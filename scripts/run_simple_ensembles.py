"""Parameter-free ensemble aggregations on the 7×8×5 matrix.

Loads persisted per-fold test predictions from
``artifacts/predictions/{dataset}/{model}_fold{f}.npz`` and applies each of
the five simple aggregators in ``ktx.ensemble.SIMPLE_AGGREGATORS`` to each of
the within-family subsets (classical-only, deep-only). Writes a long-form CSV
with one row per (dataset, fold, subset, aggregator) cell recording ensemble
AUC / ECE / Brier alongside the best single-model reference in the same
subset, so a Δmetric column can be sorted for headline-lift ranking.

Why within-family only (for now).
   Classical y_true has 93595 rows and deep y_true 92945 (algebra2005 fold 0
   reference), and pyKT walks different sources on each side (test.csv for
   classical, test_sequences.csv-with-selectmasks for deep). Aligning the two
   requires a per-row (uid, question, response, is_repeat) merger; the honest
   cross-family heterogeneous ensemble will land in a follow-up script
   Within each family, all
   models share ``y_true`` bit-exactly (verified across all 7 datasets), so
   the within-family ensembles are already row-aligned and correct.

Outputs
-------
``research/artifacts/ensembles/simple_ensembles.csv`` with columns:
    dataset, fold, subset, models_in_subset, aggregator,
    n_rows, auc, ece, brier,
    best_single_model, best_single_auc, delta_auc,
    best_single_ece,   delta_ece,
    best_single_brier, delta_brier

CLI
---
    python -m scripts.run_simple_ensembles                 # full matrix
    python -m scripts.run_simple_ensembles --datasets algebra2005
    python -m scripts.run_simple_ensembles --folds 0 --datasets assist2015
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from ktx import paths
from ktx.ensemble import SIMPLE_AGGREGATORS
from ktx.metrics import ece_equal_mass
from ktx.stats import auc_metric, brier_metric

DATASETS = ["assist2009", "assist2015", "assist2017", "algebra2005",
            "bridge2algebra2006", "assist2012", "ednet"]
FOLDS = [0, 1, 2, 3, 4]
SUBSETS: dict[str, list[str]] = {
    "classical": ["bkt", "pfa", "pfa_recency", "elorasch"],
    "deep":      ["dkt", "sakt", "akt", "simplekt"],
}


DEEP_MODELS = {"dkt", "sakt", "akt", "simplekt", "dkvmn", "saint"}


def _load_prediction(
    dataset: str, model: str, fold: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None] | None:
    """Return ``(test_y, test_p, valid_y, valid_p)`` for one component.

    Valid arrays are needed by T3c (best-single selected on valid instead of
    test — a post-selection inference fix). They are ``None``
    only for classical models that predate the valid-cache backfill.

    NB (2026-08-21): a brief attempt to prefer the
    self-aggregated `test_y_*_q_ours` fields for deep models was reverted
    after finding that pyKT concept-unrolling teacher-forcing leaks the
    label on `is_repeat=1` positions, inflating AUC by ~0.24 on ednet.
    Until the leak-free objects land,
    read pyKT `evaluate_question` outputs `y_true`/`y_prob` for both deep
    and classical.
    """
    p = paths.ARTIFACTS_DIR / "predictions" / dataset / f"{model}_fold{fold}.npz"
    if not p.exists():
        return None
    d = np.load(p)
    if "y_true" not in d.files or "y_prob" not in d.files:
        return None
    test_y = np.asarray(d["y_true"]).astype(int)
    test_p = np.asarray(d["y_prob"], dtype=np.float64)

    # Deep valid at question level. Prefer the leak-free T1b object
    # (`valid_y_*_q_pykt`, pyKT evaluate_question on the valid loader) over the
    # earlier late_mean aggregation (`valid_y_*_q`), which carries the
    # teacher-forcing leakage documented in notes/24_t1a_leakage_finding.md.
    # Cells retrained on Kaggle after 2026-08-31 only ever get the pykt fields —
    # without this preference they silently fell back to selecting best-single on
    # TEST, which is exactly the N1 defect this file was fixed for.
    if model in DEEP_MODELS:
        for tk, pk in (("valid_y_true_q_pykt", "valid_y_prob_q_pykt"),
                       ("valid_y_true_q", "valid_y_prob_q")):
            if {tk, pk} <= set(d.files):
                return (test_y, test_p,
                        np.asarray(d[tk]).astype(int),
                        np.asarray(d[pk], dtype=np.float64))

    # Classical valid from the separate cache file.
    cache = paths.ARTIFACTS_DIR / "predictions_valid_cache" / f"{dataset}__{model}__fold{fold}.npz"
    if cache.exists():
        vd = np.load(cache)
        if {"valid_y_true", "valid_y_prob"} <= set(vd.files):
            return (test_y, test_p,
                    np.asarray(vd["valid_y_true"]).astype(int),
                    np.asarray(vd["valid_y_prob"], dtype=np.float64))
    return test_y, test_p, None, None


def _stack_subset(
    dataset: str, models: list[str], fold: int
) -> tuple[np.ndarray, np.ndarray, list[str],
            np.ndarray | None, np.ndarray | None] | None:
    """Return (test_y, test_matrix, kept, valid_y, valid_matrix).

    valid_y / valid_matrix are None if any component's valid predictions
    are missing (rare; audit-tolerant fallback: best-single-on-valid then
    degrades to best-single-on-test with a warning).
    """
    test_y_ref: np.ndarray | None = None
    valid_y_ref: np.ndarray | None = None
    test_cols: list[np.ndarray] = []
    valid_cols: list[np.ndarray] = []
    kept: list[str] = []
    valid_missing = False
    for m in models:
        preds = _load_prediction(dataset, m, fold)
        if preds is None:
            continue
        ty, tp, vy, vp = preds
        if test_y_ref is None:
            test_y_ref = ty
        elif not np.array_equal(ty, test_y_ref):
            print(f"  [warn] {dataset} fold {fold}: {m} y_true does not match earlier "
                  f"models in subset (n={ty.size} vs {test_y_ref.size}); skipping subset")
            return None
        test_cols.append(tp)
        kept.append(m)

        if vy is None or vp is None:
            valid_missing = True
        elif valid_y_ref is None:
            valid_y_ref = vy
            valid_cols.append(vp)
        elif not np.array_equal(vy, valid_y_ref):
            # Between-model valid disagreement — mark as missing rather than skip.
            valid_missing = True
        else:
            valid_cols.append(vp)
    if len(kept) < 2 or test_y_ref is None:
        return None
    valid_matrix: np.ndarray | None
    if valid_missing or valid_y_ref is None or len(valid_cols) != len(kept):
        valid_y_ref = None
        valid_matrix = None
    else:
        valid_matrix = np.column_stack(valid_cols)
    return test_y_ref, np.column_stack(test_cols), kept, valid_y_ref, valid_matrix


def _best_single(y_true: np.ndarray, prob_matrix: np.ndarray,
                  models: list[str]) -> tuple[str, float]:
    """Best single model by AUC on the provided (y, probs). Ties broken by
    model-name lex order for determinism.
    """
    aucs = [auc_metric(y_true, prob_matrix[:, j]) for j in range(prob_matrix.shape[1])]
    order = sorted(range(len(models)), key=lambda j: (-aucs[j], models[j]))
    j = order[0]
    return models[j], float(aucs[j])


def process_cell(dataset: str, fold: int, subset_name: str,
                 models: list[str]) -> list[dict]:
    """Compute all 5 aggregators + best-single reference for one
    (dataset, fold, subset) cell. Returns list of row dicts (one per aggregator).

    Best-single is selected on VALID AUC when the
    valid predictions are available, then applied to test. The prior
    'best selected on test' number is also reported for backwards
    compatibility, and both are tagged in a new ``best_single_selected_on``
    column. Ties broken by model name lex order for determinism.
    """
    st = _stack_subset(dataset, models, fold)
    if st is None:
        return []
    y, matrix, kept, valid_y, valid_matrix = st

    # T3c: best-single on VALID (audit-preferred), fall back to TEST if the
    # valid predictions are missing/misaligned.
    if valid_y is not None and valid_matrix is not None:
        best_name, _ = _best_single(valid_y, valid_matrix, kept)
        selected_on = "valid"
    else:
        best_name, _ = _best_single(y, matrix, kept)
        selected_on = "test_fallback"
    j_best = kept.index(best_name)
    best_auc = auc_metric(y, matrix[:, j_best])
    best_ece = ece_equal_mass(y, matrix[:, j_best], n_bins=15)
    best_brier = brier_metric(y, matrix[:, j_best])

    # Backwards-compat: also report best selected on TEST AUC (the pre-T3c
    # protocol). Downstream analyses that want the audited number can join
    # on `best_single_on_test_*`.
    best_test_name, best_test_auc = _best_single(y, matrix, kept)
    j_best_test = kept.index(best_test_name)
    best_test_ece = ece_equal_mass(y, matrix[:, j_best_test], n_bins=15)

    rows: list[dict] = []
    for agg_name, agg_cls in SIMPLE_AGGREGATORS.items():
        pred = agg_cls().predict(matrix)
        # Rank aggregator returns a ranking-score, not a calibrated probability;
        # AUC is well-defined (rank-invariant), ECE / Brier are not meaningful.
        agg_ece = ece_equal_mass(y, pred, n_bins=15) if agg_name != "rank_mean" else float("nan")
        agg_brier = brier_metric(y, pred) if agg_name != "rank_mean" else float("nan")
        rows.append({
            "dataset": dataset,
            "fold": fold,
            "subset": subset_name,
            "models_in_subset": ",".join(kept),
            "aggregator": agg_name,
            "n_rows": int(y.size),
            "auc": auc_metric(y, pred),
            "ece": agg_ece,
            "brier": agg_brier,
            "best_single_model": best_name,
            "best_single_selected_on": selected_on,
            "best_single_auc": best_auc,
            "delta_auc": auc_metric(y, pred) - best_auc,
            "best_single_ece": best_ece,
            "delta_ece": (agg_ece - best_ece) if np.isfinite(agg_ece) else float("nan"),
            "best_single_brier": best_brier,
            "delta_brier": (agg_brier - best_brier) if np.isfinite(agg_brier) else float("nan"),
            # T3c: pre-fix numbers for the audited comparison.
            "best_single_on_test_model": best_test_name,
            "best_single_on_test_auc": best_test_auc,
            "best_single_on_test_ece": best_test_ece,
            "delta_auc_vs_test_best": auc_metric(y, pred) - best_test_auc,
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--folds", nargs="+", type=int, default=FOLDS)
    ap.add_argument("--out", type=Path,
                    default=paths.ARTIFACTS_DIR / "ensembles" / "simple_ensembles.csv")
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    for ds in args.datasets:
        for fold in args.folds:
            for subset_name, subset_models in SUBSETS.items():
                cell_rows = process_cell(ds, fold, subset_name, subset_models)
                if not cell_rows:
                    print(f"[{ds:22s} f{fold} {subset_name:9s}] SKIP (missing preds)")
                    continue
                # Compact log — one line per (dataset, fold, subset) cell
                lifts = ", ".join(
                    f"{r['aggregator']}={r['delta_auc']*100:+.3f}%"
                    for r in cell_rows if r["aggregator"] != "rank_mean"
                )
                print(f"[{ds:22s} f{fold} {subset_name:9s}] "
                      f"best={cell_rows[0]['best_single_model']:12s} "
                      f"(AUC {cell_rows[0]['best_single_auc']:.4f}) | ΔAUC: {lifts}")
                all_rows.extend(cell_rows)

    if not all_rows:
        print("No rows written (nothing loaded).")
        return

    fieldnames = list(all_rows[0].keys())
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in all_rows:
            # round floats for CSV cleanliness while keeping enough precision
            row = {k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()}
            w.writerow(row)
    print(f"\nWrote {len(all_rows)} rows → {args.out}")


if __name__ == "__main__":
    main()
