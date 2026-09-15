"""Conditional-gating ensembles on the
deep subset, at either level of row granularity.

Runs ``StaticConceptWeights``, ``GlobalStackWithConceptIntercept`` and
``LinearGating`` (from ``ktx/ensemble.py``) per (dataset, fold). Both sides
come from ``ktx.deep_inputs.load_cell``, which knows the two levels:

  * ``--granularity concept`` (default) — a row is an item-component pair.
    Fit side ``valid_y_true`` / ``valid_y_prob`` (164 550 rows on
    algebra2005), test side ``concept_y_true`` / ``concept_y_prob``
    (134 674 rows); the concept-id per row is reconstructed by
    ``ktx.concept_ids``, which repeats the pyKT evaluation walk.
  * ``--granularity question`` — a row is an item. Fit side
    ``valid_y_*_q_pykt`` (112 859 rows on algebra2005), test side
    ``y_true`` / ``y_prob`` (92 945 rows); the concept-id per row is read
    straight off the npz (``valid_cidxs_q_pykt`` / ``test_cidxs``), so no
    walk is needed.

Baselines per cell: arithmetic mean of the k component predictions, best
single component, and — for direct comparison to stacking — the
``LogisticStackedBlender`` fitted at the same level.

Scope caveats:
  * Deep-only. Classical NPZs lack persisted concept-level predictions.
    Classical + heterogeneous gating land after the pyKT cidxs re-run
    (task #26) enables cross-family alignment.
  * assist2015 has no item ids: its two levels coincide, and the
    question-level pass skips it (the npz carries no ``test_cidxs``).

Outputs
-------
``artifacts/ensembles/gating_deep.csv`` at concept level,
``gating_deep_question.csv`` at question level. Columns per
(dataset, fold, meta_learner):

    dataset, fold, granularity, meta_learner, models_in_subset,
    n_valid, n_test, n_concepts_fit,
    gated_auc, gated_ece, gated_brier,
    mean_auc,  mean_ece,  mean_brier,
    stacked_auc, stacked_ece, stacked_brier,   # LogisticStacked reference
    best_single_model, best_single_auc, best_single_ece, best_single_brier,
    lift_gated_vs_mean_auc, lift_gated_vs_best_auc, lift_gated_vs_stack_auc,
    lift_gated_vs_mean_ece, lift_gated_vs_best_ece, lift_gated_vs_stack_ece

CLI
---
    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
        OPENBLAS_NUM_THREADS=1 python -m scripts.run_gating
    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
        OPENBLAS_NUM_THREADS=1 python -m scripts.run_gating \
        --datasets algebra2005 --folds 0
    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
        OPENBLAS_NUM_THREADS=1 python -m scripts.run_gating \
        --granularity question
"""
from __future__ import annotations

import os as _os
import sys as _sys

_required_env = {
    "KMP_DUPLICATE_LIB_OK": "TRUE",
    "OMP_NUM_THREADS":      "1",
    "MKL_NUM_THREADS":      "1",
    "OPENBLAS_NUM_THREADS": "1",
}
_missing = [k for k, v in _required_env.items() if _os.environ.get(k) != v]
if _missing:
    prefix = " ".join(f"{k}={v}" for k, v in _required_env.items())
    print(f"[warn] missing env vars for macOS libomp guard: {_missing}. "
          f"Re-run as:\n  {prefix} python -m scripts.run_gating [args]",
          file=_sys.stderr)

import argparse
import csv
from pathlib import Path

import numpy as np

from ktx import deep_inputs, paths
from ktx.ensemble import (
    ArithmeticMean,
    LinearGating,
    LogisticStackedBlender,
    StaticConceptWeights,
)
from ktx.metrics import ece_equal_mass
from ktx.stats import auc_metric, brier_metric

DATASETS = ["assist2009", "assist2015", "assist2017", "algebra2005",
            "bridge2algebra2006", "assist2012", "ednet"]
FOLDS = [0, 1, 2, 3, 4]
DEEP_MODELS = ["dkt", "sakt", "akt", "simplekt"]
DEEP_ROOT = paths.ARTIFACTS_DIR / "predictions"


def _metrics(y, p, n_bins: int = 15) -> dict[str, float]:
    return {
        "auc": auc_metric(y, p),
        "ece": ece_equal_mass(y, p, n_bins=n_bins),
        "brier": brier_metric(y, p),
    }


def process_cell(dataset: str, fold: int, models: list[str],
                 granularity: str = "concept") -> list[dict]:
    try:
        cell = deep_inputs.load_cell(dataset, fold, models, granularity)
    except FileNotFoundError as e:
        print(f"  [skip] {dataset} f{fold}: concept-id lookup missing ({e})")
        return []
    if cell is None:
        return []
    vy, vmatrix, ty, tmatrix, kept = (cell.valid_y, cell.valid_matrix,
                                      cell.test_y, cell.test_matrix, cell.models)
    vconc, tconc = cell.valid_concepts, cell.test_concepts

    # Baselines shared across gating heads
    mean_pred = ArithmeticMean().predict(tmatrix)
    # Best-single must be chosen on VALID. Choosing it
    # on test makes it the maximum of several noisy test AUCs -- systematically
    # inflated -- which biases every lift-vs-best downward and lets the cluster
    # bootstrap treat a post-selection quantity as fixed. T3c fixed only
    # run_simple_ensembles.py; this file kept the test-selected baseline.
    def _argbest(y_true, mat) -> int:
        a = [auc_metric(y_true, mat[:, j]) for j in range(mat.shape[1])]
        return sorted(range(len(kept)), key=lambda j: (-a[j], kept[j]))[0]

    j_best_test = _argbest(ty, tmatrix)
    if vy is not None and vmatrix is not None and vmatrix.shape[1] == tmatrix.shape[1]:
        j_best, best_single_selected_on = _argbest(vy, vmatrix), "valid"
    else:
        j_best, best_single_selected_on = j_best_test, "test_fallback"
    single_pred = tmatrix[:, j_best]
    stacker = LogisticStackedBlender().fit(vmatrix, vy)
    stack_pred = stacker.predict(tmatrix)

    m_mean = _metrics(ty, mean_pred)
    m_best = _metrics(ty, single_pred)
    m_stack = _metrics(ty, stack_pred)

    rows: list[dict] = []
    for ml_name in ("static_concept_weights",
                     "global_stack_concept_intercept",  # T3b ablation
                     "linear_gating"):
        try:
            if ml_name == "static_concept_weights":
                ens = StaticConceptWeights(n_min=30).fit(vmatrix, vy, valid_concepts=vconc)
                gpred = ens.predict(tmatrix, test_concepts=tconc)
                n_conc_fit = ens.n_concepts_fit_
            elif ml_name == "global_stack_concept_intercept":
                # the ablation: shared component weights
                # across all concepts + per-concept intercept only.
                from ktx.ensemble import GlobalStackWithConceptIntercept
                ens = GlobalStackWithConceptIntercept(n_min=30).fit(
                    vmatrix, vy, valid_concepts=vconc)
                gpred = ens.predict(tmatrix, test_concepts=tconc)
                n_conc_fit = len(ens.concept_intercepts_)
            else:
                # LinearGating uses concept-id as a categorical context feature;
                # a linear multinomial head learns per-concept bias implicitly.
                # Log-history and item difficulty context enter in v2 alongside
                # question-level unification.
                ctx_v = vconc.reshape(-1, 1).astype(np.float64)
                ctx_t = tconc.reshape(-1, 1).astype(np.float64)
                ens = LinearGating().fit(vmatrix, vy, context=ctx_v)
                gpred = ens.predict(tmatrix, context=ctx_t)
                n_conc_fit = -1  # not a per-concept fit
        except Exception as e:
            print(f"  [warn] {dataset} f{fold} {ml_name}: fit/predict failed ({e})")
            continue

        m_g = _metrics(ty, gpred)
        rows.append({
            "dataset": dataset,
            "fold": fold,
            "granularity": granularity,
            "meta_learner": ml_name,
            "models_in_subset": ",".join(kept),
            "n_valid": int(vy.size),
            "n_test":  int(ty.size),
            "n_concepts_fit": int(n_conc_fit),
            "gated_auc": m_g["auc"],
            "gated_ece": m_g["ece"],
            "gated_brier": m_g["brier"],
            "mean_auc":  m_mean["auc"],
            "mean_ece":  m_mean["ece"],
            "mean_brier": m_mean["brier"],
            "stacked_auc": m_stack["auc"],
            "stacked_ece": m_stack["ece"],
            "stacked_brier": m_stack["brier"],
            "best_single_model": kept[j_best],
            "best_single_selected_on": best_single_selected_on,
            "best_single_on_test_model": kept[j_best_test],
            "best_single_auc": m_best["auc"],
            "best_single_ece": m_best["ece"],
            "best_single_brier": m_best["brier"],
            "lift_gated_vs_mean_auc":  m_g["auc"]  - m_mean["auc"],
            "lift_gated_vs_best_auc":  m_g["auc"]  - m_best["auc"],
            "lift_gated_vs_stack_auc": m_g["auc"]  - m_stack["auc"],
            "lift_gated_vs_mean_ece":  m_g["ece"]  - m_mean["ece"],
            "lift_gated_vs_best_ece":  m_g["ece"]  - m_best["ece"],
            "lift_gated_vs_stack_ece": m_g["ece"]  - m_stack["ece"],
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--folds", nargs="+", type=int, default=FOLDS)
    ap.add_argument("--granularity", choices=["concept", "question"], default="concept",
                    help="уровень подробности строки: пара «задание, компонент» или задание")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    if args.out is None:
        suffix = "" if args.granularity == "concept" else "_question"
        args.out = paths.ARTIFACTS_DIR / "ensembles" / f"gating_deep{suffix}.csv"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for ds in args.datasets:
        for fold in args.folds:
            cell_rows = process_cell(ds, fold, DEEP_MODELS, args.granularity)
            if not cell_rows:
                print(f"[{ds:22s} f{fold}] SKIP")
                continue
            summary = ", ".join(
                f"{r['meta_learner']}: vs stack {r['lift_gated_vs_stack_auc']*100:+.3f}% "
                f"vs best {r['lift_gated_vs_best_auc']*100:+.3f}%"
                for r in cell_rows
            )
            print(f"[{ds:22s} f{fold}] {summary}")
            rows.extend(cell_rows)

    if not rows:
        print("no rows written"); return

    keys = list(rows[0].keys())
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            row = {k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()}
            w.writerow(row)
    print(f"\nWrote {len(rows)} rows → {args.out}")


if __name__ == "__main__":
    main()
