"""Cluster-bootstrap CI on the simple-ensemble headline contrasts.

For every (dataset, subset, aggregator, fold) cell computed by
``run_simple_ensembles.py``, this script re-derives the ensemble prediction
from the underlying NPZ files and runs a paired cluster-bootstrap over
students against the fold's *best single model in the same subset*. Two
metrics are contrasted:

  * AUC   — via ``stats.auc_metric``
  * ECE   — via ``ece_equal_mass`` (equal-mass, 15 bins), matching the point
            estimates in ``run_simple_ensembles.py``

The bootstrap resamples student clusters (``groups`` field, present on every
npz), which is the honest CI for KT test sets where within-student
interactions are correlated. All ensembles here are row-independent
aggregators, so ``bootstrap_ensemble_vs_single`` precomputes the ensemble
vector once and defers to ``stats.paired_bootstrap`` — see the design note
inside ``ktx/ensemble.py``.

Outputs
-------
``artifacts/ensembles/cluster_bootstrap_simple_ensembles.csv`` with one row
per (dataset, fold, subset, aggregator) cell:

    dataset, fold, subset, aggregator, models_in_subset,
    best_single_model, best_single_selected_on, n_rows, n_students,
    ensemble_auc, best_single_auc, delta_auc, auc_ci_low, auc_ci_high, auc_p,
    ensemble_ece, best_single_ece, delta_ece, ece_ci_low, ece_ci_high, ece_p

A companion 5-fold-pooled summary:
``artifacts/ensembles/cluster_bootstrap_simple_ensembles_pooled.csv`` with
mean ΔAUC / ΔECE across folds plus min-across-folds p (loose lower bound on
significance without Fisher-combining) — used as a first ranking pass; full
Holm-across-datasets correction lives in the eventual paper-numbers audit.

CLI
---
    python -m scripts.cluster_bootstrap_simple_ensembles
    python -m scripts.cluster_bootstrap_simple_ensembles --n-boot 500 \
        --aggregators geometric_mean --datasets algebra2005
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

from ktx import paths
from ktx.bootstrap_fast import auc_metric_fast, paired_bootstrap_fast
from ktx.ensemble import SIMPLE_AGGREGATORS
from ktx.metrics import ece_equal_mass


def ece_metric(y_true, y_prob) -> float:
    """Equal-mass 15-bin ECE — pure numpy, ~50x faster than netcal.ECE in a
    bootstrap loop. Consistent with the reporting convention in
    ``run_simple_ensembles.py`` (equal_mass, n_bins=15), so per-fold ensemble
    ECE values in this table match the ECE values in the simple-ensembles
    table row-for-row."""
    return ece_equal_mass(y_true, y_prob, n_bins=15)


# Alias so the rest of this script reads the fast AUC as ``auc_metric``.
auc_metric = auc_metric_fast


def _bootstrap_ensemble_vs_single_fast(y_true, matrix, aggregator, single_pred,
                                        metric_fn, n_boot, groups, seed, name):
    """Local wrapper: precompute the ensemble vector once (row-independence)
    and defer to ``paired_bootstrap_fast``. Mirrors
    ``ensemble.bootstrap_ensemble_vs_single`` but uses the vectorized cluster
    bootstrap and numpy-native AUC."""
    ensemble_pred = aggregator.predict(matrix)
    return paired_bootstrap_fast(
        y_true, ensemble_pred, single_pred, metric_fn,
        n_boot=n_boot, groups=groups, seed=seed, metric_name=name,
    )

DATASETS = ["assist2009", "assist2015", "assist2017", "algebra2005",
            "bridge2algebra2006", "assist2012", "ednet"]
FOLDS = [0, 1, 2, 3, 4]
SUBSETS: dict[str, list[str]] = {
    "classical": ["bkt", "pfa", "pfa_recency", "elorasch"],
    "deep":      ["dkt", "sakt", "akt", "simplekt"],
}
# rank_mean produces a rank-score, not a probability; ECE is not meaningful.
# Bootstrap CI on AUC is still valid there, but we exclude it from ECE runs.
ECE_ELIGIBLE_AGGREGATORS = [a for a in SIMPLE_AGGREGATORS if a != "rank_mean"]


def _load(dataset: str, model: str, fold: int) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    p = paths.ARTIFACTS_DIR / "predictions" / dataset / f"{model}_fold{fold}.npz"
    if not p.exists():
        return None
    d = np.load(p)
    if not {"y_true", "y_prob", "groups"}.issubset(d.files):
        return None
    return (np.asarray(d["y_true"]).astype(int),
            np.asarray(d["y_prob"], dtype=np.float64),
            np.asarray(d["groups"]))


# Valid predictions, needed so the best-single baseline can be selected on
# VALID instead of on test (N1 / P0-4). Deep models carry them inside the main
# npz; classical models keep them in a separate backfill cache. Mirrors
# ``run_simple_ensembles._load_prediction`` field-for-field -- including the
# preference for the leak-free ``*_q_pykt`` object over the earlier ``*_q``
# late_mean aggregation, which carries a teacher-forcing leak.
_DEEP_MODELS = set(SUBSETS["deep"])


def _load_valid(dataset: str, model: str, fold: int
                ) -> tuple[np.ndarray, np.ndarray] | None:
    if model in _DEEP_MODELS:
        fp = paths.ARTIFACTS_DIR / "predictions" / dataset / f"{model}_fold{fold}.npz"
        if not fp.exists():
            return None
        d = np.load(fp)
        for tk, pk in (("valid_y_true_q_pykt", "valid_y_prob_q_pykt"),
                       ("valid_y_true_q", "valid_y_prob_q")):
            if {tk, pk} <= set(d.files):
                return (np.asarray(d[tk]).astype(int),
                        np.asarray(d[pk], dtype=np.float64))
        return None
    cache = (paths.ARTIFACTS_DIR / "predictions_valid_cache"
             / f"{dataset}__{model}__fold{fold}.npz")
    if not cache.exists():
        return None
    vd = np.load(cache)
    if not {"valid_y_true", "valid_y_prob"} <= set(vd.files):
        return None
    return (np.asarray(vd["valid_y_true"]).astype(int),
            np.asarray(vd["valid_y_prob"], dtype=np.float64))


def _stack_subset(dataset: str, models: list[str], fold: int
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str],
                             np.ndarray | None, np.ndarray | None] | None:
    """Return (y_true, matrix, groups, models_kept, valid_y, valid_matrix).

    ``valid_y`` / ``valid_matrix`` are None when any component's valid
    predictions are missing or disagree between models -- the audit-tolerant
    fallback that degrades best-single selection back to test, tagged as
    ``test_fallback`` in the output.
    """
    y_ref = None
    g_ref = None
    vy_ref = None
    cols: list[np.ndarray] = []
    vcols: list[np.ndarray] = []
    kept: list[str] = []
    valid_missing = False
    for m in models:
        got = _load(dataset, m, fold)
        if got is None:
            continue
        y, p, g = got
        if y_ref is None:
            y_ref, g_ref = y, g
        elif not np.array_equal(y, y_ref):
            print(f"  [warn] {dataset} f{fold}: {m} y_true disagrees; skipping subset")
            return None
        cols.append(p)
        kept.append(m)

        vgot = _load_valid(dataset, m, fold)
        if vgot is None:
            valid_missing = True
        else:
            vy, vp = vgot
            if vy_ref is None:
                vy_ref = vy
                vcols.append(vp)
            elif not np.array_equal(vy, vy_ref):
                valid_missing = True
            else:
                vcols.append(vp)
    if len(kept) < 2 or y_ref is None:
        return None
    if valid_missing or vy_ref is None or len(vcols) != len(kept):
        vy_ref, vmatrix = None, None
    else:
        vmatrix = np.column_stack(vcols)
    return y_ref, np.column_stack(cols), g_ref, kept, vy_ref, vmatrix


def _best_single_col(y: np.ndarray, matrix: np.ndarray, models: list[str]) -> tuple[int, str]:
    aucs = [auc_metric(y, matrix[:, j]) for j in range(matrix.shape[1])]
    order = sorted(range(len(models)), key=lambda j: (-aucs[j], models[j]))
    return order[0], models[order[0]]


def bootstrap_cell(dataset: str, fold: int, subset_name: str,
                   models: list[str], aggregators: list[str],
                   n_boot: int, seed: int) -> list[dict]:
    st = _stack_subset(dataset, models, fold)
    if st is None:
        return []
    y, matrix, groups, kept, valid_y, valid_matrix = st
    # The best-single baseline must be chosen
    # on VALID. Choosing it on test makes it the maximum of several noisy test
    # AUCs -- systematically inflated -- which biases every lift-vs-best
    # downward and leaves this cluster bootstrap treating a post-selection
    # quantity as fixed. T3c fixed run_simple_ensembles.py on 2026-08-25 and
    # P0-4 fixed the rest of the run_*.py family on 2026-08-31, but neither
    # touched this bootstrap counterpart.
    j_best_test, best_name_test = _best_single_col(y, matrix, kept)
    if valid_y is not None and valid_matrix is not None \
            and valid_matrix.shape[1] == matrix.shape[1]:
        j_best, best_name = _best_single_col(valid_y, valid_matrix, kept)
        selected_on = "valid"
    else:
        j_best, best_name, selected_on = j_best_test, best_name_test, "test_fallback"
    single = matrix[:, j_best]

    rows: list[dict] = []
    for agg_name in aggregators:
        agg = SIMPLE_AGGREGATORS[agg_name]()
        # AUC bootstrap (always valid, including rank_mean)
        res_auc = _bootstrap_ensemble_vs_single_fast(
            y, matrix, agg, single, auc_metric,
            n_boot=n_boot, groups=groups, seed=seed, name="auc",
        )
        # ECE bootstrap (skip rank_mean — not a probability)
        if agg_name in ECE_ELIGIBLE_AGGREGATORS:
            res_ece = _bootstrap_ensemble_vs_single_fast(
                y, matrix, agg, single, ece_metric,
                n_boot=n_boot, groups=groups, seed=seed + 1, name="ece",
            )
        else:
            res_ece = None

        rows.append({
            "dataset": dataset,
            "fold": fold,
            "subset": subset_name,
            "aggregator": agg_name,
            "models_in_subset": ",".join(kept),
            "best_single_model": best_name,
            "best_single_selected_on": selected_on,
            "n_rows": int(y.size),
            "n_students": int(np.unique(groups).size),
            "ensemble_auc": res_auc.value_a,
            "best_single_auc": res_auc.value_b,
            "delta_auc": res_auc.diff,
            "auc_ci_low": res_auc.ci_low,
            "auc_ci_high": res_auc.ci_high,
            "auc_p": res_auc.p_value,
            "ensemble_ece": res_ece.value_a if res_ece else float("nan"),
            "best_single_ece": res_ece.value_b if res_ece else float("nan"),
            "delta_ece": res_ece.diff if res_ece else float("nan"),
            "ece_ci_low": res_ece.ci_low if res_ece else float("nan"),
            "ece_ci_high": res_ece.ci_high if res_ece else float("nan"),
            "ece_p": res_ece.p_value if res_ece else float("nan"),
        })
    return rows


def pool_folds(per_fold: list[dict]) -> list[dict]:
    """5-fold pooled summary per (dataset, subset, aggregator). Reports mean
    ΔAUC / ΔECE across folds and min-across-folds p (loose lower-bound on
    significance without Fisher-combining). Also records the fold-count so
    incomplete cells are visible."""
    from collections import defaultdict
    buckets: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for r in per_fold:
        buckets[(r["dataset"], r["subset"], r["aggregator"])].append(r)

    pooled: list[dict] = []
    for (ds, subs, agg), rows in buckets.items():
        deltas_auc = np.array([r["delta_auc"] for r in rows])
        deltas_ece = np.array([r["delta_ece"] for r in rows], dtype=np.float64)
        ps_auc = np.array([r["auc_p"] for r in rows])
        ps_ece = np.array([r["ece_p"] for r in rows], dtype=np.float64)
        pooled.append({
            "dataset": ds, "subset": subs, "aggregator": agg,
            "n_folds": len(rows),
            "mean_delta_auc": float(deltas_auc.mean()),
            "std_delta_auc": float(deltas_auc.std(ddof=1)) if len(rows) > 1 else float("nan"),
            "min_auc_p": float(ps_auc.min()),
            "n_folds_sig_auc_05": int((ps_auc < 0.05).sum()),
            "mean_delta_ece": float(np.nanmean(deltas_ece)) if not np.all(np.isnan(deltas_ece)) else float("nan"),
            "std_delta_ece": float(np.nanstd(deltas_ece, ddof=1)) if np.isfinite(deltas_ece).sum() > 1 else float("nan"),
            "min_ece_p": float(np.nanmin(ps_ece)) if not np.all(np.isnan(ps_ece)) else float("nan"),
            "n_folds_sig_ece_05": int((ps_ece < 0.05).sum()) if not np.all(np.isnan(ps_ece)) else 0,
        })
    return pooled


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            row = {k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()}
            w.writerow(row)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--folds", nargs="+", type=int, default=FOLDS)
    ap.add_argument("--subsets", nargs="+", default=list(SUBSETS.keys()),
                    choices=list(SUBSETS.keys()))
    ap.add_argument("--aggregators", nargs="+", default=list(SIMPLE_AGGREGATORS.keys()),
                    choices=list(SIMPLE_AGGREGATORS.keys()))
    ap.add_argument("--n-boot", type=int, default=2000,
                    help="2000 cluster-bootstrap iterations")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-jobs", type=int, default=-1,
                    help="joblib worker count for parallel cell dispatch; "
                    "-1 uses all cores (default). 1 = serial (debugging).")
    ap.add_argument("--out", type=Path,
                    default=paths.ARTIFACTS_DIR / "ensembles"
                    / "cluster_bootstrap_simple_ensembles.csv")
    ap.add_argument("--pooled-out", type=Path,
                    default=paths.ARTIFACTS_DIR / "ensembles"
                    / "cluster_bootstrap_simple_ensembles_pooled.csv")
    args = ap.parse_args()

    t0 = time.time()
    # Fan out all (dataset, fold, subset) cells to the worker pool. Each cell
    # is independent — no shared state, so process-based parallelism
    # (loky backend) is a clean fit. On 10-core machines this cuts the total
    # wall time by ~7x over serial.
    cells = [(ds, fold, subset_name)
             for ds in args.datasets
             for fold in args.folds
             for subset_name in args.subsets]

    def _run_cell(ds, fold, subset_name):
        t_cell = time.time()
        rows = bootstrap_cell(
            ds, fold, subset_name, SUBSETS[subset_name],
            args.aggregators, n_boot=args.n_boot, seed=args.seed,
        )
        dt = time.time() - t_cell
        return ds, fold, subset_name, rows, dt

    results = Parallel(n_jobs=args.n_jobs, backend="loky", verbose=10)(
        delayed(_run_cell)(*c) for c in cells
    )

    all_rows: list[dict] = []
    for ds, fold, subset_name, rows, dt in results:
        if not rows:
            print(f"[{ds:22s} f{fold} {subset_name:9s}] SKIP [{dt:.1f}s]")
            continue
        summary = ", ".join(
            f"{r['aggregator']}: Δ={r['delta_auc']*100:+.3f}% p={r['auc_p']:.3g}"
            for r in rows
        )
        print(f"[{ds:22s} f{fold} {subset_name:9s}] [{dt:.1f}s] {summary}")
        all_rows.extend(rows)

    _write_csv(args.out, all_rows)
    pooled_rows = pool_folds(all_rows)
    _write_csv(args.pooled_out, pooled_rows)

    total_dt = time.time() - t0
    print(f"\nWrote {len(all_rows)} per-fold rows → {args.out}")
    print(f"Wrote {len(pooled_rows)} pooled rows → {args.pooled_out}")
    print(f"Total: {total_dt:.1f}s")


if __name__ == "__main__":
    main()
