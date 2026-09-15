"""Cluster-bootstrap CI on the logistic-stacked headline contrasts.

Mirrors ``cluster_bootstrap_simple_ensembles.py`` but for the stacked
predictions: for every (dataset, subset, fold) cell in the within-family
matrix, fit ``LogisticStackedBlender`` on valid-fold component predictions,
apply to the test fold, and bootstrap the AUC / ECE contrast against the
best single model in the same subset.

Row-independence still holds — the stacker is a fixed function once fit on
validation-fold predictions, so the stacked test prediction is precomputed
once per cell and the bootstrap resamples only the test-side (y, prediction)
pairs. Consistent with the rationale in the ``ktx/ensemble.py`` design note.

Outputs
-------
``artifacts/ensembles/cluster_bootstrap_stacking_all_v4.csv`` — one row per
(dataset, fold, subset). Same schema as the simple-ensemble bootstrap
minus the ``aggregator`` axis (there is only one meta-learner in this
first pass — the LogisticStackedBlender). Columns:

    dataset, fold, subset, models_in_subset, best_single_model,
    best_single_selected_on,
    n_valid, n_test, n_students,
    stacked_auc, best_single_auc, delta_auc, auc_ci_low, auc_ci_high, auc_p,
    stacked_ece, best_single_ece, delta_ece, ece_ci_low, ece_ci_high, ece_p

Pooled 5-fold summary:
``artifacts/ensembles/cluster_bootstrap_stacking_all_pooled_v4.csv``.

CLI
---
    python -m scripts.cluster_bootstrap_stacking
    python -m scripts.cluster_bootstrap_stacking --datasets algebra2005 --folds 0
    python -m scripts.cluster_bootstrap_stacking --n-boot 500 --n-jobs 4
"""
from __future__ import annotations

# macOS OpenMP conflict guard — see run_stacking.py docstring for the reason.
# Required env vars: KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1
# MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 (all set from the calling shell).
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
          f"Re-run as:\n  {prefix} python -m "
          f"scripts.cluster_bootstrap_stacking [args]", file=_sys.stderr)

import argparse
import csv
import time
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

from ktx import paths
from ktx.bootstrap_fast import auc_metric_fast, paired_bootstrap_fast
from ktx.ensemble import STACKED_BLENDERS
from ktx.metrics import ece_equal_mass

DATASETS = ["assist2009", "assist2015", "assist2017", "algebra2005",
            "bridge2algebra2006", "assist2012", "ednet"]
FOLDS = [0, 1, 2, 3, 4]
SUBSETS: dict[str, list[str]] = {
    "classical": ["bkt", "pfa", "pfa_recency", "elorasch"],
    "deep":      ["dkt", "sakt", "akt", "simplekt"],
}

CLASSICAL_CACHE = paths.ARTIFACTS_DIR / "predictions_valid_cache"
DEEP_ROOT = paths.ARTIFACTS_DIR / "predictions"


def ece_metric(y_true, y_prob) -> float:
    """Equal-mass 15-bin ECE — consistent with the reporting convention
    across ``run_simple_ensembles`` / ``cluster_bootstrap_simple_ensembles``."""
    return ece_equal_mass(y_true, y_prob, n_bins=15)


auc_metric = auc_metric_fast


# ─── loaders (share the run_stacking convention) ─────────────────────────── #


def _load_classical(dataset: str, model: str, fold: int
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    p = CLASSICAL_CACHE / f"{dataset}__{model}__fold{fold}.npz"
    if not p.exists():
        return None
    d = np.load(p)
    g = d["groups"] if "groups" in d.files and d["groups"].size > 0 else None
    return (d["valid_y_true"].astype(int), d["valid_y_prob"].astype(np.float64),
            d["y_true"].astype(int),        d["y_prob"].astype(np.float64),
            g)


def _load_deep(dataset: str, model: str, fold: int,
               valid_source: str = "v2_qlvl_late_mean"
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """See run_stacking._load_deep for the valid_source semantics."""
    p = DEEP_ROOT / dataset / f"{model}_fold{fold}.npz"
    if not p.exists():
        return None
    d = np.load(p)
    if valid_source == "v4_pykt_evaluate_question":
        need = {"valid_y_true_q_pykt", "valid_y_prob_q_pykt", "y_true", "y_prob"}
        if not need <= set(d.files):
            return None
        g = d["groups"] if "groups" in d.files and d["groups"].size > 0 else None
        return (d["valid_y_true_q_pykt"].astype(int),
                d["valid_y_prob_q_pykt"].astype(np.float64),
                d["y_true"].astype(int), d["y_prob"].astype(np.float64), g)
    if valid_source == "v2_qlvl_late_mean":
        need = {"valid_y_true_q", "valid_y_prob_q", "y_true", "y_prob"}
        if not need <= set(d.files):
            return None
        g = d["groups"] if "groups" in d.files and d["groups"].size > 0 else None
        return (d["valid_y_true_q"].astype(int),
                d["valid_y_prob_q"].astype(np.float64),
                d["y_true"].astype(int), d["y_prob"].astype(np.float64), g)
    raise ValueError(f"unknown valid_source: {valid_source}")


def _stack_subset(dataset: str, subset_name: str, models: list[str], fold: int,
                   valid_source: str = "v2_qlvl_late_mean"):
    if subset_name == "classical":
        loader = lambda ds_, m_, f_: _load_classical(ds_, m_, f_)
    else:
        loader = lambda ds_, m_, f_: _load_deep(ds_, m_, f_, valid_source=valid_source)
    v_y_ref = t_y_ref = groups = None
    v_cols, t_cols, kept = [], [], []
    for m in models:
        got = loader(dataset, m, fold)
        if got is None:
            continue
        vy, vp, ty, tp, g = got
        if v_y_ref is None:
            v_y_ref, t_y_ref, groups = vy, ty, g
        else:
            if not np.array_equal(vy, v_y_ref) or not np.array_equal(ty, t_y_ref):
                return None
        v_cols.append(vp); t_cols.append(tp); kept.append(m)
    if len(kept) < 2 or v_y_ref is None or t_y_ref is None or groups is None:
        return None
    return v_y_ref, np.column_stack(v_cols), t_y_ref, np.column_stack(t_cols), groups, kept


def _best_single(y, matrix, models) -> tuple[int, str]:
    aucs = [auc_metric(y, matrix[:, j]) for j in range(matrix.shape[1])]
    order = sorted(range(len(models)), key=lambda j: (-aucs[j], models[j]))
    return order[0], models[order[0]]


# ─── per-cell bootstrap ───────────────────────────────────────────────────── #


def bootstrap_cell(dataset: str, fold: int, subset_name: str,
                    models: list[str], meta_learners: list[str],
                    n_boot: int, seed: int,
                    valid_source: str = "v2_qlvl_late_mean") -> list[dict]:
    """Bootstrap one (dataset, fold, subset) cell for every meta-learner in
    ``meta_learners``. Baseline (best single) is computed once and shared."""
    st = _stack_subset(dataset, subset_name, models, fold, valid_source=valid_source)
    if st is None:
        return []
    vy, vmatrix, ty, tmatrix, groups, kept = st
    # The best-single baseline must be chosen
    # on VALID. Choosing it on test makes it the maximum of several noisy test
    # AUCs -- systematically inflated -- which biases every lift-vs-best downward
    # and leaves this cluster bootstrap treating a post-selection quantity as
    # fixed. P0-4 (2026-08-31) fixed the run_*.py family only; this bootstrap
    # counterpart kept the test-selected baseline, so its CIs were built around
    # a leaked reference. Mirrors run_stacking.py / run_gating.py exactly.
    j_best_test, best_name_test = _best_single(ty, tmatrix, kept)
    if vy is not None and vmatrix is not None and vmatrix.shape[1] == tmatrix.shape[1]:
        j_best, best_name = _best_single(vy, vmatrix, kept)
        selected_on = "valid"
    else:
        j_best, best_name, selected_on = j_best_test, best_name_test, "test_fallback"
    single_pred = tmatrix[:, j_best]

    rows: list[dict] = []
    for ml_name in meta_learners:
        cls = STACKED_BLENDERS[ml_name]
        try:
            stacker = cls().fit(vmatrix, vy)
            stacked_pred = stacker.predict(tmatrix)
        except Exception as e:
            print(f"  [warn] {dataset} f{fold} {subset_name} {ml_name}: fit failed ({e})")
            continue
        res_auc = paired_bootstrap_fast(
            ty, stacked_pred, single_pred, auc_metric,
            n_boot=n_boot, groups=groups, seed=seed, metric_name="auc",
        )
        res_ece = paired_bootstrap_fast(
            ty, stacked_pred, single_pred, ece_metric,
            n_boot=n_boot, groups=groups, seed=seed + 1, metric_name="ece",
        )
        rows.append({
            "dataset": dataset,
            "fold": fold,
            "subset": subset_name,
            "meta_learner": ml_name,
            "models_in_subset": ",".join(kept),
            "best_single_model": best_name,
            "best_single_selected_on": selected_on,
            "n_valid": int(vy.size),
            "n_test": int(ty.size),
            "n_students": int(np.unique(groups).size),
            "stacked_auc": res_auc.value_a,
            "best_single_auc": res_auc.value_b,
            "delta_auc": res_auc.diff,
            "auc_ci_low": res_auc.ci_low,
            "auc_ci_high": res_auc.ci_high,
            "auc_p": res_auc.p_value,
            "stacked_ece": res_ece.value_a,
            "best_single_ece": res_ece.value_b,
            "delta_ece": res_ece.diff,
            "ece_ci_low": res_ece.ci_low,
            "ece_ci_high": res_ece.ci_high,
            "ece_p": res_ece.p_value,
        })
    return rows


def pool_folds(per_fold: list[dict]) -> list[dict]:
    from collections import defaultdict
    buckets: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for r in per_fold:
        buckets[(r["dataset"], r["subset"], r["meta_learner"])].append(r)

    pooled: list[dict] = []
    for (ds, subs, ml), rows in buckets.items():
        deltas_auc = np.array([r["delta_auc"] for r in rows])
        deltas_ece = np.array([r["delta_ece"] for r in rows])
        ps_auc = np.array([r["auc_p"] for r in rows])
        ps_ece = np.array([r["ece_p"] for r in rows])
        pooled.append({
            "dataset": ds, "subset": subs, "meta_learner": ml,
            "n_folds": len(rows),
            "mean_delta_auc": float(deltas_auc.mean()),
            "std_delta_auc": float(deltas_auc.std(ddof=1)) if len(rows) > 1 else float("nan"),
            "min_auc_p": float(ps_auc.min()),
            "n_folds_sig_auc_05": int((ps_auc < 0.05).sum()),
            "mean_delta_ece": float(deltas_ece.mean()),
            "std_delta_ece": float(deltas_ece.std(ddof=1)) if len(rows) > 1 else float("nan"),
            "min_ece_p": float(ps_ece.min()),
            "n_folds_sig_ece_05": int((ps_ece < 0.05).sum()),
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


# ─── main ─────────────────────────────────────────────────────────────────── #


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--folds", nargs="+", type=int, default=FOLDS)
    ap.add_argument("--subsets", nargs="+", default=list(SUBSETS.keys()),
                    choices=list(SUBSETS.keys()))
    ap.add_argument("--meta-learners", nargs="+", default=list(STACKED_BLENDERS.keys()),
                    choices=list(STACKED_BLENDERS.keys()))
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--out", type=Path,
                    default=paths.ARTIFACTS_DIR / "ensembles" / "cluster_bootstrap_stacking_all.csv")
    ap.add_argument("--pooled-out", type=Path,
                    default=paths.ARTIFACTS_DIR / "ensembles" / "cluster_bootstrap_stacking_all_pooled.csv")
    ap.add_argument("--valid-source",
                    choices=["v4_pykt_evaluate_question", "v2_qlvl_late_mean"],
                    default="v2_qlvl_late_mean",
                    help=("Fit-side q-level probability object for deep "
                          "subsets (see run_stacking.py)."))
    args = ap.parse_args()

    # Auto-suffix defaults when v4 is selected (only when user did not override)
    default_out = paths.ARTIFACTS_DIR / "ensembles" / "cluster_bootstrap_stacking_all.csv"
    default_pooled = paths.ARTIFACTS_DIR / "ensembles" / "cluster_bootstrap_stacking_all_pooled.csv"
    if args.valid_source == "v4_pykt_evaluate_question":
        if args.out == default_out:
            args.out = paths.ARTIFACTS_DIR / "ensembles" / "cluster_bootstrap_stacking_all_v4.csv"
        if args.pooled_out == default_pooled:
            args.pooled_out = paths.ARTIFACTS_DIR / "ensembles" / "cluster_bootstrap_stacking_all_pooled_v4.csv"

    t0 = time.time()
    cells = [(ds, fold, subset_name)
             for ds in args.datasets
             for fold in args.folds
             for subset_name in args.subsets]

    def _run(ds, fold, subset_name):
        t_cell = time.time()
        rows = bootstrap_cell(ds, fold, subset_name, SUBSETS[subset_name],
                               args.meta_learners,
                               n_boot=args.n_boot, seed=args.seed,
                               valid_source=args.valid_source)
        return ds, fold, subset_name, rows, time.time() - t_cell

    results = Parallel(n_jobs=args.n_jobs, backend="loky", verbose=10)(
        delayed(_run)(*c) for c in cells
    )

    all_rows: list[dict] = []
    for ds, fold, subset_name, rows, dt in results:
        if not rows:
            print(f"[{ds:22s} f{fold} {subset_name:9s}] SKIP [{dt:.1f}s]")
            continue
        summary = ", ".join(
            f"{r['meta_learner']}:{r['delta_auc']*100:+.2f}%(p={r['auc_p']:.2g})"
            for r in rows
        )
        print(f"[{ds:22s} f{fold} {subset_name:9s}] [{dt:.1f}s] {summary}")
        all_rows.extend(rows)

    _write_csv(args.out, all_rows)
    pooled_rows = pool_folds(all_rows)
    _write_csv(args.pooled_out, pooled_rows)

    print(f"\nWrote {len(all_rows)} per-fold rows → {args.out}")
    print(f"Wrote {len(pooled_rows)} pooled rows → {args.pooled_out}")
    print(f"Total: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
