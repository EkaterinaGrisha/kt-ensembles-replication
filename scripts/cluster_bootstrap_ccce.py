"""Cluster-bootstrap CI on the three-stage ablation grid vs baselines.

For every (dataset, fold, variant) cell with variant in
{full, no_s1, global_s1, no_s3}, fits the CCCE variant on deep concept-
level valid data and bootstraps its AUC/ECE contrast against three
baselines: best single deep model, LogisticStackedBlender,
and StaticConceptWeights (Exp C1 repr — the primary §5.4 comparator).

Row-independence: CCCE is a fixed post-fit function of test inputs;
precompute the ensemble vector once per cell, defer resampling to
``bootstrap_fast``. See ``ktx/ensemble.py`` design note.

Outputs
-------
``artifacts/ensembles/cluster_bootstrap_ccce.csv`` — one row per
(dataset, fold, variant). Same schema as
``cluster_bootstrap_gating.csv`` but with three contrasts per row
(vs_best, vs_stack, vs_static).

Pooled ``..._pooled.csv``: 5-fold pooled per (dataset, variant).

CLI
---
    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
        OPENBLAS_NUM_THREADS=1 python -m \\
        scripts.cluster_bootstrap_ccce
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
          f"Re-run as:\n  {prefix} python -m "
          f"scripts.cluster_bootstrap_ccce [args]", file=_sys.stderr)

import argparse
import csv
import time
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

from ktx import deep_inputs, paths
from ktx.bootstrap_fast import auc_metric_fast, paired_bootstrap_fast
from ktx.ensemble import (
    CCCE,
    LogisticStackedBlender,
    StaticConceptWeights,
)
from ktx.metrics import ece_equal_mass

DATASETS = ["assist2009", "assist2015", "assist2017", "algebra2005",
            "bridge2algebra2006", "assist2012", "ednet"]
FOLDS = [0, 1, 2, 3, 4]
DEEP_MODELS = ["dkt", "sakt", "akt", "simplekt"]
VARIANTS = {
    "full":       dict(pre_cal="concept", post_cal=True),
    "no_s1":      dict(pre_cal="none",    post_cal=True),
    "global_s1":  dict(pre_cal="global",  post_cal=True),
    "no_s3":      dict(pre_cal="concept", post_cal=False),
}
DEEP_ROOT = paths.ARTIFACTS_DIR / "predictions"


def ece_metric(y_true, y_prob) -> float:
    return ece_equal_mass(y_true, y_prob, n_bins=15)


auc_metric = auc_metric_fast


def _best_single(y, matrix, models):
    aucs = [auc_metric(y, matrix[:, j]) for j in range(matrix.shape[1])]
    j = max(range(len(models)), key=lambda k: (aucs[k], -ord(models[k][0])))
    return j, models[j]


def bootstrap_cell(dataset, fold, n_boot, seed, granularity: str = "concept"):
    try:
        cell = deep_inputs.load_cell(dataset, fold, DEEP_MODELS, granularity)
    except FileNotFoundError:
        return []
    if cell is None:
        return []
    vy, vmatrix, ty, tmatrix, kept = (cell.valid_y, cell.valid_matrix,
                                      cell.test_y, cell.test_matrix, cell.models)
    vconc, tconc, groups = cell.valid_concepts, cell.test_concepts, cell.groups

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
    stack_pred = LogisticStackedBlender().fit(vmatrix, vy).predict(tmatrix)
    static_pred = StaticConceptWeights(n_min=30).fit(
        vmatrix, vy, valid_concepts=vconc
    ).predict(tmatrix, test_concepts=tconc)

    rows = []
    for tag, kw in VARIANTS.items():
        try:
            ens = CCCE(n_min=30, **kw).fit(vmatrix, vy, valid_concepts=vconc)
            gpred = ens.predict(tmatrix, test_concepts=tconc)
        except Exception as e:
            print(f"  [warn] {dataset} f{fold} {tag}: fit failed ({e})")
            continue

        # AUC contrasts (three baselines)
        r_bAUC = paired_bootstrap_fast(ty, gpred, single_pred, auc_metric,
                                        n_boot=n_boot, groups=groups, seed=seed)
        r_stkAUC = paired_bootstrap_fast(ty, gpred, stack_pred, auc_metric,
                                          n_boot=n_boot, groups=groups, seed=seed + 1)
        r_stAUC = paired_bootstrap_fast(ty, gpred, static_pred, auc_metric,
                                         n_boot=n_boot, groups=groups, seed=seed + 2)
        # ECE contrasts
        r_bECE = paired_bootstrap_fast(ty, gpred, single_pred, ece_metric,
                                        n_boot=n_boot, groups=groups, seed=seed + 3)
        r_stkECE = paired_bootstrap_fast(ty, gpred, stack_pred, ece_metric,
                                          n_boot=n_boot, groups=groups, seed=seed + 4)
        r_stECE = paired_bootstrap_fast(ty, gpred, static_pred, ece_metric,
                                         n_boot=n_boot, groups=groups, seed=seed + 5)

        rows.append({
            "dataset": dataset, "fold": fold, "granularity": granularity,
            "variant": tag,
            "best_single_model": best_name,
            "best_single_selected_on": selected_on,
            "n_test": int(ty.size),
            "n_students": int(np.unique(groups).size) if groups is not None else -1,
            "ccce_auc": r_bAUC.value_a,
            "ccce_ece": r_bECE.value_a,
            # vs best
            "best_auc": r_bAUC.value_b,
            "delta_auc_vs_best": r_bAUC.diff,
            "auc_p_vs_best": r_bAUC.p_value,
            "best_ece": r_bECE.value_b,
            "delta_ece_vs_best": r_bECE.diff,
            "ece_p_vs_best": r_bECE.p_value,
            # vs stack
            "stack_auc": r_stkAUC.value_b,
            "delta_auc_vs_stack": r_stkAUC.diff,
            "auc_p_vs_stack": r_stkAUC.p_value,
            "stack_ece": r_stkECE.value_b,
            "delta_ece_vs_stack": r_stkECE.diff,
            "ece_p_vs_stack": r_stkECE.p_value,
            # vs static (§5.4 primary comparator)
            "static_auc": r_stAUC.value_b,
            "delta_auc_vs_static": r_stAUC.diff,
            "auc_p_vs_static": r_stAUC.p_value,
            "static_ece": r_stECE.value_b,
            "delta_ece_vs_static": r_stECE.diff,
            "ece_p_vs_static": r_stECE.p_value,
        })
    return rows


def pool_folds(per_fold):
    from collections import defaultdict
    buckets = defaultdict(list)
    for r in per_fold:
        buckets[(r["dataset"], r["variant"])].append(r)

    pooled = []
    for (ds, var), rows in buckets.items():
        d_ab = np.array([r["delta_auc_vs_best"] for r in rows])
        d_as = np.array([r["delta_auc_vs_stack"] for r in rows])
        d_ast = np.array([r["delta_auc_vs_static"] for r in rows])
        p_ab = np.array([r["auc_p_vs_best"] for r in rows])
        p_as = np.array([r["auc_p_vs_stack"] for r in rows])
        p_ast = np.array([r["auc_p_vs_static"] for r in rows])
        pooled.append({
            "dataset": ds, "variant": var,
            "n_folds": len(rows),
            "best_single_selected_on": (sel.pop() if len(sel := {r["best_single_selected_on"]
                                        for r in rows}) == 1 else "mixed"),
            "mean_delta_auc_vs_best": float(d_ab.mean()),
            "n_folds_sig_vs_best_05": int((p_ab < 0.05).sum()),
            "mean_delta_auc_vs_stack": float(d_as.mean()),
            "n_folds_sig_vs_stack_05": int((p_as < 0.05).sum()),
            "mean_delta_auc_vs_static": float(d_ast.mean()),
            "n_folds_sig_vs_static_05": int((p_ast < 0.05).sum()),
        })
    return pooled


def _write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--folds", nargs="+", type=int, default=FOLDS)
    ap.add_argument("--granularity", choices=["concept", "question"], default="concept",
                    help="уровень подробности строки: пара «задание, компонент» или задание")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--pooled-out", type=Path, default=None)
    args = ap.parse_args()
    suffix = "" if args.granularity == "concept" else "_question"
    if args.out is None:
        args.out = paths.ARTIFACTS_DIR / "ensembles" / f"cluster_bootstrap_ccce{suffix}.csv"
    if args.pooled_out is None:
        args.pooled_out = paths.ARTIFACTS_DIR / "ensembles" / f"cluster_bootstrap_ccce{suffix}_pooled.csv"

    t0 = time.time()
    cells = [(ds, fold) for ds in args.datasets for fold in args.folds]

    def _run(ds, fold):
        t_cell = time.time()
        rows = bootstrap_cell(ds, fold, args.n_boot, args.seed, args.granularity)
        return ds, fold, rows, time.time() - t_cell

    results = Parallel(n_jobs=args.n_jobs, backend="loky", verbose=10)(
        delayed(_run)(*c) for c in cells
    )

    all_rows = []
    for ds, fold, rows, dt in results:
        if not rows:
            print(f"[{ds:22s} f{fold}] SKIP [{dt:.1f}s]"); continue
        summary = ", ".join(
            f"{r['variant']}: vs static Δ={r['delta_auc_vs_static']*100:+.3f}% p={r['auc_p_vs_static']:.2g}"
            for r in rows
        )
        print(f"[{ds:22s} f{fold}] [{dt:.1f}s] {summary}")
        all_rows.extend(rows)

    _write(args.out, all_rows)
    _write(args.pooled_out, pool_folds(all_rows))
    print(f"\nWrote {len(all_rows)} per-fold rows → {args.out}")
    print(f"Total: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
