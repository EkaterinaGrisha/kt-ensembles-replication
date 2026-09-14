"""Cluster-bootstrap CI on deep concept-level conditional gating.

Mirrors ``cluster_bootstrap_stacking.py`` but on the deep concept-level
data and against BOTH the best single deep model AND the
``LogisticStackedBlender`` reference (which is what §5.3 needs to claim
statistical significance for the "gating beats stacking" finding).

Row-independence rationale: gating predictions are computed row-by-row
from (component logits, concept-id). Precompute once per cell, defer
resampling to ``bootstrap_fast.paired_bootstrap_fast``. See the design
note in ``ktx/ensemble.py``.

Outputs
-------
``artifacts/ensembles/cluster_bootstrap_gating.csv`` — one row per
(dataset, fold, meta_learner). Columns include CI + p for BOTH the
vs-best-single contrast and the vs-stacking contrast, so §5.3 can cite
either as the primary claim.

Pooled ``..._pooled.csv`` — 5-fold pooled per (dataset, meta_learner):
mean Δ, std Δ, min-p, and count-of-folds-significant-at-α=0.05 for both
contrasts.

CLI
---
    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
        OPENBLAS_NUM_THREADS=1 python -m \
        scripts.cluster_bootstrap_gating
    (add --datasets algebra2005 --folds 0 --n-boot 500 --n-jobs 4 for a smoke)
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
          f"scripts.cluster_bootstrap_gating [args]", file=_sys.stderr)

import argparse
import csv
import time
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

from ktx import paths
from ktx.bootstrap_fast import auc_metric_fast, paired_bootstrap_fast
from ktx.downstream import concept_ids_for_test, concept_ids_for_valid
from ktx.ensemble import (
    LinearGating,
    LogisticStackedBlender,
    StaticConceptWeights,
)
from ktx.metrics import ece_equal_mass

DATASETS = ["assist2009", "assist2015", "assist2017", "algebra2005",
            "bridge2algebra2006", "assist2012", "ednet"]
FOLDS = [0, 1, 2, 3, 4]
DEEP_MODELS = ["dkt", "sakt", "akt", "simplekt"]
GATING_HEADS = ["static_concept_weights", "global_stack_concept_intercept",
                "linear_gating"]
DEEP_ROOT = paths.ARTIFACTS_DIR / "predictions"


def ece_metric(y_true, y_prob) -> float:
    return ece_equal_mass(y_true, y_prob, n_bins=15)


auc_metric = auc_metric_fast


def _load_deep(dataset, model, fold):
    p = DEEP_ROOT / dataset / f"{model}_fold{fold}.npz"
    if not p.exists():
        return None
    d = np.load(p)
    need = {"valid_y_true", "valid_y_prob", "concept_y_true", "concept_y_prob"}
    if not need <= set(d.files):
        return None
    g = d["concept_groups"] if "concept_groups" in d.files and d["concept_groups"].size > 0 else None
    return (d["valid_y_true"].astype(int),  d["valid_y_prob"].astype(np.float64),
            d["concept_y_true"].astype(int), d["concept_y_prob"].astype(np.float64),
            g)


def _stack_deep(dataset, fold):
    v_y_ref = t_y_ref = groups = None
    v_cols, t_cols, kept = [], [], []
    for m in DEEP_MODELS:
        got = _load_deep(dataset, m, fold)
        if got is None:
            continue
        vy, vp, ty, tp, g = got
        if v_y_ref is None:
            v_y_ref, t_y_ref, groups = vy, ty, g
        else:
            if not np.array_equal(vy, v_y_ref) or not np.array_equal(ty, t_y_ref):
                return None
        v_cols.append(vp); t_cols.append(tp); kept.append(m)
    if len(kept) < 2:
        return None
    return v_y_ref, np.column_stack(v_cols), t_y_ref, np.column_stack(t_cols), groups, kept


def _best_single_col(y, matrix, models):
    aucs = [auc_metric(y, matrix[:, j]) for j in range(matrix.shape[1])]
    j = max(range(len(models)), key=lambda k: (aucs[k], -ord(models[k][0])))
    return j, models[j]


def bootstrap_cell(dataset, fold, n_boot, seed):
    st = _stack_deep(dataset, fold)
    if st is None:
        return []
    vy, vmatrix, ty, tmatrix, groups, kept = st

    # Concept-ids per row
    try:
        vconc = concept_ids_for_valid(dataset, valid_fold=fold)
        tconc = concept_ids_for_test(dataset)
    except FileNotFoundError:
        return []
    if vconc.size != vy.size or tconc.size != ty.size:
        return []

    # If groups is None (concept-level groups not persisted), fall back to
    # instance-level bootstrap — noted in output.
    # The best-single baseline must be chosen
    # on VALID. Choosing it on test makes it the maximum of several noisy test
    # AUCs -- systematically inflated -- which biases every lift-vs-best downward
    # and leaves this cluster bootstrap treating a post-selection quantity as
    # fixed. P0-4 (2026-08-31) fixed the run_*.py family only; this bootstrap
    # counterpart kept the test-selected baseline, so its CIs were built around
    # a leaked reference. Mirrors run_stacking.py / run_gating.py exactly.
    j_best_test, best_name_test = _best_single_col(ty, tmatrix, kept)
    if vy is not None and vmatrix is not None and vmatrix.shape[1] == tmatrix.shape[1]:
        j_best, best_name = _best_single_col(vy, vmatrix, kept)
        selected_on = "valid"
    else:
        j_best, best_name, selected_on = j_best_test, best_name_test, "test_fallback"
    single_pred = tmatrix[:, j_best]
    stacker = LogisticStackedBlender().fit(vmatrix, vy)
    stack_pred = stacker.predict(tmatrix)

    rows = []
    for ml_name in GATING_HEADS:
        try:
            if ml_name == "static_concept_weights":
                ens = StaticConceptWeights(n_min=30).fit(vmatrix, vy, valid_concepts=vconc)
                gpred = ens.predict(tmatrix, test_concepts=tconc)
            elif ml_name == "global_stack_concept_intercept":
                # Абляция: наклоны общие, свой у компонента только свободный член.
                from ktx.ensemble import GlobalStackWithConceptIntercept
                ens = GlobalStackWithConceptIntercept(n_min=30).fit(
                    vmatrix, vy, valid_concepts=vconc)
                gpred = ens.predict(tmatrix, test_concepts=tconc)
            else:
                ctx_v = vconc.reshape(-1, 1).astype(np.float64)
                ctx_t = tconc.reshape(-1, 1).astype(np.float64)
                ens = LinearGating().fit(vmatrix, vy, context=ctx_v)
                gpred = ens.predict(tmatrix, context=ctx_t)
        except Exception as e:
            print(f"  [warn] {dataset} f{fold} {ml_name}: fit failed ({e})")
            continue

        # vs best-single (AUC + ECE)
        r_best_auc = paired_bootstrap_fast(ty, gpred, single_pred, auc_metric,
                                            n_boot=n_boot, groups=groups, seed=seed,
                                            metric_name="auc")
        r_best_ece = paired_bootstrap_fast(ty, gpred, single_pred, ece_metric,
                                            n_boot=n_boot, groups=groups, seed=seed + 1,
                                            metric_name="ece")
        # vs LogisticStacked (AUC + ECE) — the §5.3 key contrast
        r_stk_auc = paired_bootstrap_fast(ty, gpred, stack_pred, auc_metric,
                                           n_boot=n_boot, groups=groups, seed=seed + 2,
                                           metric_name="auc")
        r_stk_ece = paired_bootstrap_fast(ty, gpred, stack_pred, ece_metric,
                                           n_boot=n_boot, groups=groups, seed=seed + 3,
                                           metric_name="ece")

        rows.append({
            "dataset": dataset,
            "fold": fold,
            "meta_learner": ml_name,
            "best_single_model": best_name,
            "best_single_selected_on": selected_on,
            "n_test": int(ty.size),
            "n_students": int(np.unique(groups).size) if groups is not None else -1,
            "bootstrap_method": "cluster" if groups is not None else "instance",
            # gating value
            "gated_auc": r_best_auc.value_a,
            "gated_ece": r_best_ece.value_a,
            # vs best single
            "best_single_auc": r_best_auc.value_b,
            "delta_auc_vs_best": r_best_auc.diff,
            "auc_ci_low_vs_best": r_best_auc.ci_low,
            "auc_ci_high_vs_best": r_best_auc.ci_high,
            "auc_p_vs_best": r_best_auc.p_value,
            "best_single_ece": r_best_ece.value_b,
            "delta_ece_vs_best": r_best_ece.diff,
            "ece_p_vs_best": r_best_ece.p_value,
            # vs LogisticStacked
            "stacked_auc": r_stk_auc.value_b,
            "delta_auc_vs_stack": r_stk_auc.diff,
            "auc_ci_low_vs_stack": r_stk_auc.ci_low,
            "auc_ci_high_vs_stack": r_stk_auc.ci_high,
            "auc_p_vs_stack": r_stk_auc.p_value,
            "stacked_ece": r_stk_ece.value_b,
            "delta_ece_vs_stack": r_stk_ece.diff,
            "ece_p_vs_stack": r_stk_ece.p_value,
        })
    return rows


def pool_folds(per_fold):
    from collections import defaultdict
    buckets = defaultdict(list)
    for r in per_fold:
        buckets[(r["dataset"], r["meta_learner"])].append(r)

    pooled = []
    for (ds, ml), rows in buckets.items():
        d_ab = np.array([r["delta_auc_vs_best"] for r in rows])
        d_as = np.array([r["delta_auc_vs_stack"] for r in rows])
        d_eb = np.array([r["delta_ece_vs_best"] for r in rows])
        d_es = np.array([r["delta_ece_vs_stack"] for r in rows])
        p_ab = np.array([r["auc_p_vs_best"] for r in rows])
        p_as = np.array([r["auc_p_vs_stack"] for r in rows])
        pooled.append({
            "dataset": ds, "meta_learner": ml,
            "n_folds": len(rows),
            "best_single_selected_on": (sel.pop() if len(sel := {r["best_single_selected_on"]
                                        for r in rows}) == 1 else "mixed"),
            "mean_delta_auc_vs_best": float(d_ab.mean()),
            "std_delta_auc_vs_best":  float(d_ab.std(ddof=1)) if len(rows) > 1 else float("nan"),
            "min_auc_p_vs_best": float(p_ab.min()),
            "n_folds_sig_vs_best_05": int((p_ab < 0.05).sum()),
            "mean_delta_auc_vs_stack": float(d_as.mean()),
            "std_delta_auc_vs_stack": float(d_as.std(ddof=1)) if len(rows) > 1 else float("nan"),
            "min_auc_p_vs_stack": float(p_as.min()),
            "n_folds_sig_vs_stack_05": int((p_as < 0.05).sum()),
            "mean_delta_ece_vs_best": float(d_eb.mean()),
            "mean_delta_ece_vs_stack": float(d_es.mean()),
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
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--out", type=Path,
                    default=paths.ARTIFACTS_DIR / "ensembles" / "cluster_bootstrap_gating.csv")
    ap.add_argument("--pooled-out", type=Path,
                    default=paths.ARTIFACTS_DIR / "ensembles" / "cluster_bootstrap_gating_pooled.csv")
    args = ap.parse_args()

    t0 = time.time()
    cells = [(ds, fold) for ds in args.datasets for fold in args.folds]

    def _run(ds, fold):
        t_cell = time.time()
        rows = bootstrap_cell(ds, fold, args.n_boot, args.seed)
        return ds, fold, rows, time.time() - t_cell

    results = Parallel(n_jobs=args.n_jobs, backend="loky", verbose=10)(
        delayed(_run)(*c) for c in cells
    )

    all_rows = []
    for ds, fold, rows, dt in results:
        if not rows:
            print(f"[{ds:22s} f{fold}] SKIP [{dt:.1f}s]")
            continue
        summary = ", ".join(
            f"{r['meta_learner']}: vs stack Δ={r['delta_auc_vs_stack']*100:+.3f}% p={r['auc_p_vs_stack']:.2g}"
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
