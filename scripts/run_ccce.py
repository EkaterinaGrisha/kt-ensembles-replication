"""Concept-Conditional Calibrated Ensemble.

Runs CCCE (per-component concept-aware isotonic + concept-conditional
gating + global isotonic post-cal) on the deep concept-level matrix.
Baselines per cell:

  * ``mean``      — arithmetic-mean simple ensemble.
  * ``stacked``   — LogisticStackedBlender.
  * ``static_cg`` — StaticConceptWeights.
  * ``best_single`` — best deep model on this fold by AUC.

Outputs
-------
``artifacts/ensembles/ccce_deep.csv`` — one row per (dataset, fold). All
five metrics per configuration (CCCE + 4 baselines) plus lift columns
against each baseline.

CLI
---
    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
        OPENBLAS_NUM_THREADS=1 python -m scripts.run_ccce
    (add --datasets algebra2005 --folds 0 for a smoke)
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
          f"Re-run as:\n  {prefix} python -m scripts.run_ccce [args]",
          file=_sys.stderr)

import argparse
import csv
from pathlib import Path

import numpy as np

from ktx import paths
from ktx.downstream import concept_ids_for_test, concept_ids_for_valid
from ktx.ensemble import (
    CCCE,
    ArithmeticMean,
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


def _load_deep_concept(dataset, model, fold):
    p = DEEP_ROOT / dataset / f"{model}_fold{fold}.npz"
    if not p.exists():
        return None
    d = np.load(p)
    need = {"valid_y_true", "valid_y_prob", "concept_y_true", "concept_y_prob"}
    if not need <= set(d.files):
        return None
    return (d["valid_y_true"].astype(int),  d["valid_y_prob"].astype(np.float64),
            d["concept_y_true"].astype(int), d["concept_y_prob"].astype(np.float64))


def _stack_deep(dataset, fold):
    v_y_ref = t_y_ref = None
    v_cols, t_cols, kept = [], [], []
    for m in DEEP_MODELS:
        got = _load_deep_concept(dataset, m, fold)
        if got is None:
            continue
        vy, vp, ty, tp = got
        if v_y_ref is None:
            v_y_ref, t_y_ref = vy, ty
        else:
            if not np.array_equal(vy, v_y_ref) or not np.array_equal(ty, t_y_ref):
                return None
        v_cols.append(vp); t_cols.append(tp); kept.append(m)
    if len(kept) < 2:
        return None
    return v_y_ref, np.column_stack(v_cols), t_y_ref, np.column_stack(t_cols), kept


def _metrics(y, p, n_bins=15):
    return {
        "auc": auc_metric(y, p),
        "ece": ece_equal_mass(y, p, n_bins=n_bins),
        "brier": brier_metric(y, p),
    }


def process_cell(dataset, fold):
    st = _stack_deep(dataset, fold)
    if st is None:
        return None
    vy, vmatrix, ty, tmatrix, kept = st
    try:
        vconc = concept_ids_for_valid(dataset, valid_fold=fold)
        tconc = concept_ids_for_test(dataset)
    except FileNotFoundError as e:
        print(f"  [skip] {dataset} f{fold}: concept-id lookup missing ({e})")
        return None
    if vconc.size != vy.size or tconc.size != ty.size:
        return None

    # Baselines
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
    best_pred = tmatrix[:, j_best]
    stack_pred = LogisticStackedBlender().fit(vmatrix, vy).predict(tmatrix)
    static_pred = StaticConceptWeights(n_min=30).fit(
        vmatrix, vy, valid_concepts=vconc,
    ).predict(tmatrix, test_concepts=tconc)

    # CCCE ablation grid: pre-cal x post-cal.
    # ``full`` = default (concept-aware Stage 1 + post-cal Stage 3);
    # ``no_s1`` = skip Stage 1 pre-calibration (StaticConceptWeights + post-cal);
    # ``global_s1`` = replace concept-aware isotonic with plain global isotonic;
    # ``no_s3`` = same as full but without Stage 3 post-cal.
    variants = {
        "full":       CCCE(n_min=30, pre_cal="concept", post_cal=True),
        "no_s1":      CCCE(n_min=30, pre_cal="none",    post_cal=True),
        "global_s1":  CCCE(n_min=30, pre_cal="global",  post_cal=True),
        "no_s3":      CCCE(n_min=30, pre_cal="concept", post_cal=False),
    }
    ccce_metrics = {}
    for tag, ens in variants.items():
        pred = ens.fit(vmatrix, vy, valid_concepts=vconc) \
                  .predict(tmatrix, test_concepts=tconc)
        ccce_metrics[tag] = _metrics(ty, pred)

    m_static = _metrics(ty, static_pred)
    m_stack  = _metrics(ty, stack_pred)
    m_mean   = _metrics(ty, mean_pred)
    m_best   = _metrics(ty, best_pred)

    row = {
        "dataset": dataset, "fold": fold,
        "models_in_subset": ",".join(kept),
        "best_single_model": kept[j_best],
        "best_single_selected_on": best_single_selected_on,
        "best_single_on_test_model": kept[j_best_test],
        "n_valid": int(vy.size),
        "n_test":  int(ty.size),
        "static_auc": m_static["auc"], "static_ece": m_static["ece"], "static_brier": m_static["brier"],
        "stack_auc":  m_stack["auc"],  "stack_ece":  m_stack["ece"],  "stack_brier":  m_stack["brier"],
        "mean_auc":   m_mean["auc"],   "mean_ece":   m_mean["ece"],   "mean_brier":   m_mean["brier"],
        "best_auc":   m_best["auc"],   "best_ece":   m_best["ece"],   "best_brier":   m_best["brier"],
    }
    for tag, m in ccce_metrics.items():
        row[f"ccce_{tag}_auc"]   = m["auc"]
        row[f"ccce_{tag}_ece"]   = m["ece"]
        row[f"ccce_{tag}_brier"] = m["brier"]
        row[f"lift_{tag}_vs_best_auc"]   = m["auc"] - m_best["auc"]
        row[f"lift_{tag}_vs_stack_auc"]  = m["auc"] - m_stack["auc"]
        row[f"lift_{tag}_vs_static_auc"] = m["auc"] - m_static["auc"]
        row[f"lift_{tag}_vs_best_ece"]   = m["ece"] - m_best["ece"]
        row[f"lift_{tag}_vs_stack_ece"]  = m["ece"] - m_stack["ece"]
        row[f"lift_{tag}_vs_static_ece"] = m["ece"] - m_static["ece"]
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--folds", nargs="+", type=int, default=FOLDS)
    ap.add_argument("--out", type=Path,
                    default=paths.ARTIFACTS_DIR / "ensembles" / "ccce_deep.csv")
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for ds in args.datasets:
        for fold in args.folds:
            r = process_cell(ds, fold)
            if r is None:
                print(f"[{ds:22s} f{fold}] SKIP"); continue
            best_tag = max(("full", "no_s1", "global_s1", "no_s3"),
                            key=lambda t: r[f"ccce_{t}_auc"])
            print(f"[{ds:22s} f{fold}] "
                  f"full ΔAUC={r['lift_full_vs_static_auc']*100:+.3f}%  "
                  f"no_s1 {r['lift_no_s1_vs_static_auc']*100:+.3f}%  "
                  f"global_s1 {r['lift_global_s1_vs_static_auc']*100:+.3f}%  "
                  f"no_s3 {r['lift_no_s3_vs_static_auc']*100:+.3f}%  "
                  f"— best: {best_tag}")
            rows.append(r)

    if not rows:
        print("no rows"); return
    keys = list(rows[0].keys())
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()})
    print(f"\nWrote {len(rows)} rows → {args.out}")


if __name__ == "__main__":
    main()
