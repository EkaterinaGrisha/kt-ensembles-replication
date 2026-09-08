"""AttentionGating (torch) — deep concept-level.

Trains one ``AttentionGating`` per (dataset, fold) on the deep concept-
level valid data, evaluates on test, and compares against:
  * ``mean`` — arithmetic mean simple ensemble.
  * ``stack`` — LogisticStackedBlender.
  * ``static_cg`` — StaticConceptWeights (Exp C1 representative).
  * ``ccce_global_s1`` — the §5.4 headline variant.
  * ``best_single`` — best deep model by test-fold AUC.

Torch device is auto-picked in priority CUDA > MPS > CPU. On Mac M-series
MPS gives ~5-10x over CPU; on Kaggle T4 CUDA gives another ~3-5x.

Outputs
-------
``artifacts/ensembles/attention_gating_deep.csv`` — one row per
(dataset, fold) with all six configuration metrics + lift columns.

CLI
---
    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
        OPENBLAS_NUM_THREADS=1 python -m \\
        scripts.run_attention_gating
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
          f"scripts.run_attention_gating [args]", file=_sys.stderr)

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from ktx import paths
from ktx.downstream import concept_ids_for_test, concept_ids_for_valid
from ktx.ensemble import (
    CCCE,
    ArithmeticMean,
    AttentionGating,
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


def _load_deep(dataset, model, fold):
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
        got = _load_deep(dataset, m, fold)
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
    return {"auc": auc_metric(y, p),
            "ece": ece_equal_mass(y, p, n_bins=n_bins),
            "brier": brier_metric(y, p)}


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
        vmatrix, vy, valid_concepts=vconc
    ).predict(tmatrix, test_concepts=tconc)
    ccce_pred = CCCE(n_min=30, pre_cal="global", post_cal=True).fit(
        vmatrix, vy, valid_concepts=vconc
    ).predict(tmatrix, test_concepts=tconc)

    # AttentionGating — the star
    t_attn = time.time()
    attn_pred = AttentionGating(n_epochs=15, seed=0).fit(
        vmatrix, vy, valid_concepts=vconc
    ).predict(tmatrix, test_concepts=tconc)
    dt_attn = time.time() - t_attn

    m = {"attn": _metrics(ty, attn_pred), "ccce": _metrics(ty, ccce_pred),
         "static": _metrics(ty, static_pred), "stack": _metrics(ty, stack_pred),
         "mean": _metrics(ty, mean_pred), "best": _metrics(ty, best_pred)}

    row = {
        "dataset": dataset, "fold": fold,
        "models_in_subset": ",".join(kept),
        "best_single_model": kept[j_best],
        "best_single_selected_on": best_single_selected_on,
        "best_single_on_test_model": kept[j_best_test],
        "n_valid": int(vy.size),
        "n_test":  int(ty.size),
        "attn_train_sec": round(dt_attn, 1),
    }
    for tag, met in m.items():
        row[f"{tag}_auc"] = met["auc"]
        row[f"{tag}_ece"] = met["ece"]
        row[f"{tag}_brier"] = met["brier"]
    for cmp in ("best", "mean", "stack", "static", "ccce"):
        row[f"lift_attn_vs_{cmp}_auc"] = m["attn"]["auc"] - m[cmp]["auc"]
        row[f"lift_attn_vs_{cmp}_ece"] = m["attn"]["ece"] - m[cmp]["ece"]
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--folds", nargs="+", type=int, default=FOLDS)
    ap.add_argument("--out", type=Path,
                    default=paths.ARTIFACTS_DIR / "ensembles" / "attention_gating_deep.csv")
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for ds in args.datasets:
        for fold in args.folds:
            r = process_cell(ds, fold)
            if r is None:
                print(f"[{ds:22s} f{fold}] SKIP"); continue
            print(f"[{ds:22s} f{fold}] attn AUC={r['attn_auc']:.4f}  "
                  f"vs static {r['lift_attn_vs_static_auc']*100:+.3f}%  "
                  f"vs ccce {r['lift_attn_vs_ccce_auc']*100:+.3f}%  "
                  f"vs best {r['lift_attn_vs_best_auc']*100:+.3f}%  "
                  f"[train {r['attn_train_sec']}s]")
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
