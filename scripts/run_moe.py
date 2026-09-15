"""MixtureOfExperts (torch) — deep concept-level.

Top-2 sparse gating over 4 deep components; loads/eval mirror
``run_attention_gating.py``. Baselines: mean, stack, static_cg, ccce
(global_s1 headline variant), best_single.

Torch device auto: CUDA > MPS > CPU. Load-balancing regularisation
(Shazeer 2017 eq. 12) prevents any single component from dominating
the top-2 selection across the batch.

Outputs
-------
``artifacts/ensembles/moe_deep.csv`` — one row per (dataset, fold) with
all metrics + lift columns vs each baseline.

CLI
---
    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
        OPENBLAS_NUM_THREADS=1 python -m scripts.run_moe
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
          f"scripts.run_moe [args]", file=_sys.stderr)

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from ktx import deep_inputs, paths
from ktx.ensemble import (
    CCCE,
    ArithmeticMean,
    LogisticStackedBlender,
    MixtureOfExperts,
    StaticConceptWeights,
)
from ktx.metrics import ece_equal_mass
from ktx.stats import auc_metric, brier_metric

DATASETS = ["assist2009", "assist2015", "assist2017", "algebra2005",
            "bridge2algebra2006", "assist2012", "ednet"]
FOLDS = [0, 1, 2, 3, 4]
DEEP_MODELS = ["dkt", "sakt", "akt", "simplekt"]
DEEP_ROOT = paths.ARTIFACTS_DIR / "predictions"


def _metrics(y, p, n_bins=15):
    return {"auc": auc_metric(y, p),
            "ece": ece_equal_mass(y, p, n_bins=n_bins),
            "brier": brier_metric(y, p)}


def process_cell(dataset, fold, k_top: int = 2, granularity: str = "concept"):
    try:
        cell = deep_inputs.load_cell(dataset, fold, DEEP_MODELS, granularity)
    except FileNotFoundError as e:
        print(f"  [skip] {dataset} f{fold}: concept-id lookup missing ({e})")
        return None
    if cell is None:
        return None
    vy, vmatrix, ty, tmatrix, kept = (cell.valid_y, cell.valid_matrix,
                                      cell.test_y, cell.test_matrix, cell.models)
    vconc, tconc = cell.valid_concepts, cell.test_concepts

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

    t_moe = time.time()
    moe_pred = MixtureOfExperts(k_top=k_top, n_epochs=15, seed=0).fit(
        vmatrix, vy, valid_concepts=vconc
    ).predict(tmatrix, test_concepts=tconc)
    dt_moe = time.time() - t_moe

    m = {"moe": _metrics(ty, moe_pred), "ccce": _metrics(ty, ccce_pred),
         "static": _metrics(ty, static_pred), "stack": _metrics(ty, stack_pred),
         "mean": _metrics(ty, mean_pred), "best": _metrics(ty, best_pred)}

    row = {
        "dataset": dataset, "fold": fold,
        "granularity": granularity,
        "models_in_subset": ",".join(kept),
        "best_single_model": kept[j_best],
        "best_single_selected_on": best_single_selected_on,
        "best_single_on_test_model": kept[j_best_test],
        "n_valid": int(vy.size),
        "n_test":  int(ty.size),
        "k_top": k_top,
        "moe_train_sec": round(dt_moe, 1),
    }
    for tag, met in m.items():
        row[f"{tag}_auc"] = met["auc"]
        row[f"{tag}_ece"] = met["ece"]
        row[f"{tag}_brier"] = met["brier"]
    for cmp in ("best", "mean", "stack", "static", "ccce"):
        row[f"lift_moe_vs_{cmp}_auc"] = m["moe"]["auc"] - m[cmp]["auc"]
        row[f"lift_moe_vs_{cmp}_ece"] = m["moe"]["ece"] - m[cmp]["ece"]
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--folds", nargs="+", type=int, default=FOLDS)
    ap.add_argument("--granularity", choices=["concept", "question"], default="concept",
                    help="уровень подробности строки: пара «задание, компонент» или задание")
    ap.add_argument("--k-top", type=int, default=2)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    if args.out is None:
        suffix = "" if args.granularity == "concept" else "_question"
        args.out = paths.ARTIFACTS_DIR / "ensembles" / f"moe_deep{suffix}.csv"
    args.out.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for ds in args.datasets:
        for fold in args.folds:
            r = process_cell(ds, fold, k_top=args.k_top, granularity=args.granularity)
            if r is None:
                print(f"[{ds:22s} f{fold}] SKIP"); continue
            print(f"[{ds:22s} f{fold}] moe AUC={r['moe_auc']:.4f}  "
                  f"vs static {r['lift_moe_vs_static_auc']*100:+.3f}%  "
                  f"vs ccce {r['lift_moe_vs_ccce_auc']*100:+.3f}%  "
                  f"vs best {r['lift_moe_vs_best_auc']*100:+.3f}%  "
                  f"[train {r['moe_train_sec']}s]")
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
