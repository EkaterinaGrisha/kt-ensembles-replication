"""LogisticStackedBlender within family.

For every (dataset, subset, fold) cell in the within-family matrix — the
same 70 cells as ``run_simple_ensembles.py`` — this script:

  1. Loads the k component-model valid-fold predictions (question-level for
     both families: classical NPZ ``valid_y_true``/``valid_y_prob`` from the
     valid cache; deep NPZ ``valid_y_true_q``/``valid_y_prob_q`` from the
     question-level aggregation).
  2. Fits a logistic-regression meta-learner on those valid predictions.
  3. Applies the meta-learner to the test-fold component predictions and
     records the resulting AUC / ECE / Brier alongside the same-cell
     ``arithmetic_mean`` and ``best single`` baselines.

Cross-family stacking (heterogeneous 8-model subset) is deferred until the
pyKT-cidx patch (backlog task #13) lands; see ``ktx/alignment.py`` warning
block for the reason.

Outputs
-------
``artifacts/ensembles/stacking_logistic.csv`` — one row per
(dataset, fold, subset). Columns:

    dataset, fold, subset, models_in_subset,
    n_valid, n_test,
    stacked_auc, stacked_ece, stacked_brier,
    mean_auc,    mean_ece,    mean_brier,
    best_single_model, best_single_auc, best_single_ece, best_single_brier,
    lift_stacked_vs_mean_auc, lift_stacked_vs_best_auc,
    lift_stacked_vs_mean_ece, lift_stacked_vs_best_ece,
    weight_<model_i>...  (learned logistic weights, one column per component)

CLI
---
    python -m scripts.run_stacking
    python -m scripts.run_stacking --datasets algebra2005 --folds 0
"""
from __future__ import annotations

# macOS OpenMP conflict guard. xgboost's libomp and sklearn's libomp both
# initialise pthread mutexes at library load and can collide (OMP error #179
# pthread_mutex_init) on macOS ARM. Setting KMP_DUPLICATE_LIB_OK=TRUE tells
# the loader it's fine — this MUST come from the process environment at
# spawn time, not from an in-Python os.environ assignment (libomp is loaded
# at import time by numpy / scipy transitively, before this file gets to run
# its top-level statements). Invoke this script as:
#
#     KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m scripts.run_stacking [...]
#
# The runtime check below warns if the caller forgot.
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
          f"Re-run as:\n  {prefix} python -m scripts.run_stacking [args]",
          file=_sys.stderr)

import argparse
import csv
from pathlib import Path

import numpy as np

from ktx import paths
from ktx.ensemble import STACKED_BLENDERS, ArithmeticMean
from ktx.metrics import ece_equal_mass
from ktx.stats import auc_metric, brier_metric

DATASETS = ["assist2009", "assist2015", "assist2017", "algebra2005",
            "bridge2algebra2006", "assist2012", "ednet"]
FOLDS = [0, 1, 2, 3, 4]
SUBSETS: dict[str, list[str]] = {
    "classical": ["bkt", "pfa", "pfa_recency", "elorasch"],
    "deep":      ["dkt", "sakt", "akt", "simplekt"],
}

CLASSICAL_CACHE = paths.ARTIFACTS_DIR / "predictions_valid_cache"
DEEP_ROOT = paths.ARTIFACTS_DIR / "predictions"


def _load_classical(dataset: str, model: str, fold: int
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    p = CLASSICAL_CACHE / f"{dataset}__{model}__fold{fold}.npz"
    if not p.exists():
        return None
    d = np.load(p)
    return (d["valid_y_true"].astype(int), d["valid_y_prob"].astype(np.float64),
            d["y_true"].astype(int),        d["y_prob"].astype(np.float64))


def _load_deep(dataset: str, model: str, fold: int,
               valid_source: str = "v2_qlvl_late_mean"
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Return question-level (valid_y, valid_p, test_y, test_p) for a deep model.

    ``valid_source`` selects which q-level valid probability object to fit
    the meta-learner on. Test side is always y_true / y_prob (pyKT
    evaluate_question, leak-free by construction).

      * ``"v4_pykt_evaluate_question"`` (audit T1b, leak-free): uses
        ``valid_y_*_q_pykt`` — pyKT ``evaluate_question`` on the freshly-
        generated valid-question loader (same cq/cshft inference path as
        test). Coherent leak-free pipeline; the correct default for any
        new work post-2026-08-26.
      * ``"v2_qlvl_late_mean"`` (legacy default): uses ``valid_y_*_q`` —
        late_mean aggregation over concept_y_prob emitted by
        aggregate_deep_valid_to_question.py. Inherits pyKT concept-
        unrolling teacher-forcing leakage on ``is_repeat=1`` positions
        (see notes/24_t1a_leakage_finding.md); kept as default for
        reproducibility of the earlier stacking tables.
    """
    p = DEEP_ROOT / dataset / f"{model}_fold{fold}.npz"
    if not p.exists():
        return None
    d = np.load(p)
    if valid_source == "v4_pykt_evaluate_question":
        need = {"valid_y_true_q_pykt", "valid_y_prob_q_pykt", "y_true", "y_prob"}
        if not need <= set(d.files):
            return None
        return (d["valid_y_true_q_pykt"].astype(int),
                d["valid_y_prob_q_pykt"].astype(np.float64),
                d["y_true"].astype(int), d["y_prob"].astype(np.float64))
    if valid_source == "v2_qlvl_late_mean":
        need = {"valid_y_true_q", "valid_y_prob_q", "y_true", "y_prob"}
        if not need <= set(d.files):
            return None
        return (d["valid_y_true_q"].astype(int),
                d["valid_y_prob_q"].astype(np.float64),
                d["y_true"].astype(int), d["y_prob"].astype(np.float64))
    raise ValueError(f"unknown valid_source: {valid_source}")


def _stack_subset(dataset: str, subset_name: str, models: list[str], fold: int,
                   valid_source: str = "v2_qlvl_late_mean"):
    """Return (valid_y, valid_matrix, test_y, test_matrix, kept_models) or None."""
    if subset_name == "classical":
        loader = lambda ds_, m_, f_: _load_classical(ds_, m_, f_)
    else:
        loader = lambda ds_, m_, f_: _load_deep(ds_, m_, f_, valid_source=valid_source)
    v_y_ref = t_y_ref = None
    v_cols: list[np.ndarray] = []
    t_cols: list[np.ndarray] = []
    kept: list[str] = []
    for m in models:
        got = loader(dataset, m, fold)
        if got is None:
            continue
        vy, vp, ty, tp = got
        if v_y_ref is None:
            v_y_ref, t_y_ref = vy, ty
        else:
            if not np.array_equal(vy, v_y_ref) or not np.array_equal(ty, t_y_ref):
                print(f"  [warn] {dataset} f{fold} {subset_name}: {m} y arrays disagree; skip subset")
                return None
        v_cols.append(vp); t_cols.append(tp); kept.append(m)
    if len(kept) < 2 or v_y_ref is None or t_y_ref is None:
        return None
    return v_y_ref, np.column_stack(v_cols), t_y_ref, np.column_stack(t_cols), kept


def _metrics(y, p, n_bins: int = 15) -> dict[str, float]:
    return {
        "auc": auc_metric(y, p),
        "ece": ece_equal_mass(y, p, n_bins=n_bins),
        "brier": brier_metric(y, p),
    }


def _weights_column(stacker, kept: list[str]) -> dict[str, float]:
    """Return per-component weight dict when available; NaN otherwise.

    ``LogisticStackedBlender`` / ``RidgeStackedBlender`` expose logit-space
    coefficients via ``.weights_``. ``BayesianModelAveraging`` exposes
    probability-space convex-combination weights (same attribute name,
    different interpretation — noted in the CSV meta_learner column).
    ``XGBoostStackedBlender`` and ``MLPStackedBlender`` do not have a
    single per-model weight (nonlinear meta-learners) — fill NaN.
    """
    weights: dict[str, float] = {}
    if hasattr(stacker, "weights_") and stacker.weights_ is not None:
        try:
            arr = np.asarray(stacker.weights_).ravel()
            if arr.size == len(kept):
                for j, m in enumerate(kept):
                    weights[f"weight_{m}"] = float(arr[j])
                return weights
        except Exception:
            pass
    for m in kept:
        weights[f"weight_{m}"] = float("nan")
    return weights


def process_cell(dataset: str, fold: int, subset_name: str, models: list[str],
                 meta_learners: list[str],
                 valid_source: str = "v2_qlvl_late_mean") -> list[dict]:
    """Run every meta-learner in ``meta_learners`` on this cell. Returns one
    row per meta-learner. Baselines (mean, best_single) are computed once and
    joined to every row."""
    st = _stack_subset(dataset, subset_name, models, fold, valid_source=valid_source)
    if st is None:
        return []
    vy, vmatrix, ty, tmatrix, kept = st

    # Baselines (same across every meta-learner on this cell)
    mean_pred = ArithmeticMean().predict(tmatrix)

    # The best-single baseline must be chosen on VALID.
    # Choosing it on test makes it the maximum of several noisy test AUCs, i.e.
    # systematically inflated, which biases every lift_stacked_vs_best_* downward
    # and leaves the cluster bootstrap treating a post-selection quantity as fixed.
    # T3c fixed this in run_simple_ensembles.py on 2026-08-25 but never touched this
    # file, so 19 of 20 ensemble artifacts still carried a test-selected baseline.
    # Both numbers are reported: the valid-selected one is the result, the
    # test-selected one is kept as a witness so the pre-fix tables stay auditable.
    def _argbest(y_true, mat) -> int:
        a = [auc_metric(y_true, mat[:, j]) for j in range(mat.shape[1])]
        return sorted(range(len(kept)), key=lambda j: (-a[j], kept[j]))[0]

    j_best_test = _argbest(ty, tmatrix)
    if vy is not None and vmatrix is not None and vmatrix.shape[1] == tmatrix.shape[1]:
        j_best, selected_on = _argbest(vy, vmatrix), "valid"
    else:
        j_best, selected_on = j_best_test, "test_fallback"

    single_pred = tmatrix[:, j_best]
    m_mean = _metrics(ty, mean_pred)
    m_best = _metrics(ty, single_pred)
    m_best_test = _metrics(ty, tmatrix[:, j_best_test])

    rows: list[dict] = []
    for ml_name in meta_learners:
        cls = STACKED_BLENDERS[ml_name]
        try:
            stacker = cls().fit(vmatrix, vy)
            stacked_pred = stacker.predict(tmatrix)
        except Exception as e:
            print(f"  [warn] {dataset} f{fold} {subset_name} {ml_name}: fit failed ({e})")
            continue
        m_stack = _metrics(ty, stacked_pred)
        row: dict = {
            "dataset": dataset,
            "fold": fold,
            "subset": subset_name,
            "meta_learner": ml_name,
            "models_in_subset": ",".join(kept),
            "n_valid": int(vy.size),
            "n_test":  int(ty.size),
            "stacked_auc": m_stack["auc"],
            "stacked_ece": m_stack["ece"],
            "stacked_brier": m_stack["brier"],
            "mean_auc":  m_mean["auc"],
            "mean_ece":  m_mean["ece"],
            "mean_brier": m_mean["brier"],
            "best_single_model": kept[j_best],
            "best_single_selected_on": selected_on,
            "best_single_auc": m_best["auc"],
            "best_single_ece": m_best["ece"],
            "best_single_brier": m_best["brier"],
            # N1 witness: the pre-fix, test-selected baseline.
            "best_single_on_test_model": kept[j_best_test],
            "best_single_on_test_auc": m_best_test["auc"],
            "lift_stacked_vs_test_best_auc": m_stack["auc"] - m_best_test["auc"],
            "lift_stacked_vs_mean_auc": m_stack["auc"] - m_mean["auc"],
            "lift_stacked_vs_best_auc": m_stack["auc"] - m_best["auc"],
            "lift_stacked_vs_mean_ece": m_stack["ece"] - m_mean["ece"],
            "lift_stacked_vs_best_ece": m_stack["ece"] - m_best["ece"],
        }
        row.update(_weights_column(stacker, kept))
        rows.append(row)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--folds", nargs="+", type=int, default=FOLDS)
    ap.add_argument("--subsets", nargs="+", default=list(SUBSETS.keys()),
                    choices=list(SUBSETS.keys()))
    ap.add_argument("--meta-learners", nargs="+", default=list(STACKED_BLENDERS.keys()),
                    choices=list(STACKED_BLENDERS.keys()),
                    help="which meta-learners to run (default = all 5)")
    ap.add_argument("--out", type=Path,
                    default=paths.ARTIFACTS_DIR / "ensembles" / "stacking_all.csv")
    ap.add_argument("--valid-source",
                    choices=["v4_pykt_evaluate_question", "v2_qlvl_late_mean"],
                    default="v2_qlvl_late_mean",
                    help=("Fit-side q-level probability object for deep "
                          "subsets. v4 = leak-free pyKT evaluate_question "
                          "(audit T1b); v2 = leaked late_mean over "
                          "concept_y_prob. The default preserves earlier "
                          "reproducibility of the stacking tables."))
    args = ap.parse_args()

    # Auto-suffix default out path when v4 is selected (only if user did not
    # override --out explicitly).
    if (args.valid_source == "v4_pykt_evaluate_question" and
            args.out == paths.ARTIFACTS_DIR / "ensembles" / "stacking_all.csv"):
        args.out = paths.ARTIFACTS_DIR / "ensembles" / "stacking_all_v4.csv"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for ds in args.datasets:
        for fold in args.folds:
            for subset_name in args.subsets:
                cell_rows = process_cell(ds, fold, subset_name,
                                          SUBSETS[subset_name],
                                          args.meta_learners,
                                          valid_source=args.valid_source)
                if not cell_rows:
                    print(f"[{ds:22s} f{fold} {subset_name:9s}] SKIP")
                    continue
                # Compact log — one line per (dataset, fold, subset) cell,
                # showing per-meta-learner lift over the best single model
                lifts = ", ".join(
                    f"{r['meta_learner']}={r['lift_stacked_vs_best_auc']*100:+.3f}%"
                    for r in cell_rows
                )
                print(f"[{ds:22s} f{fold} {subset_name:9s}] "
                      f"best={cell_rows[0]['best_single_model']:12s} "
                      f"({cell_rows[0]['best_single_auc']:.4f}) | ΔAUC: {lifts}")
                rows.extend(cell_rows)

    if not rows:
        print("no rows written")
        return

    all_keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in all_keys:
                all_keys.append(k)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_keys)
        w.writeheader()
        for r in rows:
            row = {k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()}
            w.writerow(row)
    print(f"\nWrote {len(rows)} rows → {args.out}")


if __name__ == "__main__":
    main()
