"""Кластерный бутстрап для взвешивания по компонентам знания у простых моделей.

Зеркало `cluster_bootstrap_gating.py`, но на семействе простых моделей: те же
три варианта взвешивания, те же две точки отсчёта — лучшая одиночная модель,
выбранная по валидационной части, и обучаемая надстройка. Компонент знания для
строки берётся из разметки, восстановленной `classical_concepts.py`.

Учащийся для кластеризации берётся из поля `groups` файла предсказаний: у
простых моделей строка — задание, и все задания одного учащегося попадают в
перевыборку вместе.

Выход
-----
`artifacts/ensembles/cluster_bootstrap_gating_classical.csv` — строка на
(набор, разбиение, вариант взвешивания); рядом `..._pooled.csv`.

Запуск
------
    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
        OPENBLAS_NUM_THREADS=1 python -m \
        scripts.cluster_bootstrap_gating_classical
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
    print(f"[warn] не выставлены переменные окружения {_missing}. Запускать так:\n"
          f"  {prefix} python -m scripts.cluster_bootstrap_gating_classical",
          file=_sys.stderr)

import argparse
import csv
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

from ktx import paths
from ktx.bootstrap_fast import auc_metric_fast, paired_bootstrap_fast
from ktx.ensemble import (
    GlobalStackWithConceptIntercept,
    LinearGating,
    LogisticStackedBlender,
    StaticConceptWeights,
)
from ktx.metrics import ece_equal_mass

DATASETS = ["assist2009", "assist2015", "assist2017", "algebra2005",
            "bridge2algebra2006", "assist2012", "ednet"]
FOLDS = [0, 1, 2, 3, 4]
MODELS = ["bkt", "pfa", "pfa_recency", "elorasch"]
HEADS = ["static_concept_weights", "global_stack_concept_intercept", "linear_gating"]

PRED = paths.ARTIFACTS_DIR / "predictions"
VALID_CACHE = paths.ARTIFACTS_DIR / "predictions_valid_cache"
CONCEPTS = paths.ARTIFACTS_DIR / "ensembles" / "classical_concepts"

auc_metric = auc_metric_fast


def ece_metric(y_true, y_prob) -> float:
    return ece_equal_mass(y_true, y_prob, n_bins=15)


def _stack(dataset: str, fold: int):
    v_ref = t_ref = groups = None
    v_cols, t_cols, kept = [], [], []
    for m in MODELS:
        tp = PRED / dataset / f"{m}_fold{fold}.npz"
        vp = VALID_CACHE / f"{dataset}__{m}__fold{fold}.npz"
        if not tp.exists() or not vp.exists():
            continue
        t, v = np.load(tp), np.load(vp)
        if not {"y_true", "y_prob"} <= set(t.files):
            continue
        if not {"valid_y_true", "valid_y_prob"} <= set(v.files):
            continue
        vy = v["valid_y_true"].astype(int)
        ty = t["y_true"].astype(int)
        if v_ref is None:
            v_ref, t_ref = vy, ty
            groups = t["groups"] if "groups" in t.files and t["groups"].size else None
        elif not np.array_equal(vy, v_ref) or not np.array_equal(ty, t_ref):
            return None
        v_cols.append(v["valid_y_prob"].astype(np.float64))
        t_cols.append(t["y_prob"].astype(np.float64))
        kept.append(m)
    if len(kept) < 2:
        return None
    return v_ref, np.column_stack(v_cols), t_ref, np.column_stack(t_cols), groups, kept


def _concepts(dataset: str, fold: int):
    p = CONCEPTS / f"{dataset}.npz"
    if not p.exists():
        return None, None
    d = np.load(p)
    key = f"valid_cid_{fold}"
    if "test_cid" not in d.files or key not in d.files:
        return None, None
    return d[key].astype(np.int64), d["test_cid"].astype(np.int64)


def bootstrap_cell(dataset: str, fold: int, n_boot: int, seed: int) -> list[dict]:
    st = _stack(dataset, fold)
    if st is None:
        return []
    vy, vmatrix, ty, tmatrix, groups, kept = st
    vconc, tconc = _concepts(dataset, fold)
    if vconc is None or vconc.size != vy.size or tconc.size != ty.size:
        return []

    aucs = [auc_metric(vy, vmatrix[:, j]) for j in range(vmatrix.shape[1])]
    j_best = sorted(range(len(kept)), key=lambda j: (-aucs[j], kept[j]))[0]
    best_pred = tmatrix[:, j_best]
    stack_pred = LogisticStackedBlender().fit(vmatrix, vy).predict(tmatrix)

    rows = []
    for head in HEADS:
        try:
            if head == "static_concept_weights":
                ens = StaticConceptWeights(n_min=30).fit(vmatrix, vy, valid_concepts=vconc)
                pred = ens.predict(tmatrix, test_concepts=tconc)
                n_fit = ens.n_concepts_fit_
            elif head == "global_stack_concept_intercept":
                ens = GlobalStackWithConceptIntercept(n_min=30).fit(
                    vmatrix, vy, valid_concepts=vconc)
                pred = ens.predict(tmatrix, test_concepts=tconc)
                n_fit = len(ens.concept_intercepts_)
            else:
                ens = LinearGating().fit(vmatrix, vy,
                                         context=vconc.reshape(-1, 1).astype(np.float64))
                pred = ens.predict(tmatrix,
                                   context=tconc.reshape(-1, 1).astype(np.float64))
                n_fit = -1
        except Exception as e:
            print(f"  [warn] {dataset} f{fold} {head}: не обучилось ({e})")
            continue

        r_best_auc = paired_bootstrap_fast(ty, pred, best_pred, auc_metric,
                                           n_boot=n_boot, groups=groups, seed=seed,
                                           metric_name="auc")
        r_best_ece = paired_bootstrap_fast(ty, pred, best_pred, ece_metric,
                                           n_boot=n_boot, groups=groups, seed=seed + 1,
                                           metric_name="ece")
        r_stk_auc = paired_bootstrap_fast(ty, pred, stack_pred, auc_metric,
                                          n_boot=n_boot, groups=groups, seed=seed + 2,
                                          metric_name="auc")
        r_stk_ece = paired_bootstrap_fast(ty, pred, stack_pred, ece_metric,
                                          n_boot=n_boot, groups=groups, seed=seed + 3,
                                          metric_name="ece")
        rows.append({
            "dataset": dataset, "fold": fold, "meta_learner": head,
            "best_single_model": kept[j_best],
            "best_single_selected_on": "valid",
            "n_test": int(ty.size),
            "n_students": int(np.unique(groups).size) if groups is not None else -1,
            "bootstrap_method": "cluster" if groups is not None else "instance",
            "n_concepts_fit": int(n_fit),
            "gated_auc": r_best_auc.value_a,
            "gated_ece": r_best_ece.value_a,
            "best_single_auc": r_best_auc.value_b,
            "delta_auc_vs_best": r_best_auc.diff,
            "auc_ci_low_vs_best": r_best_auc.ci_low,
            "auc_ci_high_vs_best": r_best_auc.ci_high,
            "auc_p_vs_best": r_best_auc.p_value,
            "best_single_ece": r_best_ece.value_b,
            "delta_ece_vs_best": r_best_ece.diff,
            "ece_p_vs_best": r_best_ece.p_value,
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


def pool_folds(per_fold: list[dict]) -> list[dict]:
    buckets = defaultdict(list)
    for r in per_fold:
        buckets[(r["dataset"], r["meta_learner"])].append(r)
    pooled = []
    for (ds, head), rows in sorted(buckets.items()):
        entry = {"dataset": ds, "meta_learner": head, "n_folds": len(rows)}
        for base in ("best", "stack"):
            d = np.array([r[f"delta_auc_vs_{base}"] for r in rows])
            p = np.array([r[f"auc_p_vs_{base}"] for r in rows])
            entry[f"mean_delta_auc_vs_{base}"] = float(d.mean())
            entry[f"std_delta_auc_vs_{base}"] = float(d.std(ddof=1)) if len(d) > 1 else 0.0
            entry[f"min_auc_p_vs_{base}"] = float(p.min())
            entry[f"n_folds_sig_auc_vs_{base}_05"] = int((p <= 0.05).sum())
        entry["mean_delta_ece_vs_best"] = float(
            np.mean([r["delta_ece_vs_best"] for r in rows]))
        pooled.append(entry)
    return pooled


def _write(path: Path, rows: list[dict]) -> None:
    if not rows:
        print(f"[warn] нечего писать в {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 6) if isinstance(v, float) else v)
                        for k, v in r.items()})
    print(f"{path}: {len(rows)} строк")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--folds", nargs="+", type=int, default=FOLDS)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--out", type=Path,
                    default=paths.ARTIFACTS_DIR / "ensembles"
                    / "cluster_bootstrap_gating_classical.csv")
    ap.add_argument("--pooled-out", type=Path,
                    default=paths.ARTIFACTS_DIR / "ensembles"
                    / "cluster_bootstrap_gating_classical_pooled.csv")
    args = ap.parse_args()

    t0 = time.time()
    cells = [(ds, fold) for ds in args.datasets for fold in args.folds]

    def _run(ds, fold):
        t = time.time()
        return ds, fold, bootstrap_cell(ds, fold, args.n_boot, args.seed), time.time() - t

    results = Parallel(n_jobs=args.n_jobs, backend="loky", verbose=10)(
        delayed(_run)(*c) for c in cells)

    all_rows = []
    for ds, fold, rows, dt in results:
        if not rows:
            print(f"[{ds:22s} f{fold}] пропущено [{dt:.1f} с]")
            continue
        summary = ", ".join(f"{r['meta_learner'].split('_')[0]} "
                            f"Δ={r['delta_auc_vs_best']:+.4f} p={r['auc_p_vs_best']:.2g}"
                            for r in rows)
        print(f"[{ds:22s} f{fold}] [{dt:.1f} с] {summary}")
        all_rows.extend(rows)

    _write(args.out, all_rows)
    _write(args.pooled_out, pool_folds(all_rows))
    print(f"всего {time.time() - t0:.0f} с")
    return 0


if __name__ == "__main__":
    _sys.exit(main())
