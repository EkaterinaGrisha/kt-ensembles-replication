"""Взвешивание по компонентам знания для семейства простых моделей.

У глубоких моделей этот расчёт уже есть (`run_gating.py`), а у простых его не
было: строка их предсказаний — задание, а не пара «задание, компонент», и
компонент знания приходилось откуда-то брать. Он берётся из разметки,
восстановленной `classical_concepts.py` со сверкой ответов построчно.

Считаются те же три варианта, что и у глубоких моделей: полное взвешивание со
своим набором весов на каждый компонент знания; вариант, где наклоны общие, а
свой у компонента только свободный член; обучаемый распределитель весов.
Точки отсчёта тоже те же — лучшая одиночная модель, выбранная по валидационной
части, усреднение и обучаемая надстройка.

Выход: `artifacts/ensembles/gating_classical.csv`, схема совпадает с
`gating_deep.csv`, чтобы таблица статьи собиралась из обоих одинаково.

Запуск:
  KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      OPENBLAS_NUM_THREADS=1 python -m scripts.run_gating_classical
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
          f"  {prefix} python -m scripts.run_gating_classical", file=_sys.stderr)

import argparse
import csv
from pathlib import Path

import numpy as np

from ktx import paths
from ktx.ensemble import (
    ArithmeticMean,
    GlobalStackWithConceptIntercept,
    LinearGating,
    LogisticStackedBlender,
    StaticConceptWeights,
)
from ktx.metrics import ece_equal_mass
from ktx.stats import auc_metric, brier_metric

DATASETS = ["assist2009", "assist2015", "assist2017", "algebra2005",
            "bridge2algebra2006", "assist2012", "ednet"]
FOLDS = [0, 1, 2, 3, 4]
MODELS = ["bkt", "pfa", "pfa_recency", "elorasch"]
HEADS = ["static_concept_weights", "global_stack_concept_intercept", "linear_gating"]

PRED = paths.ARTIFACTS_DIR / "predictions"
VALID_CACHE = paths.ARTIFACTS_DIR / "predictions_valid_cache"
CONCEPTS = paths.ARTIFACTS_DIR / "ensembles" / "classical_concepts"


def _load(dataset: str, model: str, fold: int):
    test_path = PRED / dataset / f"{model}_fold{fold}.npz"
    valid_path = VALID_CACHE / f"{dataset}__{model}__fold{fold}.npz"
    if not test_path.exists() or not valid_path.exists():
        return None
    t = np.load(test_path)
    v = np.load(valid_path)
    if not {"y_true", "y_prob"} <= set(t.files):
        return None
    if not {"valid_y_true", "valid_y_prob"} <= set(v.files):
        return None
    return (v["valid_y_true"].astype(int), v["valid_y_prob"].astype(np.float64),
            t["y_true"].astype(int), t["y_prob"].astype(np.float64))


def _stack(dataset: str, fold: int):
    v_ref = t_ref = None
    v_cols, t_cols, kept = [], [], []
    for m in MODELS:
        got = _load(dataset, m, fold)
        if got is None:
            continue
        vy, vp, ty, tp = got
        if v_ref is None:
            v_ref, t_ref = vy, ty
        elif not np.array_equal(vy, v_ref) or not np.array_equal(ty, t_ref):
            print(f"  [warn] {dataset} f{fold}: у {m} другие ответы, ячейка пропущена")
            return None
        v_cols.append(vp)
        t_cols.append(tp)
        kept.append(m)
    if len(kept) < 2:
        return None
    return v_ref, np.column_stack(v_cols), t_ref, np.column_stack(t_cols), kept


def _concepts(dataset: str, fold: int):
    p = CONCEPTS / f"{dataset}.npz"
    if not p.exists():
        return None, None
    d = np.load(p)
    key = f"valid_cid_{fold}"
    if "test_cid" not in d.files or key not in d.files:
        return None, None
    return d[key].astype(np.int64), d["test_cid"].astype(np.int64)


def _metrics(y, p, n_bins: int = 15) -> dict[str, float]:
    return {"auc": auc_metric(y, p), "ece": ece_equal_mass(y, p, n_bins=n_bins),
            "brier": brier_metric(y, p)}


def process_cell(dataset: str, fold: int) -> list[dict]:
    st = _stack(dataset, fold)
    if st is None:
        print(f"  [пропуск] {dataset} f{fold}: нет предсказаний")
        return []
    vy, vmatrix, ty, tmatrix, kept = st
    vconc, tconc = _concepts(dataset, fold)
    if vconc is None:
        print(f"  [пропуск] {dataset} f{fold}: нет разметки компонентов")
        return []
    if vconc.size != vy.size or tconc.size != ty.size:
        print(f"  [пропуск] {dataset} f{fold}: размеры разметки не сходятся "
              f"({vconc.size} против {vy.size}, {tconc.size} против {ty.size})")
        return []

    def _argbest(y, mat) -> int:
        a = [auc_metric(y, mat[:, j]) for j in range(mat.shape[1])]
        return sorted(range(len(kept)), key=lambda j: (-a[j], kept[j]))[0]

    j_best_test = _argbest(ty, tmatrix)
    j_best = _argbest(vy, vmatrix)
    m_best = _metrics(ty, tmatrix[:, j_best])
    m_mean = _metrics(ty, ArithmeticMean().predict(tmatrix))
    stack_pred = LogisticStackedBlender().fit(vmatrix, vy).predict(tmatrix)
    m_stack = _metrics(ty, stack_pred)

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
                pred = ens.predict(tmatrix, context=tconc.reshape(-1, 1).astype(np.float64))
                n_fit = -1
        except Exception as e:
            print(f"  [warn] {dataset} f{fold} {head}: не обучилось ({e})")
            continue
        m_g = _metrics(ty, pred)
        rows.append({
            "dataset": dataset, "fold": fold, "meta_learner": head,
            "models_in_subset": ",".join(kept),
            "n_valid": int(vy.size), "n_test": int(ty.size),
            "n_concepts_fit": int(n_fit),
            "gated_auc": m_g["auc"], "gated_ece": m_g["ece"], "gated_brier": m_g["brier"],
            "mean_auc": m_mean["auc"], "mean_ece": m_mean["ece"],
            "mean_brier": m_mean["brier"],
            "stacked_auc": m_stack["auc"], "stacked_ece": m_stack["ece"],
            "stacked_brier": m_stack["brier"],
            "best_single_model": kept[j_best],
            "best_single_selected_on": "valid",
            "best_single_on_test_model": kept[j_best_test],
            "best_single_auc": m_best["auc"], "best_single_ece": m_best["ece"],
            "best_single_brier": m_best["brier"],
            "lift_gated_vs_mean_auc": m_g["auc"] - m_mean["auc"],
            "lift_gated_vs_best_auc": m_g["auc"] - m_best["auc"],
            "lift_gated_vs_stack_auc": m_g["auc"] - m_stack["auc"],
            "lift_gated_vs_mean_ece": m_g["ece"] - m_mean["ece"],
            "lift_gated_vs_best_ece": m_g["ece"] - m_best["ece"],
            "lift_gated_vs_stack_ece": m_g["ece"] - m_stack["ece"],
        })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--folds", nargs="+", type=int, default=FOLDS)
    ap.add_argument("--out", type=Path,
                    default=paths.ARTIFACTS_DIR / "ensembles" / "gating_classical.csv")
    args = ap.parse_args()

    rows: list[dict] = []
    for ds in args.datasets:
        for fold in args.folds:
            cell = process_cell(ds, fold)
            if cell:
                lifts = ", ".join(f"{r['meta_learner'].split('_')[0]} "
                                  f"{r['lift_gated_vs_best_auc']:+.4f}" for r in cell)
                print(f"[{ds:22s} f{fold}] {lifts}")
            rows.extend(cell)
    if not rows:
        _sys.exit("ФАТАЛЬНО: ни одной посчитанной ячейки")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 6) if isinstance(v, float) else v)
                        for k, v in r.items()})
    print(f"{args.out}: {len(rows)} строк")
    return 0


if __name__ == "__main__":
    _sys.exit(main())
