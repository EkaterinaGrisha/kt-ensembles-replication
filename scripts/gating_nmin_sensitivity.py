"""Чувствительность взвешивания к порогу, при котором компонент получает веса.

Взвешивание по компонентам знания настраивает собственную линейную надстройку
внутри каждого компонента, но только там, где наблюдений хватает; остальные
пользуются общими весами. Порог взят по умолчанию — тридцать наблюдений, — и на
нём держится главный вывод работы: полное взвешивание не даёт ничего сверх
варианта с одним свободным членом. Если порог низкий, полное взвешивание
переобучается на редких темах, и тогда вывод — свойство порога, а не схемы.

Сценарий считает оба варианта при нескольких порогах и записывает, как меняется
разрыв между ними. Бутстрап не гоняется: вопрос здесь не в значимости, а в том,
меняет ли порог картину.

Выход: `artifacts/ensembles/gating_nmin_sensitivity.csv`.

Запуск:
  KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      OPENBLAS_NUM_THREADS=1 python -m scripts.gating_nmin_sensitivity
"""
from __future__ import annotations

import os as _os
import sys as _sys

for _k, _v in {"KMP_DUPLICATE_LIB_OK": "TRUE", "OMP_NUM_THREADS": "1",
               "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}.items():
    _os.environ.setdefault(_k, _v)

import argparse
import csv
from pathlib import Path

import numpy as np

from ktx import paths
from ktx.ensemble import GlobalStackWithConceptIntercept, StaticConceptWeights
from ktx.stats import auc_metric
from scripts.run_gating_classical import _concepts as classical_concepts
from scripts.run_gating_classical import _stack as classical_stack
from ktx import deep_inputs

DATASETS = ["algebra2005", "assist2009", "assist2012", "assist2015",
            "assist2017", "bridge2algebra2006", "ednet"]
FOLDS = [0, 1, 2, 3, 4]
DEEP = ["dkt", "sakt", "akt", "simplekt"]
THRESHOLDS = [10, 30, 100]
OUT = paths.ARTIFACTS_DIR / "ensembles" / "gating_nmin_sensitivity.csv"


def cell_inputs(dataset: str, fold: int, subset: str):
    """Матрицы, коды компонентов и ответы для одной ячейки любого семейства."""
    if subset == "deep":
        got = deep_inputs.load(dataset, fold, DEEP, "concept")
        if got is None:
            return None
        from ktx.concept_ids import concept_ids_for_test, concept_ids_for_valid
        try:
            vconc = concept_ids_for_valid(dataset, valid_fold=fold)
            tconc = concept_ids_for_test(dataset)
        except FileNotFoundError:
            return None
        if vconc.size != got.valid_y.size or tconc.size != got.test_y.size:
            return None
        return got.valid_y, got.valid_matrix, vconc, got.test_y, got.test_matrix, tconc
    st = classical_stack(dataset, fold)
    if st is None:
        return None
    vy, vmatrix, ty, tmatrix, _ = st
    vconc, tconc = classical_concepts(dataset, fold)
    if vconc is None or vconc.size != vy.size or tconc.size != ty.size:
        return None
    return vy, vmatrix, vconc, ty, tmatrix, tconc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--folds", nargs="+", type=int, default=FOLDS)
    ap.add_argument("--thresholds", nargs="+", type=int, default=THRESHOLDS)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    rows = []
    for dataset in args.datasets:
        for subset in ("classical", "deep"):
            for fold in args.folds:
                got = cell_inputs(dataset, fold, subset)
                if got is None:
                    continue
                vy, vmatrix, vconc, ty, tmatrix, tconc = got
                aucs = [auc_metric(vy, vmatrix[:, j]) for j in range(vmatrix.shape[1])]
                best = tmatrix[:, int(np.argmax(aucs))]
                auc_best = auc_metric(ty, best)
                for n_min in args.thresholds:
                    full = StaticConceptWeights(n_min=n_min).fit(
                        vmatrix, vy, valid_concepts=vconc).predict(
                        tmatrix, test_concepts=tconc)
                    inter = GlobalStackWithConceptIntercept(n_min=n_min).fit(
                        vmatrix, vy, valid_concepts=vconc).predict(
                        tmatrix, test_concepts=tconc)
                    rows.append({
                        "dataset": dataset, "subset": subset, "fold": fold,
                        "n_min": n_min,
                        "auc_full": auc_metric(ty, full),
                        "auc_intercept": auc_metric(ty, inter),
                        "best_single_auc": auc_best,
                    })
                print(f"[{dataset:22s} {subset:9s} f{fold}] готово")
    if not rows:
        _sys.exit("ФАТАЛЬНО: ни одной посчитанной ячейки")
    for r in rows:
        r["lift_full_vs_best"] = r["auc_full"] - r["best_single_auc"]
        r["lift_intercept_vs_best"] = r["auc_intercept"] - r["best_single_auc"]
        r["gap_full_minus_intercept"] = r["auc_full"] - r["auc_intercept"]
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
