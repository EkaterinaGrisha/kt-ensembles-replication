"""Два уровня подробности у глубоких моделей: сколько строк и какая площадь.

Расчёты по глубоким моделям в этом проекте идут на двух уровнях. Усреднение без
настройки и обучаемая надстройка читают предсказания на уровне заданий.
Взвешивание по компонентам знания, трёхступенчатая схема, внимание и разреженная
смесь читают предсказания на уровне пар «задание, компонент»: у задания,
отнесённого к нескольким компонентам, ответ повторяется в каждой его строке.

Разница не только в числе строк. На развороте модель, дойдя до второй строки
задания, уже видела этот ответ в истории, и площадь под кривой на таком уровне
завышена. Сценарий измеряет обе величины, чтобы статья могла назвать их прямо, а
читатель — понимать, какие таблицы между собой сравнимы, а какие нет.

Выход: `artifacts/ensembles/deep_granularity.csv`.

Запуск:
  python -m scripts.deep_granularity
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from ktx import deep_inputs, paths
from ktx.stats import auc_metric

DATASETS = ["algebra2005", "assist2009", "assist2012", "assist2015",
            "assist2017", "bridge2algebra2006", "ednet"]
DEEP = ["dkt", "sakt", "akt", "simplekt"]
FOLDS = [0, 1, 2, 3, 4]
OUT = paths.ARTIFACTS_DIR / "ensembles" / "deep_granularity.csv"


def best_single_auc(inputs) -> float:
    """Лучшая одиночная модель выбирается по валидации, как и везде в работе."""
    v, t = inputs.valid_matrix, inputs.test_matrix
    aucs = [auc_metric(inputs.valid_y, v[:, j]) for j in range(v.shape[1])]
    j = sorted(range(len(inputs.models)), key=lambda i: (-aucs[i], inputs.models[i]))[0]
    return auc_metric(inputs.test_y, t[:, j])


def main() -> int:
    rows = []
    for dataset in DATASETS:
        entry = {"dataset": dataset}
        for gran in ("question", "concept"):
            aucs, n = [], None
            for fold in FOLDS:
                got = deep_inputs.load(dataset, fold, DEEP, gran)
                if got is None:
                    continue
                n = int(got.test_y.size)
                aucs.append(best_single_auc(got))
            entry[f"n_rows_{gran}"] = n if n is not None else -1
            entry[f"auc_best_{gran}"] = float(np.mean(aucs)) if aucs else float("nan")
        q, c = entry["n_rows_question"], entry["n_rows_concept"]
        entry["expansion"] = (c / q) if q > 0 else float("nan")
        rows.append(entry)
    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(df.round(4).to_string(index=False))
    ok = df[df.n_rows_question > 0]
    print(f"\n{OUT}: разворот от {ok.expansion.min():.2f} до {ok.expansion.max():.2f}; "
          f"площадь под кривой поднимается на "
          f"{(ok.auc_best_concept - ok.auc_best_question).max():.4f} в худшем случае")
    return 0


if __name__ == "__main__":
    sys.exit(main())
