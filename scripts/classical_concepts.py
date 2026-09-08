"""Компонент знания для каждой строки предсказаний простых моделей.

Взвешивание по компонентам знания считалось только для глубоких моделей, и
причина была техническая: у простых моделей строка — это задание, а не пара
«задание, компонент», и какой компонент приписать строке, в файлах предсказаний
не записано. Здесь эта разметка восстанавливается.

Как. Тестовая и валидационная части собираются тем же загрузчиком, каким
пользуются сами простые модели, и сворачиваются к уровню заданий той же
группировкой по паре «учащийся, номер задания». Строке приписывается первый по
номеру компонент знания задания; у заданий с одним компонентом выбор пустой, у
многокомпонентных это соглашение, и оно названо в статье.

Порядок строк не угадывается. Сценарий собирает оба возможных порядка — тот, в
каком строки лежат в файле, и отсортированный по учащемуся и позиции — и
сверяет вектор ответов с тем, что лежит в предсказаниях каждой из четырёх
моделей. Если совпадения нет ни при одном порядке, сценарий падает: молча
приписать компоненты не тем строкам — ровно та ошибка, из-за которой в этом
проекте уже пришлось пересчитывать матрицу.

Выход: `artifacts/ensembles/classical_concepts/<набор>.npz` с полями `test_cid`,
`test_n_concepts` и `valid_cid_<разбиение>` для каждого из пяти разбиений.

Запуск:
  python -m scripts.classical_concepts
  python -m scripts.classical_concepts --datasets assist2009
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ktx import paths
from ktx.classical.data import ClassicalData, load_classical_data

DATASETS = ["algebra2005", "assist2009", "assist2012", "assist2015",
            "assist2017", "bridge2algebra2006", "ednet"]
FOLDS = [0, 1, 2, 3, 4]
MODELS = ["bkt", "pfa", "pfa_recency", "elorasch"]

PRED = paths.ARTIFACTS_DIR / "predictions"
VALID_CACHE = paths.ARTIFACTS_DIR / "predictions_valid_cache"
OUT_DIR = paths.ARTIFACTS_DIR / "ensembles" / "classical_concepts"


def question_level(concept_df: pd.DataFrame, sort: bool) -> pd.DataFrame:
    df = concept_df.sort_values(["uid", "seq_pos"]) if sort else concept_df
    return ClassicalData.to_question_level(df)


def reference_responses(dataset: str, fold: int, split: str) -> np.ndarray | None:
    """Ответы, записанные рядом с предсказаниями, — эталон для сверки порядка."""
    for model in MODELS:
        if split == "test":
            p = PRED / dataset / f"{model}_fold{fold}.npz"
            key = "y_true"
        else:
            p = VALID_CACHE / f"{dataset}__{model}__fold{fold}.npz"
            key = "valid_y_true"
        if not p.exists():
            continue
        d = np.load(p)
        if key in d.files:
            return np.asarray(d[key]).astype(int)
    return None


def resolve_order(frame: pd.DataFrame, reference: np.ndarray, what: str) -> pd.DataFrame:
    """Выбрать порядок строк, при котором ответы совпадают с эталоном."""
    for sort in (False, True):
        q = question_level(frame, sort=sort)
        resp = q["response"].to_numpy().astype(int)
        if resp.size == reference.size and np.array_equal(resp, reference):
            return q
    sizes = {s: question_level(frame, sort=s).shape[0] for s in (False, True)}
    sys.exit(f"ФАТАЛЬНО: {what}: ответы не совпали ни при одном порядке строк. "
             f"В предсказаниях {reference.size} строк, в собранной таблице {sizes}.")


def first_concept(q: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    cids = np.array([c[0] for c in q["concepts"]], dtype=np.int64)
    n = np.array([len(c) for c in q["concepts"]], dtype=np.int64)
    return cids, n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--folds", nargs="+", type=int, default=FOLDS)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for dataset in args.datasets:
        t0 = time.time()
        data = load_classical_data(dataset)
        payload: dict[str, np.ndarray] = {}

        ref_test = reference_responses(dataset, args.folds[0], "test")
        if ref_test is None:
            print(f"[пропуск] {dataset}: нет предсказаний простых моделей")
            continue
        q_test = resolve_order(data.test, ref_test, f"{dataset}, тестовая часть")
        cid, ncon = first_concept(q_test)
        payload["test_cid"] = cid
        payload["test_n_concepts"] = ncon

        for fold in args.folds:
            _, valid, _ = data.split(valid_fold=fold)
            ref_valid = reference_responses(dataset, fold, "valid")
            if ref_valid is None:
                print(f"  [нет валидации] {dataset} разбиение {fold}")
                continue
            q_valid = resolve_order(valid, ref_valid,
                                    f"{dataset}, валидация, разбиение {fold}")
            vcid, _ = first_concept(q_valid)
            payload[f"valid_cid_{fold}"] = vcid

        out = OUT_DIR / f"{dataset}.npz"
        np.savez_compressed(out, **payload)
        multi = float((ncon > 1).mean())
        print(f"{dataset:22s} тест {cid.size:7d} строк, компонентов "
              f"{np.unique(cid).size:4d}, заданий с несколькими компонентами "
              f"{multi:5.1%}  [{time.time() - t0:.0f} с]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
