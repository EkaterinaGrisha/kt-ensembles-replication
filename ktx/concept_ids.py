"""Восстановление принадлежности строк предсказаний заданиям и компонентам знания.

Библиотека предобработки пишет тестовую часть последовательностями фиксированной
длины, а модель выдаёт плоский массив предсказаний; чтобы сказать, к какому
заданию, какому компоненту знания и какому учащемуся относится строка, нужно
повторить тот же обход последовательностей, каким шла оценка. Здесь он и лежит:
обход детерминирован, поэтому смещение строки в длинной таблице совпадает с её
индексом в массиве предсказаний.

Модуль выделен отдельно, потому что этим обходом пользуются независимые расчёты,
и связывать их через модуль, где лежит что-то ещё, незачем.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import paths



def concept_ids_for_valid(dataset: str, valid_fold: int) -> np.ndarray:
    """Return concept-id per row of the concept-level ``valid_y_true`` array
    stored in any deep-model npz.

    pyKT's evaluate_with_preds walks train_valid_sequences.csv filtered to the
    valid fold in CSV order; for each row it selects positions where
    selectmasks==1, drops the first (pyKT prediction shift), and appends the
    remainder to the flat validation predictions. We reproduce that walk and
    read the concept-id from the ``concepts`` column at each predicted
    position. Deterministic; matches the ordering of ``valid_y_true``.
    """
    p = paths.PYKT_ROOT / "data" / dataset / "train_valid_sequences.csv"
    if not p.exists():
        raise FileNotFoundError(str(p))
    df = pd.read_csv(p)
    df = df[df["fold"] == valid_fold]
    out: list[int] = []
    for _, r in df.iterrows():
        sm = [int(x) for x in str(r["selectmasks"]).split(",")]
        cs = [int(x) for x in str(r["concepts"]).split(",")]
        selected = [i for i, s in enumerate(sm) if s == 1]
        predicted = selected[1:]  # pyKT shift
        out.extend(cs[p] for p in predicted)
    return np.asarray(out, dtype=np.int64)


def concept_ids_for_test(dataset: str) -> np.ndarray:
    """Concept-id per row of the concept-level ``concept_y_true`` array.
    Uses extract_test_timeline() which already resolves the same CSV walk
    for the test side.
    """
    tl = extract_test_timeline(dataset)
    return tl.df.sort_values("pred_offset")["concept"].to_numpy(dtype=np.int64)


# --- Timeline extraction ---------------------------------------------------- #

@dataclass
class TestTimeline:
    """Long-form view of a dataset's test set aligned with concept-level preds.

    Columns of `df`: uid, order (per-student chronological index, monotonic
    within uid across CSV windows), question, concept, response, is_repeat,
    pred_offset (0-based global index into concept_y_prob array of any
    deep-model npz for that dataset — walk order of pyKT evaluate_with_preds).
    """
    dataset: str
    df: pd.DataFrame
    n_concept_preds: int


def extract_test_timeline(dataset: str) -> TestTimeline:
    """Reconstruct the long-form (uid, order, question, concept, response,
    is_repeat, pred_offset) representation of the test set. Deterministic:
    matches pyKT's evaluation walk, so `pred_offset` indexes into
    concept_y_prob of any deep-model npz for the same fold, and `order`
    is monotonic within each student across CSV windows (long students are
    split into multiple 200-length rows in test_sequences.csv — the previous
    per-window index was reset to 0 on every window, which broke chronology
    for anything sorting by (uid, order); see notes/14_rq2_audit.md P0-1).

    `is_repeat` follows pyKT convention: 0 = start of a new question
    interaction, 1 = continuation (another concept row of the same multi-KC
    interaction). Used by `question_level_events` to collapse concept-level
    rows to question-level events matching pyKT's late_mean fusion.

    For concept-only datasets (no `questions` column, e.g. assist2015), we
    set question=concept and is_repeat=0 for every row.
    """
    p = paths.PYKT_ROOT / "data" / dataset / "test_sequences.csv"
    if not p.exists():
        raise FileNotFoundError(str(p))
    df = pd.read_csv(p)
    has_q = "questions" in df.columns
    has_ir = "is_repeat" in df.columns
    rows: list[dict] = []
    offset = 0
    per_student_order: dict[int, int] = defaultdict(int)
    for _, r in df.iterrows():
        sm = [int(x) for x in str(r["selectmasks"]).split(",")]
        cs = [int(x) for x in str(r["concepts"]).split(",")]
        rs = [int(x) for x in str(r["responses"]).split(",")]
        qs = [int(x) for x in str(r["questions"]).split(",")] if has_q else cs
        ir = ([int(x) for x in str(r["is_repeat"]).split(",")] if has_ir
              else [0] * len(sm))
        selected = [i for i, s in enumerate(sm) if s == 1]
        predicted = selected[1:]
        uid = int(r["uid"])
        for local_i, pos in enumerate(predicted):
            rows.append({
                "uid": uid,
                "order": per_student_order[uid],
                "question": int(qs[pos]),
                "concept": int(cs[pos]),
                "response": int(rs[pos]),
                "is_repeat": int(ir[pos]),
                "pred_offset": offset + local_i,
            })
            per_student_order[uid] += 1
        offset += len(predicted)
    long_df = pd.DataFrame(rows)
    return TestTimeline(dataset=dataset, df=long_df, n_concept_preds=offset)
