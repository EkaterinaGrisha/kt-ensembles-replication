"""Tidy tabular loader for classical KT baselines (PFA, IRT, BKT).

Sequence models (DKT, SAKT, …) consume pyKT's windowed sequence files directly.
Classical models need a per-interaction tabular view instead. This module parses
pyKT's clean per-student long files — `train_valid.csv` (folds 0–4) and `test.csv`
(fold −1) — into tidy DataFrames, and reconstructs question groupings from the
`is_repeat` flag so the same interaction can be scored at concept level or
question level, comparably to pyKT's two evaluation granularities (D005).

pyKT row format (one student per CSV row, comma-separated aligned sequences):
    fold, uid, questions, concepts, responses, is_repeat[, ...]
`is_repeat == 0` marks the first concept of a new question; `is_repeat == 1`
marks a subsequent concept of the *same* question (multi-KC expansion).
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .. import config as kt_config
from .. import paths


def _explode_student_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Explode the per-student comma-separated sequences into one row per interaction."""
    records = []
    has_q = "questions" in df.columns
    for row in df.itertuples(index=False):
        uid = int(row.uid)
        fold = int(row.fold)
        concepts = str(row.concepts).split(",")
        responses = str(row.responses).split(",")
        is_repeat = str(row.is_repeat).split(",") if "is_repeat" in df.columns else ["0"] * len(concepts)
        questions = str(row.questions).split(",") if has_q and str(row.questions) != "nan" else None
        n = len(responses)
        # running question index within the student (increments when is_repeat == 0)
        qpos = -1
        for i in range(n):
            rep = int(is_repeat[i]) if is_repeat[i] not in ("", "nan") else 0
            if rep == 0:
                qpos += 1
            try:
                resp = int(responses[i])
            except ValueError:
                continue
            if resp < 0:  # padding
                continue
            records.append({
                "uid": uid,
                "fold": fold,
                "seq_pos": i,
                "qpos": qpos,
                "qid": int(questions[i]) if questions is not None else -1,
                "cid": int(concepts[i]),
                "response": resp,
                "is_repeat": rep,
            })
    return pd.DataFrame.from_records(records)


@dataclass
class ClassicalData:
    """Tidy concept-level interactions for a dataset, split by fold."""

    dataset_name: str
    train_valid: pd.DataFrame  # rows with fold in 0..4
    test: pd.DataFrame         # rows with fold == -1
    num_concepts: int
    num_questions: int

    def split(self, valid_fold: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Return (train, valid, test) concept-level frames for a given fold.

        Matches the DL convention: `valid_fold` is held out for validation, the other
        folds are training, and the canonical test set is always `self.test`.
        """
        tv = self.train_valid
        train = tv[tv["fold"] != valid_fold].copy()
        valid = tv[tv["fold"] == valid_fold].copy()
        return train, valid, self.test.copy()

    @staticmethod
    def to_question_level(concept_df: pd.DataFrame) -> pd.DataFrame:
        """Collapse concept-level rows to one row per (uid, qpos) question.

        Concepts of a multi-KC question are aggregated into a sorted tuple; the
        response is shared across the group (taken from the first row).
        """
        grp = concept_df.groupby(["uid", "qpos"], sort=False)
        out = grp.agg(
            fold=("fold", "first"),
            qid=("qid", "first"),
            response=("response", "first"),
            concepts=("cid", lambda s: tuple(sorted(set(int(x) for x in s)))),
            seq_pos=("seq_pos", "first"),
        ).reset_index()
        return out


def load_classical_data(dataset_name: str) -> ClassicalData:
    """Load and explode a dataset's train_valid.csv / test.csv into tidy frames."""
    data_config = kt_config.load_data_config()[dataset_name]
    dpath = paths.PYKT_DATA / dataset_name.split("/")[-1]
    tv_path = dpath / data_config.get("train_valid_original_file", "train_valid.csv")
    test_path = dpath / data_config.get("test_original_file", "test.csv")

    tv_raw = pd.read_csv(tv_path, dtype=str)
    test_raw = pd.read_csv(test_path, dtype=str)

    tv = _explode_student_rows(tv_raw)
    test = _explode_student_rows(test_raw)

    num_c = int(data_config.get("num_c", 0)) or int(
        max(tv["cid"].max(), test["cid"].max()) + 1)
    num_q = int(data_config.get("num_q", 0)) or int(
        max(tv["qid"].max(), test["qid"].max()) + 1)

    return ClassicalData(
        dataset_name=dataset_name,
        train_valid=tv,
        test=test,
        num_concepts=num_c,
        num_questions=num_q,
    )
