"""Входные матрицы глубоких моделей на одном уровне подробности.

Расчёты ансамблей в этом проекте исторически шли на двух разных уровнях. Простое
усреднение и обучаемая надстройка читали предсказания на уровне заданий, а
взвешивание по компонентам знания, трёхступенчатая схема, внимание и разреженная
смесь — на уровне пар «задание, компонент». Уровни различаются не только числом
строк: у задания, отнесённого к нескольким компонентам, ответ повторяется в
каждой его строке, и модель, дойдя до второй такой строки, уже видела этот ответ
в истории. Площадь под кривой на таком развороте завышена — на EdNet-KT1-5k она
поднимается с 0.657 до 0.941, а число строк растёт в 2.3 раза.

Модуль даёт один загрузчик для обоих уровней, чтобы расчёты можно было привести
к общему и сравнивать между собой.

Уровень заданий доступен там, где у набора есть идентификаторы заданий. Где их
нет (ASSISTments-2015), уровень компонентов и есть уровень заданий: разворота не
происходит, и загрузчик возвращает те же самые массивы.
"""
from __future__ import annotations

import numpy as np

from . import paths

PRED = paths.ARTIFACTS_DIR / "predictions"

# Поля предсказаний для двух уровней подробности.
FIELDS = {
    "question": {
        "valid_y": "valid_y_true_q_pykt", "valid_p": "valid_y_prob_q_pykt",
        "valid_c": "valid_cidxs_q_pykt",
        "test_y": "y_true", "test_p": "y_prob", "test_c": "test_cidxs",
        "groups": "groups",
    },
    "concept": {
        "valid_y": "valid_y_true", "valid_p": "valid_y_prob", "valid_c": None,
        "test_y": "concept_y_true", "test_p": "concept_y_prob", "test_c": None,
        "groups": "concept_groups",
    },
}


class DeepInputs:
    """Валидационная и тестовая матрицы, коды компонентов и учащиеся."""

    def __init__(self, valid_y, valid_matrix, valid_concepts,
                 test_y, test_matrix, test_concepts, groups, models, granularity):
        self.valid_y = valid_y
        self.valid_matrix = valid_matrix
        self.valid_concepts = valid_concepts
        self.test_y = test_y
        self.test_matrix = test_matrix
        self.test_concepts = test_concepts
        self.groups = groups
        self.models = models
        self.granularity = granularity

    def __iter__(self):
        """Совместимость с прежним распаковыванием кортежа."""
        return iter((self.valid_y, self.valid_matrix, self.test_y, self.test_matrix,
                     self.groups, self.models))


def available(dataset: str, fold: int, model: str, granularity: str) -> bool:
    f = FIELDS[granularity]
    p = PRED / dataset / f"{model}_fold{fold}.npz"
    if not p.exists():
        return False
    files = set(np.load(p).files)
    need = {f["valid_y"], f["valid_p"], f["test_y"], f["test_p"]}
    if f["valid_c"]:
        need |= {f["valid_c"], f["test_c"]}
    return need <= files


def load(dataset: str, fold: int, models: list[str],
         granularity: str = "question") -> DeepInputs | None:
    """Матрицы всех моделей набора на одном уровне подробности.

    Возвращает ``None``, если уровень недоступен хотя бы одной модели или если
    модели расходятся в ответах: молча считать ансамбль на разных строках нельзя.
    """
    if granularity not in FIELDS:
        raise ValueError(f"неизвестный уровень подробности: {granularity!r}")
    f = FIELDS[granularity]
    v_ref = t_ref = vconc = tconc = groups = None
    v_cols, t_cols, kept = [], [], []
    for model in models:
        p = PRED / dataset / f"{model}_fold{fold}.npz"
        if not p.exists():
            continue
        d = np.load(p)
        need = {f["valid_y"], f["valid_p"], f["test_y"], f["test_p"]}
        if f["valid_c"]:
            need |= {f["valid_c"], f["test_c"]}
        if not need <= set(d.files):
            continue
        vy = np.asarray(d[f["valid_y"]]).astype(int)
        ty = np.asarray(d[f["test_y"]]).astype(int)
        if v_ref is None:
            v_ref, t_ref = vy, ty
            if f["valid_c"]:
                vconc = np.asarray(d[f["valid_c"]]).astype(np.int64)
                tconc = np.asarray(d[f["test_c"]]).astype(np.int64)
            g = d[f["groups"]] if f["groups"] in d.files else None
            groups = g if g is not None and g.size else None
        elif not np.array_equal(vy, v_ref) or not np.array_equal(ty, t_ref):
            return None
        v_cols.append(np.asarray(d[f["valid_p"]], dtype=np.float64))
        t_cols.append(np.asarray(d[f["test_p"]], dtype=np.float64))
        kept.append(model)
    if len(kept) < 2 or v_ref is None:
        return None
    if f["valid_c"] and (vconc.size != v_ref.size or tconc.size != t_ref.size):
        return None
    if groups is not None and groups.size != t_ref.size:
        groups = None
    return DeepInputs(v_ref, np.column_stack(v_cols), vconc,
                      t_ref, np.column_stack(t_cols), tconc, groups, kept, granularity)
