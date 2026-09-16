"""Разброс внимания и разреженной смеси по случайному начальному состоянию.

Обе схемы обучаются градиентным спуском, и в работе уже отмечено, что при смене
устройства их результат уезжает на величину, сопоставимую со всем измеряемым
эффектом. Одного начального состояния тогда мало: непонятно, что именно
измеряется — схема или конкретный запуск.

Сценарий обучает обе схемы с пятью начальными состояниями на одних и тех же
данных и записывает разность с взвешиванием по компонентам знания для каждого.
Разброс по начальным состояниям — прямая мера того, насколько выводу об этих
схемах можно доверять.

Выход: `artifacts/ensembles/attention_seed_spread.csv`.

Запуск:
  KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      OPENBLAS_NUM_THREADS=1 python -m scripts.attention_seed_spread
"""
from __future__ import annotations

import os as _os
import sys as _sys

for _k, _v in {"KMP_DUPLICATE_LIB_OK": "TRUE", "OMP_NUM_THREADS": "1",
               "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}.items():
    _os.environ.setdefault(_k, _v)

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from ktx import deep_inputs, paths
from ktx.concept_ids import concept_ids_for_test, concept_ids_for_valid
from ktx.ensemble import (
    AttentionGating,
    MixtureOfExperts,
    StaticConceptWeights,
)
from ktx.stats import auc_metric

DATASETS = ["algebra2005", "assist2009", "assist2012", "assist2015",
            "assist2017", "bridge2algebra2006", "ednet"]
FOLDS = [0, 1, 2, 3, 4]
DEEP = ["dkt", "sakt", "akt", "simplekt"]
SEEDS = [0, 1, 2, 3, 4]
OUT = paths.ARTIFACTS_DIR / "ensembles" / "attention_seed_spread.csv"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--folds", nargs="+", type=int, default=FOLDS)
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--granularity", choices=["concept", "question"], default="concept",
                    help="уровень подробности строки: пара «задание, компонент» или задание")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    if args.out is None:
        suffix = "" if args.granularity == "concept" else "_question"
        args.out = OUT.with_name(f"{OUT.stem}{suffix}{OUT.suffix}")

    rows = []
    for dataset in args.datasets:
        for fold in args.folds:
            try:
                got = deep_inputs.load_cell(dataset, fold, DEEP, args.granularity)
            except FileNotFoundError:
                continue
            if got is None:
                continue
            vconc, tconc = got.valid_concepts, got.test_concepts
            static = StaticConceptWeights(n_min=30).fit(
                got.valid_matrix, got.valid_y, valid_concepts=vconc).predict(
                got.test_matrix, test_concepts=tconc)
            auc_static = auc_metric(got.test_y, static)
            t0 = time.time()
            for seed in args.seeds:
                for scheme, model in (
                        ("attention", AttentionGating(n_epochs=15, seed=seed,
                                                      device="cpu")),
                        ("moe", MixtureOfExperts(k_top=2, n_epochs=15, seed=seed,
                                                 device="cpu"))):
                    pred = model.fit(got.valid_matrix, got.valid_y,
                                     valid_concepts=vconc).predict(
                        got.test_matrix, test_concepts=tconc)
                    rows.append({
                        "dataset": dataset, "fold": fold,
                        "granularity": args.granularity, "scheme": scheme,
                        "seed": seed,
                        "auc": auc_metric(got.test_y, pred),
                        "static_auc": auc_static,
                        "lift_vs_static": auc_metric(got.test_y, pred) - auc_static,
                    })
            print(f"[{dataset:22s} f{fold}] {len(args.seeds)} начальных состояний "
                  f"за {time.time() - t0:.0f} с")
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
