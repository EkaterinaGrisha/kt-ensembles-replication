"""Насколько строки простых и глубоких моделей поддаются выравниванию.

Ансамбли в статье С5 строятся внутри семейства моделей, и раздел ограничений
обязан сказать, чего стоило бы объединить семейства между собой. Сказать это
внятно можно только числом: выравнивание возможно не на всех наборах, а там, где
возможно, оставляет от половины строк до всех.

Сценарий вызывает `ktx.alignment.align_families` с построчной сверкой ответов и
записывает, сколько строк у каждого семейства и сколько из них общие. Наборы, для
которых выравнивание невозможно, попадают в таблицу с причиной, а не молча
пропускаются.

Выход: `artifacts/ensembles/family_alignment.csv`.

Запуск:
  KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      OPENBLAS_NUM_THREADS=1 python -m scripts.family_alignment
"""
from __future__ import annotations

import sys

import pandas as pd

from ktx import paths
from ktx.alignment import align_families

DATASETS = ["algebra2005", "assist2009", "assist2012", "assist2015",
            "assist2017", "bridge2algebra2006", "ednet"]
BOOT = paths.ARTIFACTS_DIR / "ensembles" / "cluster_bootstrap_simple_ensembles.csv"
OUT = paths.ARTIFACTS_DIR / "ensembles" / "family_alignment.csv"


def main() -> int:
    size = pd.read_csv(BOOT).groupby(["dataset", "subset"]).n_rows.max().unstack()
    rows = []
    for dataset in DATASETS:
        entry = {"dataset": dataset,
                 "n_rows_classical": int(size.loc[dataset, "classical"]),
                 "n_rows_deep": int(size.loc[dataset, "deep"])}
        try:
            aligned = align_families(dataset, verify_responses=True)
            entry["n_shared"] = int(aligned.n_shared)
            entry["share_of_deep"] = aligned.n_shared / entry["n_rows_deep"]
            entry["status"] = "ok"
        except Exception as e:
            entry["n_shared"] = -1
            entry["share_of_deep"] = float("nan")
            entry["status"] = type(e).__name__
        rows.append(entry)
    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(df.to_string(index=False))
    ok = df[df.status == "ok"]
    print(f"\n{OUT}: выровнено наборов {len(ok)} из {len(df)}; "
          f"доля общих строк от {ok.share_of_deep.min():.0%} до {ok.share_of_deep.max():.0%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
