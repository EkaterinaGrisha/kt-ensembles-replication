"""Площадь под кривой и ошибка калибровки каждой компоненты ансамбля.

Статья С5 сравнивает ансамбли с лучшей одиночной моделью, и читателю нужно
видеть, из чего ансамбль собран: насколько компоненты близки друг к другу и
насколько лучшая из них оторвалась от остальных. Без этого раздел про
усреднение без настройки остаётся без объяснения — а объяснение там ровно в
разрыве между лучшей компонентой и второй.

Числа считаются из тех же файлов предсказаний, которые читают сценарии
ансамблей (`artifacts/predictions/<набор>/<модель>_fold<фолд>.npz`, поля
`y_true` и `y_prob`), теми же функциями. Брать их из таблиц калибровки было бы
короче, но там другая гранулярность на части наборов, и совпадение с
ансамблевыми таблицами не гарантировано.

Второй выход — разброс предсказаний: стандартное отклонение и межквартильный
размах у лучшей одиночной модели и у их среднего арифметического. Он нужен
разделу о калибровке: усреднение стягивает вероятности к середине шкалы, и это
утверждение должно опираться на число, а не на вид диаграммы.

Выход: `artifacts/ensembles/component_models.csv`, по строке на
(набор, модель, разбиение), и `artifacts/ensembles/prediction_spread.csv`, по
строке на (набор, семейство, разбиение).

Запуск:
  python -m scripts.component_models
"""
from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd

from ktx import paths
from ktx.metrics import ece_equal_mass
from ktx.stats import auc_metric, brier_metric

DATASETS = ["algebra2005", "assist2009", "assist2012", "assist2015",
            "assist2017", "bridge2algebra2006", "ednet"]
FOLDS = [0, 1, 2, 3, 4]
SUBSETS = {
    "classical": ["bkt", "pfa", "pfa_recency", "elorasch"],
    "deep": ["dkt", "sakt", "akt", "simplekt"],
}
ECE_BINS = 15  # как в run_simple_ensembles.py

OUT = paths.ARTIFACTS_DIR / "ensembles" / "component_models.csv"
SPREAD_OUT = paths.ARTIFACTS_DIR / "ensembles" / "prediction_spread.csv"
BOOT = paths.ARTIFACTS_DIR / "ensembles" / "cluster_bootstrap_simple_ensembles.csv"


def spread(prob: np.ndarray) -> dict:
    q75, q25 = np.percentile(prob, [75, 25])
    return {"std": float(prob.std()), "iqr": float(q75 - q25)}


def prediction_spread() -> pd.DataFrame:
    """Разброс предсказаний у лучшей одиночной модели и у усреднения."""
    boot = pd.read_csv(BOOT)
    rows = []
    for dataset in DATASETS:
        for subset, models in SUBSETS.items():
            for fold in FOLDS:
                cell = boot[(boot.dataset == dataset) & (boot.subset == subset)
                            & (boot.aggregator == "arithmetic_mean") & (boot.fold == fold)]
                if cell.empty:
                    continue
                best_model = cell.best_single_model.iloc[0]
                cols, best = [], None
                for model in models:
                    p = (paths.ARTIFACTS_DIR / "predictions" / dataset
                         / f"{model}_fold{fold}.npz")
                    if not p.exists():
                        break
                    prob = np.asarray(np.load(p)["y_prob"], dtype=np.float64)
                    cols.append(prob)
                    if model == best_model:
                        best = prob
                if best is None or len(cols) != len(models):
                    continue
                ens = np.mean(np.column_stack(cols), axis=1)
                rows.append({"dataset": dataset, "subset": subset, "fold": fold,
                             "best_single_model": best_model,
                             **{f"best_{k}": v for k, v in spread(best).items()},
                             **{f"ensemble_{k}": v for k, v in spread(ens).items()}})
    return pd.DataFrame(rows)


def main() -> int:
    rows = []
    started = time.time()
    for dataset in DATASETS:
        for subset, models in SUBSETS.items():
            for model in models:
                for fold in FOLDS:
                    p = (paths.ARTIFACTS_DIR / "predictions" / dataset
                         / f"{model}_fold{fold}.npz")
                    if not p.exists():
                        print(f"[нет] {dataset}/{model}/fold{fold}")
                        continue
                    d = np.load(p)
                    if not {"y_true", "y_prob"} <= set(d.files):
                        print(f"[нет полей] {p.name}")
                        continue
                    y = np.asarray(d["y_true"]).astype(int)
                    prob = np.asarray(d["y_prob"], dtype=np.float64)
                    rows.append({
                        "dataset": dataset, "subset": subset, "model": model,
                        "fold": fold, "n_rows": int(y.size),
                        "auc": auc_metric(y, prob),
                        "ece": ece_equal_mass(y, prob, n_bins=ECE_BINS),
                        "brier": brier_metric(y, prob),
                    })
    if not rows:
        sys.exit("ФАТАЛЬНО: не найдено ни одного файла предсказаний")
    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"{OUT}: {len(df)} строк за {time.time() - started:.0f} с")

    wide = df.groupby(["dataset", "subset", "model"]).auc.mean().unstack()
    print("\nAUC, среднее по разбиениям:")
    print(wide.round(4).to_string())

    sp = prediction_spread()
    sp.to_csv(SPREAD_OUT, index=False)
    print(f"\n{SPREAD_OUT}: {len(sp)} строк")
    print("разброс предсказаний, среднее по разбиениям:")
    print(sp.groupby(["dataset", "subset"])[["best_std", "ensemble_std"]].mean()
          .round(4).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
