"""Собрать все таблицы статьи С5 из артефактов.

Таблицы набираются не руками. Раздел 5.1 черновика показал, чем кончается ручной
набор: после пересчёта ансамблей 1 сентября колонка глубоких моделей разошлась с
артефактом во всех семи ячейках, и на ASSISTments-2017 знак оказался
противоположным. Сценарий печатает таблицы в том виде, в каком они идут в
рукопись, чтобы вставка была механической, и заодно считает величины, на которые
ссылается проза.

Все разности — доли площади под кривой или доли ошибки калибровки, как в
остальных статьях серии; проценты не используются. Знак везде один и тот же:
величина минус величина точки отсчёта, поэтому для площади под кривой
положительное значение означает выигрыш, а для ошибки калибровки — проигрыш.

Значимость считается по поправке Холма внутри семейства (колонки
`*_reject_holm` бутстрапов), а не по сырому p-значению: сравнений много, и без
поправки доля значимых ячеек завышена. Для внимания и разреженной смеси
бутстрапа нет, и значимость для них не приводится.

Режим `--into` подставляет свежие таблицы прямо в рукопись: находит «**Табл. N.**»
и заменяет идущие за ней строки таблицы. Подписи и проза при этом не трогаются —
их правит человек, а числа приезжают из артефактов. После пересчёта порядок
такой: `component_models` → `make_c5_tables --into` → `check_c5_numbers`.

Запуск:
  python -m scripts.make_c5_tables
  python -m scripts.make_c5_tables --into research/paper/c5_ensembles_ru.md
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

from ktx import paths

ENS = paths.ARTIFACTS_DIR / "ensembles"
DATA_CONFIG = paths.DATA_CONFIG_JSON

COMPONENTS = ENS / "component_models.csv"
SPREAD = ENS / "prediction_spread.csv"
GRANULARITY = ENS / "deep_granularity.csv"
NMIN = ENS / "gating_nmin_sensitivity.csv"
SEEDS = ENS / "attention_seed_spread.csv"
CLASSICAL_CONCEPTS = ENS / "classical_concepts"
SIMPLE_BOOT = ENS / "cluster_bootstrap_simple_ensembles.csv"
STACK_BOOT = ENS / "cluster_bootstrap_stacking_all_v4.csv"
GATING = {"deep": ENS / "gating_deep.csv", "classical": ENS / "gating_classical.csv"}
GATING_BOOT = {"deep": ENS / "cluster_bootstrap_gating.csv",
               "classical": ENS / "cluster_bootstrap_gating_classical.csv"}
ATTN_MOE_BOOT = ENS / "cluster_bootstrap_attention_moe.csv"
ATTN_MPS = ENS / "attention_gating_deep.csv"   # прежний прогон на ускорителе
MOE_MPS = ENS / "moe_deep.csv"
CCCE = ENS / "ccce_deep.csv"
CCCE_CF = ENS / "ccce_crossfit_deep.csv"   # первая ступень обучена вне блока
CCCE_BOOT = ENS / "cluster_bootstrap_ccce.csv"

ORDER = ["algebra2005", "assist2009", "assist2012", "assist2015",
         "assist2017", "bridge2algebra2006", "ednet"]
DISPLAY = {
    "algebra2005": "Algebra-2005", "assist2009": "ASSISTments-2009",
    "assist2012": "ASSISTments-2012", "assist2015": "ASSISTments-2015",
    "assist2017": "ASSISTments-2017", "bridge2algebra2006": "Bridge-to-Algebra-2006",
    "ednet": "EdNet-KT1-5k",
}
W = max(len(v) for v in DISPLAY.values())

MODELS = {"classical": ["bkt", "pfa", "pfa_recency", "elorasch"],
          "deep": ["dkt", "sakt", "akt", "simplekt"]}
MODEL_RU = {"bkt": "BKT", "pfa": "PFA", "pfa_recency": "PFA с забыванием",
            "elorasch": "Эло-Раш", "dkt": "DKT", "sakt": "SAKT", "akt": "AKT",
            "simplekt": "simpleKT"}
SUBSET_RU = {"classical": "простые", "deep": "глубокие"}
SUBSET_GEN = {"classical": "простых", "deep": "глубоких"}

AGGREGATORS = ["arithmetic_mean", "geometric_mean", "logit_mean", "median", "rank_mean"]
AGG_RU = {"arithmetic_mean": "среднее ар.", "geometric_mean": "среднее геом.",
          "logit_mean": "в логитах", "median": "медиана", "rank_mean": "по рангам"}
METAS = ["logistic_stacked", "ridge_stacked", "bma", "xgb_stacked", "mlp_stacked"]
META_RU = {"logistic_stacked": "логист.", "ridge_stacked": "с подбором",
           "bma": "байесовское", "xgb_stacked": "бустинг", "mlp_stacked": "сеть"}
GATES = ["static_concept_weights", "global_stack_concept_intercept", "linear_gating"]
GATE_RU = {"static_concept_weights": "полное", "global_stack_concept_intercept": "своб. член",
           "linear_gating": "распределитель"}
VARIANTS = ["full", "no_s1", "global_s1", "no_s3"]
VARIANT_RU = {"full": "полная", "no_s1": "без ст. 1",
              "global_s1": "общая ст. 1", "no_s3": "без ст. 3"}

ND = 4  # доли с четырьмя знаками: 0.0064, как в С1 и С2


def frac(x: float, nd: int = ND) -> str:
    """Знак ставится всегда: таблицы про направление сдвига, а не только про величину."""
    return f"{x:+.{nd}f}".replace("-", "−")


def plain(x: float, nd: int = ND) -> str:
    """Без знака — для самих значений метрики, а не для разностей."""
    return f"{x:.{nd}f}"


def load(path) -> pd.DataFrame:
    if not path.exists():
        sys.exit(f"ФАТАЛЬНО: нет артефакта {path}")
    return pd.read_csv(path)


def sig_counts(boot: pd.DataFrame, keys: list[str], column: str) -> pd.Series:
    """Сколько разбиений из скольких отвергают нулевую гипотезу после поправки Холма."""
    grouped = boot.groupby(keys)[column]
    return grouped.sum().astype(int).astype(str) + "/" + grouped.count().astype(str)


def head(cols: list[str], width: int, first: str = "набор") -> list[str]:
    cells = " | ".join(f"{c:>{width}}" for c in cols)
    rule = " | ".join(["-" * (width - 1) + ":"] * len(cols))
    return [f"| {first:<{W}} | {cells} |", f"| {'-' * W} | {rule} |"]


# ------------------------------------------------------------------ таблица 1
def table_datasets(boot: pd.DataFrame) -> str:
    """Учащиеся и строки — из артефакта, компоненты знания — из конфигурации pyKT."""
    if not DATA_CONFIG.exists():
        sys.exit(f"ФАТАЛЬНО: нет конфигурации {DATA_CONFIG}")
    cfg = json.loads(DATA_CONFIG.read_text(encoding="utf-8"))
    size = boot.groupby(["dataset", "subset"])[["n_rows", "n_students"]].max().unstack()
    out = ["Табл. 1. Наборы данных: объём тестовой выборки, число учащихся в ней, "
           "число компонентов знания и число заданий по разметке предобработки.", "",
           f"| {'набор':<{W}} | учащихся | компонентов | заданий | строк, простые | строк, глубокие |",
           f"| {'-' * W} | -------: | ----------: | ------: | -------------: | --------------: |"]
    for ds in ORDER:
        c = cfg.get(ds)
        if c is None or "num_c" not in c:
            sys.exit(f"ФАТАЛЬНО: в {DATA_CONFIG} нет num_c для {ds}")
        num_q = c.get("num_q", 0)
        out.append(f"| {DISPLAY[ds]:<{W}} "
                   f"| {size[('n_students', 'deep')][ds]:>8} "
                   f"| {c['num_c']:>11} "
                   f"| {(num_q if num_q else '—'):>7} "
                   f"| {size[('n_rows', 'classical')][ds]:>14} "
                   f"| {size[('n_rows', 'deep')][ds]:>15} |")
    return "\n".join(out)


# --------------------------------------------------------------- таблицы 2, 3
def table_components(comp: pd.DataFrame, subset: str, number: int) -> str:
    d = comp[comp.subset == subset]
    auc = d.groupby(["dataset", "model"]).auc.mean().unstack()
    models = MODELS[subset]
    width = max([len(MODEL_RU[m]) for m in models] + [len("отрыв")])
    out = [f"Табл. {number}. Площадь под кривой у {SUBSET_GEN[subset]} моделей, среднее "
           f"по пяти разбиениям. Последний столбец — отрыв лучшей модели набора от "
           f"второй по величине.", ""]
    out += head([MODEL_RU[m] for m in models] + ["отрыв"], width)
    for ds in ORDER:
        row = auc.loc[ds, models]
        top2 = row.sort_values(ascending=False).iloc[:2]
        cells = " | ".join(f"{plain(row[m]):>{width}}" for m in models)
        gap = f"{plain(top2.iloc[0] - top2.iloc[1]):>{width}}"
        out.append(f"| {DISPLAY[ds]:<{W}} | {cells} | {gap} |")
    return "\n".join(out)


# ------------------------------------------------------------------ таблица 4
def table_simple(boot: pd.DataFrame, number: int) -> str:
    delta = boot.groupby(["dataset", "subset", "aggregator"]).delta_auc.mean()
    sig = sig_counts(boot[boot.aggregator == "arithmetic_mean"],
                     ["dataset", "subset"], "auc_p_reject_holm")
    width = max(len(v) for v in AGG_RU.values())
    out = [f"Табл. {number}. Усреднение без настройки против лучшей одиночной модели: "
           f"разность площади под кривой, среднее по пяти разбиениям. Столбец «знач.» — "
           f"число разбиений из пяти, на которых значима разность для среднего "
           f"арифметического после поправки Холма.", ""]
    cells = " | ".join(f"{AGG_RU[a]:>{width}}" for a in AGGREGATORS)
    rule = " | ".join(["-" * (width - 1) + ":"] * len(AGGREGATORS))
    out += [f"| {'набор':<{W}} | семейство | {cells} | знач. |",
            f"| {'-' * W} | --------- | {rule} | :---: |"]
    for ds in ORDER:
        for subset in ("classical", "deep"):
            row = " | ".join(f"{frac(delta[(ds, subset, a)]):>{width}}" for a in AGGREGATORS)
            out.append(f"| {DISPLAY[ds]:<{W}} | {SUBSET_RU[subset]:<9} | {row} "
                       f"| {sig[(ds, subset)]:^5} |")
    return "\n".join(out)


# ------------------------------------------------------------------ таблица 5
def table_stacking(boot: pd.DataFrame, number: int) -> str:
    delta = boot.groupby(["dataset", "subset", "meta_learner"]).delta_auc.mean()
    sig = sig_counts(boot, ["dataset", "subset", "meta_learner"], "auc_p_reject_holm")
    width = max(len(v) for v in META_RU.values()) + 3
    out = [f"Табл. {number}. Обучаемая надстройка против лучшей одиночной модели: "
           f"разность площади под кривой, среднее по пяти разбиениям, и в скобках — "
           f"число разбиений из пяти со значимой разностью после поправки Холма.", ""]
    cells = " | ".join(f"{META_RU[m]:>{width}}" for m in METAS)
    rule = " | ".join(["-" * (width - 1) + ":"] * len(METAS))
    out += [f"| {'набор':<{W}} | семейство | {cells} |",
            f"| {'-' * W} | --------- | {rule} |"]
    for ds in ORDER:
        for subset in ("classical", "deep"):
            row = []
            for m in METAS:
                key = (ds, subset, m)
                if key in delta.index:
                    n = sig[key].split("/")[0]
                    row.append(f"{frac(delta[key]) + ' (' + n + ')':>{width}}")
                else:
                    row.append(f"{'—':>{width}}")
            out.append(f"| {DISPLAY[ds]:<{W}} | {SUBSET_RU[subset]:<9} | "
                       + " | ".join(row) + " |")
    out += ["", "Прочерк — ячейка, для которой надстройку обучить не на чем: у "
            "ASSISTments-2015 нет разметки заданий, поэтому у глубоких моделей нет "
            "валидационных предсказаний на уровне заданий."]
    return "\n".join(out)


# ------------------------------------------------------------------ таблица 6
def table_gating(points: dict, boot: dict, number: int) -> str:
    width = max(len(v) for v in GATE_RU.values()) + 3
    out = [f"Табл. {number}. Условное взвешивание и абляция свободного члена: разность "
           f"площади под кривой относительно лучшей одиночной модели, среднее по пяти "
           f"разбиениям, в скобках — число разбиений со значимой разностью после "
           f"поправки Холма. Столбец «своб. член» — вариант, в котором наклоны общие для "
           f"всех компонентов знания, а свободный член свой у каждого. У глубоких "
           f"моделей строка — пара «задание, компонент знания», у простых — задание; "
           f"числа двух семейств между собой не сравниваются (раздел 2.3).", ""]
    cells = " | ".join(f"{GATE_RU[g]:>{width}}" for g in GATES)
    rule = " | ".join(["-" * (width - 1) + ":"] * len(GATES))
    out += [f"| {'набор':<{W}} | семейство | {cells} |",
            f"| {'-' * W} | --------- | {rule} |"]
    for ds in ORDER:
        for subset in ("classical", "deep"):
            delta = points[subset].groupby(
                ["dataset", "meta_learner"]).lift_gated_vs_best_auc.mean()
            sig = sig_counts(boot[subset], ["dataset", "meta_learner"],
                             "auc_p_vs_best_reject_holm")
            row = []
            for g in GATES:
                if (ds, g) not in delta.index:
                    row.append(f"{'—':>{width}}")
                    continue
                n = sig.get((ds, g))
                label = frac(delta[(ds, g)]) + (f" ({n.split('/')[0]})" if n else "")
                row.append(f"{label:>{width}}")
            out.append(f"| {DISPLAY[ds]:<{W}} | {SUBSET_RU[subset]:<9} | "
                       + " | ".join(row) + " |")
    return "\n".join(out)


# ------------------------------------------------------------------ таблица 7
def table_gating_vs_stack(points: dict, boot: dict, number: int) -> str:
    """Главное практическое сравнение: обе схемы обучены на одной валидации и
    измерены на одних и тех же строках, поэтому числа сопоставимы напрямую."""
    width = max(len(v) for v in GATE_RU.values()) + 3
    out = [f"Табл. {number}. Условное взвешивание против обучаемой надстройки: разность "
           f"площади под кривой, среднее по пяти разбиениям, в скобках — число разбиений "
           f"со значимой разностью после поправки Холма. Обе схемы настраиваются на одной "
           f"и той же валидационной части и измеряются на одних и тех же строках.", ""]
    cells = " | ".join(f"{GATE_RU[g]:>{width}}" for g in GATES)
    rule = " | ".join(["-" * (width - 1) + ":"] * len(GATES))
    out += [f"| {'набор':<{W}} | семейство | {cells} |",
            f"| {'-' * W} | --------- | {rule} |"]
    for ds in ORDER:
        for subset in ("classical", "deep"):
            delta = points[subset].groupby(
                ["dataset", "meta_learner"]).lift_gated_vs_stack_auc.mean()
            sig = sig_counts(boot[subset], ["dataset", "meta_learner"],
                             "auc_p_vs_stack_reject_holm")
            row = []
            for g in GATES:
                if (ds, g) not in delta.index:
                    row.append(f"{'—':>{width}}")
                    continue
                n = sig.get((ds, g))
                label = frac(delta[(ds, g)]) + (f" ({n.split('/')[0]})" if n else "")
                row.append(f"{label:>{width}}")
            out.append(f"| {DISPLAY[ds]:<{W}} | {SUBSET_RU[subset]:<9} | "
                       + " | ".join(row) + " |")
    return "\n".join(out)


# ------------------------------------------------------------------ таблица 7
SCHEME_RU = {"attention": "внимание", "moe": "смесь"}


def table_attention(boot: pd.DataFrame, number: int) -> str:
    """Точечные оценки берутся из того же бутстрапа, что и значимость: обе схемы
    обучаются в нём заново на процессоре, и брать середину из одного файла, а
    разброс из другого значило бы сравнивать разные прогоны."""
    cols = ["лучшая одиночная", "взвешивание", "надстройка"]
    width = max(len(c) for c in cols) + 4
    out = [f"Табл. {number}. Внимание и разреженная смесь двух экспертов из четырёх, "
           f"семейство глубоких моделей: разность площади под кривой относительно трёх "
           f"точек отсчёта, среднее по пяти разбиениям, в скобках — число разбиений со "
           f"значимой разностью после поправки Холма.", ""]
    cells = " | ".join(f"{c:>{width}}" for c in cols)
    rule = " | ".join(["-" * (width - 1) + ":"] * len(cols))
    out += [f"| {'набор':<{W}} | способ    | {cells} |",
            f"| {'-' * W} | --------- | {rule} |"]
    for ds in ORDER:
        for scheme in ("attention", "moe"):
            d = boot[(boot.dataset == ds) & (boot.scheme == scheme)]
            row = []
            for base in ("best", "static", "stack"):
                if d.empty:
                    row.append(f"{'—':>{width}}")
                    continue
                value = frac(d[f"delta_auc_vs_{base}"].mean())
                n = int(d[f"auc_p_vs_{base}_reject_holm"].sum())
                row.append(f"{value + ' (' + str(n) + ')':>{width}}")
            out.append(f"| {DISPLAY[ds]:<{W}} | {SCHEME_RU[scheme]:<9} | "
                       + " | ".join(row) + " |")
    return "\n".join(out)


# ------------------------------------------------------------------ таблица 8
def table_ccce(points: pd.DataFrame, boot: pd.DataFrame, number: int,
               crossfit: pd.DataFrame | None = None) -> str:
    width = max(len(v) for v in VARIANT_RU.values()) + 3
    out = [f"Табл. {number}. Трёхступенчатый ансамбль, семейство глубоких моделей: "
           f"разность площади под кривой относительно взвешивания по компонентам знания "
           f"и относительно лучшей одиночной модели, среднее по пяти разбиениям, в "
           f"скобках — число разбиений со значимой разностью после поправки Холма.", ""]
    for base, title in (("static", "Против взвешивания по компонентам знания:"),
                        ("best", "Против лучшей одиночной модели:")):
        cols = {v: f"lift_{v}_vs_{base}_auc" for v in VARIANTS}
        missing = [c for c in cols.values() if c not in points.columns]
        if missing:
            sys.exit(f"ФАТАЛЬНО: в {CCCE} нет колонок {missing}")
        delta = points.groupby("dataset")[list(cols.values())].mean()
        delta.columns = VARIANTS
        sig = sig_counts(boot, ["dataset", "variant"], f"auc_p_vs_{base}_reject_holm")
        cells = " | ".join(f"{VARIANT_RU[v]:>{width}}" for v in VARIANTS)
        rule = " | ".join(["-" * (width - 1) + ":"] * len(VARIANTS))
        out += ["", title, "",
                f"| {'набор':<{W}} | {cells} |", f"| {'-' * W} | {rule} |"]
        for ds in ORDER:
            row = []
            for v in VARIANTS:
                n = sig.get((ds, v), "—/5").split("/")[0]
                row.append(f"{frac(delta.loc[ds, v]) + ' (' + n + ')':>{width}}")
            out.append(f"| {DISPLAY[ds]:<{W}} | " + " | ".join(row) + " |")

    if crossfit is not None:
        cols = {v: f"lift_{v}_vs_static_auc" for v in VARIANTS}
        delta = crossfit.groupby("dataset")[list(cols.values())].mean()
        delta.columns = VARIANTS
        cells = " | ".join(f"{VARIANT_RU[v]:>{width}}" for v in VARIANTS)
        rule = " | ".join(["-" * (width - 1) + ":"] * len(VARIANTS))
        out += ["", "То же против взвешивания, когда первая и третья ступени обучены "
                "вне блока (бутстрап для этого варианта не считался):", "",
                f"| {'набор':<{W}} | {cells} |", f"| {'-' * W} | {rule} |"]
        for ds in ORDER:
            row = " | ".join(f"{frac(delta.loc[ds, v]):>{width}}" for v in VARIANTS)
            out.append(f"| {DISPLAY[ds]:<{W}} | {row} |")
    return "\n".join(out)


# ------------------------------------------------------------------ таблица 9
def table_calibration(simple: pd.DataFrame, stack: pd.DataFrame, gate: dict,
                      ccce: pd.DataFrame, ccce_boot: pd.DataFrame,
                      gate_boot: dict, number: int) -> str:
    """Ошибка калибровки: те же способы, что и по площади под кривой, глубокие модели."""
    avg = simple[(simple.aggregator == "arithmetic_mean") & (simple.subset == "deep")]
    xgb = stack[(stack.meta_learner == "xgb_stacked") & (stack.subset == "deep")]
    gat = gate["deep"][gate["deep"].meta_learner == "static_concept_weights"]
    gat_b = gate_boot["deep"]
    gat_b = gat_b[gat_b.meta_learner == "static_concept_weights"]

    def cell(value, sig):
        return frac(value) + (f" ({sig})" if sig is not None else "")

    cols = ["усреднение", "надстройка", "взвешивание", "три ступени", "три ст. / взвеш."]
    width = max(len(c) for c in cols) + 4
    out = [f"Табл. {number}. Ошибка калибровки, семейство глубоких моделей: разность "
           f"относительно лучшей одиночной модели, среднее по пяти разбиениям, в "
           f"скобках — число разбиений со значимой разностью после поправки Холма. "
           f"Отрицательное значение означает, что ансамбль калиброван лучше. Последний "
           f"столбец — трёхступенчатая схема относительно взвешивания по компонентам "
           f"знания. Надстройка — градиентный бустинг.", ""]
    cells = " | ".join(f"{c:>{width}}" for c in cols)
    rule = " | ".join(["-" * (width - 1) + ":"] * len(cols))
    out += [f"| {'набор':<{W}} | {cells} |", f"| {'-' * W} | {rule} |"]
    for ds in ORDER:
        row = []
        a = avg[avg.dataset == ds]
        row.append(cell(a.delta_ece.mean(), int(a.ece_p_reject_holm.sum())))
        x = xgb[xgb.dataset == ds]
        row.append(cell(x.delta_ece.mean(), int(x.ece_p_reject_holm.sum()))
                   if not x.empty else "—")
        g = gat[gat.dataset == ds]
        gb = gat_b[gat_b.dataset == ds]
        row.append(cell(g.lift_gated_vs_best_ece.mean(),
                        int(gb.ece_p_vs_best_reject_holm.sum()) if not gb.empty else None))
        c = ccce[ccce.dataset == ds]
        cb = ccce_boot[(ccce_boot.dataset == ds) & (ccce_boot.variant == "full")]
        row.append(cell(c.lift_full_vs_best_ece.mean(),
                        int(cb.ece_p_vs_best_reject_holm.sum()) if not cb.empty else None))
        row.append(cell(c.lift_full_vs_static_ece.mean(),
                        int(cb.ece_p_vs_static_reject_holm.sum()) if not cb.empty else None))
        out.append(f"| {DISPLAY[ds]:<{W}} | "
                   + " | ".join(f"{v:>{width}}" for v in row) + " |")
    return "\n".join(out)


# ------------------------------------------------------------------- для прозы
def prose(comp, simple, stack, gate, gate_boot, attn_moe, ccce, spread) -> str:
    lines = ["--- величины, на которые ссылается проза ---"]

    # компоненты
    auc = comp.groupby(["dataset", "subset", "model"]).auc.mean()
    gaps = {}
    for subset in ("classical", "deep"):
        for ds in ORDER:
            row = auc[ds, subset].sort_values(ascending=False)
            gaps[(ds, subset)] = row.iloc[0] - row.iloc[1]
    for subset in ("classical", "deep"):
        g = {ds: gaps[(ds, subset)] for ds in ORDER}
        top = max(g, key=g.get)
        rest = sorted(v for k, v in g.items() if k != top)
        lines.append(f"{SUBSET_RU[subset]}: наибольший отрыв лучшей от второй — "
                     f"{plain(g[top])} на {DISPLAY[top]}, на остальных не более "
                     f"{plain(rest[-1])}")
    ed = auc["ednet", "deep"].sort_values(ascending=False)
    lines.append("EdNet, глубокие: " + ", ".join(f"{MODEL_RU[m]} {plain(v)}"
                                                 for m, v in ed.items()))
    # связь отрыва с выигрышем усреднения: тенденция есть, законом её называть нельзя
    avg = simple[simple.aggregator == "arithmetic_mean"]
    avg = avg.groupby(["dataset", "subset"]).delta_auc.mean()
    stats = pd.DataFrame({
        "gap": [gaps[k] for k in avg.index],
        "spread": [auc[k].max() - auc[k].min() for k in avg.index],
        "delta": avg.values})
    for ds in ("assist2012", "ednet"):
        r = auc[ds, "deep"].sort_values(ascending=False)
        lines.append(f"{DISPLAY[ds]}, глубокие: отрыв {plain(r.iloc[0] - r.iloc[1])}, "
                     f"размах {plain(r.iloc[0] - r.iloc[-1])}")
    lines.append(f"по четырнадцати ячейкам ранговая корреляция отрыва с выигрышем "
                 f"усреднения {stats.gap.corr(stats.delta, method='spearman'):+.2f}, "
                 f"размаха с выигрышем {stats.spread.corr(stats.delta, method='spearman'):+.2f}; "
                 f"второй по величине отрыв — {plain(sorted(gaps.values())[-2])}")

    # усреднение
    a = simple[simple.aggregator == "arithmetic_mean"]
    d = a.groupby(["dataset", "subset"]).delta_auc.mean().unstack()
    size = simple.groupby(["dataset", "subset"]).n_rows.max().unstack()
    ratio = (size.classical / size.deep - 1) * 100
    lines.append(f"у простых моделей строк больше на {ratio.min():.1f}–{ratio.max():.1f} %")
    lines += [f"усреднение, простые: от {frac(d.classical.min())} до {frac(d.classical.max())}, "
              f"отрицательно на {int((d.classical < 0).sum())} наборах из {len(d)}",
              f"усреднение, глубокие: от {frac(d.deep.min())} до {frac(d.deep.max())}, "
              f"отрицательно на {int((d.deep < 0).sum())} наборах из {len(d)}"]
    agg = simple.groupby(["dataset", "subset", "aggregator"]).delta_auc.mean().unstack()
    for name in ("median", "rank_mean", "logit_mean"):
        lines.append(f"{AGG_RU[name]} лучше среднего арифметического в "
                     f"{int((agg[name] > agg.arithmetic_mean).sum())} ячейках из {len(agg)}")
    sig = sig_counts(a, ["dataset", "subset"], "auc_p_reject_holm")
    lines.append(f"среднее арифметическое значимо на всех пяти разбиениях в "
                 f"{int((sig == '5/5').sum())} ячейках из {len(sig)}")

    # надстройка
    sd = stack.groupby(["dataset", "subset", "meta_learner"]).delta_auc.mean().unstack()
    learned = sd.drop(columns=["bma"])
    ssig = sig_counts(stack, ["dataset", "subset", "meta_learner"], "auc_p_reject_holm")
    ssig = ssig.unstack().drop(columns=["bma"])
    lines += [f"надстройка без байесовского усреднения: от {frac(learned.min().min())} "
              f"до {frac(learned.max().max())}, отрицательна в "
              f"{int((learned < 0).sum().sum())} ячейках из {learned.size}, значима на "
              f"всех пяти разбиениях в {int((ssig == '5/5').sum().sum())} ячейках",
              f"байесовское усреднение равно нулю в {int((sd.bma == 0).sum())} ячейках "
              f"из {len(sd)}, из них у простых моделей "
              f"{int((sd.bma.xs('classical', level='subset') == 0).sum())}; "
              f"минимум {frac(sd.bma.min())}",
              f"логистическая и вариант с подбором штрафа расходятся не более чем на "
              f"{(sd.logistic_stacked - sd.ridge_stacked).abs().max():.6f}",
              f"бустинг лучше логистической в {int((sd.xgb_stacked > sd.logistic_stacked).sum())} "
              f"ячейках из {len(sd)}"]

    # уровень подробности у глубоких моделей
    gr = pd.read_csv(GRANULARITY).set_index("dataset")
    ok = gr[gr.n_rows_question > 0]
    worst = ok.expansion.idxmax()
    lines += ["", "уровень подробности:",
              f"разворот строк у глубоких моделей от {ok.expansion.min():.2f} до "
              f"{ok.expansion.max():.2f}, наибольший на {DISPLAY[worst]}",
              f"{DISPLAY[worst]}: строк {int(ok.loc[worst, 'n_rows_question'])} против "
              f"{int(ok.loc[worst, 'n_rows_concept'])}, площадь под кривой у лучшей "
              f"одиночной {plain(ok.loc[worst, 'auc_best_question'])} против "
              f"{plain(ok.loc[worst, 'auc_best_concept'])}",
              f"наборов, где разворота нет: "
              f"{int((ok.expansion <= 1.01).sum())} из {len(ok)}", ""]

    # взвешивание против обучаемой надстройки — на одних и тех же строках
    for subset in ("classical", "deep"):
        g = gate[subset]
        g = g[g.meta_learner == "static_concept_weights"].groupby(
            "dataset").lift_gated_vs_stack_auc.mean()
        b = gate_boot[subset]
        b = b[b.meta_learner == "static_concept_weights"].groupby(
            "dataset").auc_p_vs_stack_reject_holm.sum()
        lines.append(f"взвешивание против надстройки, {SUBSET_GEN[subset]}: медиана "
                     f"{frac(g.median())}, от {frac(g.min())} до {frac(g.max())}, "
                     f"отрицательно на {int((g < 0).sum())} наборах из {len(g)}, "
                     f"значимо на всех пяти разбиениях на {int((b == 5).sum())}")

    # взвешивание по компонентам знания, оба семейства
    for subset in ("classical", "deep"):
        g = gate[subset].groupby(
            ["dataset", "meta_learner"]).lift_gated_vs_best_auc.mean().unstack()
        diff = g["global_stack_concept_intercept"] - g["static_concept_weights"]
        name = SUBSET_GEN[subset]
        lines += [f"полное взвешивание, {name}: от {frac(g['static_concept_weights'].min())} "
                  f"до {frac(g['static_concept_weights'].max())}, положительно на "
                  f"{int((g['static_concept_weights'] > 0).sum())} наборах из {len(g)}",
                  f"распределитель весов, {name}: отрицателен на "
                  f"{int((g['linear_gating'] < 0).sum())} наборах, ниже полного "
                  f"взвешивания на {int((g['linear_gating'] < g['static_concept_weights']).sum())}",
                  f"свободный член, {name}: отличается от полного не более чем на "
                  f"{diff.abs().max():.4f} ({DISPLAY[diff.abs().idxmax()]}), не хуже "
                  f"полного на {int((diff >= 0).sum())} наборах из {len(g)}"]
        fit = gate[subset][gate[subset].meta_learner == "static_concept_weights"]
        fit = fit.groupby("dataset").n_concepts_fit
        lines.append(f"компонентов со своими весами, {name}: от {int(fit.min().min())} "
                     f"до {int(fit.max().max())}")

    # многокомпонентные задания: у простых моделей строка — задание, и компонент
    # приходится выбирать
    import numpy as np
    shares = {}
    for ds in ORDER:
        f = CLASSICAL_CONCEPTS / f"{ds}.npz"
        if f.exists():
            n = np.load(f)["test_n_concepts"]
            shares[ds] = float((n > 1).mean())
    if shares:
        worst = sorted(shares, key=shares.get, reverse=True)
        rest = max(v for k, v in shares.items() if k not in worst[:2])
        lines += [f"заданий с несколькими компонентами: {shares[worst[0]] * 100:.1f} % на "
                  f"{DISPLAY[worst[0]]}, {shares[worst[1]] * 100:.1f} % на "
                  f"{DISPLAY[worst[1]]}, на остальных не более {rest * 100:.1f} %"]

    # внимание и разреженная смесь — из того же бутстрапа, где считалась значимость
    for scheme, ru in (("attention", "внимание"), ("moe", "смесь")):
        d = attn_moe[attn_moe.scheme == scheme]
        for base, label in (("best", "лучшей одиночной"), ("static", "взвешивания"),
                            ("stack", "надстройки")):
            m = d.groupby("dataset")[f"delta_auc_vs_{base}"].mean()
            sig = d.groupby("dataset")[f"auc_p_vs_{base}_reject_holm"].sum()
            lines.append(f"{ru} против {label}: от {frac(m.min())} до {frac(m.max())}, "
                         f"хуже на {int((m < 0).sum())} наборах из {len(m)}, значимо на "
                         f"всех пяти разбиениях на {int((sig == 5).sum())}")
    lines.append(f"время обучения: внимание {attn_moe[attn_moe.scheme == 'attention'].fit_sec.mean():.1f} с, "
                 f"смесь {attn_moe[attn_moe.scheme == 'moe'].fit_sec.mean():.1f} с на разбиение")

    # то же самое, обученное на ускорителе: расхождение прогонов
    if ATTN_MPS.exists() and MOE_MPS.exists():
        new_d = attn_moe.groupby(["dataset", "scheme"]).delta_auc_vs_static.mean().unstack()
        old_a = pd.read_csv(ATTN_MPS).groupby("dataset").lift_attn_vs_static_auc.mean()
        old_m = pd.read_csv(MOE_MPS).groupby("dataset").lift_moe_vs_static_auc.mean()
        da = (new_d["attention"] - old_a).abs()
        dm = (new_d["moe"] - old_m).abs()
        lines.append(f"те же схемы, обученные на ускорителе, дают против взвешивания другую "
                     f"величину: расхождение до {da.max():.4f} у внимания и до {dm.max():.4f} "
                     f"у смеси")

    # чувствительность к порогу, при котором компонент получает свои веса
    nm = pd.read_csv(NMIN)
    lines.append("")
    lines.append("порог для собственных весов:")
    for subset in ("classical", "deep"):
        x = nm[nm.subset == subset]
        t = x.groupby(["n_min", "dataset"]).agg(
            full=("lift_full_vs_best", "mean"),
            inter=("lift_intercept_vs_best", "mean"),
            gap=("gap_full_minus_intercept", "mean"))
        by = t.groupby("n_min")
        lo, hi = sorted(x.n_min.unique())[0], sorted(x.n_min.unique())[-1]
        lines.append(
            f"{SUBSET_GEN[subset]}: при пороге от {lo} до {hi} полное взвешивание "
            f"меняется с {frac(by.full.mean().min())} до {frac(by.full.mean().max())}, "
            f"вариант со свободным членом — с {frac(by.inter.mean().min())} до "
            f"{frac(by.inter.mean().max())}; разрыв между ними остаётся в пределах "
            f"от {frac(t.gap.min())} до {frac(t.gap.max())}")

    # разброс внимания и смеси по начальному состоянию
    sd = pd.read_csv(SEEDS)
    for scheme, ru in (("attention", "внимание"), ("moe", "смесь")):
        x = sd[sd.scheme == scheme]
        per = x.groupby(["dataset", "fold"]).lift_vs_static.agg(["min", "max", "mean"])
        flip = int(((per["min"] < 0) & (per["max"] > 0)).sum())
        lines.append(
            f"{ru}, пять начальных состояний: размах внутри ячейки медианно "
            f"{(per['max'] - per['min']).median():.4f}, наибольший "
            f"{(per['max'] - per['min']).max():.4f}; знак разности с взвешиванием "
            f"зависит от начального состояния в {flip} ячейках из {len(per)}")

    # трёхступенчатая схема
    st = ccce.groupby("dataset")[[f"lift_{v}_vs_static_auc" for v in VARIANTS]].mean()
    st.columns = VARIANTS
    bs = ccce.groupby("dataset")[[f"lift_{v}_vs_best_auc" for v in VARIANTS]].mean()
    bs.columns = VARIANTS
    cf = pd.read_csv(CCCE_CF).groupby("dataset")[
        [f"lift_{v}_vs_static_auc" for v in VARIANTS]].mean()
    cf.columns = VARIANTS
    grew = int(((cf.full - st.full) < 0).sum())
    lines += [f"кросс-фит первой ступени: полная схема от {frac(cf.full.max())} до "
              f"{frac(cf.full.min())}, хуже взвешивания на {int((cf.full < 0).sum())} "
              f"наборах из {len(cf)}; потеря выросла на {grew} наборах",
              f"кросс-фит, без первой ступени: от {frac(cf.no_s1.min(), 5)} до "
              f"{frac(cf.no_s1.max(), 5)}; с общей первой ступенью: от "
              f"{frac(cf.global_s1.min(), 5)} до {frac(cf.global_s1.max(), 5)}",
              f"кросс-фит, наибольший рост потери {frac((cf.full - st.full).min())} на "
              f"{DISPLAY[(cf.full - st.full).idxmin()]}"]
    lines += [f"полная схема против взвешивания: от {frac(st.full.min())} до "
              f"{frac(st.full.max())}, хуже на {int((st.full < 0).sum())} наборах из {len(st)}",
              f"без первой ступени: от {frac(st.no_s1.min(), 5)} до {frac(st.no_s1.max(), 5)}",
              f"общая калибровка на первой ступени: от {frac(st.global_s1.min(), 5)} до "
              f"{frac(st.global_s1.max(), 5)}",
              f"без третьей ступени: от {frac(st.no_s3.min())} до {frac(st.no_s3.max())}",
              f"против лучшей одиночной: полная от {frac(bs.full.min())} до {frac(bs.full.max())}, "
              f"без первой ступени от {frac(bs.no_s1.min())} до {frac(bs.no_s1.max())}"]

    # калибровка
    de = a.groupby(["dataset", "subset"]).delta_ece.mean().unstack()
    se = stack.groupby(["dataset", "subset", "meta_learner"]).delta_ece.mean()
    ge = gate["deep"][gate["deep"].meta_learner == "static_concept_weights"]
    ge = ge.groupby("dataset").lift_gated_vs_best_ece.mean()
    ce = ccce.groupby("dataset")[[f"lift_{v}_vs_static_ece" for v in VARIANTS]].mean()
    ce.columns = VARIANTS
    lines += ["", "калибровка:",
              f"усреднение ухудшает калибровку простых моделей на {int((de.classical > 0).sum())} "
              f"наборах из {len(de)}, глубоких — на {int((de.deep > 0).sum())}",
              f"надстройка улучшает калибровку в {int((se < 0).sum())} ячейках из {se.size}, "
              f"ухудшает в {int((se > 0).sum())}",
              f"взвешивание улучшает калибровку на {int((ge < 0).sum())} наборах из {len(ge)}, "
              f"от {frac(ge.max())} до {frac(ge.min())}",
              f"полная схема калибрована хуже взвешивания на {int((ce.full > 0).sum())} наборах "
              f"из {len(ce)}, от {frac(ce.full.min())} до {frac(ce.full.max())}",
              f"без первой ступени: от {frac(ce.no_s1.min())} до {frac(ce.no_s1.max())}; "
              f"с общей первой ступенью: от {frac(ce.global_s1.min())} до {frac(ce.global_s1.max())}"]

    sp = spread.groupby(["dataset", "subset"])[["best_std", "ensemble_std"]].mean()
    for subset in ("deep", "classical"):
        d = sp.xs(subset, level="subset")
        shrink = int((d.ensemble_std < d.best_std).sum())
        lines.append(f"усреднение сжимает разброс предсказаний у {SUBSET_GEN[subset]} "
                     f"моделей на {shrink} наборах из {len(d)}")
    for ds in ("assist2017", "ednet"):
        r = sp.loc[(ds, "deep")]
        lines.append(f"{DISPLAY[ds]}, глубокие: разброс {plain(r.best_std)} → "
                     f"{plain(r.ensemble_std)}")
    return "\n".join(lines)


def table_runs(block: str) -> list[list[str]]:
    """Строки таблиц внутри одного блока: у таблицы 8 их две, у остальных одна."""
    runs, current = [], []
    for line in block.split("\n"):
        if line.startswith("|"):
            current.append(line)
        elif current:
            runs.append(current)
            current = []
    if current:
        runs.append(current)
    return runs


def splice(path: Path, blocks: dict[int, str]) -> int:
    """Заменить таблицы в рукописи свежими, не трогая подписи и прозу."""
    text = path.read_text(encoding="utf-8")
    replaced = 0
    for number, block in blocks.items():
        anchor = f"**Табл. {number}.**"
        at = text.find(anchor)
        if at < 0:
            print(f"  таблицы {number} в рукописи нет — пропущена")
            continue
        fresh = table_runs(block)
        lines = text[at:].split("\n")
        out, run_index, i = [], 0, 0
        while i < len(lines):
            if lines[i].startswith("|") and run_index < len(fresh):
                while i < len(lines) and lines[i].startswith("|"):
                    i += 1
                out.extend(fresh[run_index])
                run_index += 1
                continue
            if run_index == len(fresh):
                break
            out.append(lines[i])
            i += 1
        text = text[:at] + "\n".join(out + lines[i:])
        replaced += run_index
    path.write_text(text, encoding="utf-8")
    return replaced


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--into", type=Path, help="подставить таблицы в рукопись")
    args = ap.parse_args()

    comp = load(COMPONENTS)
    simple = load(SIMPLE_BOOT)
    stack = load(STACK_BOOT)
    gate = {k: load(v) for k, v in GATING.items()}
    gate_boot = {k: load(v) for k, v in GATING_BOOT.items()}
    attn_moe = load(ATTN_MOE_BOOT)
    ccce, ccce_boot = load(CCCE), load(CCCE_BOOT)

    tables = {
        1: table_datasets(simple),
        2: table_components(comp, "classical", 2),
        3: table_components(comp, "deep", 3),
        4: table_simple(simple, 4),
        5: table_stacking(stack, 5),
        6: table_gating(gate, gate_boot, 6),
        7: table_gating_vs_stack(gate, gate_boot, 7),
        8: table_attention(attn_moe, 8),
        9: table_ccce(ccce, ccce_boot, 9, load(CCCE_CF)),
        10: table_calibration(simple, stack, gate, ccce, ccce_boot, gate_boot, 10),
    }
    if args.into:
        n = splice(args.into, tables)
        print(f"{args.into}: подставлено таблиц {n}")
        return 0
    print("\n\n".join(list(tables.values())
                       + [prose(comp, simple, stack, gate, gate_boot, attn_moe, ccce, load(SPREAD))]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
