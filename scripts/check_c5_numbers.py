"""Сверить каждое число рукописи с результатами расчётов.

Сверщик намеренно не пользуется `make_c5_tables.py`. Тот сценарий печатает
таблицы, этот проверяет их — и если бы оба считали одним и тем же кодом, ошибка в
нём осталась бы незамеченной. Числа берутся из тех же CSV, но соответствие
«столбец таблицы — поле артефакта» выписано здесь заново, потому что ошибка
обычно живёт именно в нём.

Покрыто:
  * таблицы 1-9, по ячейкам, включая столбцы значимости;
  * прозаические утверждения — по имени: шаблон должен найтись, и число в нём
    должно совпасть, поэтому пропавшее утверждение роняет проверку так же, как
    неверное;
  * числа, записанные словами («в девяти ячейках из четырнадцати»);
  * обе аннотации: их числа и длина в словах;
  * ключевые слова: одинаковое число позиций, не больше десяти;
  * библиография: нумерация по первому упоминанию, без пропусков и сирот;
  * гигиена: нет внутренних обозначений проекта, нет величин AUC в процентах,
    нет ссылок на неопубликованное.

Не покрыто и требует человека: верно ли проза описывает то, что в числе, и
показывают ли рисунки то, что написано в подписях.

Запуск:
  python -m scripts.check_c5_numbers [путь к рукописи]
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ktx import paths

ENS = paths.ARTIFACTS_DIR / "ensembles"
PAPER = paths.RESEARCH_DIR / "paper" / "c5_ensembles_ru.md"

DISPLAY = {
    "algebra2005": "Algebra-2005", "assist2009": "ASSISTments-2009",
    "assist2012": "ASSISTments-2012", "assist2015": "ASSISTments-2015",
    "assist2017": "ASSISTments-2017", "bridge2algebra2006": "Bridge-to-Algebra-2006",
    "ednet": "EdNet-KT1-5k",
}
BACK = {v: k for k, v in DISPLAY.items()}
SUBSET = {"простые": "classical", "глубокие": "deep"}

WORDS = {
    "нуля": 0, "одной": 1, "одном": 1, "двух": 2, "трёх": 3, "четырёх": 4, "пяти": 5,
    "шести": 6, "семи": 7, "девяти": 9, "одиннадцати": 11, "тринадцати": 13,
    "четырнадцати": 14, "пятидесяти двух": 52, "пятидесяти девяти": 59,
    "шестидесяти пяти": 65,
}

results: list[tuple[bool, str, str]] = []


def check(ok: bool, name: str, detail: str = "") -> None:
    results.append((bool(ok), name, detail))


# ---------------------------------------------------------------- текст
def manuscript(path=None) -> str:
    text = (path or PAPER).read_text(encoding="utf-8")
    marks = list(re.finditer(r"^# Текст статьи$", text, re.M))
    if not marks:
        raise SystemExit(f"{PAPER}: не найден раздел «Текст статьи»")
    body = text[marks[-1].end():]
    # рабочая заметка между заголовком и титульным блоком в статью не идёт
    start = body.find("## Титульный блок")
    return body[start:] if start >= 0 else body


def section(text: str, heading: str) -> str:
    """Тело одного раздела: утверждение проверяется там, где оно сделано.

    Обе аннотации повторяют главные числа статьи, и поиск по всему тексту был бы
    доволен аннотацией даже после того, как утверждение выпало из раздела.
    """
    m = re.search(rf"^#+\s+{re.escape(heading)}\s*$", text, re.M)
    if not m:
        return ""
    rest = text[m.end():]
    nxt = re.search(r"^#{2,4}\s+", rest, re.M)
    return rest[:nxt.start()] if nxt else rest


def num(raw: str) -> float:
    return float(raw.replace("−", "-").replace(" ", "").replace(" ", ""))


def decimals(raw: str) -> int:
    return len(raw.split(".")[1]) if "." in raw else 0


def matches(raw: str, expected: float) -> bool:
    return abs(num(raw) - expected) <= 0.5 * 10 ** (-decimals(raw)) + 1e-12


def claim(text: str, name: str, pattern: str, expected: float) -> None:
    m = re.search(pattern, text)
    if not m:
        check(False, name, f"утверждение не найдено; шаблон: {pattern}")
        return
    raw = m.group(1)
    check(matches(raw, expected), name, f"в тексте {raw!r}, из артефактов {expected}")


def word_claim(text: str, name: str, pattern: str, expected: int) -> None:
    m = re.search(pattern, text)
    if not m:
        check(False, name, f"утверждение не найдено; шаблон: {pattern}")
        return
    got = WORDS.get(m.group(1).strip())
    check(got == expected, name, f"в тексте {m.group(1)!r} = {got}, ожидалось {expected}")


def md_table(text: str, anchor: str, which: int = 0) -> list[list[str]]:
    """Строки одной markdown-таблицы после якоря; which=1 — вторая таблица блока."""
    i = text.find(anchor)
    if i < 0:
        return []
    tables, rows, started = [], [], False
    for line in text[i:].split("\n"):
        s = line.strip()
        if s.startswith("|"):
            started = True
            cells = [c.strip() for c in s.strip("|").split("|")]
            if not set("".join(cells)) <= set("-: "):
                rows.append(cells)
        elif started:
            tables.append(rows)
            rows, started = [], False
            if s.startswith("**Табл.") or s.startswith("## "):
                break
    if started:
        tables.append(rows)
    if len(tables) <= which:
        return []
    return tables[which][1:]


# ------------------------------------------------------------- артефакты
def load() -> dict:
    import json
    return {
        "comp": pd.read_csv(ENS / "component_models.csv"),
        "simple": pd.read_csv(ENS / "cluster_bootstrap_simple_ensembles.csv"),
        "stack": pd.read_csv(ENS / "cluster_bootstrap_stacking_all_v4.csv"),
        "gate": {"deep": pd.read_csv(ENS / "gating_deep.csv"),
                 "classical": pd.read_csv(ENS / "gating_classical.csv")},
        "gate_boot": {"deep": pd.read_csv(ENS / "cluster_bootstrap_gating.csv"),
                      "classical": pd.read_csv(
                          ENS / "cluster_bootstrap_gating_classical.csv")},
        "attn_moe": pd.read_csv(ENS / "cluster_bootstrap_attention_moe.csv"),
        "ccce_cf": pd.read_csv(ENS / "ccce_crossfit_deep.csv"),
        "ccce": pd.read_csv(ENS / "ccce_deep.csv"),
        "ccce_boot": pd.read_csv(ENS / "cluster_bootstrap_ccce.csv"),
        "spread": pd.read_csv(ENS / "prediction_spread.csv"),
        "align": pd.read_csv(ENS / "family_alignment.csv").set_index("dataset"),
        "gran": pd.read_csv(ENS / "deep_granularity.csv").set_index("dataset"),
        "seeds": pd.read_csv(ENS / "attention_seed_spread.csv"),
        "nmin": pd.read_csv(ENS / "gating_nmin_sensitivity.csv"),
        "attn_mps": pd.read_csv(ENS / "attention_gating_deep.csv"),
        "moe_mps": pd.read_csv(ENS / "moe_deep.csv"),
        "multi": {ds: float((np.load(ENS / "classical_concepts" / f"{ds}.npz")
                             ["test_n_concepts"] > 1).mean())
                  for ds in DISPLAY
                  if (ENS / "classical_concepts" / f"{ds}.npz").exists()},
        "cfg": json.loads(paths.DATA_CONFIG_JSON.read_text(encoding="utf-8")),
    }


def mean_of(df: pd.DataFrame, where: dict, column: str) -> float:
    d = df
    for k, v in where.items():
        d = d[d[k] == v]
    if d.empty:
        raise KeyError(f"{where} нет в артефакте")
    return float(d[column].mean())


def sig_of(df: pd.DataFrame, where: dict, column: str) -> int:
    d = df
    for k, v in where.items():
        d = d[d[k] == v]
    return int(d[column].sum())


# --------------------------------------------------------------- таблицы
def check_table1(text: str, a: dict) -> None:
    rows = md_table(text, "**Табл. 1.**")
    check(len(rows) == 7, "Т1 / число строк", f"строк {len(rows)}")
    for r in rows:
        ds = BACK.get(r[0])
        if ds is None:
            check(False, f"Т1 {r[0]}", "неизвестный набор")
            continue
        d = a["simple"][a["simple"].dataset == ds]
        check(matches(r[1], d.n_students.max()), f"Т1 {r[0]} / учащихся",
              f"в тексте {r[1]!r}, в артефакте {d.n_students.max()}")
        check(matches(r[2], a["cfg"][ds]["num_c"]), f"Т1 {r[0]} / компонентов",
              f"в тексте {r[2]!r}, в конфигурации {a['cfg'][ds]['num_c']}")
        num_q = a["cfg"][ds].get("num_q", 0)
        want = "—" if not num_q else str(num_q)
        check(r[3] == want, f"Т1 {r[0]} / заданий", f"в тексте {r[3]!r}, ожидалось {want!r}")
        for j, sub in ((4, "classical"), (5, "deep")):
            exp = d[d.subset == sub].n_rows.max()
            check(matches(r[j], exp), f"Т1 {r[0]} / строк {sub}",
                  f"в тексте {r[j]!r}, в артефакте {exp}")


def check_components(text: str, a: dict, anchor: str, subset: str, models: list[str]) -> None:
    rows = md_table(text, anchor)
    check(len(rows) == 7, f"{anchor} / число строк", f"строк {len(rows)}")
    comp = a["comp"]
    for r in rows:
        ds = BACK.get(r[0])
        if ds is None:
            check(False, f"{anchor} {r[0]}", "неизвестный набор")
            continue
        vals = {}
        for j, model in enumerate(models, start=1):
            exp = mean_of(comp, {"dataset": ds, "subset": subset, "model": model}, "auc")
            vals[model] = exp
            check(matches(r[j], exp), f"{anchor} {r[0]} / {model}",
                  f"в тексте {r[j]!r}, в артефакте {exp:.6f}")
        top = sorted(vals.values(), reverse=True)
        check(matches(r[len(models) + 1], top[0] - top[1]), f"{anchor} {r[0]} / отрыв",
              f"в тексте {r[len(models) + 1]!r}, в артефакте {top[0] - top[1]:.6f}")


AGGS = ["arithmetic_mean", "geometric_mean", "logit_mean", "median", "rank_mean"]


def check_table4(text: str, a: dict) -> None:
    rows = md_table(text, "**Табл. 4.**")
    check(len(rows) == 14, "Т4 / число строк", f"строк {len(rows)}")
    for r in rows:
        ds, sub = BACK.get(r[0]), SUBSET.get(r[1])
        for j, agg in enumerate(AGGS, start=2):
            exp = mean_of(a["simple"], {"dataset": ds, "subset": sub, "aggregator": agg},
                          "delta_auc")
            check(matches(r[j], exp), f"Т4 {r[0]}/{r[1]}/{agg}",
                  f"в тексте {r[j]!r}, в артефакте {exp:.6f}")
        n = sig_of(a["simple"], {"dataset": ds, "subset": sub,
                                 "aggregator": "arithmetic_mean"}, "auc_p_reject_holm")
        check(r[7] == f"{n}/5", f"Т4 {r[0]}/{r[1]} / значимость",
              f"в тексте {r[7]!r}, в артефакте {n}/5")


METAS = ["logistic_stacked", "ridge_stacked", "bma", "xgb_stacked", "mlp_stacked"]


def check_table5(text: str, a: dict) -> None:
    rows = md_table(text, "**Табл. 5.**")
    check(len(rows) == 14, "Т5 / число строк", f"строк {len(rows)}")
    for r in rows:
        ds, sub = BACK.get(r[0]), SUBSET.get(r[1])
        for j, meta in enumerate(METAS, start=2):
            d = a["stack"]
            d = d[(d.dataset == ds) & (d.subset == sub) & (d.meta_learner == meta)]
            if d.empty:
                check(r[j] == "—", f"Т5 {r[0]}/{r[1]}/{meta}",
                      f"в тексте {r[j]!r}, в артефакте ячейки нет")
                continue
            m = re.fullmatch(r"([−+]\d+\.\d+) \((\d)\)", r[j])
            if not m:
                check(False, f"Т5 {r[0]}/{r[1]}/{meta}", f"не разобрана ячейка {r[j]!r}")
                continue
            exp = float(d.delta_auc.mean())
            check(matches(m.group(1), exp), f"Т5 {r[0]}/{r[1]}/{meta}",
                  f"в тексте {m.group(1)!r}, в артефакте {exp:.6f}")
            n = int(d.auc_p_reject_holm.sum())
            check(int(m.group(2)) == n, f"Т5 {r[0]}/{r[1]}/{meta} / значимость",
                  f"в тексте {m.group(2)}, в артефакте {n}")


GATES = ["static_concept_weights", "global_stack_concept_intercept", "linear_gating"]


def check_table6(text: str, a: dict) -> None:
    rows = md_table(text, "**Табл. 6.**")
    check(len(rows) == 14, "Т6 / число строк", f"строк {len(rows)}")
    for r in rows:
        ds, sub = BACK.get(r[0]), SUBSET.get(r[1])
        for j, gate in enumerate(GATES, start=2):
            exp = mean_of(a["gate"][sub], {"dataset": ds, "meta_learner": gate},
                          "lift_gated_vs_best_auc")
            m = re.fullmatch(r"([−+]\d+\.\d+)(?: \((\d)\))?", r[j])
            if not m:
                check(False, f"Т6 {r[0]}/{r[1]}/{gate}", f"не разобрана ячейка {r[j]!r}")
                continue
            check(matches(m.group(1), exp), f"Т6 {r[0]}/{r[1]}/{gate}",
                  f"в тексте {m.group(1)!r}, в артефакте {exp:.6f}")
            b = a["gate_boot"][sub]
            b = b[(b.dataset == ds) & (b.meta_learner == gate)]
            if b.empty:
                check(m.group(2) is None, f"Т6 {r[0]}/{r[1]}/{gate} / значимость",
                      "бутстрапа нет, а в тексте число есть")
            else:
                n = int(b.auc_p_vs_best_reject_holm.sum())
                check(m.group(2) is not None and int(m.group(2)) == n,
                      f"Т6 {r[0]}/{r[1]}/{gate} / значимость",
                      f"в тексте {m.group(2)}, в артефакте {n}")


SCHEMES = {"внимание": "attention", "смесь": "moe"}


def check_table7(text: str, a: dict) -> None:
    """Взвешивание против надстройки: обе схемы на одних и тех же строках."""
    rows = md_table(text, "**Табл. 7.**")
    check(len(rows) == 14, "Т7 / число строк", f"строк {len(rows)}")
    for r in rows:
        ds, sub = BACK.get(r[0]), SUBSET.get(r[1])
        for j, gate in enumerate(GATES, start=2):
            exp = mean_of(a["gate"][sub], {"dataset": ds, "meta_learner": gate},
                          "lift_gated_vs_stack_auc")
            m = re.fullmatch(r"([−+]\d+\.\d+)(?: \((\d)\))?", r[j])
            if not m:
                check(False, f"Т7 {r[0]}/{r[1]}/{gate}", f"не разобрана ячейка {r[j]!r}")
                continue
            check(matches(m.group(1), exp), f"Т7 {r[0]}/{r[1]}/{gate}",
                  f"в тексте {m.group(1)!r}, в артефакте {exp:.6f}")
            b = a["gate_boot"][sub]
            b = b[(b.dataset == ds) & (b.meta_learner == gate)]
            if b.empty:
                check(m.group(2) is None, f"Т7 {r[0]}/{r[1]}/{gate} / значимость",
                      "бутстрапа нет, а в тексте число есть")
            else:
                n = int(b.auc_p_vs_stack_reject_holm.sum())
                check(m.group(2) is not None and int(m.group(2)) == n,
                      f"Т7 {r[0]}/{r[1]}/{gate} / значимость",
                      f"в тексте {m.group(2)}, в артефакте {n}")


def check_table8(text: str, a: dict) -> None:
    rows = md_table(text, "**Табл. 8.**")
    check(len(rows) == 14, "Т8 / число строк", f"строк {len(rows)}")
    boot = a["attn_moe"]
    for r in rows:
        ds, scheme = BACK.get(r[0]), SCHEMES.get(r[1])
        d = boot[(boot.dataset == ds) & (boot.scheme == scheme)]
        for j, base in enumerate(("best", "static", "stack"), start=2):
            m = re.fullmatch(r"([−+]\d+\.\d+) \((\d)\)", r[j])
            if not m:
                check(False, f"Т8 {r[0]}/{r[1]}/{base}", f"не разобрана ячейка {r[j]!r}")
                continue
            exp = float(d[f"delta_auc_vs_{base}"].mean())
            check(matches(m.group(1), exp), f"Т8 {r[0]}/{r[1]}/{base}",
                  f"в тексте {m.group(1)!r}, в артефакте {exp:.6f}")
            n = int(d[f"auc_p_vs_{base}_reject_holm"].sum())
            check(int(m.group(2)) == n, f"Т8 {r[0]}/{r[1]}/{base} / значимость",
                  f"в тексте {m.group(2)}, в артефакте {n}")


VARIANTS = ["full", "no_s1", "global_s1", "no_s3"]


def check_table9(text: str, a: dict) -> None:
    for which, base in ((0, "static"), (1, "best")):
        rows = md_table(text, "**Табл. 9.**", which)
        check(len(rows) == 7, f"Т9/{base} / число строк", f"строк {len(rows)}")
        for r in rows:
            ds = BACK.get(r[0])
            for j, v in enumerate(VARIANTS, start=1):
                exp = mean_of(a["ccce"], {"dataset": ds}, f"lift_{v}_vs_{base}_auc")
                m = re.fullmatch(r"([−+]\d+\.\d+) \((\d)\)", r[j])
                if not m:
                    check(False, f"Т9/{base} {r[0]}/{v}", f"не разобрана ячейка {r[j]!r}")
                    continue
                check(matches(m.group(1), exp), f"Т9/{base} {r[0]}/{v}",
                      f"в тексте {m.group(1)!r}, в артефакте {exp:.6f}")
                n = sig_of(a["ccce_boot"], {"dataset": ds, "variant": v},
                           f"auc_p_vs_{base}_reject_holm")
                check(int(m.group(2)) == n, f"Т9/{base} {r[0]}/{v} / значимость",
                      f"в тексте {m.group(2)}, в артефакте {n}")


def check_table10(text: str, a: dict) -> None:
    rows = md_table(text, "**Табл. 10.**")
    check(len(rows) == 7, "Т10 / число строк", f"строк {len(rows)}")
    simple, stack = a["simple"], a["stack"]
    gate, gate_boot = a["gate"]["deep"], a["gate_boot"]["deep"]
    ccce, ccce_boot = a["ccce"], a["ccce_boot"]
    for r in rows:
        ds = BACK.get(r[0])
        d_avg = simple[(simple.dataset == ds) & (simple.subset == "deep")
                       & (simple.aggregator == "arithmetic_mean")]
        d_xgb = stack[(stack.dataset == ds) & (stack.subset == "deep")
                      & (stack.meta_learner == "xgb_stacked")]
        d_gat = gate[(gate.dataset == ds)
                     & (gate.meta_learner == "static_concept_weights")]
        b_gat = gate_boot[(gate_boot.dataset == ds)
                          & (gate_boot.meta_learner == "static_concept_weights")]
        d_cc = ccce[ccce.dataset == ds]
        b_cc = ccce_boot[(ccce_boot.dataset == ds) & (ccce_boot.variant == "full")]
        cells = [
            ("усреднение", d_avg.delta_ece.mean(),
             int(d_avg.ece_p_reject_holm.sum())),
            ("надстройка", None if d_xgb.empty else d_xgb.delta_ece.mean(),
             None if d_xgb.empty else int(d_xgb.ece_p_reject_holm.sum())),
            ("взвешивание", d_gat.lift_gated_vs_best_ece.mean(),
             None if b_gat.empty else int(b_gat.ece_p_vs_best_reject_holm.sum())),
            ("три ступени", d_cc.lift_full_vs_best_ece.mean(),
             None if b_cc.empty else int(b_cc.ece_p_vs_best_reject_holm.sum())),
            ("три ст. против взвешивания", d_cc.lift_full_vs_static_ece.mean(),
             None if b_cc.empty else int(b_cc.ece_p_vs_static_reject_holm.sum())),
        ]
        for j, (name, exp, sig) in enumerate(cells, start=1):
            if exp is None:
                check(r[j] == "—", f"Т10 {r[0]} / {name}", f"в тексте {r[j]!r}")
                continue
            m = re.fullmatch(r"([−+]\d+\.\d+)(?: \((\d)\))?", r[j])
            if not m:
                check(False, f"Т10 {r[0]} / {name}", f"не разобрана ячейка {r[j]!r}")
                continue
            check(matches(m.group(1), exp), f"Т10 {r[0]} / {name}",
                  f"в тексте {m.group(1)!r}, в артефакте {exp:.6f}")
            if sig is not None:
                check(m.group(2) is not None and int(m.group(2)) == sig,
                      f"Т10 {r[0]} / {name} / значимость",
                      f"в тексте {m.group(2)}, в артефакте {sig}")


def check_crossfit(text: str, a: dict) -> None:
    """Третий блок таблицы 9: первая ступень обучена вне блока."""
    rows = md_table(text, "То же против взвешивания, когда первая и третья ступени")
    check(len(rows) == 7, "кросс-фит / число строк", f"строк {len(rows)}")
    cf = a["ccce_cf"]
    for r in rows:
        ds = BACK.get(r[0])
        for j, v in enumerate(VARIANTS, start=1):
            exp = mean_of(cf, {"dataset": ds}, f"lift_{v}_vs_static_auc")
            check(matches(r[j], exp), f"кросс-фит {r[0]}/{v}",
                  f"в тексте {r[j]!r}, в артефакте {exp:.6f}")


# ------------------------------------------------------------------ проза
def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text)


def check_prose(text: str, a: dict) -> None:
    simple, stack = a["simple"], a["stack"]
    gate, comp = a["gate"], a["comp"]
    attn_moe, ccce = a["attn_moe"], a["ccce"]

    d = simple[simple.aggregator == "arithmetic_mean"]
    d = d.groupby(["dataset", "subset"]).delta_auc.mean().unstack()
    deep_pos = d.deep[d.deep > 0]

    s = norm(section(text, "5.1 Усреднение без настройки"))
    claim(s, "5.1 простые, минимум", r"разность лежит от \+(\d+\.\d+) до", d.classical.min())
    claim(s, "5.1 простые, максимум", r"разность лежит от \+\d+\.\d+ до \+(\d+\.\d+)",
          d.classical.max())
    claim(s, "5.1 глубокие, минимум прироста", r"прирост от \+(\d+\.\d+) до", deep_pos.min())
    claim(s, "5.1 глубокие, максимум прироста", r"прирост от \+\d+\.\d+ до \+(\d+\.\d+)",
          deep_pos.max())
    claim(s, "5.1 ASSISTments-2012", r"на ASSISTments-2012 разность (−\d+\.\d+)", d.deep["assist2012"])
    claim(s, "5.1 EdNet", r"на EdNet-KT1-5k разность (−\d+\.\d+)", d.deep["ednet"])
    word_claim(s, "5.1 число значимых ячеек",
               r"значимо на всех пяти разбиениях в (\S+) ячейках из четырнадцати", 9)

    agg = simple.groupby(["dataset", "subset", "aggregator"]).delta_auc.mean().unstack()
    for label, key in (("логитах", "logit_mean"), ("геометрическое", "geometric_mean"),
                       ("арифметическое", "arithmetic_mean"), ("рангам", "rank_mean"),
                       ("медиана", "median")):
        claim(s, f"5.1 EdNet / {key}", rf"{label} (−\d+\.\d+)", agg.loc[("ednet", "deep"), key])

    ed = comp[(comp.dataset == "ednet") & (comp.subset == "deep")]
    ed = ed.groupby("model").auc.mean().sort_values(ascending=False)
    claim(s, "5.1 EdNet simpleKT", r"simpleKT даёт (\d+\.\d+)", ed.iloc[0])
    claim(s, "5.1 EdNet вторая", r"модели — (\d+\.\d+),", ed.iloc[1])
    claim(s, "5.1 EdNet третья", r"модели — \d+\.\d+, (\d+\.\d+)", ed.iloc[2])
    claim(s, "5.1 EdNet четвёртая", r"модели — \d+\.\d+, \d+\.\d+ и (\d+\.\d+)", ed.iloc[3])
    claim(s, "5.1 EdNet отрыв", r"оторвалась от второй на (\d+\.\d+)", ed.iloc[0] - ed.iloc[1])

    auc = comp.groupby(["dataset", "subset", "model"]).auc.mean()
    gaps = {}
    for sub in ("classical", "deep"):
        for ds in DISPLAY:
            row = auc[ds, sub].sort_values(ascending=False)
            gaps[(ds, sub)] = row.iloc[0] - row.iloc[1]
    deep_gaps = sorted(v for (ds, sub), v in gaps.items() if sub == "deep")
    claim(s, "5.1 отрыв на остальных", r"отрыв не превышает (\d+\.\d+)", deep_gaps[-2])
    stats = pd.DataFrame({"gap": [gaps[k] for k in d.stack().index],
                          "spread": [auc[k].max() - auc[k].min() for k in d.stack().index],
                          "delta": d.stack().values})
    claim(s, "5.1 корреляция отрыва", r"составляет (−\d+\.\d+), а корреляция",
          stats.gap.corr(stats.delta, method="spearman"))
    claim(s, "5.1 корреляция размаха", r"с выигрышем — только (−\d+\.\d+)",
          stats.spread.corr(stats.delta, method="spearman"))
    claim(s, "5.1 второй отрыв", r"Второй по величине отрыв, (\d+\.\d+),",
          sorted(gaps.values())[-2])
    a12 = auc["assist2012", "deep"].sort_values(ascending=False)
    claim(s, "5.1 ASSISTments-2012 отрыв", r"отрыв там всего (\d+\.\d+)", a12.iloc[0] - a12.iloc[1])
    claim(s, "5.1 ASSISTments-2012 размах", r"а\s*размах (\d+\.\d+)", a12.iloc[0] - a12.iloc[-1])

    s = norm(section(text, "5.2 Обучаемая надстройка"))
    sd = stack.groupby(["dataset", "subset", "meta_learner"]).delta_auc.mean().unstack()
    learned = sd.drop(columns=["bma"])
    word_claim(s, "5.2 число ячеек", r"Во всех (\S+ \S+) ячейках, кроме", 52)
    claim(s, "5.2 минимум", r"положительна — от \+(\d+\.\d+) до", learned.min().min())
    claim(s, "5.2 максимум", r"положительна — от \+\d+\.\d+ до \+(\d+\.\d+)", learned.max().max())
    claim(s, "5.2 EdNet усреднение", r"где усреднение теряло (\d+\.\d+)", -d.deep["ednet"])
    claim(s, "5.2 EdNet логистическая", r"надстройка даёт \+(\d+\.\d+)",
          sd.loc[("ednet", "deep"), "logistic_stacked"])
    claim(s, "5.2 EdNet бустинг", r"градиентный бустинг \+(\d+\.\d+)",
          sd.loc[("ednet", "deep"), "xgb_stacked"])
    claim(s, "5.2 EdNet сеть", r"нейронная сеть \+(\d+\.\d+)",
          sd.loc[("ednet", "deep"), "mlp_stacked"])
    word_claim(s, "5.2 нулевых ячеек БУМ", r"ровно ноль в (\S+) ячейках из тринадцати",
               int((sd.bma == 0).sum()))
    word_claim(s, "5.2 нулевых у простых", r"в (\S+) у простых моделей",
               int((sd.bma.xs("classical", level="subset") == 0).sum()))
    claim(s, "5.2 минимум БУМ", r"вплоть до (−\d+\.\d+) на EdNet", sd.bma.min())
    claim(s, "5.2 логистическая против подбора", r"расходятся не более чем на (\d+\.\d+)",
          (sd.logistic_stacked - sd.ridge_stacked).abs().max())
    word_claim(s, "5.2 бустинг лучше",
               r"лучше логистической надстройки в (\S+) ячейках из тринадцати",
               int((sd.xgb_stacked > sd.logistic_stacked).sum()))

    s = norm(section(text, "5.3 Условное взвешивание и вклад свободного члена"))
    g = {k: v.groupby(["dataset", "meta_learner"]).lift_gated_vs_best_auc.mean().unstack()
         for k, v in gate.items()}
    cls, dp = g["classical"], g["deep"]
    claim(s, "5.3 простые, минимум", r"на всех семи наборах, от \+(\d+\.\d+) до",
          cls["static_concept_weights"].min())
    claim(s, "5.3 простые, максимум", r"на всех семи наборах, от \+\d+\.\d+ до \+(\d+\.\d+)",
          cls["static_concept_weights"].max())
    claim(s, "5.3 глубокие, минимум", r"от (−\d+\.\d+) на EdNet-KT1-5k до",
          dp["static_concept_weights"].min())
    claim(s, "5.3 глубокие, максимум", r"до \+(\d+\.\d+) на ASSISTments-2009",
          dp["static_concept_weights"].max())
    word_claim(s, "5.3 распределитель, простые",
               r"у простых моделей он отрицателен на (\S+) наборах из семи",
               int((cls["linear_gating"] < 0).sum()))
    word_claim(s, "5.3 распределитель, глубокие", r"у глубоких — на (\S+), и в обоих",
               int((dp["linear_gating"] < 0).sum()))
    for key, name, pattern in (
            ("classical", "простые",
             r"не более чем на (\d+\.\d+) у простых моделей"),
            ("deep", "глубокие",
             r"не более чем на (\d+\.\d+) у глубоких")):
        diff = (g[key]["global_stack_concept_intercept"]
                - g[key]["static_concept_weights"])
        claim(s, f"5.3 абляция, {name}", pattern, diff.abs().max())
        if key == "deep":
            globals()["_deep_diff"] = diff
    for key, name, lo, hi in (
            ("classical", "простых", r"от (\d+) до \d+ компонентов у простых",
             r"от \d+ до (\d+) компонентов у простых"),
            ("deep", "глубоких", r"и от (\d+) до \d+ у глубоких",
             r"и от \d+ до (\d+) у глубоких")):
        fit = gate[key][gate[key].meta_learner == "static_concept_weights"]
        fit = fit.groupby("dataset").n_concepts_fit
        claim(s, f"5.3 компонентов, {name}, минимум", lo, int(fit.min().min()))
        claim(s, f"5.3 компонентов, {name}, максимум", hi, int(fit.max().max()))
    diff = (g["deep"]["global_stack_concept_intercept"]
            - g["deep"]["static_concept_weights"])

    # взвешивание против надстройки — на одних и тех же строках
    for key, name, pat_med, pat_lo, pat_hi in (
            ("classical", "простые",
             r"у надстройки медианно на \+(\d+\.\d+), у глубоких", None, None),
            ("deep", "глубокие",
             r"у глубоких — на \+(\d+\.\d+)", None, None)):
        g = gate[key]
        g = g[g.meta_learner == "static_concept_weights"].groupby(
            "dataset").lift_gated_vs_stack_auc.mean()
        claim(s, f"5.3 против надстройки, {name}", pat_med, g.median())
    gd = gate["deep"]
    gd = gd[gd.meta_learner == "static_concept_weights"].groupby(
        "dataset").lift_gated_vs_stack_auc.mean()
    claim(s, "5.3 против надстройки, худший набор",
          r"проигрыш (−\d+\.\d+) значим", gd.min())

    nm = a["nmin"]
    for key, name, pats in (
            ("classical", "простые",
             (r"полное взвешивание\s*при этом сдвигается с \+(\d+\.\d+) на \+\d+\.\d+",
              r"сдвигается с \+\d+\.\d+ на \+(\d+\.\d+), вариант",
              r"вариант со свободным членом стоит на\s*\+(\d+\.\d+)")),
            ("deep", "глубокие",
             (r"у глубоких — с \+(\d+\.\d+) на \+\d+\.\d+",
              r"у глубоких — с \+\d+\.\d+ на \+(\d+\.\d+) против",
              r"против неподвижных \+(\d+\.\d+)"))):
        x = nm[nm.subset == key]
        t = x.groupby(["n_min", "dataset"]).agg(
            full=("lift_full_vs_best", "mean"),
            inter=("lift_intercept_vs_best", "mean"),
            gap=("gap_full_minus_intercept", "mean"))
        by = t.groupby("n_min")
        claim(s, f"5.3 порог, {name}, полное минимум", pats[0], by.full.mean().min())
        claim(s, f"5.3 порог, {name}, полное максимум", pats[1], by.full.mean().max())
        claim(s, f"5.3 порог, {name}, свободный член", pats[2], by.inter.mean().max())

    s = norm(section(text, "5.4 Внимание и разреженная смесь"))
    at = attn_moe[attn_moe.scheme == "attention"]
    mo = attn_moe[attn_moe.scheme == "moe"]

    def agg(d, base):
        return d.groupby("dataset")[f"delta_auc_vs_{base}"].mean()

    def sig5(d, base):
        return int((d.groupby("dataset")[f"auc_p_vs_{base}_reject_holm"].sum() == 5).sum())

    claim(s, "5.4 внимание/лучшая минимум", r"модели — от \+(\d+\.\d+) до",
          agg(at, "best").min())
    claim(s, "5.4 внимание/лучшая максимум", r"модели — от \+\d+\.\d+ до \+(\d+\.\d+)",
          agg(at, "best").max())
    word_claim(s, "5.4 внимание/лучшая значимость",
               r"разбиениях на (\S+) наборах, — но не выигрывает", sig5(at, "best"))
    claim(s, "5.4 внимание/надстройка минимум", r"в пределах от (−\d+\.\d+) до \+\d+\.\d+, то есть",
          agg(at, "stack").min())
    claim(s, "5.4 внимание/надстройка максимум", r"от −\d+\.\d+ до \+(\d+\.\d+), то есть",
          agg(at, "stack").max())
    word_claim(s, "5.4 внимание/взвешивание, число наборов",
               r"оно хуже на (\S+) наборах из семи, в пределах",
               int((agg(at, "static") < 0).sum()))
    claim(s, "5.4 внимание/взвешивание минимум", r"в пределах от (−\d+\.\d+) до \+\d+\.\d+, и значимо",
          agg(at, "static").min())
    claim(s, "5.4 внимание/взвешивание максимум", r"от −\d+\.\d+ до \+(\d+\.\d+), и значимо",
          agg(at, "static").max())
    word_claim(s, "5.4 внимание/взвешивание значимость",
               r"разбиениях только на (\S+) наборах", sig5(at, "static"))
    word_claim(s, "5.4 смесь/лучшая, число наборов",
               r"она отрицательна на (\S+) наборах из семи, от",
               int((agg(mo, "best") < 0).sum()))
    claim(s, "5.4 смесь/лучшая минимум", r"из семи, от (−\d+\.\d+) до \+\d+\.\d+\.",
          agg(mo, "best").min())
    claim(s, "5.4 смесь/лучшая максимум", r"из семи, от −\d+\.\d+ до \+(\d+\.\d+)\.",
          agg(mo, "best").max())
    claim(s, "5.4 смесь/взвешивание, меньшая потеря",
          r"на всех семи наборах, от (−\d+\.\d+) до", agg(mo, "static").max())
    claim(s, "5.4 смесь/взвешивание, большая потеря",
          r"на всех семи наборах, от −\d+\.\d+ до (−\d+\.\d+)", agg(mo, "static").min())
    claim(s, "5.4 смесь/надстройка, меньшая потеря",
          r"тоже на всех семи, от (−\d+\.\d+) до", agg(mo, "stack").max())
    claim(s, "5.4 смесь/надстройка, большая потеря",
          r"тоже на всех семи, от −\d+\.\d+ до (−\d+\.\d+)", agg(mo, "stack").min())
    check(int((mo.groupby("dataset").auc_p_vs_static_reject_holm.sum() == 5).sum()) == 7,
          "5.4 смесь значима на всех семи наборах")

    # расхождение двух запусков одной и той же схемы на разном оборудовании
    new_static = attn_moe.groupby(["dataset", "scheme"]).delta_auc_vs_static.mean().unstack()
    d_at = (new_static["attention"]
            - a["attn_mps"].groupby("dataset").lift_attn_vs_static_auc.mean()).abs()
    d_mo = (new_static["moe"]
            - a["moe_mps"].groupby("dataset").lift_moe_vs_static_auc.mean()).abs()
    claim(s, "5.4 расхождение запусков, внимание",
          r"сместилась не более чем на (\d+\.\d+)", d_at.max())
    claim(s, "5.4 расхождение запусков, смесь", r"смеси —\s*до (\d+\.\d+)", d_mo.max())
    seeds = a["seeds"]
    for scheme, ru, pats in (
            ("attention", "внимание",
             (r"У внимания размах по начальным состояниям внутри ячейки медианно "
              r"(\d+\.\d+)", r"медианно \d+\.\d+, наибольший (\d+\.\d+), и в четырёх")),
            ("moe", "смесь",
             (r"медианно (\d+\.\d+) при наибольшем", r"при наибольшем (\d+\.\d+)"))):
        x = seeds[seeds.scheme == scheme]
        per = x.groupby(["dataset", "fold"]).lift_vs_static.agg(["min", "max"])
        span = per["max"] - per["min"]
        claim(s, f"5.4 сиды, {ru}, медиана размаха", pats[0], span.median())
        claim(s, f"5.4 сиды, {ru}, наибольший размах", pats[1], span.max())
        flip = int(((per["min"] < 0) & (per["max"] > 0)).sum())
        if scheme == "attention":
            word_claim(s, "5.4 сиды, внимание, смена знака",
                       r"и в (\S+) ячейках из тридцати пяти знак разности", flip)
    s7 = norm(section(text, "7. Ограничения"))
    claim(s7, "7 расхождение запусков", r"разошлись до (\d+\.\d+)", d_mo.max())
    seeds = a["seeds"]
    per = seeds.groupby(["dataset", "fold", "scheme"]).lift_vs_static.agg(["min", "max"])
    claim(s7, "7 размах по начальным состояниям",
          r"размах до (\d+\.\d+)", (per["max"] - per["min"]).max())
    al = a["align"]
    ok = al[al.status == "ok"]
    word_claim(s7, "7 выравнивание, число наборов",
               r"удаётся на (\S+) наборах из семи", len(ok))
    check(int((ok.share_of_deep >= 0.99).sum()) == 3, "7 три набора не ниже 0.99",
          f"в артефакте {int((ok.share_of_deep >= 0.99).sum())}")
    for ds, pattern in (("assist2009", r"ASSISTments-2009 равна (\d+\.\d+)"),
                        ("algebra2005", r"Algebra-2005 — (\d+\.\d+)"),
                        ("ednet", r"EdNet-KT1-5k — (\d+\.\d+)")):
        claim(s7, f"7 доля общих строк, {ds}", pattern, float(al.loc[ds, "share_of_deep"]))

    s = norm(section(text, "5.5 Трёхступенчатый ансамбль"))
    st = ccce.groupby("dataset")[[f"lift_{v}_vs_static_auc" for v in VARIANTS]].mean()
    st.columns = VARIANTS
    bs = ccce.groupby("dataset")[[f"lift_{v}_vs_best_auc" for v in VARIANTS]].mean()
    bs.columns = VARIANTS
    claim(s, "5.5 полная максимум", r"рис\. 3\): от (−\d+\.\d+) до", st.full.max())
    claim(s, "5.5 полная минимум", r"рис\. 3\): от −\d+\.\d+ до (−\d+\.\d+)", st.full.min())
    claim(s, "5.5 без первой максимум", r"лежит от (−\d+\.\d+) до −\d+\.\d+, то есть потери",
          st.no_s1.max())
    claim(s, "5.5 без первой минимум", r"лежит от −\d+\.\d+ до (−\d+\.\d+), то есть потери",
          st.no_s1.min())
    claim(s, "5.5 общая первая минимум", r"разность лежит от (−\d+\.\d+) до", st.global_s1.min())
    claim(s, "5.5 общая первая максимум", r"разность лежит от −\d+\.\d+ до \+(\d+\.\d+)",
          st.global_s1.max())
    claim(s, "5.5 без третьей максимум", r"почти целиком: от (−\d+\.\d+) до", st.no_s3.max())
    claim(s, "5.5 без третьей минимум", r"почти целиком: от −\d+\.\d+ до (−\d+\.\d+)", st.no_s3.min())
    claim(s, "5.5 против лучшей минимум", r"семи, от (−\d+\.\d+) до \+\d+\.\d+, но выигрывает",
          bs.full.min())
    claim(s, "5.5 против лучшей максимум", r"от −\d+\.\d+ до \+(\d+\.\d+), но выигрывает",
          bs.full.max())
    claim(s, "5.5 без первой против лучшей минимум", r"без первой ступени: от (−\d+\.\d+) до",
          bs.no_s1.min())
    claim(s, "5.5 без первой против лучшей максимум",
          r"без первой ступени: от −\d+\.\d+ до \+(\d+\.\d+)", bs.no_s1.max())

    # кросс-фит первой ступени
    s55 = norm(section(text, "5.5 Трёхступенчатый ансамбль"))
    cf = a["ccce_cf"].groupby("dataset")[
        [f"lift_{v}_vs_static_auc" for v in VARIANTS]].mean()
    cf.columns = VARIANTS
    st_full = ccce.groupby("dataset").lift_full_vs_static_auc.mean()
    claim(s55, "5.5 кросс-фит, меньшая потеря",
          r"на диапазон от (−\d+\.\d+) до −\d+\.\d+", cf.full.max())
    claim(s55, "5.5 кросс-фит, большая потеря",
          r"на диапазон от −\d+\.\d+ до (−\d+\.\d+)", cf.full.min())
    claim(s55, "5.5 кросс-фит, наибольший рост",
          r"наибольший рост потери — (\d+\.\d+)", -(cf.full - st_full).min())
    claim(s55, "5.5 кросс-фит, без первой ступени, минимум",
          r"от (−\d+\.\d+) до −\d+\.\d+ и от", cf.no_s1.min())
    claim(s55, "5.5 кросс-фит, без первой ступени, максимум",
          r"от −\d+\.\d+ до (−\d+\.\d+) и от", cf.no_s1.max())
    claim(s55, "5.5 кросс-фит, общая ступень, минимум",
          r"и от (−\d+\.\d+) до \+\d+\.\d+", cf.global_s1.min())
    claim(s55, "5.5 кросс-фит, общая ступень, максимум",
          r"и от −\d+\.\d+ до \+(\d+\.\d+)", cf.global_s1.max())

    # уровень подробности
    gr = a["gran"]
    s23 = norm(section(text, "2.3 Порядок объединения"))
    for pat, val in (
            (r"становится в 1\.18, 1\.45 и (\d+\.\d+) раза больше",
             gr.loc["ednet", "expansion"]),
            (r"она поднимается с (\d+\.\d+) до \d+\.\d+",
             gr.loc["ednet", "auc_best_question"]),
            (r"поднимается с \d+\.\d+ до (\d+\.\d+)",
             gr.loc["ednet", "auc_best_concept"])):
        claim(s23, f"2.3 уровень подробности {pat[:24]}", pat, float(val))

    s = norm(section(text, "5.6 Калибровка"))
    ge = gate["deep"][gate["deep"].meta_learner == "static_concept_weights"]
    ge = ge.groupby("dataset").lift_gated_vs_best_ece.mean()
    ce = ccce.groupby("dataset")[[f"lift_{v}_vs_static_ece" for v in VARIANTS]].mean()
    ce.columns = VARIANTS
    se = stack.groupby(["dataset", "subset", "meta_learner"]).delta_ece.mean()
    word_claim(s, "5.6 улучшено ячеек", r"уменьшилась в (\S+ \S+) ячейках из", 59)
    word_claim(s, "5.6 всего ячеек", r"ячейках из (\S+ \S+) и не выросла", 65)
    check(int((se < 0).sum()) == 59, "5.6 число улучшенных ячеек",
          f"в артефакте {int((se < 0).sum())}")
    check(int((se > 0).sum()) == 0, "5.6 ухудшенных ячеек нет",
          f"в артефакте {int((se > 0).sum())}")
    claim(s, "5.6 взвешивание максимум", r"наборах, от (−\d+\.\d+) до −\d+\.\d+", ge.max())
    claim(s, "5.6 взвешивание минимум", r"наборах, от −\d+\.\d+ до (−\d+\.\d+)", ge.min())
    claim(s, "5.6 полная минимум", r"наборах: от \+(\d+\.\d+) до", ce.full.min())
    claim(s, "5.6 полная максимум", r"наборах: от \+\d+\.\d+ до \+(\d+\.\d+)", ce.full.max())
    both = pd.concat([ce.no_s1, ce.global_s1])
    claim(s, "5.6 варианты минимум", r"в пределах от (−\d+\.\d+) до \+\d+\.\d+, то есть от простого",
          both.min())
    claim(s, "5.6 варианты максимум", r"от −\d+\.\d+ до \+(\d+\.\d+), то есть от простого",
          both.max())

    sp = a["spread"].groupby(["dataset", "subset"])[["best_std", "ensemble_std"]].mean()
    claim(s, "5.6 разброс ASSISTments-2017 до", r"с (\d+\.\d+) до \d+\.\d+ на ASSISTments-2017",
          sp.loc[("assist2017", "deep"), "best_std"])
    claim(s, "5.6 разброс ASSISTments-2017 после",
          r"с \d+\.\d+ до (\d+\.\d+) на ASSISTments-2017",
          sp.loc[("assist2017", "deep"), "ensemble_std"])
    claim(s, "5.6 разброс EdNet до", r"с (\d+\.\d+) до \d+\.\d+ на EdNet",
          sp.loc[("ednet", "deep"), "best_std"])
    claim(s, "5.6 разброс EdNet после", r"с \d+\.\d+ до (\d+\.\d+) на EdNet",
          sp.loc[("ednet", "deep"), "ensemble_std"])
    deep = sp.xs("deep", level="subset")
    classical = sp.xs("classical", level="subset")
    word_claim(s, "5.6 сжатие у глубоких",
               r"уменьшается после усреднения на всех (\S+) наборах",
               int((deep.ensemble_std < deep.best_std).sum()))
    word_claim(s, "5.6 сжатие у простых",
               r"уменьшается на (\S+) наборах из\s*семи",
               int((classical.ensemble_std < classical.best_std).sum()))
    claim(s, "5.6 подпись рис. 1 / наличие", r"Рис\. (1)\. Диаграммы надёжности", 1)
    check("Fig. 1. Reliability diagrams" in s, "5.6 подпись рис. 1 на английском")

    s5_3 = norm(section(text, "5.3 Условное взвешивание и вклад свободного члена"))
    claim(s5_3, "5.3 подпись рис. 2 / расхождение",
          r"Расхождение точек не превышает (\d+\.\d+)", diff.abs().max())
    check("Fig. 2. Full weighting" in s5_3, "5.3 подпись рис. 2 на английском")
    s5_5 = norm(section(text, "5.5 Трёхступенчатый ансамбль"))
    check("Рис. 3. Четыре варианта" in s5_5, "5.5 подпись рис. 3")
    check("Fig. 3. Four variants" in s5_5, "5.5 подпись рис. 3 на английском")

    # раздел 2
    s = norm(section(text, "2.1 Наборы данных"))
    size = simple.groupby(["dataset", "subset"]).n_rows.max().unstack()
    ratio = (size.classical / size.deep - 1) * 100
    claim(s, "2.1 разница строк, минимум", r"на (\d+\.\d+)–\d+\.\d+ % больше", ratio.min())
    claim(s, "2.1 разница строк, максимум", r"на \d+\.\d+–(\d+\.\d+) % больше", ratio.max())

    s = norm(section(text, "2.3 Порядок объединения"))
    claim(s, "2.3 многокомпонентные, Algebra-2005",
          r"Algebra-2005 таких заданий (\d+\.\d+) %", a["multi"]["algebra2005"] * 100)
    claim(s, "2.3 многокомпонентные, EdNet",
          r"EdNet-KT1-5k — (\d+\.\d+) %", a["multi"]["ednet"] * 100)
    b = gate["deep"]
    b = b[(b.dataset == "bridge2algebra2006")
          & (b.meta_learner == "static_concept_weights")]
    claim(s, "2.3 компонентов минимум", r"получают от (\d+)\s*до", b.n_concepts_fit.min())
    claim(s, "2.3 компонентов максимум", r"от \d+\s*до (\d+) компонентов", b.n_concepts_fit.max())
    claim(s, "2.3 компонентов всего", r"компонентов из (\d+)", a["cfg"]["bridge2algebra2006"]["num_c"])

    # разделы 6-8
    s = norm(section(text, "6.4 Что из этого следует для практики"))
    claim(s, "6.4 потеря на EdNet", r"теряет (\d+\.\d+) площади", -d.deep["ednet"])
    s = norm(section(text, "7. Ограничения"))
    claim(s, "7 максимальный прирост", r"не превышают (\d+\.\d+) площади", learned.max().max())
    s = norm(section(text, "8. Заключение"))
    claim(s, "8 потеря на EdNet", r"теряет (\d+\.\d+)\s*площади", -d.deep["ednet"])
    claim(s, "8 отрыв на EdNet", r"оторвалась от\s*второй на (\d+\.\d+)", ed.iloc[0] - ed.iloc[1])


# --------------------------------------------------------------- аннотации
def check_abstracts(text: str, a: dict) -> None:
    d = a["simple"][a["simple"].aggregator == "arithmetic_mean"]
    d = d.groupby(["dataset", "subset"]).delta_auc.mean().unstack()
    sd = a["stack"].groupby(["dataset", "subset", "meta_learner"]).delta_auc.mean().unstack()
    learned = sd.drop(columns=["bma"])
    st = a["ccce"].groupby("dataset")[[f"lift_{v}_vs_static_auc" for v in VARIANTS]].mean()
    st.columns = VARIANTS

    ru = norm(text.split("**Аннотация.**")[1].split("**Ключевые слова:**")[0])
    en = norm(text.split("**Abstract.**")[1].split("**Keywords:**")[0])
    for name, s, pat in (("русская", ru, r"на (\d+\.\d+) площади под кривой"),
                         ("английская", en, r"by (\d+\.\d+) of the area")):
        claim(s, f"аннотация {name} / потеря усреднения", pat, -d.deep["ednet"])
    claim(ru, "аннотация рус / надстройка минимум", r"ячейках, от (\d+\.\d+) до", learned.min().min())
    claim(ru, "аннотация рус / надстройка максимум", r"ячейках, от \d+\.\d+ до (\d+\.\d+)",
          learned.max().max())
    claim(en, "аннотация англ / надстройка минимум", r"cell, by (\d+\.\d+) to", learned.min().min())
    claim(en, "аннотация англ / надстройка максимум", r"cell, by \d+\.\d+ to (\d+\.\d+)",
          learned.max().max())
    claim(ru, "аннотация рус / схема минимум", r"наборах, от (\d+\.\d+) до \d+\.\d+ площади",
          -st.full.max())
    claim(ru, "аннотация рус / схема максимум", r"наборах, от \d+\.\d+ до (\d+\.\d+) площади",
          -st.full.min())
    claim(en, "аннотация англ / схема минимум", r"datasets, by (\d+\.\d+) to", -st.full.max())
    claim(en, "аннотация англ / схема максимум", r"datasets, by \d+\.\d+ to (\d+\.\d+)",
          -st.full.min())

    n_ru, n_en = len(ru.split()), len(en.split())
    # журнал требует 200-300 слов от обеих аннотаций
    check(200 <= n_ru <= 300, "аннотация рус / длина", f"{n_ru} слов, нужно 200-300")
    check(200 <= n_en <= 300, "аннотация англ / длина", f"{n_en} слов, нужно 200-300")

    kw_ru = [x for x in text.split("**Ключевые слова:**")[1].split("**Для цитирования")[0]
             .split(";") if x.strip()]
    kw_en = [x for x in text.split("**Keywords:**")[1].split("---")[0].split(";") if x.strip()]
    check(len(kw_ru) == len(kw_en), "ключевые слова / одинаковое число",
          f"рус {len(kw_ru)}, англ {len(kw_en)}")
    check(len(kw_ru) <= 10, "ключевые слова / не больше десяти", f"{len(kw_ru)}")


# ------------------------------------------------------------ библиография
def check_citations(text: str) -> None:
    """Нумерация по первому упоминанию, без пропусков и без сирот."""
    body, _, bib = text.partition("## Список литературы / References")
    check(bool(bib), "литература / раздел есть")
    if not bib:
        return
    listed = [int(m.group(1)) for m in re.finditer(r"^\[(\d+)\]\.", bib, re.M)]
    check(listed == list(range(1, len(listed) + 1)), "литература / нумерация без пропусков",
          f"в списке {listed}")

    mentioned: list[int] = []
    for m in re.finditer(r"\[(\d+(?:\s*[-,]\s*\d+)*)\]", body):
        for part in re.split(r"[,]", m.group(1)):
            part = part.strip()
            if "-" in part:
                lo, hi = (int(x) for x in part.split("-"))
                nums = range(lo, hi + 1)
            else:
                nums = [int(part)]
            for n in nums:
                if n not in mentioned:
                    mentioned.append(n)
    check(mentioned == sorted(mentioned), "литература / порядок первого упоминания",
          f"порядок появления {mentioned}")
    check(set(mentioned) == set(listed), "литература / нет сирот и пропущенных",
          f"в тексте {sorted(set(mentioned))}, в списке {listed}")
    check(len(listed) >= 18, "литература / не меньше восемнадцати источников",
          f"{len(listed)}")
    own = len([1 for line in bib.split("\n") if "репозитори" in line.lower()])
    check(own / max(len(listed), 1) <= 0.3, "литература / самоцитирование не больше 30 %",
          f"своих {own} из {len(listed)}")


# ---------------------------------------------------------------- гигиена
FORBIDDEN = [
    r"\bExp [A-Z]\b", r"\bT[0-9][a-z]?\b", r"аудит[а]? 23", r"audit 23",
    r"rq3_draft", r"\.csv\b", r"\bv[24]\b", r"CCCE", r"gating_deep",
    r"\bШ\d+\b", r"pyKT `", r"артефакт",
]


def check_hygiene(text: str) -> None:
    for pat in FORBIDDEN:
        m = re.search(pat, text)
        check(m is None, f"гигиена / {pat}",
              f"найдено {m.group(0)!r}" if m else "")
    sys.path.insert(0, str(paths.RESEARCH_DIR / "scripts"))
    from scripts.c5_units import percents
    bad = [m.group(0) for _, m, is_auc, _ in percents(text.split("\n")) if is_auc]
    check(not bad, "гигиена / проценты AUC", f"найдено {bad}")


def main() -> int:
    # путь можно передать аргументом: так проверяется намеренно испорченная копия
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    text = manuscript(path)
    a = load()
    check_table1(text, a)
    check_components(text, a, "**Табл. 2.**", "classical",
                     ["bkt", "pfa", "pfa_recency", "elorasch"])
    check_components(text, a, "**Табл. 3.**", "deep",
                     ["dkt", "sakt", "akt", "simplekt"])
    check_table4(text, a)
    check_table5(text, a)
    check_table6(text, a)
    check_table7(text, a)
    check_table8(text, a)
    check_table9(text, a)
    check_table10(text, a)
    check_crossfit(text, a)
    check_prose(text, a)
    check_abstracts(text, a)
    check_citations(text)
    check_hygiene(text)

    bad = [r for r in results if not r[0]]
    for ok, name, detail in results:
        if not ok:
            print(f"ПЛОХО  {name}: {detail}")
    print(f"\nпроверок {len(results)}, не прошло {len(bad)}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
