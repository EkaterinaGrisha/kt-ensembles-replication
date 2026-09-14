"""Статья С5: три рисунка, собранные как SVG и отрисованные headless-браузером.

Тот же конвейер, что у соседних статей серии, и по тем же причинам: matplotlib
плохо справляется с двуязычными подписями и точной печатной геометрией, а
рукописный SVG плюс Chrome дают PNG в 300 dpi и векторный PDF — ровно то, что
просит журнал. Вспомогательные функции продублированы, а не импортированы: каждая
статья уезжает в собственный репозиторий и не должна зависеть от чужих сценариев.

Рисунок 1 — диаграммы надёжности. Три набора, семейство глубоких моделей, первое
разбиение: лучшая одиночная модель и усреднение без настройки. Показывает, что
именно происходит с вероятностями при усреднении: точки собираются к середине
шкалы. Наборы выбраны так, чтобы был виден и проигрыш, и выигрыш.

Рисунок 2 — вклад свободного члена, семейство глубоких моделей. По одному ряду на
набор, две точки: полное взвешивание по компонентам знания и вариант, в котором
свой у компонента только свободный член. Весь смысл рисунка в том, что точки
совпадают; на простых моделях то же самое видно из таблицы 6.

Рисунок 3 — четыре варианта трёхступенчатой схемы относительно взвешивания.
Нуль — взвешивание. Полная схема и схема без третьей ступени уходят влево,
варианты без первой ступени и с общей первой ступенью стоят на нуле.

Каждое число читается из артефактов; ни одно не вписано в этот файл. Точки
диаграмм надёжности считаются из тех же файлов предсказаний, что и ансамбли, и
сохраняются в `artifacts/ensembles/reliability_points.csv`.

Геометрия: рисунок шириной 512 единиц печатается как 13 см при 300 dpi (журнал
разрешает 14).

Выход (research/paper/figures/):
  c5_fig1_reliability.{svg,png,pdf}
  c5_fig2_intercept.{svg,png,pdf}
  c5_fig3_ccce.{svg,png,pdf}

Запуск:
  python -m scripts.make_c5_figures_html
  python -m scripts.make_c5_figures_html --no-render
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from ktx import paths

ENS = paths.ARTIFACTS_DIR / "ensembles"
PRED = paths.ARTIFACTS_DIR / "predictions"
FIG_DIR = paths.RESEARCH_DIR / "paper" / "figures"
CHROME = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")

W = 512
SCALE = 3
MAX_CM = 14.0

INK = "#1B2129"
MUTED = "#6E7883"
RULE = "#C9D0D8"
SIMPLE = "#215D9E"      # простое решение
SIMPLE_2 = "#6E9BC7"    # его второй вариант
ELABORATE = "#A8403A"   # усложнение
ELABORATE_2 = "#CC8A85"  # его второй вариант
PANEL = "#F4F6F8"

FONT = ("-apple-system, 'Helvetica Neue', Helvetica, Arial, "
        "'Liberation Sans', sans-serif")

DEEP = ["dkt", "sakt", "akt", "simplekt"]
DISPLAY = {
    "assist2009": "ASSISTments-2009", "assist2012": "ASSISTments-2012",
    "assist2015": "ASSISTments-2015", "assist2017": "ASSISTments-2017",
    "algebra2005": "Algebra-2005", "bridge2algebra2006": "Bridge-to-Algebra-2006",
    "ednet": "EdNet-KT1-5k",
}
ORDER = ["algebra2005", "assist2009", "assist2012", "assist2015",
         "assist2017", "bridge2algebra2006", "ednet"]
RELIABILITY_SETS = ["assist2017", "bridge2algebra2006", "ednet"]
RELIABILITY_FOLD = 0
BINS = 15


# ----------------------------------------------------------------- svg helpers
def esc(s: str) -> str:
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def text(x, y, s, size=11, fill=INK, weight="400", anchor="start") -> str:
    return (f'<text x="{x:.2f}" y="{y:.2f}" font-family="{FONT}" font-size="{size}" '
            f'fill="{fill}" font-weight="{weight}" text-anchor="{anchor}">'
            f'{esc(s)}</text>')


def line(x1, y1, x2, y2, stroke=RULE, sw=1, dash=None) -> str:
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
            f'stroke="{stroke}" stroke-width="{sw}"{d}/>')


def rect(x, y, w, h, fill="none", stroke="none", sw=1, rx=0) -> str:
    return (f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}" rx="{rx}"/>')


def dot(cx, cy, r, fill) -> str:
    return (f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{r}" fill="{fill}" '
            f'stroke="#FFFFFF" stroke-width="1.2"/>')


def polyline(points, stroke, sw=1.6) -> str:
    d = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    return (f'<polyline points="{d}" fill="none" stroke="{stroke}" '
            f'stroke-width="{sw}" stroke-linejoin="round"/>')


def bilingual(x, y, ru, en, size=10, weight="400", anchor="start", gap=11) -> str:
    return (text(x, y, ru, size=size, weight=weight, anchor=anchor)
            + text(x, y + gap, en, size=size - 1, fill=MUTED, anchor=anchor))


def nice_step(span: float) -> float:
    """Круглый шаг шкалы: подписи вида −0.006, а не −0.00696."""
    import math
    raw = span / 4
    mag = 10 ** math.floor(math.log10(raw))
    for m in (1, 2, 2.5, 5, 10):
        if raw <= m * mag:
            return m * mag
    return 10 * mag


def axis_ticks(x, lo: float, hi: float, y: float, nd: int = 4) -> str:
    """Подписанная шкала под рядами: журналу нужна цена деления, а не только нуль."""
    import math
    step = nice_step(hi - lo)
    first = math.ceil(lo / step) * step
    parts = [line(x(lo), y, x(hi), y, stroke=RULE)]
    v = first
    while v <= hi + 1e-12:
        parts.append(line(x(v), y, x(v), y + 4, stroke=RULE))
        label = "0" if abs(v) < step / 100 else signed(v, nd)
        parts.append(text(x(v), y + 13, label, size=7, fill=MUTED, anchor="middle"))
        v += step
    return "".join(parts)


def signed(v: float, nd: int = 4) -> str:
    return f"{v:+.{nd}f}".replace("-", "−")


# ------------------------------------------------------------------- числа
def equal_mass_bins(y: np.ndarray, p: np.ndarray, n_bins: int = BINS):
    """Точки диаграммы надёжности: группы равного размера, как и в ошибке калибровки."""
    order = np.argsort(p, kind="mergesort")
    y, p = y[order], p[order]
    edges = np.linspace(0, len(p), n_bins + 1).astype(int)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi <= lo:
            continue
        out.append((float(p[lo:hi].mean()), float(y[lo:hi].mean()), int(hi - lo)))
    return out


def reliability_points() -> pd.DataFrame:
    """Считает и сохраняет точки диаграмм: лучшая одиночная модель и усреднение."""
    boot = pd.read_csv(ENS / "cluster_bootstrap_simple_ensembles.csv")
    rows = []
    for ds in RELIABILITY_SETS:
        cell = boot[(boot.dataset == ds) & (boot.subset == "deep")
                    & (boot.aggregator == "arithmetic_mean")
                    & (boot.fold == RELIABILITY_FOLD)]
        if cell.empty:
            sys.exit(f"ФАТАЛЬНО: нет строки бутстрапа для {ds}")
        best_model = cell.best_single_model.iloc[0]
        mats, y = [], None
        for model in DEEP:
            f = PRED / ds / f"{model}_fold{RELIABILITY_FOLD}.npz"
            if not f.exists():
                sys.exit(f"ФАТАЛЬНО: нет предсказаний {f}")
            d = np.load(f)
            y = np.asarray(d["y_true"]).astype(int)
            mats.append(np.asarray(d["y_prob"], dtype=np.float64))
            if model == best_model:
                best = mats[-1]
        ens = np.mean(np.column_stack(mats), axis=1)
        for name, p in (("лучшая одиночная", best), ("усреднение", ens)):
            for mean_p, frac, n in equal_mass_bins(y, p):
                rows.append({"dataset": ds, "curve": name, "mean_p": mean_p,
                             "frac_correct": frac, "n": n})
    out = pd.DataFrame(rows)
    (ENS / "reliability_points.csv").write_text(out.to_csv(index=False), encoding="utf-8")
    return out


# ---------------------------------------------------------------- рисунок 1
def figure1(points: pd.DataFrame, boot: pd.DataFrame, spread: pd.DataFrame) -> tuple[str, int]:
    pad_l, pad_r, pad_t, gap = 30, 8, 110, 16
    panel_w = ((W - pad_l - pad_r - gap * (len(RELIABILITY_SETS) - 1))
               / len(RELIABILITY_SETS))
    panel_h = 150
    height = pad_t + panel_h + 104
    parts = [rect(0, 0, W, height, fill="#FFFFFF")]
    parts.append(bilingual(0, 18, "Рис. 1. Диаграммы надёжности: усреднение стягивает "
                           "вероятности к середине",
                           "Fig. 1. Reliability diagrams: averaging pulls probabilities "
                           "toward the middle", size=11, weight="600", gap=13))
    parts.append(text(0, 58, "Глубокие модели, первое разбиение, пятнадцать групп равного "
                      "размера", size=9, fill=MUTED))
    parts.append(text(0, 69, "Deep models, first split, fifteen equal-mass bins",
                      size=8, fill=MUTED))

    for i, ds in enumerate(RELIABILITY_SETS):
        x0 = pad_l + i * (panel_w + gap)
        y0 = pad_t
        parts.append(text(x0, y0 - 9, DISPLAY[ds], size=9, weight="600"))
        parts.append(rect(x0, y0, panel_w, panel_h, fill=PANEL))
        parts.append(line(x0, y0 + panel_h, x0 + panel_w, y0, stroke=RULE, dash="3 3"))
        for curve, colour in (("лучшая одиночная", MUTED), ("усреднение", ELABORATE)):
            d = points[(points.dataset == ds) & (points.curve == curve)]
            pts = [(x0 + r.mean_p * panel_w, y0 + panel_h - r.frac_correct * panel_h)
                   for r in d.itertuples()]
            parts.append(polyline(pts, colour))
            for x, y in pts:
                parts.append(dot(x, y, 2.0, colour))
        cell = boot[(boot.dataset == ds) & (boot.subset == "deep")
                    & (boot.aggregator == "arithmetic_mean")]
        sp = spread[(spread.dataset == ds) & (spread.subset == "deep")]
        parts.append(text(x0, y0 + panel_h + 15,
                          f"ошибка калибровки {signed(cell.delta_ece.mean())}",
                          size=8, fill=INK))
        parts.append(text(x0, y0 + panel_h + 25,
                          f"calibration error {signed(cell.delta_ece.mean())}",
                          size=7, fill=MUTED))
        parts.append(text(x0, y0 + panel_h + 38,
                          f"разброс {sp.best_std.mean():.4f} → {sp.ensemble_std.mean():.4f}",
                          size=8, fill=INK))
        parts.append(text(x0, y0 + panel_h + 48,
                          f"spread {sp.best_std.mean():.4f} → {sp.ensemble_std.mean():.4f}",
                          size=7, fill=MUTED))
    parts.append(text(pad_l - 5, pad_t + panel_h + 2, "0", size=8, fill=MUTED, anchor="end"))
    parts.append(text(pad_l - 5, pad_t + 6, "1", size=8, fill=MUTED, anchor="end"))
    parts.append(text(0, pad_t - 26, "доля верных ответов / correct rate", size=8, fill=MUTED))
    parts.append(text(0, height - 34, "предсказанная вероятность / predicted probability",
                      size=8, fill=MUTED))
    legend_y = height - 12
    parts.append(dot(4, legend_y - 4, 3, MUTED))
    parts.append(text(12, legend_y, "лучшая одиночная модель / best single model",
                      size=8, fill=INK))
    parts.append(dot(252, legend_y - 4, 3, ELABORATE))
    parts.append(text(260, legend_y, "усреднение без настройки / averaging",
                      size=8, fill=INK))
    return "".join(parts), height


# ---------------------------------------------------------------- рисунок 2
def figure2(gate: pd.DataFrame) -> tuple[str, int]:
    g = gate.groupby(["dataset", "meta_learner"]).lift_gated_vs_best_auc.mean().unstack()
    row_h, pad_t, pad_l, axis_w = 30, 92, 150, 236
    height = pad_t + row_h * len(ORDER) + 76
    lo = min(0.0, float(g.min().min())) - 0.001
    hi = float(g.max().max()) + 0.002

    def x(v: float) -> float:
        return pad_l + (v - lo) / (hi - lo) * axis_w

    parts = [rect(0, 0, W, height, fill="#FFFFFF")]
    parts.append(bilingual(0, 20, "Рис. 2. Весь вклад условного взвешивания даёт свободный "
                           "член",
                           "Fig. 2. The whole gain of conditional weighting comes from the "
                           "intercept", size=11, weight="600"))
    parts.append(text(0, 56, "Глубокие модели; разность площади под кривой относительно "
                      "лучшей одиночной модели", size=9, fill=MUTED))
    parts.append(text(0, 67, "Deep models; difference in the area under the curve against "
                      "the best single model", size=8, fill=MUTED))
    parts.append(line(x(0), pad_t - 10, x(0), pad_t + row_h * len(ORDER) - 6, stroke=RULE))
    parts.append(text(x(0), pad_t - 16, "0", size=8, fill=MUTED, anchor="middle"))
    for i, ds in enumerate(ORDER):
        y = pad_t + i * row_h + 8
        parts.append(text(0, y + 4, DISPLAY[ds], size=9))
        full = float(g.loc[ds, "static_concept_weights"])
        inter = float(g.loc[ds, "global_stack_concept_intercept"])
        parts.append(line(x(min(full, inter)), y, x(max(full, inter)), y, stroke=RULE, sw=3))
        parts.append(dot(x(full), y, 4.5, ELABORATE))
        parts.append(dot(x(inter), y, 4.5, SIMPLE))
        parts.append(text(pad_l + axis_w + 10, y + 4, signed(full), size=8, fill=ELABORATE))
        parts.append(text(pad_l + axis_w + 58, y + 4, signed(inter), size=8, fill=SIMPLE))
    parts.append(axis_ticks(x, lo, hi, pad_t + row_h * len(ORDER) - 2))
    legend_y = height - 26
    parts.append(dot(4, legend_y - 4, 4, ELABORATE))
    parts.append(text(14, legend_y, "полное взвешивание / full weighting", size=8))
    parts.append(dot(4, legend_y + 12, 4, SIMPLE))
    parts.append(text(14, legend_y + 16, "только свободный член / intercept only", size=8))
    return "".join(parts), height


# ---------------------------------------------------------------- рисунок 3
VARIANTS = [("full", "полная схема", "full scheme", ELABORATE),
            ("no_s3", "без третьей ступени", "without stage 3", ELABORATE_2),
            ("no_s1", "без первой ступени", "without stage 1", SIMPLE),
            ("global_s1", "общая первая ступень", "global stage 1", SIMPLE_2)]


def figure3(ccce: pd.DataFrame) -> tuple[str, int]:
    st = ccce.groupby("dataset")[[f"lift_{v}_vs_static_auc" for v, *_ in VARIANTS]].mean()
    st.columns = [v for v, *_ in VARIANTS]
    row_h, pad_t, pad_l, axis_w = 32, 96, 150, 236
    height = pad_t + row_h * len(ORDER) + 80
    lo = float(st.min().min()) - 0.0006
    hi = max(0.0006, float(st.max().max()) + 0.0006)

    def x(v: float) -> float:
        return pad_l + (v - lo) / (hi - lo) * axis_w

    parts = [rect(0, 0, W, height, fill="#FFFFFF")]
    parts.append(bilingual(0, 20, "Рис. 3. Потеря трёхступенчатой схемы держится на первой "
                           "ступени",
                           "Fig. 3. The loss of the three-stage scheme sits in the first "
                           "stage", size=11, weight="600"))
    parts.append(text(0, 56, "Разность площади под кривой относительно взвешивания по "
                      "компонентам знания; нуль — само взвешивание", size=9, fill=MUTED))
    parts.append(text(0, 67, "Difference in the area under the curve against weighting by "
                      "knowledge component; zero is that weighting", size=8, fill=MUTED))
    parts.append(line(x(0), pad_t - 12, x(0), pad_t + row_h * len(ORDER) - 8, stroke=INK))
    parts.append(text(x(0), pad_t - 18, "0", size=8, fill=MUTED, anchor="middle"))
    for i, ds in enumerate(ORDER):
        y = pad_t + i * row_h + 8
        parts.append(text(0, y + 4, DISPLAY[ds], size=9))
        vals = [float(st.loc[ds, v]) for v, *_ in VARIANTS]
        parts.append(line(x(min(vals + [0.0])), y, x(max(vals + [0.0])), y, stroke=RULE, sw=2))
        for (v, _, _, colour), val in zip(VARIANTS, vals):
            parts.append(dot(x(val), y, 4.0, colour))
        parts.append(text(pad_l + axis_w + 10, y + 4, signed(st.loc[ds, "full"]),
                          size=8, fill=ELABORATE))
        parts.append(text(pad_l + axis_w + 58, y + 4, signed(st.loc[ds, "no_s1"], 5),
                          size=8, fill=SIMPLE))
    parts.append(axis_ticks(x, lo, hi, pad_t + row_h * len(ORDER) - 4, nd=5))
    legend_y = height - 30
    for j, (_, ru, en, colour) in enumerate(VARIANTS):
        lx = (j % 2) * 258
        ly = legend_y + (j // 2) * 14
        parts.append(dot(lx + 4, ly - 4, 4, colour))
        parts.append(text(lx + 14, ly, f"{ru} / {en}", size=8))
    return "".join(parts), height


# --------------------------------------------------------------- отрисовка
def page(svg_body: str, height: int) -> str:
    return (f'<!doctype html><html><head><meta charset="utf-8">'
            f'<style>html,body{{margin:0;padding:0;background:#fff}}</style></head>'
            f'<body><svg xmlns="http://www.w3.org/2000/svg" width="{W}" '
            f'height="{height}" viewBox="0 0 {W} {height}">{svg_body}</svg></body></html>')


def render(html: str, stem: Path, height: int, do_render: bool) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    svg = html.split("<body>", 1)[1].rsplit("</body>", 1)[0]
    stem.with_suffix(".svg").write_text(
        f'<?xml version="1.0" encoding="UTF-8"?>\n{svg}', encoding="utf-8")
    if not do_render:
        return
    if not CHROME.exists():
        print(f"  Chrome не найден по пути {CHROME} — PNG и PDF не собраны")
        return
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "f.html"
        src.write_text(html, encoding="utf-8")
        common = [str(CHROME), "--headless", "--disable-gpu", "--no-sandbox",
                  "--hide-scrollbars", "--virtual-time-budget=2000"]
        subprocess.run(common + [f"--screenshot={stem.with_suffix('.png')}",
                                 f"--window-size={W},{height}",
                                 f"--force-device-scale-factor={SCALE}", str(src)],
                       check=True, capture_output=True)
        subprocess.run(common + [f"--print-to-pdf={stem.with_suffix('.pdf')}",
                                 "--no-pdf-header-footer", str(src)],
                       check=True, capture_output=True)


def report(name: str, height: int) -> None:
    cm = W * SCALE / 300 * 2.54
    if cm > MAX_CM:
        sys.exit(f"ФАТАЛЬНО: {name} шире {MAX_CM} см ({cm:.1f})")
    print(f"  {name}: {W}x{height} единиц -> {W * SCALE}x{height * SCALE} пикселей, "
          f"{cm:.1f} см при 300 dpi (предел {MAX_CM})")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-render", action="store_true")
    args = ap.parse_args()

    boot = pd.read_csv(ENS / "cluster_bootstrap_simple_ensembles.csv")
    spread = pd.read_csv(ENS / "prediction_spread.csv")
    gate = pd.read_csv(ENS / "gating_deep.csv")
    ccce = pd.read_csv(ENS / "ccce_deep.csv")
    points = reliability_points()

    for name, (body, height) in (
            ("c5_fig1_reliability", figure1(points, boot, spread)),
            ("c5_fig2_intercept", figure2(gate)),
            ("c5_fig3_ccce", figure3(ccce))):
        render(page(body, height), FIG_DIR / name, height, not args.no_render)
        report(name, height)
    return 0


if __name__ == "__main__":
    sys.exit(main())
