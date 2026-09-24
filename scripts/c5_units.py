"""Paper C5, step S2: AUC values are written as fractions, never as percents.

The draft this paper is assembled from reports AUC differences in percents of AUC
(-0.64 %), while the sibling papers report fractions (-0.0064). The artifacts
themselves store fractions -- `delta_auc` in `simple_ensembles.csv` is 0.007125, not
0.71 -- so the fraction is also the form that needs no arithmetic on the way from the
artifact into the text. This script keeps that convention enforceable.

Two modes:

  --report FILE   list every percent literal, converted, with the line it sits on.
                  Used once to carry the draft's numbers over into the manuscript.
  --check FILE    exit non-zero if a percent literal that denotes AUC survives in the
                  text. Called from the manuscript checker (S14).

Percents that do not denote AUC stay percents: a share of held-out rows, a threshold
on correct answers, a reduction in calibration error. Those are recognised by the words
around them (NON_AUC below); anything else counts as an AUC percent, so a new kind of
percent fails loudly instead of slipping through.

Conversion keeps every significant digit: a percent quoted to d decimals becomes a
fraction quoted to d + 2, so -0.007 % becomes -0.00007 and nothing is rounded away.

Usage:
  python -m scripts.c5_units --check путь/к/рукописи.md
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# A percent literal with an optional sign: "-0.64 %", "+2.28%", "58%".
PERCENT = re.compile(r"(?P<sign>[+\-−])?(?P<value>\d+(?:\.\d+)?)\s?%")

# Words that mark a percent as something other than an AUC difference. Matched against
# the 80 characters on either side of the literal.
NON_AUC = re.compile(
    r"ECE|ошибк\w* калибровки|held-out|отложенн\w+|correct|верных ответов|"
    r"τ\s*=|throws away|выбрасывает|каждого|строк|заданий",
    re.IGNORECASE,
)

# How far from a literal the marker has to sit. The window runs over the paragraph joined
# back into one string: the manuscript is hard-wrapped, so the word that explains a percent
# lands on a neighbouring line as often as not. Whole-paragraph context would be too loose
# — a paragraph that discusses calibration error also quotes AUC values.
WINDOW = 80


def fraction(sign: str | None, value: str) -> str:
    """-0.64 -> -0.0064, keeping two more decimals than the percent had."""
    decimals = len(value.split(".")[1]) if "." in value else 0
    minus = sign in {"-", "−"}
    return f"{'−' if minus else '+'}{float(value) / 100:.{decimals + 2}f}"


def percents(lines: list[str], first: int = 1):
    """(line number, match, is_auc, line) for every percent literal in the lines.

    A literal counts as AUC unless its paragraph carries one of the NON_AUC markers.
    """
    paragraph: list[str] = []
    starts_at = 0

    def flush():
        joined = " ".join(paragraph)
        at = 0
        for offset, line in enumerate(paragraph):
            for m in PERCENT.finditer(line):
                start = at + m.start()
                context = joined[max(0, start - WINDOW): start + len(m.group(0)) + WINDOW]
                yield first + starts_at + offset, m, NON_AUC.search(context) is None, line
            at += len(line) + 1  # the space the join put back

    for i, line in enumerate(lines):
        if not line.strip():
            yield from flush()
            paragraph, starts_at = [], i + 1
        else:
            if not paragraph:
                starts_at = i
            paragraph.append(line)
    yield from flush()


def scan(path: Path) -> list[tuple[int, str, str, str, bool]]:
    """(line number, literal, fraction, line, is_auc) for every percent in the file."""
    lines = path.read_text(encoding="utf-8").split("\n")
    return [(lineno, m.group(0), fraction(m.group("sign"), m.group("value")),
             line.strip(), is_auc)
            for lineno, m, is_auc, line in percents(lines)]


def report(path: Path) -> int:
    rows = scan(path)
    auc = [r for r in rows if r[4]]
    other = [r for r in rows if not r[4]]
    print(f"{path}: {len(rows)} процентных величин, из них AUC — {len(auc)}\n")
    print("| строка | в черновике | в долях |")
    print("|---|---|---|")
    for lineno, literal, frac, _line, _ in auc:
        print(f"| {lineno} | {literal} | {frac} |")
    print(f"\nНе AUC и остаются процентами ({len(other)}):")
    for lineno, literal, _frac, line, _ in other:
        print(f"  {lineno}: {literal} — {line[:110]}")
    return 0


def check(path: Path) -> int:
    text = path.read_text(encoding="utf-8")
    # The working plan above the manuscript is not checked. The heading is matched on its
    # own line and at its last occurrence: the plan mentions it in passing near the top.
    marker = list(re.finditer(r"^# Текст статьи$", text, re.MULTILINE))
    if marker:
        cut = marker[-1].end()
        offset, body = text[:cut].count("\n") + 1, text[cut:]
    else:
        offset, body = 0, text
    bad = [(lineno, m.group(0), line.strip())
           for lineno, m, is_auc, line in percents(body.split("\n"), offset + 1)
           if is_auc]
    if not bad:
        print(f"{path}: величин AUC в процентах нет")
        return 0
    print(f"{path}: {len(bad)} величин AUC записаны в процентах:")
    for lineno, literal, line in bad:
        print(f"  {lineno}: {literal} — {line[:110]}")
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--check", type=Path)
    args = parser.parse_args()
    if args.report:
        return report(args.report)
    if args.check:
        return check(args.check)
    parser.error("нужен --report или --check")


if __name__ == "__main__":
    sys.exit(main())
