"""Unit tests for the evaluation metric suite."""
from __future__ import annotations

import importlib

import numpy as np
import pytest

from ktx.metrics import compute_metrics, reliability_bins


def _usable(module: str) -> bool:
    """Доступна ли необязательная зависимость.

    Проверяется ввозом, а не поиском файла: `find_spec` отвечает только на
    вопрос «лежит ли пакет», и сломанная установка его проходит, а потом падает
    посреди проверки. Для отсутствующего пакета ввоз обрывается сразу, поэтому
    лишнего времени это не стоит.
    """
    try:
        importlib.import_module(module)
    except Exception:
        return False
    return True


needs_netcal = pytest.mark.skipif(
    not _usable("netcal"),
    reason="ошибка калибровки в compute_metrics считается через netcal; "
           "поставьте ktx[calibration]")


@needs_netcal
def test_perfect_predictions():
    y = np.array([0, 0, 1, 1, 1, 0, 1, 0])
    p = y.astype(float) * 0.999 + (1 - y) * 0.001
    m = compute_metrics(y, p)
    assert m.auc > 0.999
    assert m.accuracy == 1.0
    assert m.brier < 1e-3
    assert m.ece < 1e-2
    assert m.n == len(y)


def test_reliability_bins_partition_all_points():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=500)
    p = rng.random(500)
    bins = reliability_bins(y, p, n_bins=10)
    assert len(bins["count"]) == 10
    assert sum(bins["count"]) == 500


@needs_netcal
def test_calibration_detects_overconfidence():
    # systematically overconfident predictions → ECE should be clearly > 0
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, size=2000)
    # predict 0.95 regardless of truth → poorly calibrated
    p = np.full(2000, 0.95)
    m = compute_metrics(y, p)
    assert m.ece > 0.2


def test_scalars_exclude_reliability_arrays():
    y = np.array([0, 1, 0, 1])
    p = np.array([0.2, 0.8, 0.4, 0.6])
    m = compute_metrics(y, p)
    s = m.scalars()
    assert "reliability" not in s
    assert "ece" in s and "auc" in s and "brier" in s
