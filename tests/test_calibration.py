"""Unit tests for post-hoc calibration methods."""
from __future__ import annotations

import importlib

import numpy as np
import pytest

from ktx.calibration import (
    IsotonicCalibration,
    PlattScaling,
    TemperatureScaling,
)
from ktx.metrics import compute_metrics


def _overconfident(n=4000, seed=0):
    """Make systematically overconfident probabilities on a held-out split.

    True P(y=1)=q but the model reports a sharpened probability, so ECE is large and a
    temperature T>1 should help. Returns (valid_p, valid_y, test_p, test_y)."""
    rng = np.random.default_rng(seed)
    def gen(m):
        q = rng.uniform(0.05, 0.95, m)
        y = (rng.random(m) < q).astype(int)
        z = np.log(q / (1 - q)) * 2.2            # sharpen logits ⇒ overconfident
        p = 1 / (1 + np.exp(-z))
        return p, y
    vp, vy = gen(n)
    tp, ty = gen(n)
    return vp, vy, tp, ty


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
    reason="проверка меряет ошибку калибровки через netcal; "
           "поставьте ktx[calibration]")


@needs_netcal
def test_temperature_reduces_ece_on_heldout():
    vp, vy, tp, ty = _overconfident()
    ece_before = compute_metrics(ty, tp).ece
    cal = TemperatureScaling().fit(vp, vy)
    ece_after = compute_metrics(ty, cal.transform(tp)).ece
    assert cal.T_ > 1.0                          # softening overconfidence
    assert ece_after < ece_before
    assert ece_after < 0.5 * ece_before          # substantial improvement


def test_temperature_near_one_for_calibrated_input():
    rng = np.random.default_rng(3)
    q = rng.uniform(0.05, 0.95, 5000)
    y = (rng.random(5000) < q).astype(int)
    cal = TemperatureScaling().fit(q, y)
    assert 0.8 < cal.T_ < 1.25                   # already calibrated ⇒ T≈1


@needs_netcal
def test_platt_and_isotonic_improve_heldout_ece():
    vp, vy, tp, ty = _overconfident(seed=1)
    ece_before = compute_metrics(ty, tp).ece
    for cal in (PlattScaling(), IsotonicCalibration()):
        out = cal.fit(vp, vy).transform(tp)
        assert out.shape == tp.shape
        assert out.min() >= 0.0 and out.max() <= 1.0
        assert compute_metrics(ty, out).ece < ece_before


def test_isotonic_monotone():
    vp, vy, tp, ty = _overconfident(seed=2)
    cal = IsotonicCalibration().fit(vp, vy)
    xs = np.linspace(0.01, 0.99, 200)
    ys = cal.transform(xs)
    assert np.all(np.diff(ys) >= -1e-9)          # non-decreasing
