"""Regression tests for BayesianModelAveraging.

Before T3a, `BayesianModelAveraging.fit` computed weights from
`softmax(-mean_NLL / T)`, which drops the sample-size factor and produces
posterior weights nearly uniform on any realistic KT valid split — making
BMA numerically indistinguishable from arithmetic mean. The audit
documented this as "BMA implementation artifact: BMA ≡ arithmetic_mean
across all 14 ensemble cells; max ΔAUC = 0.0011".

After T3a, weights are computed from `softmax(log L_m / T) =
softmax(-n · mean_NLL_m / T)` — the flat-prior Bayesian posterior from
total log-likelihood, which at large n collapses to one-hot on the
component with the smallest NLL (canonical Bayesian model selection).

Tests below assert both properties.
"""
from __future__ import annotations

import numpy as np

from ktx.ensemble import BayesianModelAveraging


def _make_two_models(n: int, gap: float, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Two-component fixture: true labels drawn from a Bernoulli(0.6);
    component 0 predicts 0.6 uniformly; component 1 predicts 0.6 + `gap`.
    Component 0 has strictly lower NLL for any nonzero gap.
    """
    rng = np.random.default_rng(seed)
    y = rng.binomial(1, 0.6, size=n).astype(int)
    p0 = np.full(n, 0.6, dtype=float)
    p1 = np.full(n, np.clip(0.6 + gap, 1e-4, 1 - 1e-4), dtype=float)
    P = np.column_stack([p0, p1])
    return P, y


def test_weights_collapse_to_one_hot_at_large_n():
    """As n grows, softmax(-n · NLL / T) → one-hot on argmin(NLL)."""
    for n in (1_000, 10_000, 100_000):
        P, y = _make_two_models(n=n, gap=0.02, seed=42)
        bma = BayesianModelAveraging(temperature=1.0).fit(P, y)
        assert bma.weights_ is not None
        # At n = 10^5, the softmax should be essentially degenerate on the
        # better component. We accept 0.999 as the practical threshold.
        top = bma.weights_.max()
        if n >= 100_000:
            assert top > 0.999, (
                f"BMA weights did not collapse at n={n}; got weights={bma.weights_}, "
                "expected top > 0.999 (canonical Bayesian large-sample behavior)."
            )
        assert bma.weights_.argmax() == 0, (
            f"BMA picked the wrong component at n={n}: argmax={bma.weights_.argmax()}, "
            "expected the lower-NLL component (index 0)."
        )


def test_weights_not_uniform_on_realistic_kt_scale():
    """Regression: pre-T3a version returned near-uniform weights for the
    small per-example NLL differences typical of KT (~0.001-0.05). Verify
    the fix produces non-uniform weights at n ≈ 50k with a modest gap.
    """
    P, y = _make_two_models(n=50_000, gap=0.02, seed=7)
    bma = BayesianModelAveraging(temperature=1.0).fit(P, y)
    assert bma.weights_ is not None
    # Uniform k=2 would be (0.5, 0.5); with n · Δmean_NLL >> 1 the weights
    # must be sharply asymmetric.
    assert abs(bma.weights_[0] - 0.5) > 0.4, (
        f"BMA weights nearly uniform: {bma.weights_}; the pre-T3a bug is not fixed."
    )


def test_temperature_flattens_posterior():
    """Larger T flattens the posterior toward uniform — the canonical
    tempered-posterior effect used by the paper's ``temperature`` param.
    """
    P, y = _make_two_models(n=50_000, gap=0.02, seed=11)
    w_low = BayesianModelAveraging(temperature=1.0).fit(P, y).weights_
    w_high = BayesianModelAveraging(temperature=1e6).fit(P, y).weights_
    assert w_low is not None and w_high is not None
    assert abs(w_high[0] - 0.5) < abs(w_low[0] - 0.5), (
        f"Temperature did not flatten posterior: T=1 -> {w_low}, T=1e6 -> {w_high}"
    )


def test_predict_is_convex_combination():
    """Sanity: predictions must equal p @ w exactly."""
    P, y = _make_two_models(n=1_000, gap=0.05)
    bma = BayesianModelAveraging(temperature=1.0).fit(P, y)
    P_test = np.array([[0.3, 0.6], [0.7, 0.2], [0.5, 0.5]])
    expected = P_test @ bma.weights_
    got = bma.predict(P_test)
    np.testing.assert_allclose(got, expected, atol=1e-12)
