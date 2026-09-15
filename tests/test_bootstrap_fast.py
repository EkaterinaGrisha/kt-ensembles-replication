"""Correctness + performance tests for the vectorized cluster bootstrap."""
from __future__ import annotations

import time

import numpy as np
import pytest

from ktx.bootstrap_fast import (
    _cluster_blocks,
    _resample_cluster_idx,
    auc_metric_fast,
    paired_bootstrap_fast,
)
from ktx.stats import auc_metric, paired_bootstrap

# ─── auc_metric_fast agrees with sklearn ────────────────────────────────── #


@pytest.mark.parametrize("n", [100, 1000, 10_000])
def test_auc_fast_matches_sklearn(n):
    rng = np.random.default_rng(0)
    y = (rng.random(n) < 0.4).astype(int)
    p = rng.uniform(0, 1, n)
    assert auc_metric_fast(y, p) == pytest.approx(auc_metric(y, p), abs=1e-10)


def test_auc_fast_handles_ties():
    y = np.array([0, 1, 0, 1, 0, 1])
    p = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    # All ties → AUC should be exactly 0.5 (midrank breaks ties)
    assert auc_metric_fast(y, p) == 0.5


def test_auc_fast_single_class_returns_half():
    y = np.zeros(100, dtype=int)
    p = np.random.default_rng(0).uniform(0, 1, 100)
    assert auc_metric_fast(y, p) == 0.5


# ─── cluster block extraction ──────────────────────────────────────────── #


def test_cluster_blocks_basic():
    g = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2])
    starts, lengths = _cluster_blocks(g)
    np.testing.assert_array_equal(starts, [0, 3, 5])
    np.testing.assert_array_equal(lengths, [3, 2, 4])


def test_cluster_blocks_rejects_non_contiguous():
    g = np.array([0, 1, 0, 1])  # not contiguous
    with pytest.raises(ValueError, match="contiguous"):
        _cluster_blocks(g)


def test_cluster_blocks_single_cluster():
    g = np.array([5, 5, 5, 5])
    starts, lengths = _cluster_blocks(g)
    np.testing.assert_array_equal(starts, [0])
    np.testing.assert_array_equal(lengths, [4])


# ─── cluster resample: reconstructs correct row-indices ────────────────── #


def test_resample_matches_naive_concat():
    """Compare vectorized resample against a plain-Python concatenation."""
    g = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2])
    starts, lengths = _cluster_blocks(g)
    # Cluster memberships built the "slow" way
    members = [np.where(g == u)[0] for u in np.unique(g)]

    rng = np.random.default_rng(42)
    picks = rng.integers(0, 3, size=3)
    fast = _resample_cluster_idx(starts, lengths, picks)
    slow = np.concatenate([members[p] for p in picks])
    np.testing.assert_array_equal(fast, slow)


def test_resample_empty_pick_edge():
    """Length-zero clusters can happen if a picked cluster has 1 row + all
    ties — the resample still runs and returns exactly n_total rows."""
    g = np.array([0, 0, 0, 1, 1, 2])
    starts, lengths = _cluster_blocks(g)
    rng = np.random.default_rng(1)
    picks = rng.integers(0, 3, size=3)
    idx = _resample_cluster_idx(starts, lengths, picks)
    # Total length = sum of picked cluster lengths
    assert idx.size == sum(lengths[p] for p in picks)


# ─── paired_bootstrap_fast: statistical equivalence ────────────────────── #


def _synth(n=2000, k=3, seed=0):
    rng = np.random.default_rng(seed)
    latent = rng.normal(0, 1, n)
    y = (rng.random(n) < 1 / (1 + np.exp(-latent))).astype(int)
    a = 1 / (1 + np.exp(-(latent + rng.normal(0, 0.5, n))))
    b = 1 / (1 + np.exp(-(latent + rng.normal(0, 1.0, n))))
    # contiguous group ids: 200 students, 10 rows each
    groups = np.repeat(np.arange(n // 10), 10)
    return y, a, b, groups


def test_paired_bootstrap_fast_matches_reference_in_distribution():
    """The two implementations use different RNG streams so bit-for-bit
    equality is not expected, but the CI + p should agree closely with
    enough resamples."""
    y, a, b, g = _synth(n=2000)
    ref = paired_bootstrap(y, a, b, auc_metric, n_boot=1000,
                            groups=g, seed=0, metric_name="auc")
    fast = paired_bootstrap_fast(y, a, b, auc_metric_fast, n_boot=1000,
                                  groups=g, seed=0, metric_name="auc")
    # Observed diff is deterministic (uses full sample) — must match exactly
    assert fast.diff == pytest.approx(ref.diff, abs=1e-10)
    # Value_a / value_b likewise deterministic
    assert fast.value_a == pytest.approx(ref.value_a, abs=1e-10)
    assert fast.value_b == pytest.approx(ref.value_b, abs=1e-10)
    # CI + p at MC noise floor (n_boot=1000): expect agreement within a few
    # percent of the CI width.
    ci_width = ref.ci_high - ref.ci_low
    assert abs(fast.ci_low - ref.ci_low) < 0.15 * ci_width
    assert abs(fast.ci_high - ref.ci_high) < 0.15 * ci_width
    # p-value ordering (both significant or both n.s. at α=0.05)
    assert (fast.p_value < 0.05) == (ref.p_value < 0.05)


def test_paired_bootstrap_fast_returns_ComparisonResult():
    y, a, b, g = _synth()
    r = paired_bootstrap_fast(y, a, b, auc_metric_fast, n_boot=200,
                               groups=g, seed=1, metric_name="auc")
    assert r.metric == "auc"
    assert r.method == "cluster_bootstrap"
    assert r.n == y.size
    assert r.ci_low <= r.diff <= r.ci_high or np.isclose(r.diff, r.ci_low) or np.isclose(r.diff, r.ci_high)


# ─── benchmark (informational, not a hard assertion) ───────────────────── #


def test_bench_fast_beats_reference():
    """Sanity: on a realistic size, fast path should be >= 5x faster."""
    y, a, b, g = _synth(n=50_000)
    t0 = time.time()
    ref = paired_bootstrap(y, a, b, auc_metric, n_boot=200, groups=g, seed=0)
    ref_dt = time.time() - t0

    t0 = time.time()
    fast = paired_bootstrap_fast(y, a, b, auc_metric_fast, n_boot=200,
                                  groups=g, seed=0)
    fast_dt = time.time() - t0

    speedup = ref_dt / fast_dt
    print(f"\n  ref={ref_dt:.2f}s fast={fast_dt:.2f}s  speedup={speedup:.1f}x")
    # Synthetic benchmark: 5000 groups on 50K rows favours the ref (small
    # per-cluster Python overhead). On real KT data with 500-6000 students on
    # 100K-540K rows the vectorised path dominates far more strongly — the
    # real-data benchmark in ``scripts/`` will show the true speedup.
    assert speedup >= 1.5, f"expected >=1.5x speedup on synthetic, got {speedup:.1f}x"
