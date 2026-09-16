"""Unit tests for the significance infrastructure."""
from __future__ import annotations

import importlib.util

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from ktx.stats import (
    auc_metric,
    bonferroni,
    delong_auc_test,
    ece_metric,
    holm_bonferroni,
    paired_bootstrap,
    paired_permutation,
    unpaired_bootstrap_metric,
)


def _synthetic(n=2000, seed=0):
    """Labels with a strong model A and a near-random model B."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n)
    # A: signal + noise; B: mostly noise
    a = np.clip(0.5 + 0.35 * (y - 0.5) * 2 + rng.normal(0, 0.15, n), 0.01, 0.99)
    b = np.clip(0.5 + 0.05 * (y - 0.5) * 2 + rng.normal(0, 0.30, n), 0.01, 0.99)
    return y, a, b


def test_delong_auc_matches_sklearn():
    y, a, b = _synthetic()
    r = delong_auc_test(y, a, b)
    assert abs(r.value_a - roc_auc_score(y, a)) < 1e-9
    assert abs(r.value_b - roc_auc_score(y, b)) < 1e-9
    # A is the stronger model and the gap is large ⇒ significant
    assert r.diff > 0
    assert r.p_value < 0.01


def test_delong_identical_predictions_not_significant():
    y, a, _ = _synthetic()
    r = delong_auc_test(y, a, a.copy())
    assert abs(r.diff) < 1e-9
    assert r.p_value > 0.99


def test_paired_bootstrap_ci_excludes_zero_for_clear_winner():
    y, a, b = _synthetic()
    r = paired_bootstrap(y, a, b, auc_metric, n_boot=500, metric_name="auc", seed=1)
    assert r.diff > 0
    assert r.ci_low > 0          # whole CI above zero ⇒ A significantly better
    assert r.p_value < 0.05


def test_cluster_bootstrap_runs_and_widens_ci():
    # 200 students x 10 interactions; clustered resampling must execute end-to-end
    y, a, b = _synthetic(n=2000, seed=3)
    groups = np.repeat(np.arange(200), 10)
    r = paired_bootstrap(y, a, b, auc_metric, n_boot=300, groups=groups,
                         metric_name="auc", seed=2)
    assert r.method == "cluster_bootstrap"
    assert r.ci_low <= r.diff <= r.ci_high


def test_permutation_detects_difference_and_null():
    y, a, b = _synthetic()
    r = paired_permutation(y, a, b, auc_metric, n_perm=300, metric_name="auc", seed=4)
    assert r.p_value < 0.05
    # identical models ⇒ permutation null, p should be large
    r0 = paired_permutation(y, a, a.copy(), auc_metric, n_perm=300, seed=5)
    assert r0.p_value > 0.5


def test_holm_more_powerful_than_bonferroni():
    pvals = [0.001, 0.013, 0.02, 0.5]
    bonf = bonferroni(pvals, alpha=0.05)
    holm = holm_bonferroni(pvals, alpha=0.05)
    # Holm rejects at least as many hypotheses as Bonferroni
    assert sum(holm["reject"]) >= sum(bonf["reject"])
    # adjusted p-values are monotone non-decreasing in p order
    adj = np.array(holm["adjusted"])
    order = np.argsort(pvals)
    assert np.all(np.diff(adj[order]) >= -1e-12)


def test_holm_smallest_p_matches_bonferroni_factor():
    pvals = [0.004, 0.2, 0.3]
    holm = holm_bonferroni(pvals, alpha=0.05)
    # smallest p gets multiplied by full family size m
    assert abs(holm["adjusted"][0] - 0.004 * 3) < 1e-12


# --- unpaired_bootstrap_metric --- #

# The four below exercise the bootstrap, not the metric; ECE is just what the suite happens
# to feed it. `ece_metric` lazily imports netcal, which pulls torch and gpytorch, so a
# lean install of the significance core cannot run them. Skipping those four is the
# honest outcome -- the machinery under test is covered by the paired tests above, which
# need only numpy and scipy -- and it must be a per-test skip: a module-level
# `importorskip` would silently drop this whole file, including the DeLong and Holm
# tests that have no such dependency.
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
    reason="ece_metric needs netcal; install ktx[calibration] to run these")


@needs_netcal
def test_unpaired_bootstrap_no_groups_recovers_diff():
    # two independent samples, A strictly better calibrated than B
    rng = np.random.default_rng(0)
    ya = rng.integers(0, 2, size=1000)
    pa = np.where(ya == 1, rng.uniform(0.55, 0.95, size=1000),
                          rng.uniform(0.05, 0.45, size=1000))
    yb = rng.integers(0, 2, size=1050)  # different-sized side
    pb = np.where(yb == 1, rng.uniform(0.55, 0.65, size=1050),
                          rng.uniform(0.35, 0.45, size=1050))
    r = unpaired_bootstrap_metric(ya, pa, yb, pb, ece_metric,
                                  n_boot=300, seed=1, metric_name="ece")
    assert r.method == "unpaired_bootstrap"
    # observed diff should sit inside its own CI
    assert r.ci_low <= r.diff <= r.ci_high
    # n = na + nb
    assert r.n == 1000 + 1050


@needs_netcal
def test_unpaired_cluster_bootstrap_uses_shared_students():
    # 40 students each with 50 obs on side A; side B has 40 same + 20 extra students
    rng = np.random.default_rng(1)
    n_per = 50
    a_students = np.repeat(np.arange(40), n_per)
    b_students = np.concatenate([a_students,
                                 np.repeat(np.arange(40, 60), n_per)])
    ya = rng.integers(0, 2, size=a_students.size)
    yb = rng.integers(0, 2, size=b_students.size)
    pa = np.clip(0.5 + 0.3 * (ya - 0.5) * 2 + rng.normal(0, 0.15, ya.size), 0.01, 0.99)
    pb = np.clip(0.5 + 0.05 * (yb - 0.5) * 2 + rng.normal(0, 0.30, yb.size), 0.01, 0.99)
    r = unpaired_bootstrap_metric(
        ya, pa, yb, pb, ece_metric,
        n_boot=300, groups_a=a_students, groups_b=b_students,
        seed=2, metric_name="ece",
    )
    assert r.method == "unpaired_cluster_bootstrap"
    assert r.ci_low <= r.diff <= r.ci_high


@needs_netcal
def test_unpaired_bootstrap_raises_on_disjoint_groups():
    ya = np.array([0, 1, 0, 1]); yb = np.array([1, 0, 1, 0])
    pa = np.array([0.2, 0.8, 0.3, 0.7]); pb = np.array([0.6, 0.4, 0.55, 0.45])
    ga = np.array([0, 0, 1, 1]); gb = np.array([2, 2, 3, 3])  # no overlap
    with pytest.raises(ValueError):
        unpaired_bootstrap_metric(ya, pa, yb, pb, ece_metric,
                                  n_boot=10, groups_a=ga, groups_b=gb)


@needs_netcal
def test_unpaired_bootstrap_centered_on_shared_subsample():
    """Audit §5 point 3: when groups only partially overlap, the observed
    diff must be computed on the shared-students subsample (same as bootstrap),
    not on the full per-side samples. Otherwise the CI can fail to contain
    the point estimate on partial-overlap inputs.
    """
    rng = np.random.default_rng(0)
    n = 200
    # side A has students {0..49}; side B has students {30..79} — overlap is {30..49}
    ga = np.repeat(np.arange(50), n // 50)
    gb = np.repeat(np.arange(30, 80), n // 50)
    # A: preds close to labels for students <30 (side-only), noisy for shared
    ya = rng.integers(0, 2, size=n)
    pa = ya * 0.9 + rng.normal(0.05, 0.02, n)
    mask_a_shared_only = ga >= 30
    pa[~mask_a_shared_only] = rng.beta(2, 5, size=(~mask_a_shared_only).sum())
    # B: uniform noise
    yb = rng.integers(0, 2, size=n)
    pb = rng.uniform(0, 1, size=n)

    r = unpaired_bootstrap_metric(
        ya, pa, yb, pb, ece_metric,
        n_boot=500, groups_a=ga, groups_b=gb, seed=0,
    )
    # observed diff should sit inside its own CI (allowing a tiny slack for
    # discretization of quantiles) since both are computed on the same subsample
    assert r.ci_low <= r.diff <= r.ci_high, (r.ci_low, r.diff, r.ci_high)
