"""Statistical-significance infrastructure for model comparison.

Every claim about ranking (in)stability must rest on significance, not eyeballed
AUC gaps. This module provides the paired tests we use throughout:

- ``delong_auc_test``  — fast DeLong test for two *correlated* ROC AUCs measured on the
  same test set (Sun & Xu 2014). The right tool for "is model A's AUC > model B's AUC?".
- ``paired_bootstrap`` — resampling CI + two-sided p for *any* metric difference
  (AUC, ECE, Brier, ...). Supports **cluster (student-level) resampling** via ``groups``,
  which is the honest choice for KT data where interactions within a student are not
  independent.
- ``paired_permutation`` — exchangeability test that swaps the two models' predictions
  (per instance, or per student when ``groups`` is given) to build the null.
- ``holm_bonferroni`` / ``bonferroni`` — family-wise error control across the many
  pairwise comparisons in a model x dataset matrix.

All comparisons are paired: ``prob_a`` and ``prob_b`` are two models' predictions on the
*same* aligned ``y_true``. Differences are reported as ``a - b`` (positive ⇒ A better,
for "higher is better" metrics).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from scipy import stats

MetricFn = Callable[[np.ndarray, np.ndarray], float]


@dataclass
class ComparisonResult:
    metric: str
    value_a: float
    value_b: float
    diff: float          # value_a - value_b
    ci_low: float        # CI on the difference (NaN for DeLong/permutation)
    ci_high: float
    p_value: float       # two-sided
    method: str
    n: int

    def __str__(self) -> str:  # compact, log-friendly
        return (f"{self.method}[{self.metric}] A={self.value_a:.4f} B={self.value_b:.4f} "
                f"diff={self.diff:+.4f} CI=[{self.ci_low:.4f},{self.ci_high:.4f}] "
                f"p={self.p_value:.4g} (n={self.n})")


# --------------------------------------------------------------------------- #
# DeLong test for two correlated AUCs (Sun & Xu 2014, fast midrank algorithm)
# --------------------------------------------------------------------------- #
def _midrank(x: np.ndarray) -> np.ndarray:
    """Mid-ranks (ties get the average rank), 1-based, vectorised."""
    order = np.argsort(x)
    z = x[order]
    n = len(x)
    t = np.zeros(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n and z[j] == z[i]:
            j += 1
        t[i:j] = 0.5 * (i + j - 1) + 1.0
        i = j
    out = np.empty(n, dtype=float)
    out[order] = t
    return out


def _fast_delong(preds_sorted: np.ndarray, m: int):
    """Core fast-DeLong. ``preds_sorted`` is (k, n) with the m positives first.

    Returns (aucs[k], covariance[k,k]).
    """
    k, total = preds_sorted.shape
    n = total - m
    pos = preds_sorted[:, :m]
    neg = preds_sorted[:, m:]
    tx = np.empty([k, m]); ty = np.empty([k, n]); tz = np.empty([k, total])
    for r in range(k):
        tx[r] = _midrank(pos[r])
        ty[r] = _midrank(neg[r])
        tz[r] = _midrank(preds_sorted[r])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    cov = sx / m + sy / n
    return aucs, np.atleast_2d(cov)


def delong_auc_test(y_true, prob_a, prob_b) -> ComparisonResult:
    """Two-sided DeLong test that AUC(A) != AUC(B) on the same test set."""
    y = np.asarray(y_true).astype(int).ravel()
    a = np.asarray(prob_a, dtype=float).ravel()
    b = np.asarray(prob_b, dtype=float).ravel()
    if len(np.unique(y)) < 2:
        raise ValueError("DeLong needs both classes present in y_true")

    order = np.argsort(-y)  # positives (label 1) first
    m = int(y.sum())
    preds_sorted = np.vstack([a[order], b[order]])
    aucs, cov = _fast_delong(preds_sorted, m)

    L = np.array([1.0, -1.0])
    var = float(L @ cov @ L)
    diff = float(aucs[0] - aucs[1])
    if var <= 0:
        # identical predictions ⇒ no difference; degenerate variance
        z, p, ci_low, ci_high = 0.0, 1.0, 0.0, 0.0
    else:
        se = np.sqrt(var)
        z = diff / se
        p = float(2.0 * stats.norm.sf(abs(z)))
        ci_low = float(diff - 1.959963985 * se)
        ci_high = float(diff + 1.959963985 * se)
    return ComparisonResult(
        metric="auc", value_a=float(aucs[0]), value_b=float(aucs[1]),
        diff=diff, ci_low=ci_low, ci_high=ci_high,
        p_value=p, method="delong", n=int(y.size),
    )


def delong_variance(y_true, y_prob) -> tuple[float, float]:
    """Single-model AUC + DeLong variance of that AUC. For unpaired AUC comparisons."""
    y = np.asarray(y_true).astype(int).ravel()
    p = np.asarray(y_prob, dtype=float).ravel()
    if len(np.unique(y)) < 2:
        raise ValueError("delong_variance needs both classes present in y_true")
    order = np.argsort(-y)
    m = int(y.sum())
    preds_sorted = np.vstack([p[order]])
    aucs, cov = _fast_delong(preds_sorted, m)
    return float(aucs[0]), float(cov[0, 0])


def unpaired_auc_test(y_true_a, prob_a, y_true_b, prob_b) -> ComparisonResult:
    """Two-sided z-test that AUC(A) != AUC(B) on **independent** test sets.

    Uses DeLong-style variance for each AUC separately, then
    z = (AUC_a - AUC_b) / sqrt(Var_a + Var_b). Less powerful than the paired DeLong
    test but the right tool when the two models can't be aligned on the same instances
    (e.g. classical vs pyKT-deep, whose preprocessing drops different rows).
    """
    auc_a, var_a = delong_variance(y_true_a, prob_a)
    auc_b, var_b = delong_variance(y_true_b, prob_b)
    diff = auc_a - auc_b
    var = var_a + var_b
    if var <= 0:
        z, p, ci_low, ci_high = 0.0, 1.0, 0.0, 0.0
    else:
        se = np.sqrt(var)
        z = diff / se
        p = float(2.0 * stats.norm.sf(abs(z)))
        ci_low = float(diff - 1.959963985 * se)
        ci_high = float(diff + 1.959963985 * se)
    return ComparisonResult(
        metric="auc", value_a=auc_a, value_b=auc_b,
        diff=diff, ci_low=ci_low, ci_high=ci_high,
        p_value=p, method="unpaired-delong-z",
        n=int(np.asarray(y_true_a).size + np.asarray(y_true_b).size),
    )


# --------------------------------------------------------------------------- #
# Resampling helpers (instance-level or student-level/cluster)
# --------------------------------------------------------------------------- #
def _group_index(groups) -> tuple[np.ndarray, list[np.ndarray]]:
    """Return (unique_groups, list_of_index_arrays) preserving first-seen order."""
    groups = np.asarray(groups)
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.where(inv == g)[0] for g in range(len(uniq))]
    return uniq, members


def paired_bootstrap(
    y_true, prob_a, prob_b, metric_fn: MetricFn,
    n_boot: int = 2000, groups: Sequence | None = None,
    alpha: float = 0.05, higher_is_better: bool = True,
    seed: int = 0, metric_name: str = "metric",
) -> ComparisonResult:
    """Percentile-bootstrap CI and two-sided p for ``metric(A) - metric(B)``.

    With ``groups`` (e.g. student id per row) resampling is done over whole clusters,
    which respects within-student dependence — the honest CI for KT test sets.
    The p-value is the bootstrap two-sided test that the difference crosses zero.
    """
    y = np.asarray(y_true).astype(int).ravel()
    a = np.asarray(prob_a, dtype=float).ravel()
    b = np.asarray(prob_b, dtype=float).ravel()
    rng = np.random.default_rng(seed)

    obs = float(metric_fn(y, a) - metric_fn(y, b))

    if groups is not None:
        _, members = _group_index(groups)
        n_groups = len(members)

        def resample_idx():
            pick = rng.integers(0, n_groups, size=n_groups)
            return np.concatenate([members[g] for g in pick])
    else:
        n = y.size

        def resample_idx():
            return rng.integers(0, n, size=n)

    diffs = np.empty(n_boot, dtype=float)
    done = 0
    for i in range(n_boot):
        idx = resample_idx()
        ys = y[idx]
        if len(np.unique(ys)) < 2:        # AUC undefined on degenerate resample
            continue
        diffs[done] = metric_fn(ys, a[idx]) - metric_fn(ys, b[idx])
        done += 1
    diffs = diffs[:done]

    lo = float(np.quantile(diffs, alpha / 2))
    hi = float(np.quantile(diffs, 1 - alpha / 2))
    # two-sided bootstrap p: how often the resampled diff is on the other side of 0
    frac_le0 = float(np.mean(diffs <= 0))
    frac_ge0 = float(np.mean(diffs >= 0))
    p = float(min(1.0, 2.0 * min(frac_le0, frac_ge0)))
    return ComparisonResult(
        metric=metric_name, value_a=float(metric_fn(y, a)), value_b=float(metric_fn(y, b)),
        diff=obs, ci_low=lo, ci_high=hi, p_value=p,
        method="cluster_bootstrap" if groups is not None else "bootstrap",
        n=int(y.size),
    )


def unpaired_bootstrap_metric(
    y_true_a, prob_a, y_true_b, prob_b, metric_fn: MetricFn,
    n_boot: int = 2000,
    groups_a: Sequence | None = None,
    groups_b: Sequence | None = None,
    alpha: float = 0.05,
    seed: int = 0,
    metric_name: str = "metric",
) -> ComparisonResult:
    """Unpaired bootstrap CI + two-sided p for ``metric(A) − metric(B)`` on
    **different** test sets.

    Use when the two sides can't share a common ``y_true`` — e.g. cross-family
    KT contrasts where pyKT drops each student's first interaction, so
    ``n_test_deep ≈ 0.98 × n_test_classical``. With ``groups_a`` / ``groups_b``
    resampling is done at the student (cluster) level **restricted to the
    students shared between both sides**, so within-student correlation is
    respected without pretending the two sides are aligned row-by-row. Without
    groups this degenerates to independent instance-level bootstrap on each
    side.

    The observed diff is computed on the *full* per-side sample; the bootstrap
    distribution is used for CI + two-sided p only.
    """
    ya = np.asarray(y_true_a).astype(int).ravel()
    pa = np.asarray(prob_a, dtype=float).ravel()
    yb = np.asarray(y_true_b).astype(int).ravel()
    pb = np.asarray(prob_b, dtype=float).ravel()
    rng = np.random.default_rng(seed)

    if groups_a is not None and groups_b is not None:
        ga = np.asarray(groups_a)
        gb = np.asarray(groups_b)
        shared = np.array(sorted(set(ga.tolist()) & set(gb.tolist())))
        if shared.size == 0:
            raise ValueError("no shared groups between the two sides")
        a_idx = {u: np.where(ga == u)[0] for u in shared}
        b_idx = {u: np.where(gb == u)[0] for u in shared}
        n_units = shared.size

        # Audit fix (§5, point 3): center the observed diff on the SAME
        # shared-students subsample the bootstrap resamples from, otherwise
        # a bootstrap CI built on the intersection can fail to contain the
        # diff computed on full per-side samples. This is a latent bug — on
        # current data the intersection is 100% so nothing changes, but the
        # unit test guards against future partial overlaps.
        a_all = np.concatenate([a_idx[u] for u in shared])
        b_all = np.concatenate([b_idx[u] for u in shared])
        obs = float(metric_fn(ya[a_all], pa[a_all]) - metric_fn(yb[b_all], pb[b_all]))

        def resample():
            pick = shared[rng.integers(0, n_units, size=n_units)]
            ai = np.concatenate([a_idx[u] for u in pick])
            bi = np.concatenate([b_idx[u] for u in pick])
            return ai, bi
    else:
        na, nb = ya.size, yb.size
        obs = float(metric_fn(ya, pa) - metric_fn(yb, pb))

        def resample():
            return rng.integers(0, na, size=na), rng.integers(0, nb, size=nb)

    diffs = np.empty(n_boot, dtype=float)
    done = 0
    for _ in range(n_boot):
        ai, bi = resample()
        yas, yes = ya[ai], yb[bi]
        if len(np.unique(yas)) < 2 or len(np.unique(yes)) < 2:
            continue
        diffs[done] = metric_fn(yas, pa[ai]) - metric_fn(yes, pb[bi])
        done += 1
    diffs = diffs[:done]

    lo = float(np.quantile(diffs, alpha / 2))
    hi = float(np.quantile(diffs, 1 - alpha / 2))
    frac_le0 = float(np.mean(diffs <= 0))
    frac_ge0 = float(np.mean(diffs >= 0))
    p = float(min(1.0, 2.0 * min(frac_le0, frac_ge0)))
    return ComparisonResult(
        metric=metric_name,
        value_a=float(metric_fn(ya, pa)),
        value_b=float(metric_fn(yb, pb)),
        diff=obs, ci_low=lo, ci_high=hi, p_value=p,
        method="unpaired_cluster_bootstrap" if (groups_a is not None and groups_b is not None)
                else "unpaired_bootstrap",
        n=int(ya.size + yb.size),
    )


def paired_permutation(
    y_true, prob_a, prob_b, metric_fn: MetricFn,
    n_perm: int = 2000, groups: Sequence | None = None,
    seed: int = 0, metric_name: str = "metric",
) -> ComparisonResult:
    """Two-sided paired permutation test for ``metric(A) - metric(B)``.

    Under the null the two models are exchangeable, so for each permutation we randomly
    swap A/B predictions (per instance, or per student when ``groups`` is given) and
    recompute the difference. p = fraction of |permuted diff| >= |observed diff|.
    """
    y = np.asarray(y_true).astype(int).ravel()
    a = np.asarray(prob_a, dtype=float).ravel()
    b = np.asarray(prob_b, dtype=float).ravel()
    rng = np.random.default_rng(seed)
    obs = float(metric_fn(y, a) - metric_fn(y, b))

    if groups is not None:
        _, members = _group_index(groups)
        n_units = len(members)
        unit_idx = members
    else:
        n_units = y.size
        unit_idx = None

    count = 0
    for _ in range(n_perm):
        swap = rng.random(n_units) < 0.5
        pa = a.copy(); pb = b.copy()
        if unit_idx is None:
            pa[swap], pb[swap] = b[swap], a[swap]
        else:
            for u in np.where(swap)[0]:
                ix = unit_idx[u]
                pa[ix], pb[ix] = b[ix], a[ix]
        d = metric_fn(y, pa) - metric_fn(y, pb)
        if abs(d) >= abs(obs) - 1e-15:
            count += 1
    p = (count + 1) / (n_perm + 1)  # add-one for unbiased small-sample p
    return ComparisonResult(
        metric=metric_name, value_a=float(metric_fn(y, a)), value_b=float(metric_fn(y, b)),
        diff=obs, ci_low=float("nan"), ci_high=float("nan"), p_value=float(p),
        method="permutation", n=int(y.size),
    )


# --------------------------------------------------------------------------- #
# Multiple-comparison correction
# --------------------------------------------------------------------------- #
def bonferroni(pvals: Sequence[float], alpha: float = 0.05):
    """Bonferroni: adjusted p = min(1, p*m); reject if adjusted <= alpha."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    adj = np.minimum(1.0, p * m)
    return {"adjusted": adj.tolist(), "reject": (adj <= alpha).tolist(), "alpha": alpha}


def holm_bonferroni(pvals: Sequence[float], alpha: float = 0.05):
    """Holm step-down. Uniformly more powerful than Bonferroni, same FWER control.

    Returns adjusted p-values (monotone, in original order) and reject flags.
    """
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj_sorted = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * p[idx]
        running = max(running, val)        # enforce monotonicity
        adj_sorted[idx] = min(1.0, running)
    return {"adjusted": adj_sorted.tolist(),
            "reject": (adj_sorted <= alpha).tolist(), "alpha": alpha}


# --------------------------------------------------------------------------- #
# Convenience metric callables (paired, share y_true)
# --------------------------------------------------------------------------- #
def auc_metric(y_true, y_prob) -> float:
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y_true, y_prob))


def ece_metric(y_true, y_prob, n_bins: int = 10) -> float:
    from netcal.metrics import ECE
    return float(ECE(bins=n_bins).measure(np.asarray(y_prob, float),
                                           np.asarray(y_true, int)))


def brier_metric(y_true, y_prob) -> float:
    from sklearn.metrics import brier_score_loss
    return float(brier_score_loss(y_true, y_prob))


def ks_calibration(y_true, y_prob) -> float:
    """Binning-free calibration statistic (Widmann et al. 2019; Gupta 2021):
    sup_t |Ĉ(t) − Û(t)|, where Ĉ(t) is the empirical CDF of predicted
    probabilities and Û(t) is the empirical CDF of realized labels ordered
    by predicted probability. Unlike ECE, has no bin-count hyperparameter and
    no Vaicenavicius (2019) discretization bias.
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()
    n = y_true.size
    if n == 0:
        return 0.0
    order = np.argsort(y_prob, kind="mergesort")
    yp = y_prob[order]
    yt = y_true[order]
    # cumulative predicted probability (Ĉ), cumulative realized labels (Û),
    # both divided by n. sup |diff| is the KS-style calibration statistic.
    cum_pred = np.cumsum(yp) / n
    cum_true = np.cumsum(yt) / n
    return float(np.max(np.abs(cum_pred - cum_true)))


def se_shrunk_ece(y_true, y_prob, n_bins: int = 10) -> float:
    """SE-shrunk ECE — heuristic per-bin variance shrinkage.

    For each equal-width bin b of size n_b with mean confidence
    ``p_b = mean(prob[b])``, subtracts ``sqrt(p_b (1 - p_b) / n_b)`` (a
    Wald-style standard-error estimate for a single-bin binomial gap)
    from the naive gap ``|conf_b - acc_b|``, clips at zero, then weights
    by ``n_b / n`` as in standard ECE. Empty bins contribute zero.

    **This is NOT the Vaicenavicius et al. (2019, Prop. 3) debiased ECE
    and NOT the Kumar-Liang-Ma (2019) verified-calibration estimator.**
    Prior versions of this function claimed the Vaicenavicius attribution,
    but Prop. 3 there is a bias-variance decomposition (a theoretical
    result about `E[naïve ECE] - true ECE`), not a subtract-the-SE-per-
    bin construction. What is implemented here is a heuristic shrinkage
    correction — informally: "reduce each bin's ECE contribution by its
    Wald SE to prevent single-bin outliers driving the average". It is
    useful as a smoothing pass on small samples but does not carry the
    formal debiasing guarantees of a proper KLM-style estimator (which
    would additionally split the sample and use held-out bin edges).

    Kept in the codebase for reproducibility of earlier results that
    reported "debiased_ece"; new code should either use the standard
    ``ece_metric`` (equal-width or equal-mass) or the KLM estimator
    (planned as a separate implementation; not required for any current
    §5.4 / §5.5 headline). Reference for KLM: Kumar, Liang & Ma,
    "Verified Uncertainty Calibration", NeurIPS 2019.

    Returns a non-negative float, clipped at 0.
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(y_prob, edges, right=True) - 1, 0, n_bins - 1)
    n = y_true.size
    ece = 0.0
    for b in range(n_bins):
        mask = (idx == b)
        n_b = int(mask.sum())
        if n_b == 0:
            continue
        conf = float(y_prob[mask].mean())
        acc = float(y_true[mask].mean())
        naive = abs(conf - acc)
        # per-bin Wald SE for a binomial gap; shrink the naive gap by it
        se = float(np.sqrt(conf * (1.0 - conf) / max(n_b, 1)))
        adj = max(0.0, naive - se)
        ece += (n_b / n) * adj
    return float(ece)


# Backward-compat alias — kept so earlier scripts (calibration_second_metric.py
# etc.) do not break. New code should import ``se_shrunk_ece`` directly.
debiased_ece = se_shrunk_ece
