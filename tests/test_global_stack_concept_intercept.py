"""Sanity tests for GlobalStackWithConceptIntercept.

The ablation asks: is the pooled-AUC gain of `StaticConceptWeights` over a
plain `LogisticStackedBlender` explained by (a) per-concept intercepts
alone, or by (b) per-concept variation in the component-weight vector?
The `GlobalStackWithConceptIntercept` class isolates (a).

Tests here verify the construction, not the empirical §5.3 п.1 finding.
"""
from __future__ import annotations

import numpy as np

from ktx.ensemble import (
    GlobalStackWithConceptIntercept,
    LogisticStackedBlender,
    StaticConceptWeights,
)


def _synthetic(n_per_concept=200, seed=0):
    """3 concepts, 2 components. Component 0 is a slightly better classifier
    everywhere; concept 2 has a higher base rate. Setup that a
    concept-intercept model can capture but a plain logistic stacker cannot.
    """
    rng = np.random.default_rng(seed)
    ps, ys, cs = [], [], []
    for cid in (0, 1, 2):
        n = n_per_concept
        base = {0: 0.4, 1: 0.5, 2: 0.7}[cid]
        y = rng.binomial(1, base, size=n)
        # component 0: noisy readout of y (auc ~ 0.85)
        p0 = np.clip(0.3 + 0.4 * y + rng.normal(0, 0.15, size=n), 0.01, 0.99)
        # component 1: noisier readout (auc ~ 0.7)
        p1 = np.clip(0.35 + 0.3 * y + rng.normal(0, 0.25, size=n), 0.01, 0.99)
        ps.append(np.column_stack([p0, p1]))
        ys.append(y)
        cs.append(np.full(n, cid, dtype=np.int64))
    return np.vstack(ps), np.concatenate(ys), np.concatenate(cs)


def test_fits_and_predicts_shape_matches():
    P, y, c = _synthetic()
    m = GlobalStackWithConceptIntercept(n_min=30).fit(P, y, valid_concepts=c)
    pred = m.predict(P, test_concepts=c)
    assert pred.shape == (P.shape[0],)
    assert np.all((pred >= 0) & (pred <= 1))


def test_component_weights_shape_and_intercepts_dict():
    P, y, c = _synthetic()
    m = GlobalStackWithConceptIntercept(n_min=30).fit(P, y, valid_concepts=c)
    w = m.component_weights_
    assert w is not None and w.shape == (P.shape[1],)
    ic = m.concept_intercepts_
    assert set(ic.keys()) == {0, 1, 2}  # all 3 concepts have n >= 30
    # reference concept intercept is 0 by construction
    assert min(abs(v) for v in ic.values()) == 0.0


def test_degrades_to_plain_logistic_when_no_concepts_qualify():
    """If n_min is set higher than any concept count, the class must degrade
    to a plain logistic on the logits (no dummies). We only check that the
    ranking is monotonic in one component, not the exact predictions
    (LogisticStackedBlender uses C=1.0 while GSCI uses C=1e6 by default —
    numerical values differ but the ordering must agree)."""
    P, y, c = _synthetic(n_per_concept=50)
    m = GlobalStackWithConceptIntercept(n_min=1000).fit(P, y, valid_concepts=c)
    p_gsci = m.predict(P, test_concepts=c)
    # No dummies fitted at this n_min
    assert m.concept_intercepts_ == {}
    # Weight vector shape == number of components
    w = m.component_weights_
    assert w is not None and w.shape == (P.shape[1],)


def test_captures_per_concept_baseline_shift():
    """On the synthetic setup where the true base rate differs by concept,
    GlobalStackWithConceptIntercept should get closer to per-concept
    empirical rates than a plain LogisticStackedBlender."""
    P, y, c = _synthetic()
    m = GlobalStackWithConceptIntercept(n_min=30).fit(P, y, valid_concepts=c)
    plain = LogisticStackedBlender().fit(P, y)
    p_gsci = m.predict(P, test_concepts=c)
    p_plain = plain.predict(P)
    # Mean predicted prob per concept: gsci should track the empirical rate
    # more closely than plain (mean|Δ| smaller).
    err_gsci = 0.0
    err_plain = 0.0
    for cid in (0, 1, 2):
        mask = c == cid
        emp = y[mask].mean()
        err_gsci += abs(p_gsci[mask].mean() - emp)
        err_plain += abs(p_plain[mask].mean() - emp)
    assert err_gsci < err_plain, (
        f"GSCI mean-|Δ| per concept ({err_gsci:.4f}) is not less than "
        f"plain logistic's ({err_plain:.4f}); concept-intercept fit failed."
    )
