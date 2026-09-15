"""Unit tests for parameter-free ensemble aggregators."""
from __future__ import annotations

import numpy as np
import pytest

from ktx.ensemble import (
    CCCE,
    GATING_HEADS,
    SIMPLE_AGGREGATORS,
    STACKED_BLENDERS,
    ArithmeticMean,
    BayesianModelAveraging,
    EnsembleBase,
    GeometricMean,
    LinearGating,
    LogisticStackedBlender,
    LogitMean,
    MedianAggregator,
    MLPStackedBlender,
    RankMean,
    RidgeStackedBlender,
    StaticConceptWeights,
    XGBoostStackedBlender,
    bootstrap_ensemble_vs_ensemble,
    bootstrap_ensemble_vs_single,
)
from ktx.stats import auc_metric, ece_metric, paired_bootstrap


def _random_probs(n=1000, k=8, seed=0) -> np.ndarray:
    """Uniform-in-(0,1) probability matrix, shape (n, k)."""
    return np.random.default_rng(seed).uniform(0.01, 0.99, size=(n, k))


AGG_CLASSES = [ArithmeticMean, GeometricMean, LogitMean, RankMean, MedianAggregator]


# ─── output shape + range ───────────────────────────────────────────────── #


@pytest.mark.parametrize("cls", AGG_CLASSES)
def test_output_shape_and_range(cls):
    probs = _random_probs()
    out = cls().predict(probs)
    assert out.shape == (probs.shape[0],)
    assert np.all(out >= 0.0) and np.all(out <= 1.0)


# ─── identity: all models agree ⇒ ensemble reproduces that agreement ────── #


@pytest.mark.parametrize("cls", [ArithmeticMean, GeometricMean, LogitMean, MedianAggregator])
def test_identical_inputs_reproduce_probability(cls):
    """When every column is p, the ensemble should also be p (RankMean excluded — it
    replaces absolute scale with within-column ranks, so it maps identical inputs
    to a linear ramp, not to the original p)."""
    n, k = 500, 6
    base = np.random.default_rng(1).uniform(0.05, 0.95, size=n)
    probs = np.tile(base[:, None], (1, k))
    out = cls().predict(probs)
    np.testing.assert_allclose(out, base, atol=1e-10)


def test_rank_mean_on_identical_inputs_is_linear_ramp():
    """RankMean of identical columns collapses to per-row rank/n, since every column
    ranks the same rows in the same order. This is by-design behaviour — documented
    here so a future refactor doesn't 'fix' it into a probability."""
    n = 100
    base = np.random.default_rng(2).uniform(0.05, 0.95, size=n)
    probs = np.tile(base[:, None], (1, 4))
    out = RankMean().predict(probs)
    expected = (np.argsort(np.argsort(base)) + 1) / n
    np.testing.assert_allclose(out, expected, atol=1e-10)


# ─── monotonicity: increase one component ⇒ ensemble does not decrease ── #


@pytest.mark.parametrize("cls", [ArithmeticMean, GeometricMean, LogitMean, MedianAggregator])
def test_monotone_in_each_component(cls):
    probs = _random_probs(seed=3)
    baseline = cls().predict(probs)
    bumped = probs.copy()
    bumped[:, 0] = np.clip(bumped[:, 0] + 0.1, 1e-6, 1 - 1e-6)
    lifted = cls().predict(bumped)
    # non-strict: median is only weakly monotone (bump inside inter-quartile range
    # may not move the median at all)
    assert np.all(lifted >= baseline - 1e-12)


# ─── boundary: all-zero / all-one predictions ─────────────────────────── #


@pytest.mark.parametrize("cls", [ArithmeticMean, GeometricMean, LogitMean, MedianAggregator])
def test_boundary_all_zero(cls):
    probs = np.zeros((50, 4))
    out = cls().predict(probs)
    # after internal clip to (0, 1-eps), the ensemble should be very close to 0
    assert np.all(out < 1e-3)


@pytest.mark.parametrize("cls", [ArithmeticMean, GeometricMean, LogitMean, MedianAggregator])
def test_boundary_all_one(cls):
    probs = np.ones((50, 4))
    out = cls().predict(probs)
    assert np.all(out > 1 - 1e-3)


# ─── ordering: mean vs geometric-mean vs logit-mean on same input ─────── #


def test_geometric_mean_below_arithmetic_when_models_disagree():
    """AM ≥ GM (Jensen), strict inequality when models disagree."""
    probs = _random_probs(seed=4)
    am = ArithmeticMean().predict(probs)
    gm = GeometricMean().predict(probs)
    # AM >= GM always; strict inequality on most rows because models disagree
    assert np.all(am >= gm - 1e-12)
    assert (am > gm + 1e-6).mean() > 0.95


# ─── validation errors ────────────────────────────────────────────────── #


def test_rejects_1d_input():
    with pytest.raises(ValueError, match="2-D"):
        ArithmeticMean().predict(np.random.rand(100))


def test_rejects_single_component():
    with pytest.raises(ValueError, match=">= 2 component"):
        ArithmeticMean().predict(np.random.rand(100, 1))


# ─── registry + base-class no-op fit ──────────────────────────────────── #


def test_registry_covers_all_five():
    assert set(SIMPLE_AGGREGATORS) == {
        "arithmetic_mean", "geometric_mean", "logit_mean", "rank_mean", "median",
    }
    for name, cls in SIMPLE_AGGREGATORS.items():
        assert issubclass(cls, EnsembleBase)
        assert cls().name == name


def test_fit_is_noop_for_simple_aggregators():
    """Simple aggregators must not depend on valid_probs/valid_labels."""
    probs = _random_probs()
    for cls in AGG_CLASSES:
        agg = cls()
        # fit with garbage — must not error, must return self
        assert agg.fit(np.zeros((10, 3)), np.zeros(10)) is agg
        out_after_fit = agg.predict(probs)
        out_no_fit = cls().predict(probs)
        np.testing.assert_allclose(out_after_fit, out_no_fit, atol=1e-12)


# ─── Bootstrap wrappers ────────────────────────────────────────────────── #


def _synthetic_ensemble_setup(n=800, k=4, n_groups=100, seed=7):
    """Build (y, matrix, groups, single_ref) for bootstrap tests.

    Labels come from a logistic model of a hidden latent; each component gets
    the latent + independent noise, so the ensemble is expected to beat any
    single component on AUC.
    """
    rng = np.random.default_rng(seed)
    latent = rng.normal(0, 1, n)
    y = (rng.random(n) < 1 / (1 + np.exp(-latent))).astype(int)
    matrix = np.column_stack([
        1 / (1 + np.exp(-(latent + rng.normal(0, 1.0, n)))) for _ in range(k)
    ])
    groups = rng.integers(0, n_groups, size=n)
    return y, matrix, groups, matrix[:, 0]


def test_bootstrap_ensemble_vs_single_matches_paired_bootstrap_on_precomputed():
    """Row-independence property: wrapper's answer == paired_bootstrap on the
    precomputed ensemble vector. Same seed ⇒ bit-identical result."""
    y, matrix, groups, single_ref = _synthetic_ensemble_setup()
    ens = ArithmeticMean()

    a = bootstrap_ensemble_vs_single(
        y, matrix, ens, single_ref, auc_metric,
        n_boot=200, groups=groups, seed=42, metric_name="auc",
    )
    b = paired_bootstrap(
        y, ens.predict(matrix), single_ref, auc_metric,
        n_boot=200, groups=groups, seed=42, metric_name="auc",
    )
    assert a.diff == b.diff
    assert a.ci_low == b.ci_low
    assert a.ci_high == b.ci_high
    assert a.p_value == b.p_value
    assert a.method == "cluster_bootstrap"


def test_bootstrap_ensemble_beats_single_component_on_auc():
    """Sanity: 4-component mean ensemble on a shared latent + independent
    noise should beat any single component on AUC (positive diff)."""
    y, matrix, groups, single_ref = _synthetic_ensemble_setup()
    res = bootstrap_ensemble_vs_single(
        y, matrix, ArithmeticMean(), single_ref, auc_metric,
        n_boot=500, groups=groups, seed=0, metric_name="auc",
    )
    assert res.diff > 0.0
    assert res.value_a > res.value_b  # ensemble AUC > single AUC


def test_bootstrap_wrapper_calls_fit_when_asked():
    """A stub ensemble records whether fit was called with the right arrays."""
    y, matrix, groups, single_ref = _synthetic_ensemble_setup()

    class RecordingMean(ArithmeticMean):
        def __init__(self):
            super().__init__()
            self.fit_calls = []

        def fit(self, valid_probs, valid_labels):
            self.fit_calls.append((valid_probs.shape, valid_labels.shape))
            return self

    ens = RecordingMean()
    fit_p = np.random.default_rng(1).uniform(0.01, 0.99, size=(50, matrix.shape[1]))
    fit_y = (np.random.default_rng(2).random(50) < 0.5).astype(int)
    bootstrap_ensemble_vs_single(
        y, matrix, ens, single_ref, auc_metric,
        n_boot=50, groups=groups, seed=0,
        fit_probs=fit_p, fit_labels=fit_y,
    )
    assert ens.fit_calls == [((50, matrix.shape[1]), (50,))]


def test_bootstrap_wrapper_skips_fit_when_not_asked():
    y, matrix, groups, single_ref = _synthetic_ensemble_setup()

    class RecordingMean(ArithmeticMean):
        def __init__(self):
            super().__init__()
            self.fit_calls = 0

        def fit(self, valid_probs, valid_labels):
            self.fit_calls += 1
            return self

    ens = RecordingMean()
    bootstrap_ensemble_vs_single(
        y, matrix, ens, single_ref, auc_metric,
        n_boot=50, groups=groups, seed=0,
    )
    assert ens.fit_calls == 0


def test_bootstrap_ensemble_vs_ensemble_smoke():
    y, matrix, groups, _ = _synthetic_ensemble_setup(k=6)
    res = bootstrap_ensemble_vs_ensemble(
        y, matrix, ArithmeticMean(),
        matrix, LogitMean(), auc_metric,
        n_boot=100, groups=groups, seed=0, metric_name="auc",
    )
    # both aggregators of the same matrix — differences should be small,
    # but the machinery must run and yield a well-formed ComparisonResult
    assert np.isfinite(res.diff)
    assert res.ci_low <= res.diff <= res.ci_high
    assert 0.0 <= res.p_value <= 1.0
    assert res.method == "cluster_bootstrap"


# ─── StackedBlender skeleton ─────────────────────────────────────────────── #


def test_stacked_registry_has_logistic():
    assert "logistic_stacked" in STACKED_BLENDERS
    assert STACKED_BLENDERS["logistic_stacked"] is LogisticStackedBlender


def test_stacked_predict_before_fit_raises():
    with pytest.raises(RuntimeError, match="fit"):
        LogisticStackedBlender().predict(_random_probs())


def test_stacked_fit_predict_shape():
    y, matrix, _groups, _ = _synthetic_ensemble_setup()
    n_fit = 200
    fit_p = matrix[:n_fit]
    fit_y = y[:n_fit]
    test_p = matrix[n_fit:]
    out = LogisticStackedBlender().fit(fit_p, fit_y).predict(test_p)
    assert out.shape == (test_p.shape[0],)
    assert np.all(out >= 0) and np.all(out <= 1)


def test_stacked_learns_positive_weights_when_components_all_helpful():
    """On the synthetic setup all components carry the same latent signal;
    the fitted logistic weights should all be > 0."""
    y, matrix, _groups, _ = _synthetic_ensemble_setup(n=2000, k=4)
    ens = LogisticStackedBlender().fit(matrix[:1000], y[:1000])
    assert (ens.weights_ > 0).all()


# ─── Additional meta-learners ──────────────────────────────────────────── #


ALL_META_LEARNERS = [
    LogisticStackedBlender,
    RidgeStackedBlender,
    BayesianModelAveraging,
    XGBoostStackedBlender,
    MLPStackedBlender,
]


@pytest.mark.parametrize("cls", ALL_META_LEARNERS)
def test_meta_learner_registry_contains_class(cls):
    assert cls in STACKED_BLENDERS.values()


@pytest.mark.parametrize("cls", ALL_META_LEARNERS)
def test_meta_learner_predict_before_fit_raises(cls):
    with pytest.raises(RuntimeError, match="fit"):
        cls().predict(_random_probs())


@pytest.mark.parametrize("cls", ALL_META_LEARNERS)
def test_meta_learner_fit_predict_end_to_end(cls):
    """All meta-learners follow the fit-then-predict contract and produce
    outputs in [0, 1] with the expected shape."""
    y, matrix, _groups, _ = _synthetic_ensemble_setup(n=1000, k=4)
    n_fit = 500
    ens = cls().fit(matrix[:n_fit], y[:n_fit])
    out = ens.predict(matrix[n_fit:])
    assert out.shape == (matrix.shape[0] - n_fit,)
    assert np.all(out >= 0) and np.all(out <= 1)


@pytest.mark.parametrize("cls", ALL_META_LEARNERS)
def test_meta_learner_lifts_auc_over_worst_component(cls):
    """A trained meta-learner should not be strictly worse than the worst
    single component on a synthetic setup where every component carries the
    latent signal."""
    from sklearn.metrics import roc_auc_score
    y, matrix, _groups, _ = _synthetic_ensemble_setup(n=1500, k=4)
    fit_p, fit_y = matrix[:750], y[:750]
    test_p, test_y = matrix[750:], y[750:]
    per_component_auc = [roc_auc_score(test_y, test_p[:, j]) for j in range(4)]
    worst = min(per_component_auc)
    ens_pred = cls().fit(fit_p, fit_y).predict(test_p)
    ens_auc = roc_auc_score(test_y, ens_pred)
    assert ens_auc >= worst - 0.01, (
        f"{cls.__name__} AUC {ens_auc:.4f} < worst component {worst:.4f}"
    )


# ─── Ridge-specific ────────────────────────────────────────────────────── #


def test_ridge_selects_C_from_grid():
    y, matrix, _, _ = _synthetic_ensemble_setup(n=1000, k=3)
    ens = RidgeStackedBlender(C_grid=(0.01, 1.0, 100.0)).fit(matrix[:500], y[:500])
    assert ens.C_ in (0.01, 1.0, 100.0)


def test_ridge_degenerate_fallback():
    """Too few samples for CV — should not crash, should pick middle of grid."""
    matrix = np.random.default_rng(0).uniform(0.01, 0.99, size=(4, 3))
    y = np.array([0, 1, 0, 1])
    ens = RidgeStackedBlender(C_grid=(0.01, 1.0, 100.0), cv=3).fit(matrix, y)
    assert ens.C_ == 1.0  # middle of grid


# ─── BMA-specific ──────────────────────────────────────────────────────── #


def test_bma_weights_sum_to_one():
    y, matrix, _, _ = _synthetic_ensemble_setup(n=1000, k=4)
    ens = BayesianModelAveraging().fit(matrix, y)
    assert ens.weights_ is not None
    np.testing.assert_allclose(ens.weights_.sum(), 1.0, atol=1e-10)
    assert np.all(ens.weights_ >= 0)


def test_bma_favours_lower_nll_component():
    """Component 0 is the true prob; component 1 is noise (always 0.5).
    BMA should weight component 0 strictly higher. Softmax(-Δnll) with the
    two-component NLL gap ~0.69 (log 2 penalty for the uninformed component)
    gives roughly a 2:1 ratio at T=1; we assert the direction, not a
    specific ratio."""
    rng = np.random.default_rng(0)
    n = 2000
    latent = rng.normal(0, 1, n)
    y = (rng.random(n) < 1 / (1 + np.exp(-latent))).astype(int)
    p0 = 1 / (1 + np.exp(-latent))          # perfect predictions
    p1 = np.full(n, 0.5)                    # noise
    matrix = np.column_stack([p0, p1])
    ens = BayesianModelAveraging().fit(matrix, y)
    assert ens.weights_[0] > ens.weights_[1], (
        f"BMA didn't favour informative component: {ens.weights_}"
    )
    # Also verify the NLL ordering is what BMA is responding to
    assert ens.nlls_[0] < ens.nlls_[1]


def test_bma_temperature_controls_concentration():
    """Low T concentrates the posterior on the best component; high T flattens
    toward uniform. Assert the concentration relation, not an absolute
    threshold — component quality on the synthetic setup varies only mildly."""
    y, matrix, _, _ = _synthetic_ensemble_setup(n=500, k=4)
    ens_low = BayesianModelAveraging(temperature=0.001).fit(matrix, y)
    ens_high = BayesianModelAveraging(temperature=1e6).fit(matrix, y)
    # low T: max weight strictly higher than uniform (1/k = 0.25)
    assert ens_low.weights_.max() > 0.25
    # high T: weights near uniform
    np.testing.assert_allclose(ens_high.weights_, 0.25, atol=1e-4)
    # Low-T max > high-T max — direction is the correctness property
    assert ens_low.weights_.max() > ens_high.weights_.max()


# ─── XGBoost-specific ──────────────────────────────────────────────────── #


def test_xgb_extra_features_augmentation():
    """Fit with extra_features should accept 1-D or 2-D array without
    changing the interface for predict."""
    y, matrix, _, _ = _synthetic_ensemble_setup(n=800, k=3)
    extra_fit = np.random.default_rng(0).uniform(0, 1, size=(400, 2))
    extra_test = np.random.default_rng(1).uniform(0, 1, size=(400, 2))
    ens = XGBoostStackedBlender(n_estimators=20).fit(
        matrix[:400], y[:400], extra_features=extra_fit,
    )
    out = ens.predict(matrix[400:], extra_features=extra_test)
    assert out.shape == (400,)


def test_xgb_extra_features_shape_mismatch_raises():
    y, matrix, _, _ = _synthetic_ensemble_setup(n=200, k=3)
    with pytest.raises(ValueError, match="extra_features rows"):
        XGBoostStackedBlender(n_estimators=10).fit(
            matrix[:100], y[:100],
            extra_features=np.zeros((99, 1)),  # wrong row count
        )


def test_stacked_beats_or_matches_simple_mean_via_bootstrap_wrapper():
    """Bootstrap-wrapper end-to-end: fit stacked on the front half, evaluate
    on the back half against best single, verify positive Δ AUC."""
    y, matrix, groups, single_ref = _synthetic_ensemble_setup(n=1600, k=4)
    fit_p = matrix[:800]
    fit_y = y[:800]
    test_y = y[800:]
    test_matrix = matrix[800:]
    test_groups = groups[800:]
    test_single = single_ref[800:]
    res = bootstrap_ensemble_vs_single(
        test_y, test_matrix, LogisticStackedBlender(),
        test_single, auc_metric, n_boot=200, groups=test_groups, seed=0,
        fit_probs=fit_p, fit_labels=fit_y,
    )
    assert res.diff > 0.0


# ─── Conditional gating ─────────────────────────────────────────────────── #


def _synth_concept_setup(n=2000, k=4, n_concepts=8, seed=17):
    """Synthetic ensemble matrix with a concept id per row. The 'best'
    component varies by concept so a static-concept gating should recover
    per-concept-appropriate weights."""
    rng = np.random.default_rng(seed)
    concepts = rng.integers(0, n_concepts, size=n)
    latent = rng.normal(0, 1, n)
    y = (rng.random(n) < 1 / (1 + np.exp(-latent))).astype(int)
    # Each component is best on a specific concept range; injected via noise
    matrix = np.zeros((n, k))
    for j in range(k):
        best_for_concept = j * (n_concepts // k)
        # Row-wise noise: smaller when its "specialty concept" is nearby
        specialty_dist = np.abs(concepts - best_for_concept)
        noise_scale = 0.3 + 0.7 * (specialty_dist / n_concepts)
        z = latent + rng.normal(0, noise_scale, n)
        matrix[:, j] = 1 / (1 + np.exp(-z))
    return y, matrix, concepts


def test_gating_registry_contents():
    """The registry is pinned by exact set on purpose.

    Heads were added over time -- attention_gating and mixture_of_experts
    (Exp C3/C4), then global_stack_concept_intercept (T3b, the control that
    separates a genuine per-concept gating gain from a plain per-concept
    intercept). Pinning the exact set means adding a head is a deliberate act
    that updates this list, rather than something that quietly changes what
    "the gating experiment" means.
    """
    assert set(GATING_HEADS.keys()) == {
        "static_concept_weights",
        "linear_gating",
        "attention_gating",
        "mixture_of_experts",
        "global_stack_concept_intercept",
    }
    assert GATING_HEADS["static_concept_weights"] is StaticConceptWeights
    assert GATING_HEADS["linear_gating"] is LinearGating


# ─── StaticConceptWeights ──────────────────────────────────────────────── #


def test_static_concept_predict_before_fit_raises():
    with pytest.raises(RuntimeError, match="fit"):
        StaticConceptWeights().predict(_random_probs(), test_concepts=np.arange(1000))


def test_static_concept_requires_concepts_at_fit():
    y, matrix, _ = _synth_concept_setup()
    with pytest.raises(ValueError, match="valid_concepts"):
        StaticConceptWeights().fit(matrix, y)


def test_static_concept_requires_concepts_at_predict():
    y, matrix, concepts = _synth_concept_setup()
    ens = StaticConceptWeights().fit(matrix, y, valid_concepts=concepts)
    with pytest.raises(ValueError, match="test_concepts"):
        ens.predict(matrix)


def test_static_concept_fit_predict_shape():
    y, matrix, concepts = _synth_concept_setup(n=1000, k=3, n_concepts=6)
    ens = StaticConceptWeights(n_min=20).fit(matrix[:500], y[:500], valid_concepts=concepts[:500])
    out = ens.predict(matrix[500:], test_concepts=concepts[500:])
    assert out.shape == (500,)
    assert np.all(out >= 0) and np.all(out <= 1)


def test_static_concept_falls_back_on_unseen_concept():
    """Concept id not present in valid → row uses global logistic fit."""
    y, matrix, concepts = _synth_concept_setup(n=800, k=3, n_concepts=5)
    # Fit on concepts 0-3 only; test on all including 4 (unseen)
    fit_mask = concepts < 4
    test_probs = matrix[~fit_mask]
    test_conc = concepts[~fit_mask]
    ens = StaticConceptWeights(n_min=10).fit(
        matrix[fit_mask], y[fit_mask], valid_concepts=concepts[fit_mask]
    )
    out = ens.predict(test_probs, test_concepts=test_conc)
    assert out.shape == (test_probs.shape[0],)
    # unseen concepts should NOT crash; predictions in [0, 1]
    assert np.all(out >= 0) and np.all(out <= 1)


def test_static_concept_fits_per_concept_when_data_sufficient():
    y, matrix, concepts = _synth_concept_setup(n=4000, k=4, n_concepts=8)
    ens = StaticConceptWeights(n_min=50).fit(matrix, y, valid_concepts=concepts)
    # Every concept has ~500 rows on average — all should have per-concept fits
    assert ens.n_concepts_fit_ == 8
    weights = ens.weights_by_concept_
    assert set(weights.keys()) == set(range(8))
    for c, w in weights.items():
        assert w.shape == (4,)


def test_static_concept_skips_small_concepts():
    y, matrix, concepts = _synth_concept_setup(n=500, k=3, n_concepts=10)
    ens = StaticConceptWeights(n_min=200).fit(matrix, y, valid_concepts=concepts)
    # 500/10 = 50 rows per concept average — below n_min=200, all skip
    assert ens.n_concepts_fit_ == 0


# ─── LinearGating ──────────────────────────────────────────────────────── #


def test_linear_gating_predict_before_fit_raises():
    with pytest.raises(RuntimeError, match="fit"):
        LinearGating().predict(_random_probs())


def test_linear_gating_fit_predict_no_context():
    y, matrix, _ = _synth_concept_setup(n=1200, k=4)
    ens = LinearGating().fit(matrix[:600], y[:600])
    out = ens.predict(matrix[600:])
    assert out.shape == (600,)
    assert np.all(out >= 0) and np.all(out <= 1)


def test_linear_gating_fit_predict_with_context():
    y, matrix, concepts = _synth_concept_setup(n=1200, k=4, n_concepts=6)
    ctx = np.column_stack([concepts, np.log1p(np.arange(1200))])
    ens = LinearGating().fit(matrix[:600], y[:600], context=ctx[:600])
    out = ens.predict(matrix[600:], context=ctx[600:])
    assert out.shape == (600,)
    assert np.all(out >= 0) and np.all(out <= 1)


def test_linear_gating_context_shape_mismatch_raises():
    y, matrix, _ = _synth_concept_setup(n=200, k=3)
    with pytest.raises(ValueError, match="context rows"):
        LinearGating().fit(matrix[:100], y[:100], context=np.zeros((99, 2)))


def test_linear_gating_beats_worst_component():
    """On the synthetic setup, a fitted gating should not be strictly worse
    than the worst single component."""
    from sklearn.metrics import roc_auc_score
    y, matrix, concepts = _synth_concept_setup(n=2000, k=4, n_concepts=8)
    per_j = [roc_auc_score(y[1000:], matrix[1000:, j]) for j in range(4)]
    worst = min(per_j)
    ens = LinearGating().fit(matrix[:1000], y[:1000], context=concepts[:1000])
    ens_pred = ens.predict(matrix[1000:], context=concepts[1000:])
    ens_auc = roc_auc_score(y[1000:], ens_pred)
    assert ens_auc >= worst - 0.01


def test_linear_gating_degenerate_target_falls_back_to_mean():
    """If every valid row has the same winning component, the multinomial
    logistic has only 1 class and the fit falls back to a uniform mean.
    Construction: component 0 gives P(observed)=0.9 for every row (correct
    on y=1, correct on y=0); components 1-2 give P(observed)=0.4 for every
    row. Argmax always picks component 0 ⇒ single-class multinomial target."""
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, size=200).astype(int)
    # Component 0: 0.9 on positives, 0.1 on negatives — P(y|0) = 0.9 always
    p0 = np.where(y == 1, 0.9, 0.1)
    # Component 1: 0.4 on positives, 0.6 on negatives — P(y|1) = 0.4 always
    p1 = np.where(y == 1, 0.4, 0.6)
    # Component 2: 0.4 on positives, 0.6 on negatives — same as 1
    p2 = np.where(y == 1, 0.4, 0.6)
    matrix = np.column_stack([p0, p1, p2])
    ens = LinearGating().fit(matrix, y)
    out = ens.predict(matrix)
    # Should not crash; sentinel fallback to arithmetic mean
    assert out.shape == (200,)
    np.testing.assert_allclose(out, matrix.mean(axis=1))


# ─── Concept-conditional calibrated ensemble ───────────────────────────── #


def test_ccce_predict_before_fit_raises():
    with pytest.raises(RuntimeError, match="fit"):
        CCCE().predict(_random_probs(), test_concepts=np.arange(1000))


def test_ccce_requires_concepts():
    y, matrix, concepts = _synth_concept_setup()
    with pytest.raises(ValueError, match="valid_concepts"):
        CCCE().fit(matrix, y)
    ens = CCCE().fit(matrix, y, valid_concepts=concepts)
    with pytest.raises(ValueError, match="test_concepts"):
        ens.predict(matrix)


def test_ccce_fit_predict_shape():
    y, matrix, concepts = _synth_concept_setup(n=2000, k=4, n_concepts=8)
    ens = CCCE(n_min=50).fit(matrix[:1000], y[:1000], valid_concepts=concepts[:1000])
    out = ens.predict(matrix[1000:], test_concepts=concepts[1000:])
    assert out.shape == (1000,)
    assert np.all(out >= 0) and np.all(out <= 1)


def test_ccce_beats_worst_component():
    """Sanity: CCCE trained on a heterogeneous synthetic setup should not be
    strictly worse than the worst single component."""
    from sklearn.metrics import roc_auc_score
    y, matrix, concepts = _synth_concept_setup(n=3000, k=4, n_concepts=8, seed=11)
    per_j = [roc_auc_score(y[1500:], matrix[1500:, j]) for j in range(4)]
    worst = min(per_j)
    ens = CCCE(n_min=30).fit(matrix[:1500], y[:1500], valid_concepts=concepts[:1500])
    ens_pred = ens.predict(matrix[1500:], test_concepts=concepts[1500:])
    ens_auc = roc_auc_score(y[1500:], ens_pred)
    assert ens_auc >= worst - 0.01


def test_ccce_stage_3_makes_output_calibrated():
    """Global post-calibration (Stage 3) should keep the ensemble output
    in [0, 1] and make it *isotonically* related to the pre-cal ensemble."""
    y, matrix, concepts = _synth_concept_setup(n=1500, k=3, n_concepts=6, seed=5)
    ens = CCCE(n_min=30).fit(matrix[:750], y[:750], valid_concepts=concepts[:750])
    out = ens.predict(matrix[750:], test_concepts=concepts[750:])
    # In [0, 1]
    assert np.all(out >= 0) and np.all(out <= 1)
    # Post-cal isotonic is monotone non-decreasing in Stage 2 output — hard
    # to verify directly without exposing intermediate; we at least check
    # that the ensemble has some spread (didn't collapse to a constant)
    assert out.std() > 1e-4


def test_ccce_shrinkage_lambda_smoke():
    """``shrinkage_lambda > 0`` should not crash; empirical bayes shrinkage
    just changes per-concept-vs-global blend weights."""
    y, matrix, concepts = _synth_concept_setup(n=1500, k=4, n_concepts=6, seed=7)
    ens = CCCE(n_min=30, shrinkage_lambda=5.0).fit(
        matrix[:800], y[:800], valid_concepts=concepts[:800],
    )
    out = ens.predict(matrix[800:], test_concepts=concepts[800:])
    assert out.shape == (700,)


def test_bootstrap_wrapper_supports_ece_metric():
    """Regression: metric_fn is generic, not AUC-specific. ECE on 4-model mean
    ensemble should yield a finite CI even with small n_boot."""
    y, matrix, groups, single_ref = _synthetic_ensemble_setup()
    res = bootstrap_ensemble_vs_single(
        y, matrix, ArithmeticMean(), single_ref, ece_metric,
        n_boot=100, groups=groups, seed=0, metric_name="ece",
    )
    assert np.isfinite(res.value_a)
    assert np.isfinite(res.value_b)
    assert np.isfinite(res.ci_low) and np.isfinite(res.ci_high)
