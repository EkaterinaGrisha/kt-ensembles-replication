"""Heterogeneous ensembles for KT predictions.

Given per-model probability vectors on an aligned test set, aggregate them into
a single ensemble prediction. The Phase-3 hypothesis is that no single KT model
dominates AUC + ECE + downstream-F1 across datasets,
so a heterogeneous ensemble should Pareto-dominate any single model.

Public surface:

- ``EnsembleBase``               — abstract aggregator.
- ``ArithmeticMean``             — plain average of probabilities.
- ``GeometricMean``              — n-th root of the product (equivalent to
  averaging log-probabilities of the positive class after normalisation).
- ``LogitMean``                  — average of logits, sigmoid back. Better
  behaved when components are individually well-calibrated.
- ``RankMean``                   — average of within-model quantile ranks,
  discarding absolute scale. Robust to a single miscalibrated component.
- ``MedianAggregator``           — per-row median. Robust to outlier models.
- ``SIMPLE_AGGREGATORS``         — registry keyed by short name.

Input convention: ``probs`` is a ``(n_samples, n_models)`` array of
probabilities in [0, 1]. Every aggregator returns a ``(n_samples,)`` vector.
Parameter-free aggregators (all of the above) have a no-op ``fit`` so they
share the ``EnsembleBase`` interface with the learned blenders that arrive in
``StackedBlender`` and ``GatingNetwork``.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

_EPS = 1e-12


def _prep(probs) -> np.ndarray:
    """Validate + clip to (0, 1). ``probs`` must be 2-D (n_samples, n_models)."""
    p = np.asarray(probs, dtype=np.float64)
    if p.ndim != 2:
        raise ValueError(f"probs must be 2-D (n_samples, n_models); got shape {p.shape}")
    if p.shape[1] < 2:
        raise ValueError(f"ensemble needs >= 2 component models; got n_models={p.shape[1]}")
    return np.clip(p, _EPS, 1.0 - _EPS)


def _logit(p: np.ndarray) -> np.ndarray:
    return np.log(p / (1.0 - p))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


class EnsembleBase(ABC):
    """Interface shared by simple, stacked, and gating ensembles.

    ``fit`` sees validation-fold predictions (with labels); simple aggregators
    ignore both arguments. ``predict`` maps a per-model probability matrix on the
    test fold to a single ensemble probability vector.
    """

    name: str = "ensemble_base"

    def fit(self, valid_probs, valid_labels) -> "EnsembleBase":  # noqa: ARG002
        return self

    @abstractmethod
    def predict(self, test_probs) -> np.ndarray:
        ...

    def fit_predict(self, valid_probs, valid_labels, test_probs) -> np.ndarray:
        return self.fit(valid_probs, valid_labels).predict(test_probs)


class ArithmeticMean(EnsembleBase):
    """Plain per-row mean over the model axis.

    Baseline aggregator. Robust to nothing in particular but always well-behaved:
    the ensemble probability is a convex combination of component probabilities
    so it stays in [0, 1] without clipping.
    """

    name = "arithmetic_mean"

    def predict(self, test_probs) -> np.ndarray:
        p = _prep(test_probs)
        return p.mean(axis=1)


class GeometricMean(EnsembleBase):
    """Per-row geometric mean over the model axis.

    ``exp(mean(log(p)))``. Amplifies agreement on low probabilities and is more
    conservative than arithmetic mean when models disagree (drags toward the
    lower prediction). Used as a sanity check that arithmetic-mean gains are not
    an artefact of the aggregation choice.
    """

    name = "geometric_mean"

    def predict(self, test_probs) -> np.ndarray:
        p = _prep(test_probs)
        return np.exp(np.log(p).mean(axis=1))


class LogitMean(EnsembleBase):
    """Per-row mean of logits, sigmoid back.

    Equivalent to product-of-odds averaging: ``sigmoid(mean(logit(p)))``. This
    is the aggregator you want when components are individually calibrated —
    logit-space is where miscalibration is additive, so averaging there is
    principled. Sensitive to a single badly-calibrated component, hence the
    The three-stage pipeline pre-calibrates components before this aggregation.
    """

    name = "logit_mean"

    def predict(self, test_probs) -> np.ndarray:
        p = _prep(test_probs)
        z = _logit(p).mean(axis=1)
        return _sigmoid(z)


class RankMean(EnsembleBase):
    """Per-model rank-transform followed by arithmetic mean.

    Each column is replaced by its within-column quantile (values in [0, 1]),
    then averaged across models. Discards absolute probability scale, so the
    output is a *ranking score* rather than a calibrated probability — do NOT
    feed it to calibration metrics without a re-calibration step. Included
    because it is the most robust simple aggregator to a single component with
    a pathological scale (e.g. logits clipped near 0/1).
    """

    name = "rank_mean"

    def predict(self, test_probs) -> np.ndarray:
        p = _prep(test_probs)
        n, k = p.shape
        # Per-model average rank in [1, n], then to [0, 1) via /n
        ranks = np.empty_like(p)
        for j in range(k):
            order = np.argsort(p[:, j])
            r = np.empty(n, dtype=np.float64)
            r[order] = np.arange(1, n + 1, dtype=np.float64)
            ranks[:, j] = r / n
        return ranks.mean(axis=1)


class MedianAggregator(EnsembleBase):
    """Per-row median across models.

    Breaks ties on even ``n_models`` by averaging the two middle values (numpy
    default). Robust to a single-model outlier but throws away information from
    the extreme components even when they are correct — reported as a robustness
    complement to ArithmeticMean, not a headline aggregator.
    """

    name = "median"

    def predict(self, test_probs) -> np.ndarray:
        p = _prep(test_probs)
        return np.median(p, axis=1)


SIMPLE_AGGREGATORS: dict[str, type[EnsembleBase]] = {
    "arithmetic_mean": ArithmeticMean,
    "geometric_mean": GeometricMean,
    "logit_mean": LogitMean,
    "rank_mean": RankMean,
    "median": MedianAggregator,
}


# ─── Stacked blenders ────────────────────────────────────────────────────── #
#
# A stacked blender fits a *meta-learner* on validation-fold predictions of the
# k component models and applies that meta-learner to test-fold predictions.
# The first skeleton included only the simplest concrete meta-learner
# (logistic regression on component logits, Wolpert 1992); XGBoost / MLP /
# Bayesian Model Averaging arrived later, once the plumbing
# has passed end-to-end validation on the full 7-dataset matrix.


class LogisticStackedBlender(EnsembleBase):
    """Logistic-regression blender on the k component logits (Wolpert 1992).

    The meta-learner is a scikit-learn ``LogisticRegression`` fitted on
    ``logit(valid_probs)`` (n_valid rows, k features) against ``valid_labels``.
    At predict time the same logit-transform is applied to the test-fold
    matrix. Regularisation defaults to L2 with C=1.0; pass ``C`` in the
    constructor to sweep.

    The blender preserves ``EnsembleBase``'s ``predict(probs)`` contract, so
    it slots into ``bootstrap_ensemble_vs_single`` unchanged: pass
    ``fit_probs=valid_matrix, fit_labels=valid_labels`` to have the wrapper
    call ``.fit`` before it precomputes the test prediction.
    """

    name = "logistic_stacked"

    def __init__(self, C: float = 1.0) -> None:
        from sklearn.linear_model import LogisticRegression
        self._lr = LogisticRegression(C=C, solver="lbfgs", max_iter=200)
        self._fitted = False

    def fit(self, valid_probs, valid_labels) -> "LogisticStackedBlender":
        p = _prep(valid_probs)
        z = _logit(p)
        y = np.asarray(valid_labels).astype(int).ravel()
        if z.shape[0] != y.size:
            raise ValueError(f"valid_probs rows ({z.shape[0]}) != valid_labels ({y.size})")
        self._lr.fit(z, y)
        self._fitted = True
        return self

    def predict(self, test_probs) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("call fit(valid_probs, valid_labels) before predict")
        z = _logit(_prep(test_probs))
        return self._lr.predict_proba(z)[:, 1]

    @property
    def weights_(self) -> np.ndarray:
        """Learned logistic weights per component (k-vector)."""
        if not self._fitted:
            raise RuntimeError("fit() first")
        return self._lr.coef_.ravel()

    @property
    def intercept_(self) -> float:
        if not self._fitted:
            raise RuntimeError("fit() first")
        return float(self._lr.intercept_[0])


class RidgeStackedBlender(EnsembleBase):
    """L2-regularised logistic blender with an inner CV sweep over the
    regularisation strength.

    Same core as ``LogisticStackedBlender`` (logistic on component logits)
    but the regularisation constant ``C`` is chosen from ``C_grid`` by
    inner K-fold cross-validation (``StratifiedKFold``) maximising valid
    AUC. Rationale: Wolpert 1992 logistic stacking is known to overfit
    with flat unregularised likelihood when component predictions are
    highly correlated (all deep KT models capture the same latent), so
    a proper ridge sweep is the honest baseline before the
    XGBoost / MLP steps arrive.
    """

    name = "ridge_stacked"

    def __init__(self, C_grid: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0),
                 cv: int = 3) -> None:
        self.C_grid = tuple(C_grid)
        self.cv = int(cv)
        self._lr = None
        self.C_ = None
        self._fitted = False

    def fit(self, valid_probs, valid_labels) -> "RidgeStackedBlender":
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import StratifiedKFold

        z = _logit(_prep(valid_probs))
        y = np.asarray(valid_labels).astype(int).ravel()
        if z.shape[0] != y.size:
            raise ValueError(f"valid_probs rows ({z.shape[0]}) != valid_labels ({y.size})")

        best_C, best_score = None, -np.inf
        # Degenerate case: too few samples for CV, fall back to middle grid point
        if y.size < 2 * self.cv or np.unique(y).size < 2:
            best_C = self.C_grid[len(self.C_grid) // 2]
        else:
            skf = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=0)
            for C in self.C_grid:
                aucs = []
                for tr, va in skf.split(z, y):
                    if np.unique(y[va]).size < 2:
                        continue
                    clf = LogisticRegression(C=C, solver="lbfgs", max_iter=200)
                    clf.fit(z[tr], y[tr])
                    aucs.append(roc_auc_score(y[va], clf.predict_proba(z[va])[:, 1]))
                if aucs:
                    m = float(np.mean(aucs))
                    if m > best_score:
                        best_score, best_C = m, C
            if best_C is None:
                best_C = self.C_grid[len(self.C_grid) // 2]

        self._lr = LogisticRegression(C=best_C, solver="lbfgs", max_iter=200)
        self._lr.fit(z, y)
        self.C_ = float(best_C)
        self._fitted = True
        return self

    def predict(self, test_probs) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("call fit(valid_probs, valid_labels) before predict")
        z = _logit(_prep(test_probs))
        return self._lr.predict_proba(z)[:, 1]

    @property
    def weights_(self) -> np.ndarray:
        return self._lr.coef_.ravel()

    @property
    def intercept_(self) -> float:
        return float(self._lr.intercept_[0])


class BayesianModelAveraging(EnsembleBase):
    """Bayesian model averaging via held-out log-likelihood posterior.

    Given the k component probabilities on the valid fold, computes the
    total log-likelihood of each component and derives posterior weights
    via ``softmax(log L_m / T) = softmax(-n · NLL_m / T)`` with a flat
    prior and temperature ``T`` (default 1.0 — the strict BMA weight;
    larger T flattens the posterior toward uniform averaging). Predictions
    are the convex combination ``Σ w_m · p_m``.

    **Fix.** Prior to 2026-08-24, this class computed
    weights from the *mean* NLL — softmax(-mean_NLL / T) — which drops the
    n factor and turns the posterior weights into a nearly uniform softmax
    over per-example log-likelihood differences that are O(0.001-0.05) in
    KT. Empirically that made BMA numerically equal to arithmetic-mean
    (max |ΔAUC(BMA - arithmetic_mean)| = 0.0011 across all 14 ensemble
    cells; median 7e-5), and the paper's headline "BMA strictly worst 14/14"
    was an implementation artifact rather than a property of Bayesian model
    selection. With the n factor, at n ≈ 10^5 valid rows the softmax
    collapses to one-hot on the min-NLL component — recovering the
    canonical Bayesian large-sample behavior. See unit test
    tests/test_bma_scaling.py.

    Interpretability angle: weights are
    a principled posterior over models — the closest we come to a mathematical
    statistics "model averaging" story (Hoeting et al.
    1999). Complements the XGBoost / MLP performance-driven baselines with
    something that ports directly to Bayesian model-selection language.
    """

    name = "bma"

    def __init__(self, temperature: float = 1.0) -> None:
        self.temperature = float(temperature)
        self.weights_: np.ndarray | None = None
        self.nlls_: np.ndarray | None = None
        self._fitted = False

    def fit(self, valid_probs, valid_labels) -> "BayesianModelAveraging":
        p = _prep(valid_probs)
        y = np.asarray(valid_labels).astype(np.float64).ravel()
        n = y.size
        if p.shape[0] != n:
            raise ValueError(f"valid_probs rows ({p.shape[0]}) != valid_labels ({n})")
        # Per-component mean NLL (kept for backwards-compat inspection)
        log_p = np.log(p)
        log_1mp = np.log(1.0 - p)
        nlls = -(y[:, None] * log_p + (1.0 - y[:, None]) * log_1mp).mean(axis=0)
        # total log-likelihood posterior (flat prior):
        #   log_evidence_m ≈ log L_m = Σ_i [y_i·log p_{m,i} + (1-y_i)·log(1-p_{m,i})]
        #                 = -n · mean_NLL_m
        # w_m ∝ exp(log_evidence_m / T) = exp(-n · nll_m / T)
        # Subtract max for numerical stability (softmax invariance).
        z = -n * nlls / self.temperature
        z = z - z.max()
        w = np.exp(z)
        self.weights_ = w / w.sum()
        self.nlls_ = nlls
        self._fitted = True
        return self

    def predict(self, test_probs) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("call fit(valid_probs, valid_labels) before predict")
        p = _prep(test_probs)
        if p.shape[1] != self.weights_.size:
            raise ValueError(f"test_probs has {p.shape[1]} components; "
                              f"fit was done with {self.weights_.size}")
        return p @ self.weights_


class XGBoostStackedBlender(EnsembleBase):
    """Gradient-boosted stacker on component logits.

    Uses ``xgboost.XGBClassifier`` with binary logistic loss. Hyperparameters
    are held to sensible defaults for the KT feature scale — the meta-learner
    is fit on ~50k-160k validation rows depending on dataset, k=4 features,
    so a shallow (max_depth=3) modest-tree (n_estimators=200) config is
    both fast and does not overfit. Rq3_plan.md §3 Exp B3 notes XGBoost is
    the "how much can we squeeze?" upper bound candidate — the tuning grid
    lives in separate sweeps, this class ships the honest default.

    Handles augmented context features via the optional ``extra_features``
    argument to ``fit`` / ``predict`` (2-D array aligned row-wise with the
    prob matrices) — used in the concept-embedding + history-length
    experiments; leave ``None`` for the first-pass logit-only stacker.
    """

    name = "xgb_stacked"

    def __init__(self, max_depth: int = 3, n_estimators: int = 200,
                 learning_rate: float = 0.1, subsample: float = 0.9,
                 random_state: int = 0, n_jobs: int = 1) -> None:
        from xgboost import XGBClassifier
        # ``n_jobs=1`` is critical on macOS + sklearn: xgboost's libomp and
        # sklearn's libomp can collide (OMP error #179 pthread_mutex_init)
        # when both are loaded in the same process. Forcing single-thread
        # sidesteps the OpenMP init clash; the meta-learner runs on ~50-500k
        # rows per cell and does not need parallelism at that scale.
        self._clf = XGBClassifier(
            max_depth=max_depth, n_estimators=n_estimators,
            learning_rate=learning_rate, subsample=subsample,
            objective="binary:logistic", tree_method="hist",
            random_state=random_state, verbosity=0, n_jobs=n_jobs,
        )
        self._fitted = False

    @staticmethod
    def _features(probs, extra_features):
        z = _logit(_prep(probs))
        if extra_features is None:
            return z
        ef = np.asarray(extra_features, dtype=np.float64)
        if ef.ndim == 1:
            ef = ef.reshape(-1, 1)
        if ef.shape[0] != z.shape[0]:
            raise ValueError(f"extra_features rows ({ef.shape[0]}) != "
                              f"probs rows ({z.shape[0]})")
        return np.column_stack([z, ef])

    def fit(self, valid_probs, valid_labels, extra_features=None
             ) -> "XGBoostStackedBlender":
        x = self._features(valid_probs, extra_features)
        y = np.asarray(valid_labels).astype(int).ravel()
        self._clf.fit(x, y)
        self._fitted = True
        return self

    def predict(self, test_probs, extra_features=None) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("call fit(valid_probs, valid_labels) before predict")
        x = self._features(test_probs, extra_features)
        return self._clf.predict_proba(x)[:, 1]

    @property
    def feature_importances_(self) -> np.ndarray:
        return self._clf.feature_importances_


class MLPStackedBlender(EnsembleBase):
    """Shallow MLP meta-learner (sklearn ``MLPClassifier``).

    Default: single hidden layer of 32 ReLU units, ``adam`` optimiser,
    ``early_stopping=True`` (holds out 10% of valid for early stop).
    Small enough to fit in seconds per cell on the 50-160k valid rows,
    large enough to capture nonlinear model-mixing patterns that ridge
    logistic misses.
    """

    name = "mlp_stacked"

    def __init__(self, hidden_layer_sizes: tuple[int, ...] = (32,),
                 max_iter: int = 200, random_state: int = 0,
                 early_stopping: bool = True) -> None:
        from sklearn.neural_network import MLPClassifier
        self._clf = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            max_iter=max_iter, random_state=random_state,
            early_stopping=early_stopping,
        )
        self._fitted = False

    def fit(self, valid_probs, valid_labels) -> "MLPStackedBlender":
        z = _logit(_prep(valid_probs))
        y = np.asarray(valid_labels).astype(int).ravel()
        self._clf.fit(z, y)
        self._fitted = True
        return self

    def predict(self, test_probs) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("call fit(valid_probs, valid_labels) before predict")
        z = _logit(_prep(test_probs))
        return self._clf.predict_proba(z)[:, 1]


STACKED_BLENDERS: dict[str, type[EnsembleBase]] = {
    "logistic_stacked": LogisticStackedBlender,
    "ridge_stacked":    RidgeStackedBlender,
    "bma":              BayesianModelAveraging,
    "xgb_stacked":      XGBoostStackedBlender,
    "mlp_stacked":      MLPStackedBlender,
}


# ─── Conditional gating ─────────────────────────────────────────────────── #
#
# Gating meta-learners predict per-row weights over the k components as a
# function of a **context** — concept-id, student-history-length, item
# difficulty, response time, etc. — instead of a single global weight
# vector. The hypothesis is that conditional weights
# recover the negative-headline cells where naive/static meta-learners
# fail (e.g. ednet deep, where simpleKT dominates by 0.10 AUC and any
# static blend adds noise).
#
# API divergence from the stacked blenders: ``predict`` takes an extra
# ``context`` argument (2-D array of features per test row). ``fit``
# likewise takes valid-side context.
#
# Fallback for concepts / contexts not seen in the valid fold: the gating
# reduces to a global mean over components (equivalent to
# ArithmeticMean). Documented per-class.


class StaticConceptWeights(EnsembleBase):
    """Per-concept static weight vector, fit by logistic regression on the
    subset of valid rows for that concept.

    Method.
      For each concept c with n_c >= n_min valid observations, fit a
      logistic regression on the k component logits restricted to those
      rows; keep the coefficient vector as w_c. Concepts with fewer than
      n_min observations, or unseen at test time, fall back to a single
      global logistic fit on ALL valid rows (equivalent to the standard
      ``LogisticStackedBlender``).

    Interpretability angle. ``weights_by_concept_[c]`` returns the fitted
    logit-space coefficients for concept c — a direct "when does each
    component dominate" table. On multi-KC datasets (algebra2005,
    bridge2algebra2006), the leading concept per event is used.

    API. ``fit(valid_probs, valid_labels, valid_concepts=...)`` and
    ``predict(test_probs, test_concepts=...)`` — the ``*_concepts``
    argument is a 1-D int array of the same length as the rows.
    ``fit_predict`` accepts both.
    """

    name = "static_concept_weights"

    def __init__(self, n_min: int = 30) -> None:
        self.n_min = int(n_min)
        self._global = LogisticStackedBlender()
        self._per_concept: dict[int, LogisticStackedBlender] = {}
        self._n_c: dict[int, int] = {}
        self._fitted = False

    def fit(self, valid_probs, valid_labels, valid_concepts=None
             ) -> "StaticConceptWeights":  # type: ignore[override]
        if valid_concepts is None:
            raise ValueError("StaticConceptWeights.fit requires valid_concepts")
        p = _prep(valid_probs)
        y = np.asarray(valid_labels).astype(int).ravel()
        c = np.asarray(valid_concepts).astype(np.int64).ravel()
        if not (p.shape[0] == y.size == c.size):
            raise ValueError(f"valid_probs/labels/concepts length mismatch: "
                              f"{p.shape[0]} / {y.size} / {c.size}")

        # Global fallback fit (used for cold concepts + unseen at test time)
        self._global.fit(p, y)

        self._per_concept.clear()
        self._n_c.clear()
        for concept in np.unique(c):
            mask = c == concept
            n = int(mask.sum())
            self._n_c[int(concept)] = n
            if n < self.n_min:
                continue
            y_c = y[mask]
            if np.unique(y_c).size < 2:
                continue  # single-class subset ⇒ logistic degenerates
            try:
                lr = LogisticStackedBlender().fit(p[mask], y_c)
                self._per_concept[int(concept)] = lr
            except Exception:
                continue  # ill-conditioned per-concept fit ⇒ silent fall-back
        self._fitted = True
        return self

    def predict(self, test_probs, test_concepts=None) -> np.ndarray:  # type: ignore[override]
        if not self._fitted:
            raise RuntimeError("call fit(valid_probs, valid_labels, valid_concepts) before predict")
        if test_concepts is None:
            raise ValueError("StaticConceptWeights.predict requires test_concepts")
        p = _prep(test_probs)
        c = np.asarray(test_concepts).astype(np.int64).ravel()
        if p.shape[0] != c.size:
            raise ValueError(f"test_probs rows ({p.shape[0]}) != test_concepts ({c.size})")

        # Start with global prediction; overwrite per-concept where a fit exists.
        out = self._global.predict(p)
        for concept, lr in self._per_concept.items():
            mask = c == concept
            if mask.any():
                out[mask] = lr.predict(p[mask])
        return out

    def fit_predict(self, valid_probs, valid_labels,  # type: ignore[override]
                     test_probs, valid_concepts=None, test_concepts=None) -> np.ndarray:
        return self.fit(valid_probs, valid_labels, valid_concepts) \
                   .predict(test_probs, test_concepts)

    @property
    def weights_by_concept_(self) -> dict[int, np.ndarray]:
        """Learned per-concept weight vectors (logit-space). Only includes
        concepts with n_c >= n_min AND both classes present in valid."""
        return {c: lr.weights_ for c, lr in self._per_concept.items()}

    @property
    def n_concepts_fit_(self) -> int:
        return len(self._per_concept)


class GlobalStackWithConceptIntercept(EnsembleBase):
    """Global logistic stacker + per-concept free intercept — the ablation.

    ``StaticConceptWeights`` fits an independent
    logistic regression per concept, so its per-cell gains combine two
    effects: (a) a per-concept baseline shift (some concepts are easier
    than average — the intercept moves), and (b) per-concept variation
    in component weighting (some concepts favor simpleKT more than the
    global weights would). If the pooled-AUC gain over the global
    ``LogisticStackedBlender`` is dominated by (a), then the "conditional
    gating recovers ednet" claim is actually about per-
    concept base-rate calibration and not about component gating at all.

    This class isolates (a) alone: **shared component weights** across all
    concepts + a per-concept free intercept. Concretely, features fed to
    a single L2-regularized ``sklearn.LogisticRegression`` are:

      [ logit(p_1), ..., logit(p_k), one_hot(concept)_1, ..., one_hot(concept)_C ]

    The first k coefficients are the global component weights (same for
    every row); the last C coefficients are per-concept dummy intercepts.
    Concepts unseen at test time fall back to the "reference concept" whose
    dummy is dropped (equivalent to the base logistic intercept). Concepts
    with n_c < n_min in valid are collapsed into the reference to prevent
    overfitting on cold slices.

    Contract:
      * If pooled AUC(StaticConceptWeights) − AUC(GlobalStackWithConceptIntercept)
        is not significant (paired cluster bootstrap), the earlier gating
        finding reduces to per-concept intercept calibration.
      * If the gap is significant, ``StaticConceptWeights`` genuinely uses
        concept-conditional component weights beyond what per-concept
        intercepts can express, and the gating finding is real.

    API mirrors ``StaticConceptWeights``.
    """

    name = "global_stack_concept_intercept"

    def __init__(self, n_min: int = 30, C_reg: float = 1e6) -> None:
        self.n_min = int(n_min)
        self.C_reg = float(C_reg)  # near-unregularised, like LogisticStackedBlender
        # populated at fit()
        self._concepts_kept: dict[int, int] = {}   # concept_id -> column index in one-hot
        self._ref_concept: int | None = None       # reference concept (dummy dropped)
        self._lr = None                            # sklearn LogisticRegression
        self._k: int | None = None                 # number of components
        self._fitted = False

    def _build_features(self, p: np.ndarray, c: np.ndarray) -> np.ndarray:
        """Concatenate [logit(p) | one_hot(concept)] with the reference concept
        dropped. Unseen or below-n_min concepts get all-zero dummies (fallback
        to reference)."""
        z = _logit(p)
        n = z.shape[0]
        C = len(self._concepts_kept)
        if C == 0:
            return z  # degenerate: just the logits
        dummies = np.zeros((n, C), dtype=np.float64)
        for i, cid in enumerate(c):
            j = self._concepts_kept.get(int(cid))
            if j is not None:
                dummies[i, j] = 1.0
        return np.column_stack([z, dummies])

    def fit(self, valid_probs, valid_labels, valid_concepts=None
             ) -> "GlobalStackWithConceptIntercept":  # type: ignore[override]
        if valid_concepts is None:
            raise ValueError(
                "GlobalStackWithConceptIntercept.fit requires valid_concepts")
        p = _prep(valid_probs)
        y = np.asarray(valid_labels).astype(int).ravel()
        c = np.asarray(valid_concepts).astype(np.int64).ravel()
        if not (p.shape[0] == y.size == c.size):
            raise ValueError(
                f"valid_probs/labels/concepts length mismatch: "
                f"{p.shape[0]} / {y.size} / {c.size}")
        self._k = int(p.shape[1])

        # Determine which concepts get a dedicated dummy: n_c >= n_min AND
        # both classes present. The concept with the largest n_c becomes the
        # reference (dummy dropped) — most stable base intercept.
        counts: dict[int, int] = {}
        for cid in np.unique(c):
            n_c = int((c == cid).sum())
            y_c = y[c == cid]
            if n_c >= self.n_min and np.unique(y_c).size == 2:
                counts[int(cid)] = n_c
        if not counts:
            # No usable concepts — behaves like a plain logistic stacker.
            self._ref_concept = None
            self._concepts_kept = {}
        else:
            self._ref_concept = max(counts, key=counts.get)  # type: ignore[arg-type]
            self._concepts_kept = {
                cid: idx for idx, cid in enumerate(
                    sorted(c for c in counts if c != self._ref_concept))
            }

        X = self._build_features(p, c)
        from sklearn.linear_model import LogisticRegression
        self._lr = LogisticRegression(C=self.C_reg, solver="lbfgs",
                                       max_iter=1000)
        self._lr.fit(X, y)
        self._fitted = True
        return self

    def predict(self, test_probs, test_concepts=None) -> np.ndarray:  # type: ignore[override]
        if not self._fitted:
            raise RuntimeError(
                "call fit(valid_probs, valid_labels, valid_concepts) before predict")
        if test_concepts is None:
            raise ValueError(
                "GlobalStackWithConceptIntercept.predict requires test_concepts")
        p = _prep(test_probs)
        c = np.asarray(test_concepts).astype(np.int64).ravel()
        if p.shape[0] != c.size:
            raise ValueError(
                f"test_probs rows ({p.shape[0]}) != test_concepts ({c.size})")
        X = self._build_features(p, c)
        return self._lr.predict_proba(X)[:, 1]

    def fit_predict(self, valid_probs, valid_labels,  # type: ignore[override]
                     test_probs, valid_concepts=None,
                     test_concepts=None) -> np.ndarray:
        return self.fit(valid_probs, valid_labels, valid_concepts) \
                   .predict(test_probs, test_concepts)

    @property
    def component_weights_(self) -> np.ndarray | None:
        """Shared component-weight vector (logit-space), length k."""
        if not self._fitted or self._lr is None or self._k is None:
            return None
        return np.asarray(self._lr.coef_[0, :self._k]).astype(np.float64)

    @property
    def concept_intercepts_(self) -> dict[int, float]:
        """Fitted per-concept intercept adjustments (logit-space), relative
        to the reference concept whose dummy was dropped."""
        if not self._fitted or self._lr is None or self._k is None:
            return {}
        coefs = np.asarray(self._lr.coef_[0]).astype(np.float64)
        out = {cid: float(coefs[self._k + j])
               for cid, j in self._concepts_kept.items()}
        if self._ref_concept is not None:
            out[int(self._ref_concept)] = 0.0
        return out


class LinearGating(EnsembleBase):
    """Multinomial-logistic gating on continuous context features.

    Fits ``sklearn.LogisticRegression`` with k ``num_models`` classes: each
    valid row (component_probabilities + context_features) is mapped to a
    per-model "vote" via cross-entropy on the winning-component target.
    At predict, softmax over the k class logits gives the gating weights;
    output probability is Σ w_m(x) · p_m(x).

    The winning component per valid row is defined as ``argmax_m p_m(x)``
    correctness-weighted — i.e. the model with the highest predicted
    probability of the observed label. This is a discriminative gating
    signal (as opposed to BMA's marginal-likelihood signal) and typically
    beats it when component quality is heterogeneous across contexts.

    Fallback. Contexts with unseen concept ids default the corresponding
    embedding row to zero; the linear head still fires. When ``context``
    is all-zero, LinearGating degenerates to a per-model bias vector
    (equivalent to a static Wolpert stacker on the intercept row).
    """

    name = "linear_gating"

    def __init__(self, C: float = 1.0) -> None:
        from sklearn.linear_model import LogisticRegression
        # sklearn >=1.5 auto-picks multinomial when the target has >2 classes;
        # the explicit ``multi_class`` kwarg is deprecated/removed in that
        # release. lbfgs handles both binary and multinomial cases.
        self._clf = LogisticRegression(C=C, solver="lbfgs", max_iter=300)
        self._fitted = False
        self._n_models: int | None = None

    @staticmethod
    def _winning_component(probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Per-row argmax over components of the correct-label probability.

        Concretely: for each row i with label y_i, compute p_m(y_i) for each
        component m, pick argmax_m. Ties broken by lowest m (stable).
        """
        # Prob of the observed class per (row, model): if y=1 use p, else 1-p.
        y = labels.reshape(-1, 1).astype(np.float64)
        p = _prep(probs)
        correct_prob = y * p + (1.0 - y) * (1.0 - p)
        return np.argmax(correct_prob, axis=1)

    def fit(self, valid_probs, valid_labels, context=None) -> "LinearGating":  # type: ignore[override]
        p = _prep(valid_probs)
        y = np.asarray(valid_labels).astype(int).ravel()
        if p.shape[0] != y.size:
            raise ValueError(f"valid_probs rows ({p.shape[0]}) != valid_labels ({y.size})")
        self._n_models = p.shape[1]
        # Features fed to the gating head = [component_logits | context]
        z = _logit(p)
        if context is not None:
            ctx = np.asarray(context, dtype=np.float64)
            if ctx.ndim == 1:
                ctx = ctx.reshape(-1, 1)
            if ctx.shape[0] != z.shape[0]:
                raise ValueError(f"context rows ({ctx.shape[0]}) != probs rows ({z.shape[0]})")
            features = np.column_stack([z, ctx])
        else:
            features = z

        target = self._winning_component(p, y)
        # sklearn LogisticRegression multinomial requires at least 2 classes;
        # if the winning-component target is degenerate (one component always
        # wins), fall back to a global mean over components at predict time.
        if np.unique(target).size < 2:
            self._clf = None  # sentinel: use uniform-mean fallback
        else:
            self._clf.fit(features, target)
        self._fitted = True
        return self

    def predict(self, test_probs, context=None) -> np.ndarray:  # type: ignore[override]
        if not self._fitted:
            raise RuntimeError("call fit(valid_probs, valid_labels, context) before predict")
        p = _prep(test_probs)
        if p.shape[1] != self._n_models:
            raise ValueError(f"test_probs has {p.shape[1]} components; "
                              f"fit was done with {self._n_models}")
        if self._clf is None:
            # Degenerate valid target ⇒ uniform average.
            return p.mean(axis=1)

        z = _logit(p)
        if context is not None:
            ctx = np.asarray(context, dtype=np.float64)
            if ctx.ndim == 1:
                ctx = ctx.reshape(-1, 1)
            if ctx.shape[0] != z.shape[0]:
                raise ValueError(f"context rows ({ctx.shape[0]}) != probs rows ({z.shape[0]})")
            features = np.column_stack([z, ctx])
        else:
            features = z

        # sklearn returns per-class softmax probabilities — these ARE our
        # gating weights (one per component). Convex-combine component
        # probs by the gating.
        gate = self._clf.predict_proba(features)  # (n, n_models)
        # gate columns are in sklearn's classes_ order — align back to
        # our component-index order (0..k-1).
        classes = self._clf.classes_
        full_gate = np.zeros((p.shape[0], self._n_models), dtype=np.float64)
        for j, cls in enumerate(classes):
            full_gate[:, int(cls)] = gate[:, j]
        # Missing components (never won in valid) get zero weight — fine
        # for the convex combination; the row is still normalised over the
        # surviving classes because sklearn softmax already sums to 1.
        return (full_gate * p).sum(axis=1)


class AttentionGating(EnsembleBase):
    """Soft-attention gating over k component models (Exp C3, torch).

    Architecture:
      * **Query**: student-context vector — concatenation of a learnable
        concept-id embedding (dim ``d_emb``) with any scalar context
        features (log-history, item-difficulty, response-time bucket) if
        provided at fit time. Projected to ``d_attn`` via a linear head.
      * **Keys**: k learnable model-embedding vectors of dim ``d_attn``
        (one per component). Values are the corresponding component
        prediction probabilities.
      * **Attention weights**: ``softmax(Q·Kᵀ / √d_attn)`` — shape
        ``(n, k)`` per batch. Entropy regularisation on weights (weight
        ``entropy_lambda``) encourages diverse mixing rather than
        winner-takes-all collapse.
      * **Loss**: BCE on the mixed prediction ``Σ_m w_m(x) · p_m(x)``,
        with early stopping on a held-out 10% split.

    Kaggle GPU is the intended runner (~few min per (dataset, fold)
    with n_valid up to 164k on a T4). Torch is imported lazily so this
    module stays CPU-only-friendly for smoke tests when torch is absent.

    API. Same as ``StaticConceptWeights`` for the concept-id path,
    plus optional ``context`` argument (extra scalar features per row).
    """

    name = "attention_gating"

    def __init__(self, d_emb: int = 16, d_attn: int = 32,
                 n_epochs: int = 15, batch_size: int = 2048,
                 lr: float = 1e-3, entropy_lambda: float = 0.01,
                 device: str = "auto", seed: int = 0) -> None:
        self.d_emb = int(d_emb)
        self.d_attn = int(d_attn)
        self.n_epochs = int(n_epochs)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.entropy_lambda = float(entropy_lambda)
        self.device = device
        self.seed = int(seed)
        self._model = None
        self._n_models: int | None = None
        self._n_concepts: int | None = None
        self._ctx_dim: int = 0
        self._fitted = False

    def _build_model(self, n_models: int, n_concepts: int, ctx_dim: int):
        import torch
        import torch.nn as nn

        class _AttentionHead(nn.Module):
            def __init__(self, d_emb, d_attn, n_models, n_concepts, ctx_dim):
                super().__init__()
                self.concept_emb = nn.Embedding(n_concepts + 1, d_emb)  # +1 for unseen
                q_in = d_emb + ctx_dim
                self.q_proj = nn.Linear(q_in, d_attn)
                self.model_keys = nn.Parameter(torch.randn(n_models, d_attn) * 0.1)
                self.scale = d_attn ** 0.5

            def forward(self, concept_idx, ctx, probs):
                # concept_idx: (B,), ctx: (B, ctx_dim) or empty, probs: (B, k)
                emb = self.concept_emb(concept_idx)  # (B, d_emb)
                if ctx.shape[1] > 0:
                    q_in = torch.cat([emb, ctx], dim=1)
                else:
                    q_in = emb
                q = self.q_proj(q_in)  # (B, d_attn)
                logits = q @ self.model_keys.T / self.scale  # (B, k)
                w = torch.softmax(logits, dim=1)             # (B, k) gating weights
                p = (w * probs).sum(dim=1)                   # (B,)  ensemble prob
                return p, w
        return _AttentionHead(self.d_emb, self.d_attn, n_models, n_concepts, ctx_dim)

    def _resolve_device(self):
        import torch
        if self.device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            if torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        return self.device

    def fit(self, valid_probs, valid_labels, valid_concepts=None,
             context=None) -> "AttentionGating":  # type: ignore[override]
        import torch
        import torch.nn as nn
        from torch.optim import Adam

        if valid_concepts is None:
            raise ValueError("AttentionGating.fit requires valid_concepts")
        p = _prep(valid_probs).astype(np.float32)
        y = np.asarray(valid_labels).astype(np.float32).ravel()
        c = np.asarray(valid_concepts).astype(np.int64).ravel()
        if context is not None:
            ctx = np.asarray(context, dtype=np.float32)
            if ctx.ndim == 1:
                ctx = ctx.reshape(-1, 1)
        else:
            ctx = np.zeros((p.shape[0], 0), dtype=np.float32)
        n, k = p.shape
        self._n_models = k
        self._n_concepts = int(c.max()) + 1
        self._ctx_dim = int(ctx.shape[1])

        device = self._resolve_device()
        torch.manual_seed(self.seed)
        model = self._build_model(k, self._n_concepts, self._ctx_dim).to(device)
        opt = Adam(model.parameters(), lr=self.lr)
        bce = nn.BCELoss()

        # 90/10 train/val split for early stopping
        rng = np.random.default_rng(self.seed)
        perm = rng.permutation(n)
        n_val = max(1, n // 10)
        val_idx = perm[:n_val]
        tr_idx = perm[n_val:]

        best_val = float("inf")
        best_state = None
        patience = 3
        no_improve = 0

        for epoch in range(self.n_epochs):
            # train
            model.train()
            for start in range(0, tr_idx.size, self.batch_size):
                batch = tr_idx[start:start + self.batch_size]
                b_c = torch.from_numpy(c[batch]).to(device)
                b_ctx = torch.from_numpy(ctx[batch]).to(device)
                b_p = torch.from_numpy(p[batch]).to(device)
                b_y = torch.from_numpy(y[batch]).to(device)
                pred, w = model(b_c, b_ctx, b_p)
                loss = bce(pred.clamp(_EPS, 1.0 - _EPS), b_y)
                if self.entropy_lambda > 0:
                    ent = -(w.clamp_min(_EPS) * w.clamp_min(_EPS).log()).sum(dim=1).mean()
                    loss = loss - self.entropy_lambda * ent  # maximise entropy → subtract
                opt.zero_grad(); loss.backward(); opt.step()

            # validate
            model.eval()
            with torch.no_grad():
                b_c = torch.from_numpy(c[val_idx]).to(device)
                b_ctx = torch.from_numpy(ctx[val_idx]).to(device)
                b_p = torch.from_numpy(p[val_idx]).to(device)
                b_y = torch.from_numpy(y[val_idx]).to(device)
                pred, _ = model(b_c, b_ctx, b_p)
                v = float(bce(pred.clamp(_EPS, 1.0 - _EPS), b_y))
            if v < best_val - 1e-5:
                best_val = v
                best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        self._model = model
        self._fitted = True
        return self

    def predict(self, test_probs, test_concepts=None,  # type: ignore[override]
                 context=None) -> np.ndarray:
        import torch
        if not self._fitted:
            raise RuntimeError("call fit(valid_probs, valid_labels, valid_concepts) before predict")
        if test_concepts is None:
            raise ValueError("AttentionGating.predict requires test_concepts")
        p = _prep(test_probs).astype(np.float32)
        c = np.asarray(test_concepts).astype(np.int64).ravel()
        # cap concept ids to +1 for unseen (matches emb table capacity)
        c = np.minimum(c, self._n_concepts)
        if context is not None:
            ctx = np.asarray(context, dtype=np.float32)
            if ctx.ndim == 1:
                ctx = ctx.reshape(-1, 1)
        else:
            ctx = np.zeros((p.shape[0], self._ctx_dim), dtype=np.float32)
        if ctx.shape[1] != self._ctx_dim:
            raise ValueError(f"context dim {ctx.shape[1]} != fit-time {self._ctx_dim}")

        device = self._resolve_device()
        self._model.eval()
        outs: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, p.shape[0], self.batch_size):
                b_c = torch.from_numpy(c[start:start + self.batch_size]).to(device)
                b_ctx = torch.from_numpy(ctx[start:start + self.batch_size]).to(device)
                b_p = torch.from_numpy(p[start:start + self.batch_size]).to(device)
                pred, _ = self._model(b_c, b_ctx, b_p)
                outs.append(pred.cpu().numpy())
        return np.concatenate(outs).astype(np.float64)


class MixtureOfExperts(EnsembleBase):
    """Top-k sparse mixture of experts (Jacobs 1991 / Shazeer 2017-style).

    At predict time, the gating network produces k logits per row. We
    take softmax over the top-``k_top`` logits (rest zeroed), and the
    ensemble output is the sparse convex combination
    :math:`\\sum_{m \\in \\text{top-k}(x)} w_m(x) \\cdot p_m(x)`.

    Compared to ``AttentionGating`` (dense soft-attention over all k
    components), MoE offers:
      * **Interpretability**: each row is explained by ``k_top`` of the
        4 components — a sparse recipe rather than a mix.
      * **Computational efficiency at inference**: skip inactive
        components. For KT with k=4 this doesn't matter, but the
        recipe generalises to k=8+ (heterogeneous cross-family).
      * **Balancing regularisation**: an auxiliary load-balancing penalty
        in the spirit of Shazeer 2017, though not their equation 12: here it
        is k times the sum of squared mean gate weights, which is minimised
        when the load is uniform. It prevents any single component from being
        chosen for every row.

    Training is identical to AttentionGating except for the top-k
    zeroing + load-balancing regularisation. Torch-lazy-imported.
    """

    name = "mixture_of_experts"

    def __init__(self, k_top: int = 2, d_emb: int = 16, d_attn: int = 32,
                 n_epochs: int = 15, batch_size: int = 2048,
                 lr: float = 1e-3, balance_lambda: float = 0.01,
                 device: str = "auto", seed: int = 0) -> None:
        self.k_top = int(k_top)
        self.d_emb = int(d_emb)
        self.d_attn = int(d_attn)
        self.n_epochs = int(n_epochs)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.balance_lambda = float(balance_lambda)
        self.device = device
        self.seed = int(seed)
        self._model = None
        self._n_models: int | None = None
        self._n_concepts: int | None = None
        self._ctx_dim: int = 0
        self._fitted = False

    def _resolve_device(self):
        import torch
        if self.device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            if torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        return self.device

    def _build_model(self, n_models: int, n_concepts: int, ctx_dim: int, k_top: int):
        import torch
        import torch.nn as nn

        class _MoEHead(nn.Module):
            def __init__(self, d_emb, d_attn, n_models, n_concepts, ctx_dim, k_top):
                super().__init__()
                self.concept_emb = nn.Embedding(n_concepts + 1, d_emb)
                q_in = d_emb + ctx_dim
                self.q_proj = nn.Linear(q_in, d_attn)
                self.model_keys = nn.Parameter(torch.randn(n_models, d_attn) * 0.1)
                self.scale = d_attn ** 0.5
                self.k_top = int(k_top)
                self.n_models = int(n_models)

            def forward(self, concept_idx, ctx, probs):
                emb = self.concept_emb(concept_idx)
                if ctx.shape[1] > 0:
                    q_in = torch.cat([emb, ctx], dim=1)
                else:
                    q_in = emb
                q = self.q_proj(q_in)
                logits = q @ self.model_keys.T / self.scale   # (B, k)
                # top-k mask: keep highest k_top per row, -inf elsewhere
                topk_vals, topk_idx = torch.topk(logits, self.k_top, dim=1)
                sparse_logits = torch.full_like(logits, float("-inf"))
                sparse_logits.scatter_(1, topk_idx, topk_vals)
                w = torch.softmax(sparse_logits, dim=1)      # zeros outside top-k
                p = (w * probs).sum(dim=1)
                return p, w
        return _MoEHead(self.d_emb, self.d_attn, n_models, n_concepts, ctx_dim, k_top)

    def fit(self, valid_probs, valid_labels, valid_concepts=None,
             context=None) -> "MixtureOfExperts":  # type: ignore[override]
        import torch
        import torch.nn as nn
        from torch.optim import Adam

        if valid_concepts is None:
            raise ValueError("MixtureOfExperts.fit requires valid_concepts")
        p = _prep(valid_probs).astype(np.float32)
        y = np.asarray(valid_labels).astype(np.float32).ravel()
        c = np.asarray(valid_concepts).astype(np.int64).ravel()
        if context is not None:
            ctx = np.asarray(context, dtype=np.float32)
            if ctx.ndim == 1:
                ctx = ctx.reshape(-1, 1)
        else:
            ctx = np.zeros((p.shape[0], 0), dtype=np.float32)
        n, k = p.shape
        if self.k_top > k:
            raise ValueError(f"k_top={self.k_top} > k={k}")
        self._n_models = k
        self._n_concepts = int(c.max()) + 1
        self._ctx_dim = int(ctx.shape[1])

        device = self._resolve_device()
        torch.manual_seed(self.seed)
        model = self._build_model(k, self._n_concepts, self._ctx_dim, self.k_top).to(device)
        opt = Adam(model.parameters(), lr=self.lr)
        bce = nn.BCELoss()

        rng = np.random.default_rng(self.seed)
        perm = rng.permutation(n)
        n_val = max(1, n // 10)
        val_idx = perm[:n_val]; tr_idx = perm[n_val:]
        best_val = float("inf"); best_state = None; no_improve = 0

        for epoch in range(self.n_epochs):
            model.train()
            for start in range(0, tr_idx.size, self.batch_size):
                batch = tr_idx[start:start + self.batch_size]
                b_c = torch.from_numpy(c[batch]).to(device)
                b_ctx = torch.from_numpy(ctx[batch]).to(device)
                b_p = torch.from_numpy(p[batch]).to(device)
                b_y = torch.from_numpy(y[batch]).to(device)
                pred, w = model(b_c, b_ctx, b_p)
                loss = bce(pred.clamp(_EPS, 1.0 - _EPS), b_y)
                # Shazeer 2017 load-balancing: variance of per-batch mean gate mass
                if self.balance_lambda > 0:
                    load = w.mean(dim=0)  # (k,) — avg gate mass per component
                    balance_loss = float(k) * (load * load).sum()  # ↑ when concentrated
                    loss = loss + self.balance_lambda * balance_loss
                opt.zero_grad(); loss.backward(); opt.step()

            # validate
            model.eval()
            with torch.no_grad():
                b_c = torch.from_numpy(c[val_idx]).to(device)
                b_ctx = torch.from_numpy(ctx[val_idx]).to(device)
                b_p = torch.from_numpy(p[val_idx]).to(device)
                b_y = torch.from_numpy(y[val_idx]).to(device)
                pred, _ = model(b_c, b_ctx, b_p)
                v = float(bce(pred.clamp(_EPS, 1.0 - _EPS), b_y))
            if v < best_val - 1e-5:
                best_val = v
                best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= 3:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        self._model = model
        self._fitted = True
        return self

    def predict(self, test_probs, test_concepts=None, context=None) -> np.ndarray:  # type: ignore[override]
        import torch
        if not self._fitted:
            raise RuntimeError("call fit(valid_probs, valid_labels, valid_concepts) before predict")
        if test_concepts is None:
            raise ValueError("MixtureOfExperts.predict requires test_concepts")
        p = _prep(test_probs).astype(np.float32)
        c = np.asarray(test_concepts).astype(np.int64).ravel()
        c = np.minimum(c, self._n_concepts)
        if context is not None:
            ctx = np.asarray(context, dtype=np.float32)
            if ctx.ndim == 1:
                ctx = ctx.reshape(-1, 1)
        else:
            ctx = np.zeros((p.shape[0], self._ctx_dim), dtype=np.float32)

        device = self._resolve_device()
        self._model.eval()
        outs: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, p.shape[0], self.batch_size):
                b_c = torch.from_numpy(c[start:start + self.batch_size]).to(device)
                b_ctx = torch.from_numpy(ctx[start:start + self.batch_size]).to(device)
                b_p = torch.from_numpy(p[start:start + self.batch_size]).to(device)
                pred, _ = self._model(b_c, b_ctx, b_p)
                outs.append(pred.cpu().numpy())
        return np.concatenate(outs).astype(np.float64)


GATING_HEADS: dict[str, type[EnsembleBase]] = {
    "static_concept_weights":         StaticConceptWeights,
    "global_stack_concept_intercept": GlobalStackWithConceptIntercept,
    "linear_gating":                  LinearGating,
    "attention_gating":               AttentionGating,
    "mixture_of_experts":             MixtureOfExperts,
}


# ─── Concept-conditional calibrated ensemble ─────────────────────────────── #


class CCCE(EnsembleBase):
    """Concept-Conditional Calibrated Ensemble.

    Three-stage pipeline, all stages concept-aware:

    1. **Per-component pre-calibration.** Each of the k component
       probability columns is put through a per-concept isotonic fit
       (``ConceptAwareIsotonic``). This produces
       ``k`` individually-well-calibrated component predictions
       aligned with ``test_concepts``.

    2. **Concept-conditional gating.** The k calibrated columns are
       combined by ``StaticConceptWeights`` (per-concept logistic
       stacking on calibrated component logits). Because Stage 1 already
       removed the per-concept miscalibration bias, the gating head is
       free to learn purely-discriminative weights.

    3. **Global post-calibration.** The ensemble output is fed through
       ONE global isotonic fit (``IsotonicCalibration``). Naeini et al.
       2015 note that combining calibrated predictions can re-introduce
       miscalibration; a single post-hoc step brings the ensemble back
       to the reliability diagonal.

    Rationale for the design choice (see
    the log): both endpoints use concept-aware isotonic (Stage 1) and
    Stage 3 is deliberately *global* — the finding of the gating experiment is that
    residual miscalibration at ensemble output is much smaller than at
    component output, so global post-cal suffices and avoids
    double-per-concept-fit overfitting.

    API is analogous to ``StaticConceptWeights``: ``fit`` takes
    ``valid_probs`` (n_valid, k), ``valid_labels``, ``valid_concepts``;
    ``predict`` takes ``test_probs`` (n_test, k) and ``test_concepts``.

    Empirical target: Pareto-dominate the best stacked blender and
    static-concept gating (Exp C1) on 5-7 datasets — especially on
    ednet, where naive/static gating recovered from stacking's disaster
    but did not surpass best-single. CCCE should push ednet above
    best-single by exploiting the per-concept calibrated components.
    """

    name = "ccce"

    def __init__(self, n_min: int = 30, shrinkage_lambda: float = 0.0,
                 pre_cal: str = "concept",  post_cal: bool = True,
                 cross_fit: int = 0) -> None:
        """
        Parameters
        ----------
        n_min, shrinkage_lambda : passed to per-component ``ConceptAwareIsotonic``
            (only used when ``pre_cal='concept'``).
        pre_cal : {"concept", "global", "none"}
            Stage 1 calibration mode.
            * ``"concept"`` — full ``ConceptAwareIsotonic`` per component
              (concept-aware isotonic, the default).
            * ``"global"`` — plain ``IsotonicCalibration`` per component (no
              concept split — ablation: does concept-awareness in Stage 1 help?).
            * ``"none"`` — skip Stage 1; equivalent to StaticConceptWeights +
              post-cal (ablation: does pre-calibration help at all?).
        post_cal : bool
            Whether Stage 3 global isotonic post-calibration is applied
            Ablation: does post-cal help beyond Stage 2 gating?
        cross_fit : int
            When greater than one, Stage 1 (and, if requested, Stage 3) are
            fitted out of fold inside the validation set: the columns the gate
            learns from are produced by calibrators that never saw the row they
            transform. With the default of zero the calibrators are fitted and
            applied on the same rows, which is what the first version did — and
            what makes a per-concept isotonic on thirty observations look better
            on the fitting set than it is. The two settings separate a real
            mechanism from ordinary overfitting of the first stage.
        """
        from .calibration import IsotonicCalibration
        if pre_cal not in {"concept", "global", "none"}:
            raise ValueError(f"pre_cal must be concept/global/none, got {pre_cal!r}")
        self.n_min = int(n_min)
        self.shrinkage_lambda = float(shrinkage_lambda)
        self.pre_cal = pre_cal
        self.post_cal = bool(post_cal)
        self.cross_fit = int(cross_fit)
        self._pre_calibrators: list = []  # length k, one per component (may be empty)
        self._gate = StaticConceptWeights(n_min=n_min)
        self._post_calibrator = IsotonicCalibration() if self.post_cal else None
        self._fitted = False

    def _apply_pre_calibration(self, probs: np.ndarray, concepts: np.ndarray) -> np.ndarray:
        """Apply Stage 1 to each column. No-op when ``pre_cal='none'``."""
        if not self._pre_calibrators:
            return probs
        out = np.empty_like(probs)
        for m, cal in enumerate(self._pre_calibrators):
            if self.pre_cal == "concept":
                out[:, m] = cal.transform(probs[:, m], concepts)
            else:  # "global"
                out[:, m] = cal.transform(probs[:, m])
        return out

    def fit(self, valid_probs, valid_labels, valid_concepts=None
             ) -> "CCCE":  # type: ignore[override]
        from .calibration import ConceptAwareIsotonic

        if valid_concepts is None:
            raise ValueError("CCCE.fit requires valid_concepts")
        p = _prep(valid_probs)
        y = np.asarray(valid_labels).astype(int).ravel()
        c = np.asarray(valid_concepts).astype(np.int64).ravel()
        if not (p.shape[0] == y.size == c.size):
            raise ValueError(f"valid_probs/labels/concepts length mismatch: "
                              f"{p.shape[0]} / {y.size} / {c.size}")
        k = p.shape[1]

        # STAGE 1: fit per-component pre-calibration (concept-aware / global / none)
        self._pre_calibrators = self._fit_pre_calibrators(p, y, c, k)

        # Columns the gate learns from. Without cross-fitting they come from
        # calibrators that already saw these very rows; with it, from calibrators
        # fitted on the other folds.
        p_cal_fit = (self._crossfit_pre(p, y, c, k) if self.cross_fit > 1
                     else self._apply_pre_calibration(p, c))

        # STAGE 2: fit concept-conditional gating on (possibly) calibrated components
        self._gate.fit(p_cal_fit, y, valid_concepts=c)

        # STAGE 3: fit global isotonic on ensemble output (if requested)
        if self._post_calibrator is not None:
            if self.cross_fit > 1:
                gate_pred = self._crossfit_gate(p_cal_fit, y, c)
            else:
                gate_pred = self._gate.predict(p_cal_fit, test_concepts=c)
            self._post_calibrator.fit(gate_pred, y)

        self._fitted = True
        return self

    # ─── stage-1 helpers ─────────────────────────────────────────────────── #

    def _fit_pre_calibrators(self, p, y, c, k):
        """Calibrators used at predict time: fitted on the whole validation set."""
        from .calibration import ConceptAwareIsotonic, IsotonicCalibration
        out = []
        if self.pre_cal == "concept":
            for m in range(k):
                out.append(ConceptAwareIsotonic(
                    n_min=self.n_min, shrinkage_lambda=self.shrinkage_lambda,
                ).fit(p[:, m], y, c))
        elif self.pre_cal == "global":
            for m in range(k):
                out.append(IsotonicCalibration().fit(p[:, m], y))
        return out

    def _folds(self, n: int):
        """Deterministic split of the validation rows into ``cross_fit`` parts."""
        rng = np.random.default_rng(0)
        return np.array_split(rng.permutation(n), self.cross_fit)

    def _crossfit_pre(self, p, y, c, k):
        """Stage-1 outputs produced out of fold: no row is calibrated by a
        calibrator that saw it."""
        if self.pre_cal == "none":
            return p
        from .calibration import ConceptAwareIsotonic, IsotonicCalibration
        out = np.empty_like(p)
        for held in self._folds(p.shape[0]):
            mask = np.ones(p.shape[0], dtype=bool)
            mask[held] = False
            for m in range(k):
                if self.pre_cal == "concept":
                    cal = ConceptAwareIsotonic(
                        n_min=self.n_min, shrinkage_lambda=self.shrinkage_lambda,
                    ).fit(p[mask, m], y[mask], c[mask])
                    out[held, m] = cal.transform(p[held, m], c[held])
                else:
                    cal = IsotonicCalibration().fit(p[mask, m], y[mask])
                    out[held, m] = cal.transform(p[held, m])
        return out

    def _crossfit_gate(self, p_cal, y, c):
        """Gate predictions produced out of fold, so stage 3 is not fitted on
        the gate's own training rows."""
        out = np.empty(p_cal.shape[0], dtype=float)
        for held in self._folds(p_cal.shape[0]):
            mask = np.ones(p_cal.shape[0], dtype=bool)
            mask[held] = False
            gate = StaticConceptWeights(n_min=self.n_min)
            gate.fit(p_cal[mask], y[mask], valid_concepts=c[mask])
            out[held] = gate.predict(p_cal[held], test_concepts=c[held])
        return out
    def predict(self, test_probs, test_concepts=None) -> np.ndarray:  # type: ignore[override]
        if not self._fitted:
            raise RuntimeError("call fit(valid_probs, valid_labels, valid_concepts) before predict")
        if test_concepts is None:
            raise ValueError("CCCE.predict requires test_concepts")
        p = _prep(test_probs)
        c = np.asarray(test_concepts).astype(np.int64).ravel()
        if p.shape[0] != c.size:
            raise ValueError(f"test_probs rows ({p.shape[0]}) != test_concepts ({c.size})")

        # Stage 1 (no-op if pre_cal='none')
        p_cal = self._apply_pre_calibration(p, c)
        # Stage 2
        gate_pred = self._gate.predict(p_cal, test_concepts=c)
        # Stage 3 (no-op if post_cal=False)
        if self._post_calibrator is not None:
            return self._post_calibrator.transform(gate_pred)
        return gate_pred

    def fit_predict(self, valid_probs, valid_labels,  # type: ignore[override]
                     test_probs, valid_concepts=None, test_concepts=None) -> np.ndarray:
        return self.fit(valid_probs, valid_labels, valid_concepts) \
                   .predict(test_probs, test_concepts)


# ─── Bootstrap CI helpers for ensemble contrasts ──────────────────────────── #
#
# All ensembles here (simple aggregators, stacked blenders, gating heads)
# are *row-independent*: the ensemble prediction at row i depends only on the
# component predictions (and, for gating, on context features) at row i, never
# on other rows. That means ``ensemble.predict(matrix[idx]) ==
# ensemble.predict(matrix)[idx]`` for any row selection ``idx``. Consequence: we
# can precompute the ensemble vector on the full test set once and defer to
# ``stats.paired_bootstrap`` for the resampling loop, instead of re-running the
# ensemble inside every bootstrap iteration. This matches the standard practice
# in benchmarking literature (Efron 1979; the meta-learner is fit on train/valid
# once and the test evaluation is what gets resampled).
#
# For stacked / gating ensembles that require an explicit fit on validation
# data, pass ``fit_probs`` + ``fit_labels`` — the wrapper calls ``.fit()`` once
# before precomputing the test prediction. Simple aggregators ignore both.


def bootstrap_ensemble_vs_single(
    y_true, test_probs_matrix, ensemble: EnsembleBase, single_pred, metric_fn,
    n_boot: int = 2000, groups=None, alpha: float = 0.05, seed: int = 0,
    metric_name: str = "metric", fit_probs=None, fit_labels=None,
):
    """Cluster-bootstrap CI + two-sided p on ``metric(ensemble) - metric(single)``.

    Parameters
    ----------
    y_true : (n,) array of {0, 1} labels.
    test_probs_matrix : (n, k) matrix of the k component-model probabilities
        on the test rows, aligned with ``y_true``.
    ensemble : an ``EnsembleBase`` instance. If it needs fitting, supply
        ``fit_probs`` (n_valid, k) and ``fit_labels`` (n_valid,).
    single_pred : (n,) probability vector of the reference single model
        (typically ``best_single_pred_for_dataset``).
    metric_fn : callable ``(y_true, y_prob) -> float``; use
        ``stats.auc_metric`` / ``stats.ece_metric`` / ``stats.brier_metric``.
    groups : optional cluster ids (student uid per row) for
        cluster-bootstrap — the honest choice for KT test sets.

    Returns
    -------
    ``stats.ComparisonResult`` with ``method='cluster_bootstrap'`` (or
    ``'bootstrap'`` if ``groups`` is None). ``value_a`` is the ensemble metric,
    ``value_b`` is the single-model metric.
    """
    from .stats import paired_bootstrap
    if fit_probs is not None:
        ensemble.fit(fit_probs, fit_labels)
    ensemble_pred = ensemble.predict(test_probs_matrix)
    return paired_bootstrap(
        y_true, ensemble_pred, single_pred, metric_fn,
        n_boot=n_boot, groups=groups, alpha=alpha,
        seed=seed, metric_name=metric_name,
    )


def bootstrap_ensemble_vs_ensemble(
    y_true, test_probs_matrix_a, ensemble_a: EnsembleBase,
    test_probs_matrix_b, ensemble_b: EnsembleBase, metric_fn,
    n_boot: int = 2000, groups=None, alpha: float = 0.05, seed: int = 0,
    metric_name: str = "metric",
    fit_probs_a=None, fit_labels_a=None,
    fit_probs_b=None, fit_labels_b=None,
):
    """Cluster-bootstrap CI + two-sided p on ``metric(A) - metric(B)`` between
    two ensembles. Same shape / alignment rules as
    ``bootstrap_ensemble_vs_single``; the two ensembles may operate on
    different component subsets (hence separate ``test_probs_matrix``
    inputs), but must be aligned to the same ``y_true`` rows.
    """
    from .stats import paired_bootstrap
    if fit_probs_a is not None:
        ensemble_a.fit(fit_probs_a, fit_labels_a)
    if fit_probs_b is not None:
        ensemble_b.fit(fit_probs_b, fit_labels_b)
    pred_a = ensemble_a.predict(test_probs_matrix_a)
    pred_b = ensemble_b.predict(test_probs_matrix_b)
    return paired_bootstrap(
        y_true, pred_a, pred_b, metric_fn,
        n_boot=n_boot, groups=groups, alpha=alpha,
        seed=seed, metric_name=metric_name,
    )
