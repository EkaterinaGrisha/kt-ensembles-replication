"""Post-hoc calibration methods for KT predictions.

Each calibrator is fit on a *held-out* set (validation-fold model outputs + labels) and
applied to the test outputs — never fit and evaluated on the same data. This is the
canonical protocol.

Methods:
- ``PlattScaling``    — 1-D logistic regression on the logit of p (Platt 1999).
- ``IsotonicCalibration`` — non-parametric monotone fit (Zadrozny & Elkan 2002).
- ``TemperatureScaling`` — single scalar T on the logit, fit by NLL (Guo et al. 2017);
  the standard neural-network calibrator. T>1 softens overconfident probabilities.

All operate on probabilities in (0,1); we convert to/from logits internally. ``fit``
takes (valid_prob, valid_label), ``transform`` maps test probabilities to calibrated
ones. ``fit_transform`` is a convenience for (valid, test) pairs.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

_EPS = 1e-6


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), _EPS, 1.0 - _EPS)
    return np.log(p / (1.0 - p))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


class PlattScaling:
    """Logistic calibration: p_cal = sigmoid(a * logit(p) + b)."""

    def __init__(self) -> None:
        self._lr = LogisticRegression(C=1e6, solver="lbfgs")  # near-unregularised
        self.a_ = None
        self.b_ = None

    def fit(self, valid_prob, valid_label) -> "PlattScaling":
        z = _logit(valid_prob).reshape(-1, 1)
        y = np.asarray(valid_label).astype(int).ravel()
        self._lr.fit(z, y)
        self.a_ = float(self._lr.coef_[0, 0])
        self.b_ = float(self._lr.intercept_[0])
        return self

    def transform(self, test_prob) -> np.ndarray:
        z = _logit(test_prob).reshape(-1, 1)
        return self._lr.predict_proba(z)[:, 1]

    def fit_transform(self, valid_prob, valid_label, test_prob) -> np.ndarray:
        return self.fit(valid_prob, valid_label).transform(test_prob)


class IsotonicCalibration:
    """Non-parametric monotone mapping fit on validation probabilities."""

    def __init__(self) -> None:
        self._iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)

    def fit(self, valid_prob, valid_label) -> "IsotonicCalibration":
        self._iso.fit(np.asarray(valid_prob, dtype=np.float64).ravel(),
                      np.asarray(valid_label).astype(float).ravel())
        return self

    def transform(self, test_prob) -> np.ndarray:
        return self._iso.predict(np.asarray(test_prob, dtype=np.float64).ravel())

    def fit_transform(self, valid_prob, valid_label, test_prob) -> np.ndarray:
        return self.fit(valid_prob, valid_label).transform(test_prob)


class TemperatureScaling:
    """Single-temperature scaling: p_cal = sigmoid(logit(p) / T), T fit by NLL."""

    def __init__(self) -> None:
        self.T_ = 1.0

    def fit(self, valid_prob, valid_label) -> "TemperatureScaling":
        z = _logit(valid_prob)
        y = np.asarray(valid_label).astype(np.float64).ravel()

        def nll(T: float) -> float:
            p = _sigmoid(z / T)
            p = np.clip(p, _EPS, 1.0 - _EPS)
            return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

        res = minimize_scalar(nll, bounds=(0.05, 20.0), method="bounded")
        self.T_ = float(res.x)
        return self

    def transform(self, test_prob) -> np.ndarray:
        return _sigmoid(_logit(test_prob) / self.T_)

    def fit_transform(self, valid_prob, valid_label, test_prob) -> np.ndarray:
        return self.fit(valid_prob, valid_label).transform(test_prob)


CALIBRATORS = {
    "platt": PlattScaling,
    "isotonic": IsotonicCalibration,
    "temperature": TemperatureScaling,
}


class ConceptAwareIsotonicKAware:
    """K-aware isotonic calibrator: two-dimensional post-hoc mapping
    (fused_prob, K) → calibrated_prob, implementing Theorem 2 (§4.5).

    Fits a separate isotonic regression per K-bucket on the validation set,
    with empirical-Bayes shrinkage towards a global isotonic. The shrinkage
    weight λ_K is inverse-variance: λ_K = n_K / (n_K + n_prior), where
    n_prior is a hyperparameter (larger = more shrinkage to global). If
    n_K is small (e.g., K=7 with 500 samples), λ_K → 0 → mostly global;
    if n_K is large, λ_K → 1 → mostly per-K.

    Interface differs from the 1-D calibrators: fit and transform take an
    extra K array (int-valued question-size). Not in the CALIBRATORS dict
    to keep the 1-D iteration protocol intact — callers must instantiate
    ConceptAwareIsotonicKAware explicitly.

    Parameters
    ----------
    k_buckets : sequence of (lo, hi_inclusive_or_None, label)
        Fixed bins for K stratification. Adaptive merging happens if any
        bucket has fewer than min_bucket_n samples during fit.
    min_bucket_n : int
        Buckets smaller than this get merged upward before fitting.
    n_prior : float
        EB shrinkage prior weight. Higher = more shrinkage to global.
    """

    def __init__(self,
                 k_buckets=((1, 1, "1"), (2, 2, "2"), (3, 3, "3"),
                            (4, None, "4+")),
                 min_bucket_n: int = 100,
                 n_prior: float = 500.0) -> None:
        self.k_buckets = list(k_buckets)
        self.min_bucket_n = int(min_bucket_n)
        self.n_prior = float(n_prior)
        self._global = None            # global isotonic
        self._per_k: dict[str, IsotonicRegression] = {}  # per-K isotonics
        self._lambda_k: dict[str, float] = {}
        self._label_of: dict[int, str] = {}  # K int → bucket label

    def _bucket_of(self, k: int) -> str:
        for lo, hi, label in self.k_buckets:
            if hi is None:
                if k >= lo:
                    return label
            elif lo <= k <= hi:
                return label
        return self.k_buckets[-1][2]  # fallback

    def _adaptive_labels(self, K: np.ndarray) -> dict[int, str]:
        """Merge buckets with <min_bucket_n adjacent-upward."""
        counts = {}
        for lo, hi, label in self.k_buckets:
            if hi is None:
                sel = K >= lo
            else:
                sel = (K >= lo) & (K <= hi)
            counts[label] = int(sel.sum())
        # merge small buckets upward
        merged = {}
        pending_labels: list[str] = []
        pending_n = 0
        for lo, hi, label in self.k_buckets:
            pending_labels.append(label)
            pending_n += counts[label]
            if pending_n >= self.min_bucket_n:
                merged_label = "∪".join(pending_labels)
                for pl in pending_labels:
                    merged[pl] = merged_label
                pending_labels, pending_n = [], 0
        # trailing small residual merges backward
        if pending_labels:
            last = list(merged.values())[-1] if merged else pending_labels[0]
            for pl in pending_labels:
                merged[pl] = last

        # K int → merged bucket label
        out = {}
        for lo, hi, label in self.k_buckets:
            merged_label = merged.get(label, label)
            if hi is None:
                for k in np.unique(K[K >= lo]):
                    out[int(k)] = merged_label
            else:
                for k in np.unique(K[(K >= lo) & (K <= hi)]):
                    out[int(k)] = merged_label
        return out

    def fit(self, valid_prob, valid_label,
            valid_K) -> "ConceptAwareIsotonicKAware":
        p = np.asarray(valid_prob, dtype=np.float64).ravel()
        y = np.asarray(valid_label).astype(float).ravel()
        K = np.asarray(valid_K).astype(int).ravel()

        # global isotonic
        self._global = IsotonicRegression(out_of_bounds="clip",
                                          y_min=0.0, y_max=1.0).fit(p, y)

        # adaptive bucket labels
        self._label_of = self._adaptive_labels(K)
        bucket_labels_arr = np.array([self._label_of[int(k)] for k in K])

        # fit per-bucket isotonic + EB shrinkage weight
        unique_labels = sorted(set(bucket_labels_arr))
        for label in unique_labels:
            sel = bucket_labels_arr == label
            n_k = int(sel.sum())
            if n_k < 30 or len(np.unique(y[sel])) < 2:
                # too small — skip, fallback to global at transform time
                continue
            iso_k = IsotonicRegression(out_of_bounds="clip",
                                       y_min=0.0, y_max=1.0)
            iso_k.fit(p[sel], y[sel])
            self._per_k[label] = iso_k
            self._lambda_k[label] = n_k / (n_k + self.n_prior)
        return self

    def transform(self, test_prob, test_K) -> np.ndarray:
        assert self._global is not None, "must call fit() first"
        p = np.asarray(test_prob, dtype=np.float64).ravel()
        K = np.asarray(test_K).astype(int).ravel()
        out = np.empty_like(p)
        global_pred = self._global.predict(p)
        for i in range(p.size):
            k = int(K[i])
            label = self._label_of.get(k)
            if label is not None and label in self._per_k:
                lam = self._lambda_k[label]
                out[i] = (lam * self._per_k[label].predict(p[i:i+1])[0]
                          + (1 - lam) * global_pred[i])
            else:
                out[i] = global_pred[i]
        return out

    def fit_transform(self, valid_prob, valid_label, valid_K,
                      test_prob, test_K) -> np.ndarray:
        return self.fit(valid_prob, valid_label, valid_K
                        ).transform(test_prob, test_K)


# ─── Concept-aware calibration ───────────────────────────────────────────── #


class ConceptAwareIsotonic:
    """Per-concept isotonic regression with optional empirical-Bayes shrinkage
    toward a global isotonic fit.

    Motivation. Global post-hoc calibrators pool across all concepts; if
    per-concept residual miscalibration is non-uniform
    that leaves signal on the table. This class fits one isotonic
    calibrator per concept with n_c >= n_min validation observations, and
    combines each per-concept prediction with a global-isotonic fallback
    via w_c = n_c / (n_c + lambda) (empirical-Bayes shrinkage).
    Concepts unseen at test time (or with n_c < n_min AND lambda == 0)
    fall through to the global fit.

    Parameters
    ----------
    n_min : int
        Minimum validation observations required to attempt a per-concept
        isotonic fit. Concepts with fewer observations rely entirely on
        the global fit (empirical default = 30).
    shrinkage_lambda : float
        Empirical-Bayes shrinkage parameter. lambda = 0 means "use
        per-concept fit as-is when available"; lambda -> inf means "always
        use global fit". Default 0.0 = no shrinkage (per-concept only,
        with global fallback for small concepts), and 0.0 is what every result
        in this repository was produced with: the parameter is exposed, not
        tuned. Sweeping it per (dataset, model) is left to the reader.
    """

    def __init__(self, n_min: int = 30, shrinkage_lambda: float = 0.0) -> None:
        self.n_min = int(n_min)
        self.shrinkage_lambda = float(shrinkage_lambda)
        self._global = IsotonicCalibration()
        self._per_concept: dict[int, IsotonicCalibration] = {}
        self._n_c: dict[int, int] = {}

    def fit(self, valid_prob, valid_label, valid_concept) -> "ConceptAwareIsotonic":
        vp = np.asarray(valid_prob, dtype=np.float64).ravel()
        vy = np.asarray(valid_label, dtype=np.float64).ravel()
        vc = np.asarray(valid_concept, dtype=np.int64).ravel()
        if vp.size != vy.size or vp.size != vc.size:
            raise ValueError(f"length mismatch: prob={vp.size} label={vy.size} concept={vc.size}")

        self._global.fit(vp, vy)
        self._per_concept.clear()
        self._n_c.clear()
        for c in np.unique(vc):
            mask = vc == c
            n = int(mask.sum())
            self._n_c[int(c)] = n
            if n < self.n_min:
                continue
            if np.unique(vy[mask]).size < 2:
                continue  # isotonic degenerates on single-class subset
            iso = IsotonicCalibration().fit(vp[mask], vy[mask])
            self._per_concept[int(c)] = iso
        return self

    def transform(self, test_prob, test_concept) -> np.ndarray:
        tp = np.asarray(test_prob, dtype=np.float64).ravel()
        tc = np.asarray(test_concept, dtype=np.int64).ravel()
        if tp.size != tc.size:
            raise ValueError(f"length mismatch: prob={tp.size} concept={tc.size}")

        global_pred = self._global.transform(tp)
        out = global_pred.copy()

        # per-concept prediction where a per-concept fit exists
        for c, iso in self._per_concept.items():
            mask = tc == c
            if not mask.any():
                continue
            per_c = iso.transform(tp[mask])
            n_c = self._n_c.get(c, 0)
            w = n_c / (n_c + self.shrinkage_lambda) if (n_c + self.shrinkage_lambda) > 0 else 0.0
            out[mask] = w * per_c + (1.0 - w) * global_pred[mask]
        return out

    def fit_transform(self, valid_prob, valid_label, valid_concept,
                      test_prob, test_concept) -> np.ndarray:
        return self.fit(valid_prob, valid_label, valid_concept) \
                   .transform(test_prob, test_concept)


class DifficultyBucketIsotonic:
    """Isotonic per predicted-difficulty bucket.

    Alternative to ConceptAwareIsotonic that avoids requiring a concept_id
    per row (relevant on question-level with multi-KC late_mean fusion where
    concept assignment is ambiguous). Buckets are defined by quantiles of
    the *validation* predicted probabilities, and each bucket gets its own
    isotonic fit. Test rows are assigned to buckets by the same quantile
    boundaries.
    """

    def __init__(self, n_buckets: int = 5) -> None:
        self.n_buckets = int(n_buckets)
        self._edges: np.ndarray | None = None
        self._global = IsotonicCalibration()
        self._per_bucket: dict[int, IsotonicCalibration] = {}

    def _assign(self, prob: np.ndarray) -> np.ndarray:
        if self._edges is None:
            raise RuntimeError("call fit() before transform()")
        return np.clip(np.digitize(prob, self._edges[1:-1]),
                       0, self.n_buckets - 1)

    def fit(self, valid_prob, valid_label) -> "DifficultyBucketIsotonic":
        vp = np.asarray(valid_prob, dtype=np.float64).ravel()
        vy = np.asarray(valid_label, dtype=np.float64).ravel()
        self._global.fit(vp, vy)
        self._edges = np.quantile(vp, np.linspace(0, 1, self.n_buckets + 1))
        self._edges[0] = -np.inf
        self._edges[-1] = np.inf
        buckets = self._assign(vp)
        self._per_bucket.clear()
        for b in range(self.n_buckets):
            mask = buckets == b
            if mask.sum() < 30 or np.unique(vy[mask]).size < 2:
                continue
            self._per_bucket[b] = IsotonicCalibration().fit(vp[mask], vy[mask])
        return self

    def transform(self, test_prob) -> np.ndarray:
        tp = np.asarray(test_prob, dtype=np.float64).ravel()
        out = self._global.transform(tp)
        buckets = self._assign(tp)
        for b, iso in self._per_bucket.items():
            mask = buckets == b
            if mask.any():
                out[mask] = iso.transform(tp[mask])
        return out

    def fit_transform(self, valid_prob, valid_label, test_prob) -> np.ndarray:
        return self.fit(valid_prob, valid_label).transform(test_prob)


# Registered separately from CALIBRATORS to avoid breaking iterator-based
# consumers that iterate
# CALIBRATORS and call fit(prob, label) with 2 args. Concept-aware needs 3.
CONCEPT_AWARE_CALIBRATORS = {
    "concept_aware_isotonic": ConceptAwareIsotonic,   # needs concept_id per row
    "difficulty_bucket_isotonic": DifficultyBucketIsotonic,  # no concept_id needed
}
