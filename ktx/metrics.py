"""Full evaluation metric suite for KT predictions.

Operates on flat aligned arrays of binary labels and predicted probabilities, so
it works identically for deep and classical models, at concept or question level.
Covers discrimination, probability accuracy and calibration.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

_EPS = 1e-12


def _clip(p: np.ndarray) -> np.ndarray:
    return np.clip(p, _EPS, 1.0 - _EPS)


def ece_equal_mass(y_true, y_prob, n_bins: int = 10) -> float:
    """Equal-MASS (quantile-binned) ECE — a robustness complement to equal-width ECE.

    Equal-width ECE is sensitive to bin placement when predictions cluster; equal-mass
    bins (each holding ~the same number of points, by prediction quantiles) reduce that
    bias (Nixon 2019, adaptive calibration error). Reported alongside the standard ECE so
    the conclusions can be checked against the binning scheme.
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_prob = np.asarray(y_prob, dtype=np.float64).ravel()
    n = y_prob.size
    if n == 0:
        return float("nan")
    order = np.argsort(y_prob)
    yp, yt = y_prob[order], y_true[order]
    edges = np.linspace(0, n, n_bins + 1).astype(int)  # equal counts per bin
    ece = 0.0
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        if hi <= lo:
            continue
        conf = yp[lo:hi].mean()
        acc = yt[lo:hi].mean()
        ece += (hi - lo) / n * abs(acc - conf)
    return float(ece)


def reliability_bins(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10):
    """Equal-width reliability-diagram bins.

    Returns dict with per-bin: bin_lower, bin_upper, count, mean_pred, frac_pos.
    Empty bins are included with NaN stats so plots have a stable x-axis.
    """
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(y_prob, edges, right=True) - 1, 0, n_bins - 1)
    out = {"bin_lower": [], "bin_upper": [], "count": [], "mean_pred": [], "frac_pos": []}
    for b in range(n_bins):
        mask = idx == b
        n = int(mask.sum())
        out["bin_lower"].append(float(edges[b]))
        out["bin_upper"].append(float(edges[b + 1]))
        out["count"].append(n)
        out["mean_pred"].append(float(y_prob[mask].mean()) if n else float("nan"))
        out["frac_pos"].append(float(y_true[mask].mean()) if n else float("nan"))
    return out


@dataclass
class MetricSet:
    # discrimination
    auc: float = float("nan")
    accuracy: float = float("nan")
    f1: float = float("nan")
    precision: float = float("nan")
    recall: float = float("nan")
    # probability accuracy
    rmse: float = float("nan")
    log_loss: float = float("nan")
    # calibration
    ece: float = float("nan")
    mce: float = float("nan")
    brier: float = float("nan")
    nll: float = float("nan")
    # bookkeeping
    n: int = 0
    base_rate: float = float("nan")
    reliability: dict = field(default_factory=dict)

    def scalars(self) -> dict:
        """Flat dict of scalar metrics (excludes the reliability bin arrays)."""
        return {
            "auc": self.auc, "accuracy": self.accuracy, "f1": self.f1,
            "precision": self.precision, "recall": self.recall,
            "rmse": self.rmse, "log_loss": self.log_loss,
            "ece": self.ece, "mce": self.mce, "brier": self.brier, "nll": self.nll,
            "n": self.n, "base_rate": self.base_rate,
        }


def compute_metrics(y_true, y_prob, n_bins: int = 10) -> MetricSet:
    """Compute the full metric suite from labels and probabilities."""
    y_true = np.asarray(y_true).astype(int).ravel()
    y_prob = np.asarray(y_prob, dtype=np.float64).ravel()
    yp = _clip(y_prob)
    pred = (y_prob >= 0.5).astype(int)

    two_classes = len(np.unique(y_true)) > 1

    ms = MetricSet(
        auc=float(roc_auc_score(y_true, y_prob)) if two_classes else float("nan"),
        accuracy=float(accuracy_score(y_true, pred)),
        f1=float(f1_score(y_true, pred, zero_division=0)),
        precision=float(precision_score(y_true, pred, zero_division=0)),
        recall=float(recall_score(y_true, pred, zero_division=0)),
        rmse=float(np.sqrt(np.mean((y_true - y_prob) ** 2))),
        log_loss=float(log_loss(y_true, yp, labels=[0, 1])),
        brier=float(brier_score_loss(y_true, y_prob)),
        nll=float(-np.mean(y_true * np.log(yp) + (1 - y_true) * np.log(1 - yp))),
        n=int(y_true.size),
        base_rate=float(y_true.mean()),
        reliability=reliability_bins(y_true, y_prob, n_bins),
    )
    # netcal calibration errors (guard tiny/degenerate inputs).
    # The import is local on purpose: netcal pulls torch and matplotlib, while
    # every ensemble artifact in this project is produced through
    # ``ece_equal_mass`` below, which needs nothing but numpy. Importing it at
    # module level would make that heavy chain a hard requirement of every
    # script that only wants the area under the curve.
    try:
        from netcal.metrics import ECE, MCE
        ms.ece = float(ECE(bins=n_bins).measure(y_prob, y_true))
        ms.mce = float(MCE(bins=n_bins).measure(y_prob, y_true))
    except Exception:
        pass
    return ms
