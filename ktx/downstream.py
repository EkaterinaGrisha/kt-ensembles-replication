"""Downstream mastery detection infrastructure.

Builds the (student, concept, timeline) long-form representation from pyKT's
test_sequences.csv and aligns it with model concept-level predictions. Then
constructs the "did-they-master-it?" ground-truth labels + mastery decisions
under two rules × several history windows × several thresholds.

Mastery evaluation protocol (§5.5 paper 2):
  Ground truth. For each (student, concept) with at least H_min history +
  K_future = 3 future attempts on that concept, GT = 1 iff all 3 future
  attempts are correct (fraction >= 0.9 with K=3 collapses to "3-in-a-row"
  binary label). This matches Ritter 2007 / Cognitive Tutor standard.
  Decision point. The LAST timestep t of the (student, concept) sub-sequence
  such that (i) history[:t] has >= N attempts, (ii) future[t:t+K] has >= 3
  attempts. One decision per (student, concept) pair. Simpler than sliding-
  window and avoids label duplication.
  Prediction rules.
    (a) "threshold": mastery_pred = 1 iff mean(last N history predictions) >= tau.
    (b) "bayesian":  Beta(1,1) prior, update with each history pred (soft),
                     posterior mean = (1 + sum p_i) / (2 + N); mastery iff
                     posterior mean >= tau.
  Sweeps.
    windows N in {3, 5, 10, -1(all)}
    thresholds tau in {0.6, 0.7, 0.8}
  Method inputs: raw or calibrated probabilities (Platt / isotonic / temperature).

Two granularities supported (matches paper 2 Layer 1 story):
  - concept-level: timeline of concept predictions, calibration on concept-valid
  - question-level: timeline aggregated to questions (late_mean), calibration
    on valid_y_*_q (needs the question-level aggregator to have run).
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import paths

# Обход последовательностей вынесен в отдельный модуль; имена
# ре-экспортируются, чтобы прежние вызовы продолжали работать.
from .concept_ids import (  # noqa: F401
    TestTimeline,
    concept_ids_for_test,
    concept_ids_for_valid,
    extract_test_timeline,
)


def extract_pykt_q_timeline(dataset: str, orirow: np.ndarray,
                             qidxs: np.ndarray) -> TestTimeline:
    """Rebuild a question-level test timeline that aligns bit-exactly with the
    pyKT-stored `y_true` / `y_prob` arrays of a deep model.

    Motivation (audit #26). Our default `question_level_events` collapses
    concept-level rows via `is_repeat`, which for ednet gives 132502 events
    while pyKT's stored q-level array has 132209 (pyKT's group_fusion uses
    a model-specific inference-path, not simple late_mean). To use pyKT's
    q-level predictions for downstream evaluation we need a matched timeline.
    The P7-patched evaluate_question emits `test_orirow_q` (CSV row per
    prediction) and `test_qids_q` (position within that row's group_fusion
    output); together with `test_question_sequences.csv` this uniquely
    determines (uid, question_id, response) per prediction.
    """
    p = paths.PYKT_ROOT / "data" / dataset / "test_question_sequences.csv"
    if not p.exists():
        raise FileNotFoundError(str(p))
    df = pd.read_csv(p)
    has_q = "questions" in df.columns
    orirow = np.asarray(orirow, dtype=np.int64)
    qidxs = np.asarray(qidxs, dtype=np.int64)
    n = orirow.size
    rows = []
    per_uid_order: dict[int, int] = defaultdict(int)
    # cache parsed columns per CSV row to avoid re-parsing the same row for
    # each prediction (rows have up to hundreds of predictions apiece)
    row_cache: dict[int, tuple[int, list[int], list[int], list[int]]] = {}
    for i in range(n):
        r_idx = int(orirow[i])
        pos = int(qidxs[i])
        if r_idx not in row_cache:
            r = df.iloc[r_idx]
            qs = [int(x) for x in str(r["questions"]).split(",")] if has_q else []
            cs = [int(x) for x in str(r["concepts"]).split(",")]
            rs = [int(x) for x in str(r["responses"]).split(",")]
            row_cache[r_idx] = (int(r["uid"]), qs, cs, rs)
        uid, qs, cs, rs = row_cache[r_idx]
        # `pos` is pyKT's per-row group-fusion index (0-based); the
        # corresponding question in test_question_sequences.csv is the pos-th
        # non-padding entry (padded entries are typically at the tail).
        # For most rows pyKT emits len(qs)-1 predictions (shift by one).
        # pos+1 targets the corresponding attempt in the flat CSV lists.
        idx = pos + 1  # +1 for pyKT prediction shift
        idx = min(idx, len(rs) - 1)
        q_id = qs[idx] if qs else cs[idx]
        c_id = cs[idx]
        rows.append({
            "uid": uid, "order": per_uid_order[uid],
            "question": int(q_id), "concept": int(c_id),
            "response": int(rs[idx]),
            "is_repeat": 0,
            "pred_offset": i,
        })
        per_uid_order[uid] += 1
    long_df = pd.DataFrame(rows)
    return TestTimeline(dataset=dataset, df=long_df, n_concept_preds=n)


def extract_classical_test_timeline(dataset: str) -> TestTimeline:
    """Long-form test timeline aligned with the classical
    `predict_question_level(test)` prediction array (one row per (uid, qpos)).

    Deep and classical models see the test set through different pipelines:

    - Deep: `test_sequences.csv` (pyKT windows), with the first prediction of
      every 200-length window dropped. That gives ~n_students fewer rows than
      the full `test.csv`, and the drop points are per-window not per-student.
    - Classical: `test.csv` (full explode), grouped by `(uid, qpos)`. No first-
      attempt drop, sorted by `(uid, seq_pos)`.

    Positional index-matching between the two arrays is therefore wrong (see
    notes/14_rq2_audit.md P0-1). To use classical predictions in downstream
    mastery evaluation we build a separate timeline whose row order matches
    exactly the classical prediction array, so `pred_offset = arange(n)` is
    a valid index into `y_prob_classical`.

    Row order: sorted by (uid ascending, seq_pos ascending), one row per
    question (first concept id kept, matching the aggregator in
    `predict_question_level`). `order` is the per-student running question
    index (== qpos), `is_repeat` is always 0 in the question-level view.
    """
    from .classical.data import ClassicalData, load_classical_data
    data = load_classical_data(dataset)
    q = ClassicalData.to_question_level(data.test)
    q = q.sort_values(["uid", "seq_pos"]).reset_index(drop=True)
    tl = pd.DataFrame({
        "uid": q["uid"].astype(int).to_numpy(),
        "order": q["qpos"].astype(int).to_numpy(),
        "question": q["qid"].astype(int).to_numpy(),
        # first concept of the (possibly multi-KC) question — same choice as
        # predict_question_level's per-group aggregate for `concepts`
        "concept": q["concepts"].apply(lambda t: int(t[0])).to_numpy(),
        "response": q["response"].astype(int).to_numpy(),
        "is_repeat": np.zeros(len(q), dtype=int),
        "pred_offset": np.arange(len(q), dtype=int),
    })
    return TestTimeline(dataset=dataset, df=tl, n_concept_preds=len(tl))


def question_level_events(concept_timeline: pd.DataFrame) -> pd.DataFrame:
    """Collapse the concept-level timeline to per-question events matching
    pyKT `evaluate_question` late_mean fusion: a new event starts at every
    is_repeat=0 position; is_repeat=1 positions are folded into the current
    event. Returns one row per question-event with the first concept row's
    fields (concept id is arbitrary-but-stable for downstream (uid, concept)
    grouping; response is identical within an interaction), and
    `pred_offset` = global question-event index aligned with pyKT's stored
    question-level `y_prob` array (walk order).

    We do NOT re-sort here: the input timeline is already in pyKT walk order
    (see `extract_test_timeline`), which is the same order in which
    `evaluate_question` emits per-question predictions. Sorting by any key
    other than pred_offset would misalign the returned `pred_offset` from
    the stored q-level y_prob.

    Concept-only datasets: since is_repeat is all zeros, every row is its
    own event → q-events == concept-events, size preserved and pred_offset
    identical to the concept-level index (which is the invariant that makes
    concept-level and question-level results bit-identical on max_kc=1
    datasets).
    """
    df = concept_timeline.reset_index(drop=True)
    df["q_event"] = (df["is_repeat"] == 0).cumsum() - 1
    first = df.groupby("q_event", as_index=False, sort=False).first()
    first = first.reset_index(drop=True)
    first["pred_offset"] = np.arange(len(first), dtype=int)
    return first[["uid", "order", "question", "concept", "response",
                  "is_repeat", "pred_offset"]]


# --- Ground truth + decision points ---------------------------------------- #

def build_mastery_decisions(
    timeline: TestTimeline,
    min_history: int,
    future_k: int = 3,
    tau_gt: float = 0.9,
) -> pd.DataFrame:
    """Build one decision-point per (uid, concept) pair with enough
    history + future attempts. Returns columns:
      uid, concept, history_offsets (list[int]), future_offsets (list[int]),
      mastery_gt (int).

    ``min_history`` is the maximum window size you plan to use; any pair with
    fewer than min_history attempts before the future_k tail is skipped.
    """
    df = timeline.df
    # pred_offset is the globally correct chronological index (walk order of
    # pyKT evaluate_with_preds); sorting by (uid, concept, pred_offset) puts
    # the K_future = last-3 attempts of each (student, concept) into the
    # "future" slice deterministically. Sorting on `order` also works after
    # the P0-1 fix (per-student monotonic), but pred_offset is the ground
    # truth of alignment, so we use it here.
    df = df.sort_values(["uid", "concept", "pred_offset"]).reset_index(drop=True)
    out: list[dict] = []
    for (uid, concept), g in df.groupby(["uid", "concept"], sort=False):
        n_total = len(g)
        if n_total < min_history + future_k:
            continue
        # Decision point: split so that last future_k attempts are held out
        history = g.iloc[:n_total - future_k]
        future = g.iloc[n_total - future_k:]
        future_correct = int(future["response"].sum())
        mastery_gt = int((future_correct / future_k) >= tau_gt)
        out.append({
            "uid": uid,
            "concept": concept,
            "n_history": int(len(history)),
            "n_future": int(len(future)),
            "history_offsets": history["pred_offset"].tolist(),
            "future_offsets": future["pred_offset"].tolist(),
            "future_correct": future_correct,
            "mastery_gt": mastery_gt,
        })
    return pd.DataFrame(out)


# --- Mastery prediction rules ---------------------------------------------- #

def rule_threshold(preds: np.ndarray, window: int, tau: float) -> int:
    """Rule (a): mastery iff mean of last-`window` predictions >= tau.
    `window = -1` means "all history"."""
    if window > 0:
        preds = preds[-window:] if window <= len(preds) else preds
    if len(preds) == 0:
        return 0
    return int(preds.mean() >= tau)


def rule_bayesian(preds: np.ndarray, window: int, tau: float,
                  alpha: float = 1.0, beta: float = 1.0) -> int:
    """Rule (b): Beta(alpha, beta) prior, treat each prediction p_i as a soft
    Bernoulli observation. Posterior mean = (alpha + sum p_i) / (alpha + beta + N).
    Mastery iff posterior mean >= tau.

    **Note.** For any fixed `tau` and window size N,
    this rule is a **monotone transformation of the arithmetic mean** — it
    fires iff `mean(preds) >= (tau*(alpha+beta+N) - alpha) / N`. With the
    Beta(1,1) defaults, at N -> inf the effective threshold converges to
    `tau` itself, so on realistic KT window sizes (N in {3, 5, 10, all})
    "bayesian" and "threshold" produce nearly identical decisions with just
    a small tau-shift ({+0.10, +0.07, +0.05, +0.02} for the four windows at
    tau=0.7). If the goal is to compare qualitatively different rules,
    prefer `rule_beta_lower_ci` below (non-monotone in the mean because it
    depends on both mean AND variance of the posterior). Prior §5.5 text
    that reported "threshold vs bayesian" as two independent methods
    should be read as reporting two thresholds slightly apart on the same
    monotone family."""
    if window > 0:
        preds = preds[-window:] if window <= len(preds) else preds
    if len(preds) == 0:
        return 0
    post = (alpha + float(preds.sum())) / (alpha + beta + len(preds))
    return int(post >= tau)


def rule_beta_lower_ci(preds: np.ndarray, window: int, tau: float,
                        alpha: float = 1.0, beta: float = 1.0,
                        ci_level: float = 0.95) -> int:
    """Rule (c): genuinely non-monotone alternative
    to `rule_bayesian`. Compute the Beta(alpha+sum(preds), beta+N-sum(preds))
    posterior lower `ci_level` credible bound; fire mastery iff the LOWER
    bound of the credible interval clears `tau`. Non-monotone in
    mean(preds): two windows with the same mean but different lengths give
    different lower bounds (longer window -> tighter posterior -> higher
    lower bound), so this rule can flip mastery calls that
    `rule_bayesian`/`rule_threshold` would not.

    Introduced as the "genuinely different rule" alternative the audit
    recommended for §5.5; not used by the current downstream_mastery
    sweep (would require rerun of the full 6528-cell tournament with
    added variance). Available for future runs.
    """
    if window > 0:
        preds = preds[-window:] if window <= len(preds) else preds
    if len(preds) == 0:
        return 0
    s = float(preds.sum())
    n = float(len(preds))
    a_post = alpha + s
    b_post = beta + n - s
    # scipy is a project dep; import lazily so this module stays cheap to load
    from scipy.stats import beta as _beta
    lower = float(_beta.ppf((1.0 - ci_level) / 2.0, a_post, b_post))
    return int(lower >= tau)


# --- Metrics --------------------------------------------------------------- #

def f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Binary F1 with graceful zero-division = 0.0."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    if tp == 0:
        return 0.0
    p = tp / (tp + fp)
    r = tp / (tp + fn)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def precision_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int); y_pred = np.asarray(y_pred).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


def recall_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int); y_pred = np.asarray(y_pred).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0
