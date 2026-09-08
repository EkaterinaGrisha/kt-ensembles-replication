"""Cross-family row alignment for heterogeneous ensembles.

Classical KT models (BKT / PFA / PFA-recency / Elo-Rasch) and deep KT models
(DKT / SAKT / AKT / simpleKT) share the same underlying test set but pyKT
walks it differently on each side:

* **classical y_true** — iterate ``test.csv`` (one row per uid), keep
  ``is_repeat==0`` positions. Question-level, temporally sorted.
* **deep y_true**      — iterate ``test_sequences.csv`` (windowed) with
  ``selectmasks``, drop the first selected position per window (pyKT
  prediction shift), aggregate multi-KC concept rows via ``late_mean``
  fusion. Includes cross-batch ``rest`` merging that a CSV-only walk
  cannot reproduce.

The reliable **row key** across both walks is the *leading* ``cidxs`` of
each ``is_repeat==0`` question-event. ``cidxs`` is a globally-unique
per-concept-row identifier assigned by pyKT preprocessing; the leading
cidx of a question-event picks out that event from either walk.

Deep-side ``cidxs`` are obtained from the ``test_cidxs`` field written
to each deep NPZ by ``research/scripts/run_dl_matrix.py`` after the
2026-08-16 pyKT patch (see ``docs/README_cross_family_alignment.md`` or
``notes/16_ensembles_kickoff.md`` §12 for the four required patches).
Classical-side ``cidxs`` are trivially reconstructed from the ``cidxs``
column of ``test.csv``.

Fresh NPZs (dumped after the patch) carry ``test_cidxs``; older NPZs do
not, and ``align_families`` raises with a clear "re-run required" message.

Public surface:

- ``align_families(dataset)`` — returns an ``AlignedIndex`` with the two
  index arrays that project each family's NPZ ``y_true`` / ``y_prob`` /
  ``groups`` into the shared aligned row-set.
- ``AlignedIndex`` — dataclass holding classical / deep index arrays
  plus the shared metadata (uid, cidxs, response).
- ``project_family``  — convenience: given the aligned index + a family
  name ("classical" | "deep"), load an npz and return the projected
  ``(y_true, y_prob, groups)`` triple.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd

from . import paths

# ─── data structures ──────────────────────────────────────────────────────── #


@dataclass
class AlignedIndex:
    """Cross-family aligned index for one dataset (test set is fold-independent).

    Attributes
    ----------
    dataset : str
    classical_idx : (n_shared,) int64
        Positions into classical NPZ ``y_true`` / ``y_prob`` / ``groups``.
    deep_idx : (n_shared,) int64
        Positions into deep NPZ ``y_true`` / ``y_prob`` / ``groups``.
    uid : (n_shared,) int64
    cidx : (n_shared,) int64
        The leading concept-row identifier of each shared question-event.
        Globally unique per event; the true match key.
    response : (n_shared,) int8
    n_classical_events : int
    n_deep_events : int
    """
    dataset: str
    classical_idx: np.ndarray
    deep_idx: np.ndarray
    uid: np.ndarray
    cidx: np.ndarray
    response: np.ndarray
    n_classical_events: int
    n_deep_events: int

    @property
    def n_shared(self) -> int:
        return int(self.classical_idx.size)


# ─── classical side: test.csv walk ────────────────────────────────────────── #


def _classical_cidxs_and_labels(dataset: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (uid, cidxs, response) per classical y_true row.

    Walks ``test.csv`` in pyKT-conventional order: for each uid, keep every
    valid (response != -1) event with ``is_repeat == 0``. Question-level;
    matches the row order of ``y_true`` in every classical NPZ for the
    dataset (verified across all 7 datasets on the 4-classical matrix).
    """
    p = paths.PYKT_DATA / dataset / "test.csv"
    df = pd.read_csv(p)
    has_ir = "is_repeat" in df.columns
    uids: list[int] = []
    cidxs: list[int] = []
    responses: list[int] = []
    for row in df.itertuples(index=False):
        uid = int(row.uid)
        rs = np.fromstring(str(row.responses), sep=",", dtype=np.int64)
        cx = np.fromstring(str(row.cidxs), sep=",", dtype=np.int64)
        mask = rs != -1
        rs_v = rs[mask]
        cx_v = cx[mask]
        if has_ir:
            ir = np.fromstring(str(row.is_repeat), sep=",", dtype=np.int64)
            m2 = min(mask.size, ir.size)
            ir_valid = ir[:m2][mask[:m2]]
            starts = np.where(ir_valid == 0)[0]
        else:
            starts = np.arange(rs_v.size, dtype=np.int64)
        for i in starts:
            uids.append(uid)
            cidxs.append(int(cx_v[i]))
            responses.append(int(rs_v[i]))
    return (np.asarray(uids, dtype=np.int64),
            np.asarray(cidxs, dtype=np.int64),
            np.asarray(responses, dtype=np.int8))


# ─── deep side: read persisted test_cidxs from NPZ ───────────────────────── #


def _deep_cidxs_and_labels(dataset: str, model: str = "dkt", fold: int = 0
                            ) -> tuple[np.ndarray, np.ndarray]:
    """Return (cidxs, response) per deep y_true row, using the pyKT-patch
    persisted ``test_cidxs`` field. Any deep model + fold will do (they
    share row alignment within family, verified in run_simple_ensembles);
    ``dkt fold 0`` is the default because it always exists.

    Raises ``RuntimeError`` if the NPZ lacks ``test_cidxs`` — the caller
    needs to re-run the deep matrix with the cidxs-aware pyKT patch.
    """
    npz_path = paths.ARTIFACTS_DIR / "predictions" / dataset / f"{model}_fold{fold}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"deep NPZ not found: {npz_path}")
    d = np.load(npz_path)
    if "test_cidxs" not in d.files:
        raise RuntimeError(
            f"deep NPZ {npz_path.name} has no test_cidxs field. "
            f"Cross-family alignment requires re-running the deep matrix "
            f"with the cidxs-aware pyKT patch — see notes/16_ensembles_kickoff.md "
            f"§12 for the four required patches plus the Kaggle GPU rerun "
            f"command."
        )
    return (np.asarray(d["test_cidxs"], dtype=np.int64),
            np.asarray(d["y_true"], dtype=np.int8))


# ─── main aligner (cached — pure function of dataset) ─────────────────────── #


@lru_cache(maxsize=32)
def align_families(dataset: str, verify_responses: bool = True) -> AlignedIndex:
    """Compute the cross-family aligned index for one dataset. Cached; the
    test set is fold-independent so one aligned index serves all folds.

    ``verify_responses=True`` (default) raises on any row where classical
    and deep disagree on the response for the same ``cidx``. Guards against
    silent data corruption on future dataset additions.
    """
    cls_uid, cls_cidx, cls_resp = _classical_cidxs_and_labels(dataset)
    dp_cidx, dp_resp = _deep_cidxs_and_labels(dataset)

    # Build classical lookup (position -> value) via unique-cidx map. Deep is
    # iterated below in its natural row order, so we do not need a reverse
    # deep-side map at this level.
    cls_pos_by_cidx: dict[int, int] = {int(c): i for i, c in enumerate(cls_cidx)}

    # Intersect on cidx. Deep is (empirically) a subset of classical on all 7
    # datasets — pyKT drops first-of-window + is_repeat=1 events; the leftover
    # classical events are the "first" event of each windowed slice on the deep
    # side. Iterate deep to preserve its natural row order.
    aligned_rows: list[tuple[int, int, int, int]] = []
    for i, cidx in enumerate(dp_cidx):
        cidx_i = int(cidx)
        cls_pos = cls_pos_by_cidx.get(cidx_i)
        if cls_pos is None:
            continue
        cls_r = int(cls_resp[cls_pos])
        dp_r = int(dp_resp[i])
        if verify_responses and cls_r != dp_r:
            raise ValueError(
                f"[{dataset}] response disagreement at cidx={cidx_i}: "
                f"classical={cls_r} deep={dp_r}"
            )
        aligned_rows.append((cls_pos, i, cidx_i, dp_r))

    if not aligned_rows:
        raise RuntimeError(f"[{dataset}] no shared cidxs between families — "
                            f"alignment failed")

    a = np.array(aligned_rows, dtype=np.int64)
    # uid recovered from classical side (deep NPZ doesn't carry uid per row here,
    # but classical does via test.csv walk)
    uids_shared = cls_uid[a[:, 0]]
    return AlignedIndex(
        dataset=dataset,
        classical_idx=a[:, 0].copy(),
        deep_idx=a[:, 1].copy(),
        uid=uids_shared.copy(),
        cidx=a[:, 2].copy(),
        response=a[:, 3].astype(np.int8),
        n_classical_events=int(cls_cidx.size),
        n_deep_events=int(dp_cidx.size),
    )


# ─── projection helper ────────────────────────────────────────────────────── #


CLASSICAL_MODELS = ("bkt", "pfa", "pfa_recency", "elorasch")
DEEP_MODELS = ("dkt", "sakt", "akt", "simplekt")


def project_family(dataset: str, model: str, fold: int, aligned: AlignedIndex
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Load one model's NPZ for (dataset, fold) and project ``y_true /
    y_prob / groups`` through the aligned index.

    ``model`` selects the index side automatically (classical → classical_idx,
    deep → deep_idx). Returns ``None`` if the NPZ is missing. Raises
    ``ValueError`` on shape mismatches or unknown model.
    """
    npz_path = paths.ARTIFACTS_DIR / "predictions" / dataset / f"{model}_fold{fold}.npz"
    if not npz_path.exists():
        return None
    d = np.load(npz_path)
    if model in CLASSICAL_MODELS:
        idx = aligned.classical_idx
        n_expected = aligned.n_classical_events
    elif model in DEEP_MODELS:
        idx = aligned.deep_idx
        n_expected = aligned.n_deep_events
    else:
        raise ValueError(f"unknown model family for '{model}'")
    y = np.asarray(d["y_true"]).astype(int)
    p = np.asarray(d["y_prob"], dtype=np.float64)
    if y.size != n_expected:
        raise ValueError(
            f"[{dataset}/{model} fold{fold}] y_true has {y.size} rows but the aligner "
            f"was built expecting {n_expected}; regenerate the aligned index"
        )
    g = np.asarray(d["groups"]) if "groups" in d.files else np.full(y.size, -1, dtype=np.int64)
    return y[idx], p[idx], g[idx]
