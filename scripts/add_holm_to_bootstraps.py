"""Post-process every bootstrap CSV in cross_dataset/ AND ensembles/ to
add `p_holm` / `reject_holm` columns with Holm-Bonferroni FWER control at
alpha=0.05.

§4.4 of the paper promises Holm; individual bootstrap scripts only produce
raw `cb_p` (or per-contrast `auc_p_vs_*`, `ece_p_vs_*`). Rather than
plumbing Holm through every producer, we do one step here so the artifact
is self-contained.

The sweep covers ``artifacts/ensembles/``
where cluster-bootstrap CSVs previously carried raw ``auc_p`` / ``ece_p``
without any FWER correction; 910 raw contrasts (350 simple + 350 stacking
+ 70 gating + 140 CCCE) were all reported un-adjusted. Ensemble family
grouping: per (dataset, subset) — matches how each contrast set is
reported in the results section.

Grouping rules (each file uses ONE family for Holm; when a natural sub-family
column exists, we group within it — this matches the way each contrast set is
reported in the text):

- cluster_bootstrap_calibration_methods{,_qlvl}.csv
    one family = all rows (best vs runner-up method per (dataset, model))
- cluster_bootstrap_calibration_models{,_qlvl}.csv
    families by `contrast_type` (overall_best; per_method:none; per_method:isotonic; …)
- cluster_bootstrap_downstream_f1.csv
    one family = all rows (raw-best vs cal-best ΔF1 per (dataset, model, granularity))
- cluster_bootstrap_cross_family{,_5fold}.csv
    families by contrast/spec column if present, else one family
- cluster_bootstrap_within_deep{,_5fold}.csv, cluster_bootstrap_concept_aware.csv,
    cluster_bootstrap_headline.csv, cluster_bootstrap_simple_ensembles.csv,
    cluster_bootstrap_subgroup_disparity{,_5fold}.csv
    one family per file

Output: overwrites each CSV in place, adds `p_holm` and `reject_holm`.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ktx import paths
from ktx.stats import holm_bonferroni

ALPHA = 0.05
GROUP_BY = {
    "cluster_bootstrap_calibration_models.csv": ["contrast_type"],
    "cluster_bootstrap_calibration_models_qlvl.csv": ["contrast_type"],
}


def _holm_column(df: pd.DataFrame, p_col: str, out_p: str, out_reject: str,
                  group_cols: list[str] | None) -> pd.DataFrame:
    """Apply Holm-Bonferroni to `p_col`, write results into `out_p` /
    `out_reject`. Groups within `group_cols` if provided and all present.
    """
    df[out_p] = 1.0
    df[out_reject] = False
    if group_cols and all(c in df.columns for c in group_cols):
        for _, sub in df.groupby(group_cols, sort=False):
            res = holm_bonferroni(sub[p_col].tolist(), alpha=ALPHA)
            df.loc[sub.index, out_p] = res["adjusted"]
            df.loc[sub.index, out_reject] = res["reject"]
    else:
        res = holm_bonferroni(df[p_col].tolist(), alpha=ALPHA)
        df[out_p] = res["adjusted"]
        df[out_reject] = res["reject"]
    return df


def _apply_holm(df: pd.DataFrame, group_cols: list[str] | None) -> pd.DataFrame:
    """Detect which raw p-value columns to Holm-adjust. Handles both the
    legacy single-column `cb_p` layout (cross_dataset/*.csv) and the
    multi-contrast layout used in ensembles/cluster_bootstrap_*.csv
    (`auc_p`, `ece_p`, or `auc_p_vs_<ref>` / `ece_p_vs_<ref>`).
    """
    df = df.copy()
    # 1) Legacy single-column layout.
    if "cb_p" in df.columns:
        return _holm_column(df, "cb_p", "p_holm", "reject_holm", group_cols)

    # 2) Multi-contrast layout: apply Holm independently per raw p-column.
    # Columns this script wrote on an earlier run must be excluded: without the
    # guard a second run treats `auc_p_vs_best_holm` as a raw p-value and adds
    # `auc_p_vs_best_holm_holm`, and the file fills up with meaningless columns.
    p_cols = [c for c in df.columns
              if (c == "auc_p" or c == "ece_p"
                  or c.startswith("auc_p_vs_")
                  or c.startswith("ece_p_vs_"))
              and not (c.endswith("_holm") or c.endswith("_reject_holm"))]
    if not p_cols:
        return df
    for pc in p_cols:
        _holm_column(df, pc, f"{pc}_holm", f"{pc}_reject_holm", group_cols)
    return df


ENSEMBLE_GROUP_BY = ["dataset", "subset"]  # T3d family definition


def _process_dir(root: Path, group_by_map: dict[str, list[str]],
                 default_group: list[str] | None) -> int:
    files = sorted(p for p in root.glob("cluster_bootstrap_*.csv"))
    total = 0
    for p in files:
        try:
            df = pd.read_csv(p)
        except Exception as e:
            print(f"[skip] {p.name}: {e}")
            continue
        # Prefer per-file group override; else default; else None.
        group_cols = group_by_map.get(p.name, default_group)
        # For ensemble pool-CSVs the group column is only "dataset" (no subset
        # column) — auto-degrade the default.
        if group_cols and any(c not in df.columns for c in group_cols):
            group_cols = [c for c in group_cols if c in df.columns] or None
        out = _apply_holm(df, group_cols)
        # Report Holm activity.
        holm_cols = [c for c in out.columns if c.endswith("_reject_holm") or c == "reject_holm"]
        if not holm_cols:
            print(f"[skip] {p.name}: no p-value column matched")
            continue
        out.to_csv(p, index=False)
        summary = "; ".join(
            f"{hc.replace('reject_holm','').rstrip('_') or 'cb'}:reject={int(out[hc].sum())}"
            for hc in holm_cols
        )
        by = f" (grouped by {group_cols})" if group_cols else ""
        print(f"[{p.name}] rows={len(out):>4d}  {summary}{by}")
        total += 1
    return total


def main() -> None:
    total = 0
    # cross_dataset/* — same behavior as before H1a (T3d rename): default
    # group is None (single family per file), overridden by GROUP_BY.
    total += _process_dir(
        paths.ARTIFACTS_DIR / "cross_dataset", GROUP_BY, default_group=None,
    )
    # ensembles/* — T3d: family per (dataset, subset). Pool CSVs auto-degrade
    # to per-dataset when no `subset` column exists.
    total += _process_dir(
        paths.ARTIFACTS_DIR / "ensembles", group_by_map={},
        default_group=ENSEMBLE_GROUP_BY,
    )
    print(f"\nprocessed {total} bootstrap CSVs")


if __name__ == "__main__":
    main()
