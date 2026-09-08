"""Кластерный бутстрап для внимания и разреженной смеси экспертов.

В статье С5 эти два способа были единственными без проверки значимости: их
посчитали, но бутстрап не гоняли, и вывод по ним опирался только на знак средних.
Здесь этот пробел закрывается — теми же тремя точками отсчёта, что и в таблице
статьи: лучшая одиночная модель, взвешивание по компонентам знания и обучаемая
надстройка.

Обе схемы обучаются заново внутри сценария, потому что предсказания на диск не
сохранялись. Устройство фиксировано процессором, а не выбирается автоматически:
внутри параллельных процессов ускоритель Apple ведёт себя непредсказуемо, а один
прогон занимает пару секунд и на процессоре. Из-за смены устройства точечные
оценки могут отличаться от `attention_gating_deep.csv` в последних знаках, поэтому
сценарий пишет свои — и таблица статьи собирается уже из них.

Выход
-----
`artifacts/ensembles/cluster_bootstrap_attention_moe.csv` — строка на
(набор, разбиение, способ), с точечными оценками, интервалами и p-значениями по
трём контрастам. Рядом `..._pooled.csv` — сводка по пяти разбиениям.

Запуск
------
    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
        OPENBLAS_NUM_THREADS=1 python -m \
        scripts.cluster_bootstrap_attention_moe
    (для дымовой проверки: --datasets algebra2005 --folds 0 --n-boot 200)
"""
from __future__ import annotations

import os as _os
import sys as _sys

_required_env = {
    "KMP_DUPLICATE_LIB_OK": "TRUE",
    "OMP_NUM_THREADS":      "1",
    "MKL_NUM_THREADS":      "1",
    "OPENBLAS_NUM_THREADS": "1",
}
_missing = [k for k, v in _required_env.items() if _os.environ.get(k) != v]
if _missing:
    prefix = " ".join(f"{k}={v}" for k, v in _required_env.items())
    print(f"[warn] не выставлены переменные окружения {_missing}. Запускать так:\n"
          f"  {prefix} python -m scripts.cluster_bootstrap_attention_moe",
          file=_sys.stderr)

import argparse
import csv
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

from ktx import paths
from ktx.bootstrap_fast import auc_metric_fast, paired_bootstrap_fast
from ktx.concept_ids import concept_ids_for_test, concept_ids_for_valid
from ktx.ensemble import (
    AttentionGating,
    LogisticStackedBlender,
    MixtureOfExperts,
    StaticConceptWeights,
)
from ktx.metrics import ece_equal_mass

DATASETS = ["assist2009", "assist2015", "assist2017", "algebra2005",
            "bridge2algebra2006", "assist2012", "ednet"]
FOLDS = [0, 1, 2, 3, 4]
DEEP_MODELS = ["dkt", "sakt", "akt", "simplekt"]
SCHEMES = ["attention", "moe"]
K_TOP = 2          # как в run_moe.py
N_EPOCHS = 15      # как в run_attention_gating.py и run_moe.py
DEVICE = "cpu"
PRED = paths.ARTIFACTS_DIR / "predictions"

auc_metric = auc_metric_fast


def ece_metric(y_true, y_prob) -> float:
    return ece_equal_mass(y_true, y_prob, n_bins=15)


def _load_deep(dataset: str, model: str, fold: int):
    p = PRED / dataset / f"{model}_fold{fold}.npz"
    if not p.exists():
        return None
    d = np.load(p)
    need = {"valid_y_true", "valid_y_prob", "concept_y_true", "concept_y_prob"}
    if not need <= set(d.files):
        return None
    groups = (d["concept_groups"]
              if "concept_groups" in d.files and d["concept_groups"].size > 0 else None)
    return (d["valid_y_true"].astype(int), d["valid_y_prob"].astype(np.float64),
            d["concept_y_true"].astype(int), d["concept_y_prob"].astype(np.float64),
            groups)


def _stack_deep(dataset: str, fold: int):
    v_ref = t_ref = groups = None
    v_cols, t_cols, kept = [], [], []
    for m in DEEP_MODELS:
        got = _load_deep(dataset, m, fold)
        if got is None:
            continue
        vy, vp, ty, tp, g = got
        if v_ref is None:
            v_ref, t_ref, groups = vy, ty, g
        elif not np.array_equal(vy, v_ref) or not np.array_equal(ty, t_ref):
            return None
        v_cols.append(vp)
        t_cols.append(tp)
        kept.append(m)
    if len(kept) < 2:
        return None
    return (v_ref, np.column_stack(v_cols), t_ref, np.column_stack(t_cols), groups, kept)


def _argbest(y, matrix, models) -> int:
    """Лучшая одиночная модель. При равенстве — по имени, чтобы выбор был воспроизводим."""
    aucs = [auc_metric(y, matrix[:, j]) for j in range(matrix.shape[1])]
    return sorted(range(len(models)), key=lambda j: (-aucs[j], models[j]))[0]


def bootstrap_cell(dataset: str, fold: int, n_boot: int, seed: int) -> list[dict]:
    st = _stack_deep(dataset, fold)
    if st is None:
        return []
    vy, vmatrix, ty, tmatrix, groups, kept = st
    try:
        vconc = concept_ids_for_valid(dataset, valid_fold=fold)
        tconc = concept_ids_for_test(dataset)
    except FileNotFoundError:
        return []
    if vconc.size != vy.size or tconc.size != ty.size:
        return []

    # Точка отсчёта выбирается по валидационной части: выбор по тестовой был бы
    # максимумом нескольких шумных оценок и завышал бы её систематически.
    j_best = _argbest(vy, vmatrix, kept)
    best_pred = tmatrix[:, j_best]
    stack_pred = LogisticStackedBlender().fit(vmatrix, vy).predict(tmatrix)
    static_pred = StaticConceptWeights(n_min=30).fit(
        vmatrix, vy, valid_concepts=vconc).predict(tmatrix, test_concepts=tconc)

    rows = []
    for scheme in SCHEMES:
        t0 = time.time()
        try:
            if scheme == "attention":
                model = AttentionGating(n_epochs=N_EPOCHS, seed=0, device=DEVICE)
            else:
                model = MixtureOfExperts(k_top=K_TOP, n_epochs=N_EPOCHS, seed=0,
                                         device=DEVICE)
            pred = model.fit(vmatrix, vy, valid_concepts=vconc).predict(
                tmatrix, test_concepts=tconc)
        except Exception as e:
            print(f"  [warn] {dataset} f{fold} {scheme}: обучение не удалось ({e})")
            continue
        fit_sec = time.time() - t0

        r_best_auc = paired_bootstrap_fast(ty, pred, best_pred, auc_metric,
                                           n_boot=n_boot, groups=groups, seed=seed,
                                           metric_name="auc")
        r_best_ece = paired_bootstrap_fast(ty, pred, best_pred, ece_metric,
                                           n_boot=n_boot, groups=groups, seed=seed + 1,
                                           metric_name="ece")
        r_static = paired_bootstrap_fast(ty, pred, static_pred, auc_metric,
                                         n_boot=n_boot, groups=groups, seed=seed + 2,
                                         metric_name="auc")
        r_stack = paired_bootstrap_fast(ty, pred, stack_pred, auc_metric,
                                        n_boot=n_boot, groups=groups, seed=seed + 3,
                                        metric_name="auc")
        rows.append({
            "dataset": dataset,
            "fold": fold,
            "scheme": scheme,
            "best_single_model": kept[j_best],
            "best_single_selected_on": "valid",
            "n_test": int(ty.size),
            "n_students": int(np.unique(groups).size) if groups is not None else -1,
            "bootstrap_method": "cluster" if groups is not None else "instance",
            "fit_sec": round(fit_sec, 1),
            "scheme_auc": r_best_auc.value_a,
            "scheme_ece": r_best_ece.value_a,
            "best_single_auc": r_best_auc.value_b,
            "delta_auc_vs_best": r_best_auc.diff,
            "auc_ci_low_vs_best": r_best_auc.ci_low,
            "auc_ci_high_vs_best": r_best_auc.ci_high,
            "auc_p_vs_best": r_best_auc.p_value,
            "best_single_ece": r_best_ece.value_b,
            "delta_ece_vs_best": r_best_ece.diff,
            "ece_p_vs_best": r_best_ece.p_value,
            "static_auc": r_static.value_b,
            "delta_auc_vs_static": r_static.diff,
            "auc_ci_low_vs_static": r_static.ci_low,
            "auc_ci_high_vs_static": r_static.ci_high,
            "auc_p_vs_static": r_static.p_value,
            "stacked_auc": r_stack.value_b,
            "delta_auc_vs_stack": r_stack.diff,
            "auc_ci_low_vs_stack": r_stack.ci_low,
            "auc_ci_high_vs_stack": r_stack.ci_high,
            "auc_p_vs_stack": r_stack.p_value,
        })
    return rows


def pool_folds(per_fold: list[dict]) -> list[dict]:
    buckets = defaultdict(list)
    for r in per_fold:
        buckets[(r["dataset"], r["scheme"])].append(r)
    pooled = []
    for (ds, scheme), rows in sorted(buckets.items()):
        entry = {"dataset": ds, "scheme": scheme, "n_folds": len(rows)}
        for base in ("best", "static", "stack"):
            d = np.array([r[f"delta_auc_vs_{base}"] for r in rows])
            p = np.array([r[f"auc_p_vs_{base}"] for r in rows])
            entry[f"mean_delta_auc_vs_{base}"] = float(d.mean())
            entry[f"std_delta_auc_vs_{base}"] = float(d.std(ddof=1)) if len(d) > 1 else 0.0
            entry[f"min_auc_p_vs_{base}"] = float(p.min())
            entry[f"n_folds_sig_auc_vs_{base}_05"] = int((p <= 0.05).sum())
        de = np.array([r["delta_ece_vs_best"] for r in rows])
        entry["mean_delta_ece_vs_best"] = float(de.mean())
        pooled.append(entry)
    return pooled


def _write(path: Path, rows: list[dict]) -> None:
    if not rows:
        print(f"[warn] нечего писать в {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 6) if isinstance(v, float) else v)
                        for k, v in r.items()})
    print(f"{path}: {len(rows)} строк")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--folds", nargs="+", type=int, default=FOLDS)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--out", type=Path,
                    default=paths.ARTIFACTS_DIR / "ensembles"
                    / "cluster_bootstrap_attention_moe.csv")
    ap.add_argument("--pooled-out", type=Path,
                    default=paths.ARTIFACTS_DIR / "ensembles"
                    / "cluster_bootstrap_attention_moe_pooled.csv")
    args = ap.parse_args()

    t0 = time.time()
    cells = [(ds, fold) for ds in args.datasets for fold in args.folds]

    def _run(ds, fold):
        t_cell = time.time()
        return ds, fold, bootstrap_cell(ds, fold, args.n_boot, args.seed), time.time() - t_cell

    results = Parallel(n_jobs=args.n_jobs, backend="loky", verbose=10)(
        delayed(_run)(*c) for c in cells)

    all_rows = []
    for ds, fold, rows, dt in results:
        if not rows:
            print(f"[{ds:22s} f{fold}] пропущено [{dt:.1f} с]")
            continue
        summary = ", ".join(
            f"{r['scheme']}: против взвешивания Δ={r['delta_auc_vs_static']:+.4f} "
            f"p={r['auc_p_vs_static']:.2g}" for r in rows)
        print(f"[{ds:22s} f{fold}] [{dt:.1f} с] {summary}")
        all_rows.extend(rows)

    _write(args.out, all_rows)
    _write(args.pooled_out, pool_folds(all_rows))
    print(f"всего {time.time() - t0:.0f} с")
    return 0


if __name__ == "__main__":
    _sys.exit(main())
