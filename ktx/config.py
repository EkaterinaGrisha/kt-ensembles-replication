"""Clean configuration loading on top of pyKT's JSON configs.

This module exists because pyKT's ``examples/wandb_train.py`` builds its configs
in a way that (a) requires cwd == examples/ for the relative ``../configs`` and
``../data`` paths, and (b) leaks control keys such as ``num_epochs`` into the
model constructor kwargs when parameters are passed programmatically. We rebuild
the same logic explicitly and correctly so experiments are reproducible from any
working directory and safe to drive from Python.
"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from typing import Any

from . import paths

# Per-model batch-size caps that pyKT applies to avoid OOM. Mirrors the logic in
# examples/wandb_train.py::main so our runs match pyKT's defaults exactly.
_OOM_BATCH_64 = {
    "dkvmn", "deep_irt", "sakt", "saint", "saint++", "akt", "atkt",
    "lpkt", "skvmn", "simplekt", "bakt_time",
}
_OOM_BATCH_16 = {"gkt"}

# Models that need seq_len injected into their constructor (pyKT does this too).
_NEEDS_SEQ_LEN = {"saint", "saint++", "sakt", "atdkt", "simplekt", "bakt_time"}

# Keys that are experiment/control concerns, never model constructor kwargs.
_CONTROL_KEYS = {
    "model_name", "dataset_name", "emb_type", "save_dir", "fold", "seed",
    "use_wandb", "add_uuid", "learning_rate", "l2", "num_epochs", "batch_size",
}


@dataclass
class ExperimentConfig:
    """Fully-resolved configuration for a single (model, dataset, fold, seed) run."""

    model_name: str
    dataset_name: str
    fold: int
    seed: int
    emb_type: str = "qid"

    # training
    num_epochs: int = 200
    batch_size: int = 256
    optimizer: str = "adam"
    learning_rate: float = 1e-3
    seq_len: int = 200
    l2: float = 1e-5

    # model hyperparameters (only the keys the model constructor accepts)
    model_config: dict[str, Any] = field(default_factory=dict)
    # the dataset block from data_config.json, with dpath made absolute
    data_config: dict[str, Any] = field(default_factory=dict)


def _load_json(path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def load_data_config() -> dict:
    """Load data_config.json and rewrite every ``dpath`` to an absolute path.

    pyKT stores dpath as ``../data/<name>`` (relative to examples/). We resolve it
    against the vendored pyKT data directory so dataset loading works regardless of
    the process working directory.
    """
    raw = _load_json(paths.DATA_CONFIG_JSON)
    for name, block in raw.items():
        dpath = block.get("dpath", "")
        # Take just the final component and re-root under the vendored data dir.
        # Degenerate values carry no dataset name and must fall back to the key:
        # the pinned fork (72c53a7) ships `"dpath": "."` for every dataset, and
        # `PYKT_DATA / "."` normalises to the data root, silently dropping the
        # dataset directory so every dataset resolved to the same path.
        base = dpath.rstrip("/").split("/")[-1]
        if base in ("", ".", ".."):
            base = name
        block["dpath"] = str(paths.PYKT_DATA / base)
    return raw


def build_experiment_config(
    model_name: str,
    dataset_name: str,
    fold: int = 0,
    seed: int = 42,
    emb_type: str = "qid",
    num_epochs: int | None = None,
    batch_size: int | None = None,
    learning_rate: float | None = None,
    extra_model_config: dict[str, Any] | None = None,
) -> ExperimentConfig:
    """Resolve pyKT JSON configs into a clean ExperimentConfig.

    ``num_epochs``/``batch_size``/``learning_rate`` overrides are applied to the
    TRAIN config only; they never leak into the model constructor kwargs.
    """
    kt = _load_json(paths.KT_CONFIG_JSON)
    train_config = copy.deepcopy(kt["train_config"])
    model_hparams = copy.deepcopy(kt.get(model_name, {}))
    # pyKT's bundled kt_config.json omits simpleKT/saint defaults; we inject
    # values from the original papers so our pipeline can run them out-of-the-box.
    if model_name == "simplekt" and not model_hparams:
        model_hparams = {  # Liu et al. 2023 (ICLR) — defaults from the paper
            "d_model": 256, "n_blocks": 2, "dropout": 0.05,
            "num_attn_heads": 8, "learning_rate": 1e-3,
        }
    if model_name == "saint" and not model_hparams:
        model_hparams = {  # Choi et al. 2020 — pyKT bundled SAINT defaults
            "emb_size": 256, "num_attn_heads": 8,
            "dropout": 0.2, "n_blocks": 4, "learning_rate": 1e-3,
        }

    data_config_all = load_data_config()
    if dataset_name not in data_config_all:
        raise KeyError(
            f"dataset '{dataset_name}' not in data_config.json; "
            f"available: {sorted(data_config_all)}"
        )
    ds_block = data_config_all[dataset_name]

    # batch-size: pyKT OOM caps, then explicit override wins
    bs = train_config["batch_size"]
    if model_name in _OOM_BATCH_64:
        bs = 64
    if model_name in _OOM_BATCH_16:
        bs = 16
    if model_name in {"qdkt", "qikt"} and dataset_name in {"algebra2005", "bridge2algebra2006"}:
        bs = 32
    if batch_size is not None:
        bs = batch_size

    # seq_len: prefer dataset maxlen if present (pyKT behaviour)
    seq_len = train_config["seq_len"]
    if "maxlen" in ds_block:
        seq_len = ds_block["maxlen"]

    # learning rate: model config default, then explicit override
    lr = model_hparams.get("learning_rate", train_config.get("learning_rate", 1e-3))
    if learning_rate is not None:
        lr = learning_rate

    # model_config = model hyperparams minus control keys (lr already extracted)
    clean_model_config = {
        k: v for k, v in model_hparams.items() if k not in _CONTROL_KEYS
    }
    if extra_model_config:
        clean_model_config.update(
            {k: v for k, v in extra_model_config.items() if k not in _CONTROL_KEYS}
        )
    if model_name in _NEEDS_SEQ_LEN:
        clean_model_config["seq_len"] = seq_len

    return ExperimentConfig(
        model_name=model_name,
        dataset_name=dataset_name,
        fold=fold,
        seed=seed,
        emb_type=emb_type,
        num_epochs=num_epochs if num_epochs is not None else train_config["num_epochs"],
        batch_size=bs,
        optimizer=train_config["optimizer"],
        learning_rate=lr,
        seq_len=seq_len,
        model_config=clean_model_config,
        data_config=ds_block,
    )
