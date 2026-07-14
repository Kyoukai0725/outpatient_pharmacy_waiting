"""
Gibbs MCMC calibration of discrete SCM conditional probability tables (CPTs).

For each node v | Pa(v) ~ Multinomial, prior Dirichlet(α,...,α).
"""

from __future__ import annotations

import json
from collections import defaultdict
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from phase2.causal_graph import PARENTS, EXOGENOUS
from phase2.config import CARDINALITY, DIRICHLET_ALPHA, MCMC_BURN, MCMC_DRAWS


def _n_levels(node: str) -> int:
    return CARDINALITY[node]


def _parent_configs(parents: tuple[str, ...]) -> list[tuple[int, ...]]:
    if not parents:
        return [()]
    ranges = [range(_n_levels(p)) for p in parents]
    return list(product(*ranges))


def _config_key(values: tuple[int, ...]) -> str:
    return "|".join(str(v) for v in values)


def build_count_tables(df: pd.DataFrame) -> dict[str, dict[str, np.ndarray]]:
    counts: dict[str, dict[str, np.ndarray]] = {}
    for node, parents in PARENTS.items():
        node_counts: dict[str, np.ndarray] = {}
        for cfg in _parent_configs(parents):
            key = _config_key(cfg)
            if parents:
                mask = np.ones(len(df), dtype=bool)
                for p, val in zip(parents, cfg):
                    mask &= df[p].to_numpy() == val
                subset = df.loc[mask, node].astype(int).to_numpy()
            else:
                subset = df[node].astype(int).to_numpy()
            cnt = np.bincount(subset, minlength=_n_levels(node))
            node_counts[key] = cnt
        counts[node] = node_counts
    return counts


def _sample_dirichlet(counts: np.ndarray, alpha: float, rng: np.random.Generator) -> np.ndarray:
    return rng.dirichlet(counts + alpha)


def gibbs_sample_cpts(
    count_tables: dict[str, dict[str, np.ndarray]],
    n_draws: int = MCMC_DRAWS,
    n_burn: int = MCMC_BURN,
    alpha: float = DIRICHLET_ALPHA,
    seed: int = 42,
) -> tuple[dict, dict]:
    rng = np.random.default_rng(seed)
    current: dict[str, dict[str, np.ndarray]] = {}
    for node, table in count_tables.items():
        current[node] = {}
        for key, cnt in table.items():
            current[node][key] = _sample_dirichlet(cnt, alpha, rng)

    stored: dict[str, list[dict[str, np.ndarray]]] = {node: [] for node in count_tables}

    total = n_burn + n_draws
    for step in range(total):
        for node, table in count_tables.items():
            for key, cnt in table.items():
                current[node][key] = _sample_dirichlet(cnt, alpha, rng)
        if step >= n_burn:
            idx = step - n_burn
            for node in count_tables:
                stored[node].append({k: v.copy() for k, v in current[node].items()})

    posterior_mean: dict[str, dict[str, list[float]]] = {}
    for node, draws in stored.items():
        posterior_mean[node] = {}
        keys = count_tables[node].keys()
        for key in keys:
            arr = np.stack([d[key] for d in draws], axis=0)
            posterior_mean[node][key] = arr.mean(axis=0).tolist()

    return posterior_mean, stored


def cpts_to_json(cpts: dict) -> dict:
    return {
        node: {key: [round(float(p), 6) for p in probs] for key, probs in table.items()}
        for node, table in cpts.items()
    }


def save_calibration(
    cpts: dict,
    count_tables: dict,
    meta: dict,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "calibrated_cpts.json").open("w", encoding="utf-8") as f:
        json.dump(cpts_to_json(cpts), f, ensure_ascii=False, indent=2)

    counts_json = {
        node: {key: cnt.tolist() for key, cnt in table.items()}
        for node, table in count_tables.items()
    }
    with (output_dir / "cpt_count_tables.json").open("w", encoding="utf-8") as f:
        json.dump(counts_json, f, ensure_ascii=False, indent=2)

    with (output_dir / "mcmc_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def train_test_split(df: pd.DataFrame, test_frac: float = 0.2, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df))
    split = int(len(df) * (1 - test_frac))
    train_idx = idx[:split]
    test_idx = idx[split:]
    return df.iloc[train_idx].copy(), df.iloc[test_idx].copy()
