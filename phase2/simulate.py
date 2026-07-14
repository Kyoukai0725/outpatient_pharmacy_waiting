"""Forward simulation from calibrated CPTs + posterior predictive checks."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from phase2.causal_graph import EXOGENOUS, ENDOGENOUS, PARENTS
from phase2.config import CARDINALITY


def _cfg_key(parents: tuple[str, ...], state: dict) -> str:
    return "|".join(str(state[p]) for p in parents)


def _sample_from_probs(probs: list[float], rng: np.random.Generator) -> int:
    p = np.asarray(probs, dtype=float)
    p = p / p.sum()
    return int(rng.choice(len(p), p=p))


def simulate_one(
    cpts: dict,
    exogenous: dict[str, int] | None = None,
    rng: np.random.Generator | None = None,
) -> dict[str, int]:
    rng = rng or np.random.default_rng()
    state: dict[str, int] = {}

    for var in EXOGENOUS:
        if exogenous and var in exogenous:
            state[var] = int(exogenous[var])
        else:
            probs = cpts[var][""]
            state[var] = _sample_from_probs(probs, rng)

    for var in ENDOGENOUS:
        key = _cfg_key(PARENTS[var], state)
        probs = cpts[var][key]
        state[var] = _sample_from_probs(probs, rng)

    return state


def simulate_batch(
    cpts: dict,
    n: int,
    exogenous: dict[str, int] | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = [simulate_one(cpts, exogenous=exogenous, rng=rng) for _ in range(n)]
    return pd.DataFrame(rows)


def predict_y_distribution(
    cpts: dict,
    row: pd.Series,
) -> np.ndarray:
    """Conditional distribution of Y0 given exogenous + observed W0,D0,V0; marginalize if stages missing."""
    state = {v: int(row[v]) for v in EXOGENOUS if v in row.index}
    n_y = CARDINALITY["Y0"]
    probs_y = np.zeros(n_y)

    if all(v in row.index and pd.notna(row[v]) for v in ("W0", "D0", "V0")):
        key = _cfg_key(PARENTS["Y0"], {**state, "W0": int(row["W0"]), "D0": int(row["D0"]), "V0": int(row["V0"])})
        return np.asarray(cpts["Y0"][key], dtype=float)

    # Missing stages: marginalize over all W,D,V combinations
    for w in range(CARDINALITY["W0"]):
        pw = cpts["W0"][_cfg_key(PARENTS["W0"], state)][w]
        for d in range(CARDINALITY["D0"]):
            pd_ = cpts["D0"][_cfg_key(PARENTS["D0"], state)][d]
            for v in range(CARDINALITY["V0"]):
                pv = cpts["V0"][_cfg_key(PARENTS["V0"], state)][v]
                y_probs = np.asarray(
                    cpts["Y0"][_cfg_key(PARENTS["Y0"], {**state, "W0": w, "D0": d, "V0": v})],
                    dtype=float,
                )
                probs_y += pw * pd_ * pv * y_probs
    return probs_y / probs_y.sum()


def evaluate(test_df: pd.DataFrame, cpts: dict) -> dict:
    loglik = []
    pred_class = []
    for _, row in test_df.iterrows():
        py = predict_y_distribution(cpts, row)
        y = int(row["Y0"])
        loglik.append(float(np.log(py[y] + 1e-12)))
        pred_class.append(int(np.argmax(py)))

    acc = float(np.mean(np.asarray(pred_class) == test_df["Y0"].astype(int).to_numpy()))
    return {
        "n_test": int(len(test_df)),
        "loglik_mean": round(float(np.mean(loglik)), 4),
        "accuracy_argmax": round(acc, 4),
        "y_true_counts": {str(k): int(v) for k, v in test_df["Y0"].value_counts().sort_index().items()},
        "y_pred_counts": {str(k): int(v) for k, v in pd.Series(pred_class).value_counts().sort_index().items()},
    }


def load_cpts(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)
