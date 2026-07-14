"""Non-stationary bandit learners: Discounted UCB, Sliding-Window TS.

When environment switches by CalBlock, full-history TS/UCB lag; these actively forget old rewards.
Reward scale: R = -Ŷ_min (consistent with continuous path).
"""

from __future__ import annotations

from collections import deque

import numpy as np
import pandas as pd


def run_stationary_or_ns_bandit(
    days: pd.DataFrame,
    daily_mu: dict,
    arms: list[str],
    algo: str,
    seed: int = 42,
    *,
    gamma: float = 0.95,
    window: int = 60,
    xi: float = 0.5,
    noise_std: float = 0.05,
) -> dict:
    """
    algo ∈ {TS, UCB, D-UCB, SW-TS, oracle, always_null, always_swap40}

    D-UCB: Garivier–Moulines discounted counts
    SW-TS: Gaussian TS using only the last window observations per arm
    """
    rng = np.random.default_rng(seed)
    K = len(arms)
    idx = {a: i for i, a in enumerate(arms)}

    # Full history (TS/UCB)
    n_pulls = np.zeros(K)
    sum_r = np.zeros(K)

    # D-UCB: discounted sufficient statistics (recursive)
    n_disc = np.zeros(K)
    sum_disc = np.zeros(K)

    # SW-TS: last window rewards per arm
    hist: list[deque] = [deque(maxlen=window) for _ in range(K)]

    cum, total = [], 0.0
    chosen = []
    step_meta = []  # per step: date, CalBlock, instant regret
    t_eff = 0  # effective steps (days with μ)

    for _, row in days.iterrows():
        key = str(row["日期"])
        if key not in daily_mu:
            continue
        yhats = daily_mu[key]
        arms_today = [a for a in arms if a in yhats]
        if not arms_today:
            continue
        opt = min(arms_today, key=lambda a: yhats[a])
        opt_r = -float(yhats[opt])
        local = [idx[a] for a in arms_today]
        cal = row["CalBlock"] if "CalBlock" in row.index else ""

        if algo == "oracle":
            a_idx = idx[opt]
        elif algo == "always_null":
            a_idx = idx["null"]
        elif algo == "always_swap40":
            a_idx = idx.get("swap40", local[0])
        elif algo == "TS":
            a_idx = _select_ts(rng, local, n_pulls, sum_r)
        elif algo == "UCB":
            a_idx = _select_ucb(local, n_pulls, sum_r, t_eff)
        elif algo == "D-UCB":
            a_idx = _select_ducb(local, n_disc, sum_disc, xi=xi)
        elif algo == "SW-TS":
            a_idx = _select_sw_ts(rng, local, hist)
        else:
            raise ValueError(algo)

        arm = arms[a_idx]
        r = -float(yhats[arm])
        r_obs = float(rng.normal(r, noise_std))

        # Update full history
        n_pulls[a_idx] += 1
        sum_r[a_idx] += r_obs

        # Update discounted stats: scale all by γ, then add today
        n_disc *= gamma
        sum_disc *= gamma
        n_disc[a_idx] += 1.0
        sum_disc[a_idx] += r_obs

        # Update sliding window
        hist[a_idx].append(r_obs)

        inst = max(0.0, opt_r - r)
        total += inst
        cum.append(total)
        chosen.append(arm)
        step_meta.append(
            {
                "date": key,
                "CalBlock": str(cal),
                "arm": arm,
                "opt": opt,
                "instant_regret": round(inst, 6),
            }
        )
        t_eff += 1

    return {
        "algo": algo,
        "final_regret_min": round(total, 4),
        "n_steps": len(chosen),
        "arm_counts": {a: int((np.array(chosen) == a).sum()) for a in arms},
        "cumulative_regret": cum,
        "step_meta": step_meta,
        "unit": "minutes",
        "params": {"gamma": gamma, "window": window, "xi": xi} if algo in ("D-UCB", "SW-TS") else {},
    }


def _select_ts(rng, local, n_pulls, sum_r) -> int:
    best_j, best_s = local[0], -1e9
    for j in local:
        if n_pulls[j] <= 0:
            s = rng.normal(-5, 1)
        else:
            mu = sum_r[j] / n_pulls[j]
            s = rng.normal(mu, 1 / np.sqrt(n_pulls[j]))
        if s > best_s:
            best_s, best_j = s, j
    return best_j


def _select_ucb(local, n_pulls, sum_r, t_eff) -> int:
    unpulled = [j for j in local if n_pulls[j] <= 0]
    if unpulled:
        return unpulled[0]
    t = max(t_eff + 1, 2)
    return max(
        local,
        key=lambda j: sum_r[j] / n_pulls[j] + np.sqrt(2 * np.log(t) / n_pulls[j]),
    )


def _select_ducb(local, n_disc, sum_disc, *, xi: float) -> int:
    """Discounted UCB: unpulled arms first; else X + 2√(ξ log(N+)/n)."""
    unpulled = [j for j in local if n_disc[j] < 1e-8]
    if unpulled:
        return unpulled[0]
    n_plus = float(n_disc.sum())
    log_term = np.log(max(n_plus, 2.0))

    def score(j: int) -> float:
        n = max(n_disc[j], 1e-8)
        x = sum_disc[j] / n
        return x + 2.0 * np.sqrt(xi * log_term / n)

    return max(local, key=score)


def _select_sw_ts(rng, local, hist: list[deque]) -> int:
    best_j, best_s = local[0], -1e9
    for j in local:
        h = hist[j]
        if len(h) == 0:
            s = rng.normal(-5, 1)
        else:
            arr = np.asarray(h, dtype=float)
            mu = float(arr.mean())
            # Within-window sample std; use prior scale if too small
            sd = float(arr.std(ddof=1)) if len(arr) > 1 else 1.0
            sd = max(sd, 0.05) / np.sqrt(len(arr))
            s = rng.normal(mu, sd)
        if s > best_s:
            best_s, best_j = s, j
    return best_j


def regret_by_calblock(step_meta: list[dict]) -> dict[str, dict]:
    """Aggregate instant regret by CalBlock."""
    from collections import defaultdict

    buckets: dict[str, list[float]] = defaultdict(list)
    for s in step_meta:
        buckets[s["CalBlock"]].append(s["instant_regret"])
    out = {}
    for b, vals in buckets.items():
        out[b] = {
            "n_days": len(vals),
            "sum_regret": round(float(np.sum(vals)), 4),
            "mean_instant": round(float(np.mean(vals)), 6),
        }
    return out


def regret_near_block_boundaries(
    step_meta: list[dict],
    days: pd.DataFrame,
    radius: int = 10,
) -> dict:
    """
    Cumulative instant regret within ±radius days of block boundaries.
    Boundary = day when CalBlock changes relative to previous day.
    """
    if not step_meta:
        return {"boundaries": [], "near_sum": 0.0, "far_sum": 0.0}

    # Effective days in calendar order
    order = [s["date"] for s in step_meta]
    blocks = [s["CalBlock"] for s in step_meta]
    inst = [s["instant_regret"] for s in step_meta]
    n = len(order)

    boundary_idx = []
    for i in range(1, n):
        if blocks[i] != blocks[i - 1]:
            boundary_idx.append(i)

    near = np.zeros(n, dtype=bool)
    for b in boundary_idx:
        lo, hi = max(0, b - radius), min(n, b + radius + 1)
        near[lo:hi] = True

    near_sum = float(np.sum(np.asarray(inst)[near]))
    far_sum = float(np.sum(np.asarray(inst)[~near]))
    return {
        "radius": radius,
        "n_boundaries": len(boundary_idx),
        "boundary_dates": [order[i] for i in boundary_idx],
        "boundary_blocks": [f"{blocks[i-1]}→{blocks[i]}" for i in boundary_idx],
        "n_near_days": int(near.sum()),
        "n_far_days": int((~near).sum()),
        "near_sum_regret": round(near_sum, 4),
        "far_sum_regret": round(far_sum, 4),
        "near_mean_instant": round(near_sum / max(int(near.sum()), 1), 6),
        "far_mean_instant": round(far_sum / max(int((~near).sum()), 1), 6),
    }
