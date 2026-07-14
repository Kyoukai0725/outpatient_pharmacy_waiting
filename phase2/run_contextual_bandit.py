"""S3: Run bandit with daily lookup A(c); main loop asserts no POMIS+ search."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from phase2.config import DATA_DIR
from phase2.ns_arms import arm_sets, oracle_mu_table
from phase2.ns_mechanisms import CAL_BLOCKS, annotate_regimes, cal_block
from phase2.run_ns_bandit import (
    align_with_swap40,
    best_arm,
    plot_regret,
    precompute_mu,
    reward_of,
    run_bandit_on_days,
)
from phase2.temporal_scm import load_scm_for_calblock

PHASE2_DIR = DATA_DIR / "phase2"
OUT_DIR = PHASE2_DIR / "ns_bandit_results" / "contextual_lookup"
TABLE_PATH = PHASE2_DIR / "contextual_graph" / "pomis_plus_table.json"
TAU_RUSH_DAY = 0.30

_POMIS_SEARCH_CALLS = 0


def assert_no_pomis_search():
    """For tests/monkey-patch: fail if POMISplusSEQ is invoked."""
    global _POMIS_SEARCH_CALLS
    _POMIS_SEARCH_CALLS += 1
    raise RuntimeError("POMIS+ search forbidden in bandit loop")


def day_context_label(calblock: str, rush_frac: float) -> str:
    regime = "rush" if float(rush_frac) >= TAU_RUSH_DAY else "other"
    return f"{calblock}_{regime}"


def build_day_index(disc: pd.DataFrame) -> pd.DataFrame:
    d = disc.copy()
    d["日期"] = pd.to_datetime(d["日期"]).dt.date
    g = (
        d.groupby("日期", as_index=False)
        .agg(
            n_rx=("处方编号", "count"),
            weekday=("日期", lambda s: pd.Timestamp(s.iloc[0]).weekday()),
            rush_frac=("MorningRush", "mean"),
        )
    )
    g["日期_ts"] = pd.to_datetime(g["日期"])
    g["CalBlock"] = g["日期_ts"].map(cal_block)
    g["is_weekday"] = g["weekday"] < 5
    g["context"] = [day_context_label(b, r) for b, r in zip(g["CalBlock"], g["rush_frac"])]
    g = g.sort_values("日期_ts").reset_index(drop=True)
    return g[g["CalBlock"].isin(CAL_BLOCKS)].reset_index(drop=True)


def run_lookup_bandit(
    days: pd.DataFrame,
    table: dict,
    mu_cache: dict,
    algo: str,
    seed: int = 0,
) -> dict:
    assert table["meta"].get("forbid_online_pomis_search") is True
    calls0 = _POMIS_SEARCH_CALLS

    by_ctx = table["by_context"]
    universe = sorted(set().union(*[set(v["arms"]) for v in by_ctx.values()]))
    index = {a: i for i, a in enumerate(universe)}
    K = len(universe)
    rng = np.random.default_rng(seed)
    n_pulls = np.zeros(K)
    sum_r = np.zeros(K)
    cum, total = [], 0.0
    chosen, rewards, ctxs = [], [], []

    for t, row in days.iterrows():
        # --- LOOKUP ONLY (search forbidden) ---
        ctx = row["context"]
        if ctx not in by_ctx:
            ctx = f"{row['CalBlock']}_other"
        arms_today = [a for a in by_ctx[ctx]["arms"] if a in index]
        if not arms_today:
            arms_today = ["null"]

        block, wd = row["CalBlock"], bool(row["is_weekday"])
        opt_arm, opt_r = best_arm(mu_cache, block, wd, arms_today)
        local = [index[a] for a in arms_today]
        mu_hat = np.full(K, 0.5)
        m = n_pulls > 0
        mu_hat[m] = sum_r[m] / n_pulls[m]

        if algo == "TS":
            best_j, best_s = local[0], -1e9
            for j in local:
                s = rng.random() if n_pulls[j] <= 0 else rng.normal(mu_hat[j], 1 / np.sqrt(n_pulls[j]))
                if s > best_s:
                    best_s, best_j = s, j
            a_idx = best_j
        elif algo == "UCB":
            unpulled = [j for j in local if n_pulls[j] <= 0]
            if unpulled:
                a_idx = unpulled[0]
            else:
                a_idx = max(
                    local,
                    key=lambda j: mu_hat[j] + np.sqrt(2 * np.log(max(t + 1, 2)) / n_pulls[j]),
                )
        elif algo == "oracle":
            a_idx = index[opt_arm]
        else:
            raise ValueError(algo)

        arm = universe[a_idx]
        r = reward_of(mu_cache, block, wd, arm)
        r_obs = float(np.clip(rng.normal(r, 0.02), 0, 1.5))
        n_pulls[a_idx] += 1
        sum_r[a_idx] += r_obs
        total += max(0.0, opt_r - r)
        cum.append(total)
        chosen.append(arm)
        rewards.append(r)
        ctxs.append(ctx)

    assert _POMIS_SEARCH_CALLS == calls0
    return {
        "algo": algo,
        "mode": "contextual_lookup",
        "final_regret": round(total, 4),
        "mean_reward": round(float(np.mean(rewards)), 4),
        "arm_counts": {a: int((np.array(chosen) == a).sum()) for a in universe},
        "context_counts": pd.Series(ctxs).value_counts().astype(int).to_dict(),
        "pomis_search_calls": 0,
        "cumulative_regret": cum,
        "arms_universe": universe,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-mc", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Monkey-patch: fail if someone mistakenly imports and calls POMISplusSEQ
    try:
        import npsem.pomis_plus as pp

        pp.POMISplusSEQ = assert_no_pomis_search  # type: ignore
    except Exception:
        pass

    print("[1] Loading lookup table...")
    with TABLE_PATH.open(encoding="utf-8") as f:
        table = json.load(f)
    print("  contexts:", list(table["by_context"].keys()))

    print("[2] Day index (τ=0.30)...")
    disc = pd.read_parquet(PHASE2_DIR / "rx_discretized.parquet")
    dates = pd.read_parquet(DATA_DIR / "rx_level.parquet", columns=["处方编号", "日期"])
    disc = disc.merge(dates, on="处方编号", how="left")
    disc = annotate_regimes(disc, disc["日期"])
    days = build_day_index(disc)
    print(f"  days={len(days)} ctx={days['context'].value_counts().to_dict()}")

    print("[3] oracle μ...")
    all_arms = sorted(set().union(*[set(v["arms"]) for v in table["by_context"].values()]))
    # Also union Brute control arms
    all_arms = sorted(set(all_arms) | set(arm_sets()["Brute"]) | set(arm_sets()["POMIS+"]))
    mu_cache = precompute_mu(days, all_arms, n_mc=args.n_mc, seed=args.seed)

    print("[4] contextual lookup bandit...")
    experiments = {}
    for algo in ("TS", "UCB", "oracle"):
        key = f"CtxLookup---{algo}"
        experiments[key] = run_lookup_bandit(days, table, mu_cache, algo, seed=args.seed)
        print(f"  {key}: regret={experiments[key]['final_regret']} search_calls={experiments[key]['pomis_search_calls']}")

    # Fixed arm-set controls (non-lookup)
    for set_name in ("POMIS+", "Brute"):
        arms = arm_sets()[set_name]
        for algo in ("TS", "UCB"):
            key = f"Fixed_{set_name}---{algo}"
            experiments[key] = run_bandit_on_days(days, arms, mu_cache, algo, seed=args.seed)
            experiments[key]["mode"] = "fixed_arm_set"

    print("[5] Saving...")
    summary = {
        k: {kk: vv for kk, vv in v.items() if kk != "cumulative_regret"}
        for k, v in experiments.items()
    }
    with (OUT_DIR / "bandit_summary.json").open("w", encoding="utf-8") as f:
        json.dump({"n_days": len(days), "experiments": summary}, f, ensure_ascii=False, indent=2)
    with (OUT_DIR / "cumulative_regret.json").open("w", encoding="utf-8") as f:
        json.dump({k: v["cumulative_regret"] for k, v in experiments.items()}, f)
    plot_regret(
        {k: v for k, v in experiments.items()},
        OUT_DIR / "cumulative_regret.png",
    )
    board = sorted(
        ((k, v["final_regret"], v.get("mean_reward")) for k, v in experiments.items()),
        key=lambda x: x[1],
    )
    print("\n=== Leaderboard ===")
    for k, fr, mr in board:
        print(f"  {k:32s} regret={fr:8.3f}  meanR={mr}")
    print("done", OUT_DIR)


if __name__ == "__main__":
    main()
