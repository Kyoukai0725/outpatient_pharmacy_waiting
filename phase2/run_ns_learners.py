"""Non-stationary learner comparison: TS/UCB vs D-UCB / SW-TS (continuous reward μ, no bonus).

Focus on whether regret near CalBlock switches is reduced by discounting/sliding window.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from phase2.config import DATA_DIR
from phase2.continuous_reward import build_feature_frame
from phase2.ns_arms import arm_sets
from phase2.ns_learners import (
    regret_by_calblock,
    regret_near_block_boundaries,
    run_stationary_or_ns_bandit,
)
from phase2.ns_mechanisms import annotate_regimes
from phase2.run_contextual_bandit import build_day_index
from phase2.staff_schedule import load_daily_staff

PHASE2_DIR = DATA_DIR / "phase2"
MU_PATH = PHASE2_DIR / "ns_bandit_results" / "continuous_reward" / "daily_mu_minutes.json"
OUT_DIR = PHASE2_DIR / "ns_bandit_results" / "ns_learners"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gamma", type=float, default=0.95, help="D-UCB discount factor")
    parser.add_argument("--window", type=int, default=60, help="SW-TS window length (days)")
    parser.add_argument("--xi", type=float, default=0.5, help="D-UCB exploration coefficient")
    parser.add_argument("--radius", type=int, default=10, help="Block-boundary neighborhood radius (days)")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not MU_PATH.exists():
        raise FileNotFoundError(f"Need {MU_PATH} first (python3 -m phase2.run_continuous_bandit)")

    with MU_PATH.open(encoding="utf-8") as f:
        daily_mu = json.load(f)

    print("[1] Loading day index...")
    rx = pd.read_parquet(DATA_DIR / "rx_level.parquet")
    disc = pd.read_parquet(PHASE2_DIR / "rx_discretized.parquet")
    rx = rx.merge(disc[["处方编号", "Peak0", "Load0"]], on="处方编号", how="left")
    rx = build_feature_frame(rx, load_daily_staff())
    rx = annotate_regimes(rx, rx["日期"])
    days = build_day_index(rx)

    arms = [a for a in arm_sets()["POMIS+"] if not a.startswith("load_do")]
    algos = ["TS", "UCB", "D-UCB", "SW-TS", "oracle", "always_swap40", "always_null"]

    print(f"[2] Running learner comparison gamma={args.gamma} window={args.window}...")
    results = {}
    for algo in algos:
        res = run_stationary_or_ns_bandit(
            days,
            daily_mu,
            arms,
            algo,
            seed=args.seed,
            gamma=args.gamma,
            window=args.window,
            xi=args.xi,
        )
        by_block = regret_by_calblock(res["step_meta"])
        near = regret_near_block_boundaries(res["step_meta"], days, radius=args.radius)
        results[algo] = {
            "final_regret_min": res["final_regret_min"],
            "n_steps": res["n_steps"],
            "arm_counts": res["arm_counts"],
            "params": res.get("params", {}),
            "by_calblock": by_block,
            "near_boundary": near,
            "cumulative_regret": res["cumulative_regret"],
        }
        print(
            f"  {algo:14s} regret={res['final_regret_min']:8.2f}  "
            f"near={near['near_sum_regret']:6.2f}  far={near['far_sum_regret']:6.2f}"
        )

    # Plot
    plt.figure(figsize=(9, 5))
    for name in ("TS", "UCB", "D-UCB", "SW-TS", "always_swap40", "oracle"):
        r = results[name]
        plt.plot(
            r["cumulative_regret"],
            label=f"{name} ({r['final_regret_min']:.1f})",
            linewidth=2 if name in ("D-UCB", "SW-TS") else 1.2,
        )
    plt.xlabel("Day")
    plt.ylabel("Cumulative regret (minutes)")
    plt.title(
        f"Non-stationary learners (γ={args.gamma}, W={args.window})\n"
        "vs stationary TS/UCB — continuous R=-Ŷ, no bonus"
    )
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "ns_learners_regret.png", dpi=140)
    plt.close()

    # Boundary comparison table
    comparison = {
        "hypothesis": (
            "D-UCB / SW-TS forget old CalBlock rewards → lower regret near block switches"
        ),
        "params": {"gamma": args.gamma, "window": args.window, "xi": args.xi, "radius": args.radius},
        "mu_source": str(MU_PATH),
        "no_bonus": True,
        "algos": {
            k: {kk: vv for kk, vv in v.items() if kk != "cumulative_regret"}
            for k, v in results.items()
        },
        "headline": {
            a: {
                "final": results[a]["final_regret_min"],
                "near_boundary": results[a]["near_boundary"]["near_sum_regret"],
                "far": results[a]["near_boundary"]["far_sum_regret"],
            }
            for a in ("TS", "UCB", "D-UCB", "SW-TS", "always_swap40")
        },
    }
    # Improvement relative to TS
    ts_f = results["TS"]["final_regret_min"]
    ts_n = results["TS"]["near_boundary"]["near_sum_regret"]
    comparison["vs_TS"] = {
        a: {
            "final_delta": round(results[a]["final_regret_min"] - ts_f, 4),
            "near_delta": round(results[a]["near_boundary"]["near_sum_regret"] - ts_n, 4),
        }
        for a in ("UCB", "D-UCB", "SW-TS", "always_swap40")
    }

    with (OUT_DIR / "ns_learners_summary.json").open("w", encoding="utf-8") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)
    with (OUT_DIR / "cumulative_regret.json").open("w", encoding="utf-8") as f:
        json.dump({k: v["cumulative_regret"] for k, v in results.items()}, f)

    print("[3] vs TS:", comparison["vs_TS"])
    print("done →", OUT_DIR)


if __name__ == "__main__":
    main()
