"""Non-stationary causal bandit: switch θ(c) by real calendar; compare Brute/POMIS/POMIS+ × TS/UCB."""

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
_NS = _ROOT / "vendor" / "NS-SCMMAB-main"
if str(_NS) not in sys.path:
    sys.path.insert(0, str(_NS))

from phase2.config import DATA_DIR
from phase2.ns_arms import (
    arm_sets,
    intervention_from_arm,
    oracle_mu_table,
    save_arm_catalog,
)
from phase2.ns_mechanisms import CAL_BLOCKS, annotate_regimes, cal_block
from phase2.temporal_scm import load_scm_for_calblock

PHASE2_DIR = DATA_DIR / "phase2"
OUT_DIR = PHASE2_DIR / "ns_bandit_results"


def thompson_sampling(mu_hat: np.ndarray, n_pulls: np.ndarray, rng: np.random.Generator) -> int:
    """Gaussian TS on [0,1]-ish rewards with unit prior variance."""
    K = len(mu_hat)
    samples = np.zeros(K)
    for a in range(K):
        if n_pulls[a] <= 0:
            samples[a] = rng.random()
        else:
            se = 1.0 / np.sqrt(n_pulls[a])
            samples[a] = rng.normal(mu_hat[a], se)
    return int(np.argmax(samples))


def ucb(mu_hat: np.ndarray, n_pulls: np.ndarray, t: int) -> int:
    K = len(mu_hat)
    if np.any(n_pulls == 0):
        return int(np.argmin(n_pulls))
    bonus = np.sqrt(2.0 * np.log(max(t, 2)) / n_pulls)
    return int(np.argmax(mu_hat + bonus))


def build_day_index(disc: pd.DataFrame) -> pd.DataFrame:
    """One row per business day: CalBlock, weekday flag, prescription count."""
    d = disc.copy()
    d["日期"] = pd.to_datetime(d["日期"]).dt.date
    g = (
        d.groupby("日期", as_index=False)
        .agg(
            n_rx=("处方编号", "count"),
            weekday=("日期", lambda s: pd.Timestamp(s.iloc[0]).weekday()),
        )
    )
    g["日期_ts"] = pd.to_datetime(g["日期"])
    g["CalBlock"] = g["日期_ts"].map(cal_block)
    g["is_weekday"] = g["weekday"] < 5
    g = g.sort_values("日期_ts").reset_index(drop=True)
    # Keep only blocks with ns CPTs
    g = g[g["CalBlock"].isin(CAL_BLOCKS)].reset_index(drop=True)
    return g


def precompute_mu(
    days: pd.DataFrame,
    all_arms: list[str],
    *,
    n_mc: int = 200,
    seed: int = 0,
) -> dict[str, dict[str, dict]]:
    """μ[calblock][weekday|weekend][arm] = reward stats。"""
    cache: dict[str, dict[str, dict]] = {}
    blocks = sorted(days["CalBlock"].unique())
    for bi, block in enumerate(blocks):
        scm = load_scm_for_calblock(block)
        cache[block] = {}
        for wd_flag, label in ((True, "weekday"), (False, "weekend")):
            print(f"  oracle μ {block}/{label} ...")
            cache[block][label] = oracle_mu_table(
                scm,
                all_arms,
                n_mc=n_mc,
                seed=seed + bi * 100 + int(wd_flag) * 50,
                weekday=wd_flag,
            )
    return cache


def reward_of(mu_cache: dict, calblock: str, is_weekday: bool, arm: str) -> float:
    label = "weekday" if is_weekday else "weekend"
    return float(mu_cache[calblock][label][arm]["reward"])


def best_arm(mu_cache: dict, calblock: str, is_weekday: bool, arms: list[str]) -> tuple[str, float]:
    label = "weekday" if is_weekday else "weekend"
    best, best_r = arms[0], -1e9
    for a in arms:
        r = float(mu_cache[calblock][label][a]["reward"])
        if r > best_r:
            best, best_r = a, r
    return best, best_r


def run_bandit_on_days(
    days: pd.DataFrame,
    arms: list[str],
    mu_cache: dict,
    algo: str,
    *,
    seed: int = 0,
) -> dict:
    rng = np.random.default_rng(seed)
    K = len(arms)
    n_pulls = np.zeros(K)
    sum_r = np.zeros(K)
    cum_regret = []
    total_regret = 0.0
    chosen = []
    rewards = []

    for t, row in days.iterrows():
        block = row["CalBlock"]
        wd = bool(row["is_weekday"])
        opt_arm, opt_r = best_arm(mu_cache, block, wd, arms)

        mu_hat = np.divide(sum_r, np.maximum(n_pulls, 1), where=n_pulls > 0, out=np.full(K, 0.5))
        if algo == "TS":
            a_idx = thompson_sampling(mu_hat, n_pulls, rng)
        elif algo == "UCB":
            a_idx = ucb(mu_hat, n_pulls, int(t) + 1)
        elif algo == "oracle":
            a_idx = arms.index(opt_arm)
        elif algo == "always_null":
            a_idx = arms.index("null") if "null" in arms else 0
        elif algo == "always_swap40":
            a_idx = arms.index("swap40") if "swap40" in arms else 0
        else:
            raise ValueError(algo)

        arm = arms[a_idx]
        r = reward_of(mu_cache, block, wd, arm)
        # stochastic observation around true μ
        r_obs = float(np.clip(rng.normal(r, 0.02), 0.0, 1.5))
        n_pulls[a_idx] += 1
        sum_r[a_idx] += r_obs
        regret = opt_r - r
        total_regret += max(0.0, regret)
        cum_regret.append(total_regret)
        chosen.append(arm)
        rewards.append(r)

    return {
        "algo": algo,
        "arms": arms,
        "cumulative_regret": cum_regret,
        "final_regret": round(total_regret, 4),
        "mean_reward": round(float(np.mean(rewards)), 4),
        "arm_counts": {a: int((np.array(chosen) == a).sum()) for a in arms},
        "chosen": chosen,
    }


def plot_regret(results: dict, out_path: Path) -> None:
    plt.figure(figsize=(9, 5))
    for name, res in results.items():
        y = res["cumulative_regret"]
        plt.plot(y, label=f"{name} (final={res['final_regret']:.2f})")
    plt.xlabel("Day (calendar order)")
    plt.ylabel("Cumulative regret")
    plt.title("NS-SCMMAB pharmacy layout bandit")
    plt.legend(fontsize=8)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=140)
    plt.close()


def align_with_swap40(mu_cache: dict) -> dict:
    """Stationary approximation: whether oracle prefers swap40 on weekday per block."""
    rows = []
    for block, by_wd in mu_cache.items():
        tab = by_wd["weekday"]
        ranked = sorted(tab.items(), key=lambda kv: -kv[1]["reward"])
        rows.append(
            {
                "calblock": block,
                "best_arm": ranked[0][0],
                "best_reward": ranked[0][1]["reward"],
                "swap40_reward": tab.get("swap40", {}).get("reward"),
                "null_reward": tab.get("null", {}).get("reward"),
                "swap40_rank": next(i for i, (a, _) in enumerate(ranked) if a == "swap40") + 1
                if "swap40" in tab
                else None,
            }
        )
    return {"weekday_oracle_by_block": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-mc", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-days", type=int, default=0, help="0=all days")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/6] Arm catalog + POMIS...")
    items = pd.read_parquet(
        DATA_DIR / "item_level.parquet",
        columns=[
            "处方编号",
            "drugid",
            "药品名称及规格",
            "品名规格",
            "是否机器_最终",
            "调配秒数_最终",
            "货架区域_最终",
        ],
    )
    catalog = save_arm_catalog(PHASE2_DIR / "ns_arm_catalog.json", items=items)
    sets = catalog["arm_sets"]
    all_arms = sorted(set().union(*[set(v) for v in sets.values()]))
    print("  arms:", all_arms)

    print("[2/6] Day index...")
    disc = pd.read_parquet(PHASE2_DIR / "rx_discretized.parquet")
    dates = pd.read_parquet(DATA_DIR / "rx_level.parquet", columns=["处方编号", "日期"])
    disc = disc.merge(dates, on="处方编号", how="left")
    disc = annotate_regimes(disc, disc["日期"])
    days = build_day_index(disc)
    if args.max_days > 0:
        days = days.head(args.max_days)
    print(f"  days={len(days)}, blocks={days['CalBlock'].value_counts().to_dict()}")

    print("[3/6] Precomputing oracle μ per CalBlock...")
    mu_cache = precompute_mu(days, all_arms, n_mc=args.n_mc, seed=args.seed)
    with (OUT_DIR / "oracle_mu.json").open("w", encoding="utf-8") as f:
        json.dump(mu_cache, f, ensure_ascii=False, indent=2)

    align = align_with_swap40(mu_cache)
    with (OUT_DIR / "align_swap40.json").open("w", encoding="utf-8") as f:
        json.dump(align, f, ensure_ascii=False, indent=2)
    print("  align:", align)

    print("[4/6] Running bandit...")
    experiments = {}
    for set_name, arms in sets.items():
        for algo in ("TS", "UCB"):
            key = f"{set_name}---{algo}"
            print(f"  {key} ...")
            experiments[key] = run_bandit_on_days(
                days, arms, mu_cache, algo, seed=args.seed
            )
            # drop long chosen list from summary later
    # baselines on POMIS+ arm set
    base_arms = sets["POMIS+"]
    for algo in ("always_null", "always_swap40", "oracle"):
        key = f"Baseline---{algo}"
        experiments[key] = run_bandit_on_days(
            days, base_arms, mu_cache, algo, seed=args.seed
        )

    print("[5/6] Plotting and saving...")
    # strip chosen lists for main json size
    summary = {}
    for k, v in experiments.items():
        summary[k] = {
            "algo": v["algo"],
            "arms": v["arms"],
            "final_regret": v["final_regret"],
            "mean_reward": v["mean_reward"],
            "arm_counts": v["arm_counts"],
            "cumulative_regret": v["cumulative_regret"],
        }
    with (OUT_DIR / "bandit_summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "n_days": len(days),
                "cal_blocks": days["CalBlock"].value_counts().to_dict(),
                "experiments": {
                    k: {kk: vv for kk, vv in s.items() if kk != "cumulative_regret"}
                    for k, s in summary.items()
                },
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    # full regret curves
    with (OUT_DIR / "cumulative_regret.json").open("w", encoding="utf-8") as f:
        json.dump({k: s["cumulative_regret"] for k, s in summary.items()}, f)

    plot_regret(summary, OUT_DIR / "cumulative_regret.png")

    # compact leaderboard
    board = sorted(
        ((k, s["final_regret"], s["mean_reward"]) for k, s in summary.items()),
        key=lambda x: x[1],
    )
    print("\n=== Regret leaderboard (lower better) ===")
    for k, fr, mr in board:
        print(f"  {k:28s} regret={fr:8.3f}  meanR={mr:.4f}")

    print("[6/6] done →", OUT_DIR)


if __name__ == "__main__":
    main()
