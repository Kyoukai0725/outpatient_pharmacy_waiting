"""S5: End-to-end closed loop — intervention sim kernel + continuous reward + dynamic graph lookup.

Three comparison curves (same daily_mu, same calendar, no bonus):
  1. Fixed_POMIS+     — daily arm set = arm_sets()["POMIS+"] (fixed G_sup)
  2. Dynamic_Gc       — daily arm set = pomis_plus_table[c] (lookup; no online POMIS)
  3. always_swap40    — single-arm baseline

Pre-check: if arm sets are identical across 8 contexts, Fixed ≈ Dynamic is expected, not a bug.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from phase2.config import DATA_DIR
from phase2.continuous_reward import (
    arm_to_scenario,
    build_feature_frame,
    check_arm_signal_gate_ttest,
    load_bundle,
    load_queue_model,
    simulate_day_under_arm,
)
from phase2.ns_arms import arm_sets
from phase2.ns_mechanisms import annotate_regimes
from phase2.run_contextual_bandit import build_day_index
from phase2.run_continuous_bandit import pooled_mu, precompute_daily_mu
from phase2.staff_schedule import load_daily_staff

PHASE2_DIR = DATA_DIR / "phase2"
TABLE_PATH = PHASE2_DIR / "contextual_graph" / "pomis_plus_table.json"
OUT_DIR = PHASE2_DIR / "ns_bandit_results" / "e2e_closed_loop"
CONT_MU_PATH = PHASE2_DIR / "ns_bandit_results" / "continuous_reward" / "daily_mu_minutes.json"


def diagnose_arm_sets(table: dict) -> dict:
    """Check whether arm sets are identical across 8 contexts."""
    by = table["by_context"]
    rows = []
    arm_tuples = []
    for ctx in sorted(by):
        arms = tuple(by[ctx]["arms"])
        arm_tuples.append(arms)
        rows.append(
            {
                "context": ctx,
                "n_arms": len(arms),
                "arms": list(arms),
                "removed_edge_ids": by[ctx].get("removed_edge_ids", []),
                "n_pomis": by[ctx].get("n_pomis"),
                "pomis_y2_sample": by[ctx].get("pomis_y2_sample"),
            }
        )
    unique = {t for t in arm_tuples}
    identical = len(unique) == 1
    shared = list(next(iter(unique))) if identical else None
    if identical and shared == ["null"]:
        implication = (
            "All contexts map to null only (no layout-feasible POMIS). "
            "Dynamic arm filter collapses exploration to baseline — "
            "G_c edge deletions do not yield distinct business arms."
        )
    elif identical:
        implication = (
            "Fixed_POMIS+ vs Dynamic_Gc expected to have ≈0 regret difference "
            "(same arms every day) — not a bug; dynamic graph changes edges not arm filter."
        )
    else:
        implication = (
            "Arm sets differ by context; Dynamic_Gc may change regret vs Fixed."
        )
    return {
        "n_contexts": len(by),
        "unique_arm_sets": len(unique),
        "arms_identical_across_contexts": identical,
        "shared_arms": shared,
        "by_context": rows,
        "implication": implication,
    }


def run_bandit_with_arm_policy(
    days: pd.DataFrame,
    daily_mu: dict,
    universe: list[str],
    algo: str,
    seed: int,
    *,
    arms_by_day: dict[str, list[str]] | None = None,
    fixed_arms: list[str] | None = None,
    force_arm: str | None = None,
    oracle_arms: list[str] | None = None,
) -> dict:
    """
    R = -Ŷ; oracle = smallest Ŷ among oracle_arms (or daily available arms).

    Arm policy:
      - force_arm: force this arm daily; regret vs oracle_arms / fixed_arms
      - arms_by_day: daily lookup
      - fixed_arms: fixed arm set
    Sort daily available arms as sorted for consistent TS/UCB paths.
    """
    rng = np.random.default_rng(seed)
    K = len(universe)
    idx = {a: i for i, a in enumerate(universe)}
    n_pulls = np.zeros(K)
    sum_r = np.zeros(K)
    cum, total = [], 0.0
    chosen = []

    for t, row in days.iterrows():
        key = str(row["日期"])
        if key not in daily_mu:
            continue
        yhats = daily_mu[key]

        if arms_by_day is not None:
            policy_arms = [a for a in arms_by_day.get(key, []) if a in yhats]
        elif fixed_arms is not None:
            policy_arms = [a for a in fixed_arms if a in yhats]
        else:
            policy_arms = [a for a in universe if a in yhats]
        # Unified order so same set → same TS/UCB path
        policy_arms = sorted(set(policy_arms))

        # Oracle comparison set: forced-arm policy still uses full arm set for regret
        if oracle_arms is not None:
            o_arms = sorted({a for a in oracle_arms if a in yhats})
        else:
            o_arms = policy_arms
        if not o_arms:
            continue
        opt = min(o_arms, key=lambda a: yhats[a])
        opt_r = -yhats[opt]

        if force_arm is not None:
            if force_arm not in yhats:
                continue
            a_idx = idx[force_arm]
        else:
            if not policy_arms:
                continue
            local = [idx[a] for a in policy_arms]
            mu_hat = np.full(K, -5.0)
            m = n_pulls > 0
            mu_hat[m] = sum_r[m] / n_pulls[m]

            if algo == "TS":
                best_j, best_s = local[0], -1e9
                for j in local:
                    s = (
                        rng.normal(-5, 1)
                        if n_pulls[j] <= 0
                        else rng.normal(mu_hat[j], 1 / np.sqrt(n_pulls[j]))
                    )
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
                        key=lambda j: mu_hat[j]
                        + np.sqrt(2 * np.log(max(t + 1, 2)) / n_pulls[j]),
                    )
            elif algo == "oracle":
                a_idx = idx[opt]
            else:
                raise ValueError(algo)

        arm = universe[a_idx]
        if arm not in yhats:
            continue
        r = -yhats[arm]
        r_obs = float(rng.normal(r, 0.05))
        n_pulls[a_idx] += 1
        sum_r[a_idx] += r_obs
        total += max(0.0, opt_r - r)
        cum.append(total)
        chosen.append(arm)

    return {
        "algo": algo if force_arm is None else f"always_{force_arm}",
        "final_regret_min": round(total, 4),
        "n_steps": len(chosen),
        "arm_counts": {a: int((np.array(chosen) == a).sum()) for a in universe},
        "cumulative_regret": cum,
        "unit": "minutes",
        "no_bonus": True,
    }


def build_arms_by_day(days: pd.DataFrame, table: dict) -> dict[str, list[str]]:
    by_ctx = table["by_context"]
    out = {}
    for _, row in days.iterrows():
        key = str(row["日期"])
        ctx = row["context"]
        if ctx not in by_ctx:
            ctx = f"{row['CalBlock']}_other"
        out[key] = list(by_ctx[ctx]["arms"])
    return out


def verify_sim_api_vs_daily_mu(
    daily_mu: dict,
    sample_day: str,
    rx: pd.DataFrame,
    items: pd.DataFrame,
    bundle,
    queue_model: dict,
    arms: list[str],
) -> dict:
    """Numeric regression: simulate_day_under_arm should align with precomputed daily_mu."""
    d = pd.to_datetime(sample_day).date()
    rx_d = rx[pd.to_datetime(rx["日期"]).dt.date == d]
    it_d = items[items["处方编号"].isin(rx_d["处方编号"])]
    cache: dict = {}
    diffs = {}
    for a in arms:
        if a not in daily_mu[sample_day]:
            continue
        out = simulate_day_under_arm(
            bundle, rx_d, it_d, a, items, queue_model, scenario_cache=cache
        )
        ref = daily_mu[sample_day][a]
        diffs[a] = {
            "sim": round(out["mu_min"], 6),
            "cached": round(ref, 6),
            "abs_diff": round(abs(out["mu_min"] - ref), 8),
        }
    max_diff = max((v["abs_diff"] for v in diffs.values()), default=0.0)
    return {"sample_day": sample_day, "max_abs_diff": max_diff, "per_arm": diffs, "ok": max_diff < 1e-5}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="ridge", choices=["ridge", "hgb"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-days", type=int, default=0)
    parser.add_argument(
        "--reuse-daily-mu",
        action="store_true",
        help="Reuse continuous_reward/daily_mu_minutes.json",
    )
    parser.add_argument(
        "--algo",
        default="both",
        choices=["TS", "UCB", "both"],
        help="Primary learner; both=produce TS+UCB",
    )
    parser.add_argument(
        "--table",
        default=str(TABLE_PATH),
        help="POMIS arm table JSON (sensitivity: pomis_plus_table_strict.json)",
    )
    parser.add_argument(
        "--out-subdir",
        default="e2e_closed_loop",
        help="Output subdirectory under ns_bandit_results",
    )
    args = parser.parse_args()

    out_dir = PHASE2_DIR / "ns_bandit_results" / args.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    table_path = Path(args.table)

    print(f"[0] Diagnosing arm table ← {table_path}...")
    with table_path.open(encoding="utf-8") as f:
        table = json.load(f)
    assert table["meta"].get("forbid_online_pomis_search") is True
    diag = diagnose_arm_sets(table)
    diag["table_path"] = str(table_path)
    diag["strict_gc_pomis"] = bool(table["meta"].get("strict_gc_pomis"))
    with (out_dir / "arm_set_diagnosis.json").open("w", encoding="utf-8") as f:
        json.dump(diag, f, ensure_ascii=False, indent=2)
    print(
        f"  contexts={diag['n_contexts']} unique_arm_sets={diag['unique_arm_sets']} "
        f"identical={diag['arms_identical_across_contexts']} "
        f"strict={diag['strict_gc_pomis']}"
    )
    print(f"  → {diag['implication']}")

    print("[1] Loading regressor / data...")
    bundle = load_bundle(args.model)
    queue_model = load_queue_model()
    mae = bundle.metrics["mae"]

    rx = pd.read_parquet(DATA_DIR / "rx_level.parquet")
    disc = pd.read_parquet(PHASE2_DIR / "rx_discretized.parquet")
    rx = rx.merge(disc[["处方编号", "Peak0", "Load0"]], on="处方编号", how="left")
    staff = load_daily_staff()
    rx = build_feature_frame(rx, staff)
    rx = annotate_regimes(rx, rx["日期"])
    days = build_day_index(rx)
    if args.max_days > 0:
        days = days.head(args.max_days)

    fixed_arms = [a for a in arm_sets()["POMIS+"] if not a.startswith("load_do")]
    universe = sorted(
        set(fixed_arms) | set().union(*[set(v["arms"]) for v in table["by_context"].values()])
    )

    if args.reuse_daily_mu and CONT_MU_PATH.exists():
        print(f"[2] Reusing daily μ ← {CONT_MU_PATH}")
        with CONT_MU_PATH.open(encoding="utf-8") as f:
            daily_mu = json.load(f)
        missing = [a for a in universe if not any(a in v for v in daily_mu.values())]
        if missing:
            print(f"  Warning: daily_mu missing arms {missing}; will recompute full table")
            args.reuse_daily_mu = False

    if not (args.reuse_daily_mu and CONT_MU_PATH.exists()):
        print("[2] Precomputing daily μ (simulate_day_under_arm / day_mean_wait)...")
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
        daily_mu = precompute_daily_mu(
            days, rx, items, universe, bundle, items, queue_model=queue_model
        )
        with (out_dir / "daily_mu_minutes.json").open("w", encoding="utf-8") as f:
            json.dump(daily_mu, f)
    else:
        with (out_dir / "daily_mu_minutes.json").open("w", encoding="utf-8") as f:
            json.dump(daily_mu, f)

    sample_day = next(iter(daily_mu))
    items_chk = pd.read_parquet(
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
    api_check = verify_sim_api_vs_daily_mu(
        daily_mu, sample_day, rx, items_chk, bundle, queue_model, fixed_arms[:4]
    )
    print(f"[2b] simulate_day_under_arm alignment check: ok={api_check['ok']} max_diff={api_check['max_abs_diff']}")
    with (out_dir / "sim_api_check.json").open("w", encoding="utf-8") as f:
        json.dump(api_check, f, ensure_ascii=False, indent=2)

    pooled = pooled_mu(daily_mu, universe)
    gate = check_arm_signal_gate_ttest(daily_mu, holdout_mae=mae)
    print(
        f"[3] t-gate: pass={gate['gate_pass']} p={gate.get('p_one_sided')} "
        f"δ̄={gate.get('mean_delta_null_minus_swap40')}"
    )

    arms_by_day = build_arms_by_day(days, table)
    algos = ["TS", "UCB"] if args.algo == "both" else [args.algo]

    all_curves: dict[str, dict] = {}
    by_algo: dict[str, dict] = {}

    for algo in algos:
        print(f"[4] Three-curve bandit (algo={algo}, no bonus)...")
        curves = {
            "Fixed_POMIS+": run_bandit_with_arm_policy(
                days,
                daily_mu,
                universe,
                algo,
                args.seed,
                fixed_arms=fixed_arms,
                oracle_arms=fixed_arms,
            ),
            "Dynamic_Gc": run_bandit_with_arm_policy(
                days,
                daily_mu,
                universe,
                algo,
                args.seed,
                arms_by_day=arms_by_day,
                oracle_arms=fixed_arms,  # same oracle as Fixed for fair exploration comparison
            ),
            "always_swap40": run_bandit_with_arm_policy(
                days,
                daily_mu,
                universe,
                algo,
                args.seed,
                force_arm="swap40",
                oracle_arms=fixed_arms,
            ),
            "oracle_Fixed": run_bandit_with_arm_policy(
                days, daily_mu, universe, "oracle", args.seed, fixed_arms=fixed_arms
            ),
            "always_null": run_bandit_with_arm_policy(
                days,
                daily_mu,
                universe,
                algo,
                args.seed,
                force_arm="null",
                oracle_arms=fixed_arms,
            ),
        }
        for name, res in curves.items():
            print(f"  {name}: regret={res['final_regret_min']} min  steps={res['n_steps']}")
            all_curves[f"{algo}__{name}"] = res

        fr = curves["Fixed_POMIS+"]["final_regret_min"]
        dr = curves["Dynamic_Gc"]["final_regret_min"]
        by_algo[algo] = {
            "fixed_minus_dynamic_regret_min": round(fr - dr, 4),
            "curves": {
                k: {kk: vv for kk, vv in v.items() if kk != "cumulative_regret"}
                for k, v in curves.items()
            },
        }

        plt.figure(figsize=(9, 5))
        for name in ("Fixed_POMIS+", "Dynamic_Gc", "always_swap40", "always_null", "oracle_Fixed"):
            res = curves[name]
            plt.plot(
                res["cumulative_regret"],
                label=f"{name} ({res['final_regret_min']:.1f})",
                linewidth=2 if name in ("Fixed_POMIS+", "Dynamic_Gc") else 1.2,
            )
        plt.xlabel("Day")
        plt.ylabel("Cumulative regret (minutes)")
        title = f"E2E closed-loop ({algo}, R=-Ŷ, no bonus)"
        if diag["arms_identical_across_contexts"]:
            title += "\n[arm sets identical across 8 contexts → Fixed≈Dynamic expected]"
        plt.title(title)
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(out_dir / f"e2e_three_curves_{algo}.png", dpi=140)
        plt.close()

    summary = {
        "status": "ok",
        "s5_stage": "e2e_closed_loop",
        "reward": "R = -predicted_wait_minutes",
        "no_bonus": True,
        "model": args.model,
        "table_path": str(table_path),
        "strict_gc_pomis": bool(table["meta"].get("strict_gc_pomis")),
        "arm_set_diagnosis": {
            "arms_identical_across_contexts": diag["arms_identical_across_contexts"],
            "unique_arm_sets": diag["unique_arm_sets"],
            "shared_arms": diag.get("shared_arms"),
            "implication": diag["implication"],
        },
        "gate": gate,
        "pooled_mu_minutes": pooled,
        "sim_api_check": {"ok": api_check["ok"], "max_abs_diff": api_check["max_abs_diff"]},
        "by_algo": by_algo,
        "note": (
            "Dynamic graph does not change available arms → regret gap = 0 (expected)."
            if diag["arms_identical_across_contexts"]
            and not table["meta"].get("strict_gc_pomis")
            else (
                "STRICT: Dynamic arms = G_c-derived only (no forced POMIS+/@1 union). "
                "Interpret Fixed−Dynamic gap as arm-filter effect."
                if table["meta"].get("strict_gc_pomis")
                else "Dynamic arm filter active; interpret Fixed−Dynamic gap."
            )
        ),
    }
    with (out_dir / "e2e_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with (out_dir / "cumulative_regret.json").open("w", encoding="utf-8") as f:
        json.dump({k: v["cumulative_regret"] for k, v in all_curves.items()}, f)
    with (out_dir / "signal_gate.json").open("w", encoding="utf-8") as f:
        json.dump({"gate": gate, "pooled_mu": pooled}, f, ensure_ascii=False, indent=2)

    print("[5] Fixed−Dynamic by algo:", {a: by_algo[a]["fixed_minus_dynamic_regret_min"] for a in by_algo})
    print("done →", out_dir)


if __name__ == "__main__":
    main()
