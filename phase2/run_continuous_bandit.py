"""S4.2–S4.3: Daily full continuous reward μ; run bandit only after gate check."""

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
    OUT_DIR,
    arm_to_scenario,
    build_feature_frame,
    check_arm_signal_gate_ttest,
    day_mean_wait,
    load_bundle,
    reward_from_yhat,
)
from phase2.ns_arms import arm_sets
from phase2.ns_mechanisms import annotate_regimes, cal_block
from phase2.run_contextual_bandit import TAU_RUSH_DAY, build_day_index, day_context_label
from phase2.staff_schedule import load_daily_staff

PHASE2_DIR = DATA_DIR / "phase2"
BANDIT_OUT = PHASE2_DIR / "ns_bandit_results" / "continuous_reward"


def precompute_daily_mu(
    days: pd.DataFrame,
    rx: pd.DataFrame,
    items: pd.DataFrame,
    arms: list[str],
    bundle,
    items_all: pd.DataFrame,
    queue_model: dict | None = None,
    affected_only: bool = False,
    affected_ref_arm: str = "swap40",
) -> dict:
    """Per day: μ[date][arm] = mean predicted wait (minutes).

    affected_only: compute μ on same prescriptions touching ref-arm swapped drugs (fair comparison).
    @1 arms via parse_arm_start_t → apply CF from rush onward only.
    """
    from phase2.continuous_reward import (
        load_queue_model,
        parse_arm_start_t,
        rx_ids_touching_drugs,
        _scenario_drug_ids,
    )

    if queue_model is None:
        queue_model = load_queue_model()
    scenarios = {a: arm_to_scenario(a, items_all) for a in arms}
    ref_sc = scenarios.get(affected_ref_arm) or scenarios.get(
        next((a for a in arms if a.split("@")[0] == affected_ref_arm), ""), None
    )
    ref_drugs = _scenario_drug_ids(ref_sc) if affected_only else set()

    rx = rx.copy()
    rx["日期_d"] = pd.to_datetime(rx["日期"]).dt.date
    items = items.copy()
    rx_day_map = rx.set_index("处方编号")["日期_d"]
    items["日期_d"] = items["处方编号"].map(rx_day_map)

    mu: dict[str, dict[str, float]] = {}
    for i, row in days.iterrows():
        d = row["日期"]
        key = str(d)
        rx_d = rx[rx["日期_d"] == d]
        it_d = items[items["日期_d"] == d]
        if len(rx_d) < 5 or len(it_d) == 0:
            continue
        restrict = rx_ids_touching_drugs(it_d, ref_drugs) if affected_only else None
        if affected_only and (not restrict or len(restrict) < 3):
            continue
        mu[key] = {}
        for a in arms:
            yhat = day_mean_wait(
                bundle,
                rx_d,
                it_d,
                scenarios[a],
                queue_model=queue_model,
                restrict_rx_ids=restrict,
                start_t=parse_arm_start_t(a),
            )
            mu[key][a] = round(yhat, 6)
        if (i + 1) % 50 == 0:
            print(f"  days done {i+1}/{len(days)}")
    return mu


def pooled_mu(daily_mu: dict, arms: list[str]) -> dict[str, float]:
    out = {}
    for a in arms:
        vals = [v[a] for v in daily_mu.values() if a in v and np.isfinite(v[a])]
        out[a] = float(np.mean(vals)) if vals else float("nan")
    return out


def run_bandit_minutes(
    days: pd.DataFrame,
    daily_mu: dict,
    arms: list[str],
    algo: str,
    seed: int = 42,
) -> dict:
    """R = -Ŷ; oracle picks smallest Ŷ that day."""
    rng = np.random.default_rng(seed)
    K = len(arms)
    idx = {a: i for i, a in enumerate(arms)}
    n_pulls = np.zeros(K)
    sum_r = np.zeros(K)
    cum, total = [], 0.0
    chosen = []

    for t, row in days.iterrows():
        key = str(row["日期"])
        if key not in daily_mu:
            continue
        yhats = daily_mu[key]
        # Arms available today
        arms_today = [a for a in arms if a in yhats]
        opt = min(arms_today, key=lambda a: yhats[a])
        opt_r = -yhats[opt]

        local = [idx[a] for a in arms_today]
        mu_hat = np.full(K, -5.0)  # approx -mean wait
        m = n_pulls > 0
        mu_hat[m] = sum_r[m] / n_pulls[m]

        if algo == "TS":
            best_j, best_s = local[0], -1e9
            for j in local:
                s = rng.normal(-5, 1) if n_pulls[j] <= 0 else rng.normal(mu_hat[j], 1 / np.sqrt(n_pulls[j]))
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
            a_idx = idx[opt]
        elif algo == "always_null":
            a_idx = idx["null"]
        elif algo == "always_swap40":
            a_idx = idx.get("swap40", idx[arms_today[0]])
        else:
            raise ValueError(algo)

        arm = arms[a_idx]
        r = -yhats[arm]
        r_obs = float(rng.normal(r, 0.05))
        n_pulls[a_idx] += 1
        sum_r[a_idx] += r_obs
        total += max(0.0, opt_r - r)
        cum.append(total)
        chosen.append(arm)

    return {
        "algo": algo,
        "final_regret_min": round(total, 4),
        "n_steps": len(chosen),
        "arm_counts": {a: int((np.array(chosen) == a).sum()) for a in arms},
        "cumulative_regret": cum,
        "unit": "minutes",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="ridge", choices=["ridge", "hgb", "mlp"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-days", type=int, default=0)
    parser.add_argument("--force-bandit", action="store_true", help="Run even if gate fails (not recommended)")
    parser.add_argument(
        "--reuse-daily-mu",
        action="store_true",
        help="Reuse existing daily_mu_minutes.json; recompute gate/bandit only",
    )
    parser.add_argument(
        "--affected-only",
        action="store_true",
        help="Compute μ on subset of prescriptions with swapped drugs",
    )
    parser.add_argument(
        "--out-subdir",
        default="",
        help="Output subdirectory; default continuous_reward or continuous_reward_affected",
    )
    args = parser.parse_args()

    from phase2.continuous_reward import load_queue_model

    out_name = args.out_subdir or (
        "continuous_reward_affected" if args.affected_only else "continuous_reward"
    )
    out_dir = PHASE2_DIR / "ns_bandit_results" / out_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[1] Loading regressor and data (S4.4 + W̃, S4.5 t-gate)...")
    bundle = load_bundle(args.model)
    mae = bundle.metrics["mae"]
    queue_model = load_queue_model()
    print(f"  model={args.model} hold-out MAE={mae} (info only) features={bundle.feature_cols}")
    print(f"  affected_only={args.affected_only}")

    arms = [a for a in arm_sets()["POMIS+"] if not a.startswith("load_do")]
    scope = "swapped-drug subset" if args.affected_only else "full daily"
    daily_mu_path = out_dir / "daily_mu_minutes.json"

    if args.reuse_daily_mu and daily_mu_path.exists():
        print(f"[2] Reusing daily μ ← {daily_mu_path}")
        with daily_mu_path.open(encoding="utf-8") as f:
            daily_mu = json.load(f)
        # days still needed for calendar-order bandit
        rx = pd.read_parquet(DATA_DIR / "rx_level.parquet")
        disc = pd.read_parquet(PHASE2_DIR / "rx_discretized.parquet")
        rx = rx.merge(disc[["处方编号", "Peak0", "Load0"]], on="处方编号", how="left")
        staff = load_daily_staff()
        rx = build_feature_frame(rx, staff)
        rx = annotate_regimes(rx, rx["日期"])
        days = build_day_index(rx)
        if args.max_days > 0:
            days = days.head(args.max_days)
    else:
        rx = pd.read_parquet(DATA_DIR / "rx_level.parquet")
        disc = pd.read_parquet(PHASE2_DIR / "rx_discretized.parquet")
        rx = rx.merge(disc[["处方编号", "Peak0", "Load0"]], on="处方编号", how="left")
        staff = load_daily_staff()
        rx = build_feature_frame(rx, staff)
        rx = annotate_regimes(rx, rx["日期"])

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
        days = build_day_index(rx)
        if args.max_days > 0:
            days = days.head(args.max_days)

        print(f"[2] Precomputing μ ({scope}, minutes)...")
        print(f"  days={len(days)} arms={arms}")
        daily_mu = precompute_daily_mu(
            days,
            rx,
            items,
            arms,
            bundle,
            items,
            queue_model=queue_model,
            affected_only=args.affected_only,
        )
        with daily_mu_path.open("w", encoding="utf-8") as f:
            json.dump(daily_mu, f)

    pooled = pooled_mu(daily_mu, arms)
    print(f"[3] Arm μ ({scope} daily mean predicted minutes):", {k: round(v, 4) for k, v in pooled.items()})

    gate = check_arm_signal_gate_ttest(daily_mu, holdout_mae=mae)
    with (out_dir / "signal_gate.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "pooled_mu": pooled,
                "gate": gate,
                "model": args.model,
                "use_counterfactual_W": True,
                "affected_only": args.affected_only,
                "s4_stage": "S4.5",
                "gate_logic": "one-sided paired t-test on daily μ diffs; MAE info-only",
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print("[4] Gate check (S4.5 t-test):", {k: gate[k] for k in (
        "mean_delta_null_minus_swap40", "se_delta", "t_stat", "p_one_sided",
        "n_days", "gate_pass", "holdout_mae_info_only", "mae_vs_delta_ratio_info",
    ) if k in gate})

    if not gate["gate_pass"] and not args.force_bandit:
        print(
            "\n※ GATE FAIL: daily μ difference one-sided t-test did not pass. "
            "Per convention, do not report bandit regret curves."
        )
        with (out_dir / "bandit_summary.json").open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "status": "skipped_gate_fail",
                    "gate": gate,
                    "pooled_mu_minutes": pooled,
                    "use_counterfactual_W": True,
                    "affected_only": args.affected_only,
                    "s4_stage": "S4.5",
                    "note": "Day-level arm ranking not statistically identifiable; do not interpret regret.",
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        return

    print("[5] Running bandit (R=-Ŷ_min)...")
    experiments = {}
    for algo in ("TS", "UCB", "oracle", "always_null", "always_swap40"):
        experiments[algo] = run_bandit_minutes(days, daily_mu, arms, algo, seed=args.seed)
        print(f"  {algo}: regret={experiments[algo]['final_regret_min']} min")

    plt.figure(figsize=(9, 5))
    for name, res in experiments.items():
        plt.plot(res["cumulative_regret"], label=f"{name} ({res['final_regret_min']:.1f})")
    plt.xlabel("Day")
    plt.ylabel("Cumulative regret (minutes)")
    plt.title("Continuous-reward bandit (R=-Ŷ_min, S4.5 t-gate, no bonus)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "cumulative_regret_minutes.png", dpi=140)
    plt.close()

    summary = {
        "status": "ok",
        "gate": gate,
        "pooled_mu_minutes": pooled,
        "holdout_mae_info_only": mae,
        "use_counterfactual_W": True,
        "affected_only": args.affected_only,
        "s4_stage": "S4.5",
        "experiments": {
            k: {kk: vv for kk, vv in v.items() if kk != "cumulative_regret"}
            for k, v in experiments.items()
        },
        "no_bonus": True,
        "reward": "R = -predicted_wait_minutes",
    }
    with (out_dir / "bandit_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with (out_dir / "cumulative_regret.json").open("w", encoding="utf-8") as f:
        json.dump({k: v["cumulative_regret"] for k, v in experiments.items()}, f)
    print("done →", out_dir)


if __name__ == "__main__":
    main()
