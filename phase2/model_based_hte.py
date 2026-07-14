"""Model-based heterogeneous effects (prescription-level τ) and high-benefit profiles.

τ_i = Ŷ_i(null) − Ŷ_i(swap40)   (full-day A2: includes queue propagation)
Also decomposable as:
  τ_disp,i  = Ŷ(null) − Ŷ(A1)
  τ_queue,i = Ŷ(A1) − Ŷ(A2)

Honest disclaimer: these are model-based heterogeneous effects under structural causal model + Ridge,
not GRF/causal-forest estimates with randomization support; must be labeled as such in the paper.

Outputs: prescription-level τ table, stratum means, shallow decision-tree rules (interpretable profiles).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor, export_text

from phase2.config import DATA_DIR
from phase2.continuous_reward import (
    _scenario_drug_ids,
    arm_to_scenario,
    build_feature_frame,
    day_rx_predictions,
    ensure_morning_rush,
    load_bundle,
    load_queue_model,
)
from phase2.effect_decomposition import _load_frames

PHASE2_DIR = DATA_DIR / "phase2"
OUT_DIR = PHASE2_DIR / "model_based_hte"

PROFILE_FEATURES = [
    "MorningRush",
    "Peak0",
    "Load0",
    "品项数",
    "机器品项占比",
    "预估配药_分钟",
    "调配人数",
    "touched_swap",
]


def collect_rx_tau(
    arm: str = "swap40",
    sample_days: int = 60,
    max_rx_per_day: int | None = None,
    seed: int = 42,
    model_name: str = "ridge",
) -> pd.DataFrame:
    print("[1] Loading...")
    rx, items, _ = _load_frames()
    bundle = load_bundle(model_name)
    qm = load_queue_model()
    scenario = arm_to_scenario(arm, items)
    swap_drugs = _scenario_drug_ids(scenario)

    rx = rx.copy()
    rx["日期_d"] = pd.to_datetime(rx["日期"]).dt.date
    items = items.copy()
    items["日期_d"] = items["处方编号"].map(rx.set_index("处方编号")["日期_d"])

    days = sorted(rx["日期_d"].dropna().unique())
    rng = np.random.default_rng(seed)
    if len(days) > sample_days:
        days = sorted(rng.choice(days, size=sample_days, replace=False))

    rows = []
    for i, d in enumerate(days):
        rx_d = rx[rx["日期_d"] == d]
        it_d = items[items["日期_d"] == d]
        if len(rx_d) < 30 or len(it_d) == 0:
            continue
        if max_rx_per_day is not None and len(rx_d) > max_rx_per_day:
            keep = set(
                rng.choice(rx_d["处方编号"].unique(), size=max_rx_per_day, replace=False)
            )
            rx_d = rx_d[rx_d["处方编号"].isin(keep)]
            it_d = it_d[it_d["处方编号"].isin(keep)]

        a0 = day_rx_predictions(bundle, rx_d, it_d, None, queue_model=qm)
        a1 = day_rx_predictions(
            bundle, rx_d, it_d, scenario, queue_model=qm, use_queue_propagation=False
        )
        a2 = day_rx_predictions(
            bundle, rx_d, it_d, scenario, queue_model=qm, use_queue_propagation=True
        )
        if a0.empty or a1.empty or a2.empty:
            continue

        touched = set(
            it_d.loc[it_d["drugid"].astype(str).isin(swap_drugs), "处方编号"].unique()
        )
        m = (
            a0[["处方编号", "yhat", "MorningRush", "Peak0", "Load0", "调配人数"]]
            .rename(columns={"yhat": "yhat_null"})
            .merge(
                a1[["处方编号", "yhat", "预估配药_分钟", "机器品项占比", "品项数"]].rename(
                    columns={"yhat": "yhat_A1"}
                ),
                on="处方编号",
            )
            .merge(
                a2[["处方编号", "yhat", "timing_active"]].rename(columns={"yhat": "yhat_A2"}),
                on="处方编号",
            )
        )
        m["date"] = str(d)
        m["arm"] = arm
        m["touched_swap"] = m["处方编号"].isin(touched).astype(int)
        m["tau_total"] = m["yhat_null"] - m["yhat_A2"]
        m["tau_disp"] = m["yhat_null"] - m["yhat_A1"]
        m["tau_queue"] = m["yhat_A1"] - m["yhat_A2"]
        rows.append(m)
        if (i + 1) % 15 == 0:
            print(f"  days {i+1}/{len(days)}")

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def stratum_summary(tau_df: pd.DataFrame) -> dict:
    def block(sub: pd.DataFrame) -> dict:
        if sub.empty:
            return {"n": 0}
        return {
            "n": int(len(sub)),
            "mean_tau_total": round(float(sub["tau_total"].mean()), 6),
            "mean_tau_disp": round(float(sub["tau_disp"].mean()), 6),
            "mean_tau_queue": round(float(sub["tau_queue"].mean()), 6),
            "p90_tau_total": round(float(sub["tau_total"].quantile(0.9)), 6),
            "frac_positive_tau": round(float((sub["tau_total"] > 1e-6).mean()), 4),
        }

    out = {"overall": block(tau_df)}
    if "MorningRush" in tau_df.columns:
        out["rush"] = block(tau_df[tau_df["MorningRush"].astype(int) == 1])
        out["other"] = block(tau_df[tau_df["MorningRush"].astype(int) == 0])
    if "touched_swap" in tau_df.columns:
        out["touched"] = block(tau_df[tau_df["touched_swap"] == 1])
        out["untouched"] = block(tau_df[tau_df["touched_swap"] == 0])
        out["touched_and_rush"] = block(
            tau_df[(tau_df["touched_swap"] == 1) & (tau_df["MorningRush"].astype(int) == 1)]
        )
    # Coarse bins by item count
    if "品项数" in tau_df.columns:
        n = pd.to_numeric(tau_df["品项数"], errors="coerce")
        out["n_items_1"] = block(tau_df[n == 1])
        out["n_items_2plus"] = block(tau_df[n >= 2])
    if "机器品项占比" in tau_df.columns:
        r = pd.to_numeric(tau_df["机器品项占比"], errors="coerce")
        out["machine_ratio_low"] = block(tau_df[r <= 0.5])
        out["machine_ratio_high"] = block(tau_df[r > 0.5])
    return out


def fit_profile_tree(tau_df: pd.DataFrame, max_depth: int = 3) -> dict:
    feats = [c for c in PROFILE_FEATURES if c in tau_df.columns]
    X = tau_df[feats].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    y = tau_df["tau_total"].astype(float)
    tree = DecisionTreeRegressor(
        max_depth=max_depth, min_samples_leaf=max(200, len(tau_df) // 50), random_state=0
    )
    tree.fit(X, y)
    # Leaf profiles: sort by predicted τ
    leaf = tree.apply(X)
    leaf_stats = []
    for lid in sorted(set(leaf)):
        m = leaf == lid
        leaf_stats.append(
            {
                "leaf_id": int(lid),
                "n": int(m.sum()),
                "mean_tau": round(float(y[m].mean()), 6),
                "frac_touched": round(float(tau_df.loc[m, "touched_swap"].mean()), 4)
                if "touched_swap" in tau_df.columns
                else None,
                "frac_rush": round(float(tau_df.loc[m, "MorningRush"].astype(int).mean()), 4)
                if "MorningRush" in tau_df.columns
                else None,
            }
        )
    leaf_stats = sorted(leaf_stats, key=lambda z: z["mean_tau"], reverse=True)
    rules = export_text(tree, feature_names=feats, decimals=3)
    return {
        "features": feats,
        "max_depth": max_depth,
        "train_r2": round(float(tree.score(X, y)), 4),
        "top_leaves": leaf_stats[:8],
        "tree_rules_text": rules,
        "note": (
            "Decision tree on model-based τ; descriptive profile only, "
            "not a causal discovery claim"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Model-based HTE / τ profiles")
    parser.add_argument("--arm", default="swap40")
    parser.add_argument("--sample-days", type=int, default=60)
    parser.add_argument("--max-rx-per-day", type=int, default=0, help="0=all prescriptions for the day")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default="ridge")
    parser.add_argument("--tree-depth", type=int, default=3)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    max_rx = args.max_rx_per_day if args.max_rx_per_day > 0 else None
    tau_df = collect_rx_tau(
        arm=args.arm,
        sample_days=args.sample_days,
        max_rx_per_day=max_rx,
        seed=args.seed,
        model_name=args.model,
    )
    if tau_df.empty:
        raise SystemExit("no tau rows")

    strata = stratum_summary(tau_df)
    profile = fit_profile_tree(tau_df, max_depth=args.tree_depth)

    # High-benefit subset: top decile of τ
    thr = float(tau_df["tau_total"].quantile(0.9))
    high = tau_df[tau_df["tau_total"] >= thr]
    high_profile = {
        "tau_p90_threshold": round(thr, 6),
        "n_high": int(len(high)),
        "frac_of_sample": round(float(len(high) / len(tau_df)), 4),
        "mean_features": {
            c: round(float(pd.to_numeric(high[c], errors="coerce").mean()), 4)
            for c in PROFILE_FEATURES
            if c in high.columns
        },
        "vs_overall_mean_features": {
            c: round(
                float(pd.to_numeric(high[c], errors="coerce").mean())
                - float(pd.to_numeric(tau_df[c], errors="coerce").mean()),
                4,
            )
            for c in PROFILE_FEATURES
            if c in tau_df.columns
        },
    }

    summary = {
        "identification": (
            "MODEL-BASED heterogeneous effects under Ridge + heuristic queue "
            "propagation; NOT randomized / NOT GRF causal estimates"
        ),
        "arm": args.arm,
        "n_rx": int(len(tau_df)),
        "n_days": int(tau_df["date"].nunique()),
        "strata": strata,
        "high_tau_decile_profile": high_profile,
        "tree_profile": {
            k: v for k, v in profile.items() if k != "tree_rules_text"
        },
    }

    tau_path = OUT_DIR / f"rx_tau_{args.arm}.parquet"
    tau_df.to_parquet(tau_path, index=False)
    with (OUT_DIR / f"hte_summary_{args.arm}.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with (OUT_DIR / f"hte_tree_rules_{args.arm}.txt").open("w", encoding="utf-8") as f:
        f.write("# MODEL-BASED τ profile tree (descriptive)\n")
        f.write(profile["tree_rules_text"])

    print("\n=== Model-based HTE ===")
    print("identification:", summary["identification"])
    print("n_rx:", summary["n_rx"], "days:", summary["n_days"])
    for k, v in strata.items():
        if v.get("n", 0) == 0:
            continue
        print(
            f"  {k}: n={v['n']}  mean_τ={v['mean_tau_total']:.4f} "
            f"(disp={v['mean_tau_disp']:.4f}, queue={v['mean_tau_queue']:.4f})"
        )
    print("top leaves:", profile["top_leaves"][:3])
    print("done →", OUT_DIR)


if __name__ == "__main__":
    main()
