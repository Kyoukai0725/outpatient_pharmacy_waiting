"""Shapley path decomposition (contrast with nested; enters main-text mechanism section when queue dominates).

Average over two orderings of the two paths {disp/direct, queue}:
  Order1: A0 → A1(direct) → A2(both)     (nested)
  Order2: A0 → A_Q(queue-only) → A2

φ_direct = 0.5[(μ0−μ1) + (μ_Q−μ2)]
φ_queue  = 0.5[(μ0−μ_Q) + (μ1−μ2)]

QM-Policy main claim: when the queue channel dominates, Shapley and nested should agree; serves as mechanistic robustness check.
Honest disclaimer: model-based.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from phase2.config import DATA_DIR
from phase2.continuous_reward import (
    arm_to_scenario,
    day_rx_predictions,
    load_bundle,
    load_queue_model,
)
from phase2.effect_decomposition import _load_frames, _mean_yhat

PHASE2_DIR = DATA_DIR / "phase2"
OUT_DIR = PHASE2_DIR / "effect_decomposition"


def shapley_day(bundle, rx_day, items_day, scenario, queue_model, restrict=None) -> dict:
    a0 = day_rx_predictions(
        bundle, rx_day, items_day, None, queue_model=queue_model, restrict_rx_ids=restrict
    )
    # A1: disp only
    a1 = day_rx_predictions(
        bundle,
        rx_day,
        items_day,
        scenario,
        queue_model=queue_model,
        restrict_rx_ids=restrict,
        use_queue_propagation=False,
        apply_layout_features=True,
    )
    # A_Q: queue only (ΔD_sys from scenario, features remain baseline)
    aq = day_rx_predictions(
        bundle,
        rx_day,
        items_day,
        scenario,
        queue_model=queue_model,
        restrict_rx_ids=restrict,
        use_queue_propagation=True,
        apply_layout_features=False,
    )
    # A2: both
    a2 = day_rx_predictions(
        bundle,
        rx_day,
        items_day,
        scenario,
        queue_model=queue_model,
        restrict_rx_ids=restrict,
        use_queue_propagation=True,
        apply_layout_features=True,
    )
    if a0.empty or a1.empty or aq.empty or a2.empty:
        return {}

    ids = set(a0["处方编号"]) & set(a1["处方编号"]) & set(aq["处方编号"]) & set(a2["处方编号"])
    a0 = a0[a0["处方编号"].isin(ids)]
    a1 = a1[a1["处方编号"].isin(ids)]
    aq = aq[aq["处方编号"].isin(ids)]
    a2 = a2[a2["处方编号"].isin(ids)]

    m0, m1, mq, m2 = _mean_yhat(a0), _mean_yhat(a1), _mean_yhat(aq), _mean_yhat(a2)
    # Order1 nested
    o1_disp, o1_queue = m0 - m1, m1 - m2
    # Order2 queue first
    o2_queue, o2_disp = m0 - mq, mq - m2
    phi_d = 0.5 * (o1_disp + o2_disp)
    phi_q = 0.5 * (o1_queue + o2_queue)
    tot = m0 - m2
    return {
        "mu_A0": round(m0, 6),
        "mu_A1": round(m1, 6),
        "mu_AQ": round(mq, 6),
        "mu_A2": round(m2, 6),
        "nested_disp": round(o1_disp, 6),
        "nested_queue": round(o1_queue, 6),
        "order2_disp": round(o2_disp, 6),
        "order2_queue": round(o2_queue, 6),
        "shapley_disp": round(phi_d, 6),
        "shapley_queue": round(phi_q, 6),
        "delta_total": round(tot, 6),
        "shapley_sum": round(phi_d + phi_q, 6),
        "n_rx": int(len(ids)),
    }


def run_shapley(
    arm: str = "swap40",
    sample_days: int = 60,
    seed: int = 42,
    model_name: str = "ridge",
) -> tuple[dict, pd.DataFrame]:
    print("[1] Loading...")
    rx, items, _ = _load_frames()
    bundle = load_bundle(model_name)
    qm = load_queue_model()
    scenario = arm_to_scenario(arm, items)

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
        out = shapley_day(bundle, rx_d, it_d, scenario, qm)
        if not out:
            continue
        out["date"] = str(d)
        out["arm"] = arm
        rows.append(out)
        if (i + 1) % 15 == 0:
            print(f"  days {i+1}/{len(days)}")

    df = pd.DataFrame(rows)

    def agg(c: str) -> dict:
        v = df[c].astype(float)
        return {
            "mean": round(float(v.mean()), 6),
            "se": round(float(v.std(ddof=1) / np.sqrt(len(v))), 6) if len(v) > 1 else 0.0,
        }

    summary = {
        "identification": (
            "Shapley average of two path orders under model-based CF; appendix robustness"
        ),
        "arm": arm,
        "n_days": int(len(df)),
        "nested": {
            "disp": agg("nested_disp"),
            "queue": agg("nested_queue"),
        },
        "order2_queue_first": {
            "disp": agg("order2_disp"),
            "queue": agg("order2_queue"),
        },
        "shapley": {
            "disp": agg("shapley_disp"),
            "queue": agg("shapley_queue"),
            "sum": agg("shapley_sum"),
            "delta_total": agg("delta_total"),
        },
        "share_shapley_queue": round(
            float(df["shapley_queue"].mean() / max(df["delta_total"].mean(), 1e-9)), 4
        ),
        "share_nested_queue": round(
            float(df["nested_queue"].mean() / max(df["delta_total"].mean(), 1e-9)), 4
        ),
    }
    return summary, df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", default="swap40")
    parser.add_argument("--sample-days", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default="ridge")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary, df = run_shapley(
        arm=args.arm,
        sample_days=args.sample_days,
        seed=args.seed,
        model_name=args.model,
    )
    with (OUT_DIR / f"shapley_{args.arm}.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    df.to_csv(OUT_DIR / f"shapley_daily_{args.arm}.csv", index=False)

    print("\n=== Shapley (appendix) ===")
    print(summary["identification"])
    s = summary["shapley"]
    print(
        f"  φ_disp={s['disp']['mean']:.4f}  φ_queue={s['queue']['mean']:.4f}  "
        f"sum={s['sum']['mean']:.4f}  total={s['delta_total']['mean']:.4f}  "
        f"queue_share={summary['share_shapley_queue']}"
    )
    print(
        f"  nested queue_share={summary['share_nested_queue']}  "
        f"(compare for robustness)"
    )
    print("done →", OUT_DIR)


if __name__ == "__main__":
    main()
