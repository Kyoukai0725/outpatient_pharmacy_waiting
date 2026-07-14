"""Quarterly/semi-annual block rolling swap: dynamic swap40 under nonstationarity.

Three decision windows (2024H1 → 2024H2 → 2025H1), each re-estimating 40 slots from in-window frequency,
evaluating μ on the next window, and comparing to "full-sample static swap40"; report slot Jaccard/turnover.

Narrative: static plans become suboptimal as item frequency drifts; aligns with NS bandit / CalBlock.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from phase2.config import DATA_DIR
from phase2.continuous_reward import (
    build_feature_frame,
    day_mean_wait,
    load_bundle,
    load_queue_model,
)
from phase2.intervention import LayoutScenario, build_machine_swap
from phase2.ns_mechanisms import annotate_regimes, cal_block
from phase2.staff_schedule import load_daily_staff

PHASE2_DIR = DATA_DIR / "phase2"
OUT_DIR = PHASE2_DIR / "rolling_swap"

# Decision window → evaluation window
ROLL_PAIRS = (
    ("2024H1", "2024H2"),
    ("2024H2", "2025H1"),
    ("2025H1", "2025H2"),
)


def _slot_sets(sc: LayoutScenario | None) -> tuple[set[str], set[str], set[str]]:
    if sc is None:
        return set(), set(), set()
    inn = set(map(str, sc.to_machine))
    out = set(map(str, sc.to_manual))
    return inn, out, inn | out


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    u = a | b
    return float(len(a & b) / len(u)) if u else 1.0


def turnover(a: set[str], b: set[str]) -> dict:
    return {
        "n_a": len(a),
        "n_b": len(b),
        "n_intersection": len(a & b),
        "n_only_a": len(a - b),
        "n_only_b": len(b - a),
        "jaccard": round(jaccard(a, b), 4),
        "change_rate": round(1.0 - jaccard(a, b), 4),
    }


def filter_items_by_block(items: pd.DataFrame, rx: pd.DataFrame, block: str) -> pd.DataFrame:
    rx_b = rx[rx["CalBlock"] == block]
    ids = set(rx_b["处方编号"])
    return items[items["处方编号"].isin(ids)].copy()


def eval_mu_on_days(
    bundle,
    rx: pd.DataFrame,
    items: pd.DataFrame,
    scenario: LayoutScenario | None,
    queue_model: dict,
    block: str,
    max_days: int = 40,
    seed: int = 42,
) -> dict:
    rx_b = rx[rx["CalBlock"] == block].copy()
    rx_b["日期_d"] = pd.to_datetime(rx_b["日期"]).dt.date
    days = sorted(rx_b["日期_d"].unique())
    rng = np.random.default_rng(seed)
    if len(days) > max_days:
        days = sorted(rng.choice(days, size=max_days, replace=False))

    items = items.copy()
    items["日期_d"] = items["处方编号"].map(rx_b.set_index("处方编号")["日期_d"])

    deltas = []
    for d in days:
        rx_d = rx_b[rx_b["日期_d"] == d]
        it_d = items[items["日期_d"] == d]
        if len(rx_d) < 20 or len(it_d) == 0:
            continue
        m0 = day_mean_wait(bundle, rx_d, it_d, None, queue_model=queue_model)
        m1 = day_mean_wait(bundle, rx_d, it_d, scenario, queue_model=queue_model)
        if np.isfinite(m0) and np.isfinite(m1):
            deltas.append(m0 - m1)
    if not deltas:
        return {"n_days": 0, "mean_delta": None}
    return {
        "n_days": int(len(deltas)),
        "mean_delta": round(float(np.mean(deltas)), 6),
        "se_delta": round(float(np.std(deltas, ddof=1) / np.sqrt(len(deltas))), 6)
        if len(deltas) > 1
        else 0.0,
    }


def run_rolling(
    n_swap: int = 40,
    max_eval_days: int = 40,
    seed: int = 42,
    model_name: str = "ridge",
) -> dict:
    print("[1] Loading...")
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
    bundle = load_bundle(model_name)
    qm = load_queue_model()

    # Static freeze: use first decision-window frequency only, no updates thereafter (fair baseline)
    first_decide = ROLL_PAIRS[0][0]
    items_first = filter_items_by_block(items, rx, first_decide)
    static = build_machine_swap(
        items_first,
        n_swap,
        enforce_machine_eligibility=True,
        prefer_remove_ineligible=False,
    )
    static_in, static_out, static_all = _slot_sets(static)

    decisions = []
    prev_all = None
    for decide_b, eval_b in ROLL_PAIRS:
        print(f"[2] decide={decide_b} → eval={eval_b}")
        items_dec = filter_items_by_block(items, rx, decide_b)
        sc = build_machine_swap(
            items_dec,
            n_swap,
            enforce_machine_eligibility=True,
            prefer_remove_ineligible=False,
        )
        inn, out, all_slots = _slot_sets(sc)
        vs_static = {
            "in": turnover(static_in, inn),
            "out": turnover(static_out, out),
            "all_slots": turnover(static_all, all_slots),
        }
        vs_prev = turnover(prev_all, all_slots) if prev_all is not None else None

        mu_roll = eval_mu_on_days(
            bundle, rx, items, sc, qm, eval_b, max_days=max_eval_days, seed=seed
        )
        mu_static = eval_mu_on_days(
            bundle, rx, items, static, qm, eval_b, max_days=max_eval_days, seed=seed
        )
        gap = None
        if mu_roll["mean_delta"] is not None and mu_static["mean_delta"] is not None:
            gap = round(mu_roll["mean_delta"] - mu_static["mean_delta"], 6)

        decisions.append(
            {
                "decide_block": decide_b,
                "eval_block": eval_b,
                "n_items_decide": int(len(items_dec)),
                "n_swap": int(sc.meta.get("n_swap", n_swap)),
                "expected_net_hours_in_window": sc.meta.get("expected_net_hours"),
                "slots_in": sorted(inn),
                "slots_out": sorted(out),
                "vs_static": vs_static,
                "vs_prev_decision": vs_prev,
                "mu_delta_rolling_on_eval": mu_roll,
                "mu_delta_static_on_eval": mu_static,
                "rolling_minus_static_delta": gap,
            }
        )
        prev_all = all_slots

    # Cross-decision slot drift summary
    drift = []
    for i in range(1, len(decisions)):
        a = set(decisions[i - 1]["slots_in"]) | set(decisions[i - 1]["slots_out"])
        b = set(decisions[i]["slots_in"]) | set(decisions[i]["slots_out"])
        drift.append(
            {
                "from": decisions[i - 1]["decide_block"],
                "to": decisions[i]["decide_block"],
                **turnover(a, b),
            }
        )

    return {
        "identification": (
            "Rolling frequency-based swap recomputed each CalBlock; "
            "μ from Ridge+propagate (model-based)"
        ),
        "n_swap": n_swap,
        "static_built_on": first_decide,
        "static_n_in": len(static_in),
        "static_n_out": len(static_out),
        "roll_pairs": [{"decide": a, "eval": b} for a, b in ROLL_PAIRS],
        "decisions": decisions,
        "decision_to_decision_drift": drift,
        "note": (
            "static = swap frozen at first decide block (no peek at future freqs); "
            "change_rate = 1 - Jaccard; "
            "rolling_minus_static_delta > 0 means rolling beats frozen static on eval"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-swap", type=int, default=40)
    parser.add_argument("--max-eval-days", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default="ridge")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = run_rolling(
        n_swap=args.n_swap,
        max_eval_days=args.max_eval_days,
        seed=args.seed,
        model_name=args.model,
    )
    # Write summary without long slot lists; slot tables go to sidecar file
    slim = dict(summary)
    slot_tables = []
    for d in slim["decisions"]:
        slot_tables.append(
            {
                "decide_block": d["decide_block"],
                "slots_in": d.pop("slots_in"),
                "slots_out": d.pop("slots_out"),
            }
        )
    with (OUT_DIR / "rolling_swap_summary.json").open("w", encoding="utf-8") as f:
        json.dump(slim, f, ensure_ascii=False, indent=2)
    with (OUT_DIR / "rolling_swap_slots.json").open("w", encoding="utf-8") as f:
        json.dump(slot_tables, f, ensure_ascii=False, indent=2)

    print("\n=== Rolling swap ===")
    for d in slim["decisions"]:
        print(
            f"  {d['decide_block']}→{d['eval_block']}: "
            f"vs_static change={d['vs_static']['all_slots']['change_rate']:.3f}  "
            f"μ_roll={d['mu_delta_rolling_on_eval'].get('mean_delta')}  "
            f"μ_static={d['mu_delta_static_on_eval'].get('mean_delta')}  "
            f"gap={d['rolling_minus_static_delta']}"
        )
    print("drift:", slim["decision_to_decision_drift"])
    print("done →", OUT_DIR)


if __name__ == "__main__":
    main()
