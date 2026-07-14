"""Nested counterfactual path decomposition (model-based).

Decompose predicted wait reduction of swap/shelf vs null into:
  A0: do(null) — baseline
  A1: change D/M/N only, W held at observed (direct dispensing path)
  A2: change D/M/N + propagate(W̃) (+ queue propagation)

  Δ_disp  = μ(A0) − μ(A1)
  Δ_queue = μ(A1) − μ(A2)
  Δ_total = μ(A0) − μ(A2)

Morning rush focus: repeat reporting on MorningRush subset (interaction/stratification, not a third independent edge).

Honest disclaimer: all are model-based inferences under Ridge + heuristic queue propagation, not randomized causal estimates.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from phase2.config import DATA_DIR
from phase2.continuous_reward import (
    _scenario_drug_ids,
    arm_to_scenario,
    build_feature_frame,
    day_rx_predictions,
    ensure_morning_rush,
    load_bundle,
    load_queue_model,
    rx_ids_touching_drugs,
)
from phase2.ns_mechanisms import annotate_regimes
from phase2.staff_schedule import load_daily_staff

PHASE2_DIR = DATA_DIR / "phase2"
OUT_DIR = PHASE2_DIR / "effect_decomposition"

ARMS_DEFAULT = ("swap20", "swap40", "shelf50")


def _load_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rx = pd.read_parquet(DATA_DIR / "rx_level.parquet")
    disc = pd.read_parquet(PHASE2_DIR / "rx_discretized.parquet")
    rx = rx.merge(disc[["处方编号", "Peak0", "Load0"]], on="处方编号", how="left")
    staff = load_daily_staff()
    rx = build_feature_frame(rx, staff)
    rx = annotate_regimes(rx, rx["日期"])
    rx = ensure_morning_rush(rx)
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
    return rx, items, staff


def _mean_yhat(pred: pd.DataFrame, mask: np.ndarray | None = None) -> float:
    if pred.empty or "yhat" not in pred.columns:
        return float("nan")
    y = pred["yhat"].to_numpy(dtype=float)
    if mask is not None:
        if not mask.any():
            return float("nan")
        y = y[mask]
    return float(np.nanmean(y))


def decompose_day(
    bundle,
    rx_day: pd.DataFrame,
    items_day: pd.DataFrame,
    scenario,
    queue_model: dict,
    *,
    affected_only: bool = False,
) -> dict:
    """Single-day nested decomposition; when affected_only, average only over prescriptions touching swapped drugs."""
    restrict = None
    if affected_only:
        drugs = _scenario_drug_ids(scenario)
        restrict = rx_ids_touching_drugs(items_day, drugs)
        if not restrict or len(restrict) < 3:
            return {}

    a0 = day_rx_predictions(
        bundle, rx_day, items_day, None, queue_model=queue_model, restrict_rx_ids=restrict
    )
    a1 = day_rx_predictions(
        bundle,
        rx_day,
        items_day,
        scenario,
        queue_model=queue_model,
        restrict_rx_ids=restrict,
        use_queue_propagation=False,
    )
    a2 = day_rx_predictions(
        bundle,
        rx_day,
        items_day,
        scenario,
        queue_model=queue_model,
        restrict_rx_ids=restrict,
        use_queue_propagation=True,
    )
    if a0.empty or a1.empty or a2.empty:
        return {}

    # Align by prescription id
    ids = set(a0["处方编号"]) & set(a1["处方编号"]) & set(a2["处方编号"])
    a0 = a0[a0["处方编号"].isin(ids)].sort_values("处方编号")
    a1 = a1[a1["处方编号"].isin(ids)].sort_values("处方编号")
    a2 = a2[a2["处方编号"].isin(ids)].sort_values("处方编号")

    rush = (
        a0["MorningRush"].astype(int).to_numpy() == 1
        if "MorningRush" in a0.columns
        else np.zeros(len(a0), dtype=bool)
    )

    mu0, mu1, mu2 = _mean_yhat(a0), _mean_yhat(a1), _mean_yhat(a2)
    mu0_r, mu1_r, mu2_r = (
        _mean_yhat(a0, rush),
        _mean_yhat(a1, rush),
        _mean_yhat(a2, rush),
    )
    mu0_o, mu1_o, mu2_o = (
        _mean_yhat(a0, ~rush),
        _mean_yhat(a1, ~rush),
        _mean_yhat(a2, ~rush),
    )

    def pack(m0, m1, m2) -> dict:
        d_disp = m0 - m1
        d_queue = m1 - m2
        d_tot = m0 - m2
        return {
            "mu_A0": round(m0, 6),
            "mu_A1": round(m1, 6),
            "mu_A2": round(m2, 6),
            "delta_disp": round(d_disp, 6),
            "delta_queue": round(d_queue, 6),
            "delta_total": round(d_tot, 6),
            "share_disp": round(d_disp / d_tot, 4) if abs(d_tot) > 1e-9 else None,
            "share_queue": round(d_queue / d_tot, 4) if abs(d_tot) > 1e-9 else None,
        }

    sys_d = float(a2["system_disp_saved_min"].iloc[0]) if "system_disp_saved_min" in a2 else 0.0
    return {
        "n_rx": int(len(a0)),
        "n_rush": int(rush.sum()),
        "system_disp_saved_min": round(sys_d, 6),
        "all": pack(mu0, mu1, mu2),
        "rush": pack(mu0_r, mu1_r, mu2_r),
        "other": pack(mu0_o, mu1_o, mu2_o),
        "rush_minus_other_total": round(
            (mu0_r - mu2_r) - (mu0_o - mu2_o), 6
        )
        if np.isfinite(mu0_r) and np.isfinite(mu0_o)
        else None,
    }


def run_decomposition(
    arms: list[str],
    sample_days: int = 80,
    seed: int = 42,
    affected_only: bool = False,
    model_name: str = "ridge",
) -> dict:
    print("[1] Loading data and models...")
    rx, items, _staff = _load_frames()
    bundle = load_bundle(model_name)
    qm = load_queue_model()
    items_all = items

    rx = rx.copy()
    rx["日期_d"] = pd.to_datetime(rx["日期"]).dt.date
    rx_day_map = rx.set_index("处方编号")["日期_d"]
    items = items.copy()
    items["日期_d"] = items["处方编号"].map(rx_day_map)

    days = sorted(rx["日期_d"].dropna().unique())
    rng = np.random.default_rng(seed)
    if len(days) > sample_days:
        days = list(rng.choice(days, size=sample_days, replace=False))
        days = sorted(days)

    scenarios = {a: arm_to_scenario(a, items_all) for a in arms}
    daily_rows = []
    for i, d in enumerate(days):
        rx_d = rx[rx["日期_d"] == d]
        it_d = items[items["日期_d"] == d]
        if len(rx_d) < 30 or len(it_d) == 0:
            continue
        for arm in arms:
            sc = scenarios[arm]
            if sc is None:
                continue
            dec = decompose_day(
                bundle, rx_d, it_d, sc, qm, affected_only=affected_only
            )
            if not dec:
                continue
            daily_rows.append(
                {
                    "date": str(d),
                    "arm": arm,
                    "n_rx": dec["n_rx"],
                    "n_rush": dec["n_rush"],
                    "system_disp_saved_min": dec["system_disp_saved_min"],
                    "delta_disp": dec["all"]["delta_disp"],
                    "delta_queue": dec["all"]["delta_queue"],
                    "delta_total": dec["all"]["delta_total"],
                    "share_disp": dec["all"]["share_disp"],
                    "share_queue": dec["all"]["share_queue"],
                    "mu_A0": dec["all"]["mu_A0"],
                    "mu_A2": dec["all"]["mu_A2"],
                    "rush_delta_total": dec["rush"]["delta_total"],
                    "other_delta_total": dec["other"]["delta_total"],
                    "rush_delta_disp": dec["rush"]["delta_disp"],
                    "rush_delta_queue": dec["rush"]["delta_queue"],
                    "other_delta_disp": dec["other"]["delta_disp"],
                    "other_delta_queue": dec["other"]["delta_queue"],
                    "rush_minus_other_total": dec["rush_minus_other_total"],
                }
            )
        if (i + 1) % 20 == 0:
            print(f"  days {i+1}/{len(days)}")

    df = pd.DataFrame(daily_rows)
    summary_arms = {}
    for arm in arms:
        sub = df[df["arm"] == arm]
        if sub.empty:
            continue

        def agg_col(c: str) -> dict:
            v = sub[c].dropna().astype(float)
            if v.empty:
                return {"mean": None, "se": None, "n": 0}
            return {
                "mean": round(float(v.mean()), 6),
                "se": round(float(v.std(ddof=1) / np.sqrt(len(v))), 6)
                if len(v) > 1
                else 0.0,
                "n": int(len(v)),
            }

        tot = agg_col("delta_total")
        disp = agg_col("delta_disp")
        que = agg_col("delta_queue")
        summary_arms[arm] = {
            "n_days": int(len(sub)),
            "delta_total": tot,
            "delta_disp": disp,
            "delta_queue": que,
            "share_disp_mean": round(float(sub["share_disp"].dropna().mean()), 4)
            if sub["share_disp"].notna().any()
            else None,
            "share_queue_mean": round(float(sub["share_queue"].dropna().mean()), 4)
            if sub["share_queue"].notna().any()
            else None,
            "rush_delta_total": agg_col("rush_delta_total"),
            "other_delta_total": agg_col("other_delta_total"),
            "rush_minus_other_total": agg_col("rush_minus_other_total"),
            "note_shares": "shares = path/total averaged across days (not path of averages)",
        }

    return {
        "identification": (
            "model-based nested counterfactual path decomposition under Ridge + "
            "heuristic queue propagation; NOT randomized causal estimates"
        ),
        "layers": {
            "A0": "do(null): observed D, observed W",
            "A1": "layout CF on D/M/N only; W = observed (dispense path)",
            "A2": "A1 + propagate_call_wait (queue path)",
            "rush_vs_other": "stratum interaction on same A0/A1/A2; not a third independent edge",
        },
        "affected_only": affected_only,
        "model": model_name,
        "sample_days_requested": sample_days,
        "arms": summary_arms,
        "n_daily_rows": int(len(df)),
    }, df


def main() -> None:
    parser = argparse.ArgumentParser(description="Nested path decomposition (model-based)")
    parser.add_argument("--arms", nargs="+", default=list(ARMS_DEFAULT))
    parser.add_argument("--sample-days", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--affected-only", action="store_true")
    parser.add_argument("--model", default="ridge")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary, daily = run_decomposition(
        arms=args.arms,
        sample_days=args.sample_days,
        seed=args.seed,
        affected_only=args.affected_only,
        model_name=args.model,
    )
    tag = "affected" if args.affected_only else "all"
    with (OUT_DIR / f"path_decomposition_{tag}.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    daily.to_csv(OUT_DIR / f"path_decomposition_daily_{tag}.csv", index=False)

    print("\n=== Nested path decomposition (model-based) ===")
    print("identification:", summary["identification"])
    for arm, s in summary["arms"].items():
        print(
            f"  {arm}: Δtotal={s['delta_total']['mean']:.4f} "
            f"(disp={s['delta_disp']['mean']:.4f}, queue={s['delta_queue']['mean']:.4f}) "
            f"shares disp/queue={s['share_disp_mean']}/{s['share_queue_mean']} "
            f"| rush Δ={s['rush_delta_total']['mean']:.4f} "
            f"other Δ={s['other_delta_total']['mean']:.4f}"
        )
    print("done →", OUT_DIR)


if __name__ == "__main__":
    main()
