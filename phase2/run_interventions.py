"""Run layout intervention simulation and print comparison report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "vendor" / "NS-SCMMAB-main") not in sys.path:
    sys.path.insert(0, str(_ROOT / "vendor" / "NS-SCMMAB-main"))

from phase2.config import DATA_DIR
from phase2.intervention import build_default_scenarios, run_intervention_study, save_results
from phase2.queue_propagation import fit_queue_propagation_model, save_queue_model
from phase2.simulate import load_cpts
from phase2.staff_schedule import N_WINDOWS, load_daily_staff, save_daily_staff

PHASE2_DIR = DATA_DIR / "phase2"


def print_report(results: dict, queue_model: dict) -> None:
    print()
    print("=" * 72)
    print("Layout intervention simulation (two-mode comparison)")
    print(f"  Fixed window count={N_WINDOWS}")
    print(
        f"  Queue propagation: dispensing speedup ratio × load/peak × staffing sensitivity"
        f" (β_D={queue_model['beta_disp_min']}, "
        f"β_调配={queue_model['beta_dispense_staff']}, "
        f"β_窗口={queue_model['beta_window']}, "
        f"R²={queue_model['rx_regression_r2']})"
    )
    print("=" * 72)

    for name, r in results.items():
        if "error" in r:
            print(f"\n[{name}] Error: {r['error']}")
            continue
        print(f"\n--- {name}: {r['scenario']['description']} ---")
        for mode, label in [
            ("fixed_w", "Old logic: fixed W0"),
            ("with_queue", "New logic: queue propagation D0→W0 (incl. staffing)"),
        ]:
            m = r[mode]
            print(f"  [{label}]")
            print(f"    System dispensing saved: {m['system_disp_saved_min']:.3f} min/Rx")
            print(
                f"    Call wait: W0 {m['mean_W0_base']:.2f}→{m['mean_W0_cf']:.2f}"
                f"  ({m['mean_w_min_saved']:.3f} min, downgraded {m['W0_downgraded_rate']:.1%})"
            )
            print(
                f"    Dispensing: D0 {m['mean_D0_base']:.2f}→{m['mean_D0_cf']:.2f}"
                f"  (saved {m['disp_min_saved']:.3f} min)"
            )
            print(
                f"    E[Y0]: {m['mean_EY_base']:.3f}→{m['mean_EY_cf']:.3f}"
                f"  (↓{m['EY_reduction']:.3f} levels)"
            )
            print(
                f"    P(Q4): {m['P_Y3_base']:.1%}→{m['P_Y3_cf']:.1%}"
                f"  (↓{m['P_Y3_reduction']:.1%})"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Layout intervention simulation")
    parser.add_argument("--sample-n", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("[1/6] Loading data...")
    cpts = load_cpts(PHASE2_DIR / "calibrated_cpts.json")
    with (PHASE2_DIR / "discretize_thresholds.json").open(encoding="utf-8") as f:
        thresholds = json.load(f)
    rx_disc = pd.read_parquet(PHASE2_DIR / "rx_discretized.parquet")
    rx_dates = pd.read_parquet(DATA_DIR / "rx_level.parquet", columns=["处方编号", "日期"])
    rx_disc = rx_disc.merge(rx_dates, on="处方编号", how="left")

    print("[2/6] Parsing schedule staffing (fixed windows=6)...")
    staff = load_daily_staff()
    save_daily_staff(staff, PHASE2_DIR / "daily_staff.parquet")
    print(
        f"  Days={len(staff)}, mean 调配人数={staff['调配人数'].mean():.1f}, "
        f"mean 发药人数={staff['发药人数'].mean():.1f}, mean 产能代理={staff['产能代理'].mean():.1f}"
    )

    print("[3/6] Calibrating queue propagation model (incl. staffing)...")
    queue_model = fit_queue_propagation_model(rx_disc, staff=staff)
    save_queue_model(queue_model, PHASE2_DIR / "queue_model.json")
    print(
        f"  β_D={queue_model['beta_disp_min']}, β_调配={queue_model['beta_dispense_staff']}, "
        f"β_窗口={queue_model['beta_window']}, R²={queue_model['rx_regression_r2']}"
    )
    print(f"  Schedule match rate={queue_model['staff_match_rate']}")

    # Attach staffing to prescriptions for per-day capacity adjustment during interventions
    from phase2.queue_propagation import attach_staff

    rx_disc = attach_staff(rx_disc, staff)

    print("[4/6] Building layout scenarios...")
    items_for_scenarios = pd.read_parquet(
        DATA_DIR / "item_level.parquet",
        columns=["处方编号", "drugid", "是否机器_最终", "调配秒数_最终", "货架区域_最终"],
    )
    scenarios = build_default_scenarios(items_for_scenarios)

    print("[5/6] Intervention simulation...")
    rng = np.random.default_rng(args.seed)
    eligible = rx_disc[rx_disc["M0"] < 3]["处方编号"].unique()
    chosen = rng.choice(eligible, size=min(args.sample_n, len(eligible)), replace=False)
    items = items_for_scenarios[items_for_scenarios["处方编号"].isin(chosen)]

    results = run_intervention_study(
        cpts=cpts,
        rx_disc=rx_disc,
        items=items,
        scenarios=scenarios,
        d0_thresholds=thresholds["D0"],
        w0_thresholds=thresholds["W0"],
        queue_model=queue_model,
        chosen_ids=chosen,
    )

    out_path = PHASE2_DIR / "intervention_results.json"
    payload = {
        "queue_model": queue_model,
        "n_windows_fixed": N_WINDOWS,
        "scenarios": results,
    }
    save_results(payload, out_path)
    print(f"[6/6] Saved {out_path}")
    print_report(results, queue_model)


if __name__ == "__main__":
    main()
