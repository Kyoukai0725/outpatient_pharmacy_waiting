"""Run feasible intervention simulation: in-machine swaps + near-end shelf rearrangement."""

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
from phase2.intervention import build_feasible_scenarios, run_intervention_study, save_results
from phase2.queue_propagation import attach_staff, fit_queue_propagation_model, save_queue_model
from phase2.simulate import load_cpts
from phase2.staff_schedule import N_WINDOWS, load_daily_staff, save_daily_staff

PHASE2_DIR = DATA_DIR / "phase2"


def print_report(results: dict, queue_model: dict) -> None:
    print()
    print("=" * 72)
    print("Feasible intervention simulation (machine full: swap / shelf rearrange)")
    print(f"  Fixed window count={N_WINDOWS}")
    print(
        f"  Queue propagation includes staffing (β_D={queue_model['beta_disp_min']}, "
        f"β_调配={queue_model['beta_dispense_staff']})"
    )
    print("=" * 72)

    for name, r in results.items():
        if "error" in r:
            print(f"\n[{name}] Error: {r['error']}")
            continue
        sc = r["scenario"]
        print(f"\n--- {name}: {sc['description']} ---")
        if sc.get("meta"):
            meta = sc["meta"]
            if "expected_net_hours" in meta:
                print(
                    f"  Rough net savings: {meta['expected_net_hours']} hours"
                    f" (saved {meta.get('expected_save_hours')} / cost {meta.get('expected_cost_hours')})"
                )
            if "fast_zones" in meta:
                print(f"  Fast zones={meta['fast_zones']} slow zones={meta['slow_zones']}")
                print(
                    f"  Fast-zone mean={meta['fast_zone_mean_sec']}s slow-zone mean={meta['slow_zone_mean_sec']}s"
                )

        m = r["with_queue"]
        print("  [with queue propagation]")
        print(f"    Affected prescriptions: {m['n_prescriptions']:,}")
        print(f"    System dispensing saved: {m['system_disp_saved_min']:.3f} min/Rx")
        print(
            f"    Call wait: W0 {m['mean_W0_base']:.2f}→{m['mean_W0_cf']:.2f}"
            f"  (shortened {m['mean_w_min_saved']:.3f} min)"
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
    parser = argparse.ArgumentParser(description="Feasible intervention simulation")
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

    print("[2/6] Parsing schedule staffing...")
    staff_path = PHASE2_DIR / "daily_staff.parquet"
    if staff_path.exists():
        staff = pd.read_parquet(staff_path)
    else:
        staff = load_daily_staff()
        save_daily_staff(staff, staff_path)
    print(f"  Days={len(staff)}, mean 调配人数={staff['调配人数'].mean():.1f}")

    print("[3/6] Queue model...")
    queue_path = PHASE2_DIR / "queue_model.json"
    if queue_path.exists():
        with queue_path.open(encoding="utf-8") as f:
            queue_model = json.load(f)
    else:
        queue_model = fit_queue_propagation_model(rx_disc, staff=staff)
        save_queue_model(queue_model, queue_path)
    rx_disc = attach_staff(rx_disc, staff)

    print("[4/6] Building feasible scenarios (machine swap / shelf rearrange)...")
    items_all = pd.read_parquet(
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
    scenarios = build_feasible_scenarios(items_all, constrained_machine=True)
    for sc in scenarios:
        print(f"  - {sc.name}: {sc.description}")
        if sc.kind == "machine_swap" and sc.meta.get("pairs"):
            print(
                f"    Eligible manual pool={sc.meta.get('n_eligible_manual')}, "
                f"ineligible removals={sc.meta.get('n_outs_ineligible')}, "
                f"rough net savings={sc.meta.get('expected_net_hours')}h"
            )

    print("[5/6] Intervention simulation...")
    rng = np.random.default_rng(args.seed)
    chosen = rng.choice(
        rx_disc["处方编号"].unique(),
        size=min(args.sample_n, rx_disc["处方编号"].nunique()),
        replace=False,
    )
    items = items_all[items_all["处方编号"].isin(chosen)]

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

    out_path = PHASE2_DIR / "feasible_intervention_constrained.json"
    payload = {
        "constraint": (
            "Machine full, no net increase; entry excludes P/S suffix and name contains 口服液/合剂/注射; "
            "removal prioritizes ineligible items already in machine"
        ),
        "n_windows_fixed": N_WINDOWS,
        "queue_model": queue_model,
        "scenarios": results,
    }
    save_results(payload, out_path)
    print(f"[6/6] Saved {out_path}")
    print_report(results, queue_model)


if __name__ == "__main__":
    main()
