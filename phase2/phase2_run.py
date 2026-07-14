"""
Phase 2: causal graph definition + Gibbs MCMC CPT calibration + simulation validation

Dispensing node D0 uses the quartile of 预估配药_分钟 (not system timestamps).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

# Wire up NS-SCMMAB
_ROOT = Path(__file__).resolve().parents[1]
_NS = _ROOT / "vendor" / "NS-SCMMAB-main"
if str(_NS) not in sys.path:
    sys.path.insert(0, str(_NS))

from phase2.causal_graph import PharmacyCausalGraph, build_causal_diagram
from phase2.config import DATA_DIR, MCMC_BURN, MCMC_DRAWS, RX_LEVEL
from phase2.discretize import build_discretized_rx, save_thresholds
from phase2.mcmc_calibrate import (
    build_count_tables,
    gibbs_sample_cpts,
    save_calibration,
    train_test_split,
)
from phase2.simulate import evaluate, simulate_batch


def run(
    rx_path: Path = RX_LEVEL,
    output_dir: Path | None = None,
    test_frac: float = 0.2,
    n_draws: int = MCMC_DRAWS,
    n_burn: int = MCMC_BURN,
    seed: int = 42,
) -> None:
    output_dir = output_dir or (DATA_DIR / "phase2")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[1/7] Loading prescription-level data...")
    rx = pd.read_parquet(rx_path)
    print(f"  Raw prescriptions: {len(rx):,}")

    print("[2/7] Discretizing causal nodes (D0=预估配药_分钟)...")
    disc, thresholds = build_discretized_rx(rx)
    disc_path = output_dir / "rx_discretized.parquet"
    disc.to_parquet(disc_path, index=False)
    save_thresholds(thresholds, output_dir / "discretize_thresholds.json")
    print(f"  Valid samples: {len(disc):,}")

    print("[3/7] Saving causal graph...")
    graph = PharmacyCausalGraph()
    with (output_dir / "causal_graph.json").open("w", encoding="utf-8") as f:
        json.dump(graph.to_dict(), f, ensure_ascii=False, indent=2)
    cd = build_causal_diagram()
    print(f"  Nodes: {sorted(cd.V)}")
    print(f"  Edges: {cd.edges}")

    print("[4/7] Splitting train/test...")
    train_df, test_df = train_test_split(disc, test_frac=test_frac, seed=seed)
    print(f"  Train: {len(train_df):,} | Test: {len(test_df):,}")

    print("[5/7] Gibbs MCMC CPT calibration...")
    counts = build_count_tables(train_df)
    cpts, _draws = gibbs_sample_cpts(counts, n_draws=n_draws, n_burn=n_burn, seed=seed)
    meta = {
        "mcmc_draws": n_draws,
        "mcmc_burn": n_burn,
        "train_size": int(len(train_df)),
        "test_size": int(len(test_df)),
        "dispense_node": "D0",
        "dispense_source": "预估配药_分钟",
        "seed": seed,
    }
    save_calibration(cpts, counts, meta, output_dir)
    print("  Saved calibrated_cpts.json")

    print("[6/7] Posterior predictive check...")
    metrics = evaluate(test_df, cpts)
    with (output_dir / "validation_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"  Test log-lik: {metrics['loglik_mean']}")
    print(f"  Test argmax accuracy: {metrics['accuracy_argmax']:.2%}")

    print("[7/7] Forward simulation example (pure-machine prescriptions M0=3)...")
    sim = simulate_batch(cpts, n=5000, exogenous={"M0": 3}, seed=seed)
    sim_summary = {
        "scenario": "M0=3 (纯机器)",
        "Y0_distribution": sim["Y0"].value_counts(normalize=True).sort_index().round(4).to_dict(),
        "D0_distribution": sim["D0"].value_counts(normalize=True).sort_index().round(4).to_dict(),
        "mean_D0_level": round(float(sim["D0"].mean()), 4),
    }
    with (output_dir / "sim_example_pure_machine.json").open("w", encoding="utf-8") as f:
        json.dump(sim_summary, f, ensure_ascii=False, indent=2)

    print()
    print("Phase 2 complete. Output directory:", output_dir)
    print("  rx_discretized.parquet")
    print("  causal_graph.json")
    print("  calibrated_cpts.json")
    print("  validation_metrics.json")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 2: causal graph + MCMC calibration")
    p.add_argument("--rx-path", type=Path, default=RX_LEVEL)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--test-frac", type=float, default=0.2)
    p.add_argument("--draws", type=int, default=MCMC_DRAWS)
    p.add_argument("--burn", type=int, default=MCMC_BURN)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        rx_path=args.rx_path,
        output_dir=args.output_dir,
        test_frac=args.test_frac,
        n_draws=args.draws,
        n_burn=args.burn,
        seed=args.seed,
    )
