"""S4.1: Train wait-time-minute regressors (Ridge primary + HGB/MLP controls)."""

from __future__ import annotations

import argparse

import pandas as pd

from phase2.config import DATA_DIR
from phase2.continuous_reward import (
    OUT_DIR,
    build_feature_frame,
    save_bundles,
    train_wait_regressors,
)
from phase2.staff_schedule import load_daily_staff


def main() -> None:
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    print("[1] Loading rx + staff schedule...")
    rx = pd.read_parquet(DATA_DIR / "rx_level.parquet")
    disc = pd.read_parquet(DATA_DIR / "phase2" / "rx_discretized.parquet")
    rx = rx.merge(
        disc[["处方编号", "Peak0", "Load0"]],
        on="处方编号",
        how="left",
    )
    staff = load_daily_staff()
    rx = build_feature_frame(rx, staff)
    print(f"  n={len(rx):,}")

    print("[2] Training ridge + hgb + mlp...")
    bundles = train_wait_regressors(rx, staff=None)
    save_bundles(bundles, OUT_DIR)
    print("[3] saved →", OUT_DIR)
    for name, b in bundles.items():
        print(f"  {name}: {b.metrics}")


if __name__ == "__main__":
    main()
