"""Backfill summary and new columns from existing parquet files without re-reading all xls."""

import json
from pathlib import Path

import pandas as pd

from preprocess.phase1_preprocess import summarize
from preprocess.config import DATA_DIR


def main() -> None:
    rx_path = DATA_DIR / "rx_level.parquet"
    thresholds_path = DATA_DIR / "quartile_thresholds.json"
    summary_path = DATA_DIR / "preprocess_summary.json"
    item_path = DATA_DIR / "item_level.parquet"

    rx = pd.read_parquet(rx_path)
    items = pd.read_parquet(item_path)

    if "预估配药_分钟" not in rx.columns:
        rx["预估配药_分钟"] = rx["预估调配总秒"] / 60.0
    if "阶段时长可信" not in rx.columns:
        rx["阶段时长可信"] = (
            rx["叫号等待_分钟"].notna()
            & rx["配药时长_分钟"].notna()
            & rx["核对发药_分钟"].notna()
            & (rx["配药时长_分钟"] > 0)
        )
        rx.to_parquet(rx_path, index=False)

    if thresholds_path.exists():
        with thresholds_path.open(encoding="utf-8") as f:
            quartile_meta = json.load(f)
    else:
        import numpy as np

        q1, q2, q3 = rx["候药时长_分钟"].quantile([0.25, 0.50, 0.75]).tolist()
        quartile_meta = {
            "Q1_upper_min": round(q1, 4),
            "Q2_upper_min": round(q2, 4),
            "Q3_upper_min": round(q3, 4),
            "labels": {
                "0": f"Q1 (<= {q1:.2f} min)",
                "1": f"Q2 ({q1:.2f}, {q2:.2f}] min",
                "2": f"Q3 ({q2:.2f}, {q3:.2f}] min",
                "3": f"Q4 (> {q3:.2f} min)",
            },
        }
        with thresholds_path.open("w", encoding="utf-8") as f:
            json.dump(quartile_meta, f, ensure_ascii=False, indent=2)

    summary = summarize(items, rx, quartile_meta)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("updated:", rx_path)
    print("written:", summary_path)


if __name__ == "__main__":
    main()
