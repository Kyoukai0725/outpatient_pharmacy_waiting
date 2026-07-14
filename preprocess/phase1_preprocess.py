"""
Phase 1: waiting-time data preprocessing.

Pipeline aligned with the causal chain: check-in → call → dispense (machine/manual) → verify & hand out.
- Item level: join dispense time, bin location, shelf zone
- Prescription level: aggregate dispense features, discretize wait duration into quartiles
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from preprocess.config import (
    DATA_DIR,
    DISPENSE_FILE,
    LAYOUT_FILE,
    MACHINE_DISPENSE_SEC,
    PROJECT_ROOT,
    WAIT_COLS,
    WAIT_DATA_DIRS,
)


def extract_drugid(series: pd.Series) -> pd.Series:
    return series.astype(str).str.extract(r"^(\S+)", expand=False)


def shelf_zone(loc) -> str | None:
    if pd.isna(loc):
        return None
    text = str(loc).strip()
    if not text:
        return None
    return text[0].upper()


def load_dispense_lookup() -> pd.DataFrame:
    disp = pd.read_excel(DISPENSE_FILE, engine="openpyxl")
    disp = disp.rename(columns={"材料编号": "drugid"})
    disp["drugid"] = disp["drugid"].astype(str).str.strip()
    disp["料位号"] = disp["料位号"].astype(str).str.strip()
    disp["货架区域"] = disp["料位号"].map(shelf_zone)
    disp["是否机器"] = disp["是否为药包机品项"] == "Y"
    disp["调配秒数"] = np.where(
        disp["是否机器"],
        disp["从机器掉落到进入药框时间/秒"].fillna(MACHINE_DISPENSE_SEC),
        disp["单品项人工调配时间/秒"],
    )
    return disp[
        [
            "drugid",
            "品名规格",
            "料位号",
            "货架区域",
            "是否机器",
            "调配秒数",
        ]
    ].drop_duplicates("drugid")


def load_layout_lookup() -> pd.DataFrame:
    layout = pd.read_excel(LAYOUT_FILE, engine="openpyxl")
    layout = layout.rename(columns={"药品编码": "drugid"})
    layout["drugid"] = layout["drugid"].astype(str).str.strip()
    layout["料位号_方位图"] = layout["料位号"].astype(str).str.strip()
    layout["货架区域_方位图"] = layout["料位号_方位图"].map(shelf_zone)
    return layout[["drugid", "料位号_方位图", "货架区域_方位图", "是否使用"]].drop_duplicates(
        "drugid"
    )


def load_wait_items(years: list[int]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for year in years:
        folder = WAIT_DATA_DIRS[year]
        pattern = f"侯药-{year}*.xls"
        files = sorted(folder.glob(pattern))
        if not files:
            raise FileNotFoundError(
                f"No waiting-time data found for {year}: {folder / pattern}"
            )
        for path in files:
            df = pd.read_excel(path, usecols=WAIT_COLS)
            df["数据年份"] = year
            df["来源文件"] = path.name
            frames.append(df)
    items = pd.concat(frames, ignore_index=True)
    items["drugid"] = extract_drugid(items["药品编码"])

    for col in ["报到时间", "呼叫时间", "摆药完成时间", "发药E化时间"]:
        items[f"{col}_dt"] = pd.to_datetime(items[col], errors="coerce")

    items["候药时长_分钟"] = (
        items["发药E化时间_dt"] - items["报到时间_dt"]
    ).dt.total_seconds() / 60.0
    items["叫号等待_分钟"] = (
        items["呼叫时间_dt"] - items["报到时间_dt"]
    ).dt.total_seconds() / 60.0
    items["配药时长_分钟"] = (
        items["摆药完成时间_dt"] - items["呼叫时间_dt"]
    ).dt.total_seconds() / 60.0
    items["核对发药_分钟"] = (
        items["发药E化时间_dt"] - items["摆药完成时间_dt"]
    ).dt.total_seconds() / 60.0

    items["报到小时"] = items["报到时间_dt"].dt.hour
    items["星期几"] = items["报到时间_dt"].dt.dayofweek
    items["月份"] = items["报到时间_dt"].dt.month
    items["日期"] = items["报到时间_dt"].dt.date

    return items


def merge_reference_tables(items: pd.DataFrame, disp: pd.DataFrame, layout: pd.DataFrame) -> pd.DataFrame:
    merged = items.merge(disp, on="drugid", how="left", indicator="_disp_match")
    merged = merged.merge(layout, on="drugid", how="left")

    merged["料位号_最终"] = merged["料位号"].fillna(merged["料位号_方位图"])
    merged["货架区域_最终"] = merged["货架区域"].fillna(merged["货架区域_方位图"])
    merged["调配秒数_最终"] = merged["调配秒数"]
    merged["是否机器_最终"] = merged["是否机器"]

    manual_mean = merged.loc[merged["是否机器_最终"] == False, "调配秒数_最终"].mean()
    merged["调配秒数_最终"] = merged["调配秒数_最终"].fillna(manual_mean)
    merged["是否机器_最终"] = merged["是否机器"].astype("boolean").fillna(False).astype(bool)
    merged["匹配调配表"] = merged["_disp_match"] == "both"
    merged = merged.drop(columns=["_disp_match"])
    return merged


def prepare_for_export(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    raw_time_cols = ["报到时间", "呼叫时间", "摆药完成时间", "发药E化时间"]
    out = out.drop(columns=[c for c in raw_time_cols if c in out.columns])
    if "日期" in out.columns:
        out["日期"] = pd.to_datetime(out["日期"])
    return out


def build_rx_level(items: pd.DataFrame, quartile_bins: np.ndarray) -> pd.DataFrame:
    valid_rx = (
        items.groupby("处方编号", as_index=False)
        .agg(
            候药时长_分钟=("候药时长_分钟", "first"),
            报到时间_dt=("报到时间_dt", "first"),
            报到小时=("报到小时", "first"),
            星期几=("星期几", "first"),
            月份=("月份", "first"),
            日期=("日期", "first"),
            数据年份=("数据年份", "first"),
            叫号等待_分钟=("叫号等待_分钟", "first"),
            配药时长_分钟=("配药时长_分钟", "first"),
            核对发药_分钟=("核对发药_分钟", "first"),
        )
        .query("候药时长_分钟 > 0")
        .copy()
    )

    daily_volume = valid_rx.groupby("日期")["处方编号"].transform("count")
    valid_rx["当日处方数"] = daily_volume

    item_stats = items.groupby("处方编号").agg(
        品项数=("drugid", "count"),
        机器品项数=("是否机器_最终", lambda s: int(s.sum())),
        人工品项数=("是否机器_最终", lambda s: int((~s).sum())),
        未知调配方式数=("匹配调配表", lambda s: int((~s).sum())),
        预估调配总秒=("调配秒数_最终", "sum"),
        平均单品调配秒=("调配秒数_最终", "mean"),
        最大单品调配秒=("调配秒数_最终", "max"),
        货架区域数=("货架区域_最终", "nunique"),
        机器品项占比=("是否机器_最终", "mean"),
    )
    item_stats["调配方式"] = np.select(
        [
            item_stats["机器品项数"] == item_stats["品项数"],
            item_stats["人工品项数"] == item_stats["品项数"],
        ],
        ["纯机器", "纯人工"],
        default="混合",
    )

    rx = valid_rx.merge(item_stats, left_on="处方编号", right_index=True, how="left")
    rx["候药四分位"] = pd.cut(
        rx["候药时长_分钟"],
        bins=quartile_bins,
        labels=[0, 1, 2, 3],
        include_lowest=True,
        right=True,
    ).astype("Int64")

    rx["叫号等待_分钟"] = rx["叫号等待_分钟"].clip(lower=0)
    rx["配药时长_分钟"] = rx["配药时长_分钟"].clip(lower=0)
    rx["核对发药_分钟"] = rx["核对发药_分钟"].clip(lower=0)
    rx["预估配药_分钟"] = rx["预估调配总秒"] / 60.0
    rx["阶段时长可信"] = (
        rx["叫号等待_分钟"].notna()
        & rx["配药时长_分钟"].notna()
        & rx["核对发药_分钟"].notna()
        & (rx["配药时长_分钟"] > 0)
    )
    return rx


def compute_quartile_bins(wait_minutes: pd.Series) -> tuple[np.ndarray, dict]:
    q1, q2, q3 = wait_minutes.quantile([0.25, 0.50, 0.75]).tolist()
    bins = np.array([-np.inf, q1, q2, q3, np.inf], dtype=float)
    meta = {
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
    return bins, meta


def summarize(items: pd.DataFrame, rx: pd.DataFrame, quartile_meta: dict) -> dict:
    return {
        "project_root": str(PROJECT_ROOT),
        "item_rows": int(len(items)),
        "rx_rows": int(len(rx)),
        "years": sorted(items["数据年份"].unique().tolist()),
        "drugid_count": int(items["drugid"].nunique()),
        "dispense_match_rate": round(float(items["匹配调配表"].mean()), 4),
        "machine_item_rate": round(float(items["是否机器_最终"].mean()), 4),
        "call_time_coverage": round(float(items["呼叫时间_dt"].notna().mean()), 4),
        "dispense_done_coverage": round(float(items["摆药完成时间_dt"].notna().mean()), 4),
        "wait_minutes": {
            "mean": round(float(rx["候药时长_分钟"].mean()), 4),
            "median": round(float(rx["候药时长_分钟"].median()), 4),
            "p90": round(float(rx["候药时长_分钟"].quantile(0.9)), 4),
        },
        "stage_minutes_rx_mean": {
            "叫号等待": round(float(rx["叫号等待_分钟"].mean()), 4),
            "配药时长_系统时间戳": round(float(rx["配药时长_分钟"].mean()), 4),
            "核对发药": round(float(rx["核对发药_分钟"].mean()), 4),
            "预估配药_调配表": round(float(rx["预估配药_分钟"].mean()), 4),
        },
        "stage_timestamp_reliable_rate": round(float(rx["阶段时长可信"].mean()), 4),
        "quartile_counts": {
            str(k): int(v) for k, v in rx["候药四分位"].value_counts(sort=True).items()
        },
        "quartile_thresholds": quartile_meta,
        "dispense_type_counts": rx["调配方式"].value_counts().astype(int).to_dict(),
    }


def run(years: list[int], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[1/6] Loading dispense times and layout map...")
    disp = load_dispense_lookup()
    layout = load_layout_lookup()
    print(f"  Dispense table: {len(disp)} items | Layout map: {len(layout)} items")

    print(f"[2/6] Loading waiting-time data ({years})...")
    items = load_wait_items(years)
    print(f"  Item records: {len(items):,}")

    print("[3/6] Joining dispense features...")
    items = merge_reference_tables(items, disp, layout)
    valid_wait = items.loc[items["候药时长_分钟"] > 0, "候药时长_分钟"]
    quartile_bins, quartile_meta = compute_quartile_bins(valid_wait)

    print("[4/6] Building prescription-level features and quartile labels...")
    rx = build_rx_level(items, quartile_bins)

    print("[5/6] Saving results...")
    item_path = output_dir / "item_level.parquet"
    rx_path = output_dir / "rx_level.parquet"
    prepare_for_export(items).to_parquet(item_path, index=False)
    prepare_for_export(rx).to_parquet(rx_path, index=False)

    summary = summarize(items, rx, quartile_meta)
    summary_path = output_dir / "preprocess_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    thresholds_path = output_dir / "quartile_thresholds.json"
    with thresholds_path.open("w", encoding="utf-8") as f:
        json.dump(quartile_meta, f, ensure_ascii=False, indent=2)

    print("[6/6] Done")
    print(f"  Item level: {item_path}")
    print(f"  Rx level:   {rx_path}")
    print(f"  Summary:    {summary_path}")
    print()
    print("Quartile thresholds (minutes):")
    print(f"  Q1 <= {quartile_meta['Q1_upper_min']}")
    print(f"  Q2 <= {quartile_meta['Q2_upper_min']}")
    print(f"  Q3 <= {quartile_meta['Q3_upper_min']}")
    print(f"  Q4 >  {quartile_meta['Q3_upper_min']}")
    print()
    print("Rx-level stage means (minutes):")
    for k, v in summary["stage_minutes_rx_mean"].items():
        print(f"  {k}: {v}")
    print()
    print("Wait-time quartile distribution:", summary["quartile_counts"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 1: waiting-time data preprocessing")
    parser.add_argument(
        "--years",
        type=int,
        nargs="+",
        default=[2024, 2025],
        help="Years to process (default: 2024 2025)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DATA_DIR,
        help="Output directory",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(years=args.years, output_dir=args.output_dir)
