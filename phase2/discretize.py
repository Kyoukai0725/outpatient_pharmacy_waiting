"""Discretize continuous features into quartile states 0..3."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from phase2.config import N_LEVELS, PEAK_HOURS


def is_peak_hour(hour: int) -> int:
    for start, end in PEAK_HOURS:
        if start <= hour <= end:
            return 1
    return 0


def quartile_bin(series: pd.Series, prefix: str) -> tuple[pd.Series, dict]:
    valid = series.dropna()
    q1, q2, q3 = valid.quantile([0.25, 0.50, 0.75]).tolist()
    bins = [-np.inf, q1, q2, q3, np.inf]
    labels = pd.cut(
        series,
        bins=bins,
        labels=list(range(N_LEVELS)),
        include_lowest=True,
        right=True,
    )
    meta = {
        "variable": prefix,
        "Q1_upper": round(q1, 6),
        "Q2_upper": round(q2, 6),
        "Q3_upper": round(q3, 6),
    }
    return labels.astype("Int64"), meta


def discretize_n_items(n: pd.Series) -> pd.Series:
    return pd.cut(
        n,
        bins=[0, 1, 2, 4, np.inf],
        labels=[0, 1, 2, 3],
        include_lowest=True,
        right=True,
    ).astype("Int64")


def discretize_machine_ratio(ratio: pd.Series, dispense_type: pd.Series) -> pd.Series:
    out = pd.Series(index=ratio.index, dtype="Int64")
    out[dispense_type == "纯人工"] = 0
    out[(dispense_type == "混合") & (ratio <= 0.5)] = 1
    out[(dispense_type == "混合") & (ratio > 0.5)] = 2
    out[dispense_type == "纯机器"] = 3
    return out


def build_discretized_rx(rx: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    out = rx.copy()
    thresholds: dict = {}

    out["Peak0"] = out["报到小时"].apply(is_peak_hour).astype(int)
    out["Load0"], thresholds["Load0"] = quartile_bin(out["当日处方数"], "Load0")
    out["N0"] = discretize_n_items(out["品项数"])
    out["M0"] = discretize_machine_ratio(out["机器品项占比"], out["调配方式"])

    out["W0"], thresholds["W0"] = quartile_bin(out["叫号等待_分钟"], "W0")
    out["D0"], thresholds["D0"] = quartile_bin(out["预估配药_分钟"], "D0")
    out["V0"], thresholds["V0"] = quartile_bin(out["核对发药_分钟"], "V0")
    out["Y0"] = out["候药四分位"].astype("Int64")

    thresholds["N0"] = {"bins": [1, 2, 4, "inf"], "labels": ["1品项", "2品项", "3-4品项", "5+品项"]}
    thresholds["M0"] = {
        "labels": ["0=纯人工", "1=混合低", "2=混合高", "3=纯机器"],
    }
    thresholds["Peak0"] = {"labels": ["0=非高峰", "1=高峰"]}

    keep = [
        "处方编号",
        "Peak0",
        "Load0",
        "N0",
        "M0",
        "W0",
        "D0",
        "V0",
        "Y0",
        "叫号等待_分钟",
        "预估配药_分钟",
        "核对发药_分钟",
        "候药时长_分钟",
        "品项数",
        "机器品项占比",
        "调配方式",
        "当日处方数",
        "报到小时",
        "数据年份",
    ]
    return out[keep].dropna(subset=["W0", "D0", "V0", "Y0"]), thresholds


def save_thresholds(thresholds: dict, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(thresholds, f, ensure_ascii=False, indent=2)
