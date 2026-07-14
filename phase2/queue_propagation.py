"""Queue propagation: faster dispensing → shorter call wait (with fixed windows and schedule staff)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from phase2.staff_schedule import N_WINDOWS


def attach_staff(rx: pd.DataFrame, staff: pd.DataFrame) -> pd.DataFrame:
    out = rx.copy()
    if "日期" not in out.columns:
        raise KeyError("rx needs a 日期 column to join schedules")
    out["日期"] = pd.to_datetime(out["日期"]).dt.date
    staff = staff.copy()
    staff["日期"] = pd.to_datetime(staff["日期"]).dt.date
    return out.merge(staff, on="日期", how="left")


def fit_queue_propagation_model(rx: pd.DataFrame, staff: pd.DataFrame | None = None) -> dict:
    """Calibrate elasticity of W0 to system dispensing load and staffing from observed data."""
    valid = rx.dropna(subset=["叫号等待_分钟", "预估配药_分钟"]).copy()
    if staff is not None:
        if "日期" not in valid.columns and "日期" in rx.columns:
            valid["日期"] = rx["日期"]
        valid = attach_staff(valid, staff)
        valid = valid.dropna(subset=["调配人数", "发药人数", "产能代理"])
        valid = valid[valid["调配人数"] > 0]

    cols = ["预估配药_分钟", "Peak0", "Load0"]
    if "调配人数" in valid.columns:
        cols += ["调配人数", "有效窗口"]

    x_rx = valid[cols].astype(float)
    x_rx = sm.add_constant(x_rx)
    m_rx = sm.OLS(valid["叫号等待_分钟"].astype(float), x_rx).fit()

    mean_disp = float(valid["预估配药_分钟"].mean())
    mean_wait = float(valid["叫号等待_分钟"].mean())
    mean_cap = float(valid["产能代理"].mean()) if "产能代理" in valid.columns else float(N_WINDOWS)
    mean_dispense_staff = float(valid["调配人数"].mean()) if "调配人数" in valid.columns else 4.0

    beta_disp = float(m_rx.params.get("预估配药_分钟", 0.4))
    beta_window = float(m_rx.params.get("有效窗口", 0.0)) if "有效窗口" in m_rx.params else 0.0
    beta_dispense_staff = (
        float(m_rx.params.get("调配人数", 0.0)) if "调配人数" in m_rx.params else 0.0
    )

    return {
        "description": "Call wait rises when system dispensing slows; staffing/fixed windows modulate queue sensitivity",
        "n_windows_fixed": N_WINDOWS,
        "beta_disp_min": round(beta_disp, 4),
        "beta_window": round(beta_window, 4),
        "beta_dispense_staff": round(beta_dispense_staff, 4),
        "mean_disp_min": round(mean_disp, 4),
        "mean_wait_min": round(mean_wait, 4),
        "mean_capacity": round(mean_cap, 4),
        "mean_dispense_staff": round(mean_dispense_staff, 4),
        "queue_amplify": 1.0,
        "load_amplify": 0.5,
        "peak_amplify": 0.5,
        "staff_amplify": 1.0,
        "rx_regression_r2": round(float(m_rx.rsquared), 4),
        "staff_match_rate": round(float(valid["调配人数"].notna().mean()), 4)
        if "调配人数" in valid.columns
        else None,
        "coef": {str(k): round(float(v), 4) for k, v in m_rx.params.items()},
        "note": "ΔW ≈ W × (ΔD/D̄) × load/peak × (C̄/C_today); windows fixed at 6, C=min(6,发药人数)+0.5×调配人数",
    }


def propagate_call_wait_minutes(
    w_base_min: float,
    system_disp_saved_min: float,
    peak: int,
    load0: int,
    model: dict,
    capacity: float | None = None,
) -> float:
    """
    After system-average dispensing shortens, call wait shortens proportionally.

    Lower staffing / capacity increases marginal queue impact of the same dispensing speedup.
    """
    d_mean = model["mean_disp_min"]
    speedup_ratio = system_disp_saved_min / max(d_mean, 0.01)
    load_factor = 1.0 + model["load_amplify"] * (load0 / 3.0)
    peak_factor = 1.0 + model["peak_amplify"] * float(peak)

    cap_mean = model.get("mean_capacity", float(N_WINDOWS))
    cap = capacity if capacity is not None and capacity > 0 else cap_mean
    # Queue is more sensitive when staffing is low
    staff_factor = 1.0 + model.get("staff_amplify", 1.0) * max(0.0, (cap_mean / cap) - 1.0)

    reduction_frac = min(
        0.85,
        speedup_ratio * load_factor * peak_factor * staff_factor * model["queue_amplify"],
    )
    return max(0.0, w_base_min * (1.0 - reduction_frac))


def save_queue_model(model: dict, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(model, f, ensure_ascii=False, indent=2)
