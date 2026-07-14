"""Fluid queue approximation: sensitivity experiment alternative to heuristic propagate_call_wait.

Does not assume Poisson arrivals. Fluid model by hourly arrival rate λ(t), service rate μ, window capacity c:
  dQ/dt = λ(t) − μ·c   (when Q>0 or λ>μc; otherwise no backlog)
Waiting at arrival approximated as W(t) ≈ Q(t)/(μ·c).

S8 baseline: hourly aggregation + single s for full day.
S8.1 refined sensitivity: prescription-level arrival times + time-varying s (rush 9–10 vs other).
Intervention shortens per-prescription service time to raise μ, observing Q and W changes. Not on main track.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from phase2.config import DATA_DIR
from phase2.queue_propagation import attach_staff, propagate_call_wait_minutes
from phase2.staff_schedule import N_WINDOWS, load_daily_staff

PHASE2_DIR = DATA_DIR / "phase2"
OUT_DIR = PHASE2_DIR / "fluid_queue"
HOURS = list(range(8, 18))  # business hours
DT_MIN = 1.0  # fluid step size (minutes)
RUSH_HOURS = {9, 10}
DAY_START_MIN = 8 * 60  # 08:00
DAY_END_MIN = 18 * 60  # 18:00


def _day_capacity(staff_row: pd.Series | None) -> float:
    if staff_row is None:
        return float(N_WINDOWS)
    if "产能代理" in staff_row.index and pd.notna(staff_row["产能代理"]):
        return max(float(staff_row["产能代理"]), 1.0)
    if "有效窗口" in staff_row.index and pd.notna(staff_row["有效窗口"]):
        return max(float(staff_row["有效窗口"]), 1.0)
    return float(N_WINDOWS)


def hourly_arrivals(rx_day: pd.DataFrame) -> dict[int, int]:
    h = pd.to_numeric(rx_day["报到小时"], errors="coerce").dropna().astype(int)
    counts = h.value_counts().to_dict()
    return {hour: int(counts.get(hour, 0)) for hour in HOURS}


def calibrate_service_minutes(
    rx: pd.DataFrame,
    staff: pd.DataFrame,
    sample_days: int = 40,
    seed: int = 42,
) -> dict:
    """
    Calibrate equivalent per-prescription service time s so fluid-predicted daily mean W
    matches observed call-wait. Search s on a grid, minimizing daily mean |W_fluid − W_obs|.
    """
    rng = np.random.default_rng(seed)
    rx = rx.copy()
    rx["日期"] = pd.to_datetime(rx["日期"]).dt.date
    days = sorted(rx["日期"].unique())
    if len(days) > sample_days:
        days = list(rng.choice(days, size=sample_days, replace=False))

    staff = staff.copy()
    staff["日期"] = pd.to_datetime(staff["日期"]).dt.date
    staff_map = staff.set_index("日期")

    grid = np.linspace(0.3, 2.5, 23)  # minutes per prescription
    best_s, best_mae = float(grid[0]), 1e9
    records = []

    for s in grid:
        errs = []
        for d in days:
            rx_d = rx[rx["日期"] == d]
            if len(rx_d) < 30 or rx_d["叫号等待_分钟"].notna().sum() < 20:
                continue
            cap = _day_capacity(staff_map.loc[d] if d in staff_map.index else None)
            pred = simulate_fluid_day(
                hourly_arrivals(rx_d),
                service_min=float(s),
                capacity=cap,
            )
            w_obs = float(rx_d["叫号等待_分钟"].mean())
            w_hat = float(pred["mean_wait_min"])
            errs.append(abs(w_hat - w_obs))
        if not errs:
            continue
        mae = float(np.mean(errs))
        records.append({"service_min": round(float(s), 4), "mae_day_mean_w": round(mae, 4)})
        if mae < best_mae:
            best_mae, best_s = mae, float(s)

    return {
        "service_min": round(best_s, 4),
        "calibrate_mae_day_mean_w": round(best_mae, 4),
        "n_cal_days": len(days),
        "grid": records,
        "note": "s chosen to match day-mean call-wait; fluid not Poisson",
    }


def simulate_fluid_day(
    arrivals: dict[int, int],
    service_min: float,
    capacity: float,
    dt: float = DT_MIN,
) -> dict:
    """
    Run fluid simulation for one day; return hourly mean wait and full-day mean.
    μ = 1/service_min (prescriptions/min/window), total service rate = μ * capacity.
    """
    s = max(float(service_min), 0.05)
    mu = 1.0 / s
    c = max(float(capacity), 1.0)
    rate_out = mu * c  # prescriptions/minute

    q = 0.0
    waits_by_hour: dict[int, list[float]] = {h: [] for h in HOURS}
    q_path = []

    for hour in HOURS:
        n = int(arrivals.get(hour, 0))
        lam = n / 60.0  # prescriptions/minute
        # Step integration within this hour
        for m in range(60):
            # Spread arrivals uniformly within this minute
            q = max(0.0, q + lam * dt - rate_out * dt)
            w = q / rate_out  # minutes
            waits_by_hour[hour].append(w)
            q_path.append(q)

    hour_mean = {
        h: float(np.mean(ws)) if ws else 0.0 for h, ws in waits_by_hour.items()
    }
    # Full-day mean wait weighted by arrivals
    total_n = sum(arrivals.get(h, 0) for h in HOURS)
    if total_n > 0:
        mean_w = sum(arrivals.get(h, 0) * hour_mean[h] for h in HOURS) / total_n
    else:
        mean_w = float(np.mean(list(hour_mean.values()))) if hour_mean else 0.0

    rush_n = arrivals.get(9, 0) + arrivals.get(10, 0)
    rush_w = (
        (arrivals.get(9, 0) * hour_mean[9] + arrivals.get(10, 0) * hour_mean[10]) / rush_n
        if rush_n > 0
        else float("nan")
    )

    return {
        "mean_wait_min": mean_w,
        "rush_wait_min": rush_w,
        "hour_wait": hour_mean,
        "max_queue": float(max(q_path) if q_path else 0.0),
        "service_min": s,
        "capacity": c,
        "mu_per_server": mu,
        "service_rate_total": rate_out,
    }


def fluid_wait_under_speedup(
    arrivals: dict[int, int],
    service_min: float,
    capacity: float,
    system_disp_saved_min: float,
    mean_disp_min: float,
) -> dict:
    """
    Dispensing speedup → equivalent service time shortened.
    s_cf = s * (1 - κ * saved/mean_disp), κ≤1 to avoid overshooting.
    """
    frac = float(system_disp_saved_min) / max(float(mean_disp_min), 0.01)
    frac = float(np.clip(frac, -0.5, 0.85))
    s_cf = max(0.05, float(service_min) * (1.0 - frac))
    base = simulate_fluid_day(arrivals, service_min, capacity)
    cf = simulate_fluid_day(arrivals, s_cf, capacity)
    return {
        "w_base": base["mean_wait_min"],
        "w_cf": cf["mean_wait_min"],
        "delta_w": base["mean_wait_min"] - cf["mean_wait_min"],
        "w_rush_base": base["rush_wait_min"],
        "w_rush_cf": cf["rush_wait_min"],
        "delta_w_rush": (
            float(base["rush_wait_min"] - cf["rush_wait_min"])
            if np.isfinite(base["rush_wait_min"]) and np.isfinite(cf["rush_wait_min"])
            else float("nan")
        ),
        "service_min_base": service_min,
        "service_min_cf": s_cf,
        "speedup_frac": frac,
        "base": base,
        "cf": cf,
    }


def compare_fluid_vs_heuristic(
    rx: pd.DataFrame,
    staff: pd.DataFrame,
    queue_model: dict,
    fluid_cal: dict,
    sample_days: int = 80,
    seed: int = 42,
    system_disp_saved: float = 0.05,
) -> dict:
    """Daily: fluid-predicted W vs observed; and ΔW under speedup (fluid vs heuristic propagate)."""
    rng = np.random.default_rng(seed)
    rx = rx.copy()
    rx["日期"] = pd.to_datetime(rx["日期"]).dt.date
    days = sorted(rx["日期"].unique())
    if len(days) > sample_days:
        days = list(rng.choice(days, size=sample_days, replace=False))

    staff = staff.copy()
    staff["日期"] = pd.to_datetime(staff["日期"]).dt.date
    staff_map = staff.set_index("日期")
    s = float(fluid_cal["service_min"])
    d_mean = float(queue_model.get("mean_disp_min", 0.65))

    rows = []
    for d in days:
        rx_d = rx[rx["日期"] == d]
        if len(rx_d) < 30:
            continue
        w_obs = rx_d["叫号等待_分钟"].mean()
        if not np.isfinite(w_obs):
            continue
        cap = _day_capacity(staff_map.loc[d] if d in staff_map.index else None)
        arr = hourly_arrivals(rx_d)
        fluid = simulate_fluid_day(arr, s, cap)
        # Heuristic: use daily mean observed W as base, propagate
        peak = 1  # coarse daily comparison uses rush factor
        load0 = 2
        w_heur_cf = propagate_call_wait_minutes(
            float(w_obs), system_disp_saved, peak, load0, queue_model, capacity=cap
        )
        fluid_cf = fluid_wait_under_speedup(arr, s, cap, system_disp_saved, d_mean)
        rows.append(
            {
                "date": str(d),
                "n_rx": int(len(rx_d)),
                "w_obs": float(w_obs),
                "w_fluid": float(fluid["mean_wait_min"]),
                "w_fluid_rush": float(fluid["rush_wait_min"]) if np.isfinite(fluid["rush_wait_min"]) else None,
                "abs_err_fluid": abs(float(fluid["mean_wait_min"]) - float(w_obs)),
                "delta_w_fluid": float(fluid_cf["delta_w"]),
                "delta_w_fluid_rush": float(fluid_cf["delta_w_rush"])
                if np.isfinite(fluid_cf["delta_w_rush"])
                else None,
                "delta_w_heur": float(w_obs - w_heur_cf),
                "max_queue_fluid": float(fluid["max_queue"]),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return {"n_days": 0}

    summary = {
        "n_days": int(len(df)),
        "system_disp_saved_min": system_disp_saved,
        "fluid_cal": fluid_cal,
        "fit": {
            "mae_fluid_vs_obs": round(float(df["abs_err_fluid"].mean()), 4),
            "mean_w_obs": round(float(df["w_obs"].mean()), 4),
            "mean_w_fluid": round(float(df["w_fluid"].mean()), 4),
            "corr_fluid_obs": round(float(df["w_obs"].corr(df["w_fluid"])), 4),
        },
        "counterfactual_delta_w": {
            "mean_delta_fluid": round(float(df["delta_w_fluid"].mean()), 4),
            "mean_delta_fluid_rush": round(
                float(df["delta_w_fluid_rush"].dropna().mean()), 4
            )
            if df["delta_w_fluid_rush"].notna().any()
            else None,
            "mean_delta_heur": round(float(df["delta_w_heur"].mean()), 4),
            "ratio_fluid_over_heur": round(
                float(df["delta_w_fluid"].mean() / max(df["delta_w_heur"].mean(), 1e-6)),
                4,
            ),
        },
        "note": (
            "Fluid may not beat heuristic on MAE; value is structural CF dynamics "
            "(buildup/drain). Compare delta_w under same system_disp_saved."
        ),
    }
    return {"summary": summary, "daily": rows}


# ---------------------------------------------------------------------------
# S8.1: prescription-level arrivals + time-varying s
# ---------------------------------------------------------------------------


def _arrival_minutes(rx_day: pd.DataFrame) -> np.ndarray:
    """Check-in time → minutes from midnight. Fall back to hour midpoint if 报到时间_dt missing."""
    if "报到时间_dt" in rx_day.columns and rx_day["报到时间_dt"].notna().any():
        t = pd.to_datetime(rx_day["报到时间_dt"], errors="coerce")
        mins = (t.dt.hour * 60 + t.dt.minute + t.dt.second / 60.0).to_numpy(dtype=float)
        # Rows without timestamp fall back to hour midpoint
        bad = ~np.isfinite(mins)
        if bad.any() and "报到小时" in rx_day.columns:
            h = pd.to_numeric(rx_day["报到小时"], errors="coerce").to_numpy(dtype=float)
            mins[bad] = h[bad] * 60.0 + 30.0
        return mins
    h = pd.to_numeric(rx_day["报到小时"], errors="coerce").fillna(12).to_numpy(dtype=float)
    return h * 60.0 + 30.0


def _service_at_minute(minute: float, s_other: float, s_rush: float) -> float:
    hour = int(minute // 60) % 24
    return float(s_rush if hour in RUSH_HOURS else s_other)


def simulate_fluid_rx(
    arrival_mins: np.ndarray,
    capacity: float,
    s_other: float,
    s_rush: float | None = None,
) -> dict:
    """
    Prescription-level fluid: sort by actual check-in times; drain at μc between arrivals,
    at arrival W=Q/(μc), Q+=1. When s_rush is None, use s_other for full day.
    """
    s_r = float(s_other if s_rush is None else s_rush)
    s_o = max(float(s_other), 0.05)
    s_r = max(s_r, 0.05)
    c = max(float(capacity), 1.0)

    order = np.argsort(arrival_mins)
    arr = arrival_mins[order]
    n = len(arr)
    if n == 0:
        return {
            "mean_wait_min": 0.0,
            "rush_wait_min": float("nan"),
            "waits": np.array([]),
            "order": order,
            "max_queue": 0.0,
        }

    waits = np.zeros(n, dtype=float)
    q = 0.0
    t_prev = float(arr[0])
    max_q = 0.0

    for i, t in enumerate(arr):
        t = float(t)
        dt = max(0.0, t - t_prev)
        # Segment drain within gap (split by minute at rush boundary to limit μ jump error)
        if dt > 0 and q > 0:
            rem = dt
            cursor = t_prev
            while rem > 1e-9 and q > 0:
                step = min(rem, 1.0)
                s = _service_at_minute(cursor, s_o, s_r)
                rate = (1.0 / s) * c
                q = max(0.0, q - rate * step)
                cursor += step
                rem -= step
        s_now = _service_at_minute(t, s_o, s_r)
        rate_now = (1.0 / s_now) * c
        waits[i] = q / rate_now
        q += 1.0
        max_q = max(max_q, q)
        t_prev = t

    # Map waits back to original row order (for alignment with observations)
    waits_orig = np.empty(n, dtype=float)
    waits_orig[order] = waits

    hours = ((arr // 60).astype(int) % 24)
    rush_mask = np.isin(hours, list(RUSH_HOURS))
    rush_w = float(np.mean(waits[rush_mask])) if rush_mask.any() else float("nan")

    return {
        "mean_wait_min": float(np.mean(waits)),
        "rush_wait_min": rush_w,
        "waits": waits_orig,
        "order": order,
        "max_queue": float(max_q),
        "s_other": s_o,
        "s_rush": s_r,
        "capacity": c,
    }


def simulate_fluid_day_segmented(
    arrivals: dict[int, int],
    s_other: float,
    s_rush: float,
    capacity: float,
    dt: float = DT_MIN,
) -> dict:
    """Hourly aggregated fluid with different s for rush vs other."""
    s_o = max(float(s_other), 0.05)
    s_r = max(float(s_rush), 0.05)
    c = max(float(capacity), 1.0)

    q = 0.0
    waits_by_hour: dict[int, list[float]] = {h: [] for h in HOURS}
    q_path = []

    for hour in HOURS:
        n = int(arrivals.get(hour, 0))
        lam = n / 60.0
        s = s_r if hour in RUSH_HOURS else s_o
        rate_out = (1.0 / s) * c
        for _ in range(60):
            q = max(0.0, q + lam * dt - rate_out * dt)
            waits_by_hour[hour].append(q / rate_out)
            q_path.append(q)

    hour_mean = {h: float(np.mean(ws)) if ws else 0.0 for h, ws in waits_by_hour.items()}
    total_n = sum(arrivals.get(h, 0) for h in HOURS)
    if total_n > 0:
        mean_w = sum(arrivals.get(h, 0) * hour_mean[h] for h in HOURS) / total_n
    else:
        mean_w = float(np.mean(list(hour_mean.values()))) if hour_mean else 0.0
    rush_n = arrivals.get(9, 0) + arrivals.get(10, 0)
    rush_w = (
        (arrivals.get(9, 0) * hour_mean[9] + arrivals.get(10, 0) * hour_mean[10]) / rush_n
        if rush_n > 0
        else float("nan")
    )
    return {
        "mean_wait_min": mean_w,
        "rush_wait_min": rush_w,
        "hour_wait": hour_mean,
        "max_queue": float(max(q_path) if q_path else 0.0),
    }


def calibrate_service_refined(
    rx: pd.DataFrame,
    staff: pd.DataFrame,
    sample_days: int = 30,
    seed: int = 42,
) -> dict:
    """
    Calibrate four parameter sets (daily mean W MAE):
      hourly_single / hourly_seg / rx_single / rx_seg
    """
    rng = np.random.default_rng(seed)
    rx = rx.copy()
    rx["日期"] = pd.to_datetime(rx["日期"]).dt.date
    days = sorted(rx["日期"].unique())
    if len(days) > sample_days:
        days = list(rng.choice(days, size=sample_days, replace=False))

    staff = staff.copy()
    staff["日期"] = pd.to_datetime(staff["日期"]).dt.date
    staff_map = staff.set_index("日期")

    day_packs = []
    for d in days:
        rx_d = rx[rx["日期"] == d]
        if len(rx_d) < 30 or rx_d["叫号等待_分钟"].notna().sum() < 20:
            continue
        cap = _day_capacity(staff_map.loc[d] if d in staff_map.index else None)
        day_packs.append(
            {
                "arr_h": hourly_arrivals(rx_d),
                "arr_m": _arrival_minutes(rx_d),
                "w_obs": float(rx_d["叫号等待_分钟"].mean()),
                "w_obs_rx": pd.to_numeric(rx_d["叫号等待_分钟"], errors="coerce").to_numpy(
                    dtype=float
                ),
                "cap": cap,
            }
        )
    if not day_packs:
        raise RuntimeError("no calibration days")

    single_grid = np.linspace(0.4, 1.6, 13)
    # Time-varying: s_other × rush_mult (slower rush → mult>1)
    other_grid = np.linspace(0.4, 1.4, 6)
    mult_grid = np.array([0.7, 0.85, 1.0, 1.15, 1.3, 1.5, 1.8])

    def mae_hourly_single(s: float) -> float:
        errs = [
            abs(
                simulate_fluid_day(p["arr_h"], s, p["cap"])["mean_wait_min"] - p["w_obs"]
            )
            for p in day_packs
        ]
        return float(np.mean(errs))

    def mae_hourly_seg(s_o: float, s_r: float) -> float:
        errs = [
            abs(
                simulate_fluid_day_segmented(p["arr_h"], s_o, s_r, p["cap"])["mean_wait_min"]
                - p["w_obs"]
            )
            for p in day_packs
        ]
        return float(np.mean(errs))

    def mae_rx(s_o: float, s_r: float | None) -> tuple[float, float]:
        """Return (daily mean W MAE, prescription-level W MAE)."""
        day_errs, rx_errs = [], []
        for p in day_packs:
            sim = simulate_fluid_rx(p["arr_m"], p["cap"], s_o, s_r)
            day_errs.append(abs(sim["mean_wait_min"] - p["w_obs"]))
            obs = p["w_obs_rx"]
            pred = sim["waits"]
            m = np.isfinite(obs) & np.isfinite(pred)
            if m.any():
                rx_errs.append(float(np.mean(np.abs(pred[m] - obs[m]))))
        return float(np.mean(day_errs)), float(np.mean(rx_errs)) if rx_errs else float("nan")

    best_hs, best_hs_mae = float(single_grid[0]), 1e9
    for s in single_grid:
        mae = mae_hourly_single(float(s))
        if mae < best_hs_mae:
            best_hs_mae, best_hs = mae, float(s)

    best_hseg, best_hseg_mae = (0.8, 0.8), 1e9
    for so in other_grid:
        for m in mult_grid:
            sr = float(so) * float(m)
            mae = mae_hourly_seg(float(so), sr)
            if mae < best_hseg_mae:
                best_hseg_mae, best_hseg = mae, (float(so), sr)

    best_rs, best_rs_day, best_rs_rx = float(single_grid[0]), 1e9, float("nan")
    for s in single_grid:
        d_mae, r_mae = mae_rx(float(s), None)
        if d_mae < best_rs_day:
            best_rs_day, best_rs_rx, best_rs = d_mae, r_mae, float(s)

    best_rseg, best_rseg_day, best_rseg_rx = (0.8, 0.8), 1e9, float("nan")
    for so in other_grid:
        for m in mult_grid:
            sr = float(so) * float(m)
            d_mae, r_mae = mae_rx(float(so), sr)
            if d_mae < best_rseg_day:
                best_rseg_day, best_rseg_rx, best_rseg = d_mae, r_mae, (float(so), sr)

    return {
        "n_cal_days": len(day_packs),
        "hourly_single": {
            "service_min": round(best_hs, 4),
            "mae_day_mean_w": round(best_hs_mae, 4),
        },
        "hourly_seg": {
            "s_other": round(best_hseg[0], 4),
            "s_rush": round(best_hseg[1], 4),
            "mae_day_mean_w": round(best_hseg_mae, 4),
        },
        "rx_single": {
            "service_min": round(best_rs, 4),
            "mae_day_mean_w": round(best_rs_day, 4),
            "mae_rx_w": round(best_rs_rx, 4) if np.isfinite(best_rs_rx) else None,
        },
        "rx_seg": {
            "s_other": round(best_rseg[0], 4),
            "s_rush": round(best_rseg[1], 4),
            "mae_day_mean_w": round(best_rseg_day, 4),
            "mae_rx_w": round(best_rseg_rx, 4) if np.isfinite(best_rseg_rx) else None,
        },
        "note": "calibrate on day-mean call-wait MAE; rx_* also report rx-level MAE",
    }


def _speedup_s(s: float, saved: float, mean_disp: float) -> float:
    frac = float(np.clip(saved / max(mean_disp, 0.01), -0.5, 0.85))
    return max(0.05, float(s) * (1.0 - frac))


def compare_refined_sensitivity(
    rx: pd.DataFrame,
    staff: pd.DataFrame,
    queue_model: dict,
    cal: dict,
    sample_days: int = 80,
    seed: int = 42,
    system_disp_saved: float = 0.05,
) -> dict:
    """Four fluid variants + heuristic: fit and CF ΔW comparison."""
    rng = np.random.default_rng(seed)
    rx = rx.copy()
    rx["日期"] = pd.to_datetime(rx["日期"]).dt.date
    days = sorted(rx["日期"].unique())
    if len(days) > sample_days:
        days = list(rng.choice(days, size=sample_days, replace=False))

    staff = staff.copy()
    staff["日期"] = pd.to_datetime(staff["日期"]).dt.date
    staff_map = staff.set_index("日期")
    d_mean = float(queue_model.get("mean_disp_min", 0.65))

    hs = float(cal["hourly_single"]["service_min"])
    hso, hsr = float(cal["hourly_seg"]["s_other"]), float(cal["hourly_seg"]["s_rush"])
    rs = float(cal["rx_single"]["service_min"])
    rso, rsr = float(cal["rx_seg"]["s_other"]), float(cal["rx_seg"]["s_rush"])

    rows = []
    for d in days:
        rx_d = rx[rx["日期"] == d]
        if len(rx_d) < 30:
            continue
        w_obs = float(rx_d["叫号等待_分钟"].mean())
        if not np.isfinite(w_obs):
            continue
        cap = _day_capacity(staff_map.loc[d] if d in staff_map.index else None)
        arr_h = hourly_arrivals(rx_d)
        arr_m = _arrival_minutes(rx_d)
        obs_rx = pd.to_numeric(rx_d["叫号等待_分钟"], errors="coerce").to_numpy(dtype=float)

        # base
        f_hs = simulate_fluid_day(arr_h, hs, cap)
        f_hseg = simulate_fluid_day_segmented(arr_h, hso, hsr, cap)
        f_rs = simulate_fluid_rx(arr_m, cap, rs, None)
        f_rseg = simulate_fluid_rx(arr_m, cap, rso, rsr)

        # CF
        f_hs_cf = simulate_fluid_day(arr_h, _speedup_s(hs, system_disp_saved, d_mean), cap)
        f_hseg_cf = simulate_fluid_day_segmented(
            arr_h,
            _speedup_s(hso, system_disp_saved, d_mean),
            _speedup_s(hsr, system_disp_saved, d_mean),
            cap,
        )
        f_rs_cf = simulate_fluid_rx(
            arr_m, cap, _speedup_s(rs, system_disp_saved, d_mean), None
        )
        f_rseg_cf = simulate_fluid_rx(
            arr_m,
            cap,
            _speedup_s(rso, system_disp_saved, d_mean),
            _speedup_s(rsr, system_disp_saved, d_mean),
        )

        w_heur_cf = propagate_call_wait_minutes(
            w_obs, system_disp_saved, 1, 2, queue_model, capacity=cap
        )

        def rx_mae(pred: np.ndarray) -> float:
            m = np.isfinite(obs_rx) & np.isfinite(pred)
            return float(np.mean(np.abs(pred[m] - obs_rx[m]))) if m.any() else float("nan")

        rows.append(
            {
                "date": str(d),
                "n_rx": int(len(rx_d)),
                "w_obs": w_obs,
                "w_hourly_single": float(f_hs["mean_wait_min"]),
                "w_hourly_seg": float(f_hseg["mean_wait_min"]),
                "w_rx_single": float(f_rs["mean_wait_min"]),
                "w_rx_seg": float(f_rseg["mean_wait_min"]),
                "mae_rx_single": rx_mae(f_rs["waits"]),
                "mae_rx_seg": rx_mae(f_rseg["waits"]),
                "delta_hourly_single": float(
                    f_hs["mean_wait_min"] - f_hs_cf["mean_wait_min"]
                ),
                "delta_hourly_seg": float(
                    f_hseg["mean_wait_min"] - f_hseg_cf["mean_wait_min"]
                ),
                "delta_rx_single": float(f_rs["mean_wait_min"] - f_rs_cf["mean_wait_min"]),
                "delta_rx_seg": float(f_rseg["mean_wait_min"] - f_rseg_cf["mean_wait_min"]),
                "delta_heur": float(w_obs - w_heur_cf),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return {"n_days": 0}

    def fit_block(col: str) -> dict:
        return {
            "mae_day_mean_w": round(float((df[col] - df["w_obs"]).abs().mean()), 4),
            "mean_w_pred": round(float(df[col].mean()), 4),
            "corr_obs": round(float(df["w_obs"].corr(df[col])), 4),
        }

    variants = {
        "hourly_single": fit_block("w_hourly_single"),
        "hourly_seg": fit_block("w_hourly_seg"),
        "rx_single": {
            **fit_block("w_rx_single"),
            "mae_rx_w": round(float(df["mae_rx_single"].mean()), 4),
        },
        "rx_seg": {
            **fit_block("w_rx_seg"),
            "mae_rx_w": round(float(df["mae_rx_seg"].mean()), 4),
        },
    }
    cf = {
        "mean_delta_hourly_single": round(float(df["delta_hourly_single"].mean()), 4),
        "mean_delta_hourly_seg": round(float(df["delta_hourly_seg"].mean()), 4),
        "mean_delta_rx_single": round(float(df["delta_rx_single"].mean()), 4),
        "mean_delta_rx_seg": round(float(df["delta_rx_seg"].mean()), 4),
        "mean_delta_heur": round(float(df["delta_heur"].mean()), 4),
    }
    # Relative to heuristic
    h = max(cf["mean_delta_heur"], 1e-6)
    cf["ratio_vs_heur"] = {
        k.replace("mean_delta_", ""): round(cf[k] / h, 4)
        for k in cf
        if k.startswith("mean_delta_") and k != "mean_delta_heur"
    }

    summary = {
        "n_days": int(len(df)),
        "system_disp_saved_min": system_disp_saved,
        "mean_w_obs": round(float(df["w_obs"].mean()), 4),
        "calibration": cal,
        "fit": variants,
        "counterfactual_delta_w": cf,
        "note": (
            "S8.1 refined sensitivity: rx-level arrivals + segmented s. "
            "Does not replace heuristic propagate on main path."
        ),
    }
    return {"summary": summary, "daily": rows}


def main() -> None:
    import argparse

    from phase2.continuous_reward import build_feature_frame, load_queue_model
    from phase2.ns_mechanisms import annotate_regimes

    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-days", type=int, default=80)
    parser.add_argument("--saved", type=float, default=0.05, help="counterfactual system dispensing savings (minutes)")
    parser.add_argument(
        "--refined",
        action="store_true",
        help="S8.1: prescription-level arrivals + time-varying s, four sensitivity variants",
    )
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("[1] Loading data...")
    rx = pd.read_parquet(DATA_DIR / "rx_level.parquet")
    disc = pd.read_parquet(PHASE2_DIR / "rx_discretized.parquet")
    rx = rx.merge(disc[["处方编号", "Peak0", "Load0"]], on="处方编号", how="left")
    staff = load_daily_staff()
    rx = build_feature_frame(rx, staff)
    rx = annotate_regimes(rx, rx["日期"])
    qm = load_queue_model()

    if args.refined:
        print("[2] Calibrating four fluid parameter sets (hourly/rx × single/segmented)...")
        cal = calibrate_service_refined(rx, staff, sample_days=min(30, args.sample_days))
        print("  cal:", {k: cal[k] for k in ("hourly_single", "hourly_seg", "rx_single", "rx_seg")})
        print("[3] Refined sensitivity comparison...")
        cmp_ = compare_refined_sensitivity(
            rx, staff, qm, cal, sample_days=args.sample_days, system_disp_saved=args.saved
        )
        print("  fit:", cmp_["summary"]["fit"])
        print("  CF ΔW:", cmp_["summary"]["counterfactual_delta_w"])
        with (OUT_DIR / "fluid_refined_calibration.json").open("w", encoding="utf-8") as f:
            json.dump(cal, f, ensure_ascii=False, indent=2)
        with (OUT_DIR / "fluid_refined_vs_heuristic.json").open("w", encoding="utf-8") as f:
            json.dump(cmp_["summary"], f, ensure_ascii=False, indent=2)
        pd.DataFrame(cmp_["daily"]).to_csv(OUT_DIR / "fluid_refined_daily.csv", index=False)
        print("done →", OUT_DIR)
        return

    print("[2] Calibrating fluid service time s...")
    cal = calibrate_service_minutes(rx, staff, sample_days=min(40, args.sample_days))
    print(f"  s={cal['service_min']} min  calibrate_MAE={cal['calibrate_mae_day_mean_w']}")

    print("[3] Fluid vs heuristic comparison...")
    cmp_ = compare_fluid_vs_heuristic(
        rx, staff, qm, cal, sample_days=args.sample_days, system_disp_saved=args.saved
    )
    print("  fit:", cmp_["summary"]["fit"])
    print("  CF ΔW:", cmp_["summary"]["counterfactual_delta_w"])

    with (OUT_DIR / "fluid_calibration.json").open("w", encoding="utf-8") as f:
        json.dump(cal, f, ensure_ascii=False, indent=2)
    with (OUT_DIR / "fluid_vs_heuristic.json").open("w", encoding="utf-8") as f:
        json.dump(cmp_["summary"], f, ensure_ascii=False, indent=2)
    pd.DataFrame(cmp_["daily"]).to_csv(OUT_DIR / "fluid_daily.csv", index=False)
    print("done →", OUT_DIR)


if __name__ == "__main__":
    main()
