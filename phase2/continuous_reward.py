"""S4: Continuous wait-time-minute reward (no hand-crafted bonus).

Decisions:
  - S4.4: features include counterfactual W~ (train=observed call wait; do(a)=propagate_call_wait)
  - Primary model Ridge; HGB control
  - Primary report R = -Yhat_min; affine for plotting only
  - Full daily prescriptions (optional subset with swapped drugs)
  - Gate (S4.5): cross-day μ difference one-sided t-test p<0.05 and correct direction (μ_swap40 < μ_null)
    MAE for prediction accuracy only, not a bandit feasibility gate
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from phase2.config import DATA_DIR
from phase2.intervention import (
    DEFAULT_MANUAL_SEC,
    MACHINE_SEC,
    LayoutScenario,
    build_machine_swap,
    build_shelf_rearrange,
    counts_to_m0,
    n_items_to_n0,
)
from phase2.ns_mechanisms import annotate_regimes, cal_block
from phase2.queue_propagation import attach_staff, propagate_call_wait_minutes
from phase2.staff_schedule import load_daily_staff

PHASE2_DIR = DATA_DIR / "phase2"
OUT_DIR = PHASE2_DIR / "continuous_reward"
QUEUE_MODEL_PATH = PHASE2_DIR / "queue_model.json"

# S4.4: W~ uses column 叫号等待_分钟 (train on observed; intervene via queue propagation)
FEATURE_COLS = [
    "预估配药_分钟",
    "机器品项占比",
    "品项数",
    "Peak0",
    "Load0",
    "调配人数",
    "叫号等待_分钟",
]
TARGET_COL = "候药时长_分钟"
W_FEATURE = "叫号等待_分钟"
WINSOR_P = 0.99
USE_COUNTERFACTUAL_W = True


@dataclass
class WaitRegressorBundle:
    model_name: str
    pipeline: Pipeline
    feature_cols: list[str]
    y_cap: float
    q01: float
    q99: float
    metrics: dict


def _is_peak(hour: float) -> int:
    h = int(hour) if pd.notna(hour) else -1
    return 1 if (9 <= h <= 11) or (14 <= h <= 16) else 0


def load_queue_model(path: Path = QUEUE_MODEL_PATH) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def build_feature_frame(rx: pd.DataFrame, staff: pd.DataFrame | None = None) -> pd.DataFrame:
    out = rx.copy()
    if "Peak0" not in out.columns:
        out["Peak0"] = out["报到小时"].map(_is_peak).astype(int)
    if "Load0" not in out.columns:
        # Consistent with discretization: daily prescription count quartiles
        q = out["当日处方数"].quantile([0.25, 0.5, 0.75]).tolist()
        out["Load0"] = pd.cut(
            out["当日处方数"],
            bins=[-np.inf, q[0], q[1], q[2], np.inf],
            labels=[0, 1, 2, 3],
        ).astype(int)
    if staff is not None:
        need = [c for c in ("调配人数", "产能代理", "有效窗口") if c not in out.columns]
        if need:
            out = attach_staff(out, staff)
    out["调配人数"] = out["调配人数"].fillna(out["调配人数"].median())
    if W_FEATURE in out.columns:
        out[W_FEATURE] = pd.to_numeric(out[W_FEATURE], errors="coerce")
        med_w = float(out[W_FEATURE].median()) if out[W_FEATURE].notna().any() else 3.5
        out[W_FEATURE] = out[W_FEATURE].fillna(med_w)
    return out


def winsorize_y(y: pd.Series, p: float = WINSOR_P) -> tuple[pd.Series, float]:
    cap = float(y.quantile(p))
    return y.clip(upper=cap), cap


def time_split_mask(dates: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Train: CalBlock ∈ {2024H1,2024H2,2025H1}; test: 2025H2."""
    blocks = pd.to_datetime(dates).map(cal_block)
    test = blocks == "2025H2"
    train = ~test
    return train.to_numpy(), test.to_numpy()


def train_wait_regressors(
    rx: pd.DataFrame,
    staff: pd.DataFrame | None = None,
) -> dict[str, WaitRegressorBundle]:
    df = build_feature_frame(rx, staff)
    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL, "日期"]).copy()
    y_raw = df[TARGET_COL].astype(float)
    y, y_cap = winsorize_y(y_raw)
    X = df[FEATURE_COLS].astype(float)
    train_m, test_m = time_split_mask(df["日期"])

    bundles: dict[str, WaitRegressorBundle] = {}
    specs = {
        "ridge": Pipeline(
            [("scaler", StandardScaler()), ("model", Ridge(alpha=1.0, random_state=42))]
        ),
        "hgb": HistGradientBoostingRegressor(
            max_depth=6,
            learning_rate=0.08,
            max_iter=200,
            random_state=42,
        ),
        # Lightweight MLP: captures peak/load nonlinearity; same features, no extra dims
        "mlp": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    MLPRegressor(
                        hidden_layer_sizes=(64, 32),
                        activation="relu",
                        solver="adam",
                        alpha=1e-4,
                        learning_rate_init=1e-3,
                        max_iter=200,
                        early_stopping=True,
                        validation_fraction=0.1,
                        n_iter_no_change=15,
                        random_state=42,
                    ),
                ),
            ]
        ),
    }

    for name, est in specs.items():
        if name == "hgb":
            pipe = Pipeline([("model", est)])
        else:
            pipe = est
        pipe.fit(X.loc[train_m], y.loc[train_m])
        pred_te = pipe.predict(X.loc[test_m])
        y_te = y.loc[test_m].to_numpy()
        metrics = {
            "n_train": int(train_m.sum()),
            "n_test": int(test_m.sum()),
            "y_cap_p99": round(y_cap, 4),
            "mae": round(float(mean_absolute_error(y_te, pred_te)), 4),
            "rmse": round(float(np.sqrt(mean_squared_error(y_te, pred_te))), 4),
            "r2": round(float(r2_score(y_te, pred_te)), 4),
        }
        # Stratified MAE by morning rush
        if "报到小时" in df.columns:
            rush = df.loc[test_m, "报到小时"].between(9, 10) & (
                pd.to_datetime(df.loc[test_m, "日期"]).dt.weekday < 5
            )
            other = ~rush
            if rush.any():
                metrics["mae_morning_rush"] = round(
                    float(mean_absolute_error(y_te[rush.to_numpy()], pred_te[rush.to_numpy()])),
                    4,
                )
            if other.any():
                metrics["mae_other"] = round(
                    float(mean_absolute_error(y_te[other.to_numpy()], pred_te[other.to_numpy()])),
                    4,
                )
        q01, q99 = float(y.loc[train_m].quantile(0.01)), float(y.loc[train_m].quantile(0.99))
        bundles[name] = WaitRegressorBundle(
            model_name=name,
            pipeline=pipe,
            feature_cols=list(FEATURE_COLS),
            y_cap=y_cap,
            q01=q01,
            q99=q99,
            metrics=metrics,
        )
        print(
            f"  [{name}] hold-out MAE={metrics['mae']} RMSE={metrics['rmse']} R2={metrics['r2']}"
            + (
                f" rush_MAE={metrics.get('mae_morning_rush')}"
                if "mae_morning_rush" in metrics
                else ""
            )
        )
    return bundles


def save_bundles(bundles: dict[str, WaitRegressorBundle], out_dir: Path = OUT_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = {}
    for name, b in bundles.items():
        joblib.dump(
            {
                "pipeline": b.pipeline,
                "feature_cols": b.feature_cols,
                "y_cap": b.y_cap,
                "q01": b.q01,
                "q99": b.q99,
                "metrics": b.metrics,
                "model_name": b.model_name,
            },
            out_dir / f"wait_regressor_{name}.joblib",
        )
        metrics[name] = b.metrics
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "features": FEATURE_COLS,
                "target": TARGET_COL,
                "winsor_p": WINSOR_P,
                "split": "train=2024H1+2024H2+2025H1, test=2025H2",
                "use_counterfactual_W": USE_COUNTERFACTUAL_W,
                "W_feature": W_FEATURE,
                "W_train": "observed 叫号等待_分钟",
                "W_do": "propagate_call_wait_minutes(W_obs, ΔD_sys, Peak, Load, queue_model, capacity)",
                "models": metrics,
                "primary": "ridge",
                "s4_stage": "S4.4+mlp",
                "mlp_note": "MLP(64,32) same features; sensitivity vs ridge/hgb",
            },
            f,
            ensure_ascii=False,
            indent=2,
        )


def load_bundle(name: str = "ridge", out_dir: Path = OUT_DIR) -> WaitRegressorBundle:
    obj = joblib.load(out_dir / f"wait_regressor_{name}.joblib")
    return WaitRegressorBundle(
        model_name=obj["model_name"],
        pipeline=obj["pipeline"],
        feature_cols=obj["feature_cols"],
        y_cap=obj["y_cap"],
        q01=obj["q01"],
        q99=obj["q99"],
        metrics=obj["metrics"],
    )


def predict_minutes(bundle: WaitRegressorBundle, X: pd.DataFrame) -> np.ndarray:
    return np.asarray(bundle.pipeline.predict(X[bundle.feature_cols].astype(float)), dtype=float)


def affine_reward(yhat: float, bundle: WaitRegressorBundle) -> float:
    """Affine [0,1] mapping for plotting only."""
    denom = max(bundle.q99 - bundle.q01, 1e-6)
    return float(np.clip(1.0 - (yhat - bundle.q01) / denom, 0.0, 1.5))


# ----- Vectorized do(a) recomputation -----


def apply_scenario_to_items(items: pd.DataFrame, scenario: LayoutScenario | None) -> pd.DataFrame:
    out = items.copy()
    if scenario is None or scenario.kind == "baseline":
        return out
    is_m = out["是否机器_最终"].astype(bool).to_numpy().copy()
    sec = out["调配秒数_最终"].astype(float).to_numpy().copy()
    did = out["drugid"].astype(str)

    if scenario.kind == "machine_swap":
        to_m = did.isin(scenario.to_machine).to_numpy() & (~is_m)
        to_a = did.isin(scenario.to_manual).to_numpy() & is_m
        is_m[to_m] = True
        sec[to_m] = MACHINE_SEC
        is_m[to_a] = False
        sec[to_a] = DEFAULT_MANUAL_SEC
    elif scenario.kind == "service_time_map":
        # Generic: arbitrary per-unit service-second override (cross-scenario interface)
        for i, d in enumerate(did):
            if d in scenario.new_seconds:
                sec[i] = float(scenario.new_seconds[d])
    elif scenario.kind == "shelf_rearrange":
        for i, d in enumerate(did):
            if (not is_m[i]) and d in scenario.new_seconds:
                sec[i] = float(scenario.new_seconds[d])

    out["是否机器_最终"] = is_m
    out["调配秒数_最终"] = sec
    return out


def aggregate_rx_features(items: pd.DataFrame) -> pd.DataFrame:
    g = items.groupby("处方编号", sort=False).agg(
        预估配药_分钟=("调配秒数_最终", lambda s: float(s.sum()) / 60.0),
        机器品项数=("是否机器_最终", "sum"),
        品项数=("drugid", "count"),
    )
    g["机器品项占比"] = g["机器品项数"] / g["品项数"].clip(lower=1)
    g["N0"] = g["品项数"].map(n_items_to_n0)
    g["M0"] = [
        counts_to_m0(int(m), int(n)) for m, n in zip(g["机器品项数"], g["品项数"])
    ]
    return g.reset_index()


def arm_to_scenario(arm_id: str, items_all: pd.DataFrame) -> LayoutScenario | None:
    base = arm_id.split("@")[0]
    if base == "null" or base == "load_do":
        return None
    if base == "swap20":
        return build_machine_swap(
            items_all, 20, enforce_machine_eligibility=True, prefer_remove_ineligible=False
        )
    if base == "swap40":
        return build_machine_swap(
            items_all, 40, enforce_machine_eligibility=True, prefer_remove_ineligible=False
        )
    if base == "shelf50":
        return build_shelf_rearrange(items_all, 50)
    return None


def parse_arm_start_t(arm_id: str) -> int:
    """POMIS+ timing: swap40 → 0 (full day); swap40@1 → 1 (MorningRush prescriptions only)."""
    if "@" not in arm_id:
        return 0
    try:
        return int(arm_id.split("@", 1)[1])
    except ValueError:
        return 0


def ensure_morning_rush(rx: pd.DataFrame) -> pd.DataFrame:
    """Ensure MorningRush exists (weekday 9–10), consistent with ns_mechanisms.morning_rush."""
    out = rx.copy()
    if "MorningRush" in out.columns:
        return out
    if "报到小时" not in out.columns or "日期" not in out.columns:
        out["MorningRush"] = 0
        return out
    from phase2.ns_mechanisms import morning_rush

    dt = pd.to_datetime(out["日期"])
    out["MorningRush"] = [
        morning_rush(h, int(w)) for h, w in zip(out["报到小时"], dt.dt.weekday)
    ]
    return out


def active_rx_ids_for_timing(rx_day: pd.DataFrame, start_t: int) -> set | None:
    """
    Return prescription IDs that receive layout intervention under this timing.
    start_t=0: full day (None = no restriction).
    start_t=1: MorningRush==1 only (weekday 9–10) — business definition of @1.
    start_t>=2: afternoon rest only (weekday hour>=11, or all weekend).
    """
    if start_t <= 0:
        return None
    rx = ensure_morning_rush(rx_day)
    if start_t == 1:
        active = rx.loc[rx["MorningRush"].astype(int) == 1, "处方编号"]
        return set(active.unique())
    hour = pd.to_numeric(rx["报到小时"], errors="coerce").fillna(-1).astype(int)
    wd = pd.to_datetime(rx["日期"]).dt.weekday
    rest = (wd >= 5) | (hour >= 11)
    return set(rx.loc[rest, "处方编号"].unique())


def _scenario_drug_ids(scenario: LayoutScenario | None) -> set[str]:
    if scenario is None or scenario.kind == "baseline":
        return set()
    if scenario.kind == "machine_swap":
        return set(scenario.to_machine) | set(scenario.to_manual)
    if scenario.kind in ("shelf_rearrange", "service_time_map"):
        return set(scenario.new_seconds.keys())
    return set()


def _propagate_call_wait_vectorized(
    w_base: np.ndarray,
    system_disp_saved_min: float,
    peak: np.ndarray,
    load0: np.ndarray,
    model: dict,
    capacity: np.ndarray | None = None,
) -> np.ndarray:
    """Vectorized version of the same formula as propagate_call_wait_minutes."""
    d_mean = model["mean_disp_min"]
    speedup_ratio = system_disp_saved_min / max(d_mean, 0.01)
    load_factor = 1.0 + model["load_amplify"] * (load0.astype(float) / 3.0)
    peak_factor = 1.0 + model["peak_amplify"] * peak.astype(float)
    cap_mean = float(model.get("mean_capacity", 8.0))
    if capacity is None:
        cap = np.full(len(w_base), cap_mean, dtype=float)
    else:
        cap = np.asarray(capacity, dtype=float)
        cap = np.where(np.isfinite(cap) & (cap > 0), cap, cap_mean)
    staff_factor = 1.0 + float(model.get("staff_amplify", 1.0)) * np.maximum(
        0.0, (cap_mean / cap) - 1.0
    )
    reduction_frac = np.minimum(
        0.85,
        speedup_ratio
        * load_factor
        * peak_factor
        * staff_factor
        * float(model["queue_amplify"]),
    )
    return np.maximum(0.0, w_base.astype(float) * (1.0 - reduction_frac))


def rx_ids_touching_drugs(items_day: pd.DataFrame, drug_ids: set[str]) -> set:
    if not drug_ids:
        return set()
    hit = items_day["drugid"].astype(str).isin(drug_ids)
    return set(items_day.loc[hit, "处方编号"].unique())


def day_rx_predictions(
    bundle: WaitRegressorBundle,
    rx_day: pd.DataFrame,
    items_day: pd.DataFrame,
    scenario: LayoutScenario | None,
    queue_model: dict | None = None,
    restrict_rx_ids: set | None = None,
    start_t: int = 0,
    *,
    use_queue_propagation: bool = True,
    apply_layout_features: bool = True,
) -> pd.DataFrame:
    """
    Per-prescription predictions under do(a) for one day.

    S4.4: system ΔD_sys = mean(D_obs − D_cf) (on prescriptions receiving intervention);
    W~ = propagate_call_wait(W_obs, ΔD_sys, ...).
    start_t=0: full-day intervention; start_t=1: MorningRush prescriptions only.
    restrict_rx_ids: restrict mean to same prescriptions for fair null vs swap comparison.
    use_queue_propagation=False: change D/M/N features only, keep observed W (nested path decomposition A1).
    apply_layout_features=False: keep baseline D/M/N but still use ΔD_sys for queue propagation (Shapley queue-only layer).
    """
    if len(rx_day) == 0 or len(items_day) == 0:
        return pd.DataFrame()

    rx_day = ensure_morning_rush(rx_day)
    timing_active = active_rx_ids_for_timing(rx_day, start_t)

    items_cf_all = apply_scenario_to_items(items_day, scenario)
    d_base_all = items_day.groupby("处方编号")["调配秒数_最终"].sum() / 60.0
    d_cf_all = items_cf_all.groupby("处方编号")["调配秒数_最终"].sum() / 60.0
    aligned = pd.concat(
        [d_base_all.rename("base"), d_cf_all.rename("cf")], axis=1, join="inner"
    )
    if timing_active is not None:
        aligned_sys = aligned.loc[aligned.index.isin(timing_active)]
    else:
        aligned_sys = aligned
    system_disp_saved = (
        float((aligned_sys["base"] - aligned_sys["cf"]).mean()) if len(aligned_sys) else 0.0
    )

    if restrict_rx_ids is not None:
        rx_day = rx_day[rx_day["处方编号"].isin(restrict_rx_ids)]
        items_day = items_day[items_day["处方编号"].isin(restrict_rx_ids)]
        if len(rx_day) == 0 or len(items_day) == 0:
            return pd.DataFrame()
        items_cf = apply_scenario_to_items(items_day, scenario)
    else:
        items_cf = items_cf_all

    feat_cf = aggregate_rx_features(items_cf)
    feat_base = aggregate_rx_features(items_day)

    meta_cols = ["处方编号"]
    for c in ("Peak0", "Load0", "调配人数", "产能代理", W_FEATURE, "报到小时", "MorningRush"):
        if c in rx_day.columns:
            meta_cols.append(c)
    meta = rx_day[meta_cols].drop_duplicates("处方编号")

    cf_use = feat_cf.rename(
        columns={
            "预估配药_分钟": "预估配药_分钟_cf",
            "机器品项占比": "机器品项占比_cf",
            "品项数": "品项数_cf",
        }
    )
    base_use = feat_base.rename(
        columns={
            "预估配药_分钟": "预估配药_分钟_base",
            "机器品项占比": "机器品项占比_base",
            "品项数": "品项数_base",
        }
    )
    keep_cf = ["处方编号", "预估配药_分钟_cf", "机器品项占比_cf", "品项数_cf"]
    keep_base = ["处方编号", "预估配药_分钟_base", "机器品项占比_base", "品项数_base"]
    merged = (
        cf_use[keep_cf]
        .merge(base_use[keep_base], on="处方编号", how="inner")
        .merge(meta, on="处方编号", how="inner")
    )
    if merged.empty:
        return merged

    if timing_active is None:
        active_mask = np.ones(len(merged), dtype=bool)
    else:
        active_mask = merged["处方编号"].isin(timing_active).to_numpy()

    if apply_layout_features:
        merged["预估配药_分钟"] = np.where(
            active_mask, merged["预估配药_分钟_cf"], merged["预估配药_分钟_base"]
        )
        merged["机器品项占比"] = np.where(
            active_mask, merged["机器品项占比_cf"], merged["机器品项占比_base"]
        )
        merged["品项数"] = np.where(active_mask, merged["品项数_cf"], merged["品项数_base"])
    else:
        # Queue-only layer: baseline features; ΔD_sys still from scenario (computed above)
        merged["预估配药_分钟"] = merged["预估配药_分钟_base"]
        merged["机器品项占比"] = merged["机器品项占比_base"]
        merged["品项数"] = merged["品项数_base"]

    if "Peak0" not in merged.columns:
        if "报到小时" in merged.columns:
            merged["Peak0"] = merged["报到小时"].map(_is_peak).astype(int)
        else:
            merged["Peak0"] = 0
    if "Load0" not in merged.columns:
        merged["Load0"] = int(rx_day["Load0"].mode().iloc[0]) if "Load0" in rx_day.columns else 1
    if "调配人数" not in merged.columns:
        merged["调配人数"] = (
            float(rx_day["调配人数"].median()) if "调配人数" in rx_day.columns else 4.0
        )
    merged["调配人数"] = merged["调配人数"].fillna(4.0)

    if W_FEATURE not in merged.columns:
        merged[W_FEATURE] = float(queue_model.get("mean_wait_min", 3.7)) if queue_model else 3.7
    w_obs = pd.to_numeric(merged[W_FEATURE], errors="coerce")
    w_med = float(w_obs.median()) if w_obs.notna().any() else 3.5
    w_obs = w_obs.fillna(w_med)

    is_baseline = scenario is None or getattr(scenario, "kind", "baseline") == "baseline"
    w_out = w_obs.to_numpy(dtype=float).copy()
    if (
        use_queue_propagation
        and (not is_baseline)
        and abs(system_disp_saved) >= 1e-12
        and queue_model is not None
        and active_mask.any()
    ):
        w_cf = _propagate_call_wait_vectorized(
            w_obs.to_numpy(dtype=float),
            system_disp_saved,
            merged["Peak0"].to_numpy(dtype=float),
            merged["Load0"].to_numpy(dtype=float),
            queue_model,
            capacity=(
                merged["产能代理"].to_numpy(dtype=float)
                if "产能代理" in merged.columns
                else None
            ),
        )
        w_out = np.where(active_mask, w_cf, w_obs.to_numpy(dtype=float))
    merged[W_FEATURE] = w_out
    merged["use_queue_propagation"] = int(bool(use_queue_propagation))
    merged["apply_layout_features"] = int(bool(apply_layout_features))

    for c in bundle.feature_cols:
        if c not in merged.columns:
            merged[c] = 0.0
        merged[c] = pd.to_numeric(merged[c], errors="coerce")
        med = float(merged[c].median()) if merged[c].notna().any() else 0.0
        merged[c] = merged[c].fillna(med)

    merged["yhat"] = predict_minutes(bundle, merged)
    merged["system_disp_saved_min"] = system_disp_saved
    merged["timing_active"] = active_mask.astype(int)
    merged["start_t"] = int(start_t)
    return merged


def day_mean_wait(
    bundle: WaitRegressorBundle,
    rx_day: pd.DataFrame,
    items_day: pd.DataFrame,
    scenario: LayoutScenario | None,
    queue_model: dict | None = None,
    restrict_rx_ids: set | None = None,
    start_t: int = 0,
    *,
    use_queue_propagation: bool = True,
) -> float:
    """Mean predicted wait (minutes) under do(a) for daily prescriptions."""
    if queue_model is None and USE_COUNTERFACTUAL_W:
        queue_model = load_queue_model()
    pred = day_rx_predictions(
        bundle,
        rx_day,
        items_day,
        scenario,
        queue_model=queue_model,
        restrict_rx_ids=restrict_rx_ids,
        start_t=start_t,
        use_queue_propagation=use_queue_propagation,
    )
    if pred.empty or "yhat" not in pred.columns:
        return float("nan")
    return float(np.nanmean(pred["yhat"].to_numpy()))


def simulate_day_under_arm(
    bundle: WaitRegressorBundle,
    rx_day: pd.DataFrame,
    items_day: pd.DataFrame,
    arm_id: str,
    items_all: pd.DataFrame,
    queue_model: dict | None = None,
    *,
    scenario_cache: dict[str, LayoutScenario | None] | None = None,
    restrict_rx_ids: set | None = None,
) -> dict:
    """
    End-to-end daily intervention simulation kernel (S5): arm → LayoutScenario → item CF + W~ → Ŷ.

    @1 timing: apply layout CF only to MorningRush prescriptions; others stay at baseline.
    No reward_bonus / d_shift.
    """
    if queue_model is None and USE_COUNTERFACTUAL_W:
        queue_model = load_queue_model()
    start_t = parse_arm_start_t(arm_id)
    cache_key = arm_id.split("@")[0]
    if scenario_cache is not None and cache_key in scenario_cache:
        scenario = scenario_cache[cache_key]
    else:
        scenario = arm_to_scenario(arm_id, items_all)
        if scenario_cache is not None:
            scenario_cache[cache_key] = scenario

    pred = day_rx_predictions(
        bundle,
        rx_day,
        items_day,
        scenario,
        queue_model=queue_model,
        restrict_rx_ids=restrict_rx_ids,
        start_t=start_t,
    )
    if pred.empty or "yhat" not in pred.columns:
        return {
            "arm": arm_id,
            "mu_min": float("nan"),
            "reward": float("nan"),
            "n_rx": 0,
            "n_timing_active": 0,
            "system_disp_saved_min": 0.0,
            "scenario_kind": None if scenario is None else scenario.kind,
            "start_t": start_t,
            "timing_note": None,
        }
    mu = float(np.nanmean(pred["yhat"].to_numpy()))
    sys_d = float(pred["system_disp_saved_min"].iloc[0]) if "system_disp_saved_min" in pred else 0.0
    n_act = int(pred["timing_active"].sum()) if "timing_active" in pred.columns else int(len(pred))
    return {
        "arm": arm_id,
        "mu_min": mu,
        "reward": reward_from_yhat(mu),
        "n_rx": int(len(pred)),
        "n_timing_active": n_act,
        "system_disp_saved_min": sys_d,
        "scenario_kind": None if scenario is None else scenario.kind,
        "start_t": start_t,
        "timing_note": (
            f"start_t={start_t}: CF on {n_act}/{len(pred)} rx (MorningRush only)"
            if start_t >= 1
            else "full-day intervention"
        ),
    }



def reward_from_yhat(yhat_min: float) -> float:
    """Primary report: R = -Ŷ_min."""
    return float(-yhat_min)


def check_arm_signal_gate_ttest(
    daily_mu: dict,
    arm_null: str = "null",
    arm_alt: str = "swap40",
    alpha: float = 0.05,
    holdout_mae: float | None = None,
) -> dict:
    """
    S4.5 gate: one-sided t-test on daily δ_t = μ_null(t) − μ_swap40(t).

    H1: E[δ] > 0 (i.e. μ_swap40 < μ_null, shorter wait)
    Pass: p_one < alpha and mean(δ) > 0.

    Hold-out MAE written for documentation only; not used for gating.
    """
    from scipy import stats

    diffs = []
    for arms in daily_mu.values():
        if arm_null not in arms or arm_alt not in arms:
            continue
        n, s = float(arms[arm_null]), float(arms[arm_alt])
        if np.isfinite(n) and np.isfinite(s):
            diffs.append(n - s)
    diffs = np.asarray(diffs, dtype=float)
    n_days = int(len(diffs))
    if n_days < 3:
        return {
            "gate_pass": False,
            "rule": f"one-sided t-test on daily (μ_{arm_null}−μ_{arm_alt}), p<{alpha}",
            "note": "FAIL: fewer than 3 days with both arms",
            "n_days": n_days,
            "holdout_mae_info_only": holdout_mae,
        }

    mean_d = float(diffs.mean())
    std_d = float(diffs.std(ddof=1))
    se = float(std_d / np.sqrt(n_days))
    t_stat, p_two = stats.ttest_1samp(diffs, 0.0)
    t_stat = float(t_stat)
    p_two = float(p_two)
    # One-sided: H1 mean > 0
    if t_stat > 0:
        p_one = p_two / 2.0
    else:
        p_one = 1.0 - p_two / 2.0
    direction_ok = mean_d > 0
    ok = bool(direction_ok and p_one < alpha)
    ci_lo, ci_hi = stats.t.interval(0.95, n_days - 1, loc=mean_d, scale=se)

    out = {
        "mu_null_pooled": round(float(np.mean([v[arm_null] for v in daily_mu.values() if arm_null in v])), 4),
        "mu_swap40_pooled": round(float(np.mean([v[arm_alt] for v in daily_mu.values() if arm_alt in v])), 4),
        "mean_delta_null_minus_swap40": round(mean_d, 4),
        "abs_mean_delta": round(abs(mean_d), 4),
        "std_delta": round(std_d, 4),
        "se_delta": round(se, 4),
        "t_stat": round(t_stat, 4),
        "p_one_sided": float(p_one),
        "p_two_sided": float(p_two),
        "alpha": alpha,
        "n_days": n_days,
        "frac_days_alt_better": round(float((diffs > 0).mean()), 4),
        "ci95_mean_delta": [round(float(ci_lo), 4), round(float(ci_hi), 4)],
        "direction_ok": bool(direction_ok),
        "gate_pass": ok,
        "rule": f"one-sided t-test H1: E[μ_{arm_null}−μ_{arm_alt}]>0, p<{alpha}",
        "note": (
            "PASS: day-level arm ranking statistically identifiable — may report regret"
            if ok
            else "FAIL: arm difference not significant or wrong direction — do NOT report regret"
        ),
        # MAE demoted to documentation; does not affect gate_pass
        "holdout_mae_info_only": round(float(holdout_mae), 4) if holdout_mae is not None else None,
        "mae_vs_delta_ratio_info": (
            round(abs(mean_d) / float(holdout_mae), 4)
            if holdout_mae is not None and holdout_mae > 0
            else None
        ),
    }
    return out


# Back-compat alias: forwards to t-test gate (passing pooled scalars will error; use daily_mu)
def check_arm_signal_gate(*args, **kwargs):
    """MAE gate removed; use check_arm_signal_gate_ttest(daily_mu, ...) instead."""
    if args and isinstance(args[0], dict):
        return check_arm_signal_gate_ttest(*args, **kwargs)
    raise TypeError(
        "MAE gate removed. Use check_arm_signal_gate_ttest(daily_mu, holdout_mae=mae)."
    )
