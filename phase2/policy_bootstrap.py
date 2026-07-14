"""Day-level cluster bootstrap policy interval (not standard IPS/DR OPE).

On existing daily_mu: δ_t = μ_null(t) − μ_swap40(t), cluster bootstrap by CalBlock,
yielding point estimate and confidence interval for daily mean wait savings of swap40 vs null.

Honest disclaimer: this is a policy-value interval on model-inferred μ, not classic OPE with propensity scores.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from phase2.config import DATA_DIR
from phase2.ns_mechanisms import cal_block

PHASE2_DIR = DATA_DIR / "phase2"
DEFAULT_MU = PHASE2_DIR / "ns_bandit_results" / "continuous_reward" / "daily_mu_minutes.json"
OUT_DIR = PHASE2_DIR / "policy_bootstrap"


def load_day_deltas(
    mu_path: Path,
    arm: str = "swap40",
    baseline: str = "null",
) -> pd.DataFrame:
    with mu_path.open(encoding="utf-8") as f:
        daily_mu = json.load(f)
    rows = []
    for d, arms in daily_mu.items():
        if baseline not in arms or arm not in arms:
            continue
        m0, ma = float(arms[baseline]), float(arms[arm])
        if not (np.isfinite(m0) and np.isfinite(ma)):
            continue
        dt = pd.Timestamp(d)
        rows.append(
            {
                "date": d,
                "CalBlock": cal_block(dt),
                "mu_null": m0,
                f"mu_{arm}": ma,
                "delta": m0 - ma,
            }
        )
    return pd.DataFrame(rows)


def day_bootstrap_ci(
    df: pd.DataFrame,
    n_boot: int = 2000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict:
    """Day-level iid bootstrap (reference; cluster CI is wide when few clusters)."""
    vals = df["delta"].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=float)
    n = len(vals)
    for b in range(n_boot):
        boots[b] = float(vals[rng.integers(0, n, size=n)].mean())
    return {
        "n_boot": n_boot,
        "alpha": alpha,
        "ci_low": round(float(np.quantile(boots, alpha / 2)), 6),
        "ci_high": round(float(np.quantile(boots, 1 - alpha / 2)), 6),
        "boot_se": round(float(boots.std(ddof=1)), 6),
        "method": "iid day bootstrap (secondary; not cluster-robust)",
    }


def cluster_bootstrap_ci(
    df: pd.DataFrame,
    n_boot: int = 2000,
    seed: int = 42,
    alpha: float = 0.05,
    cluster_col: str = "CalBlock",
) -> dict:
    """Percentile CI for daily mean δ via cluster resampling with replacement."""
    clusters = sorted(df[cluster_col].unique())
    by = {c: df.loc[df[cluster_col] == c, "delta"].to_numpy(dtype=float) for c in clusters}
    rng = np.random.default_rng(seed)
    point = float(df["delta"].mean())
    boots = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        # Standard cluster bootstrap: resample C clusters with replacement, use all obs within cluster
        chosen = rng.choice(clusters, size=len(clusters), replace=True)
        parts = [by[c] for c in chosen]
        boots[b] = float(np.concatenate(parts).mean())

    lo = float(np.quantile(boots, alpha / 2))
    hi = float(np.quantile(boots, 1 - alpha / 2))
    # Days per block
    block_n = {c: int((df[cluster_col] == c).sum()) for c in clusters}
    block_mean = {
        c: round(float(df.loc[df[cluster_col] == c, "delta"].mean()), 6) for c in clusters
    }
    return {
        "n_days": int(len(df)),
        "n_clusters": int(len(clusters)),
        "clusters": clusters,
        "block_n_days": block_n,
        "block_mean_delta": block_mean,
        "point_mean_delta_min": round(point, 6),
        "bootstrap": {
            "n_boot": n_boot,
            "alpha": alpha,
            "ci_low": round(lo, 6),
            "ci_high": round(hi, 6),
            "boot_mean": round(float(boots.mean()), 6),
            "boot_se": round(float(boots.std(ddof=1)), 6),
            "method": "cluster bootstrap by CalBlock (resample clusters with replacement)",
        },
        "rough_hours_saved_per_year": None,  # filled by caller if needed
    }


def rough_hours(
    delta_min: float,
    rx_per_day: float = 2000.0,
    days_per_year: float = 250.0,
) -> float:
    """Rough estimate: Δ min/rx × rx/day × business days/year / 60."""
    return float(delta_min * rx_per_day * days_per_year / 60.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Cluster-bootstrap policy interval")
    parser.add_argument("--mu", type=Path, default=DEFAULT_MU)
    parser.add_argument("--arm", default="swap40")
    parser.add_argument("--baseline", default="null")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rx-per-day", type=float, default=0.0, help="0=estimate daily rx volume from data")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_day_deltas(args.mu, arm=args.arm, baseline=args.baseline)
    if df.empty:
        raise SystemExit("no daily deltas")

    res = cluster_bootstrap_ci(df, n_boot=args.n_boot, seed=args.seed)
    res["day_bootstrap"] = day_bootstrap_ci(df, n_boot=args.n_boot, seed=args.seed)
    # Daily rx volume: prefer rx_level
    if args.rx_per_day > 0:
        rpd = args.rx_per_day
    else:
        rx = pd.read_parquet(DATA_DIR / "rx_level.parquet", columns=["日期", "处方编号"])
        rx["日期"] = pd.to_datetime(rx["日期"]).dt.date
        rpd = float(rx.groupby("日期")["处方编号"].nunique().mean())

    d = res["point_mean_delta_min"]
    lo = res["bootstrap"]["ci_low"]
    hi = res["bootstrap"]["ci_high"]
    res["rx_per_day_used"] = round(rpd, 1)
    res["rough_hours_saved_per_year"] = {
        "point": round(rough_hours(d, rpd), 1),
        "ci_low": round(rough_hours(lo, rpd), 1),
        "ci_high": round(rough_hours(hi, rpd), 1),
        "note": "Δmin/rx × rx/day × 250 / 60; rough estimate, not a clinical trial endpoint",
    }
    res["identification"] = (
        "Day-level cluster bootstrap on model-based daily μ differences; "
        "NOT inverse-propensity / doubly-robust OPE"
    )
    res["arm"] = args.arm
    res["baseline"] = args.baseline
    res["mu_path"] = str(args.mu)

    out = OUT_DIR / f"policy_interval_{args.arm}.json"
    with out.open("w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    df.to_csv(OUT_DIR / f"day_deltas_{args.arm}.csv", index=False)

    print("=== Policy interval (cluster bootstrap) ===")
    print(res["identification"])
    print(
        f"  {args.arm} vs {args.baseline}: "
        f"Δ={d:.4f} min/day  "
        f"cluster 95% CI [{lo:.4f}, {hi:.4f}]  "
        f"day 95% CI [{res['day_bootstrap']['ci_low']:.4f}, {res['day_bootstrap']['ci_high']:.4f}]  "
        f"(n_days={res['n_days']}, blocks={res['clusters']})"
    )
    print("  rough hours/year:", res["rough_hours_saved_per_year"])
    print("done →", out)


if __name__ == "__main__":
    main()
