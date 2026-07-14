"""Dual non-stationary mechanism family θ(r, c): stratified CPT by MorningRush × CalBlock + drift metrics."""

from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from phase2.causal_graph import PARENTS
from phase2.config import CARDINALITY, DATA_DIR, DIRICHLET_ALPHA
from phase2.mcmc_calibrate import build_count_tables, cpts_to_json, gibbs_sample_cpts

PHASE2_DIR = DATA_DIR / "phase2"
NS_CPT_DIR = PHASE2_DIR / "ns_cpts"

CAL_BLOCKS = ("2024H1", "2024H2", "2025H1", "2025H2")
RUSH_LABELS = ("other", "rush")


def cal_block(ts: pd.Timestamp) -> str:
    y = int(ts.year)
    h = "H1" if int(ts.month) <= 6 else "H2"
    return f"{y}{h}"


def morning_rush(hour: float, weekday: int) -> int:
    """Weekday and check-in hour ∈ {9,10}."""
    if weekday >= 5:
        return 0
    h = int(hour) if pd.notna(hour) else -1
    return 1 if h in (9, 10) else 0


def annotate_regimes(disc: pd.DataFrame, dates: pd.Series) -> pd.DataFrame:
    out = disc.copy()
    dt = pd.to_datetime(dates)
    out["日期"] = dt
    out["CalBlock"] = dt.map(cal_block)
    wd = dt.dt.weekday
    out["MorningRush"] = [
        morning_rush(h, int(w)) for h, w in zip(out["报到小时"], wd)
    ]
    out["regime"] = out["MorningRush"].map({0: "other", 1: "rush"})
    return out


def dirichlet_posterior_mean_cpts(
    count_tables: dict[str, dict[str, np.ndarray]],
    alpha: float = DIRICHLET_ALPHA,
) -> dict[str, dict[str, list[float]]]:
    """Dirichlet posterior mean (equivalent to the mean limit of infinite Gibbs draws)."""
    cpts: dict[str, dict[str, list[float]]] = {}
    for node, table in count_tables.items():
        cpts[node] = {}
        for key, cnt in table.items():
            arr = cnt.astype(float) + alpha
            cpts[node][key] = (arr / arr.sum()).tolist()
    return cpts


def fit_stratum_cpt(
    df: pd.DataFrame,
    *,
    use_gibbs: bool = False,
    n_draws: int = 200,
    n_burn: int = 50,
    seed: int = 42,
) -> dict:
    need = list(PARENTS.keys())
    sub = df.dropna(subset=need).copy()
    for c in need:
        sub[c] = sub[c].astype(int)
    counts = build_count_tables(sub)
    if use_gibbs:
        cpts, _ = gibbs_sample_cpts(counts, n_draws=n_draws, n_burn=n_burn, seed=seed)
    else:
        cpts = dirichlet_posterior_mean_cpts(counts)
    return {
        "n": int(len(sub)),
        "cpts": cpts_to_json(cpts),
        "counts": {n: {k: v.tolist() for k, v in t.items()} for n, t in counts.items()},
    }


def _tv(p: np.ndarray, q: np.ndarray) -> float:
    return 0.5 * float(np.abs(p - q).sum())


def _kl(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))


def compare_cpts(cpts_a: dict, cpts_b: dict, nodes: list[str] | None = None) -> dict:
    nodes = nodes or ["W0", "D0", "V0", "Y0"]
    out: dict = {}
    for node in nodes:
        if node not in cpts_a or node not in cpts_b:
            continue
        tvs, kls = [], []
        keys = set(cpts_a[node]) & set(cpts_b[node])
        for key in keys:
            pa = np.asarray(cpts_a[node][key], dtype=float)
            pb = np.asarray(cpts_b[node][key], dtype=float)
            tvs.append(_tv(pa, pb))
            kls.append(_kl(pa, pb))
        out[node] = {
            "n_parent_configs": len(keys),
            "mean_tv": round(float(np.mean(tvs)), 4) if tvs else None,
            "max_tv": round(float(np.max(tvs)), 4) if tvs else None,
            "mean_kl": round(float(np.mean(kls)), 4) if kls else None,
        }
    return out


def fit_all_strata(
    disc: pd.DataFrame,
    *,
    use_gibbs: bool = False,
    min_n: int = 200,
) -> dict:
    results: dict = {"strata": {}, "meta": {"use_gibbs": use_gibbs, "min_n": min_n}}
    for block, regime in product(CAL_BLOCKS, RUSH_LABELS):
        mask = (disc["CalBlock"] == block) & (disc["regime"] == regime)
        sub = disc.loc[mask]
        key = f"{block}_{regime}"
        if len(sub) < min_n:
            results["strata"][key] = {"n": int(len(sub)), "skipped": True}
            continue
        fitted = fit_stratum_cpt(sub, use_gibbs=use_gibbs)
        results["strata"][key] = fitted
        print(f"  fitted {key}: n={fitted['n']}")
    return results


def compute_drift_metrics(strata: dict) -> dict:
    """rush vs other (same CalBlock); adjacent half-year blocks (same regime)."""
    rush_vs_other = {}
    for block in CAL_BLOCKS:
        a = strata.get(f"{block}_rush", {})
        b = strata.get(f"{block}_other", {})
        if a.get("skipped") or b.get("skipped") or "cpts" not in a or "cpts" not in b:
            rush_vs_other[block] = {"skipped": True}
            continue
        rush_vs_other[block] = {
            "n_rush": a["n"],
            "n_other": b["n"],
            "drift": compare_cpts(a["cpts"], b["cpts"]),
        }

    calendar_drift = {}
    for regime in RUSH_LABELS:
        pairs = []
        for i in range(len(CAL_BLOCKS) - 1):
            b0, b1 = CAL_BLOCKS[i], CAL_BLOCKS[i + 1]
            a = strata.get(f"{b0}_{regime}", {})
            b = strata.get(f"{b1}_{regime}", {})
            if a.get("skipped") or b.get("skipped") or "cpts" not in a or "cpts" not in b:
                pairs.append({"from": b0, "to": b1, "skipped": True})
                continue
            pairs.append(
                {
                    "from": b0,
                    "to": b1,
                    "n_from": a["n"],
                    "n_to": b["n"],
                    "drift": compare_cpts(a["cpts"], b["cpts"]),
                }
            )
        calendar_drift[regime] = pairs

    # summary: mean TV on W0 for rush vs other
    w0_tvs = [
        v["drift"]["W0"]["mean_tv"]
        for v in rush_vs_other.values()
        if not v.get("skipped") and v.get("drift", {}).get("W0", {}).get("mean_tv") is not None
    ]
    return {
        "rush_vs_other_by_calblock": rush_vs_other,
        "calendar_drift_by_regime": calendar_drift,
        "summary": {
            "mean_W0_tv_rush_vs_other": round(float(np.mean(w0_tvs)), 4) if w0_tvs else None,
            "n_calblocks_compared": len(w0_tvs),
        },
    }


def save_ns_cpts(results: dict, out_dir: Path = NS_CPT_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for key, payload in results["strata"].items():
        if payload.get("skipped") or "cpts" not in payload:
            continue
        path = out_dir / f"{key}.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(
                {"key": key, "n": payload["n"], "cpts": payload["cpts"]},
                f,
                ensure_ascii=False,
                indent=2,
            )
    with (out_dir / "index.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "cal_blocks": list(CAL_BLOCKS),
                "regimes": list(RUSH_LABELS),
                "strata": {
                    k: {"n": v.get("n"), "skipped": bool(v.get("skipped"))}
                    for k, v in results["strata"].items()
                },
                "meta": results.get("meta", {}),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )


def load_ns_cpt(calblock: str, regime: str, out_dir: Path = NS_CPT_DIR) -> dict:
    path = out_dir / f"{calblock}_{regime}.json"
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def load_all_ns_cpts(out_dir: Path = NS_CPT_DIR) -> dict[str, dict]:
    index_path = out_dir / "index.json"
    with index_path.open(encoding="utf-8") as f:
        index = json.load(f)
    out = {}
    for key, meta in index["strata"].items():
        if meta.get("skipped"):
            continue
        with (out_dir / f"{key}.json").open(encoding="utf-8") as f:
            out[key] = json.load(f)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Dual non-stationary mechanisms θ(r,c)")
    parser.add_argument("--gibbs", action="store_true", help="Use short Gibbs instead of Dirichlet mean")
    parser.add_argument("--min-n", type=int, default=200)
    args = parser.parse_args()

    print("[1/4] Loading discretized prescriptions...")
    disc = pd.read_parquet(PHASE2_DIR / "rx_discretized.parquet")
    rx_dates = pd.read_parquet(DATA_DIR / "rx_level.parquet", columns=["处方编号", "日期"])
    disc = disc.merge(rx_dates, on="处方编号", how="left")

    print("[2/4] Annotating MorningRush / CalBlock...")
    disc = annotate_regimes(disc, disc["日期"])
    print(disc.groupby(["CalBlock", "regime"]).size().unstack(fill_value=0))

    print("[3/4] Stratified CPT fitting...")
    results = fit_all_strata(disc, use_gibbs=args.gibbs, min_n=args.min_n)
    save_ns_cpts(results)
    print(f"  saved under {NS_CPT_DIR}")

    print("[4/4] Drift metrics...")
    drift = compute_drift_metrics(results["strata"])
    drift_path = PHASE2_DIR / "ns_drift_metrics.json"
    with drift_path.open("w", encoding="utf-8") as f:
        json.dump(drift, f, ensure_ascii=False, indent=2)
    print("  summary:", drift["summary"])
    print("saved:", drift_path)


if __name__ == "__main__":
    main()
