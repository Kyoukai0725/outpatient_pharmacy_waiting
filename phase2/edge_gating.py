"""S1: Data-driven edge gating tests (8 atomic contexts × candidate edges).

Thresholds (pre-locked):
  ε=0.05, BF>3, n_min=2000 (500 for D1→W2), stability 80% (20×50% subsample)
Day-level regime: day is rush when MorningRush fraction ≥ τ=0.30.
"""

from __future__ import annotations

import argparse
import json
import math
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from phase2.config import CARDINALITY, DATA_DIR
from phase2.ns_mechanisms import CAL_BLOCKS, annotate_regimes
from phase2.temporal_scm import build_temporal_diagram

PHASE2_DIR = DATA_DIR / "phase2"
OUT_DIR = PHASE2_DIR / "contextual_graph"

# --- Pre-locked thresholds ---
EPS_TV = 0.05
BF_THRESHOLD = 3.0
N_MIN_DEFAULT = 2000
N_MIN_D1_W2 = 500
STABILITY_ROUNDS = 20
STABILITY_FRAC = 0.5
STABILITY_AGREE = 0.80
TAU_RUSH_DAY = 0.30
ALPHA = 1.0

# Candidate edges: id -> specification
CANDIDATE_EDGES = (
    {
        "edge_id": "D→W",
        "kind": "slice",
        "child": "W0",
        "drop_parent": "D0",
        "parents_full": ("Peak0", "Load0", "D0"),
        "n_min": N_MIN_DEFAULT,
    },
    {
        "edge_id": "Load→W",
        "kind": "slice",
        "child": "W0",
        "drop_parent": "Load0",
        "parents_full": ("Peak0", "Load0", "D0"),
        "n_min": N_MIN_DEFAULT,
    },
    {
        "edge_id": "Load→V",
        "kind": "slice",
        "child": "V0",
        "drop_parent": "Load0",
        "parents_full": ("Peak0", "Load0"),
        "n_min": N_MIN_DEFAULT,
    },
    {
        "edge_id": "D1→W2",
        "kind": "proxy_d1_w2",
        "child": "W0",
        "drop_parent": "rush_D_proxy",
        "parents_full": ("Peak0", "Load0", "rush_D_proxy"),
        "n_min": N_MIN_D1_W2,
    },
    {
        "edge_id": "Wprev→W",
        "kind": "proxy_wprev",
        "child": "W0",
        "drop_parent": "rush_W_proxy",
        "parents_full": ("Peak0", "Load0", "D0", "rush_W_proxy"),
        "n_min": N_MIN_DEFAULT,
    },
)


def _log_gamma(x: float) -> float:
    return math.lgamma(x)


def dirichlet_multinomial_log_marginal(
    counts_by_cfg: dict[tuple, np.ndarray],
    n_states: int,
    alpha: float = ALPHA,
) -> float:
    """Σ_cfg [ log B(n+α) - log B(α) ] for Multinomial-Dirichlet."""
    log_b_alpha = n_states * _log_gamma(alpha) - _log_gamma(n_states * alpha)
    total = 0.0
    for cnt in counts_by_cfg.values():
        n = float(cnt.sum())
        if n <= 0:
            continue
        s = _log_gamma(n_states * alpha) - _log_gamma(n + n_states * alpha)
        for j in range(n_states):
            s += _log_gamma(float(cnt[j]) + alpha) - _log_gamma(alpha)
        # s = log B(n+α) - log B(α)  already via expansion; subtract log_b_alpha form:
        total += s - log_b_alpha
    return total


def _parent_key(row: pd.Series, parents: tuple[str, ...]) -> tuple:
    return tuple(int(row[p]) for p in parents)


def build_counts(
    df: pd.DataFrame,
    child: str,
    parents: tuple[str, ...],
    n_states: int,
) -> dict[tuple, np.ndarray]:
    tables: dict[tuple, np.ndarray] = {}
    if not parents:
        cnt = np.bincount(df[child].astype(int).to_numpy(), minlength=n_states).astype(float)
        tables[()] = cnt[:n_states]
        return tables
    g = df.groupby(list(parents) + [child], sort=False).size().reset_index(name="n")
    for _, row in g.iterrows():
        key = tuple(int(row[p]) for p in parents)
        y = int(row[child])
        if key not in tables:
            tables[key] = np.zeros(n_states, dtype=float)
        if 0 <= y < n_states:
            tables[key][y] += float(row["n"])
    return tables


def mean_conditional_tv(
    counts_full: dict[tuple, np.ndarray],
    drop_idx: int,
    n_states: int,
    alpha: float = ALPHA,
) -> float:
    """
    For each full parent config, compare P(Y|pa) vs marginalized P(Y|pa\\X) after dropping drop dim.
    Smooth with counts+α; weighted average TV by config sample size.
    """
    # Aggregate reduced
    reduced: dict[tuple, np.ndarray] = {}
    for key, cnt in counts_full.items():
        rkey = key[:drop_idx] + key[drop_idx + 1 :]
        reduced[rkey] = reduced.get(rkey, np.zeros(n_states, dtype=float)) + cnt

    def _prob(cnt: np.ndarray) -> np.ndarray:
        a = cnt + alpha
        return a / a.sum()

    num, den = 0.0, 0.0
    for key, cnt in counts_full.items():
        n = float(cnt.sum())
        if n <= 0:
            continue
        rkey = key[:drop_idx] + key[drop_idx + 1 :]
        p_full = _prob(cnt)
        p_red = _prob(reduced[rkey])
        tv = 0.5 * float(np.abs(p_full - p_red).sum())
        num += n * tv
        den += n
    return float(num / den) if den > 0 else 1.0


def test_edge_once(df: pd.DataFrame, edge: dict) -> dict:
    child = edge["child"]
    parents_full = edge["parents_full"]
    drop = edge["drop_parent"]
    n_states = int(CARDINALITY.get(child, 4))
    if child == "W0":
        n_states = 4
    if drop not in df.columns or child not in df.columns:
        return {
            "n": 0,
            "mean_tv": None,
            "log_bf_remove": None,
            "bf_remove": None,
            "decision_raw": "skip_missing_cols",
            "remove": False,
        }

    need = list(parents_full) + [child]
    sub = df.dropna(subset=need).copy()
    for c in need:
        sub[c] = sub[c].astype(int)
    n = len(sub)
    n_min = int(edge["n_min"])
    if n < n_min:
        return {
            "n": n,
            "mean_tv": None,
            "log_bf_remove": None,
            "bf_remove": None,
            "decision_raw": "insufficient_n",
            "remove": False,
        }

    parents_red = tuple(p for p in parents_full if p != drop)
    drop_idx = parents_full.index(drop)

    counts_full = build_counts(sub, child, parents_full, n_states)
    counts_red = build_counts(sub, child, parents_red, n_states)

    ll_full = dirichlet_multinomial_log_marginal(counts_full, n_states)
    ll_red = dirichlet_multinomial_log_marginal(counts_red, n_states)
    # BF(remove vs keep) = p(data|red) / p(data|full)
    log_bf = ll_red - ll_full
    bf = float(math.exp(min(max(log_bf, -50), 50)))

    mtv = mean_conditional_tv(counts_full, drop_idx, n_states)
    remove_stat = (bf > BF_THRESHOLD) or (mtv < EPS_TV)
    return {
        "n": n,
        "mean_tv": round(mtv, 6),
        "log_bf_remove": round(log_bf, 4),
        "bf_remove": round(bf, 4),
        "decision_raw": "remove_candidate" if remove_stat else "keep",
        "remove": bool(remove_stat),
    }


def stability_agree(df: pd.DataFrame, edge: dict, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    decisions = []
    n = len(df)
    if n < 10:
        return 0.0
    for i in range(STABILITY_ROUNDS):
        idx = rng.choice(n, size=max(1, int(n * STABILITY_FRAC)), replace=False)
        sub = df.iloc[idx]
        r = test_edge_once(sub, edge)
        decisions.append(bool(r["remove"]))
    if not decisions:
        return 0.0
    # Compare against full-sample decision: majority vote as stability target
    majority = sum(decisions) >= (len(decisions) / 2)
    return float(sum(d == majority for d in decisions) / len(decisions))


def finalize_decision(stat: dict, stability: float, n_min: int) -> dict:
    out = dict(stat)
    out["stability"] = round(stability, 4)
    out["n_min"] = n_min
    if stat.get("decision_raw") == "insufficient_n":
        out["remove_final"] = False
        out["decision"] = "keep_insufficient_n"
        return out
    if stat.get("decision_raw") == "skip_missing_cols":
        out["remove_final"] = False
        out["decision"] = "keep_missing"
        return out
    if stat.get("remove") and stability >= STABILITY_AGREE and stat["n"] >= n_min:
        out["remove_final"] = True
        out["decision"] = "remove"
    else:
        out["remove_final"] = False
        reason = "keep"
        if stat.get("remove") and stability < STABILITY_AGREE:
            reason = "keep_unstable"
        out["decision"] = reason
    return out


def add_day_proxies(disc: pd.DataFrame) -> pd.DataFrame:
    """Build same-day rush proxy features for cross-slice edges; day regime uses τ=0.30."""
    out = disc.copy()
    out["日期_d"] = pd.to_datetime(out["日期"]).dt.date

    rush_only = out[out["MorningRush"] == 1]
    rush_D = rush_only.groupby("日期_d")["D0"].mean()
    rush_W = rush_only.groupby("日期_d")["W0"].mean()
    day_regime = (
        out.groupby("日期_d")["MorningRush"]
        .mean()
        .ge(TAU_RUSH_DAY)
        .map({True: "rush", False: "other"})
    )

    out["day_regime"] = out["日期_d"].map(day_regime)
    out["rush_D_mean"] = out["日期_d"].map(rush_D)
    out["rush_W_mean"] = out["日期_d"].map(rush_W)

    def _bin_mean(x):
        if pd.isna(x):
            return np.nan
        return int(np.clip(int(round(float(x))), 0, 3))

    out["rush_D_proxy"] = out["rush_D_mean"].map(_bin_mean)
    out["rush_W_proxy"] = out["rush_W_mean"].map(_bin_mean)
    return out


def stratum_mask(df: pd.DataFrame, regime: str, calblock: str, edge: dict) -> pd.Series:
    """
    Prescription-level stratum: prescription MorningRush and CalBlock (consistent with ns_cpts).
    Cross-slice proxy edges: test on other (or rest) prescriptions; require non-null proxy.
    """
    base = (df["CalBlock"] == calblock) & (df["regime"] == regime)
    if edge["kind"] == "slice":
        return base
    if edge["kind"] == "proxy_d1_w2":
        # Test on other layer for afternoon/non-rush dependence on same-day rush dispensing; require rush data that day
        return (df["CalBlock"] == calblock) & (df["regime"] == "other") & df["rush_D_proxy"].notna()
    if edge["kind"] == "proxy_wprev":
        # Non-rush period depends on same-day rush wait
        return (df["CalBlock"] == calblock) & (df["regime"] == "other") & df["rush_W_proxy"].notna()
    return base


def run_all_tests(disc: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    rows = []
    for regime, calblock in product(("rush", "other"), CAL_BLOCKS):
        for edge in CANDIDATE_EDGES:
            # proxy edges reported only in other cell (rush cell marked N/A)
            if edge["kind"].startswith("proxy") and regime == "rush":
                rows.append(
                    {
                        "calblock": calblock,
                        "regime": regime,
                        "context": f"{calblock}_{regime}",
                        "edge_id": edge["edge_id"],
                        "n": 0,
                        "mean_tv": None,
                        "bf_remove": None,
                        "stability": None,
                        "decision": "na_proxy_on_rush_cell",
                        "remove_final": False,
                        "n_min": edge["n_min"],
                    }
                )
                continue

            mask = stratum_mask(disc, regime, calblock, edge)
            sub = disc.loc[mask]
            # For proxy: context label still uses calblock_other
            ctx = f"{calblock}_{regime}" if edge["kind"] == "slice" else f"{calblock}_other"

            stat = test_edge_once(sub, edge)
            if stat["n"] >= edge["n_min"]:
                stab = stability_agree(sub, edge, seed=seed + hash(edge["edge_id"] + ctx) % 10000)
            else:
                stab = 0.0
            final = finalize_decision(stat, stab, edge["n_min"])
            rows.append(
                {
                    "calblock": calblock,
                    "regime": regime if edge["kind"] == "slice" else "other",
                    "context": ctx,
                    "edge_id": edge["edge_id"],
                    "kind": edge["kind"],
                    **{k: final[k] for k in (
                        "n", "mean_tv", "log_bf_remove", "bf_remove",
                        "stability", "decision", "remove_final", "n_min",
                    )},
                }
            )
            print(
                f"  {ctx:16s} {edge['edge_id']:8s} n={final['n']:6d} "
                f"tv={final.get('mean_tv')} bf={final.get('bf_remove')} "
                f"stab={final.get('stability')} → {final['decision']}"
            )
    return pd.DataFrame(rows)


def build_Gc_edge_lists(tests: pd.DataFrame) -> dict:
    """
    Start from hypergraph edge set; remove edges per test results.
    Single-slice edges map to all t; cross-slice edges removed by id.
    """
    G = build_temporal_diagram()
    base_edges = sorted((u, v) for u, v in G.edges)

    # remove set per context
    contexts = sorted({f"{b}_{r}" for b in CAL_BLOCKS for r in ("rush", "other")})
    graphs = {}
    for ctx in contexts:
        remove_ids = set(
            tests.loc[
                (tests["context"] == ctx) & (tests["remove_final"] == True),
                "edge_id",
            ]
        )
        # proxy results written to calblock_other: sync to that calblock's other graph
        kept = []
        removed = []
        for u, v in base_edges:
            eid = _map_temporal_edge_to_candidate(u, v)
            if eid and eid in remove_ids:
                removed.append([u, v])
            else:
                kept.append([u, v])
        graphs[ctx] = {
            "context": ctx,
            "removed_edge_ids": sorted(remove_ids),
            "removed_edges": removed,
            "kept_edges": kept,
            "n_kept": len(kept),
            "n_removed": len(removed),
        }
    return graphs


def _map_temporal_edge_to_candidate(u: str, v: str) -> str | None:
    """Map temporal edge name → candidate edge_id."""
    # D_t → W_t
    if u.startswith("D") and v.startswith("W") and u[1:] == v[1:]:
        return "D→W"
    if u.startswith("Load") and v.startswith("W") and u[4:] == v[1:]:
        return "Load→W"
    if u.startswith("Load") and v.startswith("V") and u[4:] == v[1:]:
        return "Load→V"
    if u == "D1" and v == "W2":
        return "D1→W2"
    if u.startswith("W") and v.startswith("W"):
        try:
            if int(v[1:]) == int(u[1:]) + 1:
                return "Wprev→W"
        except ValueError:
            pass
    return None


def summarize(tests: pd.DataFrame, graphs: dict) -> dict:
    nontrivial = []
    for eid in tests["edge_id"].unique():
        sub = tests[tests["edge_id"] == eid]
        rem = sub[sub["remove_final"] == True]
        keep = sub[sub["decision"].astype(str).str.startswith("keep") | (sub["remove_final"] == False)]
        if len(rem) and len(sub) > len(rem):
            nontrivial.append(
                {
                    "edge_id": eid,
                    "n_contexts_removed": int(len(rem)),
                    "contexts_removed": rem["context"].tolist(),
                }
            )
    n_any_remove = int(tests["remove_final"].sum())
    return {
        "thresholds": {
            "eps_tv": EPS_TV,
            "bf": BF_THRESHOLD,
            "n_min_default": N_MIN_DEFAULT,
            "n_min_D1_W2": N_MIN_D1_W2,
            "stability_agree": STABILITY_AGREE,
            "tau_rush_day": TAU_RUSH_DAY,
        },
        "n_tests": int(len(tests)),
        "n_remove_final": n_any_remove,
        "nontrivial_edges": nontrivial,
        "honest_fallback_fixed_graph": n_any_remove == 0,
        "graphs_index": {k: {"n_removed": v["n_removed"], "removed_ids": v["removed_edge_ids"]} for k, v in graphs.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="S1 edge gating tests")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "graphs").mkdir(parents=True, exist_ok=True)

    print("[1/4] Loading discretized prescriptions...")
    disc = pd.read_parquet(PHASE2_DIR / "rx_discretized.parquet")
    dates = pd.read_parquet(DATA_DIR / "rx_level.parquet", columns=["处方编号", "日期"])
    disc = disc.merge(dates, on="处方编号", how="left")
    disc = annotate_regimes(disc, disc["日期"])
    disc = add_day_proxies(disc)
    print(f"  n={len(disc):,}, regimes={disc['regime'].value_counts().to_dict()}")

    print("[2/4] Per-cell tests...")
    tests = run_all_tests(disc, seed=args.seed)
    tests_path = OUT_DIR / "edge_gating_tests.csv"
    tests.to_csv(tests_path, index=False, encoding="utf-8-sig")
    print("  saved", tests_path)

    print("[3/4] Building G_c...")
    graphs = build_Gc_edge_lists(tests)
    for ctx, g in graphs.items():
        with (OUT_DIR / "graphs" / f"G_{ctx}.json").open("w", encoding="utf-8") as f:
            json.dump(g, f, ensure_ascii=False, indent=2)

    print("[4/4] Summary...")
    summary = summarize(tests, graphs)
    with (OUT_DIR / "gating_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary["honest_fallback_fixed_graph"]:
        print("\n※ Data does not support edge removal → formal conclusion option: honest fallback to fixed G_sup")
    else:
        print("\n※ Removable edges exist; see nontrivial_edges for non-trivial patterns")


if __name__ == "__main__":
    main()
