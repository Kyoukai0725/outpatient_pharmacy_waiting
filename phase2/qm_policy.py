"""Queue-Mediated Policy (QM-Policy): identification framework, not a new algorithm/system.

Claim (portable to registration, lab, OR, and other capacity-constrained batch scheduling):
  intervention value = direct service channel + queue channel; under capacity constraints the queue channel often dominates.

Inputs: any layout/service-time intervention (LayoutScenario or drug→seconds map)
        + outcome model (default Ridge) + queue propagator (default heuristic propagate)
Outputs: nested path decomposition + Shapley (main-text mechanism) + affected-subset HTE + day-level/cluster bootstrap CI

Block-Reoptimize: deployment protocol under nonstationarity (periodic re-estimation), attached to this framework, not a separate mainline.

Identification disclaimer: model-based; not randomized, not classic OPE.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from phase2.config import DATA_DIR
from phase2.continuous_reward import (
    WaitRegressorBundle,
    _scenario_drug_ids,
    arm_to_scenario,
    day_rx_predictions,
    load_bundle,
    load_queue_model,
    rx_ids_touching_drugs,
)
from phase2.effect_decomposition import _load_frames, _mean_yhat, decompose_day
from phase2.intervention import LayoutScenario
from phase2.ns_mechanisms import cal_block
from phase2.policy_bootstrap import cluster_bootstrap_ci, day_bootstrap_ci
from phase2.shapley_paths import shapley_day

PHASE2_DIR = DATA_DIR / "phase2"
OUT_DIR = PHASE2_DIR / "qm_policy"

IDENTIFICATION = (
    "Queue-Mediated Policy is an identification framework that decomposes "
    "model-based intervention value into a direct service channel and a queue "
    "channel under capacity-constrained batch scheduling. "
    "It is NOT a new learning algorithm or production system. "
    "Estimates are model-based (outcome regressor + queue propagator), "
    "not randomized ATEs and not IPS/DR OPE."
)


# ---------------------------------------------------------------------------
# Intervention input
# ---------------------------------------------------------------------------


def intervention_from_arm(arm_id: str, items_all: pd.DataFrame) -> LayoutScenario | None:
    """Pharmacy convenience: business arm name → LayoutScenario."""
    return arm_to_scenario(arm_id, items_all)


def intervention_from_service_seconds(
    name: str,
    new_seconds: dict[str, float],
    description: str = "",
) -> LayoutScenario:
    """Generic input: unit id → new service seconds (analogous for lab stations, registration windows, etc.)."""
    return LayoutScenario(
        name=name,
        description=description or f"service-time map ({len(new_seconds)} units)",
        kind="service_time_map",
        new_seconds={str(k): float(v) for k, v in new_seconds.items()},
        meta={"source": "service_seconds_map", "n_units": len(new_seconds)},
    )


# ---------------------------------------------------------------------------
# Result schema
# ---------------------------------------------------------------------------


@dataclass
class ChannelShares:
    delta_direct: float
    delta_queue: float
    delta_total: float
    share_direct: float | None
    share_queue: float | None


@dataclass
class QMPolicyReport:
    """Standard framework output (main-text mechanism tables can use directly)."""

    identification: str
    intervention_name: str
    n_days: int
    nested: dict[str, Any]
    shapley: dict[str, Any]
    hte_strata: dict[str, Any]
    policy_interval: dict[str, Any]
    queue_dominance: dict[str, Any]
    deployment_protocol: dict[str, Any] | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Core framework
# ---------------------------------------------------------------------------


class QueueMediatedPolicy:
    """
    Reusable analysis framework:
      analyze(intervention, data) → channel decomposition + affected HTE + policy interval

    outcome_predict: (bundle, rx_day, items_day, scenario, **kwargs) → pred frame
    Default binds pharmacy Ridge + propagate; other settings can inject a similar interface.
    """

    def __init__(
        self,
        bundle: WaitRegressorBundle,
        queue_model: dict,
        *,
        claim: str | None = None,
    ):
        self.bundle = bundle
        self.queue_model = queue_model
        self.claim = claim or (
            "Under capacity-constrained batch scheduling, intervention value "
            "decomposes into a direct service channel and a queue channel; "
            "the queue channel often dominates."
        )

    def _four_layers(
        self,
        rx_day: pd.DataFrame,
        items_day: pd.DataFrame,
        scenario: LayoutScenario | None,
        restrict: set | None,
    ) -> dict[str, pd.DataFrame]:
        """A0, A1(direct), A_Q(queue-only), A2(both)."""
        kw = dict(queue_model=self.queue_model, restrict_rx_ids=restrict)
        return {
            "A0": day_rx_predictions(self.bundle, rx_day, items_day, None, **kw),
            "A1": day_rx_predictions(
                self.bundle,
                rx_day,
                items_day,
                scenario,
                use_queue_propagation=False,
                apply_layout_features=True,
                **kw,
            ),
            "AQ": day_rx_predictions(
                self.bundle,
                rx_day,
                items_day,
                scenario,
                use_queue_propagation=True,
                apply_layout_features=False,
                **kw,
            ),
            "A2": day_rx_predictions(
                self.bundle,
                rx_day,
                items_day,
                scenario,
                use_queue_propagation=True,
                apply_layout_features=True,
                **kw,
            ),
        }

    @staticmethod
    def _shares(direct: float, queue: float) -> ChannelShares:
        tot = direct + queue
        return ChannelShares(
            delta_direct=round(direct, 6),
            delta_queue=round(queue, 6),
            delta_total=round(tot, 6),
            share_direct=round(direct / tot, 4) if abs(tot) > 1e-9 else None,
            share_queue=round(queue / tot, 4) if abs(tot) > 1e-9 else None,
        )

    def analyze_day(
        self,
        rx_day: pd.DataFrame,
        items_day: pd.DataFrame,
        intervention: LayoutScenario | None,
        *,
        affected_only: bool = False,
    ) -> dict:
        """Single day: nested + Shapley + affected flags."""
        restrict = None
        drugs = _scenario_drug_ids(intervention)
        if affected_only:
            restrict = rx_ids_touching_drugs(items_day, drugs)
            if not restrict or len(restrict) < 3:
                return {}

        nested = decompose_day(
            self.bundle,
            rx_day,
            items_day,
            intervention,
            self.queue_model,
            affected_only=affected_only,
        )
        shap = shapley_day(
            self.bundle,
            rx_day,
            items_day,
            intervention,
            self.queue_model,
            restrict=restrict,
        )
        if not nested or not shap:
            return {}

        # Prescription-level τ (full-day A2) for affected/rush stratification
        layers = self._four_layers(rx_day, items_day, intervention, restrict=None)
        a0, a2 = layers["A0"], layers["A2"]
        if a0.empty or a2.empty:
            return {"nested": nested, "shapley": shap}

        m = a0[["处方编号", "yhat", "MorningRush"]].rename(columns={"yhat": "y0"}).merge(
            a2[["处方编号", "yhat"]].rename(columns={"yhat": "y2"}), on="处方编号"
        )
        touched = rx_ids_touching_drugs(items_day, drugs)
        m["touched"] = m["处方编号"].isin(touched).astype(int)
        m["tau"] = m["y0"] - m["y2"]
        rush = m["MorningRush"].astype(int) == 1 if "MorningRush" in m.columns else None

        def mean_tau(mask: pd.Series | None = None) -> dict:
            sub = m if mask is None else m[mask]
            if sub.empty:
                return {"n": 0, "mean_tau": None}
            return {
                "n": int(len(sub)),
                "mean_tau": round(float(sub["tau"].mean()), 6),
            }

        hte = {
            "overall": mean_tau(),
            "touched": mean_tau(m["touched"] == 1),
            "untouched": mean_tau(m["touched"] == 0),
            "rush": mean_tau(rush) if rush is not None else None,
            "touched_and_rush": mean_tau((m["touched"] == 1) & rush)
            if rush is not None
            else None,
        }
        return {
            "nested": nested,
            "shapley": shap,
            "hte": hte,
            "delta_total_day": shap["delta_total"],
        }

    def analyze(
        self,
        intervention: LayoutScenario | None,
        rx: pd.DataFrame,
        items: pd.DataFrame,
        *,
        sample_days: int = 60,
        seed: int = 42,
        intervention_name: str | None = None,
        n_boot: int = 2000,
        include_deployment: bool = False,
        deployment_fn: Callable[[], dict] | None = None,
    ) -> QMPolicyReport:
        """
        Main framework entry point.

        Parameters
        ----------
        intervention : layout/service-time intervention (domain-agnostic input slot)
        rx, items : unit-level and aggregate observations (pharmacy = prescription/items)
        """
        name = intervention_name or (
            intervention.name if intervention is not None else "null"
        )
        rx = rx.copy()
        rx["日期_d"] = pd.to_datetime(rx["日期"]).dt.date
        items = items.copy()
        items["日期_d"] = items["处方编号"].map(rx.set_index("处方编号")["日期_d"])

        days = sorted(rx["日期_d"].dropna().unique())
        rng = np.random.default_rng(seed)
        if len(days) > sample_days:
            days = sorted(rng.choice(days, size=sample_days, replace=False))

        daily = []
        hte_acc = []
        for i, d in enumerate(days):
            rx_d = rx[rx["日期_d"] == d]
            it_d = items[items["日期_d"] == d]
            if len(rx_d) < 30 or len(it_d) == 0:
                continue
            one = self.analyze_day(rx_d, it_d, intervention, affected_only=False)
            if not one:
                continue
            sh = one["shapley"]
            nest = one["nested"]["all"]
            daily.append(
                {
                    "date": str(d),
                    "CalBlock": cal_block(pd.Timestamp(str(d))),
                    "delta": sh["delta_total"],
                    "nested_direct": nest["delta_disp"],
                    "nested_queue": nest["delta_queue"],
                    "shapley_direct": sh["shapley_disp"],
                    "shapley_queue": sh["shapley_queue"],
                    "n_rx": sh["n_rx"],
                }
            )
            h = one["hte"]
            h["date"] = str(d)
            hte_acc.append(h)
            if (i + 1) % 15 == 0:
                print(f"  QM-Policy days {i+1}/{len(days)}")

        df = pd.DataFrame(daily)
        if df.empty:
            raise RuntimeError("QM-Policy: no valid days")

        def agg_mean(col: str) -> dict:
            v = df[col].astype(float)
            return {
                "mean": round(float(v.mean()), 6),
                "se": round(float(v.std(ddof=1) / np.sqrt(len(v))), 6)
                if len(v) > 1
                else 0.0,
                "n": int(len(v)),
            }

        nested_block = {
            "direct": agg_mean("nested_direct"),
            "queue": agg_mean("nested_queue"),
            "total": {
                "mean": round(
                    float(df["nested_direct"].mean() + df["nested_queue"].mean()), 6
                )
            },
            "share_queue": round(
                float(
                    df["nested_queue"].mean()
                    / max(df["nested_direct"].mean() + df["nested_queue"].mean(), 1e-9)
                ),
                4,
            ),
        }
        shapley_block = {
            "direct": agg_mean("shapley_direct"),
            "queue": agg_mean("shapley_queue"),
            "total": agg_mean("delta"),
            "share_queue": round(
                float(
                    df["shapley_queue"].mean()
                    / max(df["shapley_direct"].mean() + df["shapley_queue"].mean(), 1e-9)
                ),
                4,
            ),
        }

        # Pool affected HTE (daily mean of pooled prescription means across days)
        def pool_stratum(key: str) -> dict:
            vals, ns = [], []
            for h in hte_acc:
                s = h.get(key)
                if s and s.get("mean_tau") is not None and s.get("n", 0) > 0:
                    vals.append(float(s["mean_tau"]))
                    ns.append(int(s["n"]))
            if not vals:
                return {"n_day_means": 0}
            return {
                "n_day_means": len(vals),
                "mean_of_day_mean_tau": round(float(np.mean(vals)), 6),
                "mean_n_rx_per_day": round(float(np.mean(ns)), 1),
            }

        hte_strata = {
            k: pool_stratum(k)
            for k in ("overall", "touched", "untouched", "rush", "touched_and_rush")
        }

        boot_df = df[["date", "CalBlock", "delta"]].copy()
        cluster = cluster_bootstrap_ci(boot_df, n_boot=n_boot, seed=seed)
        day_b = day_bootstrap_ci(boot_df, n_boot=n_boot, seed=seed)
        policy_interval = {
            "point_mean_delta": cluster["point_mean_delta_min"],
            "cluster_bootstrap": cluster["bootstrap"],
            "day_bootstrap": day_b,
            "block_mean_delta": cluster["block_mean_delta"],
            "note": (
                "Dual intervals: cluster (CalBlock) vs iid day; "
                "sensitivity to aggregation granularity is itself a finding"
            ),
        }

        qdom = {
            "claim": self.claim,
            "nested_share_queue": nested_block["share_queue"],
            "shapley_share_queue": shapley_block["share_queue"],
            "agreement": abs(nested_block["share_queue"] - shapley_block["share_queue"])
            < 0.02,
            "interpretation": (
                "If share_queue ≫ share_direct, layout/service redesign value is "
                "realized primarily via congestion relief, not unit service seconds."
            ),
        }

        deploy = None
        if include_deployment and deployment_fn is not None:
            deploy = {
                "role": (
                    "Deployment protocol under nonstationarity — natural extension "
                    "of QM-Policy, not a separate methodological mainline"
                ),
                "result": deployment_fn(),
            }

        return QMPolicyReport(
            identification=IDENTIFICATION,
            intervention_name=name,
            n_days=int(len(df)),
            nested=nested_block,
            shapley=shapley_block,
            hte_strata=hte_strata,
            policy_interval=policy_interval,
            queue_dominance=qdom,
            deployment_protocol=deploy,
            meta={
                "sample_days_requested": sample_days,
                "channels": {
                    "direct_service": "A1: change service features only (W fixed)",
                    "queue": "A_Q / nested residual: congestion channel via propagator",
                    "both": "A2",
                },
                "daily": daily,
            },
        )


# ---------------------------------------------------------------------------
# Deployment protocol (attached, not a mainline)
# ---------------------------------------------------------------------------


def block_reoptimize_protocol_summary() -> dict:
    """Load rolling_swap results and wrap as deployment protocol summary."""
    path = PHASE2_DIR / "rolling_swap" / "rolling_swap_summary.json"
    if not path.exists():
        return {
            "status": "not_run",
            "hint": "python3 -m phase2.rolling_swap",
        }
    with path.open(encoding="utf-8") as f:
        raw = json.load(f)
    decisions = []
    for d in raw.get("decisions", []):
        decisions.append(
            {
                "decide": d["decide_block"],
                "eval": d["eval_block"],
                "slot_change_vs_frozen": d["vs_static"]["all_slots"]["change_rate"],
                "mu_gap_roll_minus_frozen": d["rolling_minus_static_delta"],
            }
        )
    return {
        "name": "Block-Reoptimize",
        "static_frozen_at": raw.get("static_built_on"),
        "decisions": decisions,
        "drift": raw.get("decision_to_decision_drift"),
        "limitation": (
            "Negative deltas in some blocks (e.g. 2025H2) are retained as evidence "
            "of nonstationary environment; do not explain them away"
        ),
    }


# ---------------------------------------------------------------------------
# CLI (engineering entry point, not an academic contribution)
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="QM-Policy identification framework (not a new algorithm)"
    )
    parser.add_argument("--arm", default="swap40", help="pharmacy convenience arm id")
    parser.add_argument("--sample-days", type=int, default=60)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default="ridge")
    parser.add_argument(
        "--with-deployment",
        action="store_true",
        help="attach Block-Reoptimize protocol summary if available",
    )
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("[1] load frames + models...")
    rx, items, _ = _load_frames()
    bundle = load_bundle(args.model)
    qm = load_queue_model()
    intervention = intervention_from_arm(args.arm, items)

    framework = QueueMediatedPolicy(bundle, qm)
    print(f"[2] analyze intervention={args.arm} ...")
    report = framework.analyze(
        intervention,
        rx,
        items,
        sample_days=args.sample_days,
        seed=args.seed,
        intervention_name=args.arm,
        n_boot=args.n_boot,
        include_deployment=args.with_deployment,
        deployment_fn=block_reoptimize_protocol_summary if args.with_deployment else None,
    )

    out = report.to_dict()
    # Save daily detail separately; keep main JSON readable
    daily = out["meta"].pop("daily", [])
    with (OUT_DIR / f"qm_report_{args.arm}.json").open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    pd.DataFrame(daily).to_csv(OUT_DIR / f"qm_daily_{args.arm}.csv", index=False)

    print("\n=== QM-Policy (identification framework) ===")
    print(IDENTIFICATION)
    print(f"  intervention: {report.intervention_name}  n_days={report.n_days}")
    print(
        f"  nested:  direct={report.nested['direct']['mean']:.4f}  "
        f"queue={report.nested['queue']['mean']:.4f}  "
        f"share_queue={report.nested['share_queue']}"
    )
    print(
        f"  shapley: direct={report.shapley['direct']['mean']:.4f}  "
        f"queue={report.shapley['queue']['mean']:.4f}  "
        f"share_queue={report.shapley['share_queue']}"
    )
    pi = report.policy_interval
    print(
        f"  Δ={pi['point_mean_delta']:.4f}  "
        f"cluster CI [{pi['cluster_bootstrap']['ci_low']:.4f}, "
        f"{pi['cluster_bootstrap']['ci_high']:.4f}]  "
        f"day CI [{pi['day_bootstrap']['ci_low']:.4f}, {pi['day_bootstrap']['ci_high']:.4f}]"
    )
    for k, v in report.hte_strata.items():
        if v.get("mean_of_day_mean_tau") is not None:
            print(f"  HTE {k}: τ≈{v['mean_of_day_mean_tau']:.4f}")
    if report.deployment_protocol:
        print("  deployment:", report.deployment_protocol.get("role", "")[:80])
    print("done →", OUT_DIR)


if __name__ == "__main__":
    main()
