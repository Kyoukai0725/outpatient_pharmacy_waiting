"""Feasible interventions: in-machine swaps (capacity-conserving) and near-end shelf rearrangement (less walking)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

MACHINE_SEC = 17.0
DEFAULT_MANUAL_SEC = 30.0  # Default manual seconds after low-frequency machine items exit

# Not machine-eligible: generic name contains keyword, or material code ends with P/S
MACHINE_NAME_BLOCKLIST = ("口服液", "合剂", "注射")
MACHINE_SUFFIX_BLOCKLIST = ("P", "S")


def machine_ineligible(drugid: str, name: str = "") -> tuple[bool, str]:
    """Return (machine-ineligible, reason). Ineligible items already in machine may still be removal candidates."""
    did = str(drugid).strip()
    if did and did[-1] in MACHINE_SUFFIX_BLOCKLIST:
        return True, f"suffix_{did[-1]}"
    text = str(name or "")
    for kw in MACHINE_NAME_BLOCKLIST:
        if kw in text:
            return True, f"name_{kw}"
    return False, ""


@dataclass
class LayoutScenario:
    name: str
    description: str
    kind: str = "baseline"  # baseline | machine_swap | shelf_rearrange
    # machine_swap: manual→machine / machine→manual
    to_machine: set[str] = field(default_factory=set)
    to_manual: set[str] = field(default_factory=set)
    # shelf_rearrange: drugid -> new dispensing seconds
    new_seconds: dict[str, float] = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "kind": self.kind,
            "n_to_machine": len(self.to_machine),
            "n_to_manual": len(self.to_manual),
            "n_shelf_moved": len(self.new_seconds),
            "to_machine": sorted(self.to_machine),
            "to_manual": sorted(self.to_manual),
            "meta": self.meta,
        }


def n_items_to_n0(n_items: int) -> int:
    if n_items <= 1:
        return 0
    if n_items == 2:
        return 1
    if n_items <= 4:
        return 2
    return 3


def counts_to_m0(n_machine: int, n_items: int) -> int:
    if n_machine == 0:
        return 0
    if n_machine == n_items:
        return 3
    ratio = n_machine / n_items
    return 1 if ratio <= 0.5 else 2


def minutes_to_quartile(minutes: float, thresholds: dict) -> int:
    q1, q2, q3 = thresholds["Q1_upper"], thresholds["Q2_upper"], thresholds["Q3_upper"]
    if minutes <= q1:
        return 0
    if minutes <= q2:
        return 1
    if minutes <= q3:
        return 2
    return 3


def drug_stats(items: pd.DataFrame) -> pd.DataFrame:
    name_col = None
    for c in ("药品名称及规格", "品名规格"):
        if c in items.columns:
            name_col = c
            break
    agg = {
        "次数": ("处方编号", "count"),
        "秒数": ("调配秒数_最终", "first"),
        "区域": ("货架区域_最终", "first"),
    }
    if name_col:
        agg["名称"] = (name_col, "first")
    stats = items.groupby(["drugid", "是否机器_最终"], as_index=False).agg(**agg)
    if "名称" not in stats.columns:
        stats["名称"] = ""
    flags = [
        machine_ineligible(d, n) for d, n in zip(stats["drugid"], stats["名称"])
    ]
    stats["禁入机器"] = [f[0] for f in flags]
    stats["禁入原因"] = [f[1] for f in flags]
    return stats


def build_machine_swap(
    items: pd.DataFrame,
    n_swap: int,
    *,
    enforce_machine_eligibility: bool = True,
    prefer_remove_ineligible: bool = True,
) -> LayoutScenario:
    """Low-frequency machine items exit → high-frequency manual items enter, 1:1 slot conservation.

    enforce_machine_eligibility: exclude P/S suffix and name keywords 口服液/合剂/注射 from machine entry.
    prefer_remove_ineligible: prioritize removing ineligible items already in machine (then by ascending frequency).
    """
    stats = drug_stats(items)
    machine = stats[stats["是否机器_最终"]].copy()
    manual = stats[~stats["是否机器_最终"]].copy()

    if prefer_remove_ineligible:
        machine = machine.sort_values(["禁入机器", "次数"], ascending=[False, True])
    else:
        machine = machine.sort_values("次数", ascending=True)

    if enforce_machine_eligibility:
        manual = manual[~manual["禁入机器"]]
    manual = manual.sort_values("次数", ascending=False)

    n_avail_in = int(len(manual))
    n_avail_out = int(len(machine))
    n_eff = min(n_swap, n_avail_in, n_avail_out)
    outs = machine.head(n_eff)
    ins = manual.head(n_eff)

    save_hours = float(((ins["秒数"] - MACHINE_SEC) * ins["次数"]).sum() / 3600)
    cost_hours = float(((DEFAULT_MANUAL_SEC - MACHINE_SEC) * outs["次数"]).sum() / 3600)

    pairs = [
        {
            "out_drugid": str(o.drugid),
            "out_name": str(o.名称),
            "out_freq": int(o.次数),
            "out_ineligible": bool(o.禁入机器),
            "out_reason": str(o.禁入原因),
            "in_drugid": str(i.drugid),
            "in_name": str(i.名称),
            "in_freq": int(i.次数),
            "in_sec": float(i.秒数),
            "pair_net_hours": round(
                float((i.秒数 - MACHINE_SEC) * i.次数 / 3600)
                - float((DEFAULT_MANUAL_SEC - MACHINE_SEC) * o.次数 / 3600),
                2,
            ),
        }
        for o, i in zip(outs.itertuples(), ins.itertuples())
    ]

    tag = "constrained_" if enforce_machine_eligibility else ""
    return LayoutScenario(
        name=f"{tag}machine_swap_{n_eff}",
        description=(
            f"In-machine swap {n_eff} slots (ineligible: P/S suffix, name contains 口服液/合剂/注射): "
            f"low-frequency/ineligible machine items out, high-frequency eligible manual items in"
        ),
        kind="machine_swap",
        to_machine=set(ins["drugid"].astype(str)),
        to_manual=set(outs["drugid"].astype(str)),
        meta={
            "n_swap": n_eff,
            "n_swap_requested": n_swap,
            "enforce_machine_eligibility": enforce_machine_eligibility,
            "prefer_remove_ineligible": prefer_remove_ineligible,
            "n_eligible_manual": n_avail_in,
            "n_machine": n_avail_out,
            "expected_save_hours": round(save_hours, 2),
            "expected_cost_hours": round(cost_hours, 2),
            "expected_net_hours": round(save_hours - cost_hours, 2),
            "out_freq_max": int(outs["次数"].max()) if len(outs) else 0,
            "in_freq_min": int(ins["次数"].min()) if len(ins) else 0,
            "n_outs_ineligible": int(outs["禁入机器"].sum()) if len(outs) else 0,
            "pairs": pairs,
        },
    )


def zone_mean_seconds(items: pd.DataFrame) -> pd.Series:
    man = items[~items["是否机器_最终"]]
    return man.groupby("货架区域_最终")["调配秒数_最终"].mean()


def build_shelf_rearrange(items: pd.DataFrame, n_pairs: int) -> LayoutScenario:
    """
    Slow-zone high-frequency ↔ fast-zone low-frequency: swap bin locations, approximating walk-time change via zone means.
    Machine items are not moved.
    """
    man = items[~items["是否机器_最终"]].copy()
    zmean = zone_mean_seconds(items).dropna()
    if len(zmean) < 4:
        return LayoutScenario("shelf_rearrange_0", "Shelf rearrange (insufficient data)", kind="shelf_rearrange")

    # Zones with enough items
    zone_n = man.groupby("货架区域_最终")["drugid"].nunique()
    eligible = zmean[zone_n.reindex(zmean.index).fillna(0) >= 5]
    fast_zones = set(eligible.nsmallest(max(3, len(eligible) // 4)).index)
    slow_zones = set(eligible.nlargest(max(3, len(eligible) // 4)).index)

    drug_zone = (
        man.groupby("drugid")
        .agg(次数=("处方编号", "count"), 秒数=("调配秒数_最终", "first"), 区域=("货架区域_最终", "first"))
        .reset_index()
    )
    high_slow = drug_zone[drug_zone["区域"].isin(slow_zones)].sort_values("次数", ascending=False)
    low_fast = drug_zone[drug_zone["区域"].isin(fast_zones)].sort_values("次数", ascending=True)

    n = min(n_pairs, len(high_slow), len(low_fast))
    movers = high_slow.head(n)
    vacate = low_fast.head(n)

    new_seconds: dict[str, float] = {}
    # High-frequency slow-zone drugs → fast-zone mean seconds
    fast_sec = float(eligible[eligible.index.isin(fast_zones)].mean())
    slow_sec = float(eligible[eligible.index.isin(slow_zones)].mean())
    for _, row in movers.iterrows():
        # Keep item-specific relative difference but shift overall to fast-zone level
        delta = fast_sec - float(row["秒数"])
        # Only allow faster, and not much below fast-zone mean
        new_seconds[str(row["drugid"])] = max(20.0, min(float(row["秒数"]), fast_sec))
    for _, row in vacate.iterrows():
        new_seconds[str(row["drugid"])] = max(float(row["秒数"]), slow_sec)

    expected_save = 0.0
    for _, row in movers.iterrows():
        expected_save += (float(row["秒数"]) - new_seconds[str(row["drugid"])]) * row["次数"]
    expected_cost = 0.0
    for _, row in vacate.iterrows():
        expected_cost += (new_seconds[str(row["drugid"])] - float(row["秒数"])) * row["次数"]

    return LayoutScenario(
        name=f"shelf_near_{n_pairs}",
        description=f"Near-end shelf rearrange {n} pairs: slow-zone high-freq ↔ fast-zone low-freq (less walking)",
        kind="shelf_rearrange",
        new_seconds=new_seconds,
        meta={
            "n_pairs": n,
            "fast_zones": sorted(fast_zones),
            "slow_zones": sorted(slow_zones),
            "fast_zone_mean_sec": round(fast_sec, 2),
            "slow_zone_mean_sec": round(slow_sec, 2),
            "expected_save_hours": round(expected_save / 3600, 2),
            "expected_cost_hours": round(expected_cost / 3600, 2),
            "expected_net_hours": round((expected_save - expected_cost) / 3600, 2),
            "moved_high_freq": movers["drugid"].astype(str).tolist(),
            "moved_low_freq": vacate["drugid"].astype(str).tolist(),
        },
    )


def build_feasible_scenarios(
    items: pd.DataFrame,
    *,
    constrained_machine: bool = True,
) -> list[LayoutScenario]:
    # Remove from machine by ascending frequency (no ineligible priority) to maximize wait-time improvement; entry still strictly compliant
    kw = dict(
        enforce_machine_eligibility=constrained_machine,
        prefer_remove_ineligible=False,
    )
    return [
        LayoutScenario("baseline", "Current layout (no intervention)", kind="baseline"),
        build_machine_swap(items, 20, **kw),
        build_machine_swap(items, 30, **kw),
        build_machine_swap(items, 40, **kw),
        build_shelf_rearrange(items, 20),
        build_shelf_rearrange(items, 50),
    ]


def recompute_rx_under_scenario(items: pd.DataFrame, scenario: LayoutScenario) -> dict:
    """Recompute dispensing features for one prescription under a scenario."""
    is_machine = items["是否机器_最终"].astype(bool).copy()
    seconds = items["调配秒数_最终"].astype(float).copy()
    drugids = items["drugid"].astype(str)

    changed = 0

    if scenario.kind == "machine_swap":
        to_m = drugids.isin(scenario.to_machine) & (~is_machine)
        to_a = drugids.isin(scenario.to_manual) & is_machine
        is_machine.loc[to_m] = True
        seconds.loc[to_m] = MACHINE_SEC
        is_machine.loc[to_a] = False
        seconds.loc[to_a] = DEFAULT_MANUAL_SEC
        changed = int(to_m.sum() + to_a.sum())

    elif scenario.kind == "shelf_rearrange":
        for idx, did in drugids.items():
            if did in scenario.new_seconds and not bool(is_machine.loc[idx]):
                old = float(seconds.loc[idx])
                new = float(scenario.new_seconds[did])
                if abs(old - new) > 1e-6:
                    seconds.loc[idx] = new
                    changed += 1

    n_items = len(items)
    n_machine = int(is_machine.sum())
    ratio = n_machine / n_items if n_items else 0.0
    if n_machine == 0:
        disp_type = "纯人工"
    elif n_machine == n_items:
        disp_type = "纯机器"
    else:
        disp_type = "混合"

    return {
        "预估配药_分钟": float(seconds.sum()) / 60.0,
        "机器品项占比": ratio,
        "调配方式": disp_type,
        "机器品项数": n_machine,
        "N0": n_items_to_n0(n_items),
        "M0": counts_to_m0(n_machine, n_items),
        "moved_count": changed,
    }


Y0_PARENTS = ("W0", "D0", "V0")


def _y_probs(cpts: dict, row: pd.Series, w0: int, d0: int) -> np.ndarray:
    state = {
        "Peak0": int(row["Peak0"]),
        "Load0": int(row["Load0"]),
        "N0": int(row["N0"]),
        "M0": int(row["M0"]),
        "W0": int(w0),
        "D0": int(d0),
        "V0": int(row["V0"]),
    }
    key = "|".join(str(state[p]) for p in Y0_PARENTS)
    return np.asarray(cpts["Y0"][key], dtype=float)


def expected_y(cpts: dict, row: pd.Series, w0: int, d0: int) -> float:
    py = _y_probs(cpts, row, w0=w0, d0=d0)
    return float(np.dot(np.arange(len(py)), py))


def prob_long_wait(cpts: dict, row: pd.Series, w0: int, d0: int) -> float:
    py = _y_probs(cpts, row, w0=w0, d0=d0)
    return float(py[3]) if len(py) > 3 else 0.0


def _summarize_intervention(df: pd.DataFrame, system_disp_saved: float) -> dict:
    return {
        "n_prescriptions": int(len(df)),
        "system_disp_saved_min": round(system_disp_saved, 4),
        "mean_disp_min_base": round(float(df["disp_min_base"].mean()), 4),
        "mean_disp_min_cf": round(float(df["disp_min_cf"].mean()), 4),
        "disp_min_saved": round(float((df["disp_min_base"] - df["disp_min_cf"]).mean()), 4),
        "mean_W0_base": round(float(df["W0_base"].mean()), 4),
        "mean_W0_cf": round(float(df["W0_cf"].mean()), 4),
        "mean_w_min_saved": round(float((df["w_min_base"] - df["w_min_cf"]).mean()), 4),
        "W0_downgraded_rate": round(float((df["W0_cf"] < df["W0_base"]).mean()), 4),
        "mean_D0_base": round(float(df["D0_base"].mean()), 4),
        "mean_D0_cf": round(float(df["D0_cf"].mean()), 4),
        "mean_M0_base": round(float(df["M0_base"].mean()), 4),
        "mean_M0_cf": round(float(df["M0_cf"].mean()), 4),
        "mean_EY_base": round(float(df["EY_base"].mean()), 4),
        "mean_EY_cf": round(float(df["EY_cf"].mean()), 4),
        "EY_reduction": round(float((df["EY_base"] - df["EY_cf"]).mean()), 4),
        "P_Y3_base": round(float(df["P_Y3_base"].mean()), 4),
        "P_Y3_cf": round(float(df["P_Y3_cf"].mean()), 4),
        "P_Y3_reduction": round(float((df["P_Y3_base"] - df["P_Y3_cf"]).mean()), 4),
        "D0_downgraded_rate": round(float((df["D0_cf"] < df["D0_base"]).mean()), 4),
        "M0_upgraded_rate": round(float((df["M0_cf"] > df["M0_base"]).mean()), 4),
        "mean_moved_items": round(float(df["moved_items"].mean()), 4),
    }


def _scenario_touches(scenario: LayoutScenario) -> set[str]:
    return set(scenario.to_machine) | set(scenario.to_manual) | set(scenario.new_seconds)


def run_intervention_study(
    cpts: dict,
    rx_disc: pd.DataFrame,
    items: pd.DataFrame,
    scenarios: list[LayoutScenario],
    d0_thresholds: dict,
    w0_thresholds: dict,
    queue_model: dict,
    sample_n: int = 30000,
    seed: int = 42,
    chosen_ids: np.ndarray | None = None,
) -> dict:
    from phase2.queue_propagation import propagate_call_wait_minutes

    rng = np.random.default_rng(seed)
    # Evaluate full sample (including pure-machine Rx, since machine swaps affect them too)
    all_ids = rx_disc["处方编号"].unique()
    if chosen_ids is None:
        if len(all_ids) > sample_n:
            chosen = rng.choice(all_ids, size=sample_n, replace=False)
        else:
            chosen = all_ids
    else:
        chosen = chosen_ids

    rx_sub = rx_disc[rx_disc["处方编号"].isin(chosen)].copy()
    rx_index = rx_sub.set_index("处方编号", drop=False)
    item_sub = items[items["处方编号"].isin(chosen)]
    grouped = {k: g for k, g in item_sub.groupby("处方编号", sort=False)}
    rx_drugs: dict[int, set[str]] = {
        rid: set(g["drugid"].astype(str)) for rid, g in grouped.items()
    }

    results = {}
    for scenario in scenarios:
        touches = _scenario_touches(scenario)

        cf_cache: dict[int, dict] = {}
        disp_saved_all: list[float] = []
        for rx_id in chosen:
            if rx_id not in grouped:
                continue
            base_min = float(rx_index.loc[rx_id, "预估配药_分钟"])
            cf = recompute_rx_under_scenario(grouped[rx_id], scenario)
            cf_cache[rx_id] = cf
            disp_saved_all.append(base_min - cf["预估配药_分钟"])
        system_disp_saved = float(np.mean(disp_saved_all)) if disp_saved_all else 0.0

        if scenario.kind == "baseline":
            target_ids = list(chosen)
        else:
            target_ids = [rid for rid in chosen if rid in rx_drugs and rx_drugs[rid] & touches]

        records_fixed: list[dict] = []
        records_queue: list[dict] = []

        for rx_id in target_ids:
            if rx_id not in grouped or rx_id not in cf_cache:
                continue
            rx_row = rx_index.loc[rx_id]
            cf = cf_cache[rx_id]
            if scenario.kind != "baseline" and cf["moved_count"] == 0:
                continue

            w0_base = int(rx_row["W0"])
            d0_base = int(rx_row["D0"])
            d0_cf = minutes_to_quartile(cf["预估配药_分钟"], d0_thresholds)

            w_base_min = float(rx_row["叫号等待_分钟"])
            capacity = None
            if "产能代理" in rx_row.index and pd.notna(rx_row["产能代理"]):
                capacity = float(rx_row["产能代理"])
            w_cf_min = propagate_call_wait_minutes(
                w_base_min,
                system_disp_saved,
                int(rx_row["Peak0"]),
                int(rx_row["Load0"]),
                queue_model,
                capacity=capacity,
            )
            w0_cf = minutes_to_quartile(w_cf_min, w0_thresholds)

            base_common = {
                "D0_base": d0_base,
                "D0_cf": d0_cf,
                "M0_base": int(rx_row["M0"]),
                "M0_cf": int(cf["M0"]),
                "disp_min_base": float(rx_row["预估配药_分钟"]),
                "disp_min_cf": cf["预估配药_分钟"],
                "moved_items": cf["moved_count"],
                "W0_base": w0_base,
                "w_min_base": w_base_min,
            }

            records_fixed.append(
                {
                    **base_common,
                    "W0_cf": w0_base,
                    "w_min_cf": w_base_min,
                    "EY_base": expected_y(cpts, rx_row, w0_base, d0_base),
                    "EY_cf": expected_y(cpts, rx_row, w0_base, d0_cf),
                    "P_Y3_base": prob_long_wait(cpts, rx_row, w0_base, d0_base),
                    "P_Y3_cf": prob_long_wait(cpts, rx_row, w0_base, d0_cf),
                }
            )
            records_queue.append(
                {
                    **base_common,
                    "W0_cf": w0_cf,
                    "w_min_cf": w_cf_min,
                    "EY_base": expected_y(cpts, rx_row, w0_base, d0_base),
                    "EY_cf": expected_y(cpts, rx_row, w0_cf, d0_cf),
                    "P_Y3_base": prob_long_wait(cpts, rx_row, w0_base, d0_base),
                    "P_Y3_cf": prob_long_wait(cpts, rx_row, w0_cf, d0_cf),
                }
            )

        df_fixed = pd.DataFrame(records_fixed)
        df_queue = pd.DataFrame(records_queue)
        if df_fixed.empty:
            results[scenario.name] = {"error": "no affected prescriptions"}
            continue

        results[scenario.name] = {
            "scenario": scenario.to_dict(),
            "fixed_w": _summarize_intervention(df_fixed, system_disp_saved),
            "with_queue": _summarize_intervention(df_queue, system_disp_saved),
        }

    return results


def save_results(results: dict, path) -> None:
    import json
    from pathlib import Path

    path = Path(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


# Backward-compatible import alias
def build_default_scenarios(items: pd.DataFrame) -> list[LayoutScenario]:
    return build_feasible_scenarios(items)
