"""Business intervention arms ↔ do() + POMIS / POMIS+ filtering and sequence arms."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from phase2.intervention import build_machine_swap, build_shelf_rearrange
from phase2.temporal_scm import InterventionSpec, PharmacyTemporalSCM, build_temporal_diagram

_ROOT = Path(__file__).resolve().parents[1]
_NS = _ROOT / "vendor" / "NS-SCMMAB-main"
if str(_NS) not in sys.path:
    sys.path.insert(0, str(_NS))

from npsem.where_do import POMISs
from npsem.pomis_plus import POMISplusSEQ


# Offline simulation-calibrated relative effects (vs null; magnitude from constrained results)
# DEPRECATED for bandit reward (S4): reward_bonus / d_shift / m_shift
# Kept only for legacy discrete SCM demos; continuous reward path must not use bonus.
# Continuous reward: see phase2/continuous_reward.py — R = -Ŷ_min(do(a))
ARM_EFFECTS = {
    "null": {"d_shift": 0.0, "m_shift": 0.0, "reward_bonus": 0.0, "manipulates": frozenset(), "deprecated_for_reward": True},
    "swap20": {
        "d_shift": 0.35,
        "m_shift": 0.8,
        "reward_bonus": 0.025,
        "manipulates": frozenset({"M", "D"}),
        "deprecated_for_reward": True,
    },
    "swap40": {
        "d_shift": 0.55,
        "m_shift": 1.2,
        "reward_bonus": 0.032,
        "manipulates": frozenset({"M", "D"}),
        "deprecated_for_reward": True,
    },
    "shelf50": {
        "d_shift": 0.25,
        "m_shift": 0.0,
        "reward_bonus": 0.031,
        "manipulates": frozenset({"D"}),
        "deprecated_for_reward": True,
    },
    # Decoy: pretends to do(Load), no layout benefit (filtered by POMIS)
    "load_do": {
        "d_shift": 0.0,
        "m_shift": 0.0,
        "reward_bonus": -0.01,
        "manipulates": frozenset({"Load"}),
        "deprecated_for_reward": True,
    },
}


def build_business_arm_library(items=None) -> dict[str, dict]:
    """Build compliant business-arm metadata (optionally with swap details)."""
    lib = {}
    for name, eff in ARM_EFFECTS.items():
        lib[name] = {
            "name": name,
            **{k: (list(v) if isinstance(v, frozenset) else v) for k, v in eff.items()},
            "start_t_options": [0] if name == "null" else [0, 1],
        }
    if items is not None:
        for n in (20, 40):
            sc = build_machine_swap(
                items,
                n,
                enforce_machine_eligibility=True,
                prefer_remove_ineligible=False,
            )
            key = f"swap{n}"
            if key in lib:
                lib[key]["scenario"] = sc.to_dict()
        shelf = build_shelf_rearrange(items, 50)
        lib["shelf50"]["scenario"] = shelf.to_dict()
    return lib


def intervention_from_arm(arm_id: str) -> InterventionSpec:
    """
    Arm ID encoding:
      null
      swap20 / swap40 / shelf50          → start_t=0 (full day)
      swap20@1 / swap40@1 / shelf50@1    → start_t=1 (effective from rush, POMIS+ sequence)
    """
    if "@" in arm_id:
        base, st = arm_id.split("@", 1)
        start_t = int(st)
    else:
        base, start_t = arm_id, 0
    if base not in ARM_EFFECTS:
        raise KeyError(f"unknown arm base: {base}")
    eff = ARM_EFFECTS[base]
    # Slightly reduce bonus when intervention starts late (misses part of pre slice)
    bonus = eff["reward_bonus"]
    if start_t == 1:
        bonus *= 0.85
    elif start_t >= 2:
        bonus *= 0.5
    return InterventionSpec(
        name=arm_id,
        start_t=start_t,
        d_shift=eff["d_shift"],
        m_shift=eff["m_shift"],
        reward_bonus=bonus,
        meta={"base": base, "manipulates": list(eff["manipulates"])},
    )


def all_brute_arm_ids() -> list[str]:
    """Brute = business arms + timing arms + non-manipulable decoy (do Load, filtered by POMIS)."""
    ids = ["null"]
    for base in ("swap20", "swap40", "shelf50"):
        ids.append(base)
        ids.append(f"{base}@1")
        ids.append(f"{base}@2")  # too late to take effect
    ids += ["load_do", "load_do@1"]
    return ids


def pomis_nodes_for_Y2() -> dict:
    """Compute POMIS on Y2 on the temporal graph; interpret minimal manipulable sets."""
    G = build_temporal_diagram()
    pomis = POMISs(G, "Y2")
    # also myopic per-slice on Yt
    per_t = {t: [sorted(s) for s in POMISs(G, f"Y{t}")] for t in range(3)}
    return {
        "Y2_pomis": [sorted(s) for s in pomis],
        "Y2_pomis_sizes": sorted({len(s) for s in pomis}),
        "per_slice_pomis": per_t,
        "interpretation": (
            "POMIS on Y2 includes interventions on M_t/D_t/W_t ancestors; "
            "Load_t is exogenous (not a layout arm). Business arms only manipulate M/D."
        ),
    }


def pomis_plus_sequences() -> dict:
    """Invoke POMISplusSEQ; return simplified sequence arm set if graph is too large."""
    G = build_temporal_diagram()
    Vs = [{f"{b}{t}" for b in ("Load", "N", "M", "D", "W", "V", "Y")} for t in range(3)]
    Ys = ["Y0", "Y1", "Y2"]
    try:
        import phase2.temporal_scm as _  # noqa: F401
        from npsem.pomis_plus import Sequences

        Sequences.clear()
        seqs = POMISplusSEQ(G=G, Vs=Vs, Ys=Ys, T=2)
        # cap for readability
        shown = [tuple(sorted(map(str, frozenset().union(*s)))) for s in list(seqs)[:50]]
        return {"n_sequences": len(seqs), "sample": shown, "ok": True}
    except Exception as e:
        return {
            "ok": False,
            "error": str(e),
            "fallback_sequences": [
                "null",
                "swap40",  # full-day M/D
                "swap40@1",  # from rush (non-myopic timing)
                "swap20",
                "swap20@1",
                "shelf50",
                "shelf50@1",
            ],
            "note": "POMIS+ timing arms: @1 = intervene starting at morning-rush slice",
        }


def arm_sets() -> dict[str, list[str]]:
    """
    Brute: all business arms + late timing + Load decoy
    POMIS: full-day arms manipulating M/D only
    POMIS+: POMIS ∪ cross-period timing arms effective from rush (@1)
    """
    brute = all_brute_arm_ids()
    pomis = ["null", "swap20", "swap40", "shelf50"]
    pomis_plus = ["null", "swap20", "swap40", "shelf50", "swap20@1", "swap40@1", "shelf50@1"]
    return {"Brute": brute, "POMIS": pomis, "POMIS+": pomis_plus}


def oracle_mu_table(
    scm: PharmacyTemporalSCM,
    arm_ids: list[str],
    *,
    n_mc: int = 300,
    seed: int = 0,
    weekday: bool = True,
) -> dict[str, dict]:
    table = {}
    for i, aid in enumerate(arm_ids):
        interv = intervention_from_arm(aid)
        table[aid] = scm.expected_reward(
            interv, n_mc=n_mc, seed=seed + i * 17, weekday=weekday
        )
    return table


def save_arm_catalog(path: Path, items=None) -> dict:
    catalog = {
        "business_library": build_business_arm_library(items),
        "arm_sets": arm_sets(),
        "pomis_y2": pomis_nodes_for_Y2(),
        "pomis_plus": pomis_plus_sequences(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(catalog, f, ensure_ascii=False, indent=2)
    return catalog


if __name__ == "__main__":
    from phase2.config import DATA_DIR

    items = None
    try:
        import pandas as pd

        items = pd.read_parquet(
            DATA_DIR / "item_level.parquet",
            columns=[
                "处方编号",
                "drugid",
                "药品名称及规格",
                "品名规格",
                "是否机器_最终",
                "调配秒数_最终",
                "货架区域_最终",
            ],
        )
    except Exception:
        pass
    out = DATA_DIR / "phase2" / "ns_arm_catalog.json"
    cat = save_arm_catalog(out, items=items)
    print("arm sets:", {k: len(v) for k, v in cat["arm_sets"].items()})
    print("POMIS Y2 sizes:", cat["pomis_y2"]["Y2_pomis_sizes"])
    print("saved", out)
