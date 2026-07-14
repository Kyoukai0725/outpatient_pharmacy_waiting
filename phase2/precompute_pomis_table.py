"""S2: Offline precompute POMIS+ per G_c → business arm table (lookup only; no online recomputation)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from phase2.config import DATA_DIR
from phase2.ns_arms import ARM_EFFECTS, all_brute_arm_ids, arm_sets
from phase2.temporal_scm import T_SLICES, build_temporal_diagram

_ROOT = Path(__file__).resolve().parents[1]
_NS = _ROOT / "vendor" / "NS-SCMMAB-main"
if str(_NS) not in sys.path:
    sys.path.insert(0, str(_NS))

from npsem.where_do import POMISs
from npsem.model import CD

PHASE2_DIR = DATA_DIR / "phase2"
GRAPH_DIR = PHASE2_DIR / "contextual_graph" / "graphs"
OUT_PATH = PHASE2_DIR / "contextual_graph" / "pomis_plus_table.json"
OUT_PATH_STRICT = PHASE2_DIR / "contextual_graph" / "pomis_plus_table_strict.json"
OUT_PATH_PROJECTED = PHASE2_DIR / "contextual_graph" / "pomis_plus_table_projected.json"

# Layout-manipulable node prefixes (business arms map only to these)
MANIPULABLE_PREFIXES = ("M", "D")


def load_Gc(context: str):
    path = GRAPH_DIR / f"G_{context}.json"
    with path.open(encoding="utf-8") as f:
        meta = json.load(f)
    edges = [tuple(e) for e in meta["kept_edges"]]
    vs = set()
    for u, v in edges:
        vs.add(u)
        vs.add(v)
    # Ensure Y_t nodes are in the graph
    for t in range(T_SLICES):
        vs.add(f"Y{t}")
    G = CD(vs, edges)
    return G, meta


def pomis_nodes_for_Y2(G) -> list[list[str]]:
    pomis = POMISs(G, "Y2")
    return [sorted(s) for s in pomis]


def set_is_layout_feasible(node_set: set[str]) -> bool:
    """Intervention set is empty or every node is M*/D* (layout-feasible); empty set allowed (=null)."""
    if not node_set:
        return True
    for x in node_set:
        if not x.startswith(MANIPULABLE_PREFIXES):
            return False
    return True


def involves_timing_from_rush(node_set: set[str]) -> bool:
    """Map to @1 timing arm if intervention affects only M/D at t≥1."""
    if not node_set:
        return False
    times = []
    for x in node_set:
        if x[-1].isdigit():
            times.append(int(x[-1]))
    if not times:
        return False
    return min(times) >= 1 and max(times) >= 1


def project_pomis_to_manipulable(pomis_sets: list[list[str]]) -> list[list[str]]:
    """Layout projection: keep only M*/D* nodes in each POMIS set (drop V/W/Load etc.)."""
    out: list[list[str]] = []
    for s in pomis_sets:
        proj = sorted(x for x in s if x.startswith(MANIPULABLE_PREFIXES))
        out.append(proj)
    return out


def map_projected_pomis_to_business_arms(pomis_sets: list[list[str]]) -> list[str]:
    """
    Honest mapping after projection (no "fill full-day arms when timing"):
      - empty set → null
      - non-empty with t≥1 → null + *@1
      - non-empty including t=0 → null + full-day swap/shelf
      - no feasible set → null
    """
    arms: set[str] = set()
    for s in pomis_sets:
        fs = set(s)
        if not set_is_layout_feasible(fs):
            continue
        if not fs:
            arms.add("null")
            continue
        if involves_timing_from_rush(fs):
            arms.update(["null", "swap20@1", "swap40@1", "shelf50@1"])
        else:
            arms.update(["null", "swap20", "swap40", "shelf50"])
    if not arms:
        arms = {"null"}
    return sorted(arms)


def map_pomis_to_business_arms(
    pomis_sets: list[list[str]],
    *,
    strict: bool = False,
) -> list[str]:
    """
    Coarse mapping:
      - empty set → null
      - M/D only → swap20, swap40, shelf50 (full day)
      - M/D at t≥1 only → same with @1
      - includes Load/V/W etc. → discard (entire set not layout-feasible)

    strict=False (default): fallback to full business POMIS set when no feasible set.
    strict=True: keep null only when no feasible set (no full-set injection / no forced @1).
    """
    arms: set[str] = set()
    any_layout = False
    any_timing = False
    for s in pomis_sets:
        fs = set(s)
        if not set_is_layout_feasible(fs):
            continue
        if not fs:
            arms.add("null")
            continue
        any_layout = True
        if involves_timing_from_rush(fs):
            any_timing = True
        else:
            arms.update(["null", "swap20", "swap40", "shelf50"])
    if any_layout and not any(a.startswith("swap") or a == "shelf50" for a in arms):
        arms.update(["null", "swap20", "swap40", "shelf50"])
    if any_timing:
        arms.update(["swap20@1", "swap40@1", "shelf50@1"])
    if not arms:
        if strict:
            arms = {"null"}
        else:
            arms = set(arm_sets()["POMIS"])
    return sorted(arms)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fallback-sup",
        action="store_true",
        help="Ignore G_c; use G_sup throughout (honest fallback control)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Strict G_c-POMIS: no forced union of POMIS+/@1; "
            "keep null only when no layout-feasible POMIS. Writes pomis_plus_table_strict.json"
        ),
    )
    parser.add_argument(
        "--project",
        action="store_true",
        help=(
            "Layout projection: project POMIS sets to M*/D* first, then strict mapping without forced union. "
            "Writes pomis_plus_table_projected.json (sensitivity; does not change main table)"
        ),
    )
    args = parser.parse_args()
    if args.strict and args.project:
        raise SystemExit("Choose one: --strict or --project")

    summary_path = PHASE2_DIR / "contextual_graph" / "gating_summary.json"
    with summary_path.open(encoding="utf-8") as f:
        gating = json.load(f)

    contexts = sorted(gating["graphs_index"].keys())
    if args.project:
        out_path = OUT_PATH_PROJECTED
        mode = "project"
    elif args.strict:
        out_path = OUT_PATH_STRICT
        mode = "strict"
    else:
        out_path = OUT_PATH
        mode = "default"

    table = {
        "meta": {
            "note": {
                "project": (
                    "PROJECTED G_c-POMIS: keep only M*/D* nodes then map; "
                    "no forced POMIS+/@1 union; bandit LOOKUP only"
                ),
                "strict": (
                    "STRICT G_c-POMIS: no forced POMIS+/@1 union; "
                    "infeasible POMIS → null only; bandit LOOKUP only"
                ),
                "default": "Offline POMIS+ arm table; bandit must LOOKUP only",
            }[mode],
            "fallback_sup": args.fallback_sup,
            "strict_gc_pomis": mode == "strict",
            "project_md": mode == "project",
            "honest_fallback_fixed_graph_flag": gating.get("honest_fallback_fixed_graph"),
            "forbid_online_pomis_search": True,
        },
        "by_context": {},
    }

    G_sup = build_temporal_diagram()

    for ctx in contexts:
        if args.fallback_sup:
            G = G_sup
            meta = {"removed_edge_ids": [], "n_removed": 0}
        else:
            G, meta = load_Gc(ctx)
        tag = {"project": "POMIS-project", "strict": "POMIS-strict", "default": "POMIS+"}[mode]
        print(f"[{tag}] {ctx} |V|={len(G.V)} removed={meta.get('removed_edge_ids')}")
        try:
            pomis = pomis_nodes_for_Y2(G)
        except Exception as e:
            print(f"  POMIS failed: {e}; {'null only' if mode != 'default' else 'fallback POMIS'}")
            pomis = []
            arms = ["null"] if mode != "default" else arm_sets()["POMIS"]
            table["by_context"][ctx] = {
                "pomis_y2": [],
                "arms": arms,
                "removed_edge_ids": meta.get("removed_edge_ids", []),
                "error": str(e),
            }
            continue

        pomis_for_map = project_pomis_to_manipulable(pomis) if mode == "project" else pomis
        if mode == "project":
            arms = map_projected_pomis_to_business_arms(pomis_for_map)
        else:
            arms = map_pomis_to_business_arms(
                pomis_for_map, strict=(mode == "strict")
            )
        if mode == "default":
            # Default: force union of POMIS+ timing arms (main table behavior, unchanged)
            if meta.get("n_removed", 0) >= 0:
                plus = set(arm_sets()["POMIS+"])
                arms = sorted(set(arms) | (plus & set(arms)) | {"null"})
                if any(a in ("swap20", "swap40", "shelf50") for a in arms):
                    arms = sorted(set(arms) | {"swap20@1", "swap40@1", "shelf50@1"})

        table["by_context"][ctx] = {
            "pomis_y2_sample": pomis[:20],
            "pomis_projected_sample": pomis_for_map[:20] if mode == "project" else None,
            "n_pomis": len(pomis),
            "arms": arms,
            "removed_edge_ids": meta.get("removed_edge_ids", []),
            "layout_feasible_pomis": [
                s for s in pomis_for_map if set_is_layout_feasible(set(s))
            ],
        }
        print(f"  n_pomis={len(pomis)} arms={arms}")
        if mode == "project":
            print(f"  projected={pomis_for_map[:5]}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(table, f, ensure_ascii=False, indent=2)
    print("saved", out_path)


if __name__ == "__main__":
    main()
