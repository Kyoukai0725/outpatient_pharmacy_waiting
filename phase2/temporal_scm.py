"""T=3 temporal causal graph + CPT lookup SCM adapter layer (NS-SCMMAB graph interface)."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_NS = _ROOT / "vendor" / "NS-SCMMAB-main"
if str(_NS) not in sys.path:
    sys.path.insert(0, str(_NS))

from npsem.model import CD, CausalDiagram

from phase2.config import CARDINALITY

T_SLICES = 3  # t=0 pre-rush, t=1 morning-rush, t=2 rest
SLICE_NAMES = ("pre", "rush", "rest")


def _slice_vars(t: int) -> dict[str, str]:
    return {
        "Load": f"Load{t}",
        "N": f"N{t}",
        "M": f"M{t}",
        "D": f"D{t}",
        "W": f"W{t}",
        "V": f"V{t}",
        "Y": f"Y{t}",
    }


def build_temporal_diagram() -> CausalDiagram:
    """
    Intraday T=3 unfolding:
      Within-slice: Load,N,M→D; Load,D→W; Load→V; W,D,V→Y
      Cross-slice: W0→W1→W2; D1→W2
    """
    vs: set[str] = set()
    edges: list[tuple[str, str]] = []
    for t in range(T_SLICES):
        v = _slice_vars(t)
        vs.update(v.values())
        edges += [
            (v["N"], v["D"]),
            (v["M"], v["D"]),
            (v["Load"], v["D"]),
            (v["Load"], v["W"]),
            (v["D"], v["W"]),
            (v["Load"], v["V"]),
            (v["W"], v["Y"]),
            (v["D"], v["Y"]),
            (v["V"], v["Y"]),
        ]
    # residual queue + rush dispense spillover
    edges += [("W0", "W1"), ("W1", "W2"), ("D1", "W2")]
    return CD(vs, edges)


TEMPORAL_PARENTS: dict[str, tuple[str, ...]] = {}
for t in range(T_SLICES):
    v = _slice_vars(t)
    TEMPORAL_PARENTS[v["Load"]] = ()
    TEMPORAL_PARENTS[v["N"]] = ()
    TEMPORAL_PARENTS[v["M"]] = ()
    TEMPORAL_PARENTS[v["D"]] = (v["N"], v["M"], v["Load"])
    w_pa = [v["Load"], v["D"]]
    if t >= 1:
        w_pa.append(f"W{t-1}")
    if t == 2:
        w_pa.append("D1")
    TEMPORAL_PARENTS[v["W"]] = tuple(w_pa)
    TEMPORAL_PARENTS[v["V"]] = (v["Load"],)
    TEMPORAL_PARENTS[v["Y"]] = (v["W"], v["D"], v["V"])


def _cfg_key(parents: tuple[str, ...], state: dict) -> str:
    return "|".join(str(int(state[p])) for p in parents)


def _sample_probs(probs: list[float] | np.ndarray, rng: np.random.Generator) -> int:
    p = np.asarray(probs, dtype=float)
    p = np.clip(p, 1e-12, None)
    p = p / p.sum()
    return int(rng.choice(len(p), p=p))


def _get_cpt_row(cpts: dict, node: str, key: str) -> list[float]:
    table = cpts[node]
    if key in table:
        return table[key]
    # fallback: average over keys / empty
    if "" in table:
        return table[""]
    mats = np.stack([np.asarray(v, dtype=float) for v in table.values()], axis=0)
    return mats.mean(axis=0).tolist()


@dataclass
class InterventionSpec:
    """Layout intervention on the temporal SCM: change D/M distributions from start_t onward."""

    name: str
    start_t: int = 0  # 0=full day; 1=from rush onward; 2=rest only
    # Expected D-bin shift down relative to baseline (larger is better)
    d_shift: float = 0.0
    # Expected M-bin shift up
    m_shift: float = 0.0
    # Extra reward for directly lowering P(Y=Q4) (from offline simulation calibration)
    reward_bonus: float = 0.0
    meta: dict = field(default_factory=dict)


@dataclass
class PharmacyTemporalSCM:
    """
    CPT lookup temporal SCM.
    cpts_other / cpts_rush: single-slice node names Peak0/Load0/.../Y0 CPTs (from ns_mechanisms).
    """

    cpts_other: dict
    cpts_rush: dict
    diagram: CausalDiagram = field(default_factory=build_temporal_diagram)
    weekday: bool = True

    def regime_cpts(self, t: int) -> dict:
        if t == 1 and self.weekday:
            return self.cpts_rush
        return self.cpts_other

    def _map_single_slice_key(self, node0: str, parents0: tuple[str, ...], state_t: dict, t: int) -> str:
        """Map temporal state to single-slice CPT parent config key (Peak0/Load0/...)."""
        # Peak0: rush slice on weekday → 1
        peak = 1 if (t == 1 and self.weekday) else 0
        mapping = {
            "Peak0": peak,
            "Load0": int(state_t.get(f"Load{t}", state_t.get("Load0", 1))),
            "N0": int(state_t.get(f"N{t}", 1)),
            "M0": int(state_t.get(f"M{t}", 2)),
            "D0": int(state_t.get(f"D{t}", 1)),
            "W0": int(state_t.get(f"W{t}", 1)),
            "V0": int(state_t.get(f"V{t}", 1)),
        }
        return "|".join(str(mapping[p]) for p in parents0)

    def sample_exogenous_slice(
        self,
        t: int,
        rng: np.random.Generator,
        exo: dict[str, int] | None = None,
    ) -> dict[str, int]:
        cpts = self.regime_cpts(t)
        out = {}
        for base, card_key in (("Load", "Load0"), ("N", "N0"), ("M", "M0")):
            name = f"{base}{t}"
            if exo and name in exo:
                out[name] = int(exo[name])
                continue
            if exo and card_key in exo:
                out[name] = int(exo[card_key])
                continue
            probs = _get_cpt_row(cpts, card_key, "")
            out[name] = _sample_probs(probs, rng)
        return out

    def _apply_m_shift(self, m: int, shift: float, rng: np.random.Generator) -> int:
        if shift <= 0:
            return m
        # probabilistic bump toward higher machine ratio
        if rng.random() < min(0.95, shift / 2.0):
            return min(3, m + 1)
        return m

    def _apply_d_shift(self, d: int, shift: float, rng: np.random.Generator) -> int:
        if shift <= 0:
            return d
        # shift expected D down
        steps = int(shift) + (1 if rng.random() < (shift % 1) else 0)
        return max(0, d - steps)

    def sample_day(
        self,
        rng: np.random.Generator | None = None,
        intervention: InterventionSpec | None = None,
        exo_by_t: dict[int, dict[str, int]] | None = None,
    ) -> dict[str, int]:
        rng = rng or np.random.default_rng()
        interv = intervention or InterventionSpec("null")
        state: dict[str, int] = {}

        from phase2.causal_graph import PARENTS as SINGLE_PARENTS

        for t in range(T_SLICES):
            cpts = self.regime_cpts(t)
            exo = self.sample_exogenous_slice(t, rng, exo=(exo_by_t or {}).get(t))
            state.update(exo)

            active = t >= interv.start_t
            m_name, n_name, load_name = f"M{t}", f"N{t}", f"Load{t}"
            if active and interv.m_shift > 0:
                state[m_name] = self._apply_m_shift(state[m_name], interv.m_shift, rng)

            # D | N, M  (ignore Load in single CPT; Load used for W/V)
            d_key = self._map_single_slice_key(
                "D0",
                SINGLE_PARENTS["D0"],
                {**state, "N0": state[n_name], "M0": state[m_name]},
                t,
            )
            # D0 parents are N0,M0 only in single graph
            d_key = "|".join(str(x) for x in (state[n_name], state[m_name]))
            d = _sample_probs(_get_cpt_row(cpts, "D0", d_key), rng)
            if active:
                d = self._apply_d_shift(d, interv.d_shift, rng)
            state[f"D{t}"] = d

            # W | Peak, Load, D  (+ residual from W_{t-1})
            peak = 1 if (t == 1 and self.weekday) else 0
            w_key = f"{peak}|{state[load_name]}|{d}"
            # single CPT W0 parents: Peak0, Load0, D0
            w = _sample_probs(_get_cpt_row(cpts, "W0", w_key), rng)
            if t >= 1:
                # residual queue: cannot be much better than previous slice wait
                prev = state[f"W{t-1}"]
                if rng.random() < 0.4:
                    w = max(w, prev)
            if t == 2:
                # D1 spillover into afternoon wait
                if state["D1"] >= 2 and rng.random() < 0.25:
                    w = min(3, w + 1)
            state[f"W{t}"] = w

            v_key = f"{peak}|{state[load_name]}"
            # V0 parents Peak0, Load0
            v = _sample_probs(_get_cpt_row(cpts, "V0", v_key), rng)
            state[f"V{t}"] = v

            y_key = f"{w}|{d}|{v}"
            y = _sample_probs(_get_cpt_row(cpts, "Y0", y_key), rng)
            state[f"Y{t}"] = y

        return state

    def expected_reward(
        self,
        intervention: InterventionSpec | None = None,
        n_mc: int = 400,
        seed: int = 0,
        weekday: bool | None = None,
    ) -> dict[str, float]:
        """
        Daily reward: MC estimate of R = 1 - mean_t P(Y_t=3), plus reward_bonus.
        Also returns E[Y], P(Q4).
        """
        if weekday is not None:
            self.weekday = weekday
        rng = np.random.default_rng(seed)
        interv = intervention or InterventionSpec("null")
        ys, y1s, q4s = [], [], []
        for _ in range(n_mc):
            s = self.sample_day(rng=rng, intervention=interv)
            y_mean = np.mean([s["Y0"], s["Y1"], s["Y2"]])
            ys.append(y_mean)
            y1s.append(s["Y1"])
            q4s.append(np.mean([s["Y0"] == 3, s["Y1"] == 3, s["Y2"] == 3]))
        ey = float(np.mean(ys))
        pq4 = float(np.mean(q4s))
        r = (1.0 - pq4) + float(interv.reward_bonus)
        return {
            "reward": round(r, 6),
            "E_Y": round(ey, 4),
            "P_Q4": round(pq4, 4),
            "E_Y1_rush": round(float(np.mean(y1s)), 4),
            "arm": interv.name,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "T": T_SLICES,
            "slices": list(SLICE_NAMES),
            "nodes": sorted(self.diagram.V),
            "edges": list(self.diagram.edges),
            "parents": {k: list(v) for k, v in TEMPORAL_PARENTS.items()},
        }


def load_scm_for_calblock(calblock: str, ns_cpt_dir: Path | None = None) -> PharmacyTemporalSCM:
    from phase2.ns_mechanisms import NS_CPT_DIR, load_ns_cpt

    d = ns_cpt_dir or NS_CPT_DIR
    other = load_ns_cpt(calblock, "other", d)["cpts"]
    rush = load_ns_cpt(calblock, "rush", d)["cpts"]
    return PharmacyTemporalSCM(cpts_other=other, cpts_rush=rush)
