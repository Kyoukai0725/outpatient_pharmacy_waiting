"""
Outpatient pharmacy waiting-time causal graph

    Peak0 ──→ W0 ──────────────→ Y0
    Load0 ──┘      ↑            ↑
              N0 ──→ D0 ────────┤   (D0 = quartile of 预估配药_分钟)
              M0 ──┘      │     ↑
                          └──→ W0   (queue propagation: faster upstream dispensing → shorter call wait)
    Peak0 ──→ V0 ───────────────┘
    Load0 ──┘

- W0: call wait
- D0: dispensing (prefer 预估配药_分钟)
- V0: verification and dispensing
- Y0: waiting-time quartile (outcome)
"""

from __future__ import annotations

from dataclasses import dataclass, field

EXOGENOUS = ("Peak0", "Load0", "N0", "M0")
ENDOGENOUS = ("W0", "D0", "V0", "Y0")
ALL_VARS = EXOGENOUS + ENDOGENOUS

DIRECTED_EDGES = [
    ("Peak0", "W0"),
    ("Load0", "W0"),
    ("N0", "D0"),
    ("M0", "D0"),
    ("D0", "W0"),  # Queue propagation: system dispensing efficiency → call wait
    ("Peak0", "V0"),
    ("Load0", "V0"),
    ("W0", "Y0"),
    ("D0", "Y0"),
    ("V0", "Y0"),
]

PARENTS = {
    "Peak0": (),
    "Load0": (),
    "N0": (),
    "M0": (),
    "W0": ("Peak0", "Load0", "D0"),
    "D0": ("N0", "M0"),
    "V0": ("Peak0", "Load0"),
    "Y0": ("W0", "D0", "V0"),
}


def build_causal_diagram():
    from npsem.model import CD

    return CD(set(ALL_VARS), DIRECTED_EDGES)


@dataclass
class PharmacyCausalGraph:
    """Causal graph metadata for serialization and simulation."""

    variables: list[str] = field(default_factory=lambda: list(ALL_VARS))
    exogenous: list[str] = field(default_factory=lambda: list(EXOGENOUS))
    endogenous: list[str] = field(default_factory=lambda: list(ENDOGENOUS))
    edges: list[list[str]] = field(default_factory=lambda: [list(e) for e in DIRECTED_EDGES])
    parents: dict[str, list[str]] = field(default_factory=lambda: {k: list(v) for k, v in PARENTS.items()})
    n_levels: int = 4
    dispense_node: str = "D0"
    dispense_source: str = "预估配药_分钟"
    outcome_node: str = "Y0"

    def to_dict(self) -> dict:
        return {
            "variables": self.variables,
            "exogenous": self.exogenous,
            "endogenous": self.endogenous,
            "edges": self.edges,
            "parents": self.parents,
            "n_levels": self.n_levels,
            "dispense_node": self.dispense_node,
            "dispense_source": self.dispense_source,
            "outcome_node": self.outcome_node,
            "description": "check-in → call wait (W0, incl. queue propagation) ← dispensing (D0) → verification (V0) → waiting time (Y0)",
        }
