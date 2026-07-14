# Outpatient Pharmacy Waiting-Time Layout Optimization

Domain-adapted **non-stationary structural causal bandits** for outpatient pharmacy layout decisions under binding capacity: the automated dispensing cabinet is full and the number of dispensing windows is fixed.

**Research question.** How can compliance-constrained **machine slot swaps** and **shelf near-end rearrangements** shorten patient waiting, and through which causal channel (direct service vs. call-queue congestion)?

## Key results (model-based)

| Quantity | Estimate |
|----------|----------|
| Recommended policy | Compliance-constrained **40-slot machine swap** |
| Call wait ↓ (affected Rx) | ~**0.29 min** |
| P(worst waiting quartile) ↓ | ~**3.2 pp** |
| Queue-channel share of effect | ~**98.7%** (nested ≈ Shapley) |
| Rough net hours saved | ~**275 h** (two-year scale, approximate) |

Identification is **model-based** (outcome regression + heuristic queue propagation), not a randomized trial. Historical single-item changes serve as **negative controls**; a prospective pilot is still needed.

## Repository layout

```
outpatient-pharmacy-waiting/
├── preprocess/          # Stage-1: fuse wait / dispense / layout / roster → Rx table
├── phase2/              # Causal graph, interventions, NS bandits, QM-Policy
├── vendor/NS-SCMMAB/    # Vendored NeurIPS 2025 NS-SCMMAB (npsem)
├── docs/                # Paper draft and notes
├── figures/             # Result figures for the report
├── data/                # Derived outputs (mostly gitignored)
└── raw/                 # Place hospital extracts here (not committed)
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e vendor/NS-SCMMAB   # provides `npsem`
export PYTHONPATH="$(pwd):$PYTHONPATH"
```

### Data (not in git)

Place extracts under `raw/` following `raw/README.md`. Expected sources:

- Wait-time monthly extracts 
- Dispense timing table (machine 17s / manual seconds)
- Shelf layout map
- Daily staff rosters
- Optional: historical machine in/out log for validation

Then:

```bash
python3 -m preprocess.phase1_preprocess
python3 -m phase2.phase2_run
python3 -m phase2.run_feasible_interventions
python3 -m phase2.run_continuous_bandit --reuse-daily-mu
```

See module docstrings under `phase2/` for path decomposition, QM-Policy, rolling swap, and sensitivity experiments.

## Method (short)

1. **Single-slice SCM + Dirichlet CPT**; dispense node uses table-based seconds (system timestamps unreliable).
2. **Non-stationary mechanisms** θ(r,c): MorningRush × calendar half-year blocks.
3. **T=3 temporal SCM** with residual waiting across slices.
4. **Business arms**: constrained machine swaps / shelf moves; POMIS(+)-filtered menus.
5. **Continuous reward path**: Ridge wait predictor; counterfactual call wait via queue map; reward \(R=-\hat Y\) (no hand-tuned bonus).
6. **QM-Policy**: nested / Shapley decomposition into direct service vs. queue channels.

## Citation

If you use the NS-SCMMAB library:

```
Kwon et al., Non-Stationary Structural Causal Bandits, NeurIPS 2025.
```

Paper draft for this pharmacy application: `docs/paper_draft_SCM_layout_intervention_pharmacy_waiting.md`.

## License

Application code in this repository: MIT (see `LICENSE`).  
Vendored `vendor/NS-SCMMAB` retains its upstream license.
