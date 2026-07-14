# Methods overview

## Pipeline

1. **Preprocess** (`python -m preprocess.phase1_preprocess`)  
   Fuse wait extracts, dispense timing table, shelf map → `data/item_level.parquet`, `data/rx_level.parquet`.

2. **Calibrate discrete SCM** (`python -m phase2.phase2_run`)  
   Quartile discretization, Dirichlet CPT calibration, queue regression helpers.

3. **Feasible interventions** (`python -m phase2.run_feasible_interventions`)  
   Compliance-constrained machine swaps and shelf near-end moves; writes pair tables under `data/phase2/`.

4. **Continuous-reward bandit** (`python -m phase2.run_continuous_bandit`)  
   Ridge wait predictor + heuristic call-wait propagation; calendar-ordered TS/UCB.

5. **Mechanism reporting**  
   - `phase2.effect_decomposition` / `shapley_paths` / `qm_policy` — queue vs service channels  
   - `phase2.model_based_hte` — stratum τ profiles  
   - `phase2.policy_bootstrap` / `rolling_swap` — intervals and re-optimization

## Identification note

All intervention contrasts are **model-based** counterfactuals. They are not randomized ATEs and not inverse-propensity / doubly robust off-policy estimates. Historical single-item layout changes are used as **negative controls**.

## Column names

Hospital extracts use Chinese column headers. The code keeps those string literals so that the same exports can be dropped into `raw/` without renaming. Comments and documentation are English.
