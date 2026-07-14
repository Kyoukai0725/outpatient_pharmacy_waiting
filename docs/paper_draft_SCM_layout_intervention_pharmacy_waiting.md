# Structural Causal Analysis of Layout Interventions for Outpatient Pharmacy Waiting Time

**Draft for internal circulation.** Publication figures (JPG, 320 dpi) are in `../figure/`.

*Identification note.* Unless otherwise stated, path decompositions, heterogeneous effects, and policy intervals are obtained from a fitted outcome model together with a heuristic queue-propagation map. They should be read as *model-based* counterfactual comparisons, not as randomized average treatment effects and not as inverse-propensity or doubly robust off-policy estimates.

---

## Abstract

Outpatient pharmacies often operate under capacity limits that rule out the most obvious remedies for long waits: the automated dispensing cabinet may already be full, and the number of dispensing windows may be institutionally fixed. In that setting, layout changes—moving drugs into or out of the machine, or rearranging shelves—alter fill times only modestly at the prescription level. Their operational value, if any, must therefore travel through the call queue that dominates total waiting.

This paper develops a structural causal account of that pathway. We represent the pharmacy as a temporal structural causal model, encode compliance-constrained layout policies as interventions, and decompose each policy’s predicted effect on waiting into a *direct service channel* and a *queue channel*. Using two years of prescription-level data from a large outpatient pharmacy, we find that a forty-slot machine swap shortens call waiting by about 0.29 minutes and lowers the share of prescriptions in the worst waiting quartile by about 3.2 percentage points among affected cases. Nested path decomposition attributes nearly all of the corresponding reduction in predicted waiting—approximately 99 percent—to the queue channel. Heterogeneity analysis, dual bootstrap intervals, and a rolling re-optimization protocol reinforce the same conclusion: under binding capacity, layout redesign matters chiefly as congestion control.

**Keywords:** structural causal models; outpatient pharmacy; waiting time; layout intervention; queue mediation

---

## 1. Introduction

Long waits at the outpatient pharmacy are a familiar source of patient dissatisfaction and staff pressure. Managers typically respond by asking for more windows, more machines, or faster picking. Those levers are not always available. In the pharmacy studied here, the automated dispensing machine is already saturated, and the number of dispensing windows is fixed at six. The remaining degrees of freedom are rearrangements of *what* sits in the machine and *where* manual items are stored.

Such rearrangements change the seconds required to assemble a prescription. Taken one prescription at a time, those changes look small: machine items dispense in a fixed interval (seventeen seconds in our timing table), while manual items take longer and vary with walking distance. The operational hypothesis is subtler. If enough high-volume items move onto the machine, system throughput rises, the call queue drains faster, and waiting falls for many patients—including those whose prescriptions never contain a swapped drug. Conversely, if waiting is driven mainly by verification or by idiosyncratic prescription complexity, layout change will disappoint.

Existing pharmacy-operations studies document staffing, automation, and shelf placement, yet they rarely cast layout policies as interventions in a causal system, and they seldom separate the direct effect of faster filling from the indirect effect that runs through congestion. Causal banding frameworks supply formal language for temporal interventions, but continuous online exploration is a poor match for a near-saturated pharmacy whose feasible actions are few and whose best policy is already close to obvious.

We therefore pursue a different question: *through which causal channel does a compliance-constrained layout change affect waiting, and how large is that effect under realistic constraints?* The paper answers in three steps. Section 4 builds a temporal structural causal model of the pharmacy day. Section 5 defines feasible swaps and shelf moves, maps them into counterfactual waiting, and decomposes predicted effects into service and queue channels. Section 6 examines who benefits, how uncertain the policy value is across calendar regimes, and how often the recommended slot set should be refreshed. Throughout, we keep the identification status explicit: the quantitative claims are model-based counterfactuals grounded in observational data and a transparent queue map, not a field experiment.

---

## 2. Related Work

Pharmacy operations research has long studied window staffing, automated dispensing, and shelf organization. That literature clarifies physical constraints and cost trade-offs, but counterfactual evaluation under *machine-capacity conservation*—one drug in only if another leaves—remains uncommon, as does explicit modeling of eligibility rules that forbid certain dosage forms from the machine.

Structural causal models provide a language for such evaluation. Do-calculus and related graphical criteria characterize when an intervention’s effect is identifiable; temporal and nonstationary extensions allow mechanisms to change across regimes. We use that language to define layout actions and to state which paths from action to outcome we intend to measure. We do not propose a new online learning algorithm.

Queueing theory supplies the complementary intuition: when servers are busy, small changes in service rate can produce large changes in waiting. Outpatient pharmacies violate textbook assumptions—arrival intensity surges in the morning, staffing varies by roster, and multiple stages intervene between check-in and handout. Rather than fit a full stochastic queue, we embed a simple, auditable propagation rule inside the causal model and then *ask the model how much of the predicted benefit survives if the queue pathway is closed*. That question is the methodological center of the paper.

---

## 3. Setting, Data, and Outcome Decomposition

### 3.1 Institutional constraints

The decision problem is defined at the level of a business day. A layout policy specifies which drugs occupy machine slots and, optionally, which manual items move closer to the picking path. Policies may be applied all day or only during the morning rush. Machine capacity is conserved: every drug that enters the machine displaces another. Eligibility rules exclude codes ending in `P` or `S` and products whose names indicate oral liquids, mixtures, or injectables. Dispensing windows remain fixed at six. We summarize staffing by a capacity proxy

\[
C=\min(6,N_{\text{dispense}})+0.5\,N_{\text{compound}},
\]

where \(N_{\text{dispense}}\) and \(N_{\text{compound}}\) are daily headcounts from the roster.

### 3.2 Data

We analyze outpatient pharmacy records spanning 2024 and 2025, linked to a dispense-timing table, a shelf map, and daily staff schedules.

**Table 1.** Summary of the study data.

| Quantity | Value |
|:---------|------:|
| Item-level records | 1,364,979 |
| Prescriptions | 724,648 |
| Distinct drugs | 1,366 |
| Match rate to timing table | 97.4% |
| Share of item lines from machine | 70% |
| Waiting time: mean / median / 90th percentile (min) | 4.57 / 3.73 / 7.92 |
| Pure machine / pure manual / mixed prescriptions | 55.8% / 23.5% / 20.7% |
| Waiting quartile cut-points (min) | 2.28, 3.73, 5.60 |

System timestamps that attempt to isolate “dispense duration” are unreliable (about 7.5% of prescriptions yield internally consistent stage times). For that reason, the dispense node in our model uses table-based estimates—seventeen seconds per machine item and measured manual seconds including walking—rather than the noisy timestamps.

### 3.3 Why the call queue is the right target

Total waiting decomposes approximately as

\[
Y \approx W + D + V,
\]

where \(W\) is time from check-in to call, \(D\) is dispense work, and \(V\) is verification and handout. Empirically, mean \(W\) is 3.59 minutes, mean table-based \(D\) is 0.66 minutes, and mean \(V\) is 2.15 minutes (Figure 1). The call queue, not the fill step, accounts for most of the wait. Layout interventions act directly on \(D\) and only indirectly—through congestion—on \(W\). Any credible evaluation must therefore track that second step.

**Figure 1.** Stage composition of outpatient pharmacy waiting. Call waiting dominates both the mean duration and the share of the stage sum.

![Figure 1](../figure/fig04_waiting_stages.jpg)

---

## 4. A Temporal Causal Model of the Pharmacy Day

### 4.1 Within-slice structure

Within a time slice we distinguish peak status and daily load, prescription complexity (item count and machine-item share), dispense intensity \(D\), call waiting \(W\), verification \(V\), and the waiting outcome \(Y\). Layout actions manipulate machine membership and shelf location, which shift \(D\) and the machine-share composition of prescriptions. Congestion links \(D\) and load into \(W\); \(W\), \(D\), and \(V\) jointly determine \(Y\).

### 4.2 Three slices and memory across the day

A pharmacy day is not exchangeable hour by hour. We unfold the system into three slices: pre-rush opening, weekday morning rush (09:00–10:00), and the remainder of the day. Waiting can carry forward across slices, and rush-hour dispense load can spill into later waiting. Figure 2 sketches this temporal organization.

**Figure 2.** Temporal structural causal model with three slices. Within each slice, layout-related features influence dispense intensity and waiting; across slices, waiting persists and rush dispense load may affect later congestion.

![Figure 2](../figure/fig01_temporal_scm.jpg)

### 4.3 Nonstationarity

Mechanisms are allowed to depend on a rush indicator and on calendar half-years (2024H1 through 2025H2). Stratified estimation shows that queue-related relationships drift more than dispense relationships: the nonstationarity that matters for waiting is concentrated on the congestion side of the graph. We also conducted data-driven tests of edge negligibility across contexts. Those tests remove a number of edges and clarify which dependencies are unstable, but after mapping graphical intervention sets to business policies the *feasible action menu* remains the same across contexts. We therefore treat edge gating as structural diagnosis rather than as a device that unlocks context-specific menus (Appendix).

---

## 5. From Layout Actions to Waiting: Interventions and Channels

### 5.1 Feasible policies

Under capacity conservation and eligibility rules we construct machine swaps of twenty, thirty, and forty slots, together with near-shelf rearrangements of twenty and fifty pairs. Larger swaps buy more throughput at the cost of more operational disruption; forty slots strike a practical balance. The recommended forty-slot list is archived with the study materials.

### 5.2 Counterfactual waiting under a policy

Let \(\Delta D_{\mathrm{sys}}\) denote the average reduction in table-based dispense minutes among prescriptions subject to a policy. Observed call waiting \(W\) is mapped to a counterfactual

\[
\tilde W = W\cdot\Bigl(1-\operatorname{clip}\bigl(\tfrac{\Delta D_{\mathrm{sys}}}{\bar D}\,f(\mathrm{Load},\mathrm{Peak},C)\bigr)\Bigr),
\]

where \(f\) amplifies the effect under high load, peak hours, and thin staffing. An outcome regression trained on observed features—including observed \(W\)—is then evaluated at the counterfactual features, with \(\tilde W\) in place of \(W\). Predicted waiting under the null policy and under each alternative yields a day-level contrast.

### 5.3 Separating the service channel from the queue channel

The central identification move is nested and order-averaged path decomposition. Write \(\mu(A)\) for mean predicted waiting under counterfactual regime \(A\):

- \(A_0\): null layout, observed waiting;
- \(A_1\): layout applied to dispense features only, waiting held at its observed value (*direct service channel*);
- \(A_2\): layout applied to dispense features *and* waiting updated by the queue map (*full policy*).

Then

\[
\Delta_{\mathrm{direct}}=\mu(A_0)-\mu(A_1),\qquad
\Delta_{\mathrm{queue}}=\mu(A_1)-\mu(A_2),\qquad
\Delta_{\mathrm{total}}=\mu(A_0)-\mu(A_2).
\]

Averaging the two intervention orders (service first versus queue first) yields a Shapley attribution. When the two orders agree, the channel shares are not an artifact of nesting direction.

**Table 2.** Nested path decomposition of predicted waiting reductions (minutes per day; model-based).

| Policy | Total | Direct service | Queue | Queue share | Rush only | Non-rush |
|:-------|------:|---------------:|------:|------------:|----------:|---------:|
| swap20 | 0.139 | 0.002 | 0.137 | 99% | 0.270 | 0.084 |
| swap40 | 0.196 | 0.003 | 0.194 | 99% | 0.344 | 0.136 |
| shelf50 | 0.158 | ≈0 | 0.158 | ≈100% | 0.291 | 0.102 |

Among prescriptions that actually contain a swapped drug, the direct channel rises to roughly thirteen percent of the total for swap40; the queue channel still accounts for the large majority. Figure 3 visualizes the dominance of the queue term.

**Figure 3.** Nested path decomposition of predicted waiting reductions. Across policies, nearly all of the effect is assigned to the queue channel.

![Figure 3](../figure/fig02_path_decomposition.jpg)

The substantive reading is sharp. Layout redesign in this pharmacy does not work primarily by making individual prescriptions a few seconds faster to assemble. It works by relieving the call queue. That is the mechanism a capacity-constrained service system would lead one to expect, and it is the claim most worth exporting to registration desks, laboratories, and operating-room boards.

### 5.4 Magnitude among affected prescriptions

Table 3 reports effects on the subset of prescriptions touched by each policy, including queue propagation. A forty-slot swap saves on the order of 275 staff hours in a rough two-year extrapolation, shortens call waiting by 0.29 minutes, and reduces the probability of falling into the worst waiting quartile by 3.2 percentage points (Figure 4). Shelf moves help, but less efficiently for congestion. A cautious pilot would begin with the twenty highest-yield pairs and expand only after several weeks of monitoring.

**Figure 4.** Compliance-constrained layout policies compared on call-wait reduction, worst-quartile share, and approximate hours saved (affected prescriptions).

![Figure 4](../figure/fig03_intervention_effects.jpg)

**Table 3.** Estimated effects of compliance-constrained layout policies on affected prescriptions.

| Policy | Approx. net hours saved | Affected prescriptions | Call-wait reduction (min) | Reduction in \(P(Q4)\) | Reduction in mean quartile level |
|:-------|------------------------:|-----------------------:|--------------------------:|----------------------:|---------------------------------:|
| swap20 | 207 | 2,625 | 0.21 | 2.5 pp | 0.07 |
| swap30 | 248 | 3,169 | 0.26 | 3.0 pp | 0.09 |
| **swap40** | **275** | **3,585** | **0.29** | **3.2 pp** | **0.10** |
| shelf20 | 188 | 1,711 | 0.18 | 2.5 pp | 0.07 |
| shelf50 | 242 | 2,156 | 0.23 | 3.1 pp | 0.08 |

---

## 6. Who Benefits, How Uncertain, and How Often to Refresh

### 6.1 Heterogeneity

For each prescription we compute a model-based contrast \(\tau_i=\hat Y_i(\mathrm{null})-\hat Y_i(\mathrm{swap40})\). Average \(\tau\) is about 0.19 minutes overall, 0.25 among prescriptions containing a swapped drug, and 0.18 among prescriptions that do not—evidence of queue spillover. During morning rush the mean rises to 0.36 minutes, and to 0.40 minutes for rush prescriptions that also contain a swapped drug (Figure 5). High-benefit cases concentrate when load is high and compounding staff are thin. These patterns are descriptive profiles under the model; they are not claims of randomized conditional average treatment effects.

**Figure 5.** Mean model-based effect of swap40 by prescription stratum. Untouched prescriptions still improve through queue spillover.

![Figure 5](../figure/fig05_heterogeneous_effects.jpg)

### 6.2 Policy intervals at two granularities

Aggregating day-level contrasts \(\delta_t=\mu_{\mathrm{null}}(t)-\mu_{\mathrm{swap40}}(t)\) over 310 days yields a mean reduction of 0.148 minutes per day. A day-level bootstrap interval, \([0.124,\,0.171]\), excludes zero. A cluster bootstrap that resamples the four calendar half-years yields \([-0.022,\,0.244]\), which includes zero (Figure 6). The discrepancy is informative rather than embarrassing: block means decline from 0.28 in 2024H1 to a *negative* value in 2025H2, so four coarse clusters cannot support a tight interval. Reporting both intervals shows that the signal is real at the daily scale and fragile when one insists on half-year clusters—an honest description of a nonstationary environment.

**Figure 6.** Dual policy intervals for swap40 versus null, with half-year block means. The day-level interval excludes zero; the four-block cluster interval does not.

![Figure 6](../figure/fig06_policy_intervals.jpg)

**Table 4.** Dual policy intervals for the day-mean waiting contrast (swap40 versus null).

| Estimand | Estimate |
|:---------|:---------|
| Mean daily contrast | 0.148 min |
| Day-level 95% bootstrap interval | [0.124, 0.171] |
| CalBlock cluster 95% bootstrap interval | [−0.022, 0.244] |
| Block means (H1 → H2 → 2025H1 → 2025H2) | 0.279, 0.213, 0.122, −0.191 |

### 6.3 Refreshing the slot set

Drug frequencies drift. Re-estimating the forty-slot set from each half-year’s frequencies and evaluating it on the next half-year produces slot turnover of roughly 45–60 percent relative to a policy frozen in 2024H1, with modest gains in predicted waiting on later windows (Figure 7; Table 5). In 2025H2 both the rolling and the frozen policies show negative contrasts—retained here as evidence of regime shift, not smoothed away. The practical implication is periodic refresh, not continuous experimentation.

**Figure 7.** Rolling re-estimation of the forty-slot set. Left: slot turnover against a plan frozen in 2024H1. Right: predicted waiting contrasts on the subsequent block.

![Figure 7](../figure/fig07_rolling_reoptimize.jpg)

**Table 5.** Rolling re-estimation of the forty-slot set against a policy frozen in 2024H1.

| Decision block → evaluation block | Slot turnover vs. frozen | Rolling contrast | Frozen contrast | Difference |
|:----------------------------------| ------------------------:| ----------------:| ---------------:| ----------:|
| 2024H1 → 2024H2 | 0 | 0.199 | 0.199 | 0 |
| 2024H2 → 2025H1 | 0.45 | 0.128 | 0.094 | +0.034 |
| 2025H1 → 2025H2 | 0.60 | −0.189 | −0.215 | +0.025 |

As a sanity check on ranking among a small menu of policies, an offline bandit that plays the continuous-reward contrasts accumulates less regret under a constant swap40 rule than under Thompson sampling or UCB (Figure 8). That finding supports *confirm-and-execute* rather than ongoing exploration; it is not offered as the paper’s methodological contribution.

**Figure 8.** Cumulative regret under continuous predicted waiting (lower is better). A constant swap40 policy dominates standard bandit rules on this menu.

![Figure 8](../figure/fig08_cumulative_regret.jpg)

---

## 7. Discussion

The empirical pattern coheres around a single causal story. Binding capacity makes the call queue the scarce resource. Layout changes that raise throughput therefore pay off mainly by shortening that queue, including for patients whose prescriptions never touch the swapped drugs. The near-equality of nested and order-averaged attributions (queue share ≈ 99 percent) indicates that this conclusion does not hinge on the order in which channels are opened.

The same story limits what online learning can add. When the action set is small and one compliant swap already sits near the top of the ranking, the decision problem collapses to validation, deployment, and periodic refresh. Dynamic graphs that delete unstable edges remain useful for interpretation; they need not rewrite the business menu.

Several limits remain. All headline numbers are model-based. The queue map is deliberately simple. Inventory stock-outs are unobserved. Most importantly, the recommended swap has not yet been fielded; historical single-drug changes serve only as negative controls in which both observation and model find negligible effects. A prospective pilot—beginning with twenty pairs—is the natural next step.

---

## 8. Conclusion

Under a full dispensing machine and fixed windows, outpatient pharmacy layout policy should be evaluated as an intervention on a congested service system, not as a collection of per-item time savings. A temporal structural causal model, paired with nested path decomposition, shows that compliance-constrained machine swaps reduce waiting almost entirely through the queue channel. The forty-slot policy offers a concrete operational plan; heterogeneity and dual intervals clarify who benefits and how fragile the estimate is across calendar regimes; rolling re-estimation sketches how often the plan should be revisited. The broader lesson is portable: when service capacity binds, redesign the layout for congestion, and measure success along the queue path that carries the effect.

---

## Appendix

### A. Edge gating (summary)

Forty negligibility tests produced twenty-two edge removals across eight rush×calendar contexts. Recurring patterns include removal of load-to-verification links in most contexts and removal of the rush-dispense-to-later-waiting link in non-rush contexts. After mapping to business policies, the available arm set is identical in all eight contexts. Detailed tests are filed with the replication materials.

### B. Reproduction commands

```bash
cd 排布研究
python3 -m phase2.qm_policy --arm swap40 --sample-days 60 --with-deployment
python3 -m phase2.policy_bootstrap --arm swap40
python3 -m phase2.rolling_swap --n-swap 40
python3 -m phase2.run_continuous_bandit --reuse-daily-mu
```

### C. Figure and table index

| Item | File | Role in the argument |
|:-----|:-----|:---------------------|
| Figure 1 | `figure/fig04_waiting_stages.jpg` | Waiting is queue-dominated |
| Figure 2 | `figure/fig01_temporal_scm.jpg` | Temporal causal scaffold |
| Figure 3 | `figure/fig02_path_decomposition.jpg` | Queue-channel dominance |
| Figure 4 | `figure/fig03_intervention_effects.jpg` | Operational effect sizes |
| Figure 5 | `figure/fig05_heterogeneous_effects.jpg` | Who benefits under the model |
| Figure 6 | `figure/fig06_policy_intervals.jpg` | Dual intervals / nonstationarity |
| Figure 7 | `figure/fig07_rolling_reoptimize.jpg` | Periodic refresh |
| Figure 8 | `figure/fig08_cumulative_regret.jpg` | Ranking sanity check |
| Tables 1–5 | this draft | Data, channels, effects, CIs, rolling |
