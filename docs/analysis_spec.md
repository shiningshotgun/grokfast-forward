# Prospective Analysis Specification for Five-Seed Grokfast Comparison

**Study**: Grokfast modular-multiplication grokking repro  
**Scope**: Five random seeds (Seeds 1, 2, 3, 4, 5), four arms each (`none_wd0`, `ema_wd0`, `none_wd0.005`, `ema_wd0.005`).  
**Commit Status**: Committed prospectively prior to any comparative metric computation.

---

## 1. Analysis Questions

Within each weight-decay level ($wd = 0$ and $wd = 0.005$):
1. **Primary**: Does the Grokfast-EMA arm cross the generalization threshold sooner (in fewer update steps) than the baseline (`none`) arm?
2. **Secondary**: Does the Grokfast-EMA arm reach a higher final validation accuracy at the end of the update budget ($50,000$ updates) than the baseline (`none`) arm?

---

## 2. Record Schema and Evaluation Grid Evidence

Inspection of the canonical study records across all five seeds (`study_outputs/multiseed_seed{1_attempt3, 2, 3, 4_attempt2, 5}/`) confirms:
- **Accuracy Metric Units**: Validation accuracy (`acc`) is recorded as a floating-point fraction in the closed interval $[0.0, 1.0]$ (e.g., $1.0 = 100\%$, $0.99 = 99\%$).
- **Evaluation Cadence**: Full-set validation evaluations (`full_set = True`, $N = 4,656$ validation pairs) occur uniformly every $100$ updates starting at update $0$ through update $50,000$ ($501$ evaluations per arm).
- **Endpoint Integrity**: Every arm across all five seeds reached status `completed_cap` at exactly $50,000$ updates with its final evaluation at step $50,000$.

---

## 3. Generalization Crossing Definition & Conflict Resolution

### Conflict Analysis of Pre-existing Repo Definitions
Two distinct crossing concepts appear in the existing codebase:
1. `runner.py` (lines 930–935): Defines crossing as the first evaluation with `acc >= 0.95`. This definition is transient: it records the earliest step meeting the threshold regardless of whether performance subsequently collapses.
2. `recompute_study.py` (lines 110–120) and `paper_vs_study.md`: Document that transient $\ge 0.95$ crossing exhibits severe post-crossing instability and reversals (e.g., accuracy dropping back down to $57.32\%$ for baseline and $85.57\%$ for EMA). Both sources explicitly warn that *"reaching 0.95 once does not mean staying there"* and record subsequent sub-0.95 observations.

### Adopted Crossing Rule
To resolve this instability and align with canonical grokking literature (Power et al., 2022, which defines grokking as terminal, sustained transition to near-perfect generalization), we adopt the declared rule:
> **Adopted Crossing Rule**: A crossing is defined as the first recorded full-set evaluation step at or above $0.99$ ($99\%$) validation accuracy that is **never followed by a drop below $0.99$** through the final recorded evaluation at step $50,000$.

An arm that never achieves sustained $\ge 0.99$ accuracy through step $50,000$ is classified as **RIGHT-CENSORED** at the $50,000$-update cap.

---

## 4. Metrics and Statistical Treatment

### 4.1 Primary Metric: Steps to Crossing
- Measured as discrete updates on the $100$-update grid.
- **Right-Censoring**: An arm that does not meet the sustained crossing rule by step $50,000$ is treated strictly as right-censored at $> 50,000$ updates. It is **never** assigned an artificial crossing step of $50,000$, nor is it omitted from reporting.
- **Marginal (Unpaired) Medians**: For any single arm across the five seeds, the marginal median crossing step is estimable if and only if at least $3$ of the $5$ seeds cross ($\ge 60\%$). If fewer than $3$ seeds cross, the median is reported as `"not estimable within the 50,000-update horizon"` alongside the observed crossing count (e.g., $k/5$).

### 4.2 Paired Differences & Censored Handling
For each seed and weight-decay condition, the pair $(\text{EMA}, \text{none})$ is classified into exactly one of four quadrants:
1. `both-cross`: Both arms achieved sustained crossing.
2. `EMA-only-cross`: Only the Grokfast-EMA arm achieved sustained crossing; the baseline arm was right-censored.
3. `none-only-cross`: Only the baseline arm achieved sustained crossing; the EMA arm was right-censored.
4. `neither`: Neither arm achieved sustained crossing before the $50,000$-update horizon.

**Rules for Paired Calculations**:
- Numeric paired differences in crossing steps ($\Delta_{\text{cross}} = \text{step}_{\text{EMA}} - \text{step}_{\text{none}}$) are computed **ONLY** for pairs in the `both-cross` quadrant.
- Numeric differences must always state the conditional denominator (e.g., $n / 5$ pairs) and be explicitly labelled as a **conditional subset**, never reported as an unconditional population median.
- Pairs in `EMA-only-cross`, `none-only-cross`, and `neither` provide directional / rank information (e.g., EMA crossing while baseline censors indicates earlier crossing for EMA), but do not have an exact numeric difference under finite horizons.

### 4.3 Secondary Metric: Final Validation Accuracy
- Measured as the validation accuracy (`acc`, fraction in $[0, 1]$) at the final evaluation step ($50,000$).
- Verification: Each arm's final record must be verified to be at update $50,000$. If any arm ended prior to $50,000$, its data is defective and analysis must halt.
- Paired difference: $\Delta_{\text{acc}} = \text{acc}_{\text{EMA}} - \text{acc}_{\text{none}}$ at step $50,000$.

### 4.4 Sign Conventions
- **Crossing-Time Difference ($\text{EMA} - \text{none}$)**:  
  **NEGATIVE** means the Grokfast-EMA arm crossed sooner (fewer updates).  
  **POSITIVE** means the baseline arm crossed sooner.
- **Final-Accuracy Difference ($\text{EMA} - \text{none}$)**:  
  **POSITIVE** means the Grokfast-EMA arm achieved higher validation accuracy at the cap.  
  **NEGATIVE** means the baseline arm achieved higher validation accuracy.

---

## 5. Aggregation and Interpretation Scope

1. **Per-Seed Reporting**: Every seed's individual arm values, statuses, and differences will be presented in full tabular form. No aggregation will obscure individual seed variation.
2. **Summary Statistics**: Where estimable, median and range will be reported alongside sign-consistency counts (e.g., $k$ out of $5$ seeds favoring EMA).
3. **Statistical Interpretation**:
   - A sample of $5$ seeds is descriptive and exploratory, not inferential proof.
   - A consistent direction across all seeds is suggestive of an effect.
   - Mixed or contradictory signs across seeds constitutes a finding of seed instability.
   - A null result (e.g., no difference or failure of both to cross) indicates that this configuration found no consistent effect under this horizon—not proof that no effect could exist under other architectures, learning rates, or budgets.

---

## 6. Methodological Disclosure: Budget Amendments

- Seeds 2, 3, and 5 were executed under the original runner budget ($1,500$s per-arm deadline, $6,000$s total deadline, $50,000$-update cap).
- Seeds 1 (attempt 3) and 4 (attempt 2) were executed under an approved amended budget ($2,100$s per-arm deadline, derived $9,000$s total deadline, $50,000$-update cap) to eliminate time-budget censoring that occurred in their earlier attempts.
- In both budget regimes, the algorithm, hyperparameters, data split, batch order, and update cap ($50,000$) were identical.
- Comparability between the $1,500$s and $2,100$s runs is treated as a stated methodological assumption. Part B empirically tests this assumption via prefix-consistency checks against the superseded attempts over shared updates.
