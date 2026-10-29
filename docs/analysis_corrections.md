# Analysis Corrections Disclosure

**Scope**: Five-Seed Comparative Analysis of Grokfast-EMA vs Baseline (`grokfast-forward`)  
**Status**: Correction in place of analysis implementation and generated artifacts (the prospective specification `docs/analysis_spec.md` stands unchanged; no history rewritten; no models retrained).

---

## 1. Summary of Defects and Rationale

Review of the committed analysis at `bb41d12` identified four defects in the analysis implementation script (`analysis/five_seed_comparison.py`) and generated artifacts:

1. **Marginal Medians Under Right-Censoring**:
   - *Defect*: The original script evaluated marginal medians using Python's standard `statistics.median(observed_crossings)` on the subset of arms that completed crossing (`completer-only` median). This dropped censored arms that did not reach the threshold within 50,000 updates.
   - *Violation of Committed Spec*: `docs/analysis_spec.md` forbids dropping censored seeds or replacing them with arbitrary values. Censored seeds lie beyond the 50,000-update horizon and must be ordered above any observed crossing. With $N=5$ seeds, the marginal median is the 3rd-ordered value in the full distribution. Thus, the median is estimable if and only if $\ge 3$ seeds cross AND the 3rd-ordered value is an observed crossing; otherwise, the median falls in the censored region and is reported as not estimable.
2. **Prefix Evidence Overclaim**:
   - *Defect*: The original text claimed "bit-for-bit trajectory match" and asserted that the evidence confirmed full trajectory comparability at step resolution.
   - *Correction*: The empirical prefix check evaluated exactly two arm-pairs: Seed 4 `none_wd0` (Attempt 1 vs Attempt 2) and Seed 1 `ema_wd0` (Attempt 2 vs Attempt 3). The evidence demonstrated exact point-by-point agreement of recorded validation-accuracy curves on the shared 100-update evaluation grid—not full model state. Comparability of the remaining rerun arms rests on the methodological assumption of identical code, seeds, and hyperparameters under the same cap.
3. **Accuracy Difference Units**:
   - *Defect*: Differences of two percentages ($\Delta_{\text{acc}} = \text{acc}_{\text{EMA}} - \text{acc}_{\text{none}}$) were formatted as `%`.
   - *Correction*: Accurately labelled as percentage points (`pp`) across all tables, prose, CSV column headers, and plot axes.
4. **Speed Advantage Interpretation Precision**:
   - *Defect*: Prose stating "earlier crossing in 4 of 5 paired comparisons" was imprecise regarding the breakdown between uncensored and censored pairs.
   - *Correction*: Precisely decomposed to state that for $wd=0.005$, in 3 of 4 uncensored pairs EMA crossed earlier, and in 1 additional pair EMA crossed while baseline was right-censored.

---

## 2. Quantitative Before-and-After Comparison

| Metric / Arm Condition | Committed Value (`bb41d12`) | Corrected Value | Status / Impact |
|:---|:---:|:---:|:---|
| **Marginal Median**: $wd = 0.0$ Baseline (`none`) | Not estimable ($0/5$ crossed) | Not estimable within the 50,000-update horizon ($0/5$ crossed) | Unchanged (0 crossed) |
| **Marginal Median**: $wd = 0.0$ Grokfast-EMA | 49,550 updates (completer-only average of 49,400 & 49,700) | **49,700 updates** (3rd ordered crossing of 5 seeds) | Corrected |
| **Marginal Median**: $wd = 0.005$ Baseline (`none`) | 44,100 updates (completer-only average of 41,300 & 46,900) | **46,900 updates** (3rd ordered crossing of 5 seeds) | Corrected |
| **Marginal Median**: $wd = 0.005$ Grokfast-EMA | 44,700 updates (5/5 crossed) | **44,700 updates** (5/5 crossed) | Unchanged (all 5 crossed) |
| **Paired Crossing Differences** ($\Delta_{\text{cross}}$) | Seeds 1, 2, 3, 5: [-1,100, -300, -29,300, +23,700] | Same values, restricted to uncensored pairs | Unchanged (primary analysis) |
| **Final Accuracy Difference Units** ($\Delta_{\text{acc}}$) | Formatted as `%` | Formatted as **`pp`** (percentage points) | Unit relabeling |

### Scientific Note on Marginal Median Shift
In the original completer-only computation at $wd=0.005$, the baseline marginal median (44,100) appeared lower than EMA (44,700) because the slowest baseline seed (Seed 4, which never crossed by 50,000 updates) was omitted. When properly ordering the censored seed beyond the horizon, the true marginal median crossing for baseline is 46,900 updates, showing that EMA (44,700) has a faster marginal median crossing than baseline. The paired comparison—which is primary—remains unaffected.

---

## 3. Implementation and Verification

- The logic in `analysis/five_seed_comparison.py` was corrected in place to compute marginal medians under censoring and output updated summaries, tables, and figures.
- Synthetic unit tests were introduced in `analysis/test_five_seed_comparison.py` covering marginal median computation with 0, 2, 3, 4, and 5 observed crossings, censored 3rd-value detection, and pair classification under censoring.
- All artifact files (`five_seed_summary.md`, `prefix_check.md`, `paired_comparison.csv`, `per_seed_arms.csv`, `crossing_and_accuracy_comparison.png`) were regenerated via pipeline execution.
