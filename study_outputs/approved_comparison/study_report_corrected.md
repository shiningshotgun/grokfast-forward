# Approved comparison (Corrected Derived Report)

> [!IMPORTANT]
> **Provenance & Supersedes Notice**: This report supersedes `study_report.md` and `study_summary.json`. The original derived files reported an invalid timing ratio (0.6784x) caused by dividing cumulative global study timestamps from sequentially executed arms, conflated maximum accuracy ties with minimum loss, and lacked explicit distinction for partial validation passes. The original summary, report, and raw metrics remain frozen and unmodified as audit provenance. All values herein are recomputed directly from the complete validation records in `arm_*.json`.

## Study Overview & Execution Parameters

- **Study Directory**: `study_outputs/approved_comparison`
- **Recorded Work Elapsed**: 5679.615 s / 6000 s ceiling (94m 39.615s (~94 minutes 40 seconds))
- **Peak RSS**: 309510144 bytes (~309.5 MB macOS `ru_maxrss`)
- **Per-Arm Deadline**: 1500 s | **Cap Per Arm**: 50000 updates
- **Evaluation Grid**: Every 100 updates on full validation set ($n = 4656$)
- **Timing Boundary**: Timer starts after initial marker/thread setup; does not certify a literal all-setup-inclusive 100 minutes.
- **Integrity & Bounds**: Honest in-flight deadline overruns and partial final validations are retained (no hard-kill enforcement).
- **Verification Context**: Saved test log records 32 tests passing (`OK`, `EXIT_STATUS=0`), but lacks command/time headers and does not establish independent execution order. Two new tests have weak assertions (chunked-vs-continuous checks update count rather than parameter equivalence; deadline test checks directory existence rather than per-arm skip logic).
- **Scope of Claim**: Evaluated on one paired seed (seed 0) with unequal final caps. Demonstrates earlier crossing for EMA with weight decay on this configuration, not a universal mechanism or full-paper replication.

## Summary Table: Recomputed Full Validations

All accuracy and loss metrics are computed strictly on complete full validations ($n = 4656$, `complete = True`, `full_set = True`).

| Arm | Status | Updates Done | First $\ge 0.95$ Step | Best Complete Acc (First Step) | Min Complete Loss (Step) | Last Complete Acc | Last Complete Loss | Censored |
|---|---|---|---|---|---|---|---|---|
| `none_wd0` | `completed_cap` | 50,000 | None (censored) | 93.9003% (step 50000) | 0.934071 (step 48700) | 93.9003% (step 50000) | 0.999960 | Yes |
| `ema_wd0` | `deadline_stopped` | 41,919 | None (censored) | 61.3832% (step 41900) | 4.607068 (step 100) | 61.3832% (step 41900) | 5.522185 | Yes |
| `none_wd0.005` | `deadline_stopped` | 46,400 | 5900 | 100.0000% (step 9600) | 0.032604 (step 34400) | 100.0000% (step 46300) | 0.040629 | No |
| `ema_wd0.005` | `deadline_stopped` | 43,886 | 1200 | 100.0000% (step 1800) | 0.005651 (step 8800) | 100.0000% (step 43800) | 0.008490 | No |

### Key Audit Distinctions
- **Partial Final Evaluation in `none_wd0.005`**: The final recorded validation at update 46,400 evaluated only 1,536 of 4,656 examples before deadline expiration (`complete=False`, `full_set=False`). It is excluded from full-set conclusions. Its last complete evaluation occurred at update 46,300 (100% accuracy, loss 0.040629).
- **Best Accuracy Ties vs. Minimum Loss**: For arms reaching 100% accuracy, maximum accuracy was tied across hundreds of checkpoints (337 ties for baseline, 396 ties for EMA). The first tie step does not reflect minimum loss. In `none_wd0.005`, best accuracy first occurred at step 9,600 (loss 0.042892), but minimum loss was 0.032604 at step 34,400. In `ema_wd0.005`, best accuracy first occurred at step 1,800 (loss 0.025220), but minimum loss was 0.005651 at step 8,800.
- **Censoring of Non-Crossers**: Neither `none_wd0` nor `ema_wd0` reached 95% full validation accuracy within stopping limits (capped at 50,000 updates and 41,919 updates respectively). They are censored at their last complete evaluations (step 50,000 at 93.9003% and step 41,900 at 61.3832%), not proven incapable of crossing given further updates.

## Crossing Comparisons & Timing Recovery

### 1. Zero Weight Decay (`none_wd0` vs `ema_wd0`)
- **Crossing Speedup**: Not estimable.
- **Reason**: Neither arm achieved the 0.95 validation accuracy threshold prior to stopping limits. Unequal update counts (50,000 vs 41,919) reflect per-arm deadline and cap constraints, not equal-effort endpoints.

### 2. Coupled Weight Decay 0.005 (`none_wd0.005` vs `ema_wd0.005`)
- **Update Ratio**: Baseline crossed 0.95 at update 5,900; EMA crossed at update 1,200.
  $$\text{Update Ratio} = \frac{5900}{1200} = 4.9166667 \approx 4.92\times$$
  *Note*: Crossing is observed on a 100-update evaluation grid and represents the first observed evaluation meeting threshold, not an exact latent crossing time.

- **Rejection of Old 0.6784x Time Ratio**:
  The original `study_summary.json` reported `work_time_ratio_baseline_over_ema: 0.6784258`. This was calculated as $2861.595496\text{s} / 4217.993332\text{s}$.
  **This ratio is invalid and rejected.** In `runner.py`, `work_elapsed_s` is measured from the global start of the entire study (`clock() - global_start`). Because the arms ran sequentially, EMA was executed fourth; its cumulative timestamp of 4,217.993s incorporated the execution time of all preceding arms (~4,179.194s of prior study runtime). Baseline ran third; its timestamp incorporated ~2,678.329s of prior runtime. Dividing these sequential cumulative timestamps compared non-coincident durations.

- **Recovered Valid Interval (Step 0 Eval END to First Crossing Eval END)**:
  A precise, well-defined timing interval can be recovered from the saved records:
  - **Baseline (`none_wd0.005`)**:
    - Step 0 validation END: $2678.328588667\text{ s}$
    - Crossing (step 5900) validation END: $2861.595496125\text{ s}$
    - Interval: $2861.595496125\text{ s} - 2678.328588667\text{ s} = 183.266907458\text{ s}$
  - **EMA (`ema_wd0.005`)**:
    - Step 0 validation END: $4179.193923250\text{ s}$
    - Crossing (step 1200) validation END: $4217.993332375\text{ s}$
    - Interval: $4217.993332375\text{ s} - 4179.193923250\text{ s} = 38.799409125\text{ s}$
  - **Interval Ratio**:
    $$\text{Ratio} = \frac{183.266907458\text{ s}}{38.799409125\text{ s}} = 4.72344584 \approx 4.72\times$$
  - **Inclusions**: Intervening training updates, periodic 100-step full validations, evaluation compute, and checkpoint/disk I/O between step 0 eval END and crossing eval END.
  - **Exclusions**: Initial model/data/optimizer setup and initial step 0 validation execution.
  - **Scope Limitation**: This is NOT a setup-inclusive arm runtime or whole-study speedup. Exact arm-start timestamps were not recorded by `runner.py`.

## Post-Crossing Stability & Reversal Analysis

Reaching 0.95 validation accuracy at an early checkpoint does not guarantee continuous generalization. Saved full-set validation records show subsequent temporary performance drops:

- **Baseline (`none_wd0.005`)**:
  - Post-crossing full evaluations: 404 evaluations.
  - Subsequent observations below 0.95: **12 observations**.
  - Post-crossing minimum accuracy: **57.3239%** (loss 1.271428 at update 11,900).
  - Other notable drops include updates 16,200 (65.94%), 17,200 (64.69%), and 30,500 (58.10%).

- **EMA (`ema_wd0.005`)**:
  - Post-crossing full evaluations: 426 evaluations.
  - Subsequent observations below 0.95: **3 observations**.
  - Post-crossing minimum accuracy: **85.5670%** (loss 0.529261 at update 19,700).
  - Other drops occurred at updates 2,200 (93.77%) and 9,900 (88.42%).

*Observation Definition Note*: These counts represent discrete 100-update grid evaluation points falling below 0.95 accuracy. They do not quantify time duration or distinct independent collapse events.
