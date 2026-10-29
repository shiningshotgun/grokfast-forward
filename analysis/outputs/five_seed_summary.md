# Five-Seed Comparative Analysis: Grokfast-EMA vs Baseline

## 1. Study Configuration & Prospective Methodology
- **Evaluation Grid**: 501 evaluation steps on [0, 50,000] updates (every 100 updates).
- **Primary Crossing Threshold**: First step with sustained validation accuracy $\ge 0.99$ (never falling below 0.99 through step 50,000).
- **Censoring Rule**: Non-crossing arms are strictly right-censored at $> 50,000$ updates.
- **Sign Convention**: $\Delta_{\text{cross}} = \text{EMA} - \text{none}$ (negative favors EMA); $\Delta_{\text{acc}} = \text{EMA} - \text{none}$ (positive favors EMA).
- **Accuracy Difference Units**: Reported as percentage points (pp).
- **Budget Disclosure & Trajectory Comparability**: Seeds 1 (attempt 3) and 4 (attempt 2) were run under amended 2,100s/arm budgets (total 9,000s); Seeds 2, 3, 5 under 1,500s/arm budgets (total 6,000s). All 5 seeds executed under update cap 50,000 with identical random seed streams and hyperparameters. Exact agreement of recorded validation-accuracy curves over the shared prefix for the two reported arm-pairs supports (cannot prove) treating those two reruns as the same trajectory. For the other rerun arms, comparability rests on same-code/same-seed/same-cap reasoning, stated as an assumption.

## 2. Per-Seed / Per-Arm Results Table

| Seed | Arm | Filter | Weight Decay | Crossing Step ($\ge 0.99$ sustained) | Final Val Acc (Step 50k) |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | `none_wd0` | none | 0.0 | **RIGHT-CENSORED** | 79.60% |
| 1 | `ema_wd0` | ema | 0.0 | 46,800 | 99.55% |
| 1 | `none_wd0.005` | none | 0.005 | 41,300 | 100.00% |
| 1 | `ema_wd0.005` | ema | 0.005 | 40,200 | 100.00% |
| 2 | `none_wd0` | none | 0.0 | **RIGHT-CENSORED** | 68.71% |
| 2 | `ema_wd0` | ema | 0.0 | 49,900 | 99.01% |
| 2 | `none_wd0.005` | none | 0.005 | 49,000 | 100.00% |
| 2 | `ema_wd0.005` | ema | 0.005 | 48,700 | 100.00% |
| 3 | `none_wd0` | none | 0.0 | **RIGHT-CENSORED** | 98.48% |
| 3 | `ema_wd0` | ema | 0.0 | 49,700 | 99.59% |
| 3 | `none_wd0.005` | none | 0.005 | 46,900 | 100.00% |
| 3 | `ema_wd0.005` | ema | 0.005 | 17,600 | 100.00% |
| 4 | `none_wd0` | none | 0.0 | **RIGHT-CENSORED** | 85.42% |
| 4 | `ema_wd0` | ema | 0.0 | **RIGHT-CENSORED** | 73.24% |
| 4 | `none_wd0.005` | none | 0.005 | **RIGHT-CENSORED** | 94.57% |
| 4 | `ema_wd0.005` | ema | 0.005 | 48,700 | 100.00% |
| 5 | `none_wd0` | none | 0.0 | **RIGHT-CENSORED** | 86.86% |
| 5 | `ema_wd0` | ema | 0.0 | 49,400 | 99.55% |
| 5 | `none_wd0.005` | none | 0.005 | 21,000 | 100.00% |
| 5 | `ema_wd0.005` | ema | 0.005 | 44,700 | 100.00% |

## 3. Paired Comparisons within Weight Decay Levels

| Weight Decay | Seed | Classification | Baseline Crossing | EMA Crossing | $\Delta_{\text{cross}}$ (EMA - none) | Baseline Final Acc | EMA Final Acc | $\Delta_{\text{acc}}$ (EMA - none) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0.0 | 1 | `EMA-only-cross` | CENSORED | 46,800 | N/A (censored) | 79.60% | 99.55% | +19.95 pp |
| 0.0 | 2 | `EMA-only-cross` | CENSORED | 49,900 | N/A (censored) | 68.71% | 99.01% | +30.30 pp |
| 0.0 | 3 | `EMA-only-cross` | CENSORED | 49,700 | N/A (censored) | 98.48% | 99.59% | +1.12 pp |
| 0.0 | 4 | `neither` | CENSORED | CENSORED | N/A (censored) | 85.42% | 73.24% | -12.18 pp |
| 0.0 | 5 | `EMA-only-cross` | CENSORED | 49,400 | N/A (censored) | 86.86% | 99.55% | +12.69 pp |
| 0.005 | 1 | `both-cross` | 41,300 | 40,200 | -1,100 | 100.00% | 100.00% | +0.00 pp |
| 0.005 | 2 | `both-cross` | 49,000 | 48,700 | -300 | 100.00% | 100.00% | +0.00 pp |
| 0.005 | 3 | `both-cross` | 46,900 | 17,600 | -29,300 | 100.00% | 100.00% | +0.00 pp |
| 0.005 | 4 | `EMA-only-cross` | CENSORED | 48,700 | N/A (censored) | 94.57% | 100.00% | +5.43 pp |
| 0.005 | 5 | `both-cross` | 21,000 | 44,700 | 23,700 | 100.00% | 100.00% | +0.00 pp |

## 4. Aggregate Comparison & Sign Consistency

### Weight Decay = 0.0
- **Pair Quadrants**: `both-cross` = 0/5, `EMA-only-cross` = 4/5, `none-only-cross` = 0/5, `neither` = 1/5.
- **Marginal Crossing Median (Baseline)**: not estimable within the 50,000-update horizon (0/5 crossed)
- **Marginal Crossing Median (EMA)**: 49,700
- **Conditional Paired Crossing Difference**: Not estimable (0 pairs in `both-cross`).
- **Directional Speed Advantage**: 4 of 5 seeds (in 0 of 0 uncensored pairs EMA crossed earlier, and in 4 pair(s) EMA crossed while baseline was censored).
- **Final Validation Accuracy Difference (EMA - none)**: median = +12.69 pp, range = [-12.18 pp, +30.30 pp].
- **Sign Consistency for Final Accuracy (EMA > none)**: 4 of 5 seeds.

### Weight Decay = 0.005
- **Pair Quadrants**: `both-cross` = 4/5, `EMA-only-cross` = 1/5, `none-only-cross` = 0/5, `neither` = 0/5.
- **Marginal Crossing Median (Baseline)**: 46,900
- **Marginal Crossing Median (EMA)**: 44,700
- **Conditional Paired Crossing Difference** (conditional subset $n = 4/5$ pairs): median = -700.0 updates, range = [-29,300, 23,700].
- **Directional Speed Advantage**: 4 of 5 seeds (in 3 of 4 uncensored pairs EMA crossed earlier, and in 1 pair(s) EMA crossed while baseline was censored).
- **Final Validation Accuracy Difference (EMA - none)**: median = +0.00 pp, range = [+0.00 pp, +5.43 pp].
- **Sign Consistency for Final Accuracy (EMA > none)**: 1 of 5 seeds.
