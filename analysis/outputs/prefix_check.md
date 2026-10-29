# Part B: Prefix-Consistency Check Report

Empirical examination of the recorded validation-accuracy curves between initial and amended-budget reruns.
Over the shared update range, recorded validation-accuracy curves at the shared evaluation points are compared.

**Scope Note**: This comparison covers exactly two arm-pairs: Seed 4 `none_wd0` (Attempt 1 vs Attempt 2) and Seed 1 `ema_wd0` (Attempt 2 vs Attempt 3). The other rerun arms were not part of this prefix comparison, and no claim is made about whether superseded records exist for them. Exact agreement of the recorded validation-accuracy curves over the shared prefix for these two arm-pairs supports (cannot prove) treating those two reruns as the same trajectory. For the other rerun arms, comparability rests on same-code/same-seed/same-cap reasoning, stated as an assumption.

### Seed 4: none_wd0 (Attempt 1 vs Attempt 2)
- **Record Source**: archive/superseded-attempts branch
- **Shared Maximum Update**: 47,000
- **Aligned Evaluation Count**: 471
- **Maximum Absolute Accuracy Difference**: 0.0000000000
- **Mean Absolute Accuracy Difference**: 0.0000000000
- **Curve Agreement**: Exact match of recorded validation accuracy across all shared evaluation points (max diff 0.0000000000)

### Seed 1: ema_wd0 (Attempt 2 vs Attempt 3)
- **Record Source**: archive/superseded-attempts branch
- **Shared Maximum Update**: 43,900
- **Aligned Evaluation Count**: 440
- **Maximum Absolute Accuracy Difference**: 0.0000000000
- **Mean Absolute Accuracy Difference**: 0.0000000000
- **Curve Agreement**: Exact match of recorded validation accuracy across all shared evaluation points (max diff 0.0000000000)
