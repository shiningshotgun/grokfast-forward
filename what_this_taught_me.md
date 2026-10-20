# What this taught me

This experiment gave me a result worth keeping. It also gave me a timing claim that was wrong. I don't need another training run to separate the two. I need to read what was actually recorded.

## Fewer updates and less time are different claims

With Adam's coupled weight decay at 0.005, the baseline first recorded at least 95% validation accuracy at update 5,900. EMA did it at 1,200. That's a 4.92x ratio in updates to the first observed crossing, on a 100-update evaluation grid.

The reported 0.68x time ratio was invalid. Each timestamp started at the beginning of the whole study. The arms ran sequentially, so the later arm's timestamp included the earlier arms. Dividing those timestamps doesn't compare how long the two arms took to learn.

The saved records still support a useful comparison. From the END of each arm's initial validation to the END of its first 95% evaluation, baseline took 183.27 seconds and EMA took 38.80 seconds. That's a 4.72x ratio over that interval. It includes the training, evaluations and I/O in between. It excludes initialization and the initial validation. Calling it whole-run speedup would be wrong again.

I need the start point, end point and denominator before I use a timing number. The calculation can be correct and the comparison can still be meaningless.

## The decay setting is part of the result

At decay 0.005, EMA crossed earlier and both arms eventually recorded 100% full-validation accuracy. Without decay, neither crossed 95% within its stopping limit. Baseline's last full result was 93.90%; EMA's was 61.38%.

Those zero-decay endpoints came from different limits: baseline reached 50,000 updates; EMA stopped at 41,919, with its last full evaluation at 41,900. They aren't equal-duration or equal-update endpoints.

The result is EMA helping the first crossing in this seed and decay setting. It isn't proof that EMA always helps, that decay alone explains the behavior, or that this reproduces the whole paper. Four arms make the comparison more informative. They don't turn one seed into a universal result.

## First crossing doesn't mean it stayed there

After first crossing 95%, the baseline had 12 later full evaluations below it. EMA had 3. The baseline fell as low as 57.32%; EMA fell to 85.57% after crossing.

Both last complete evaluations were 100%, but that doesn't erase those reversals. The claim I can make is earlier first observed crossing. I can't replace that with uninterrupted generalization.

The evaluation boundary matters too. Baseline's evaluation at 46,400 updates covered only 1,536 of 4,656 validation examples before the deadline. Its last complete evaluation was at 46,300. EMA's last complete evaluation was at 43,800, though training reached 43,886. A partial pass showing 100% is not a full-set 100% result.

I need to report the crossing, what happened afterward, and the last complete observation. One threshold number leaves too much out.

## Passing tests don't automatically prove the claim in their names

The saved log shows 32 tests passing. That matters. But the new test called chunked-versus-continuous state equivalence only checked that both paths reached 20 updates. It didn't compare the states. The new deadline test checked that an output directory existed, not that both deadline rules worked.

I need to read the assertions, not count green lines. A test needs to fail when the property it claims to protect is broken. Existing resume tests, source inspection and contiguous recorded updates support using this run. They don't make the weak assertions stronger or turn a saved log into an independent execution.

## Keeping the evidence made the correction possible

The raw metrics let me recover the timing interval, exclude the partial validation, count reversals and keep the actual result without retraining. The source fingerprints matched the verified model, data and filter files. The recorded work time was 94 minutes 40 seconds, but the timer omitted a little initial setup, so it isn't an exactly all-inclusive runtime claim.

For the next experiment, I want explicit arm-start and arm-end timestamps, separate setup/evaluation/checkpoint timing, and tests that compare the state they name. More seeds would answer a different question. They need a separate experiment, not a quiet extension of this one.

For this run, the raw evidence remains frozen and unchanged. The corrected summary and report are saved separately alongside the originals, without modifying the training records or starting another run.
