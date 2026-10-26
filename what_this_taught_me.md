# What this taught me

I ran a Grokfast-style experiment on my Mac with four arms: EMA of gradients on or off, weight decay 0.005 or 0. One seed. Recorded work time was 94 minutes 40 seconds, and the timer missed a little initial setup.

With weight decay 0.005, the baseline first reached 95% validation accuracy at update 5,900 and the EMA arm at update 1,200, a 4.92x difference in updates to first crossing on a 100-update evaluation grid. With no weight decay, neither arm crossed 95% before its stopping limit: baseline's last full evaluation was 93.90% at 50,000 updates, EMA's was 61.38% at 41,900, with training stopped at 41,919. The EMA result depends on the decay setting, and this was one seed.

My first timing claim, 0.68x, was wrong. Each arm's timestamps started at the beginning of the whole study and the arms ran one after another, so the later arm's clock included the earlier arms. Dividing those numbers doesn't compare the arms. The saved records do support a narrower interval: from the end of each arm's initial validation to the end of its first 95% evaluation, baseline took 183.27 seconds and EMA 38.80 seconds, 4.72x over that stretch. That stretch includes training, evaluations, and I/O, and excludes initialization. I should have logged explicit arm start and end times and broken timing out by phase from the beginning.

After first crossing 95%, the baseline fell back below it in 12 later full evaluations, EMA in 3. The baseline's lowest was 57.32%, EMA's 85.57%. Both last complete evaluations were 100%. Baseline's evaluation at 46,400 updates covered only 1,536 of 4,656 validation examples before the deadline stopped it, so for endpoint numbers I used its last complete evaluation at 46,300. EMA's last complete evaluation was at 43,800, with training reaching 43,886.

The suite logged 32 passing tests. Two of the new tests were weaker than their names: the chunked-versus-continuous test checked that both paths reached 20 updates and never compared the states, and the deadline test checked that an output directory existed. The run is still usable on the strength of the existing resume tests, source inspection, and contiguous update records; that doesn't fix the two weak tests.

All raw records from this run are frozen. The corrected summary and report sit alongside the originals. Next time: arm-level timestamps, timing by phase, and tests that compare the states they name.
