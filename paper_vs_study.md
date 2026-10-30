# Paper versus this study

This page compares this study with the Grokfast paper, so the scope of the results is not read wider than it is.

Paper: Lee et al., "Grokfast: Accelerated Grokking by Amplifying Slow Gradients", https://arxiv.org/abs/2405.20233 (PDF: https://arxiv.org/pdf/2405.20233). Section and appendix references below are to that text.
Official code: https://github.com/ironjr/grokfast, commit `93b2248e351d4bb9e4dda8ac32a6999b8970404b` (see `THIRD_PARTY_NOTICES.md`).
This study: `README.md`, `study_outputs/approved_comparison/study_report_corrected.md` and `study_outputs/approved_comparison/config.json` in this repository.

This is the Grokfast paper only. The sibling grokking study reproduces a different paper (Power et al., 2022) and has its own page.

## Side by side

| Item | Paper | This study |
|---|---|---|
| Task | Modular binary multiplication `x * y mod 97` on algorithmic data, 50% of the data for training (Section 2.2, Appendix B.1, Figure 2). | Same task as implemented: `x * y % 97`, x in 0..96, y in 1..96, 9,312 examples, 50/50 split (`data.py`, `config.json`). |
| Model | Two-layer decoder-only transformer, hidden dimension 128, 4 heads (Appendix B.1). | Two layers, width 128, 4 heads, causal mask, pre-norm (`README.md`, `config.json`). |
| Base optimizer | Adam, betas (0.9, 0.98), constant learning rate 1e-3, batch size 512, linear warmup over the first 10 iterations (Appendix B.1). | Adam (not AdamW), lr 1e-3, betas (0.9, 0.98), batch 512, 10-update warmup. Weight decay is Adam's coupled `weight_decay`, set per arm (`README.md`). |
| Filter | Grokfast-MA (window w, gain lambda) and Grokfast-EMA (alpha, lambda). | EMA only: alpha 0.98, lambda 2.0, applied between backward and the optimizer step, state started from the first raw gradient. No MA arm. |
| Arms | Baseline against Grokfast with different hyperparameters, including weight-decay ablations (Figures 4, 7, 8). | Four arms: none or EMA, crossed with weight decay 0 or 0.005. The baseline has the same weight decay as its EMA partner. |
| Seeds | Table 2 and Figure 12b report five instances; Figure 13 (Appendix E) shows "five runs" of the Figure 2 experiment with different seeds. The text I read does not give a seed count for the headline curves in Figures 2 or 8. | One paired seed (seed 0): the model, split and batch order each use seed 0 in independent streams, identical across arms. Four arms is 4 runs. As arithmetic only, five seeds for each of the four arms would be 20 runs. That is an illustration, not a prescribed seed count. |
| Headline result | Figure 2 reports about 50x faster generalization with Grokfast-MA (annotated x50.49); Grokfast-EMA is annotated x43.84 faster in Figure 8. | EMA at weight decay 0.005 reached 95% validation accuracy at update 1,200, against 5,900 for its baseline: 4.92x in updates, first observed on a 100-update grid. |
| Budget | I did not find the paper's training budget in the text I read. The plotted step axes in Figures 4 to 8 run from 10^1 to about 10^5 steps; that is the plotted range, not a stated budget. | Cap of 50,000 updates and 1,500 s per arm, 6,000 s total. Arms stopped at 50,000, 41,919, 46,400 and 43,886 updates. |
| Metric | Iterations until validation accuracy reaches 0.95 (Section 2.2 and figure annotations). Exact evaluation interval not read here. | First full-set validation (4,656 examples, every 100 updates) at or above 0.95, from the saved records. |
| Hardware | The paper mentions a single 1080 Ti GPU for its timing comparison. | CPU on an M1 MacBook Air, 8 GB, Python 3.9.6, torch 2.8.0 (`environment.json`). |

## Where the numbers must not be mixed

- These ratios must not be treated as a like-for-like replication comparison: 4.92x here, x43.84 (Grokfast-EMA, Figure 8e) and x50.49 (Grokfast-MA with weight decay 0.01, Section 2.3, the Figure 2 demonstration) in the paper. The paper's plotted baseline reaches 0.95 at 39,890 iterations (the 3.99 x 10^4 label in Figure 8e), against 9.10 x 10^2 for the annotated Grokfast curve. The paper's text I read does not state that baseline's weight decay. It separately describes applying the same weight decay without Grokfast-MA (Figure 7) and says that makes training unstable. Ours compares EMA with a baseline that shares weight decay 0.005. At weight decay 0, neither of our arms reached 95% inside its limit, so no zero-decay ratio exists.
- 4.92x counts updates to a first observed crossing. 4.72x is a different, scoped quantity: the interval from the end of each arm's initial validation to the end of its first 95% evaluation (183.27 s for baseline, 38.80 s for EMA). It includes training, periodic validation and checkpoint I/O in that interval and excludes setup and the initial validation. It is not whole-run speedup. An earlier time ratio of 0.6784 was invalid and is rejected in the corrected report.
- Reaching 95% once does not mean staying there. After crossing, the weight-decay baseline had 12 later full evaluations below 0.95 and EMA had 3. These are counts of grid points, not independent events (`study_report_corrected.md`).
- The five-run trajectory figures in the paper are parameter-space visualizations. They are not seed counts for our accuracy comparison.

## What this study supports and does not

Supports: with one paired seed at weight decay 0.005, EMA reached 95% validation accuracy in fewer updates than the matched baseline, and both ended with 100% last complete evaluations.
Does not support: a general EMA speedup, the paper's 50x claim, a result for Grokfast-MA, a claim that weight decay alone explains the behavior, equal-effort endpoints at zero decay, or a replication of the whole paper. The public CI run (https://github.com/shiningshotgun/grokfast-forward/actions/runs/40989911192) runs exactly the 32 unit tests in tests/; two weak assertions are noted in the corrected report. The runtime is not a certified all-inclusive figure.

## Unknown or not checked

- The number of seeds behind the paper's headline curves in Figures 2 and 8.
- The weight decay of the paper's plotted baseline, which the text I read does not state.
- Whether the paper's evaluation interval matches our 100-update grid.
- Whether our result would hold across more seeds. The study makes no claim either way. This page does not set a required seed count.


## Addendum: the five-seed extension against the paper

The sections above compare the paper with the original single-seed study. This addendum covers the five-seed extension (five paired seeds, same four arms, 50,000 updates per arm, prospective analysis specification in docs/analysis_spec.md).

- Where the single-seed study could only illustrate, the extension gives a small-sample read at our scale. With weight decay 0.005, EMA crossed the sustained-99% threshold in all 5 seeds against 4 of 5 for baseline, with marginal medians of 44,700 updates (EMA) and 46,900 (baseline); the four pairs where both crossed had a median paired difference of -700 updates (EMA minus baseline) with a wide range (-29,300 to +23,700), including one seed where baseline crossed far earlier. The direction favors EMA on crossing; the scatter is large.
- With weight decay 0, baseline never crossed within the horizon (0 of 5 seeds), while EMA crossed in 4 of 5, late in training (marginal median 49,700). The single-seed study could not establish anything here because neither of its wd-0 arms crossed.
- Consistency, not validation: these results are directionally consistent with the acceleration the paper reports, at our smaller scale and different setup (the setup differences listed above still apply). Agreement with the paper is consistency between references, not independent validation of either. No causal claim and no cross-architecture generalization follows from five descriptive seeds.
- Method notes: crossing required sustained 99% validation accuracy through the final evaluation; arms not crossing were treated as right-censored, never substituted at the cap; marginal medians are reported only where at least 3 seeds crossed. An earlier analysis version dropped censored seeds from two marginal medians and was corrected; see docs/analysis_corrections.md.
