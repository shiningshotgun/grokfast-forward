# Grokfast-EMA comparison study

Small, self-contained CPU runner for the Grokfast modular-multiplication
grokking setup, with a resource pilot and a capped four-arm comparison study.

## Study outputs

The authoritative study results are in `study_outputs/approved_comparison/`:
- **`study_report_corrected.md`** and **`study_summary_corrected.json`** are the
  corrected, authoritative derived documents.
- `study_report.md` and `study_summary.json` are the original derived documents,
  kept unmodified for provenance (they contain an invalid timing ratio and other
  issues documented in the corrected files).
- `arm_*.json` are the raw per-arm metric records.
- `what_this_taught_me.md` (repo root) is the author's reflection on the study.

## Sources

Model/data/filter logic reused with attribution from `ironjr/grokfast`
(commit `93b2248e351d4bb9e4dda8ac32a6999b8970404b`):

- `main.py`, `grokfast.py`, `README.md` (see `config.json` in pilot output).

Upstream notices remain with the upstream project (MIT). Local adaptations are
documented in `grokfast.py`.

## Task

- Multiplication mod 97: `x * y % 97`, `x in 0..96`, `y in 1..96`
  (9312 examples, 50/50 train/valid split).
- Five tokens per example `[x, op, y, eq, result]`, loss only on result.
- Vocab size 99 (results 0..96 plus op/eq tokens).
- Model: 2 layers, width 128, 4 heads, causal mask, pre-norm residuals.
- Optimizer: Adam (not AdamW), lr 1e-3, betas (0.9, 0.98), batch 512.
- Scheduler: `LambdaLR(lambda update: 1 if update > 10 else update/10)` (10 updates).
- Weight decay is coupled (Adam `weight_decay`), set per arm.

## Arms (sequential, identical init/split/order)

1. `none_wd0` — no filter, weight_decay 0
2. `ema_wd0` — EMA filter, weight_decay 0
3. `none_wd0.005` — no filter, weight_decay 0.005
4. `ema_wd0.005` — EMA filter, weight_decay 0.005

EMA: alpha 0.98, lambda 2.0, applied after backward and before `optimizer.step()`.
State initialized from the first raw gradient (first modified grad = 3x raw).

## RNG / seeding

- Model init: `torch.manual_seed(0)` immediately before construction.
- Split: separate `torch.Generator().manual_seed(0)`.
- Batch order: separate `torch.Generator().manual_seed(0)`.
- These independent streams differ from upstream's single global stream, so exact
  upstream trajectories are not reproduced, but all arms share identical
  init/split/order.

Train metrics are recorded online per update (no repeated full-train passes).
Validation runs at steps 0 and 100 (if reached) on the full validation set.

## Running tests

```
PYTHONPATH=. /path/to/python -m unittest discover -s tests
```

## Running the pilot

```
python runner.py --pilot
```

Inspects existing records first; refuses to rerun if `pilot_outputs/` is non-empty.

## Pilot result

All four arms completed 100 updates. Recorded total elapsed 8.356290792 s
(limit 600 s). Peak RSS 322748416 bytes (~307.8 MiB, process peak, CPU).

Most updates use 512 examples and take ~0.019-0.020 s; every tenth update uses
the 48-example final partial batch and takes ~0.005-0.006 s. The latter is NOT
steady-state speed.

Per-arm 100-update means (from recorded update times):
- none/0: 0.0181545 s
- EMA/0: 0.0186113 s
- none/0.005: 0.0181268 s
- EMA/0.005: 0.0177931 s

All 400 updates sum 7.268574705 s. Recorded pilot total 8.356290792 s; the
residual 1.087716087 s combines setup, eight validation passes, checkpoint/I/O
and other overhead, NOT separately measured validation time. Final report writes
were outside that recorded total.

| Arm | Status | Updates | Val@0 acc | Val@100 acc |
|---|---|---|---|---|
| none_wd0 | completed_100_updates | 100 | 0.0120 | 0.0114 |
| ema_wd0 | completed_100_updates | 100 | 0.0120 | 0.0157 |
| none_wd0.005 | completed_100_updates | 100 | 0.0120 | 0.0146 |
| ema_wd0.005 | completed_100_updates | 100 | 0.0120 | 0.0178 |

This pilot establishes resources only, not grokking speedup.

## Future budget (no approved run)

The recorded means above are the only supported timing evidence. A 50k-update
per-arm extrapolation is ~61 minutes training alone across four arms, with no
measured long-run completion guarantee. No validation-pass duration was measured
separately. No run budget is approved here.
