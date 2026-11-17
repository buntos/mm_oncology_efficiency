# Stage 32: internal test evaluation

All nine previously selected checkpoints; the same 186 test patients and 64 patches per evaluation.
Selection used validation CE. No test-based selection or threshold tuning. LUAD=0, LUSC=1; threshold=0.5.

| Training K | Test AUROC mean ± seed SD | BA mean ± seed SD | CE mean |
|---|---|---|---|
| 64 | 0.8886 ± 0.0044 | 0.7926 ± 0.0030 | 0.4419 |
| 32 | 0.9169 ± 0.0067 | 0.8307 ± 0.0223 | 0.3776 |
| 16 | 0.9286 ± 0.0076 | 0.8495 ± 0.0103 | 0.3567 |

## Paired differences

- K16-K64 (primary): ΔAUROC=0.0400; 95% patient-bootstrap CI [0.0198, 0.0625].
- K32-K64 (secondary_exploratory): ΔAUROC=0.0284; 95% patient-bootstrap CI [0.0123, 0.0468].

Intervals condition on these three fitted seeds. They do not quantify the population of training seeds. Secondary intervals are pointwise exploratory. A CI containing zero does not establish equivalence or noninferiority.

## Measured training resources

| K | Training seconds/epoch (mean) | Full training session minutes (mean) | Peak allocated training GPU GiB (mean) |
|---|---|---|---|
| 64 | 191.93 | 56.88 | 1.298 |
| 32 | 100.76 | 67.69 | 0.808 |
| 16 | 53.32 | 54.00 | 0.556 |

Full training cost includes epochs after the selected epoch until the fixed stopping rule. Source lower-bound and replay flags are preserved in CSVs. GPU allocated, GPU reserved and process RSS are separate metrics. All evaluation budgets are 64: these results do not demonstrate inference-time savings or energy savings.

## Scope and limitations

- three seeds; internal patient split, not a held-out-site evaluation
- TSS/class confounding and heterogeneous physical patch scale
- 20 pilot-validation patients used in earlier development
- test has not informed this analysis plan; plan follows validation inspection
- training resource savings do not establish inference or energy savings
