# Resource benchmark: frozen follow-up protocol v1.0.0

## Scope and selection
Post-stage26 follow-up, exploratory, not external validation. First 10 patients
per class by previously frozen class_sample_rank from the stage26 30+30 draw.
No replacement or selection by observed effect. Includes original QC failures.
The resource-constrained sample size is not a power calculation.
Thresholds 0.70/0.80/0.90, 64 patches, same DX1, native level0 256x256,
original candidate mask/order and all other QC unchanged. Tissue is not tumor.
Existing stage26 primary endpoint remains replacement_fraction 0.90 vs 0.80.

## Benchmark endpoints
Main benchmark endpoint: mean patient-level wall-time ratio 0.90/0.80 among
patients reaching 64 at BOTH thresholds. Prespecified additional contrast:
0.70/0.80. First take the median of three repeats at each threshold per patient,
then form patient ratios; do not pool repeat observations as independent data.
Secondary: process CPU time and OS peak resident memory ratio/difference.
Also report all-patient attempt costs including candidate exhaustion and
counts of both complete, baseline only, target only and neither complete.
An unsuccessful attempt is not time-to-64 and is not a technical error.
Overall percentile 95% CI: 10000 class-stratified patient bootstrap resamples,
fixed seed. No p-value search or arbitrary outlier removal. CI is conditional
on sample and benchmark environment; three repeats describe runtime variability.

## Execution and cache
Three repeats. Per patient, a cyclic Latin schedule gives each threshold each
position once. Base permutation cycles over six permutations across patients.
Each threshold/round: one untimed warm-up in a fresh subprocess, followed by
one measured fresh subprocess, same WSI/threshold. Warm-up is validated too.
OS caches are not flushed; actual OS residency cannot be guaranteed. This is
warm-cache-oriented local processing, not cold-start or network latency.
One worker at a time, native numerical thread pools set to one before imports.
Use same machine, disk, power plan and low background load throughout.

## Measurement boundary
perf_counter: from before OpenSlide open through close. process_time: same
interval, all CPU threads of this process. Includes thumbnail, tissue mask,
candidate list/order, region decoding, QC and first-64 selection/exhaustion.
Excludes startup/imports, downloads/MD5, warm-up, reference comparisons and
JSON/log output. No candidate CSV writes, patch saving, RNA or model inference.
Native artifact metrics run only after tissue/background/white pass for the
single current threshold, preserving original sequential filtering semantics.
Compare ordered coordinates, accepted QC metrics, stopping/rejection counts,
candidate pool size and seed with stage26 for EVERY warm-up and measurement.

## Memory interpretation
Windows GetProcessMemoryInfo PeakWorkingSetSize; Linux ru_maxrss converted
from KiB (macOS bytes). Process-lifetime peak through scan completion includes
Python and imports; also retain pre-scan peak. Fresh processes exclude prior
warm-up allocations. No claims about interval peak, energy or GPU memory.

## Recovery and provenance
Code/dependency/machine identity, source inputs, stage26 configuration and
selected patient checkpoint/candidate evidence hashes are frozen. Technical
failures stop and remain distinct from QC. Complete patients resume unchanged;
partial patients restart ALL rounds, never selecting fastest repeats. Writes
only to benchmark output; deletes only owned downloaded WSI after checkpoint.
Final aggregate estimates are suppressed until all 20 patients validate.
