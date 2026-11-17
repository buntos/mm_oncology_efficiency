# Paired tissue-threshold study: fixed protocol v1

## Population and sampling
Source: frozen 950 paired TCGA LUAD/LUSC patients (474/476). Draw 30 per class
using ascending SHA256('WSI_PAIRED60_V1|20261111|' + case_submitter_id), with case
ID as tie-breaker. This draw includes all baseline QC statuses. Do not redraw,
replace failed patients, enlarge the sample after examining significance, or
present this balanced subsample as a new independent cohort. Prior stage-1
membership and TSS codes are recorded. The sample size is resource-constrained,
not a prospective power calculation. Descriptive/exploratory single-cohort study.

## Intervention
One fixed original primary DX1 per patient; no alternate-slide fallback at any
threshold. Thresholds 0.70, 0.75, 0.80, 0.85, 0.90. Native level 0, 256x256 pixels,
stride 256, candidate mask threshold 0.70, 64 patches. Original HSV/morphology,
background <=0.10, white <=0.10 and artifact rules are unchanged. First 64 valid
patches in the original deterministic order. Scan until all five thresholds
reach 64 or the candidate pool is exhausted. Every original PASS case must
reproduce 64 exact ordered coordinates and numerical QC metrics at 0.80; an
original excluded case must remain below 64 at 0.80. A mismatch blocks results.

## Primary endpoint and estimand
Primary contrast: 0.90 versus 0.80. For patients with BOTH full sets, calculate
replacement_fraction = 1 - |coordinates_080 intersection coordinates_090| / 64.
The primary estimate is the mean of this patient-level fraction in eligible
sampled patients. This is conditional on full-set availability; it does not
assign zero to patients who cannot supply a full set. Report the eligible n
and separate losses/gains/neither-complete counts alongside the endpoint.

## Secondary outcomes
Other threshold contrasts against 0.80 are exploratory. Report full-set
availability across the complete sample, coordinate Jaccard, patient-level
mean tissue/background/white fractions, paired differences in these means,
checked_to_64 difference and ratio. Per-threshold quality summaries condition
on a full 64-patch set; their populations can differ. Paired differences use
only patients complete at BOTH thresholds. Do not infer a quality change from
an unpaired comparison of different retained groups.

## Uncertainty
Unit of analysis: patient, never patch. 95% Wilson intervals for complete-set
proportions; 10000 deterministic percentile bootstrap patient resamples for
paired means (numpy quantile method='linear'). Overall bootstrap is stratified
by class among eligible pairs, preserving each stratum's eligible n. Class
analyses resample patients in that class. n<2 in any represented stratum:
no bootstrap interval; constant
observations produce a degenerate empirical interval, labelled explicitly.
No confirmatory p-value claims or multiple-comparison discovery. The ALL group
is the observed balanced subsample, NOT a prevalence-weighted 950-patient estimate.
Original cohort retention at 0.70-0.80 comes from the separate stage-1 analysis.

## Confounding and scope
Record native MPP, objective and TSS. Within-patient contrasts hold the slide
and resolution fixed; between-class differences can still reflect source/site,
scale or staining. TCGA-05 deserves explicit sensitivity discussion; no causal
histology claim from class contrasts. The tissue detector is not a tumor-cell
measurement or independent expert quality label. Coordinate replacement is not
proof of changed model embeddings or diagnostic accuracy. No classification or
clinical-quality claims without additional validation.

## Computation and missingness
One shared scan is used for five virtual stopping rules. checked_to_64 is a
candidate-count cost measure; shared scan_seconds is NOT a per-threshold time
benchmark. Failed thresholds have no checked_to_64, not zero. Candidate logs
contain sequential rejection reasons (first applicable filter), not independent
artifact/white/background prevalence. Completed insufficient sets are scientific
observations; interrupted/download/read failures are technical missingness.
Do not replace such patients. Final estimates/intervals require all 60 complete,
valid baseline controls and unchanged input hashes; partial files are labelled.
