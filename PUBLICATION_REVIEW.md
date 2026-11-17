# Publication review, 17 November 2026

- Incoming archive: 8,582 files, 198.30 MiB uncompressed.
- All 8,577 copied research files match their packaging SHA256 values.
- Nine model files are intentionally external; their hashes agree between publication_checksums.json, best_model.json and the fixed Stage32 model registry (metadata verification only; model bytes not supplied here).
- Stage28 loader passed: 929 patients, 59,456 patch records; frozen split and manifests verified without opening PNGs.
- Python source syntax parsed without errors.
- No matches for the scanned private-key, GitHub-token, AWS-key or quoted password/API-secret assignment patterns in the inspected text. Environment snapshots were also inspected after UTF-16 decoding. This is a bounded pattern scan, not a proof of absence of secrets.
- Added CPU verification passed: all nine runs / 186 patients, point metrics and budget means, primary/secondary paired AUROC intervals with 10,000 resamples, and per-run resource totals. NumPy used for this check: 2.3.5.
- Statistical functions are reused from hash-verified Stage32; the replay does not constitute independent algorithmic validation, image inference, GPU benchmark replication or data regeneration.

## Changes in this revision

Added tools/verify_published_results.py and this review. Updated README.md and PUBLICATION_NOTES.md. Original scripts, protocols, predictions and results are unchanged. Added publication_added_files_sha256.json for the publication-only files. publication_checksums.json remains the original copy manifest, including separate model assets.

## Remaining publication steps

Choose a code license with both authors; no license has been silently assigned. Create the GitHub repository and upload selected_models.zip as a release asset. Add the actual release URL and article-version tag/commit to README and manuscript. External TCGA data and third-party weights retain their own applicable terms.

The full original training workflow needs data and pretrained/final weights. Archived failed smoke runs and checkpoint commit metadata without state.pt remain historical records, not runnable checkpoints. For a new research run use a separate documented setup; do not bypass frozen integrity checks.
