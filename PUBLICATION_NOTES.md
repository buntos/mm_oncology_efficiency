# Publication copy and omitted assets

This copy preserves the original relative paths and bytes of the selected research files. Content checks and a CPU statistical replay have been completed; see PUBLICATION_REVIEW.md. The archive has not been tested as a complete installation on a fresh machine. Original protocols intentionally check exact hashes and environment properties.

Included: research scripts 01-32 (as present in the source inventory; numbering has gaps), cohort manifests and metadata, Stage 28 preparation, Stage 31/32 protocols and results, resource journals, diagnostic/pilot reports, and small pilot thumbnails. Historic failed smoke reports are retained as development history, not primary results. Archived checkpoint metadata can refer to omitted binary states and must not be mistaken for resumable runs. The cleanup script 23 is historical; it is not a reproduction prerequisite.

## Large assets outside Git

- `data/tiles/`: 59,456 PNG tiles. Reconstruct from the fixed GDC file IDs, slide selection, coordinates, processing settings, and hashes. This reconstruction must be checked before claiming end-to-end reproducibility.
- `data/rna/`: original RNA tables from the broader multimodal project; not model inputs for this article. Cohort metadata are retained.
- `results/figures/`: bulk QC masks and contact sheets omitted, except the small pilot thumbnail folder. Regenerate when needed.
- Nine `results/stage31_patch_budget_v1_1/runs/K{K}_seed{seed}/best_model.pt` files: intended for a separate `selected_models.zip` release asset. Extract at repository root preserving paths. Validate against the accompanying `best_model.json` hashes. No release URL exists yet.
- Official ResNet18 weights omitted: restore `results/wsi_patch_budget_v1/stage29_pilot_v1/weights/resnet18-f37072fd.pth` from https://download.pytorch.org/models/resnet18-f37072fd.pth and verify against the frozen protocol. These are third-party pretrained weights.
- `.venv/`, intermediate optimizer checkpoints, smoke-test weights, bundled executables, and ZIP snapshots omitted. Obtain GDC/OpenSlide tools from their official distributions.

## Before publication

The subsequent review inspected Python syntax, frozen inputs, saved predictions, resource summaries and common credential patterns in text files, with separate decoding of UTF-16 environment snapshots. No matching credential patterns were found; this is not a guarantee that every possible secret format was detected. Add a code license with the authors' agreement; a code license does not automatically cover TCGA data or third-party packages. Complete and test fresh-machine setup/reconstruction instructions before claiming end-to-end reproducibility. The added tools/verify_published_results.py provides a read-only CPU statistical replay. The existing Stage 32 `--analyze` still verifies model files and source protocols.

`.gitattributes` disables line-ending conversion to preserve SHA256 checks. Do not reformat frozen scripts or replace protocols to suppress integrity checks. Use a separate documented adaptation for a new machine or run series. `publication_checksums.json` records the bytes copied, not just file sizes.
