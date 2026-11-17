from pathlib import Path
import re

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_DIR = ROOT / "data" / "manifests"

WSI_FILE = MANIFEST_DIR / "wsi_candidates.csv"
RNA_FILE = MANIFEST_DIR / "rna_star_counts_candidates.csv"
PAIRED_STAGE1_FILE = MANIFEST_DIR / "paired_cases_stage1.csv"


def first_entity_id(value):
    if pd.isna(value):
        return None

    value = str(value).strip()

    if not value:
        return None

    return value.split(";")[0].strip()


def sample_code(barcode):
    """
    Example:
    TCGA-44-6147-01A-01-TS1
                 ^^
                 01
    """

    if not barcode:
        return None

    parts = barcode.split("-")

    if len(parts) < 4:
        return None

    token = parts[3]

    if len(token) < 2:
        return None

    return token[:2]


def sample_vial(barcode):
    """
    Example:
    01A -> vial A
    01B -> vial B
    """

    if not barcode:
        return None

    parts = barcode.split("-")

    if len(parts) < 4:
        return None

    token = parts[3]

    if len(token) < 3:
        return None

    return token[2].upper()


def vial_rank(vial):
    if vial is None:
        return 999

    if len(vial) != 1 or not vial.isalpha():
        return 999

    return ord(vial.upper()) - ord("A")


def slide_token(barcode):
    """
    Returns e.g. DX1, TS1, BS1.
    """

    if not barcode:
        return None

    parts = barcode.split("-")

    if not parts:
        return None

    return parts[-1].upper()


def slide_prefix(token):
    if not token:
        return None

    match = re.match(r"([A-Z]+)", token)

    if not match:
        return None

    return match.group(1)


def dx_index(token):
    """
    DX1 -> 1
    DX2 -> 2
    """

    if not token:
        return 9999

    match = re.match(r"DX(\d+)$", token)

    if not match:
        return 9999

    return int(match.group(1))


# ============================================================
# LOAD STAGE 1 DATA
# ============================================================

wsi = pd.read_csv(WSI_FILE)
rna = pd.read_csv(RNA_FILE)
stage1 = pd.read_csv(PAIRED_STAGE1_FILE)

stage1_cases = set(
    stage1["case_submitter_id"].dropna()
)

wsi = wsi[
    wsi["case_submitter_id"].isin(stage1_cases)
].copy()

rna = rna[
    rna["case_submitter_id"].isin(stage1_cases)
].copy()


# ============================================================
# PARSE TCGA BIOSPECIMEN BARCODES
# ============================================================

wsi["associated_barcode"] = (
    wsi["associated_entity_submitter_ids"]
    .apply(first_entity_id)
)

rna["associated_barcode"] = (
    rna["associated_entity_submitter_ids"]
    .apply(first_entity_id)
)


wsi["sample_code"] = (
    wsi["associated_barcode"]
    .apply(sample_code)
)

rna["sample_code"] = (
    rna["associated_barcode"]
    .apply(sample_code)
)


wsi["slide_token"] = (
    wsi["associated_barcode"]
    .apply(slide_token)
)

wsi["slide_prefix"] = (
    wsi["slide_token"]
    .apply(slide_prefix)
)

wsi["dx_index"] = (
    wsi["slide_token"]
    .apply(dx_index)
)


rna["sample_vial"] = (
    rna["associated_barcode"]
    .apply(sample_vial)
)

rna["vial_rank"] = (
    rna["sample_vial"]
    .apply(vial_rank)
)


# ============================================================
# FILTER PRIMARY TUMOR DIAGNOSTIC WSI
# ============================================================

wsi_primary_dx = wsi[
    (wsi["sample_code"] == "01")
    &
    (wsi["slide_prefix"] == "DX")
].copy()


# ============================================================
# FILTER PRIMARY TUMOR RNA
# ============================================================

rna_primary = rna[
    rna["sample_code"] == "01"
].copy()


wsi_primary_dx.to_csv(
    MANIFEST_DIR / "wsi_primary_diagnostic_candidates.csv",
    index=False,
)

rna_primary.to_csv(
    MANIFEST_DIR / "rna_primary_tumor_candidates.csv",
    index=False,
)


# ============================================================
# DETERMINISTIC WSI SELECTION
#
# Priority:
#   DX1 -> DX2 -> DX3 ...
#   barcode
#   UUID
# ============================================================

wsi_selected = (
    wsi_primary_dx
    .sort_values(
        by=[
            "case_submitter_id",
            "dx_index",
            "associated_barcode",
            "file_id",
        ],
        kind="stable",
    )
    .drop_duplicates(
        subset="case_submitter_id",
        keep="first",
    )
    .copy()
)


# ============================================================
# DETERMINISTIC RNA SELECTION
#
# Priority:
#   vial A -> B -> C ...
#   aliquot barcode
#   UUID
# ============================================================

rna_selected = (
    rna_primary
    .sort_values(
        by=[
            "case_submitter_id",
            "vial_rank",
            "associated_barcode",
            "file_id",
        ],
        kind="stable",
    )
    .drop_duplicates(
        subset="case_submitter_id",
        keep="first",
    )
    .copy()
)


# ============================================================
# BUILD SELECTED PAIRED COHORT
# ============================================================

wsi_cases = set(
    wsi_selected["case_submitter_id"]
)

rna_cases = set(
    rna_selected["case_submitter_id"]
)

final_cases = sorted(
    wsi_cases.intersection(rna_cases)
)


wsi_keep = wsi_selected[
    wsi_selected["case_submitter_id"].isin(final_cases)
][[
    "case_submitter_id",
    "project_id",

    "file_id",
    "file_name",
    "file_size_bytes",
    "file_size_gb",
    "md5sum",

    "associated_barcode",
    "sample_code",
    "slide_token",
]].copy()


wsi_keep = wsi_keep.rename(columns={
    "file_id": "wsi_file_id",
    "file_name": "wsi_file_name",
    "file_size_bytes": "wsi_file_size_bytes",
    "file_size_gb": "wsi_file_size_gb",
    "md5sum": "wsi_md5sum",
    "associated_barcode": "wsi_barcode",
    "sample_code": "wsi_sample_code",
    "slide_token": "wsi_slide_token",
})


rna_keep = rna_selected[
    rna_selected["case_submitter_id"].isin(final_cases)
][[
    "case_submitter_id",

    "file_id",
    "file_name",
    "file_size_bytes",
    "file_size_gb",
    "md5sum",

    "associated_barcode",
    "sample_code",
    "sample_vial",
]].copy()


rna_keep = rna_keep.rename(columns={
    "file_id": "rna_file_id",
    "file_name": "rna_file_name",
    "file_size_bytes": "rna_file_size_bytes",
    "file_size_gb": "rna_file_size_gb",
    "md5sum": "rna_md5sum",
    "associated_barcode": "rna_barcode",
    "sample_code": "rna_sample_code",
    "sample_vial": "rna_sample_vial",
})


cohort = wsi_keep.merge(
    rna_keep,
    on="case_submitter_id",
    how="inner",
    validate="one_to_one",
)


cohort["class_label"] = cohort["project_id"].map({
    "TCGA-LUAD": "LUAD",
    "TCGA-LUSC": "LUSC",
})


cohort = cohort.sort_values(
    "case_submitter_id"
).reset_index(drop=True)


cohort.to_csv(
    MANIFEST_DIR / "paired_cohort_stage2.csv",
    index=False,
)


# ============================================================
# BUILD EXCLUSION TABLE
# ============================================================

primary_dx_cases = set(
    wsi_primary_dx["case_submitter_id"]
)

primary_rna_cases = set(
    rna_primary["case_submitter_id"]
)


exclusion_rows = []

for case in sorted(stage1_cases):

    has_wsi = case in primary_dx_cases
    has_rna = case in primary_rna_cases

    if has_wsi and has_rna:
        continue

    reasons = []

    if not has_wsi:
        reasons.append(
            "no_primary_tumor_diagnostic_WSI"
        )

    if not has_rna:
        reasons.append(
            "no_primary_tumor_RNA"
        )

    exclusion_rows.append({
        "case_submitter_id": case,
        "reason": ";".join(reasons),
    })


exclusions = pd.DataFrame(exclusion_rows)

exclusions.to_csv(
    MANIFEST_DIR / "cohort_exclusions_stage2.csv",
    index=False,
)


# ============================================================
# SUMMARY STATISTICS
# ============================================================

multi_wsi = (
    wsi_primary_dx
    .groupby("case_submitter_id")
    .size()
)

multi_rna = (
    rna_primary
    .groupby("case_submitter_id")
    .size()
)


print()
print("=" * 68)
print(" TCGA-LUAD / TCGA-LUSC — STAGE 2 COHORT SELECTION")
print("=" * 68)

print(
    f"Stage 1 paired patients:             "
    f"{len(stage1_cases)}"
)

print("-" * 68)

print(
    f"Primary diagnostic WSI files:        "
    f"{len(wsi_primary_dx)}"
)

print(
    f"Patients with primary diagnostic WSI:"
    f" {wsi_primary_dx['case_submitter_id'].nunique()}"
)

print(
    f"Patients with >1 diagnostic WSI:     "
    f"{(multi_wsi > 1).sum()}"
)

print("-" * 68)

print(
    f"Primary tumor RNA files:             "
    f"{len(rna_primary)}"
)

print(
    f"Patients with primary tumor RNA:     "
    f"{rna_primary['case_submitter_id'].nunique()}"
)

print(
    f"Patients with >1 primary RNA:        "
    f"{(multi_rna > 1).sum()}"
)

print("-" * 68)

print(
    f"SELECTED PAIRED COHORT:              "
    f"{len(cohort)}"
)

print("\nPatients by class:")

class_counts = (
    cohort["class_label"]
    .value_counts()
    .sort_index()
)

for label, count in class_counts.items():
    print(
        f"  {label:<6} {count}"
    )


print("\nSelected WSI diagnostic slide indices:")

dx_counts = (
    cohort["wsi_slide_token"]
    .value_counts()
    .sort_index()
)

for slide, count in dx_counts.items():
    print(
        f"  {slide:<6} {count}"
    )


print("\nSelected RNA vial distribution:")

vial_counts = (
    cohort["rna_sample_vial"]
    .value_counts(dropna=False)
    .sort_index()
)

for vial, count in vial_counts.items():
    print(
        f"  vial {vial}: {count}"
    )


print("\nSelected data size:")

print(
    f"  WSI: "
    f"{cohort['wsi_file_size_bytes'].sum() / 1024**3:.2f} GB"
)

print(
    f"  RNA: "
    f"{cohort['rna_file_size_bytes'].sum() / 1024**3:.2f} GB"
)


print("\nExcluded Stage 1 patients:")

print(
    f"  {len(stage1_cases) - len(cohort)}"
)


if not exclusions.empty:

    print("\nExclusion reasons:")

    reason_counts = (
        exclusions["reason"]
        .value_counts()
    )

    for reason, count in reason_counts.items():
        print(
            f"  {reason}: {count}"
        )


print("\nSaved:")

print(
    "  data/manifests/"
    "wsi_primary_diagnostic_candidates.csv"
)

print(
    "  data/manifests/"
    "rna_primary_tumor_candidates.csv"
)

print(
    "  data/manifests/"
    "paired_cohort_stage2.csv"
)

print(
    "  data/manifests/"
    "cohort_exclusions_stage2.csv"
)

print()
print(
    "NOTE: Stage 2 is the deterministic file-level cohort."
)

print(
    "It becomes final only after file availability/QC checks."
)

print("=" * 68)