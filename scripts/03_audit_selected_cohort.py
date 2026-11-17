from pathlib import Path
import hashlib
import json
import shutil

import pandas as pd
import requests


BASE_URL = "https://api.gdc.cancer.gov"

ROOT = Path(__file__).resolve().parents[1]

MANIFEST_DIR = ROOT / "data" / "manifests"
METADATA_DIR = ROOT / "data" / "metadata"
LOG_DIR = ROOT / "logs"

COHORT_FILE = MANIFEST_DIR / "paired_cohort_stage2.csv"
WSI_CANDIDATES_FILE = MANIFEST_DIR / "wsi_candidates.csv"
RNA_CANDIDATES_FILE = MANIFEST_DIR / "rna_star_counts_candidates.csv"

METADATA_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)


def sha256_file(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


def fmt_gb(value):
    return f"{value / 1024**3:.2f} GB"


def query_live_files(file_ids, batch_size=200):
    """
    Re-query selected UUIDs from GDC to verify that the
    files are still present and metadata still match.
    """

    all_hits = []

    for start in range(0, len(file_ids), batch_size):
        batch = file_ids[start:start + batch_size]

        filters = {
            "op": "in",
            "content": {
                "field": "files.file_id",
                "value": batch,
            },
        }

        payload = {
            "filters": filters,
            "format": "JSON",
            "fields": ",".join([
                "file_id",
                "file_name",
                "file_size",
                "md5sum",
                "access",
                "state",
                "data_type",
                "data_format",
                "analysis.workflow_type",
            ]),
            "size": str(len(batch) + 10),
        }

        response = requests.post(
            f"{BASE_URL}/files",
            json=payload,
            timeout=180,
        )

        response.raise_for_status()

        result = response.json()
        hits = result["data"]["hits"]

        all_hits.extend(hits)

        print(
            f"  verified {min(start + batch_size, len(file_ids))}"
            f"/{len(file_ids)} UUIDs"
        )

    return pd.DataFrame(all_hits)


# ============================================================
# LOAD
# ============================================================

cohort = pd.read_csv(COHORT_FILE)

wsi_candidates = pd.read_csv(WSI_CANDIDATES_FILE)
rna_candidates = pd.read_csv(RNA_CANDIDATES_FILE)


# ============================================================
# INTERNAL CONSISTENCY
# ============================================================

problems = []

if cohort["case_submitter_id"].duplicated().any():
    problems.append("duplicate patient IDs")

if cohort["wsi_file_id"].duplicated().any():
    problems.append("duplicate selected WSI UUIDs")

if cohort["rna_file_id"].duplicated().any():
    problems.append("duplicate selected RNA UUIDs")


expected_labels = {
    "TCGA-LUAD": "LUAD",
    "TCGA-LUSC": "LUSC",
}

bad_labels = cohort[
    cohort["class_label"]
    != cohort["project_id"].map(expected_labels)
]

if not bad_labels.empty:
    problems.append("project/class label mismatch")


if not cohort["wsi_sample_code"].astype(str).eq("1").all():
    # pandas may read "01" as integer 1
    bad = cohort[
        ~cohort["wsi_sample_code"].astype(str).isin(["1", "01"])
    ]

    if not bad.empty:
        problems.append("non-primary-tumor WSI found")


if not cohort["rna_sample_code"].astype(str).eq("1").all():
    bad = cohort[
        ~cohort["rna_sample_code"].astype(str).isin(["1", "01"])
    ]

    if not bad.empty:
        problems.append("non-primary-tumor RNA found")


bad_dx = cohort[
    ~cohort["wsi_slide_token"]
    .astype(str)
    .str.startswith("DX")
]

if not bad_dx.empty:
    problems.append("non-diagnostic WSI found")


required_columns = [
    "case_submitter_id",
    "project_id",
    "class_label",

    "wsi_file_id",
    "wsi_file_name",
    "wsi_file_size_bytes",
    "wsi_md5sum",
    "wsi_barcode",

    "rna_file_id",
    "rna_file_name",
    "rna_file_size_bytes",
    "rna_md5sum",
    "rna_barcode",
]

for col in required_columns:
    if cohort[col].isna().any():
        problems.append(f"missing values in {col}")


# ============================================================
# RESTORE ACCESS / STATE FROM STAGE 1 TABLES
# ============================================================

wsi_info = (
    wsi_candidates[
        [
            "file_id",
            "access",
            "state",
        ]
    ]
    .drop_duplicates("file_id")
    .rename(columns={
        "file_id": "wsi_file_id",
        "access": "wsi_access_stage1",
        "state": "wsi_state_stage1",
    })
)

rna_info = (
    rna_candidates[
        [
            "file_id",
            "access",
            "state",
        ]
    ]
    .drop_duplicates("file_id")
    .rename(columns={
        "file_id": "rna_file_id",
        "access": "rna_access_stage1",
        "state": "rna_state_stage1",
    })
)

audit = (
    cohort
    .merge(
        wsi_info,
        on="wsi_file_id",
        how="left",
        validate="one_to_one",
    )
    .merge(
        rna_info,
        on="rna_file_id",
        how="left",
        validate="one_to_one",
    )
)


# ============================================================
# LIVE GDC VERIFICATION
# ============================================================

selected_ids = (
    cohort["wsi_file_id"].tolist()
    +
    cohort["rna_file_id"].tolist()
)

print()
print("Re-validating selected UUIDs against live GDC API...")

live = query_live_files(selected_ids)

live.to_csv(
    METADATA_DIR / "selected_files_live_gdc_stage3.csv",
    index=False,
)


found_ids = set(live["file_id"])
expected_ids = set(selected_ids)

missing_live = sorted(expected_ids - found_ids)

if missing_live:
    problems.append(
        f"{len(missing_live)} selected UUIDs missing from live GDC"
    )


# ============================================================
# COMPARE SAVED METADATA WITH CURRENT GDC
# ============================================================

inventory_rows = []

for _, row in cohort.iterrows():

    inventory_rows.append({
        "case_submitter_id": row["case_submitter_id"],
        "project_id": row["project_id"],
        "class_label": row["class_label"],
        "modality": "WSI",
        "file_id": row["wsi_file_id"],
        "file_name": row["wsi_file_name"],
        "file_size": int(row["wsi_file_size_bytes"]),
        "md5sum": row["wsi_md5sum"],
    })

    inventory_rows.append({
        "case_submitter_id": row["case_submitter_id"],
        "project_id": row["project_id"],
        "class_label": row["class_label"],
        "modality": "RNA",
        "file_id": row["rna_file_id"],
        "file_name": row["rna_file_name"],
        "file_size": int(row["rna_file_size_bytes"]),
        "md5sum": row["rna_md5sum"],
    })


inventory = pd.DataFrame(inventory_rows)


live_compare = live[
    [
        "file_id",
        "file_name",
        "file_size",
        "md5sum",
        "access",
        "state",
    ]
].copy()

live_compare = live_compare.rename(columns={
    "file_name": "live_file_name",
    "file_size": "live_file_size",
    "md5sum": "live_md5sum",
    "access": "live_access",
    "state": "live_state",
})


inventory = inventory.merge(
    live_compare,
    on="file_id",
    how="left",
    validate="one_to_one",
)


inventory["name_match"] = (
    inventory["file_name"]
    == inventory["live_file_name"]
)

inventory["size_match"] = (
    inventory["file_size"]
    == inventory["live_file_size"]
)

inventory["md5_match"] = (
    inventory["md5sum"]
    == inventory["live_md5sum"]
)


metadata_mismatches = inventory[
    ~(
        inventory["name_match"]
        &
        inventory["size_match"]
        &
        inventory["md5_match"]
    )
].copy()


if not metadata_mismatches.empty:
    problems.append(
        f"{len(metadata_mismatches)} live metadata mismatches"
    )


inventory.to_csv(
    MANIFEST_DIR / "selected_file_inventory_stage3.csv",
    index=False,
)

metadata_mismatches.to_csv(
    MANIFEST_DIR / "metadata_mismatches_stage3.csv",
    index=False,
)


# ============================================================
# SIZE ANALYSIS
# ============================================================

wsi_sizes = cohort["wsi_file_size_bytes"]
rna_sizes = cohort["rna_file_size_bytes"]

largest_wsi = (
    cohort[
        [
            "case_submitter_id",
            "class_label",
            "wsi_file_name",
            "wsi_file_size_bytes",
        ]
    ]
    .sort_values(
        "wsi_file_size_bytes",
        ascending=False,
    )
    .head(20)
    .copy()
)

largest_wsi["wsi_file_size_gb"] = (
    largest_wsi["wsi_file_size_bytes"]
    / 1024**3
)

largest_wsi.to_csv(
    MANIFEST_DIR / "largest_wsi_stage3.csv",
    index=False,
)


# ============================================================
# DISK
# ============================================================

disk = shutil.disk_usage(ROOT)

free_gb = disk.free / 1024**3


# ============================================================
# SHA256
# ============================================================

cohort_sha256 = sha256_file(COHORT_FILE)


# ============================================================
# SAVE AUDIT SUMMARY JSON
# ============================================================

summary = {
    "patients": int(len(cohort)),
    "luad": int((cohort["class_label"] == "LUAD").sum()),
    "lusc": int((cohort["class_label"] == "LUSC").sum()),

    "selected_files": int(len(inventory)),
    "wsi_files": int(len(cohort)),
    "rna_files": int(len(cohort)),

    "wsi_total_gb": float(wsi_sizes.sum() / 1024**3),
    "rna_total_gb": float(rna_sizes.sum() / 1024**3),

    "wsi_min_gb": float(wsi_sizes.min() / 1024**3),
    "wsi_median_gb": float(wsi_sizes.median() / 1024**3),
    "wsi_mean_gb": float(wsi_sizes.mean() / 1024**3),
    "wsi_p95_gb": float(wsi_sizes.quantile(0.95) / 1024**3),
    "wsi_max_gb": float(wsi_sizes.max() / 1024**3),

    "wsi_over_1gb": int((wsi_sizes > 1024**3).sum()),
    "wsi_over_2gb": int((wsi_sizes > 2 * 1024**3).sum()),

    "live_uuid_missing": int(len(missing_live)),
    "metadata_mismatches": int(len(metadata_mismatches)),

    "free_disk_gb": float(free_gb),

    "cohort_sha256": cohort_sha256,

    "problems": problems,
}


(METADATA_DIR / "cohort_audit_stage3.json").write_text(
    json.dumps(summary, indent=2),
    encoding="utf-8",
)


# ============================================================
# PRINT
# ============================================================

print()
print("=" * 70)
print(" TCGA-LUAD / TCGA-LUSC — STAGE 3 PRE-DOWNLOAD AUDIT")
print("=" * 70)

print(
    f"Patients:                    {len(cohort)}"
)
print(
    f"  LUAD / LUSC:               "
    f"{(cohort['class_label'] == 'LUAD').sum()} / "
    f"{(cohort['class_label'] == 'LUSC').sum()}"
)

print(
    f"Selected files:              {len(inventory)} "
    f"(950 WSI + 950 RNA)"
)

print("-" * 70)

print(
    f"Unique patient IDs:          "
    f"{cohort['case_submitter_id'].nunique()}"
)

print(
    f"Unique WSI UUIDs:            "
    f"{cohort['wsi_file_id'].nunique()}"
)

print(
    f"Unique RNA UUIDs:            "
    f"{cohort['rna_file_id'].nunique()}"
)

print("-" * 70)

print(
    f"Live GDC UUIDs found:        "
    f"{len(found_ids)} / {len(expected_ids)}"
)

print(
    f"Metadata mismatches:         "
    f"{len(metadata_mismatches)}"
)

print("\nLive access distribution:")

print(
    inventory["live_access"]
    .value_counts(dropna=False)
    .to_string()
)

print("\nLive state distribution:")

print(
    inventory["live_state"]
    .value_counts(dropna=False)
    .to_string()
)

print("-" * 70)

print("WSI size distribution:")

print(
    f"  min:                       "
    f"{fmt_gb(wsi_sizes.min())}"
)

print(
    f"  median:                    "
    f"{fmt_gb(wsi_sizes.median())}"
)

print(
    f"  mean:                      "
    f"{fmt_gb(wsi_sizes.mean())}"
)

print(
    f"  p95:                       "
    f"{fmt_gb(wsi_sizes.quantile(0.95))}"
)

print(
    f"  max:                       "
    f"{fmt_gb(wsi_sizes.max())}"
)

print(
    f"  >1 GB:                     "
    f"{(wsi_sizes > 1024**3).sum()} files"
)

print(
    f"  >2 GB:                     "
    f"{(wsi_sizes > 2 * 1024**3).sum()} files"
)

print("-" * 70)

print(
    f"Total selected WSI:          "
    f"{fmt_gb(wsi_sizes.sum())}"
)

print(
    f"Total selected RNA:          "
    f"{fmt_gb(rna_sizes.sum())}"
)

print(
    f"Current free disk:           "
    f"{free_gb:.2f} GB"
)

print("-" * 70)

print(
    f"Cohort SHA256:"
)
print(
    f"  {cohort_sha256}"
)

print()

if problems:
    print("AUDIT STATUS: CHECK REQUIRED")

    for p in problems:
        print(f"  ! {p}")

else:
    print("AUDIT STATUS: PASS")
    print(
        "No manifest-level problems detected."
    )

print()
print("Saved:")
print(
    "  data/manifests/selected_file_inventory_stage3.csv"
)
print(
    "  data/manifests/largest_wsi_stage3.csv"
)
print(
    "  data/manifests/metadata_mismatches_stage3.csv"
)
print(
    "  data/metadata/selected_files_live_gdc_stage3.csv"
)
print(
    "  data/metadata/cohort_audit_stage3.json"
)

print("=" * 70)