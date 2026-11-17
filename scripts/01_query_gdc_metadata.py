from pathlib import Path
import json

import pandas as pd
import requests


BASE_URL = "https://api.gdc.cancer.gov"

PROJECTS = [
    "TCGA-LUAD",
    "TCGA-LUSC",
]

ROOT = Path(__file__).resolve().parents[1]

METADATA_DIR = ROOT / "data" / "metadata"
MANIFEST_DIR = ROOT / "data" / "manifests"

METADATA_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST_DIR.mkdir(parents=True, exist_ok=True)


# Fields returned by the GDC /files endpoint.
FIELDS = [
    "file_id",
    "file_name",
    "file_size",
    "md5sum",
    "data_category",
    "data_type",
    "data_format",
    "experimental_strategy",
    "access",
    "state",
    "analysis.workflow_type",
    "cases.case_id",
    "cases.submitter_id",
    "cases.project.project_id",
    "cases.samples.sample_id",
    "cases.samples.submitter_id",
    "cases.samples.sample_type",
    "cases.samples.tissue_type",
    "cases.samples.tumor_descriptor",
    "associated_entities.entity_id",
    "associated_entities.entity_submitter_id",
    "associated_entities.entity_type",
]


def eq(field, value):
    return {
        "op": "=",
        "content": {
            "field": field,
            "value": value,
        },
    }


def in_(field, values):
    return {
        "op": "in",
        "content": {
            "field": field,
            "value": values,
        },
    }


def and_(*conditions):
    return {
        "op": "and",
        "content": list(conditions),
    }


def query_files(name, filters):
    """
    Query GDC /files endpoint and save full raw JSON response.
    """

    payload = {
        "filters": filters,
        "format": "JSON",
        "fields": ",".join(FIELDS),
        "size": "10000",
    }

    print(f"\nQuerying: {name}")

    response = requests.post(
        f"{BASE_URL}/files",
        json=payload,
        timeout=180,
    )

    response.raise_for_status()

    result = response.json()

    raw_path = METADATA_DIR / f"{name}_raw.json"

    raw_path.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    hits = result["data"]["hits"]
    total = result["data"]["pagination"]["total"]

    print(f"Returned files: {len(hits)}")
    print(f"GDC reported total: {total}")

    if len(hits) != total:
        raise RuntimeError(
            f"Query was truncated: returned {len(hits)} of {total} records"
        )

    return hits


def flatten_files(hits, modality):
    """
    Convert GDC nested JSON into a simple file-level table.
    """

    rows = []

    for hit in hits:
        analysis = hit.get("analysis") or {}
        cases = hit.get("cases") or []
        associated = hit.get("associated_entities") or []

        associated_types = sorted({
            str(item.get("entity_type"))
            for item in associated
            if item.get("entity_type")
        })

        associated_submitter_ids = sorted({
            str(item.get("entity_submitter_id"))
            for item in associated
            if item.get("entity_submitter_id")
        })

        for case in cases:
            project = case.get("project") or {}

            samples = case.get("samples") or []

            sample_types = sorted({
                str(sample.get("sample_type"))
                for sample in samples
                if sample.get("sample_type")
            })

            tissue_types = sorted({
                str(sample.get("tissue_type"))
                for sample in samples
                if sample.get("tissue_type")
            })

            sample_submitter_ids = sorted({
                str(sample.get("submitter_id"))
                for sample in samples
                if sample.get("submitter_id")
            })

            file_size = hit.get("file_size")

            rows.append({
                "modality": modality,

                "project_id": project.get("project_id"),
                "case_id": case.get("case_id"),
                "case_submitter_id": case.get("submitter_id"),

                "file_id": hit.get("file_id"),
                "file_name": hit.get("file_name"),
                "file_size_bytes": file_size,
                "file_size_gb": (
                    file_size / 1024**3
                    if file_size is not None
                    else None
                ),
                "md5sum": hit.get("md5sum"),

                "data_category": hit.get("data_category"),
                "data_type": hit.get("data_type"),
                "data_format": hit.get("data_format"),
                "experimental_strategy": hit.get(
                    "experimental_strategy"
                ),

                "workflow_type": analysis.get("workflow_type"),

                "access": hit.get("access"),
                "state": hit.get("state"),

                "case_sample_types": ";".join(sample_types),
                "case_tissue_types": ";".join(tissue_types),
                "case_sample_submitter_ids": ";".join(
                    sample_submitter_ids
                ),

                "associated_entity_types": ";".join(
                    associated_types
                ),
                "associated_entity_submitter_ids": ";".join(
                    associated_submitter_ids
                ),
            })

    return pd.DataFrame(rows)


# ============================================================
# 1. SAVE GDC API STATUS
# ============================================================

print("Checking GDC API status...")

status_response = requests.get(
    f"{BASE_URL}/status",
    timeout=60,
)

status_response.raise_for_status()

status = status_response.json()

(METADATA_DIR / "gdc_api_status.json").write_text(
    json.dumps(status, indent=2),
    encoding="utf-8",
)

print("GDC API connection: OK")


# ============================================================
# 2. QUERY WSI
# ============================================================

wsi_filters = and_(
    in_(
        "cases.project.project_id",
        PROJECTS,
    ),
    eq(
        "files.data_type",
        "Slide Image",
    ),
    eq(
        "files.data_format",
        "SVS",
    ),
)

wsi_hits = query_files(
    "wsi_candidates",
    wsi_filters,
)

wsi_df = flatten_files(
    wsi_hits,
    modality="WSI",
)

wsi_df.to_csv(
    MANIFEST_DIR / "wsi_candidates.csv",
    index=False,
)


# ============================================================
# 3. QUERY RNA-SEQ / STAR - COUNTS
# ============================================================

rna_filters = and_(
    in_(
        "cases.project.project_id",
        PROJECTS,
    ),
    eq(
        "files.data_type",
        "Gene Expression Quantification",
    ),
    eq(
        "files.experimental_strategy",
        "RNA-Seq",
    ),
    eq(
        "files.analysis.workflow_type",
        "STAR - Counts",
    ),
)

rna_hits = query_files(
    "rna_star_counts_candidates",
    rna_filters,
)

rna_df = flatten_files(
    rna_hits,
    modality="RNA",
)

rna_df.to_csv(
    MANIFEST_DIR / "rna_star_counts_candidates.csv",
    index=False,
)


# ============================================================
# 4. FIND PATIENTS WITH BOTH WSI AND RNA
# ============================================================

wsi_cases = set(
    wsi_df["case_submitter_id"].dropna()
)

rna_cases = set(
    rna_df["case_submitter_id"].dropna()
)

paired_cases = sorted(
    wsi_cases.intersection(rna_cases)
)


project_lookup = (
    pd.concat([
        wsi_df[
            ["case_submitter_id", "project_id"]
        ],
        rna_df[
            ["case_submitter_id", "project_id"]
        ],
    ])
    .dropna()
    .drop_duplicates("case_submitter_id")
    .set_index("case_submitter_id")["project_id"]
    .to_dict()
)


paired_df = pd.DataFrame({
    "case_submitter_id": paired_cases
})

paired_df["project_id"] = paired_df[
    "case_submitter_id"
].map(project_lookup)


paired_df.to_csv(
    MANIFEST_DIR / "paired_cases_stage1.csv",
    index=False,
)


# ============================================================
# 5. SUMMARY
# ============================================================

paired_wsi = wsi_df[
    wsi_df["case_submitter_id"].isin(paired_cases)
]

paired_rna = rna_df[
    rna_df["case_submitter_id"].isin(paired_cases)
]


print("\n")
print("=" * 64)
print(" GDC TCGA-LUAD / TCGA-LUSC — STAGE 1 SUMMARY")
print("=" * 64)

print(
    f"WSI files:                  {len(wsi_df)}"
)
print(
    f"Unique patients with WSI:   "
    f"{wsi_df['case_submitter_id'].nunique()}"
)
print(
    f"All WSI size:               "
    f"{wsi_df['file_size_bytes'].sum() / 1024**3:.2f} GB"
)

print("-" * 64)

print(
    f"STAR-Counts RNA files:      {len(rna_df)}"
)
print(
    f"Unique patients with RNA:   "
    f"{rna_df['case_submitter_id'].nunique()}"
)
print(
    f"All RNA size:               "
    f"{rna_df['file_size_bytes'].sum() / 1024**3:.2f} GB"
)

print("-" * 64)

print(
    f"PAIRED PATIENTS:            {len(paired_df)}"
)

print("\nPaired patients by project:")

project_counts = (
    paired_df["project_id"]
    .value_counts()
    .sort_index()
)

for project, count in project_counts.items():
    print(
        f"  {project:<12} {count}"
    )


print("\nWSI count per paired patient:")

wsi_counts = (
    paired_wsi
    .groupby("case_submitter_id")
    .size()
    .value_counts()
    .sort_index()
)

for file_count, patient_count in wsi_counts.items():
    print(
        f"  {file_count} WSI -> "
        f"{patient_count} patients"
    )


print("\nRNA files per paired patient:")

rna_counts = (
    paired_rna
    .groupby("case_submitter_id")
    .size()
    .value_counts()
    .sort_index()
)

for file_count, patient_count in rna_counts.items():
    print(
        f"  {file_count} RNA -> "
        f"{patient_count} patients"
    )


print("\nPotential WSI size for paired patients:")

print(
    f"  {paired_wsi['file_size_bytes'].sum() / 1024**3:.2f} GB"
)


print("\nSaved files:")

print(
    "  data/manifests/wsi_candidates.csv"
)
print(
    "  data/manifests/rna_star_counts_candidates.csv"
)
print(
    "  data/manifests/paired_cases_stage1.csv"
)
print(
    "  data/metadata/gdc_api_status.json"
)

print("\nIMPORTANT:")
print(
    "paired_cases_stage1.csv is NOT the final cohort."
)
print(
    "Tumor-sample filtering and deterministic file selection "
    "will be performed in Stage 2."
)

print("=" * 64)