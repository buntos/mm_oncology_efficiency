from pathlib import Path
import hashlib
import json

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

MANIFEST_DIR = ROOT / "data" / "manifests"
METADATA_DIR = ROOT / "data" / "metadata"

COHORT_FILE = MANIFEST_DIR / "paired_cohort_stage2.csv"

PILOT_SIZE = 10
TARGET_PER_CLASS = {
    "LUAD": 5,
    "LUSC": 5,
}


def sha256_file(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


# ============================================================
# LOAD
# ============================================================

cohort = pd.read_csv(COHORT_FILE).copy()

cohort["wsi_size_gb"] = (
    cohort["wsi_file_size_bytes"] / 1024**3
)

cohort = cohort.sort_values(
    "case_submitter_id"
).reset_index(drop=True)


selected = {}


def add_patient(row, reason):
    case_id = row["case_submitter_id"]

    if case_id in selected:
        selected[case_id]["pilot_selection_reason"] += f";{reason}"
        return False

    data = row.to_dict()
    data["pilot_selection_reason"] = reason

    selected[case_id] = data

    return True


# ============================================================
# 1. SMALLEST WSI
# ============================================================

smallest = (
    cohort
    .sort_values(
        ["wsi_file_size_bytes", "case_submitter_id"],
        ascending=[True, True],
    )
    .iloc[0]
)

add_patient(
    smallest,
    "smallest_WSI",
)


# ============================================================
# 2. LARGEST WSI
# ============================================================

largest = (
    cohort
    .sort_values(
        ["wsi_file_size_bytes", "case_submitter_id"],
        ascending=[False, True],
    )
    .iloc[0]
)

add_patient(
    largest,
    "largest_WSI",
)


# ============================================================
# 3. RNA VIAL B
#
# Prefer a vial-B case near the overall median WSI size.
# ============================================================

overall_median = cohort["wsi_file_size_bytes"].median()

vial_b = cohort[
    cohort["rna_sample_vial"]
    .astype(str)
    .str.upper()
    .eq("B")
].copy()

vial_b = vial_b[
    ~vial_b["case_submitter_id"].isin(selected)
]

if not vial_b.empty:

    vial_b["distance_to_median"] = (
        vial_b["wsi_file_size_bytes"] - overall_median
    ).abs()

    vial_b = vial_b.sort_values(
        [
            "distance_to_median",
            "case_submitter_id",
        ]
    )

    add_patient(
        vial_b.iloc[0],
        "RNA_vial_B",
    )


# ============================================================
# 4. FILL TO 5 LUAD + 5 LUSC
#
# Choose ordinary cases nearest the class median.
# ============================================================

for class_label, target_count in TARGET_PER_CLASS.items():

    while True:

        current_count = sum(
            1
            for row in selected.values()
            if row["class_label"] == class_label
        )

        if current_count >= target_count:
            break

        candidates = cohort[
            (cohort["class_label"] == class_label)
            &
            (~cohort["case_submitter_id"].isin(selected))
        ].copy()

        if candidates.empty:
            raise RuntimeError(
                f"No candidates left for class {class_label}"
            )

        class_median = candidates[
            "wsi_file_size_bytes"
        ].median()

        candidates["distance_to_class_median"] = (
            candidates["wsi_file_size_bytes"]
            - class_median
        ).abs()

        candidates = candidates.sort_values(
            [
                "distance_to_class_median",
                "case_submitter_id",
            ]
        )

        add_patient(
            candidates.iloc[0],
            f"typical_{class_label}",
        )


# ============================================================
# VALIDATE
# ============================================================

pilot = pd.DataFrame(
    selected.values()
)

pilot = pilot.sort_values(
    [
        "class_label",
        "case_submitter_id",
    ]
).reset_index(drop=True)


if len(pilot) != PILOT_SIZE:
    raise RuntimeError(
        f"Expected {PILOT_SIZE} pilot patients, got {len(pilot)}"
    )


counts = pilot["class_label"].value_counts()

for label, expected in TARGET_PER_CLASS.items():

    actual = int(counts.get(label, 0))

    if actual != expected:
        raise RuntimeError(
            f"Expected {expected} {label}, got {actual}"
        )


if pilot["case_submitter_id"].duplicated().any():
    raise RuntimeError(
        "Duplicate patients detected in pilot cohort"
    )


# ============================================================
# SAVE PILOT COHORT
# ============================================================

pilot_path = (
    MANIFEST_DIR
    / "pilot_cohort_stage4.csv"
)

pilot.to_csv(
    pilot_path,
    index=False,
)


# ============================================================
# BUILD FILE-LEVEL PILOT INVENTORY
# ============================================================

file_rows = []

for _, row in pilot.iterrows():

    file_rows.append({
        "case_submitter_id": row["case_submitter_id"],
        "class_label": row["class_label"],
        "modality": "WSI",
        "file_id": row["wsi_file_id"],
        "file_name": row["wsi_file_name"],
        "file_size_bytes": int(row["wsi_file_size_bytes"]),
        "md5sum": row["wsi_md5sum"],
        "pilot_selection_reason": row[
            "pilot_selection_reason"
        ],
    })

    file_rows.append({
        "case_submitter_id": row["case_submitter_id"],
        "class_label": row["class_label"],
        "modality": "RNA",
        "file_id": row["rna_file_id"],
        "file_name": row["rna_file_name"],
        "file_size_bytes": int(row["rna_file_size_bytes"]),
        "md5sum": row["rna_md5sum"],
        "pilot_selection_reason": row[
            "pilot_selection_reason"
        ],
    })


pilot_files = pd.DataFrame(file_rows)

pilot_files.to_csv(
    MANIFEST_DIR / "pilot_files_stage4.csv",
    index=False,
)


# ============================================================
# SAVE UUID LIST
# ============================================================

uuid_path = (
    MANIFEST_DIR
    / "pilot_file_uuids_stage4.txt"
)

with open(
    uuid_path,
    "w",
    encoding="utf-8",
) as f:

    for file_id in pilot_files["file_id"]:
        f.write(f"{file_id}\n")


# ============================================================
# HASHES
# ============================================================

source_hash = sha256_file(
    COHORT_FILE
)

pilot_hash = sha256_file(
    pilot_path
)


# ============================================================
# SUMMARY JSON
# ============================================================

summary = {
    "source_cohort_sha256": source_hash,
    "pilot_cohort_sha256": pilot_hash,

    "pilot_patients": int(len(pilot)),
    "luad": int(
        (pilot["class_label"] == "LUAD").sum()
    ),
    "lusc": int(
        (pilot["class_label"] == "LUSC").sum()
    ),

    "wsi_total_gb": float(
        pilot["wsi_file_size_bytes"].sum()
        / 1024**3
    ),

    "rna_total_gb": float(
        pilot["rna_file_size_bytes"].sum()
        / 1024**3
    ),
}


(
    METADATA_DIR
    / "pilot_selection_stage4.json"
).write_text(
    json.dumps(
        summary,
        indent=2,
    ),
    encoding="utf-8",
)


# ============================================================
# PRINT SUMMARY
# ============================================================

print()
print("=" * 76)
print(" TCGA-LUAD / TCGA-LUSC — STAGE 4 PILOT COHORT")
print("=" * 76)

print(
    f"Pilot patients:             {len(pilot)}"
)

print(
    f"LUAD / LUSC:               "
    f"{(pilot['class_label'] == 'LUAD').sum()} / "
    f"{(pilot['class_label'] == 'LUSC').sum()}"
)

print(
    f"Pilot WSI total:            "
    f"{pilot['wsi_file_size_bytes'].sum() / 1024**3:.2f} GB"
)

print(
    f"Pilot RNA total:            "
    f"{pilot['rna_file_size_bytes'].sum() / 1024**3:.3f} GB"
)

print()
print("Selected patients:")
print()

display_cols = [
    "case_submitter_id",
    "class_label",
    "wsi_size_gb",
    "wsi_slide_token",
    "rna_sample_vial",
    "pilot_selection_reason",
]

display = pilot[
    display_cols
].copy()

display["wsi_size_gb"] = (
    display["wsi_size_gb"]
    .round(3)
)

print(
    display.to_string(
        index=False
    )
)

print()
print("Source cohort SHA256:")
print(
    f"  {source_hash}"
)

print()
print("Pilot cohort SHA256:")
print(
    f"  {pilot_hash}"
)

print()
print("Saved:")
print(
    "  data/manifests/pilot_cohort_stage4.csv"
)
print(
    "  data/manifests/pilot_files_stage4.csv"
)
print(
    "  data/manifests/pilot_file_uuids_stage4.txt"
)
print(
    "  data/metadata/pilot_selection_stage4.json"
)

print("=" * 76)