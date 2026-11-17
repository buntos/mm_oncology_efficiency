from pathlib import Path
import hashlib
import json

import pandas as pd
import openslide


ROOT = Path(__file__).resolve().parents[1]

DOWNLOAD_DIR = ROOT / "data" / "pilot_download"
MANIFEST_DIR = ROOT / "data" / "manifests"
METADATA_DIR = ROOT / "data" / "metadata"
THUMB_DIR = ROOT / "results" / "figures" / "pilot_thumbnails"

PILOT_FILES = MANIFEST_DIR / "pilot_files_stage4.csv"

METADATA_DIR.mkdir(parents=True, exist_ok=True)
THUMB_DIR.mkdir(parents=True, exist_ok=True)


def md5_file(path, chunk_size=8 * 1024 * 1024):
    h = hashlib.md5()

    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


def find_downloaded_file(filename):
    matches = list(
        DOWNLOAD_DIR.rglob(filename)
    )

    if len(matches) == 0:
        return None, "missing"

    if len(matches) > 1:
        return None, f"duplicate_paths:{len(matches)}"

    return matches[0], "ok"


# ============================================================
# LOAD EXPECTED FILES
# ============================================================

pilot = pd.read_csv(PILOT_FILES)

integrity_rows = []
wsi_rows = []
rna_rows = []

print()
print("=" * 72)
print(" TCGA-LUAD / TCGA-LUSC — PILOT DOWNLOAD QC")
print("=" * 72)

print(f"Expected files: {len(pilot)}")
print()


# ============================================================
# FILE INTEGRITY
# ============================================================

for i, row in pilot.iterrows():

    filename = row["file_name"]
    modality = row["modality"]
    expected_size = int(row["file_size_bytes"])
    expected_md5 = str(row["md5sum"]).lower()

    print(
        f"[{i + 1:02d}/{len(pilot)}] "
        f"{modality:<3} {filename[:55]}"
    )

    path, find_status = find_downloaded_file(filename)

    if path is None:

        integrity_rows.append({
            "case_submitter_id": row["case_submitter_id"],
            "class_label": row["class_label"],
            "modality": modality,
            "file_id": row["file_id"],
            "file_name": filename,
            "path": None,
            "exists": False,
            "size_match": False,
            "md5_match": False,
            "status": find_status,
        })

        continue

    actual_size = path.stat().st_size

    size_match = (
        actual_size == expected_size
    )

    actual_md5 = md5_file(path)

    md5_match = (
        actual_md5.lower() == expected_md5
    )

    integrity_rows.append({
        "case_submitter_id": row["case_submitter_id"],
        "class_label": row["class_label"],
        "modality": modality,
        "file_id": row["file_id"],
        "file_name": filename,
        "path": str(path),
        "exists": True,
        "expected_size": expected_size,
        "actual_size": actual_size,
        "size_match": size_match,
        "expected_md5": expected_md5,
        "actual_md5": actual_md5,
        "md5_match": md5_match,
        "status": (
            "PASS"
            if size_match and md5_match
            else "FAIL"
        ),
    })


integrity = pd.DataFrame(integrity_rows)

integrity.to_csv(
    METADATA_DIR / "pilot_file_integrity_stage4.csv",
    index=False,
)


# ============================================================
# WSI QC
# ============================================================

print()
print("Checking WSI with OpenSlide...")

wsi_integrity = integrity[
    (integrity["modality"] == "WSI")
    &
    (integrity["status"] == "PASS")
]

for _, row in wsi_integrity.iterrows():

    path = Path(row["path"])

    record = {
        "case_submitter_id": row["case_submitter_id"],
        "class_label": row["class_label"],
        "file_name": row["file_name"],
        "path": str(path),
    }

    try:

        slide = openslide.OpenSlide(str(path))

        width, height = slide.dimensions

        level_count = slide.level_count

        objective = slide.properties.get(
            openslide.PROPERTY_NAME_OBJECTIVE_POWER
        )

        mpp_x = slide.properties.get(
            openslide.PROPERTY_NAME_MPP_X
        )

        mpp_y = slide.properties.get(
            openslide.PROPERTY_NAME_MPP_Y
        )

        vendor = slide.properties.get(
            openslide.PROPERTY_NAME_VENDOR
        )

        # Force actual image decoding rather than only parsing headers.
        thumbnail = slide.get_thumbnail(
            (512, 512)
        ).convert("RGB")

        thumb_path = (
            THUMB_DIR
            / f"{row['case_submitter_id']}.jpg"
        )

        thumbnail.save(
            thumb_path,
            quality=90,
        )

        record.update({
            "openslide_ok": True,
            "width": width,
            "height": height,
            "level_count": level_count,
            "level_dimensions": str(
                list(slide.level_dimensions)
            ),
            "objective_power": objective,
            "mpp_x": mpp_x,
            "mpp_y": mpp_y,
            "vendor": vendor,
            "thumbnail_path": str(thumb_path),
            "error": None,
        })

        slide.close()

    except Exception as exc:

        record.update({
            "openslide_ok": False,
            "width": None,
            "height": None,
            "level_count": None,
            "level_dimensions": None,
            "objective_power": None,
            "mpp_x": None,
            "mpp_y": None,
            "vendor": None,
            "thumbnail_path": None,
            "error": repr(exc),
        })

    wsi_rows.append(record)


wsi_qc = pd.DataFrame(wsi_rows)

wsi_qc.to_csv(
    METADATA_DIR / "pilot_wsi_qc_stage4.csv",
    index=False,
)


# ============================================================
# RNA QC
# ============================================================

print("Checking RNA STAR-Counts files...")

rna_integrity = integrity[
    (integrity["modality"] == "RNA")
    &
    (integrity["status"] == "PASS")
]

for _, row in rna_integrity.iterrows():

    path = Path(row["path"])

    record = {
        "case_submitter_id": row["case_submitter_id"],
        "class_label": row["class_label"],
        "file_name": row["file_name"],
        "path": str(path),
    }

    try:

        df = pd.read_csv(
            path,
            sep="\t",
            comment="#",
            low_memory=False,
        )

        columns = list(df.columns)

        gene_id_present = (
            "gene_id" in columns
        )

        record.update({
            "rna_read_ok": True,
            "rows": len(df),
            "columns_count": len(columns),
            "columns": ";".join(columns),
            "gene_id_present": gene_id_present,
            "error": None,
        })

    except Exception as exc:

        record.update({
            "rna_read_ok": False,
            "rows": None,
            "columns_count": None,
            "columns": None,
            "gene_id_present": False,
            "error": repr(exc),
        })

    rna_rows.append(record)


rna_qc = pd.DataFrame(rna_rows)

rna_qc.to_csv(
    METADATA_DIR / "pilot_rna_qc_stage4.csv",
    index=False,
)


# ============================================================
# SUMMARY
# ============================================================

files_pass = int(
    (integrity["status"] == "PASS").sum()
)

wsi_pass = int(
    wsi_qc["openslide_ok"].sum()
) if not wsi_qc.empty else 0

rna_pass = int(
    (
        rna_qc["rna_read_ok"]
        &
        rna_qc["gene_id_present"]
    ).sum()
) if not rna_qc.empty else 0


overall_pass = (
    files_pass == 20
    and wsi_pass == 10
    and rna_pass == 10
)


summary = {
    "expected_files": 20,
    "files_integrity_pass": files_pass,

    "expected_wsi": 10,
    "wsi_openslide_pass": wsi_pass,

    "expected_rna": 10,
    "rna_read_pass": rna_pass,

    "overall_pass": overall_pass,
}


(
    METADATA_DIR
    / "pilot_download_qc_stage4.json"
).write_text(
    json.dumps(
        summary,
        indent=2,
    ),
    encoding="utf-8",
)


# ============================================================
# PRINT RESULTS
# ============================================================

print()
print("-" * 72)

print(
    f"File integrity:             "
    f"{files_pass}/20 PASS"
)

print(
    f"WSI OpenSlide:              "
    f"{wsi_pass}/10 PASS"
)

print(
    f"RNA STAR-Counts:            "
    f"{rna_pass}/10 PASS"
)

print("-" * 72)

if not wsi_qc.empty:

    print()
    print("WSI dimensions:")

    display = wsi_qc[
        [
            "case_submitter_id",
            "class_label",
            "width",
            "height",
            "level_count",
            "objective_power",
            "mpp_x",
        ]
    ]

    print(
        display.to_string(
            index=False
        )
    )


if not rna_qc.empty:

    print()
    print("RNA structure:")

    display = rna_qc[
        [
            "case_submitter_id",
            "rows",
            "columns_count",
            "gene_id_present",
        ]
    ]

    print(
        display.to_string(
            index=False
        )
    )


print()
print("-" * 72)

if overall_pass:

    print("PILOT DOWNLOAD QC STATUS: PASS")

else:

    print("PILOT DOWNLOAD QC STATUS: CHECK REQUIRED")


print()
print("Saved:")

print(
    "  data/metadata/pilot_file_integrity_stage4.csv"
)

print(
    "  data/metadata/pilot_wsi_qc_stage4.csv"
)

print(
    "  data/metadata/pilot_rna_qc_stage4.csv"
)

print(
    "  data/metadata/pilot_download_qc_stage4.json"
)

print(
    "  results/figures/pilot_thumbnails/*.jpg"
)

print("=" * 72)