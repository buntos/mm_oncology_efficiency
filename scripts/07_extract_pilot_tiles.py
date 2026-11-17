from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import openslide

from skimage.filters import threshold_otsu
from skimage.morphology import (
    remove_small_holes,
    remove_small_objects,
)


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

MANIFEST_DIR = ROOT / "data" / "manifests"
METADATA_DIR = ROOT / "data" / "metadata"

DOWNLOAD_DIR = ROOT / "data" / "pilot_download"
TILE_ROOT = ROOT / "data" / "tiles" / "pilot"
FIGURE_DIR = ROOT / "results" / "figures" / "pilot_tile_contact_sheets"

PILOT_COHORT = MANIFEST_DIR / "pilot_cohort_stage4.csv"
INTEGRITY_FILE = METADATA_DIR / "pilot_file_integrity_stage4.csv"

TILE_SIZE = 256
STRIDE = 256

TARGET_TILES = 64

# Low-resolution mask is only used to find promising candidates.
CANDIDATE_TISSUE_FRACTION = 0.80

# Final decision is made on the actual level-0 tile.
FINAL_TISSUE_FRACTION = 0.90

THUMBNAIL_MAX_SIZE = 4096

TILE_ROOT.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# HELPERS
# ============================================================

def deterministic_seed(text):
    digest = hashlib.sha256(
        text.encode("utf-8")
    ).digest()

    return int.from_bytes(
        digest[:8],
        byteorder="little",
        signed=False,
    )


def integral_image(mask):
    return (
        mask.astype(np.uint32)
        .cumsum(axis=0)
        .cumsum(axis=1)
    )


def rect_sum(ii, x1, y1, x2, y2):

    x1 = max(0, x1)
    y1 = max(0, y1)

    x2 = min(ii.shape[1], x2)
    y2 = min(ii.shape[0], y2)

    if x2 <= x1 or y2 <= y1:
        return 0

    result = ii[y2 - 1, x2 - 1]

    if x1 > 0:
        result -= ii[y2 - 1, x1 - 1]

    if y1 > 0:
        result -= ii[y1 - 1, x2 - 1]

    if x1 > 0 and y1 > 0:
        result += ii[y1 - 1, x1 - 1]

    return int(result)


def create_contact_sheet(tile_paths, out_path):

    images = [
        Image.open(path).convert("RGB")
        for path in tile_paths
    ]

    if len(images) != 64:
        return

    sheet = Image.new(
        "RGB",
        (8 * TILE_SIZE, 8 * TILE_SIZE),
        "white",
    )

    for i, img in enumerate(images):

        x = (i % 8) * TILE_SIZE
        y = (i // 8) * TILE_SIZE

        sheet.paste(
            img,
            (x, y),
        )

    sheet.save(
        out_path,
        quality=95,
    )

    for img in images:
        img.close()


# ============================================================
# LOAD DATA
# ============================================================

pilot = pd.read_csv(
    PILOT_COHORT
)

integrity = pd.read_csv(
    INTEGRITY_FILE
)

wsi_integrity = integrity[
    (integrity["modality"] == "WSI")
    &
    (integrity["status"] == "PASS")
].copy()


path_lookup = (
    wsi_integrity
    .set_index("case_submitter_id")["path"]
    .to_dict()
)


tile_rows = []
qc_rows = []


# ============================================================
# PROCESS PILOT
# ============================================================

print()
print("=" * 76)
print(" TCGA-LUAD / TCGA-LUSC — PILOT TILE EXTRACTION")
print("=" * 76)

print(
    f"Tile size:                   {TILE_SIZE} x {TILE_SIZE}"
)

print(
    f"Stride:                      {STRIDE}"
)

print(
    f"Target tiles per patient:    {TARGET_TILES}"
)

print(
    f"Final tissue threshold:      "
    f"{FINAL_TISSUE_FRACTION:.0%}"
)

print()


for patient_index, row in pilot.iterrows():

    case_id = row["case_submitter_id"]
    class_label = row["class_label"]

    if case_id not in path_lookup:

        qc_rows.append({
            "case_submitter_id": case_id,
            "class_label": class_label,
            "status": "FAIL",
            "reason": "WSI path missing",
        })

        continue

    slide_path = Path(
        path_lookup[case_id]
    )

    print(
        f"[{patient_index + 1:02d}/{len(pilot)}] "
        f"{case_id} ({class_label})"
    )

    slide = openslide.OpenSlide(
        str(slide_path)
    )

    width, height = slide.dimensions


    # ========================================================
    # LOW-RES THUMBNAIL
    # ========================================================

    thumbnail = slide.get_thumbnail(
        (
            THUMBNAIL_MAX_SIZE,
            THUMBNAIL_MAX_SIZE,
        )
    ).convert("RGB")

    thumb_np = np.asarray(
        thumbnail
    )

    gray = (
        0.299 * thumb_np[:, :, 0]
        + 0.587 * thumb_np[:, :, 1]
        + 0.114 * thumb_np[:, :, 2]
    ).astype(np.uint8)


    otsu_threshold = float(
        threshold_otsu(gray)
    )

    tissue_mask = (
        gray < otsu_threshold
    )

    # Remove tiny isolated scanner/background artifacts.
    tissue_mask = remove_small_objects(
        tissue_mask,
        min_size=32,
    )

    tissue_mask = remove_small_holes(
        tissue_mask,
        area_threshold=32,
    )


    mask_fraction = float(
        tissue_mask.mean()
    )


    # ========================================================
    # MAP LEVEL-0 COORDINATES TO THUMBNAIL
    # ========================================================

    thumb_h, thumb_w = tissue_mask.shape

    scale_x = width / thumb_w
    scale_y = height / thumb_h

    ii = integral_image(
        tissue_mask
    )


    candidates = []


    for y in range(
        0,
        height - TILE_SIZE + 1,
        STRIDE,
    ):

        ty1 = int(
            np.floor(y / scale_y)
        )

        ty2 = int(
            np.ceil(
                (y + TILE_SIZE) / scale_y
            )
        )

        for x in range(
            0,
            width - TILE_SIZE + 1,
            STRIDE,
        ):

            tx1 = int(
                np.floor(x / scale_x)
            )

            tx2 = int(
                np.ceil(
                    (x + TILE_SIZE) / scale_x
                )
            )

            area = (
                max(1, tx2 - tx1)
                *
                max(1, ty2 - ty1)
            )

            tissue_pixels = rect_sum(
                ii,
                tx1,
                ty1,
                tx2,
                ty2,
            )

            approx_fraction = (
                tissue_pixels / area
            )

            if (
                approx_fraction
                >= CANDIDATE_TISSUE_FRACTION
            ):

                candidates.append(
                    (
                        x,
                        y,
                        approx_fraction,
                    )
                )


    print(
        f"    slide:                  "
        f"{width} x {height}"
    )

    print(
        f"    thumbnail:              "
        f"{thumb_w} x {thumb_h}"
    )

    print(
        f"    Otsu threshold:         "
        f"{otsu_threshold:.1f}"
    )

    print(
        f"    tissue mask fraction:   "
        f"{mask_fraction:.1%}"
    )

    print(
        f"    candidate grid tiles:   "
        f"{len(candidates)}"
    )


    # ========================================================
    # DETERMINISTIC RANDOM ORDER
    #
    # Same patient always produces the same candidate order.
    # ========================================================

    seed = deterministic_seed(
        case_id
    )

    rng = np.random.default_rng(
        seed
    )

    order = np.arange(
        len(candidates)
    )

    rng.shuffle(
        order
    )


    patient_dir = (
        TILE_ROOT
        / case_id
    )

    patient_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    # Remove old generated tiles for reproducible reruns.
    for old_tile in patient_dir.glob(
        "*.png"
    ):
        old_tile.unlink()


    selected_paths = []

    checked = 0
    accepted = 0


    for candidate_index in order:

        x, y, approx_fraction = (
            candidates[candidate_index]
        )

        checked += 1

        tile = slide.read_region(
            (int(x), int(y)),
            0,
            (
                TILE_SIZE,
                TILE_SIZE,
            ),
        ).convert("RGB")


        tile_np = np.asarray(
            tile
        )

        tile_gray = (
            0.299 * tile_np[:, :, 0]
            + 0.587 * tile_np[:, :, 1]
            + 0.114 * tile_np[:, :, 2]
        )


        # Use the slide-level Otsu threshold for the final
        # full-resolution tile decision.
        final_fraction = float(
            (
                tile_gray
                < otsu_threshold
            ).mean()
        )


        if (
            final_fraction
            < FINAL_TISSUE_FRACTION
        ):
            continue


        tile_number = accepted + 1

        tile_name = (
            f"tile_{tile_number:03d}"
            f"_x{x}_y{y}.png"
        )

        tile_path = (
            patient_dir
            / tile_name
        )


        tile.save(
            tile_path,
            format="PNG",
        )


        selected_paths.append(
            tile_path
        )


        tile_rows.append({
            "case_submitter_id": case_id,
            "class_label": class_label,

            "tile_index": tile_number,

            "x_level0": int(x),
            "y_level0": int(y),

            "tile_size": TILE_SIZE,

            "approx_tissue_fraction": float(
                approx_fraction
            ),

            "final_tissue_fraction": float(
                final_fraction
            ),

            "otsu_threshold": otsu_threshold,

            "deterministic_seed": seed,

            "tile_path": str(
                tile_path
            ),
        })


        accepted += 1


        if accepted >= TARGET_TILES:
            break


    slide.close()


    # ========================================================
    # CONTACT SHEET
    # ========================================================

    if accepted == TARGET_TILES:

        contact_path = (
            FIGURE_DIR
            / f"{case_id}_64_tiles.jpg"
        )

        create_contact_sheet(
            selected_paths,
            contact_path,
        )


        status = "PASS"
        reason = None

    else:

        status = "FAIL"
        reason = (
            f"only {accepted} valid tiles "
            f"found"
        )


    qc_rows.append({
        "case_submitter_id": case_id,
        "class_label": class_label,

        "width": width,
        "height": height,

        "thumbnail_width": thumb_w,
        "thumbnail_height": thumb_h,

        "otsu_threshold": otsu_threshold,
        "tissue_mask_fraction": mask_fraction,

        "candidate_tiles": len(candidates),
        "candidate_tiles_checked": checked,

        "valid_tiles_selected": accepted,

        "status": status,
        "reason": reason,
    })


    print(
        f"    candidates checked:     "
        f"{checked}"
    )

    print(
        f"    selected:               "
        f"{accepted}/{TARGET_TILES}"
    )

    print(
        f"    STATUS:                 "
        f"{status}"
    )

    print()


# ============================================================
# SAVE RESULTS
# ============================================================

tiles_df = pd.DataFrame(
    tile_rows
)

qc_df = pd.DataFrame(
    qc_rows
)


tiles_df.to_csv(
    MANIFEST_DIR
    / "pilot_tile_manifest_stage5.csv",
    index=False,
)


qc_df.to_csv(
    METADATA_DIR
    / "pilot_tiling_qc_stage5.csv",
    index=False,
)


patients_pass = int(
    (qc_df["status"] == "PASS").sum()
)

total_tiles = int(
    len(tiles_df)
)


overall_pass = (
    patients_pass == len(pilot)
    and total_tiles
    == len(pilot) * TARGET_TILES
)


summary = {
    "patients": int(
        len(pilot)
    ),

    "patients_pass": patients_pass,

    "tile_size": TILE_SIZE,
    "stride": STRIDE,

    "target_tiles_per_patient": TARGET_TILES,

    "candidate_tissue_fraction": (
        CANDIDATE_TISSUE_FRACTION
    ),

    "final_tissue_fraction": (
        FINAL_TISSUE_FRACTION
    ),

    "total_tiles": total_tiles,

    "overall_pass": overall_pass,
}


(
    METADATA_DIR
    / "pilot_tiling_summary_stage5.json"
).write_text(
    json.dumps(
        summary,
        indent=2,
    ),
    encoding="utf-8",
)


# ============================================================
# FINAL PRINT
# ============================================================

print()
print("-" * 76)

print(
    f"Patients passed:             "
    f"{patients_pass}/{len(pilot)}"
)

print(
    f"Total extracted tiles:       "
    f"{total_tiles}"
)

print(
    f"Expected tiles:              "
    f"{len(pilot) * TARGET_TILES}"
)

print("-" * 76)


display = qc_df[
    [
        "case_submitter_id",
        "class_label",
        "candidate_tiles",
        "candidate_tiles_checked",
        "valid_tiles_selected",
        "status",
    ]
]

print()
print(
    display.to_string(
        index=False
    )
)

print()


if overall_pass:

    print(
        "PILOT TILING STATUS: PASS"
    )

else:

    print(
        "PILOT TILING STATUS: CHECK REQUIRED"
    )


print()
print("Saved:")

print(
    "  data/manifests/"
    "pilot_tile_manifest_stage5.csv"
)

print(
    "  data/metadata/"
    "pilot_tiling_qc_stage5.csv"
)

print(
    "  data/metadata/"
    "pilot_tiling_summary_stage5.json"
)

print(
    "  results/figures/"
    "pilot_tile_contact_sheets/"
)

print("=" * 76)