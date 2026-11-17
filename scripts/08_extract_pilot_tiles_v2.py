from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
from PIL import Image
import openslide

from skimage.color import rgb2hsv
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

TILE_ROOT = ROOT / "data" / "tiles" / "pilot_v2"

CONTACT_DIR = (
    ROOT
    / "results"
    / "figures"
    / "pilot_tile_contact_sheets_v2"
)

MASK_DIR = (
    ROOT
    / "results"
    / "figures"
    / "pilot_tissue_masks_v2"
)

PILOT_COHORT = (
    MANIFEST_DIR
    / "pilot_cohort_stage4.csv"
)

INTEGRITY_FILE = (
    METADATA_DIR
    / "pilot_file_integrity_stage4.csv"
)


# ------------------------------------------------------------
# Tile protocol
# ------------------------------------------------------------

TILE_SIZE = 256
STRIDE = 256

TARGET_TILES = 64

# Thumbnail mask only determines which level-0 tiles are worth
# opening. Final acceptance is always performed on level 0.
CANDIDATE_TISSUE_FRACTION = 0.70

# Final tile must contain at least 90% tissue pixels.
FINAL_TISSUE_FRACTION = 0.90

THUMBNAIL_MAX_SIZE = 4096


# ------------------------------------------------------------
# Colour-based tissue detection
# ------------------------------------------------------------

# Otsu is calculated independently for every slide.
# This floor prevents nearly grey scanner background from being
# treated as tissue when Otsu produces an extremely low value.
SATURATION_FLOOR = 0.03

# Very bright pixels are considered background.
MAX_TISSUE_VALUE = 0.95

# Explicit grey/background detector.
BACKGROUND_SATURATION_MAX = 0.04
BACKGROUND_VALUE_MIN = 0.55

# A valid tile may contain no more than 10% such background.
MAX_BACKGROUND_FRACTION = 0.10

# Additional near-white protection.
WHITE_VALUE_MIN = 0.97
MAX_WHITE_FRACTION = 0.10


# ------------------------------------------------------------
# Morphological cleanup of the low-resolution mask
# ------------------------------------------------------------

MIN_OBJECT_SIZE = 32
MAX_HOLE_SIZE = 32


TILE_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

CONTACT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

MASK_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# HELPERS
# ============================================================

def deterministic_seed(text):
    """
    Stable seed derived from patient ID.
    The same patient always produces the same shuffled order.
    """

    digest = hashlib.sha256(
        text.encode("utf-8")
    ).digest()

    return int.from_bytes(
        digest[:8],
        byteorder="little",
        signed=False,
    )


def integral_image(mask):
    """
    Integral image for fast rectangular sums.
    """

    return (
        mask.astype(np.uint32)
        .cumsum(axis=0)
        .cumsum(axis=1)
    )


def rect_sum(ii, x1, y1, x2, y2):
    """
    Sum of an integral-image rectangle:
    [x1, x2), [y1, y2)
    """

    x1 = max(0, x1)
    y1 = max(0, y1)

    x2 = min(ii.shape[1], x2)
    y2 = min(ii.shape[0], y2)

    if x2 <= x1 or y2 <= y1:
        return 0

    result = int(
        ii[y2 - 1, x2 - 1]
    )

    if x1 > 0:
        result -= int(
            ii[y2 - 1, x1 - 1]
        )

    if y1 > 0:
        result -= int(
            ii[y1 - 1, x2 - 1]
        )

    if x1 > 0 and y1 > 0:
        result += int(
            ii[y1 - 1, x1 - 1]
        )

    return result


def rgb_to_hsv_arrays(rgb_uint8):
    """
    uint8 RGB -> saturation and value arrays in [0, 1].
    """

    rgb_float = (
        rgb_uint8.astype(np.float32)
        / 255.0
    )

    hsv = rgb2hsv(
        rgb_float
    )

    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    return saturation, value


def calculate_slide_mask(thumbnail):
    """
    Build slide-specific tissue mask from HSV saturation.

    Returns:
        mask
        raw Otsu threshold
        threshold actually used
        saturation array
        value array
    """

    thumb_np = np.asarray(
        thumbnail
    )

    saturation, value = (
        rgb_to_hsv_arrays(
            thumb_np
        )
    )


    # Otsu on saturation separates colour-rich H&E tissue
    # from low-saturation white/grey background.
    raw_otsu = float(
        threshold_otsu(
            saturation
        )
    )


    saturation_threshold = max(
        raw_otsu,
        SATURATION_FLOOR,
    )


    tissue_mask = (
        (saturation >= saturation_threshold)
        &
        (value <= MAX_TISSUE_VALUE)
    )


    tissue_mask = remove_small_objects(
        tissue_mask,
        min_size=MIN_OBJECT_SIZE,
    )


    tissue_mask = remove_small_holes(
        tissue_mask,
        area_threshold=MAX_HOLE_SIZE,
    )


    return (
        tissue_mask,
        raw_otsu,
        saturation_threshold,
        saturation,
        value,
    )


def evaluate_level0_tile(
    tile,
    saturation_threshold,
):
    """
    Final full-resolution tile evaluation.

    Returns:
        tissue_fraction
        background_fraction
        white_fraction
    """

    tile_np = np.asarray(
        tile
    )

    saturation, value = (
        rgb_to_hsv_arrays(
            tile_np
        )
    )


    tissue_pixels = (
        (saturation >= saturation_threshold)
        &
        (value <= MAX_TISSUE_VALUE)
    )


    # Specifically detect white/grey scanner background.
    background_pixels = (
        (saturation <= BACKGROUND_SATURATION_MAX)
        &
        (value >= BACKGROUND_VALUE_MIN)
    )


    white_pixels = (
        value >= WHITE_VALUE_MIN
    )


    tissue_fraction = float(
        tissue_pixels.mean()
    )

    background_fraction = float(
        background_pixels.mean()
    )

    white_fraction = float(
        white_pixels.mean()
    )


    return (
        tissue_fraction,
        background_fraction,
        white_fraction,
    )


def create_contact_sheet(
    tile_paths,
    out_path,
):
    """
    Create an 8 x 8 sheet containing all 64 selected tiles.
    """

    if len(tile_paths) != 64:
        return


    images = [
        Image.open(path).convert("RGB")
        for path in tile_paths
    ]


    sheet = Image.new(
        "RGB",
        (
            8 * TILE_SIZE,
            8 * TILE_SIZE,
        ),
        "white",
    )


    for i, image in enumerate(images):

        x = (
            i % 8
        ) * TILE_SIZE

        y = (
            i // 8
        ) * TILE_SIZE

        sheet.paste(
            image,
            (x, y),
        )


    sheet.save(
        out_path,
        quality=95,
    )


    for image in images:
        image.close()


def save_mask_preview(
    thumbnail,
    tissue_mask,
    out_path,
):
    """
    Save a compact visual representation:
    tissue stays visible, detected background is dimmed.
    """

    rgb = np.asarray(
        thumbnail
    ).copy()


    overlay = rgb.astype(
        np.float32
    )


    background = (
        ~tissue_mask
    )


    # Darken background while leaving detected tissue untouched.
    overlay[background] *= 0.25


    overlay = np.clip(
        overlay,
        0,
        255,
    ).astype(
        np.uint8
    )


    Image.fromarray(
        overlay
    ).save(
        out_path,
        quality=92,
    )


# ============================================================
# LOAD INPUT DATA
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
    .set_index(
        "case_submitter_id"
    )["path"]
    .to_dict()
)


tile_rows = []
qc_rows = []


# ============================================================
# PROCESS PILOT
# ============================================================

print()
print("=" * 78)
print(
    " TCGA-LUAD / TCGA-LUSC — PILOT TILE EXTRACTION V2"
)
print("=" * 78)

print(
    f"Tile size:                      "
    f"{TILE_SIZE} x {TILE_SIZE}"
)

print(
    f"Stride:                         "
    f"{STRIDE}"
)

print(
    f"Target tiles per patient:       "
    f"{TARGET_TILES}"
)

print(
    f"Candidate tissue threshold:     "
    f"{CANDIDATE_TISSUE_FRACTION:.0%}"
)

print(
    f"Final tissue threshold:         "
    f"{FINAL_TISSUE_FRACTION:.0%}"
)

print(
    f"Max grey background fraction:   "
    f"{MAX_BACKGROUND_FRACTION:.0%}"
)

print(
    f"Max near-white fraction:        "
    f"{MAX_WHITE_FRACTION:.0%}"
)

print()


for patient_index, row in pilot.iterrows():

    case_id = row[
        "case_submitter_id"
    ]

    class_label = row[
        "class_label"
    ]


    print(
        f"[{patient_index + 1:02d}/{len(pilot)}] "
        f"{case_id} ({class_label})"
    )


    if case_id not in path_lookup:

        qc_rows.append({
            "case_submitter_id": case_id,
            "class_label": class_label,
            "status": "FAIL",
            "reason": "WSI path missing",
        })

        print(
            "    STATUS: FAIL — WSI path missing"
        )

        print()

        continue


    slide_path = Path(
        path_lookup[case_id]
    )


    slide = openslide.OpenSlide(
        str(slide_path)
    )


    width, height = (
        slide.dimensions
    )


    objective_power = (
        slide.properties.get(
            openslide.PROPERTY_NAME_OBJECTIVE_POWER
        )
    )


    mpp_x = (
        slide.properties.get(
            openslide.PROPERTY_NAME_MPP_X
        )
    )


    mpp_y = (
        slide.properties.get(
            openslide.PROPERTY_NAME_MPP_Y
        )
    )


    # ========================================================
    # THUMBNAIL + SATURATION-BASED TISSUE MASK
    # ========================================================

    thumbnail = slide.get_thumbnail(
        (
            THUMBNAIL_MAX_SIZE,
            THUMBNAIL_MAX_SIZE,
        )
    ).convert(
        "RGB"
    )


    (
        tissue_mask,
        raw_otsu,
        saturation_threshold,
        _,
        _,
    ) = calculate_slide_mask(
        thumbnail
    )


    thumb_h, thumb_w = (
        tissue_mask.shape
    )


    mask_fraction = float(
        tissue_mask.mean()
    )


    mask_preview_path = (
        MASK_DIR
        / f"{case_id}_tissue_mask.jpg"
    )


    save_mask_preview(
        thumbnail,
        tissue_mask,
        mask_preview_path,
    )


    # ========================================================
    # MAP LEVEL-0 COORDINATES TO THUMBNAIL
    # ========================================================

    scale_x = (
        width / thumb_w
    )

    scale_y = (
        height / thumb_h
    )


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
            np.floor(
                y / scale_y
            )
        )

        ty2 = int(
            np.ceil(
                (y + TILE_SIZE)
                / scale_y
            )
        )


        for x in range(
            0,
            width - TILE_SIZE + 1,
            STRIDE,
        ):

            tx1 = int(
                np.floor(
                    x / scale_x
                )
            )

            tx2 = int(
                np.ceil(
                    (x + TILE_SIZE)
                    / scale_x
                )
            )


            area = (
                max(
                    1,
                    tx2 - tx1,
                )
                *
                max(
                    1,
                    ty2 - ty1,
                )
            )


            tissue_pixels = rect_sum(
                ii,
                tx1,
                ty1,
                tx2,
                ty2,
            )


            approx_fraction = (
                tissue_pixels
                / area
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
        f"    slide:                     "
        f"{width} x {height}"
    )

    print(
        f"    objective / MPP:           "
        f"{objective_power} / {mpp_x}"
    )

    print(
        f"    thumbnail:                 "
        f"{thumb_w} x {thumb_h}"
    )

    print(
        f"    saturation Otsu raw:       "
        f"{raw_otsu:.4f}"
    )

    print(
        f"    saturation threshold used: "
        f"{saturation_threshold:.4f}"
    )

    print(
        f"    tissue mask fraction:      "
        f"{mask_fraction:.1%}"
    )

    print(
        f"    candidate grid tiles:      "
        f"{len(candidates)}"
    )


    # ========================================================
    # DETERMINISTIC CANDIDATE ORDER
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


    # ========================================================
    # OUTPUT DIRECTORY
    # ========================================================

    patient_dir = (
        TILE_ROOT
        / case_id
    )


    patient_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    # Re-running V2 always produces a clean deterministic set.
    for old_tile in patient_dir.glob(
        "*.png"
    ):
        old_tile.unlink()


    selected_paths = []


    checked = 0
    accepted = 0

    rejected_tissue = 0
    rejected_background = 0
    rejected_white = 0


    # ========================================================
    # LEVEL-0 VALIDATION
    # ========================================================

    for candidate_index in order:

        (
            x,
            y,
            approx_fraction,
        ) = candidates[
            candidate_index
        ]


        checked += 1


        tile = slide.read_region(
            (
                int(x),
                int(y),
            ),
            0,
            (
                TILE_SIZE,
                TILE_SIZE,
            ),
        ).convert(
            "RGB"
        )


        (
            final_tissue_fraction,
            background_fraction,
            white_fraction,
        ) = evaluate_level0_tile(
            tile,
            saturation_threshold,
        )


        # ----------------------------------------------------
        # Filter 1: actual H&E tissue fraction
        # ----------------------------------------------------

        if (
            final_tissue_fraction
            < FINAL_TISSUE_FRACTION
        ):

            rejected_tissue += 1
            continue


        # ----------------------------------------------------
        # Filter 2: grey scanner/background regions
        # ----------------------------------------------------

        if (
            background_fraction
            > MAX_BACKGROUND_FRACTION
        ):

            rejected_background += 1
            continue


        # ----------------------------------------------------
        # Filter 3: near-white areas
        # ----------------------------------------------------

        if (
            white_fraction
            > MAX_WHITE_FRACTION
        ):

            rejected_white += 1
            continue


        # ====================================================
        # ACCEPT TILE
        # ====================================================

        tile_number = (
            accepted + 1
        )


        tile_name = (
            f"tile_{tile_number:03d}"
            f"_x{x}"
            f"_y{y}.png"
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
                final_tissue_fraction
            ),

            "background_fraction": float(
                background_fraction
            ),

            "white_fraction": float(
                white_fraction
            ),

            "raw_otsu_saturation": float(
                raw_otsu
            ),

            "saturation_threshold": float(
                saturation_threshold
            ),

            "deterministic_seed": seed,

            "objective_power": objective_power,
            "mpp_x": mpp_x,
            "mpp_y": mpp_y,

            "tile_path": str(
                tile_path
            ),
        })


        accepted += 1


        if (
            accepted
            >= TARGET_TILES
        ):
            break


    slide.close()


    # ========================================================
    # CONTACT SHEET
    # ========================================================

    if (
        accepted
        == TARGET_TILES
    ):

        contact_path = (
            CONTACT_DIR
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
            f"only {accepted} valid tiles found"
        )


    qc_rows.append({
        "case_submitter_id": case_id,
        "class_label": class_label,

        "width": width,
        "height": height,

        "objective_power": objective_power,
        "mpp_x": mpp_x,
        "mpp_y": mpp_y,

        "thumbnail_width": thumb_w,
        "thumbnail_height": thumb_h,

        "raw_otsu_saturation": raw_otsu,
        "saturation_threshold": saturation_threshold,

        "tissue_mask_fraction": mask_fraction,

        "candidate_tiles": len(
            candidates
        ),

        "candidate_tiles_checked": checked,

        "rejected_low_tissue": rejected_tissue,

        "rejected_background": rejected_background,

        "rejected_white": rejected_white,

        "valid_tiles_selected": accepted,

        "status": status,
        "reason": reason,

        "mask_preview_path": str(
            mask_preview_path
        ),
    })


    print(
        f"    candidates checked:        "
        f"{checked}"
    )

    print(
        f"    rejected low tissue:       "
        f"{rejected_tissue}"
    )

    print(
        f"    rejected grey background:  "
        f"{rejected_background}"
    )

    print(
        f"    rejected near-white:       "
        f"{rejected_white}"
    )

    print(
        f"    selected:                  "
        f"{accepted}/{TARGET_TILES}"
    )

    print(
        f"    STATUS:                    "
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


tile_manifest_path = (
    MANIFEST_DIR
    / "pilot_tile_manifest_stage5_v2.csv"
)


qc_path = (
    METADATA_DIR
    / "pilot_tiling_qc_stage5_v2.csv"
)


summary_path = (
    METADATA_DIR
    / "pilot_tiling_summary_stage5_v2.json"
)


tiles_df.to_csv(
    tile_manifest_path,
    index=False,
)


qc_df.to_csv(
    qc_path,
    index=False,
)


# ============================================================
# SUMMARY
# ============================================================

patients_pass = int(
    (
        qc_df["status"]
        == "PASS"
    ).sum()
)


total_tiles = int(
    len(
        tiles_df
    )
)


expected_tiles = (
    len(pilot)
    * TARGET_TILES
)


overall_pass = (
    patients_pass
    == len(pilot)
    and
    total_tiles
    == expected_tiles
)


summary = {
    "protocol_version": "V2_HSV_saturation",

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

    "saturation_floor": (
        SATURATION_FLOOR
    ),

    "max_tissue_value": (
        MAX_TISSUE_VALUE
    ),

    "background_saturation_max": (
        BACKGROUND_SATURATION_MAX
    ),

    "background_value_min": (
        BACKGROUND_VALUE_MIN
    ),

    "max_background_fraction": (
        MAX_BACKGROUND_FRACTION
    ),

    "white_value_min": (
        WHITE_VALUE_MIN
    ),

    "max_white_fraction": (
        MAX_WHITE_FRACTION
    ),

    "total_tiles": (
        total_tiles
    ),

    "overall_pass": (
        overall_pass
    ),
}


summary_path.write_text(
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
print("=" * 78)
print(
    " PILOT TILE EXTRACTION V2 — SUMMARY"
)
print("=" * 78)


print(
    f"Patients passed:                "
    f"{patients_pass}/{len(pilot)}"
)


print(
    f"Total extracted tiles:          "
    f"{total_tiles}"
)


print(
    f"Expected tiles:                 "
    f"{expected_tiles}"
)


print("-" * 78)


display_columns = [
    "case_submitter_id",
    "class_label",
    "candidate_tiles",
    "candidate_tiles_checked",
    "rejected_low_tissue",
    "rejected_background",
    "rejected_white",
    "valid_tiles_selected",
    "status",
]


print()

print(
    qc_df[
        display_columns
    ].to_string(
        index=False
    )
)

print()


if overall_pass:

    print(
        "PILOT TILING V2 STATUS: PASS"
    )

else:

    print(
        "PILOT TILING V2 STATUS: CHECK REQUIRED"
    )


print()
print("Saved:")

print(
    "  data/tiles/pilot_v2/"
)

print(
    "  data/manifests/"
    "pilot_tile_manifest_stage5_v2.csv"
)

print(
    "  data/metadata/"
    "pilot_tiling_qc_stage5_v2.csv"
)

print(
    "  data/metadata/"
    "pilot_tiling_summary_stage5_v2.json"
)

print(
    "  results/figures/"
    "pilot_tile_contact_sheets_v2/"
)

print(
    "  results/figures/"
    "pilot_tissue_masks_v2/"
)

print("=" * 78)