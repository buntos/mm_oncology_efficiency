from pathlib import Path
import argparse
import hashlib
import json
import shutil
import subprocess
import time

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
# ROOT / INPUT FILES
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

MANIFEST_DIR = ROOT / "data" / "manifests"
METADATA_DIR = ROOT / "data" / "metadata"

COHORT_FILE = (
    MANIFEST_DIR
    / "paired_cohort_stage2.csv"
)

WSI_CANDIDATES_FILE = (
    MANIFEST_DIR
    / "wsi_primary_diagnostic_candidates.csv"
)

PILOT_FILE = (
    MANIFEST_DIR
    / "pilot_cohort_stage4.csv"
)

GDC_CLIENT = (
    ROOT
    / "tools"
    / "gdc-client.exe"
)


# ============================================================
# OUTPUT ROOTS
# ============================================================

TILE_ROOT = (
    ROOT
    / "data"
    / "tiles"
    / "production_v2"
)

RNA_ROOT = (
    ROOT
    / "data"
    / "rna"
    / "production_v2"
)

TEMP_GDC_ROOT = (
    ROOT
    / "data"
    / "temp_gdc_v2prod"
)

TEMP_TILE_ROOT = (
    ROOT
    / "data"
    / "temp_tiles_v2prod"
)

PATIENT_TILE_MANIFEST_ROOT = (
    MANIFEST_DIR
    / "production_v2"
    / "patient_tiles"
)

WSI_ATTEMPT_ROOT = (
    METADATA_DIR
    / "production_v2"
    / "wsi_attempts"
)

MASK_ROOT = (
    ROOT
    / "results"
    / "figures"
    / "production_v2_masks"
)

CONTACT_ROOT = (
    ROOT
    / "results"
    / "figures"
    / "production_v2_contact_sheets"
)

LOG_ROOT = (
    ROOT
    / "logs"
    / "production_v2"
)


for directory in [
    TILE_ROOT,
    RNA_ROOT,
    TEMP_GDC_ROOT,
    TEMP_TILE_ROOT,
    PATIENT_TILE_MANIFEST_ROOT,
    WSI_ATTEMPT_ROOT,
    MASK_ROOT,
    CONTACT_ROOT,
    LOG_ROOT,
]:
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# V2 FROZEN PREPROCESSING CONFIGURATION
# ============================================================

PROTOCOL_NAME = "V2_PROD_HSV_64x256"

TILE_SIZE = 256
STRIDE = 256
TARGET_TILES = 64

CANDIDATE_TISSUE_FRACTION = 0.70
FINAL_TISSUE_FRACTION = 0.90

THUMBNAIL_MAX_SIZE = 4096

SATURATION_FLOOR = 0.03
MAX_TISSUE_VALUE = 0.95

BACKGROUND_SATURATION_MAX = 0.04
BACKGROUND_VALUE_MIN = 0.55

MAX_BACKGROUND_FRACTION = 0.10

WHITE_VALUE_MIN = 0.97
MAX_WHITE_FRACTION = 0.10


# Old V2 used:
#
#   min_size=32
#   area_threshold=32
#
# In skimage 0.26:
#
#   max_size removes <= threshold.
#
# max_size=31 therefore preserves the old "< 32" behaviour.
MORPH_MAX_SIZE = 31


# ============================================================
# DOWNLOAD CONFIG
# ============================================================

DOWNLOAD_RETRIES = 3
DOWNLOAD_RETRY_DELAY_SECONDS = 5

MIN_FREE_SPACE_BUFFER_GB = 8.0


# ============================================================
# HELPERS
# ============================================================

def md5_file(
    path,
    chunk_size=8 * 1024 * 1024,
):
    h = hashlib.md5()

    with open(path, "rb") as f:

        while True:

            chunk = f.read(
                chunk_size
            )

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


def deterministic_seed(text):

    digest = hashlib.sha256(
        text.encode("utf-8")
    ).digest()

    return int.from_bytes(
        digest[:8],
        byteorder="little",
        signed=False,
    )


def deterministic_sort_key(text):

    return hashlib.sha256(
        (
            "V2_PROD_BATCH_SELECTION|"
            + str(text)
        ).encode("utf-8")
    ).hexdigest()


def integral_image(mask):

    return (
        mask.astype(np.uint32)
        .cumsum(axis=0)
        .cumsum(axis=1)
    )


def rect_sum(
    ii,
    x1,
    y1,
    x2,
    y2,
):

    x1 = max(0, x1)
    y1 = max(0, y1)

    x2 = min(
        ii.shape[1],
        x2,
    )

    y2 = min(
        ii.shape[0],
        y2,
    )

    if (
        x2 <= x1
        or
        y2 <= y1
    ):
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

    if (
        x1 > 0
        and
        y1 > 0
    ):
        result += int(
            ii[y1 - 1, x1 - 1]
        )

    return result


def rgb_to_hsv_arrays(
    rgb_uint8,
):

    rgb_float = (
        rgb_uint8.astype(
            np.float32
        )
        / 255.0
    )

    hsv = rgb2hsv(
        rgb_float
    )

    return (
        hsv[:, :, 1],
        hsv[:, :, 2],
    )


def create_tissue_mask(
    thumbnail,
):

    rgb = np.asarray(
        thumbnail
    )

    saturation, value = (
        rgb_to_hsv_arrays(
            rgb
        )
    )

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
        max_size=MORPH_MAX_SIZE,
    )

    tissue_mask = remove_small_holes(
        tissue_mask,
        max_size=MORPH_MAX_SIZE,
    )

    return (
        tissue_mask,
        raw_otsu,
        saturation_threshold,
    )


def evaluate_tile(
    tile,
    saturation_threshold,
):

    rgb = np.asarray(
        tile
    )

    saturation, value = (
        rgb_to_hsv_arrays(
            rgb
        )
    )

    tissue_pixels = (
        (saturation >= saturation_threshold)
        &
        (value <= MAX_TISSUE_VALUE)
    )

    background_pixels = (
        (saturation <= BACKGROUND_SATURATION_MAX)
        &
        (value >= BACKGROUND_VALUE_MIN)
    )

    white_pixels = (
        value >= WHITE_VALUE_MIN
    )

    return (
        float(
            tissue_pixels.mean()
        ),
        float(
            background_pixels.mean()
        ),
        float(
            white_pixels.mean()
        ),
    )


def save_mask_preview(
    thumbnail,
    tissue_mask,
    path,
):

    rgb = np.asarray(
        thumbnail
    ).copy()

    preview = rgb.astype(
        np.float32
    )

    preview[
        ~tissue_mask
    ] *= 0.25

    preview = np.clip(
        preview,
        0,
        255,
    ).astype(
        np.uint8
    )

    Image.fromarray(
        preview
    ).save(
        path,
        quality=90,
    )


def create_contact_sheet(
    tile_paths,
    out_path,
):

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
        quality=94,
    )

    for image in images:
        image.close()


def free_disk_gb():

    usage = shutil.disk_usage(
        ROOT
    )

    return (
        usage.free
        / 1024**3
    )


def remove_uuid_download(
    patient_temp_dir,
    file_id,
):

    uuid_dir = (
        patient_temp_dir
        / str(file_id)
    )

    if uuid_dir.exists():

        shutil.rmtree(
            uuid_dir,
            ignore_errors=True,
        )


def locate_downloaded_file(
    patient_temp_dir,
    file_id,
    file_name,
):

    expected = (
        patient_temp_dir
        / str(file_id)
        / str(file_name)
    )

    if expected.exists():
        return expected

    matches = list(
        patient_temp_dir.rglob(
            str(file_name)
        )
    )

    if len(matches) == 1:
        return matches[0]

    return None


def verify_downloaded_file(
    path,
    expected_size,
    expected_md5,
):

    if (
        path is None
        or
        not path.exists()
    ):
        return (
            False,
            "file_missing",
        )

    actual_size = (
        path.stat().st_size
    )

    if (
        int(actual_size)
        !=
        int(expected_size)
    ):
        return (
            False,
            (
                f"size_mismatch:"
                f"{actual_size}"
                f"!="
                f"{expected_size}"
            ),
        )

    actual_md5 = (
        md5_file(path)
        .lower()
    )

    if (
        actual_md5
        !=
        str(expected_md5).lower()
    ):
        return (
            False,
            "md5_mismatch",
        )

    return (
        True,
        "PASS",
    )


# ============================================================
# GDC DOWNLOAD
# ============================================================

def download_gdc_file(
    file_id,
    file_name,
    expected_size,
    expected_md5,
    patient_temp_dir,
):

    required_gb = (
        int(expected_size)
        / 1024**3
    )

    if (
        free_disk_gb()
        <
        required_gb
        + MIN_FREE_SPACE_BUFFER_GB
    ):

        raise RuntimeError(
            "Insufficient free disk space. "
            f"Need file {required_gb:.2f} GB "
            f"+ {MIN_FREE_SPACE_BUFFER_GB:.1f} GB buffer."
        )


    existing = locate_downloaded_file(
        patient_temp_dir,
        file_id,
        file_name,
    )

    ok, reason = verify_downloaded_file(
        existing,
        expected_size,
        expected_md5,
    )

    if ok:

        print(
            "      existing download verified"
        )

        return existing


    for attempt in range(
        1,
        DOWNLOAD_RETRIES + 1,
    ):

        print(
            f"      GDC download attempt "
            f"{attempt}/{DOWNLOAD_RETRIES}"
        )


        command = [
            str(GDC_CLIENT),
            "download",
            str(file_id),

            "-d",
            str(patient_temp_dir),

            "--no-related-files",
            "--no-annotations",
        ]


        subprocess.run(
            command,
            cwd=str(ROOT),
        )


        path = locate_downloaded_file(
            patient_temp_dir,
            file_id,
            file_name,
        )


        ok, reason = verify_downloaded_file(
            path,
            expected_size,
            expected_md5,
        )


        if ok:

            print(
                "      download + MD5: PASS"
            )

            return path


        print(
            f"      verification failed: "
            f"{reason}"
        )


        partials = list(
            (
                patient_temp_dir
                / str(file_id)
            ).glob(
                "*.partial"
            )
        )


        if partials:

            print(
                "      partial file detected; "
                "next attempt will resume"
            )

        else:

            if path is not None:

                remove_uuid_download(
                    patient_temp_dir,
                    file_id,
                )


        if (
            attempt
            < DOWNLOAD_RETRIES
        ):

            time.sleep(
                DOWNLOAD_RETRY_DELAY_SECONDS
            )


    raise RuntimeError(
        "GDC download failed after retries: "
        f"{file_id} ({file_name})"
    )


# ============================================================
# RNA QC
# ============================================================

def validate_rna(
    path,
):

    try:

        df = pd.read_csv(
            path,
            sep="\t",
            comment="#",
            low_memory=False,
        )

    except Exception as exc:

        return {
            "status": "FAIL",
            "reason": repr(exc),
        }


    columns = list(
        df.columns
    )


    if (
        "gene_id"
        not in columns
    ):

        return {
            "status": "FAIL",
            "reason": "gene_id_missing",
        }


    if len(df) < 1000:

        return {
            "status": "FAIL",
            "reason": (
                f"unexpected_row_count:"
                f"{len(df)}"
            ),
        }


    return {
        "status": "PASS",
        "reason": None,
        "rows": int(
            len(df)
        ),
        "columns_count": int(
            len(columns)
        ),
        "columns": ";".join(
            columns
        ),
    }


# ============================================================
# WSI TILE EXTRACTION
# ============================================================

def extract_wsi_tiles(
    slide_path,
    case_id,
    class_label,
    candidate_row,
    temporary_tile_dir,
    mask_preview_path,
):

    if temporary_tile_dir.exists():

        shutil.rmtree(
            temporary_tile_dir,
            ignore_errors=True,
        )

    temporary_tile_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    tile_records = []


    try:

        slide = openslide.OpenSlide(
            str(slide_path)
        )

    except Exception as exc:

        return (
            {
                "status": "FAIL",
                "reason": (
                    "openslide_open_failed:"
                    + repr(exc)
                ),
            },
            [],
        )


    try:

        width, height = (
            slide.dimensions
        )


        objective = (
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
        ) = create_tissue_mask(
            thumbnail
        )


        save_mask_preview(
            thumbnail,
            tissue_mask,
            mask_preview_path,
        )


        thumb_h, thumb_w = (
            tissue_mask.shape
        )


        scale_x = (
            width
            / thumb_w
        )

        scale_y = (
            height
            / thumb_h
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


                tissue_pixels = (
                    rect_sum(
                        ii,
                        tx1,
                        ty1,
                        tx2,
                        ty2,
                    )
                )


                approx_fraction = (
                    tissue_pixels
                    / area
                )


                if (
                    approx_fraction
                    >=
                    CANDIDATE_TISSUE_FRACTION
                ):

                    candidates.append(
                        (
                            x,
                            y,
                            approx_fraction,
                        )
                    )


        seed = deterministic_seed(
            case_id
            + "|"
            + str(
                candidate_row[
                    "file_id"
                ]
            )
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


        checked = 0
        accepted = 0

        rejected_low_tissue = 0
        rejected_background = 0
        rejected_white = 0


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
            ) = evaluate_tile(
                tile,
                saturation_threshold,
            )


            if (
                final_tissue_fraction
                <
                FINAL_TISSUE_FRACTION
            ):

                rejected_low_tissue += 1
                continue


            if (
                background_fraction
                >
                MAX_BACKGROUND_FRACTION
            ):

                rejected_background += 1
                continue


            if (
                white_fraction
                >
                MAX_WHITE_FRACTION
            ):

                rejected_white += 1
                continue


            tile_number = (
                accepted + 1
            )


            tile_name = (
                f"tile_{tile_number:03d}"
                f"_x{x}"
                f"_y{y}.png"
            )


            tile_path = (
                temporary_tile_dir
                / tile_name
            )


            tile.save(
                tile_path,
                format="PNG",
            )


            tile_records.append({
                "case_submitter_id": (
                    case_id
                ),

                "class_label": (
                    class_label
                ),

                "selected_wsi_file_id": (
                    candidate_row[
                        "file_id"
                    ]
                ),

                "selected_wsi_slide_token": (
                    candidate_row[
                        "slide_token"
                    ]
                ),

                "tile_index": (
                    tile_number
                ),

                "tile_name": (
                    tile_name
                ),

                "x_level0": int(
                    x
                ),

                "y_level0": int(
                    y
                ),

                "tile_size": (
                    TILE_SIZE
                ),

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

                "deterministic_seed": (
                    seed
                ),
            })


            accepted += 1


            if (
                accepted
                >= TARGET_TILES
            ):
                break


        qc = {
            "status": (
                "PASS"
                if accepted
                == TARGET_TILES
                else "FAIL"
            ),

            "reason": (
                None
                if accepted
                == TARGET_TILES
                else
                "insufficient_valid_tissue_tiles"
            ),

            "width": int(
                width
            ),

            "height": int(
                height
            ),

            "level_count": int(
                slide.level_count
            ),

            "objective_power": (
                objective
            ),

            "mpp_x": (
                mpp_x
            ),

            "mpp_y": (
                mpp_y
            ),

            "thumbnail_width": int(
                thumb_w
            ),

            "thumbnail_height": int(
                thumb_h
            ),

            "raw_otsu_saturation": float(
                raw_otsu
            ),

            "saturation_threshold": float(
                saturation_threshold
            ),

            "tissue_mask_fraction": float(
                tissue_mask.mean()
            ),

            "candidate_tiles": int(
                len(candidates)
            ),

            "candidate_tiles_checked": int(
                checked
            ),

            "rejected_low_tissue": int(
                rejected_low_tissue
            ),

            "rejected_background": int(
                rejected_background
            ),

            "rejected_white": int(
                rejected_white
            ),

            "valid_tiles_selected": int(
                accepted
            ),
        }


        return (
            qc,
            tile_records,
        )


    except Exception as exc:

        return (
            {
                "status": "FAIL",
                "reason": (
                    "tiling_exception:"
                    + repr(exc)
                ),
            },
            [],
        )


    finally:

        slide.close()


# ============================================================
# BATCH SELECTION
# ============================================================

def create_or_load_batch(
    cohort,
    pilot_cases,
    batch_path,
    limit,
    process_all,
    include_pilot,
):

    if batch_path.exists():

        print(
            "Using existing frozen batch manifest:"
        )

        print(
            f"  {batch_path}"
        )

        return pd.read_csv(
            batch_path
        )


    eligible = cohort.copy()


    if not include_pilot:

        eligible = eligible[
            ~eligible[
                "case_submitter_id"
            ].isin(
                pilot_cases
            )
        ].copy()


    if process_all:

        batch = eligible.sort_values(
            "case_submitter_id"
        ).copy()

    else:

        eligible[
            "_selection_key"
        ] = (
            eligible[
                "case_submitter_id"
            ]
            .apply(
                deterministic_sort_key
            )
        )


        luad_target = (
            limit // 2
        )

        lusc_target = (
            limit - luad_target
        )


        luad = (
            eligible[
                eligible[
                    "class_label"
                ]
                == "LUAD"
            ]
            .sort_values(
                "_selection_key"
            )
            .head(
                luad_target
            )
        )


        lusc = (
            eligible[
                eligible[
                    "class_label"
                ]
                == "LUSC"
            ]
            .sort_values(
                "_selection_key"
            )
            .head(
                lusc_target
            )
        )


        batch = (
            pd.concat(
                [
                    luad,
                    lusc,
                ],
                ignore_index=True,
            )
            .sort_values(
                "case_submitter_id"
            )
            .copy()
        )


        batch = batch.drop(
            columns=[
                "_selection_key"
            ],
            errors="ignore",
        )


    batch.to_csv(
        batch_path,
        index=False,
    )


    print(
        "Created frozen batch manifest:"
    )

    print(
        f"  {batch_path}"
    )


    return batch


# ============================================================
# RESULT UPDATE
# ============================================================

def update_patient_results(
    result_path,
    new_record,
):

    new_df = pd.DataFrame(
        [
            new_record
        ]
    )


    if result_path.exists():

        old = pd.read_csv(
            result_path
        )

        old = old[
            old[
                "case_submitter_id"
            ]
            !=
            new_record[
                "case_submitter_id"
            ]
        ]

        combined = pd.concat(
            [
                old,
                new_df,
            ],
            ignore_index=True,
        )

    else:

        combined = (
            new_df
        )


    combined = (
        combined
        .sort_values(
            "case_submitter_id"
        )
    )


    combined.to_csv(
        result_path,
        index=False,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Production V2 WSI/RNA preprocessing "
            "with deterministic DX fallback."
        )
    )


    parser.add_argument(
        "--batch-name",
        default="validation30_v2prod",
    )


    parser.add_argument(
        "--limit",
        type=int,
        default=30,
    )


    parser.add_argument(
        "--all",
        action="store_true",
        help=(
            "Process all eligible cohort patients "
            "instead of a validation subset."
        ),
    )


    parser.add_argument(
        "--include-pilot",
        action="store_true",
    )


    parser.add_argument(
        "--keep-wsi",
        action="store_true",
        help=(
            "Do not delete downloaded WSI files "
            "after QC."
        ),
    )


    parser.add_argument(
        "--contact-sheets",
        action="store_true",
        help=(
            "Save an 8x8 contact sheet for each "
            "successfully processed patient."
        ),
    )


    args = parser.parse_args()


    if not GDC_CLIENT.exists():

        raise FileNotFoundError(
            f"GDC client not found: "
            f"{GDC_CLIENT}"
        )


    cohort = pd.read_csv(
        COHORT_FILE
    )


    wsi_candidates = pd.read_csv(
        WSI_CANDIDATES_FILE
    )


    pilot_cases = set()


    if PILOT_FILE.exists():

        pilot_df = pd.read_csv(
            PILOT_FILE
        )

        pilot_cases = set(
            pilot_df[
                "case_submitter_id"
            ]
        )


    batch_path = (
        MANIFEST_DIR
        / f"{args.batch_name}_patients.csv"
    )


    patient_result_path = (
        METADATA_DIR
        / f"{args.batch_name}_patient_qc.csv"
    )


    combined_tile_manifest_path = (
        MANIFEST_DIR
        / f"{args.batch_name}_tile_manifest.csv"
    )


    batch = create_or_load_batch(
        cohort,
        pilot_cases,
        batch_path,
        args.limit,
        args.all,
        args.include_pilot,
    )


    print()
    print("=" * 80)
    print(
        " V2_PROD MULTIMODAL COHORT PROCESSING"
    )
    print("=" * 80)

    print(
        f"Protocol:                  "
        f"{PROTOCOL_NAME}"
    )

    print(
        f"Batch:                     "
        f"{args.batch_name}"
    )

    print(
        f"Patients:                  "
        f"{len(batch)}"
    )

    print(
        f"LUAD / LUSC:               "
        f"{(batch['class_label'] == 'LUAD').sum()} / "
        f"{(batch['class_label'] == 'LUSC').sum()}"
    )

    print(
        f"Delete WSI after QC:        "
        f"{not args.keep_wsi}"
    )

    print(
        f"Contact sheets:             "
        f"{args.contact_sheets}"
    )

    print(
        f"Free disk now:              "
        f"{free_disk_gb():.2f} GB"
    )

    print("=" * 80)
    print()


    # PASS and genuine QC exclusions are terminal.
    # CHECK_REQUIRED remains retryable.
    terminal_cases = set()

    if patient_result_path.exists():

        existing_results = pd.read_csv(
            patient_result_path
        )

        terminal_cases = set(
            existing_results.loc[
                existing_results[
                    "status"
                ].isin([
                    "PASS",
                    "EXCLUDED_WSI_QC",
                ]),
                "case_submitter_id",
            ]
        )


    for patient_number, patient in batch.iterrows():

        case_id = (
            patient[
                "case_submitter_id"
            ]
        )

        class_label = (
            patient[
                "class_label"
            ]
        )


        print()
        print("=" * 80)

        print(
            f"[{patient_number + 1:03d}/{len(batch):03d}] "
            f"{case_id} ({class_label})"
        )

        print("=" * 80)


        if case_id in terminal_cases:

            print(
                "Already has terminal result — SKIP"
            )

            continue


        patient_start = time.perf_counter()


        patient_temp_dir = (
            TEMP_GDC_ROOT
            / case_id
        )

        patient_temp_dir.mkdir(
            parents=True,
            exist_ok=True,
        )


        # ====================================================
        # WSI CANDIDATES / DX FALLBACK
        # ====================================================

        patient_wsi = (
            wsi_candidates[
                wsi_candidates[
                    "case_submitter_id"
                ]
                == case_id
            ]
            .sort_values(
                [
                    "dx_index",
                    "associated_barcode",
                    "file_id",
                ]
            )
            .copy()
        )


        print(
            f"Diagnostic WSI candidates: "
            f"{len(patient_wsi)}"
        )


        attempt_records = []

        selected_candidate = None
        selected_tile_records = None
        selected_temp_tile_dir = None


        for attempt_number, (
            _,
            candidate,
        ) in enumerate(
            patient_wsi.iterrows(),
            start=1,
        ):

            slide_token = (
                candidate[
                    "slide_token"
                ]
            )

            file_id = (
                candidate[
                    "file_id"
                ]
            )


            print()
            print(
                f"  WSI attempt "
                f"{attempt_number}/{len(patient_wsi)}: "
                f"{slide_token}"
            )

            print(
                f"    UUID: {file_id}"
            )

            print(
                f"    size: "
                f"{candidate['file_size_bytes'] / 1024**3:.3f} GB"
            )


            attempt_start = (
                time.perf_counter()
            )


            try:

                wsi_path = download_gdc_file(
                    file_id=file_id,
                    file_name=(
                        candidate[
                            "file_name"
                        ]
                    ),
                    expected_size=(
                        candidate[
                            "file_size_bytes"
                        ]
                    ),
                    expected_md5=(
                        candidate[
                            "md5sum"
                        ]
                    ),
                    patient_temp_dir=(
                        patient_temp_dir
                    ),
                )


                temp_tile_dir = (
                    TEMP_TILE_ROOT
                    / case_id
                    / str(file_id)
                )


                mask_batch_dir = (
                    MASK_ROOT
                    / args.batch_name
                )

                mask_batch_dir.mkdir(
                    parents=True,
                    exist_ok=True,
                )


                mask_path = (
                    mask_batch_dir
                    /
                    (
                        f"{case_id}_"
                        f"{slide_token}_mask.jpg"
                    )
                )


                (
                    wsi_qc,
                    tile_records,
                ) = extract_wsi_tiles(
                    slide_path=wsi_path,
                    case_id=case_id,
                    class_label=class_label,
                    candidate_row=candidate,
                    temporary_tile_dir=temp_tile_dir,
                    mask_preview_path=mask_path,
                )


                attempt_record = {
                    "case_submitter_id": (
                        case_id
                    ),

                    "class_label": (
                        class_label
                    ),

                    "attempt_number": (
                        attempt_number
                    ),

                    "slide_token": (
                        slide_token
                    ),

                    "file_id": (
                        file_id
                    ),

                    "file_name": (
                        candidate[
                            "file_name"
                        ]
                    ),

                    "file_size_gb": float(
                        candidate[
                            "file_size_bytes"
                        ]
                        / 1024**3
                    ),

                    **wsi_qc,

                    "attempt_seconds": (
                        time.perf_counter()
                        - attempt_start
                    ),
                }


                attempt_records.append(
                    attempt_record
                )


                print(
                    f"    candidates: "
                    f"{wsi_qc.get('candidate_tiles')}"
                )

                print(
                    f"    checked:    "
                    f"{wsi_qc.get('candidate_tiles_checked')}"
                )

                print(
                    f"    selected:   "
                    f"{wsi_qc.get('valid_tiles_selected')}/64"
                )

                print(
                    f"    WSI QC:     "
                    f"{wsi_qc['status']}"
                )


                if (
                    wsi_qc[
                        "status"
                    ]
                    == "PASS"
                ):

                    selected_candidate = (
                        candidate
                    )

                    selected_tile_records = (
                        tile_records
                    )

                    selected_temp_tile_dir = (
                        temp_tile_dir
                    )

                    break


                # A slide that really reached WSI QC but failed
                # can be removed before trying DX2/DX3.
                if temp_tile_dir.exists():

                    shutil.rmtree(
                        temp_tile_dir,
                        ignore_errors=True,
                    )


                if not args.keep_wsi:

                    remove_uuid_download(
                        patient_temp_dir,
                        file_id,
                    )


            except Exception as exc:

                print(
                    f"    WSI attempt ERROR: "
                    f"{repr(exc)}"
                )


                attempt_records.append({
                    "case_submitter_id": (
                        case_id
                    ),

                    "class_label": (
                        class_label
                    ),

                    "attempt_number": (
                        attempt_number
                    ),

                    "slide_token": (
                        slide_token
                    ),

                    "file_id": (
                        file_id
                    ),

                    "status": "ERROR",

                    "reason": repr(
                        exc
                    ),

                    "attempt_seconds": (
                        time.perf_counter()
                        - attempt_start
                    ),
                })


        # Save all DX attempts for reproducibility.
        attempts_df = pd.DataFrame(
            attempt_records
        )

        attempts_df.to_csv(
            WSI_ATTEMPT_ROOT
            / f"{case_id}.csv",
            index=False,
        )


        # ====================================================
        # NO WSI PASSED
        # ====================================================

        if selected_candidate is None:

            attempt_statuses = [
                str(
                    record.get(
                        "status",
                        "",
                    )
                )
                for record
                in attempt_records
            ]

            technical_error_count = sum(
                status == "ERROR"
                for status
                in attempt_statuses
            )

            qc_fail_count = sum(
                status == "FAIL"
                for status
                in attempt_statuses
            )


            # No candidate in the manifest is not a WSI QC fail.
            if len(patient_wsi) == 0:

                patient_status = (
                    "CHECK_REQUIRED"
                )

                patient_reason = (
                    "no_diagnostic_wsi_candidates_in_manifest"
                )


            # If at least one diagnostic slide could not even
            # be evaluated because of download/network/etc.,
            # the patient must remain retryable.
            elif technical_error_count > 0:

                patient_status = (
                    "CHECK_REQUIRED"
                )

                patient_reason = (
                    "wsi_technical_error_retry_required"
                )


            # Only if ALL diagnostic slides were actually
            # evaluated and none passed do we declare a real
            # WSI QC exclusion.
            else:

                patient_status = (
                    "EXCLUDED_WSI_QC"
                )

                patient_reason = (
                    "all_diagnostic_slides_failed_wsi_qc"
                )


            print()
            print(
                f"PATIENT RESULT: "
                f"{patient_status}"
            )

            print(
                f"  WSI QC fails:      "
                f"{qc_fail_count}"
            )

            print(
                f"  technical errors:  "
                f"{technical_error_count}"
            )


            patient_record = {
                "case_submitter_id": (
                    case_id
                ),

                "class_label": (
                    class_label
                ),

                "status": (
                    patient_status
                ),

                "reason": (
                    patient_reason
                ),

                "diagnostic_slides_attempted": (
                    len(
                        attempt_records
                    )
                ),

                "wsi_qc_fail_count": (
                    qc_fail_count
                ),

                "wsi_technical_error_count": (
                    technical_error_count
                ),

                "selected_slide_token": (
                    None
                ),

                "selected_wsi_file_id": (
                    None
                ),

                "rna_file_id": (
                    patient[
                        "rna_file_id"
                    ]
                ),

                "patient_seconds": (
                    time.perf_counter()
                    - patient_start
                ),
            }


            update_patient_results(
                patient_result_path,
                patient_record,
            )

            continue


        # ====================================================
        # RNA DOWNLOAD + QC
        # ====================================================

        print()
        print(
            "  WSI selected:"
        )

        print(
            f"    "
            f"{selected_candidate['slide_token']}"
        )


        rna_file_id = (
            patient[
                "rna_file_id"
            ]
        )

        rna_file_name = (
            patient[
                "rna_file_name"
            ]
        )


        print()
        print(
            "  RNA download/QC"
        )


        try:

            final_rna_dir = (
                RNA_ROOT
                / case_id
            )

            final_rna_dir.mkdir(
                parents=True,
                exist_ok=True,
            )


            final_rna_path = (
                final_rna_dir
                / str(
                    rna_file_name
                )
            )


            # Reuse final RNA if already valid.
            rna_ok, _ = (
                verify_downloaded_file(
                    final_rna_path,
                    patient[
                        "rna_file_size_bytes"
                    ],
                    patient[
                        "rna_md5sum"
                    ],
                )
            )


            if not rna_ok:

                downloaded_rna = (
                    download_gdc_file(
                        file_id=(
                            rna_file_id
                        ),

                        file_name=(
                            rna_file_name
                        ),

                        expected_size=(
                            patient[
                                "rna_file_size_bytes"
                            ]
                        ),

                        expected_md5=(
                            patient[
                                "rna_md5sum"
                            ]
                        ),

                        patient_temp_dir=(
                            patient_temp_dir
                        ),
                    )
                )


                shutil.copy2(
                    downloaded_rna,
                    final_rna_path,
                )


                rna_ok, rna_reason = (
                    verify_downloaded_file(
                        final_rna_path,
                        patient[
                            "rna_file_size_bytes"
                        ],
                        patient[
                            "rna_md5sum"
                        ],
                    )
                )


                if not rna_ok:

                    raise RuntimeError(
                        "RNA copy verification failed: "
                        f"{rna_reason}"
                    )


            rna_qc = validate_rna(
                final_rna_path
            )


            if (
                rna_qc[
                    "status"
                ]
                != "PASS"
            ):

                raise RuntimeError(
                    "RNA QC failed: "
                    f"{rna_qc['reason']}"
                )


            print(
                f"    RNA QC: PASS "
                f"({rna_qc['rows']} rows)"
            )


        except Exception as exc:

            print(
                f"    RNA ERROR: "
                f"{repr(exc)}"
            )


            if (
                selected_temp_tile_dir
                is not None
                and
                selected_temp_tile_dir.exists()
            ):

                shutil.rmtree(
                    selected_temp_tile_dir,
                    ignore_errors=True,
                )


            patient_record = {
                "case_submitter_id": (
                    case_id
                ),

                "class_label": (
                    class_label
                ),

                "status": (
                    "CHECK_REQUIRED"
                ),

                "reason": (
                    "RNA_QC_OR_DOWNLOAD_FAILED:"
                    + repr(exc)
                ),

                "diagnostic_slides_attempted": (
                    len(
                        attempt_records
                    )
                ),

                "selected_slide_token": (
                    selected_candidate[
                        "slide_token"
                    ]
                ),

                "selected_wsi_file_id": (
                    selected_candidate[
                        "file_id"
                    ]
                ),

                "rna_file_id": (
                    rna_file_id
                ),

                "patient_seconds": (
                    time.perf_counter()
                    - patient_start
                ),
            }


            update_patient_results(
                patient_result_path,
                patient_record,
            )

            continue


        # ====================================================
        # COMMIT FINAL TILES
        # ====================================================

        final_tile_dir = (
            TILE_ROOT
            / case_id
        )


        if final_tile_dir.exists():

            shutil.rmtree(
                final_tile_dir,
                ignore_errors=True,
            )


        final_tile_dir.parent.mkdir(
            parents=True,
            exist_ok=True,
        )


        shutil.move(
            str(
                selected_temp_tile_dir
            ),
            str(
                final_tile_dir
            ),
        )


        for record in selected_tile_records:

            record[
                "tile_path"
            ] = str(
                final_tile_dir
                / record[
                    "tile_name"
                ]
            )


        patient_tile_manifest = (
            pd.DataFrame(
                selected_tile_records
            )
        )


        patient_tile_manifest.to_csv(
            PATIENT_TILE_MANIFEST_ROOT
            / f"{case_id}.csv",
            index=False,
        )


        if args.contact_sheets:

            contact_batch_dir = (
                CONTACT_ROOT
                / args.batch_name
            )

            contact_batch_dir.mkdir(
                parents=True,
                exist_ok=True,
            )


            selected_paths = [
                final_tile_dir
                / record[
                    "tile_name"
                ]
                for record
                in selected_tile_records
            ]


            create_contact_sheet(
                selected_paths,
                contact_batch_dir
                / f"{case_id}_64_tiles.jpg",
            )


        # ====================================================
        # CLEAN DOWNLOADS
        # ====================================================

        if not args.keep_wsi:

            remove_uuid_download(
                patient_temp_dir,
                selected_candidate[
                    "file_id"
                ],
            )


        remove_uuid_download(
            patient_temp_dir,
            rna_file_id,
        )


        try:

            if (
                patient_temp_dir.exists()
                and
                not any(
                    patient_temp_dir.iterdir()
                )
            ):

                patient_temp_dir.rmdir()

        except OSError:
            pass


        # ====================================================
        # PATIENT PASS
        # ====================================================

        failed_tokens = [
            str(
                record.get(
                    "slide_token"
                )
            )
            for record
            in attempt_records
            if (
                record.get(
                    "status"
                )
                != "PASS"
            )
        ]


        patient_record = {
            "case_submitter_id": (
                case_id
            ),

            "class_label": (
                class_label
            ),

            "status": "PASS",

            "reason": (
                None
            ),

            "diagnostic_slides_attempted": (
                len(
                    attempt_records
                )
            ),

            "failed_slide_tokens": (
                ";".join(
                    failed_tokens
                )
            ),

            "selected_slide_token": (
                selected_candidate[
                    "slide_token"
                ]
            ),

            "selected_wsi_file_id": (
                selected_candidate[
                    "file_id"
                ]
            ),

            "selected_wsi_file_name": (
                selected_candidate[
                    "file_name"
                ]
            ),

            "rna_file_id": (
                rna_file_id
            ),

            "rna_file_name": (
                rna_file_name
            ),

            "valid_tiles": (
                TARGET_TILES
            ),

            "rna_rows": (
                rna_qc[
                    "rows"
                ]
            ),

            "patient_seconds": (
                time.perf_counter()
                - patient_start
            ),
        }


        update_patient_results(
            patient_result_path,
            patient_record,
        )


        print()
        print(
            "PATIENT RESULT: PASS"
        )

        print(
            f"  selected slide: "
            f"{selected_candidate['slide_token']}"
        )

        print(
            f"  tiles:          64"
        )

        print(
            f"  elapsed:        "
            f"{patient_record['patient_seconds'] / 60:.1f} min"
        )

        print(
            f"  free disk:      "
            f"{free_disk_gb():.2f} GB"
        )


    # ========================================================
    # COMBINE TILE MANIFESTS FOR THIS BATCH
    # ========================================================

    tile_frames = []


    for case_id in batch[
        "case_submitter_id"
    ]:

        path = (
            PATIENT_TILE_MANIFEST_ROOT
            / f"{case_id}.csv"
        )

        if path.exists():

            tile_frames.append(
                pd.read_csv(
                    path
                )
            )


    if tile_frames:

        combined_tiles = pd.concat(
            tile_frames,
            ignore_index=True,
        )

        combined_tiles.to_csv(
            combined_tile_manifest_path,
            index=False,
        )


    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print("=" * 80)
    print(
        " V2_PROD BATCH SUMMARY"
    )
    print("=" * 80)


    if patient_result_path.exists():

        results = pd.read_csv(
            patient_result_path
        )


        results = results[
            results[
                "case_submitter_id"
            ].isin(
                set(
                    batch[
                        "case_submitter_id"
                    ]
                )
            )
        ]


        status_counts = (
            results[
                "status"
            ]
            .value_counts()
        )


        for status, count in (
            status_counts.items()
        ):

            print(
                f"{status:<25} "
                f"{count}"
            )


        pass_df = results[
            results[
                "status"
            ]
            == "PASS"
        ]


        if not pass_df.empty:

            print()
            print(
                "Selected slide distribution:"
            )

            print(
                pass_df[
                    "selected_slide_token"
                ]
                .value_counts()
                .sort_index()
                .to_string()
            )


            print()
            print(
                "DX fallback used:"
            )

            fallback_count = int(
                (
                    pass_df[
                        "diagnostic_slides_attempted"
                    ]
                    > 1
                ).sum()
            )

            print(
                f"  {fallback_count} patients"
            )


    print()
    print(
        "Patient QC:"
    )

    print(
        f"  {patient_result_path}"
    )

    print(
        "Combined tile manifest:"
    )

    print(
        f"  {combined_tile_manifest_path}"
    )

    print(
        "WSI attempt logs:"
    )

    print(
        f"  {WSI_ATTEMPT_ROOT}"
    )

    print("=" * 80)


if __name__ == "__main__":
    main()