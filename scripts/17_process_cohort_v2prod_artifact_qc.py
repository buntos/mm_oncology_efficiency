from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import math
import shutil
import sys

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


# ============================================================
# LOAD FROZEN V2_PROD IMPLEMENTATION FROM SCRIPT 11
# ============================================================

THIS_FILE = Path(__file__).resolve()
ROOT = THIS_FILE.parents[1]

BASE_SCRIPT = (
    THIS_FILE.parent
    / "11_process_cohort_batch_v2_prod.py"
)

if not BASE_SCRIPT.exists():
    raise FileNotFoundError(
        f"Base V2_PROD script not found: {BASE_SCRIPT}"
    )


spec = importlib.util.spec_from_file_location(
    "v2prod_base",
    BASE_SCRIPT,
)

base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)


# ============================================================
# PROTOCOL
# ============================================================

PROTOCOL_NAMESPACE = (
    "production_v2_artifact_qc_candidate_v1"
)

PROTOCOL_NAME = (
    "V2_PROD_HSV_64x256"
    "+ARTIFACT_QC_SENSITIVE_CANDIDATE_V1"
)

ARTIFACT_PROTOCOL_NAME = (
    "SENSITIVE_CANDIDATE_V1"
)


# ============================================================
# FROZEN CANDIDATE ARTIFACT RULES
# ============================================================
#
# IMPORTANT:
#
# These thresholds were selected after the initial
# validation30 diagnostic investigation.
#
# They are now FROZEN for independent validation.
#
# DO NOT tune these values on the next independent batch.
# ============================================================

ARTIFACT_RULES = {

    "near_black_fraction_min": 0.30,

    "flat_gradient_max": 0.30,
    "flat_red_excess_min": 125.0,
    "flat_cyan_excess_min": 8.0,

    "hyper_red_excess_min": 140.0,
    "hyper_red_saturation_min": 0.75,

    "hyper_cyan_excess_min": 18.0,
    "hyper_cyan_saturation_min": 0.60,
}


# ============================================================
# KEEP ORIGINAL INPUT ROOTS
# ============================================================

ORIGINAL_MANIFEST_DIR = base.MANIFEST_DIR
ORIGINAL_METADATA_DIR = base.METADATA_DIR


# ============================================================
# NEW PROTOCOL-SPECIFIC OUTPUT ROOTS
# ============================================================

PROTOCOL_MANIFEST_ROOT = (
    ORIGINAL_MANIFEST_DIR
    / PROTOCOL_NAMESPACE
)

PROTOCOL_BATCH_MANIFEST_ROOT = (
    PROTOCOL_MANIFEST_ROOT
    / "batches"
)

PROTOCOL_METADATA_ROOT = (
    ORIGINAL_METADATA_DIR
    / PROTOCOL_NAMESPACE
)

PROTOCOL_BATCH_METADATA_ROOT = (
    PROTOCOL_METADATA_ROOT
    / "batches"
)

TILE_ROOT = (
    ROOT
    / "data"
    / "tiles"
    / PROTOCOL_NAMESPACE
)

TEMP_GDC_ROOT = (
    ROOT
    / "data"
    / "temp_gdc_v2prod_artifact_qc"
)

TEMP_TILE_ROOT = (
    ROOT
    / "data"
    / "temp_tiles_v2prod_artifact_qc"
)

PATIENT_TILE_MANIFEST_ROOT = (
    PROTOCOL_MANIFEST_ROOT
    / "patient_tiles"
)

WSI_ATTEMPT_ROOT = (
    PROTOCOL_METADATA_ROOT
    / "wsi_attempts"
)

ARTIFACT_AUDIT_ROOT = (
    PROTOCOL_METADATA_ROOT
    / "artifact_rejections"
)

MASK_ROOT = (
    ROOT
    / "results"
    / "figures"
    / "production_v2_artifact_qc_masks"
)

CONTACT_ROOT = (
    ROOT
    / "results"
    / "figures"
    / "production_v2_artifact_qc_contact_sheets"
)

ARTIFACT_IMAGE_ROOT = (
    ROOT
    / "results"
    / "figures"
    / "production_v2_artifact_qc_rejections"
)

LOG_ROOT = (
    ROOT
    / "logs"
    / PROTOCOL_NAMESPACE
)


for directory in [
    PROTOCOL_MANIFEST_ROOT,
    PROTOCOL_BATCH_MANIFEST_ROOT,
    PROTOCOL_METADATA_ROOT,
    PROTOCOL_BATCH_METADATA_ROOT,
    TILE_ROOT,
    TEMP_GDC_ROOT,
    TEMP_TILE_ROOT,
    PATIENT_TILE_MANIFEST_ROOT,
    WSI_ATTEMPT_ROOT,
    ARTIFACT_AUDIT_ROOT,
    MASK_ROOT,
    CONTACT_ROOT,
    ARTIFACT_IMAGE_ROOT,
    LOG_ROOT,
]:
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# PATCH BASE OUTPUT PATHS
# ============================================================
#
# RNA_ROOT intentionally stays production_v2.
#
# RNA preprocessing has not changed, and files are MD5
# verified before reuse. There is no reason to duplicate
# the same STAR Counts files.
# ============================================================

base.PROTOCOL_NAME = PROTOCOL_NAME

base.MANIFEST_DIR = (
    PROTOCOL_BATCH_MANIFEST_ROOT
)

base.METADATA_DIR = (
    PROTOCOL_BATCH_METADATA_ROOT
)

base.TILE_ROOT = TILE_ROOT

base.TEMP_GDC_ROOT = TEMP_GDC_ROOT
base.TEMP_TILE_ROOT = TEMP_TILE_ROOT

base.PATIENT_TILE_MANIFEST_ROOT = (
    PATIENT_TILE_MANIFEST_ROOT
)

base.WSI_ATTEMPT_ROOT = (
    WSI_ATTEMPT_ROOT
)

base.MASK_ROOT = MASK_ROOT
base.CONTACT_ROOT = CONTACT_ROOT
base.LOG_ROOT = LOG_ROOT


# ============================================================
# WRAPPER RUNTIME STATE
# ============================================================

ACTIVE_BATCH_NAME = None

EXPLICIT_CASES = None

FORCE_REPROCESS = False

SAVE_REJECTED_ARTIFACTS = False


# ============================================================
# FREEZE ARTIFACT RULE CONFIG
# ============================================================

FROZEN_RULES_PATH = (
    PROTOCOL_METADATA_ROOT
    / "artifact_qc_candidate_v1_frozen_rules.json"
)


def write_or_verify_frozen_rules():

    payload = {
        "protocol_namespace": (
            PROTOCOL_NAMESPACE
        ),

        "protocol_name": (
            PROTOCOL_NAME
        ),

        "artifact_protocol_name": (
            ARTIFACT_PROTOCOL_NAME
        ),

        "status": (
            "FROZEN_FOR_INDEPENDENT_VALIDATION"
        ),

        "source_validation_batch": (
            "validation30_v2prod"
        ),

        "rules": (
            ARTIFACT_RULES
        ),

        "important_note": (
            "These thresholds must not be tuned "
            "on the independent validation batch."
        ),
    }


    if FROZEN_RULES_PATH.exists():

        existing = json.loads(
            FROZEN_RULES_PATH.read_text(
                encoding="utf-8"
            )
        )


        if existing != payload:

            raise RuntimeError(
                "Frozen artifact-QC rule file differs "
                "from the implementation. "
                "Refusing to continue."
            )


        return


    FROZEN_RULES_PATH.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


# ============================================================
# ARTIFACT METRICS
# ============================================================

def calculate_artifact_metrics(
    tile,
):

    rgb = np.asarray(
        tile.convert("RGB"),
        dtype=np.float32,
    )


    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]


    max_channel = rgb.max(
        axis=2
    )


    near_black_fraction = float(
        (
            max_channel
            < 50
        ).mean()
    )


    gray = (
        0.299 * r
        +
        0.587 * g
        +
        0.114 * b
    )


    grad_x = np.abs(
        np.diff(
            gray,
            axis=1,
        )
    )

    grad_y = np.abs(
        np.diff(
            gray,
            axis=0,
        )
    )


    gradient_mean = float(
        (
            grad_x.mean()
            +
            grad_y.mean()
        )
        / 2.0
    )


    red_excess_map = (
        r
        -
        (
            g + b
        )
        / 2.0
    )


    cyan_excess_map = (
        (
            g + b
        )
        / 2.0
        -
        r
    )


    red_excess = float(
        red_excess_map.mean()
    )

    cyan_excess = float(
        cyan_excess_map.mean()
    )


    saturation, _ = (
        base.rgb_to_hsv_arrays(
            rgb.astype(
                np.uint8
            )
        )
    )


    saturation_mean = float(
        saturation.mean()
    )


    return {
        "near_black_fraction": (
            near_black_fraction
        ),

        "gradient_mean": (
            gradient_mean
        ),

        "red_excess": (
            red_excess
        ),

        "cyan_excess": (
            cyan_excess
        ),

        "saturation_mean": (
            saturation_mean
        ),
    }


# ============================================================
# ARTIFACT DECISION
# ============================================================

def evaluate_artifact_qc(
    metrics,
):

    reasons = []


    if (
        metrics[
            "near_black_fraction"
        ]
        >=
        ARTIFACT_RULES[
            "near_black_fraction_min"
        ]
    ):

        reasons.append(
            "near_black"
        )


    if (
        metrics[
            "gradient_mean"
        ]
        <=
        ARTIFACT_RULES[
            "flat_gradient_max"
        ]
        and
        metrics[
            "red_excess"
        ]
        >=
        ARTIFACT_RULES[
            "flat_red_excess_min"
        ]
    ):

        reasons.append(
            "flat_extreme_red"
        )


    if (
        metrics[
            "gradient_mean"
        ]
        <=
        ARTIFACT_RULES[
            "flat_gradient_max"
        ]
        and
        metrics[
            "cyan_excess"
        ]
        >=
        ARTIFACT_RULES[
            "flat_cyan_excess_min"
        ]
    ):

        reasons.append(
            "flat_extreme_cyan"
        )


    if (
        metrics[
            "red_excess"
        ]
        >=
        ARTIFACT_RULES[
            "hyper_red_excess_min"
        ]
        and
        metrics[
            "saturation_mean"
        ]
        >=
        ARTIFACT_RULES[
            "hyper_red_saturation_min"
        ]
    ):

        reasons.append(
            "hyper_red_saturated"
        )


    if (
        metrics[
            "cyan_excess"
        ]
        >=
        ARTIFACT_RULES[
            "hyper_cyan_excess_min"
        ]
        and
        metrics[
            "saturation_mean"
        ]
        >=
        ARTIFACT_RULES[
            "hyper_cyan_saturation_min"
        ]
    ):

        reasons.append(
            "hyper_cyan_saturated"
        )


    return (
        len(reasons) > 0,
        reasons,
    )


# ============================================================
# ARTIFACT CONTACT SHEET
# ============================================================

def load_font(
    size,
):

    candidates = [
        Path(
            r"C:\Windows\Fonts\segoeui.ttf"
        ),
        Path(
            r"C:\Windows\Fonts\arial.ttf"
        ),
    ]


    for path in candidates:

        if path.exists():

            try:

                return ImageFont.truetype(
                    str(path),
                    size=size,
                )

            except Exception:
                pass


    return ImageFont.load_default()


def create_artifact_contact_sheet(
    artifact_records,
    output_path,
):

    records = [
        record
        for record in artifact_records
        if (
            record.get(
                "artifact_image_path"
            )
            and
            Path(
                record[
                    "artifact_image_path"
                ]
            ).exists()
        )
    ]


    if not records:
        return


    columns = 4

    preview_size = 256
    label_height = 68
    header_height = 56


    rows = math.ceil(
        len(records)
        / columns
    )


    sheet = Image.new(
        "RGB",
        (
            columns
            * preview_size,
            header_height
            +
            rows
            *
            (
                preview_size
                +
                label_height
            ),
        ),
        "white",
    )


    draw = ImageDraw.Draw(
        sheet
    )


    title_font = load_font(
        22
    )

    label_font = load_font(
        13
    )


    draw.text(
        (
            10,
            10,
        ),
        (
            "Artifact-QC rejected candidates — "
            f"{ARTIFACT_PROTOCOL_NAME}"
        ),
        fill="black",
        font=title_font,
    )


    for i, record in enumerate(
        records
    ):

        column = (
            i % columns
        )

        row = (
            i // columns
        )


        x = (
            column
            * preview_size
        )

        y = (
            header_height
            +
            row
            *
            (
                preview_size
                +
                label_height
            )
        )


        with Image.open(
            record[
                "artifact_image_path"
            ]
        ) as source:

            image = source.convert(
                "RGB"
            )


            if image.size != (
                preview_size,
                preview_size,
            ):

                image = image.resize(
                    (
                        preview_size,
                        preview_size,
                    ),
                    Image.Resampling.LANCZOS,
                )


            sheet.paste(
                image,
                (
                    x,
                    y,
                ),
            )


        draw.text(
            (
                x + 3,
                y + preview_size + 3,
            ),
            (
                f"candidate #{record['candidate_checked_index']} "
                f"x={record['x_level0']} "
                f"y={record['y_level0']}"
            ),
            fill="black",
            font=label_font,
        )


        draw.text(
            (
                x + 3,
                y + preview_size + 25,
            ),
            str(
                record[
                    "artifact_reasons"
                ]
            ),
            fill="black",
            font=label_font,
        )


        draw.text(
            (
                x + 3,
                y + preview_size + 47,
            ),
            (
                f"grad={record['gradient_mean']:.3f} "
                f"red={record['red_excess']:.1f} "
                f"cyan={record['cyan_excess']:.1f}"
            ),
            fill="black",
            font=label_font,
        )


    sheet.save(
        output_path,
        quality=94,
        subsampling=0,
    )


# ============================================================
# SAVE ARTIFACT AUDIT
# ============================================================

def save_artifact_audit(
    case_id,
    file_id,
    artifact_records,
):

    case_dir = (
        ARTIFACT_AUDIT_ROOT
        / ACTIVE_BATCH_NAME
    )

    case_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    path = (
        case_dir
        /
        (
            f"{case_id}_"
            f"{file_id}.csv"
        )
    )


    columns = [
        "case_submitter_id",
        "class_label",
        "file_id",
        "slide_token",
        "candidate_checked_index",
        "x_level0",
        "y_level0",
        "approx_tissue_fraction",
        "final_tissue_fraction",
        "background_fraction",
        "white_fraction",
        "near_black_fraction",
        "gradient_mean",
        "red_excess",
        "cyan_excess",
        "saturation_mean",
        "artifact_reasons",
        "artifact_image_path",
        "artifact_qc_protocol",
    ]


    pd.DataFrame(
        artifact_records,
        columns=columns,
    ).to_csv(
        path,
        index=False,
    )


    return path


# ============================================================
# REPLACEMENT-AWARE WSI EXTRACTION
# ============================================================

def extract_wsi_tiles_with_artifact_qc(
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
    artifact_records = []


    file_id = str(
        candidate_row[
            "file_id"
        ]
    )

    slide_token = str(
        candidate_row[
            "slide_token"
        ]
    )


    artifact_image_dir = (
        ARTIFACT_IMAGE_ROOT
        / ACTIVE_BATCH_NAME
        / case_id
        / (
            f"{slide_token}_"
            f"{file_id}"
        )
    )


    if artifact_image_dir.exists():

        shutil.rmtree(
            artifact_image_dir,
            ignore_errors=True,
        )


    if SAVE_REJECTED_ARTIFACTS:

        artifact_image_dir.mkdir(
            parents=True,
            exist_ok=True,
        )


    try:

        slide = base.openslide.OpenSlide(
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

                "artifact_qc_protocol": (
                    ARTIFACT_PROTOCOL_NAME
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
                base.openslide.PROPERTY_NAME_OBJECTIVE_POWER
            )
        )

        mpp_x = (
            slide.properties.get(
                base.openslide.PROPERTY_NAME_MPP_X
            )
        )

        mpp_y = (
            slide.properties.get(
                base.openslide.PROPERTY_NAME_MPP_Y
            )
        )


        thumbnail = slide.get_thumbnail(
            (
                base.THUMBNAIL_MAX_SIZE,
                base.THUMBNAIL_MAX_SIZE,
            )
        ).convert(
            "RGB"
        )


        (
            tissue_mask,
            raw_otsu,
            saturation_threshold,
        ) = base.create_tissue_mask(
            thumbnail
        )


        base.save_mask_preview(
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


        ii = base.integral_image(
            tissue_mask
        )


        candidates = []


        for y in range(
            0,
            height
            - base.TILE_SIZE
            + 1,
            base.STRIDE,
        ):

            ty1 = int(
                np.floor(
                    y / scale_y
                )
            )

            ty2 = int(
                np.ceil(
                    (
                        y
                        +
                        base.TILE_SIZE
                    )
                    / scale_y
                )
            )


            for x in range(
                0,
                width
                - base.TILE_SIZE
                + 1,
                base.STRIDE,
            ):

                tx1 = int(
                    np.floor(
                        x / scale_x
                    )
                )

                tx2 = int(
                    np.ceil(
                        (
                            x
                            +
                            base.TILE_SIZE
                        )
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
                    base.rect_sum(
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
                    base.CANDIDATE_TISSUE_FRACTION
                ):

                    candidates.append(
                        (
                            x,
                            y,
                            approx_fraction,
                        )
                    )


        seed = base.deterministic_seed(
            case_id
            + "|"
            + file_id
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

        rejected_artifact = 0

        artifact_reason_counts = {
            "near_black": 0,
            "flat_extreme_red": 0,
            "flat_extreme_cyan": 0,
            "hyper_red_saturated": 0,
            "hyper_cyan_saturated": 0,
        }


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
                    base.TILE_SIZE,
                    base.TILE_SIZE,
                ),
            ).convert(
                "RGB"
            )


            (
                final_tissue_fraction,
                background_fraction,
                white_fraction,
            ) = base.evaluate_tile(
                tile,
                saturation_threshold,
            )


            if (
                final_tissue_fraction
                <
                base.FINAL_TISSUE_FRACTION
            ):

                rejected_low_tissue += 1
                continue


            if (
                background_fraction
                >
                base.MAX_BACKGROUND_FRACTION
            ):

                rejected_background += 1
                continue


            if (
                white_fraction
                >
                base.MAX_WHITE_FRACTION
            ):

                rejected_white += 1
                continue


            # =================================================
            # NEW ARTIFACT SANITY QC
            # =================================================

            artifact_metrics = (
                calculate_artifact_metrics(
                    tile
                )
            )


            (
                artifact_reject,
                artifact_reasons,
            ) = evaluate_artifact_qc(
                artifact_metrics
            )


            if artifact_reject:

                rejected_artifact += 1


                for reason in artifact_reasons:

                    artifact_reason_counts[
                        reason
                    ] += 1


                artifact_image_path = None


                if SAVE_REJECTED_ARTIFACTS:

                    artifact_image_path = (
                        artifact_image_dir
                        /
                        (
                            f"reject_"
                            f"{rejected_artifact:03d}"
                            f"_candidate_{checked:05d}"
                            f"_x{x}"
                            f"_y{y}.png"
                        )
                    )


                    tile.save(
                        artifact_image_path,
                        format="PNG",
                    )


                artifact_records.append({

                    "case_submitter_id": (
                        case_id
                    ),

                    "class_label": (
                        class_label
                    ),

                    "file_id": (
                        file_id
                    ),

                    "slide_token": (
                        slide_token
                    ),

                    "candidate_checked_index": (
                        checked
                    ),

                    "x_level0": int(
                        x
                    ),

                    "y_level0": int(
                        y
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

                    **artifact_metrics,

                    "artifact_reasons": (
                        ";".join(
                            artifact_reasons
                        )
                    ),

                    "artifact_image_path": (
                        str(
                            artifact_image_path
                        )
                        if artifact_image_path
                        else None
                    ),

                    "artifact_qc_protocol": (
                        ARTIFACT_PROTOCOL_NAME
                    ),
                })


                # IMPORTANT:
                #
                # Do NOT reduce patient tile count.
                # Continue searching for a replacement.
                continue


            # =================================================
            # ACCEPTED TILE
            # =================================================

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
                    file_id
                ),

                "selected_wsi_slide_token": (
                    slide_token
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
                    base.TILE_SIZE
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

                "artifact_qc_protocol": (
                    ARTIFACT_PROTOCOL_NAME
                ),

                "artifact_qc_status": (
                    "PASS"
                ),

                **artifact_metrics,
            })


            accepted += 1


            if (
                accepted
                >=
                base.TARGET_TILES
            ):

                break


        # =====================================================
        # AUDIT OUTPUT
        # =====================================================

        artifact_audit_path = (
            save_artifact_audit(
                case_id,
                file_id,
                artifact_records,
            )
        )


        artifact_contact_path = None


        if (
            SAVE_REJECTED_ARTIFACTS
            and
            artifact_records
        ):

            artifact_contact_path = (
                artifact_image_dir
                / "artifact_rejections_contact_sheet.jpg"
            )


            create_artifact_contact_sheet(
                artifact_records,
                artifact_contact_path,
            )


        print(
            f"    artifact rejected: "
            f"{rejected_artifact}"
        )


        if artifact_records:

            print(
                "    artifact reasons:  "
                +
                ", ".join(
                    (
                        f"{name}="
                        f"{count}"
                    )
                    for name, count
                    in artifact_reason_counts.items()
                    if count > 0
                )
            )


        qc = {

            "status": (
                "PASS"
                if accepted
                == base.TARGET_TILES
                else "FAIL"
            ),

            "reason": (
                None
                if accepted
                == base.TARGET_TILES
                else
                "insufficient_valid_nonartifact_tissue_tiles"
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

            "rejected_artifact": int(
                rejected_artifact
            ),

            "artifact_near_black": int(
                artifact_reason_counts[
                    "near_black"
                ]
            ),

            "artifact_flat_extreme_red": int(
                artifact_reason_counts[
                    "flat_extreme_red"
                ]
            ),

            "artifact_flat_extreme_cyan": int(
                artifact_reason_counts[
                    "flat_extreme_cyan"
                ]
            ),

            "artifact_hyper_red_saturated": int(
                artifact_reason_counts[
                    "hyper_red_saturated"
                ]
            ),

            "artifact_hyper_cyan_saturated": int(
                artifact_reason_counts[
                    "hyper_cyan_saturated"
                ]
            ),

            "valid_tiles_selected": int(
                accepted
            ),

            "artifact_qc_protocol": (
                ARTIFACT_PROTOCOL_NAME
            ),

            "artifact_audit_csv": str(
                artifact_audit_path
            ),

            "artifact_contact_sheet": (
                str(
                    artifact_contact_path
                )
                if artifact_contact_path
                else None
            ),
        }


        return (
            qc,
            tile_records,
        )


    except Exception as exc:

        try:

            save_artifact_audit(
                case_id,
                file_id,
                artifact_records,
            )

        except Exception:
            pass


        return (
            {
                "status": "FAIL",

                "reason": (
                    "tiling_exception:"
                    + repr(exc)
                ),

                "artifact_qc_protocol": (
                    ARTIFACT_PROTOCOL_NAME
                ),

                "rejected_artifact": int(
                    len(
                        artifact_records
                    )
                ),
            },
            [],
        )


    finally:

        slide.close()


# ============================================================
# GLOBAL PROTOCOL TERMINAL CASES
# ============================================================

def collect_protocol_terminal_cases():

    terminal = set()


    for path in (
        PROTOCOL_BATCH_METADATA_ROOT.glob(
            "*_patient_qc.csv"
        )
    ):

        try:

            df = pd.read_csv(
                path
            )

        except Exception:
            continue


        if (
            "case_submitter_id"
            not in df.columns
            or
            "status"
            not in df.columns
        ):

            continue


        cases = df.loc[
            df[
                "status"
            ].isin(
                [
                    "PASS",
                    "EXCLUDED_WSI_QC",
                ]
            ),
            "case_submitter_id",
        ]


        terminal.update(
            cases.astype(
                str
            )
        )


    return terminal


# ============================================================
# HISTORICAL DEVELOPMENT / VALIDATION CASES
# ============================================================

def collect_historical_batch_cases():

    cases = set()


    # Old V2_PROD / smoke / validation batch manifests.
    for path in (
        ORIGINAL_MANIFEST_DIR.glob(
            "*_patients.csv"
        )
    ):

        try:

            df = pd.read_csv(
                path
            )

        except Exception:
            continue


        if (
            "case_submitter_id"
            in df.columns
        ):

            cases.update(
                df[
                    "case_submitter_id"
                ]
                .astype(str)
            )


    # Pilot is stored under a different filename.
    if base.PILOT_FILE.exists():

        try:

            pilot = pd.read_csv(
                base.PILOT_FILE
            )


            if (
                "case_submitter_id"
                in pilot.columns
            ):

                cases.update(
                    pilot[
                        "case_submitter_id"
                    ]
                    .astype(str)
                )

        except Exception:
            pass


    return cases


# ============================================================
# DETERMINISTIC NEW-PROTOCOL SELECTION KEY
# ============================================================

def new_protocol_sort_key(
    case_id,
):

    return hashlib.sha256(
        (
            PROTOCOL_NAMESPACE
            + "|INDEPENDENT_BATCH|"
            + str(
                case_id
            )
        ).encode(
            "utf-8"
        )
    ).hexdigest()


# ============================================================
# PATCHED BATCH CREATION
# ============================================================

def create_or_load_batch_candidate_v1(
    cohort,
    pilot_cases,
    batch_path,
    limit,
    process_all,
    include_pilot,
):

    global EXPLICIT_CASES
    global FORCE_REPROCESS


    # ========================================================
    # EXISTING FROZEN BATCH
    # ========================================================

    if batch_path.exists():

        print(
            "Using existing frozen batch manifest:"
        )

        print(
            f"  {batch_path}"
        )


        batch = pd.read_csv(
            batch_path
        )


        if EXPLICIT_CASES:

            requested = set(
                map(
                    str,
                    EXPLICIT_CASES,
                )
            )

            existing = set(
                batch[
                    "case_submitter_id"
                ].astype(str)
            )


            if requested != existing:

                raise RuntimeError(
                    "Existing frozen batch does not match "
                    "--cases. Use another --batch-name."
                )


        return batch.reset_index(
            drop=True
        )


    # ========================================================
    # EXPLICIT TEST CASES
    # ========================================================

    if EXPLICIT_CASES:

        requested = list(
            dict.fromkeys(
                map(
                    str,
                    EXPLICIT_CASES,
                )
            )
        )


        available = set(
            cohort[
                "case_submitter_id"
            ].astype(str)
        )


        missing = [
            case_id
            for case_id in requested
            if case_id
            not in available
        ]


        if missing:

            raise RuntimeError(
                "Explicit cases not found in cohort: "
                f"{missing}"
            )


        eligible = cohort[
            cohort[
                "case_submitter_id"
            ]
            .astype(str)
            .isin(
                requested
            )
        ].copy()


        order_map = {
            case_id: index
            for index, case_id
            in enumerate(
                requested
            )
        }


        eligible[
            "_explicit_order"
        ] = (
            eligible[
                "case_submitter_id"
            ]
            .astype(str)
            .map(
                order_map
            )
        )


        batch = (
            eligible
            .sort_values(
                "_explicit_order"
            )
            .drop(
                columns=[
                    "_explicit_order"
                ]
            )
            .reset_index(
                drop=True
            )
        )


    # ========================================================
    # NORMAL BATCH SELECTION
    # ========================================================

    else:

        eligible = cohort.copy()


        if not include_pilot:

            eligible = eligible[
                ~eligible[
                    "case_submitter_id"
                ].isin(
                    pilot_cases
                )
            ].copy()


        # Cases already terminal under THIS artifact-QC
        # protocol must not be processed again.
        if not FORCE_REPROCESS:

            protocol_terminal = (
                collect_protocol_terminal_cases()
            )


            if protocol_terminal:

                eligible = eligible[
                    ~eligible[
                        "case_submitter_id"
                    ]
                    .astype(str)
                    .isin(
                        protocol_terminal
                    )
                ].copy()


                print(
                    "Excluded already-terminal "
                    f"{PROTOCOL_NAMESPACE} cases: "
                    f"{len(protocol_terminal)}"
                )


        # For a NEW limited validation batch, exclude all
        # patients used by prior pilot/smoke/validation work.
        #
        # For --all we do NOT apply this development exclusion:
        # final cohort processing must eventually cover the
        # complete eligible cohort under the final protocol.
        if not process_all:

            historical_cases = (
                collect_historical_batch_cases()
            )


            before = len(
                eligible
            )


            eligible = eligible[
                ~eligible[
                    "case_submitter_id"
                ]
                .astype(str)
                .isin(
                    historical_cases
                )
            ].copy()


            print(
                "Excluded prior pilot/smoke/"
                "validation cases: "
                f"{before - len(eligible)}"
            )


        if process_all:

            batch = (
                eligible
                .sort_values(
                    "case_submitter_id"
                )
                .reset_index(
                    drop=True
                )
            )


        else:

            eligible[
                "_selection_key"
            ] = (
                eligible[
                    "case_submitter_id"
                ]
                .apply(
                    new_protocol_sort_key
                )
            )


            luad_target = (
                limit // 2
            )

            lusc_target = (
                limit
                -
                luad_target
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
                .drop(
                    columns=[
                        "_selection_key"
                    ],
                    errors="ignore",
                )
                .reset_index(
                    drop=True
                )
            )


    batch.to_csv(
        batch_path,
        index=False,
    )


    metadata_path = (
        batch_path.with_suffix(
            ".json"
        )
    )


    metadata = {

        "protocol_namespace": (
            PROTOCOL_NAMESPACE
        ),

        "protocol_name": (
            PROTOCOL_NAME
        ),

        "artifact_protocol": (
            ARTIFACT_PROTOCOL_NAME
        ),

        "artifact_rules": (
            ARTIFACT_RULES
        ),

        "batch_name": (
            ACTIVE_BATCH_NAME
        ),

        "patient_count": int(
            len(
                batch
            )
        ),

        "luad": int(
            (
                batch[
                    "class_label"
                ]
                == "LUAD"
            ).sum()
        ),

        "lusc": int(
            (
                batch[
                    "class_label"
                ]
                == "LUSC"
            ).sum()
        ),

        "explicit_cases": (
            EXPLICIT_CASES
        ),

        "process_all": bool(
            process_all
        ),

        "important_note": (
            "Artifact-QC thresholds are frozen "
            "for independent validation."
        ),
    }


    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


    print(
        "Created frozen batch manifest:"
    )

    print(
        f"  {batch_path}"
    )


    return batch


# ============================================================
# FORCE REPROCESS FOR EXPLICIT TEST CASES
# ============================================================

def prepare_force_reprocess():

    if (
        not FORCE_REPROCESS
        or
        not EXPLICIT_CASES
    ):

        return


    patient_result_path = (
        PROTOCOL_BATCH_METADATA_ROOT
        /
        (
            f"{ACTIVE_BATCH_NAME}"
            "_patient_qc.csv"
        )
    )


    if not patient_result_path.exists():

        return


    results = pd.read_csv(
        patient_result_path
    )


    requested = set(
        map(
            str,
            EXPLICIT_CASES,
        )
    )


    before = len(
        results
    )


    results = results[
        ~results[
            "case_submitter_id"
        ]
        .astype(str)
        .isin(
            requested
        )
    ].copy()


    if len(results) == 0:

        patient_result_path.unlink()

    else:

        results.to_csv(
            patient_result_path,
            index=False,
        )


    print(
        "Force reprocess: removed "
        f"{before - len(results)} terminal "
        "records from current batch."
    )


# ============================================================
# CUSTOM ARGUMENTS
# ============================================================

def parse_wrapper_arguments():

    parser = argparse.ArgumentParser(
        add_help=False
    )


    parser.add_argument(
        "--batch-name",
        default=(
            "artifact_qc_candidate_v1_test"
        ),
    )


    parser.add_argument(
        "--cases",
        nargs="+",
        default=None,
        help=(
            "Process exactly these TCGA case IDs. "
            "Useful for replacement-search tests."
        ),
    )


    parser.add_argument(
        "--force-reprocess",
        action="store_true",
        help=(
            "Allow explicit cases to be rerun "
            "inside an existing protocol batch."
        ),
    )


    parser.add_argument(
        "--save-rejected-artifacts",
        action="store_true",
        help=(
            "Save every artifact-rejected candidate "
            "tile and a review contact sheet."
        ),
    )


    args, remaining = (
        parser.parse_known_args()
    )


    # Rebuild argv for script 11's existing parser.
    #
    # --cases / --force-reprocess /
    # --save-rejected-artifacts are wrapper-only.
    #
    # --batch-name must be passed back to base.main().
    sys.argv = [
        sys.argv[0],
        "--batch-name",
        args.batch_name,
        *remaining,
    ]


    return args


# ============================================================
# MAIN
# ============================================================

def main():

    global ACTIVE_BATCH_NAME
    global EXPLICIT_CASES
    global FORCE_REPROCESS
    global SAVE_REJECTED_ARTIFACTS


    wrapper_args = (
        parse_wrapper_arguments()
    )


    ACTIVE_BATCH_NAME = (
        wrapper_args.batch_name
    )

    EXPLICIT_CASES = (
        wrapper_args.cases
    )

    FORCE_REPROCESS = (
        wrapper_args.force_reprocess
    )

    SAVE_REJECTED_ARTIFACTS = (
        wrapper_args.save_rejected_artifacts
    )


    write_or_verify_frozen_rules()


    # Replace only the two behaviours that differ from
    # the already tested V2_PROD implementation.
    base.extract_wsi_tiles = (
        extract_wsi_tiles_with_artifact_qc
    )

    base.create_or_load_batch = (
        create_or_load_batch_candidate_v1
    )


    prepare_force_reprocess()


    print()
    print("=" * 88)
    print(
        " V2_PROD + FROZEN ARTIFACT QC CANDIDATE V1"
    )
    print("=" * 88)

    print(
        f"Protocol:                 "
        f"{PROTOCOL_NAME}"
    )

    print(
        f"Batch:                    "
        f"{ACTIVE_BATCH_NAME}"
    )

    print(
        f"Explicit cases:           "
        f"{EXPLICIT_CASES}"
    )

    print(
        f"Save rejected artifacts:  "
        f"{SAVE_REJECTED_ARTIFACTS}"
    )

    print(
        f"Force reprocess:          "
        f"{FORCE_REPROCESS}"
    )

    print()
    print(
        "Frozen artifact rules:"
    )

    for key, value in (
        ARTIFACT_RULES.items()
    ):

        print(
            f"  {key:<34} {value}"
        )

    print()
    print(
        "NOTE: thresholds are frozen for "
        "independent validation."
    )

    print("=" * 88)


    base.main()


if __name__ == "__main__":
    main()