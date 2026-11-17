from pathlib import Path
import argparse
import importlib.util
import json
import math
import shutil

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


# ============================================================
# ROOT
# ============================================================

THIS_FILE = Path(__file__).resolve()
ROOT = THIS_FILE.parents[1]


# ============================================================
# LOAD SCRIPT 17
# ============================================================

PIPELINE17_PATH = (
    THIS_FILE.parent
    / "17_process_cohort_v2prod_artifact_qc.py"
)

if not PIPELINE17_PATH.exists():
    raise FileNotFoundError(
        f"Script 17 not found: {PIPELINE17_PATH}"
    )


spec = importlib.util.spec_from_file_location(
    "artifact_pipeline_v1",
    PIPELINE17_PATH,
)

pipeline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline)

base = pipeline.base


# ============================================================
# AUDIT IDENTITY
# ============================================================

AUDIT_NAME = "final_tissue_threshold_audit_v1"

AUDIT_PROTOCOL = (
    "V2_PROD_TISSUE_THRESHOLD_SWEEP"
    "+FROZEN_ARTIFACT_QC_CANDIDATE_V1"
)


# ============================================================
# THRESHOLDS
# ============================================================
#
# ONLY FINAL_TISSUE_FRACTION changes.
#
# Everything else stays frozen.
# ============================================================

TISSUE_THRESHOLDS = [
    0.90,
    0.85,
    0.80,
    0.75,
    0.70,
]

MIN_TISSUE_THRESHOLD = min(TISSUE_THRESHOLDS)


# ============================================================
# DEFAULT CASES
# ============================================================

DEFAULT_FAILURE_CASES = [
    "TCGA-05-4244",
    "TCGA-05-4396",
    "TCGA-05-4397",
    "TCGA-05-4434",
    "TCGA-66-2753",
]


DEFAULT_CONTROL_CASES = [
    "TCGA-55-6986",
    "TCGA-39-5040",
    "TCGA-64-5774",
    "TCGA-66-2789",
    "TCGA-21-5782",
]


# ============================================================
# INPUTS
# ============================================================

COHORT_FILE = (
    ROOT
    / "data"
    / "manifests"
    / "paired_cohort_stage2.csv"
)

WSI_CANDIDATES_FILE = (
    ROOT
    / "data"
    / "manifests"
    / "wsi_primary_diagnostic_candidates.csv"
)


# ============================================================
# OUTPUTS
# ============================================================

OUTPUT_ROOT = (
    ROOT
    / "results"
    / AUDIT_NAME
)

FIGURE_ROOT = (
    ROOT
    / "results"
    / "figures"
    / AUDIT_NAME
)

METADATA_ROOT = (
    ROOT
    / "data"
    / "metadata"
    / AUDIT_NAME
)

TEMP_GDC_ROOT = (
    ROOT
    / "data"
    / "temp_gdc_tissue_threshold_audit"
)


for directory in [
    OUTPUT_ROOT,
    FIGURE_ROOT,
    METADATA_ROOT,
    TEMP_GDC_ROOT,
]:
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# VISUAL CONFIG
# ============================================================

CONTACT_COLUMNS = 8
CONTACT_ROWS = 8

HEADER_HEIGHT = 64

CONTACT_TILE_SIZE = base.TILE_SIZE

CASE_OVERVIEW_PREVIEW_WIDTH = 480

GLOBAL_OVERVIEW_WIDTH = 1200

JPEG_QUALITY = 94


# ============================================================
# FONT
# ============================================================

def load_font(size):

    candidates = [
        Path(r"C:\Windows\Fonts\segoeui.ttf"),
        Path(r"C:\Windows\Fonts\arial.ttf"),
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


# ============================================================
# SMALL HELPERS
# ============================================================

def threshold_slug(threshold):

    return str(
        int(
            round(
                threshold * 100
            )
        )
    )


def close_image_safely(image):

    try:
        image.close()

    except Exception:
        pass


# ============================================================
# FROZEN AUDIT CONFIG
# ============================================================

FROZEN_CONFIG_PATH = (
    METADATA_ROOT
    / "frozen_audit_configuration.json"
)


def write_or_verify_frozen_config(
    failure_cases,
    control_cases,
):

    payload = {
        "audit_name": AUDIT_NAME,
        "audit_protocol": AUDIT_PROTOCOL,

        "tissue_thresholds": TISSUE_THRESHOLDS,

        "candidate_tissue_fraction": (
            float(
                base.CANDIDATE_TISSUE_FRACTION
            )
        ),

        "max_background_fraction": (
            float(
                base.MAX_BACKGROUND_FRACTION
            )
        ),

        "max_white_fraction": (
            float(
                base.MAX_WHITE_FRACTION
            )
        ),

        "artifact_qc_protocol": (
            pipeline.ARTIFACT_PROTOCOL_NAME
        ),

        "artifact_rules": (
            pipeline.ARTIFACT_RULES
        ),

        "failure_cases": list(
            failure_cases
        ),

        "control_cases": list(
            control_cases
        ),

        "important_note": (
            "Only FINAL_TISSUE_FRACTION varies. "
            "Candidate tissue detection, background QC, "
            "white-pixel QC and artifact-QC remain frozen."
        ),
    }


    if FROZEN_CONFIG_PATH.exists():

        existing = json.loads(
            FROZEN_CONFIG_PATH.read_text(
                encoding="utf-8"
            )
        )

        if existing != payload:

            raise RuntimeError(
                "Existing frozen audit configuration differs "
                "from the current implementation. "
                "Do not overwrite it silently. "
                "Use a new AUDIT_NAME if the experiment changes."
            )

        return


    FROZEN_CONFIG_PATH.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


# ============================================================
# RESOLVE WSI
# ============================================================

def resolve_wsi_candidate(
    wsi_candidates,
    case_id,
):

    rows = wsi_candidates[
        wsi_candidates[
            "case_submitter_id"
        ].astype(str)
        ==
        str(case_id)
    ].copy()


    if rows.empty:

        raise RuntimeError(
            f"No diagnostic WSI candidates found "
            f"for {case_id}"
        )


    rows = rows.sort_values(
        [
            "dx_index",
            "associated_barcode",
            "file_id",
        ]
    )


    return rows.iloc[0]


# ============================================================
# GENERATE THUMBNAIL-LEVEL CANDIDATES
# ============================================================

def generate_candidates(
    slide,
    tissue_mask,
):

    width, height = slide.dimensions

    thumb_h, thumb_w = (
        tissue_mask.shape
    )


    scale_x = (
        width
        /
        thumb_w
    )

    scale_y = (
        height
        /
        thumb_h
    )


    ii = base.integral_image(
        tissue_mask
    )


    candidates = []


    for y in range(
        0,
        height - base.TILE_SIZE + 1,
        base.STRIDE,
    ):

        ty1 = int(
            np.floor(
                y
                /
                scale_y
            )
        )

        ty2 = int(
            np.ceil(
                (
                    y
                    +
                    base.TILE_SIZE
                )
                /
                scale_y
            )
        )


        for x in range(
            0,
            width - base.TILE_SIZE + 1,
            base.STRIDE,
        ):

            tx1 = int(
                np.floor(
                    x
                    /
                    scale_x
                )
            )

            tx2 = int(
                np.ceil(
                    (
                        x
                        +
                        base.TILE_SIZE
                    )
                    /
                    scale_x
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
                /
                area
            )


            if (
                approx_fraction
                >=
                base.CANDIDATE_TISSUE_FRACTION
            ):

                candidates.append(
                    (
                        int(x),
                        int(y),
                        float(
                            approx_fraction
                        ),
                    )
                )


    return candidates


# ============================================================
# CREATE CONTACT SHEET
# ============================================================

def create_contact_sheet(
    selected_records,
    image_cache,
    output_path,
    case_id,
    threshold,
    checked_to_64,
):

    width = (
        CONTACT_COLUMNS
        *
        CONTACT_TILE_SIZE
    )

    height = (
        HEADER_HEIGHT
        +
        CONTACT_ROWS
        *
        CONTACT_TILE_SIZE
    )


    sheet = Image.new(
        "RGB",
        (
            width,
            height,
        ),
        "white",
    )


    draw = ImageDraw.Draw(
        sheet
    )


    title_font = load_font(
        23
    )

    small_font = load_font(
        15
    )


    draw.text(
        (
            10,
            7,
        ),
        (
            f"{case_id} — "
            f"FINAL_TISSUE_FRACTION >= "
            f"{threshold:.2f}"
        ),
        fill="black",
        font=title_font,
    )


    if checked_to_64 is None:

        checked_text = (
            "NOT REACHED"
        )

    else:

        checked_text = str(
            checked_to_64
        )


    draw.text(
        (
            10,
            38,
        ),
        (
            f"selected="
            f"{len(selected_records)}/64"
            f" | checked_to_64="
            f"{checked_text}"
        ),
        fill="black",
        font=small_font,
    )


    for index in range(
        base.TARGET_TILES
    ):

        column = (
            index
            %
            CONTACT_COLUMNS
        )

        row = (
            index
            //
            CONTACT_COLUMNS
        )


        paste_x = (
            column
            *
            CONTACT_TILE_SIZE
        )

        paste_y = (
            HEADER_HEIGHT
            +
            row
            *
            CONTACT_TILE_SIZE
        )


        if (
            index
            <
            len(
                selected_records
            )
        ):

            record = (
                selected_records[
                    index
                ]
            )


            key = (
                int(
                    record[
                        "x_level0"
                    ]
                ),
                int(
                    record[
                        "y_level0"
                    ]
                ),
            )


            image = (
                image_cache[
                    key
                ]
            )


            sheet.paste(
                image,
                (
                    paste_x,
                    paste_y,
                ),
            )


        else:

            blank = Image.new(
                "RGB",
                (
                    CONTACT_TILE_SIZE,
                    CONTACT_TILE_SIZE,
                ),
                (
                    225,
                    225,
                    225,
                ),
            )


            sheet.paste(
                blank,
                (
                    paste_x,
                    paste_y,
                ),
            )


            blank.close()


    sheet.save(
        output_path,
        quality=JPEG_QUALITY,
        subsampling=0,
    )


    sheet.close()


# ============================================================
# CREATE CASE OVERVIEW
# ============================================================

def create_case_overview(
    contact_paths,
    output_path,
    case_id,
):

    previews = []


    for threshold in TISSUE_THRESHOLDS:

        path = contact_paths[
            threshold
        ]


        with Image.open(path) as source:

            source = source.convert(
                "RGB"
            )


            ratio = (
                CASE_OVERVIEW_PREVIEW_WIDTH
                /
                source.width
            )


            preview_height = int(
                round(
                    source.height
                    *
                    ratio
                )
            )


            preview = source.resize(
                (
                    CASE_OVERVIEW_PREVIEW_WIDTH,
                    preview_height,
                ),
                Image.Resampling.LANCZOS,
            )


            previews.append(
                preview.copy()
            )


            preview.close()


    gap = 12
    top_header = 55


    width = (
        len(previews)
        *
        CASE_OVERVIEW_PREVIEW_WIDTH
        +
        (
            len(previews)
            - 1
        )
        *
        gap
    )


    max_preview_height = max(
        image.height
        for image in previews
    )


    height = (
        top_header
        +
        max_preview_height
    )


    overview = Image.new(
        "RGB",
        (
            width,
            height,
        ),
        "white",
    )


    draw = ImageDraw.Draw(
        overview
    )


    font = load_font(
        25
    )


    draw.text(
        (
            10,
            10,
        ),
        (
            f"{case_id} — "
            "FINAL_TISSUE_FRACTION sweep"
        ),
        fill="black",
        font=font,
    )


    current_x = 0


    for image in previews:

        overview.paste(
            image,
            (
                current_x,
                top_header,
            ),
        )


        current_x += (
            CASE_OVERVIEW_PREVIEW_WIDTH
            +
            gap
        )


    overview.save(
        output_path,
        quality=JPEG_QUALITY,
        subsampling=0,
    )


    overview.close()


    for image in previews:
        close_image_safely(
            image
        )


# ============================================================
# AUDIT ONE CASE
# ============================================================

def audit_case(
    case_id,
    group_name,
    class_label,
    candidate,
    keep_wsi,
):

    print()
    print("=" * 88)

    print(
        f" {case_id} "
        f"({class_label}) — "
        f"{group_name}"
    )

    print("=" * 88)


    file_id = str(
        candidate[
            "file_id"
        ]
    )

    slide_token = str(
        candidate[
            "slide_token"
        ]
    )


    print(
        f"WSI UUID:       "
        f"{file_id}"
    )

    print(
        f"slide token:    "
        f"{slide_token}"
    )

    print(
        f"WSI size:       "
        f"{candidate['file_size_bytes'] / 1024**3:.3f} GB"
    )


    patient_temp_dir = (
        TEMP_GDC_ROOT
        /
        case_id
    )


    patient_temp_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    wsi_path = base.download_gdc_file(
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


    case_output_dir = (
        OUTPUT_ROOT
        /
        case_id
    )

    case_figure_dir = (
        FIGURE_ROOT
        /
        case_id
    )


    case_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    case_figure_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    slide = None


    try:

        slide = base.openslide.OpenSlide(
            str(
                wsi_path
            )
        )


        width, height = (
            slide.dimensions
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


        mask_path = (
            case_figure_dir
            /
            "thumbnail_tissue_mask.jpg"
        )


        base.save_mask_preview(
            thumbnail,
            tissue_mask,
            mask_path,
        )


        thumbnail.close()


        candidates = generate_candidates(
            slide,
            tissue_mask,
        )


        seed = base.deterministic_seed(
            case_id
            +
            "|"
            +
            file_id
        )


        rng = np.random.default_rng(
            seed
        )


        order = np.arange(
            len(
                candidates
            )
        )


        rng.shuffle(
            order
        )


        selected = {
            threshold: []
            for threshold
            in TISSUE_THRESHOLDS
        }


        checked_to_64 = {
            threshold: None
            for threshold
            in TISSUE_THRESHOLDS
        }


        pass_count_seen = {
            threshold: 0
            for threshold
            in TISSUE_THRESHOLDS
        }


        image_cache = {}


        checked = 0

        background_fail_seen = 0
        white_fail_seen = 0

        artifact_rejected_seen = 0


        artifact_reason_counts = {
            "near_black": 0,
            "flat_extreme_red": 0,
            "flat_extreme_cyan": 0,
            "hyper_red_saturated": 0,
            "hyper_cyan_saturated": 0,
        }


        print(
            f"thumbnail candidates: "
            f"{len(candidates)}"
        )

        print(
            "Evaluating the same deterministic "
            "candidate order for all thresholds..."
        )


        for candidate_index in order:

            x, y, approx_fraction = (
                candidates[
                    candidate_index
                ]
            )


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


            background_fail = (
                background_fraction
                >
                base.MAX_BACKGROUND_FRACTION
            )


            white_fail = (
                white_fraction
                >
                base.MAX_WHITE_FRACTION
            )


            if background_fail:
                background_fail_seen += 1


            if white_fail:
                white_fail_seen += 1


            artifact_reject = False
            artifact_reasons = []


            artifact_metrics = {
                "near_black_fraction": None,
                "gradient_mean": None,
                "red_excess": None,
                "cyan_excess": None,
                "saturation_mean": None,
            }


            # Only calculate artifact-QC for tiles that
            # could pass at least one threshold.
            if (
                final_tissue_fraction
                >=
                MIN_TISSUE_THRESHOLD
                and
                not background_fail
                and
                not white_fail
            ):

                artifact_metrics = (
                    pipeline.calculate_artifact_metrics(
                        tile
                    )
                )


                (
                    artifact_reject,
                    artifact_reasons,
                ) = (
                    pipeline.evaluate_artifact_qc(
                        artifact_metrics
                    )
                )


                if artifact_reject:

                    artifact_rejected_seen += 1


                    for reason in artifact_reasons:

                        if (
                            reason
                            in
                            artifact_reason_counts
                        ):

                            artifact_reason_counts[
                                reason
                            ] += 1


            selected_this_candidate = False


            for threshold in TISSUE_THRESHOLDS:

                passes = (
                    final_tissue_fraction
                    >=
                    threshold
                    and
                    not background_fail
                    and
                    not white_fail
                    and
                    not artifact_reject
                )


                if not passes:
                    continue


                pass_count_seen[
                    threshold
                ] += 1


                if (
                    len(
                        selected[
                            threshold
                        ]
                    )
                    >=
                    base.TARGET_TILES
                ):

                    continue


                selected_index = (
                    len(
                        selected[
                            threshold
                        ]
                    )
                    +
                    1
                )


                record = {
                    "case_submitter_id": (
                        case_id
                    ),

                    "class_label": (
                        class_label
                    ),

                    "group": (
                        group_name
                    ),

                    "threshold": float(
                        threshold
                    ),

                    "selected_index": int(
                        selected_index
                    ),

                    "candidate_checked_index": int(
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

                    "near_black_fraction": (
                        artifact_metrics[
                            "near_black_fraction"
                        ]
                    ),

                    "gradient_mean": (
                        artifact_metrics[
                            "gradient_mean"
                        ]
                    ),

                    "red_excess": (
                        artifact_metrics[
                            "red_excess"
                        ]
                    ),

                    "cyan_excess": (
                        artifact_metrics[
                            "cyan_excess"
                        ]
                    ),

                    "saturation_mean": (
                        artifact_metrics[
                            "saturation_mean"
                        ]
                    ),
                }


                selected[
                    threshold
                ].append(
                    record
                )


                selected_this_candidate = True


                if (
                    len(
                        selected[
                            threshold
                        ]
                    )
                    ==
                    base.TARGET_TILES
                    and
                    checked_to_64[
                        threshold
                    ]
                    is None
                ):

                    checked_to_64[
                        threshold
                    ] = int(
                        checked
                    )


            if selected_this_candidate:

                key = (
                    int(x),
                    int(y),
                )


                if key not in image_cache:

                    image_cache[
                        key
                    ] = tile.copy()


            tile.close()


            # If even the strictest threshold already has
            # 64 tiles, every looser threshold has 64 as well.
            # No need to inspect the rest of the WSI.
            all_complete = all(
                len(
                    selected[
                        threshold
                    ]
                )
                >=
                base.TARGET_TILES
                for threshold
                in TISSUE_THRESHOLDS
            )


            if all_complete:
                break


            if (
                checked
                %
                2000
                ==
                0
            ):

                count_text = " | ".join(
                    [
                        (
                            f"{threshold:.2f}:"
                            f"{len(selected[threshold])}"
                        )
                        for threshold
                        in TISSUE_THRESHOLDS
                    ]
                )


                print(
                    f"  checked "
                    f"{checked}/"
                    f"{len(candidates)}"
                    f" | "
                    f"{count_text}"
                )


        scan_exhausted = (
            checked
            >=
            len(
                candidates
            )
        )


        print()

        print(
            f"candidate scan exhausted: "
            f"{scan_exhausted}"
        )

        print(
            f"candidates checked:       "
            f"{checked}/"
            f"{len(candidates)}"
        )


        # ====================================================
        # SAVE PER-THRESHOLD RESULTS
        # ====================================================

        summary_rows = []
        contact_paths = {}


        for threshold in TISSUE_THRESHOLDS:

            records = (
                selected[
                    threshold
                ]
            )


            slug = threshold_slug(
                threshold
            )


            manifest_path = (
                case_output_dir
                /
                (
                    f"selected_threshold_"
                    f"{slug}.csv"
                )
            )


            pd.DataFrame(
                records
            ).to_csv(
                manifest_path,
                index=False,
            )


            contact_path = (
                case_figure_dir
                /
                (
                    f"threshold_"
                    f"{slug}_contact.jpg"
                )
            )


            create_contact_sheet(
                selected_records=records,

                image_cache=image_cache,

                output_path=contact_path,

                case_id=case_id,

                threshold=threshold,

                checked_to_64=(
                    checked_to_64[
                        threshold
                    ]
                ),
            )


            contact_paths[
                threshold
            ] = contact_path


            reached_64 = (
                len(records)
                >=
                base.TARGET_TILES
            )


            summary_rows.append(
                {
                    "case_submitter_id": (
                        case_id
                    ),

                    "class_label": (
                        class_label
                    ),

                    "group": (
                        group_name
                    ),

                    "file_id": (
                        file_id
                    ),

                    "slide_token": (
                        slide_token
                    ),

                    "threshold": float(
                        threshold
                    ),

                    "selected_tiles": int(
                        len(records)
                    ),

                    "reached_64": bool(
                        reached_64
                    ),

                    "checked_to_64": (
                        checked_to_64[
                            threshold
                        ]
                    ),

                    "pass_count_seen": int(
                        pass_count_seen[
                            threshold
                        ]
                    ),

                    "candidates_total": int(
                        len(candidates)
                    ),

                    "candidates_checked": int(
                        checked
                    ),

                    "scan_exhausted": bool(
                        scan_exhausted
                    ),

                    "background_fail_seen": int(
                        background_fail_seen
                    ),

                    "white_fail_seen": int(
                        white_fail_seen
                    ),

                    "artifact_rejected_seen": int(
                        artifact_rejected_seen
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
                }
            )


            if reached_64:

                result_text = (
                    "PASS"
                )

            else:

                result_text = (
                    "FAIL"
                )


            if (
                checked_to_64[
                    threshold
                ]
                is None
            ):

                checked_text = "-"

            else:

                checked_text = str(
                    checked_to_64[
                        threshold
                    ]
                )


            print(
                f"  threshold "
                f"{threshold:.2f}: "
                f"{len(records):2d}/64 "
                f"{result_text}"
                f" | checked_to_64="
                f"{checked_text}"
            )


        # ====================================================
        # CASE OVERVIEW
        # ====================================================

        overview_path = (
            case_figure_dir
            /
            "threshold_comparison_overview.jpg"
        )


        create_case_overview(
            contact_paths=contact_paths,
            output_path=overview_path,
            case_id=case_id,
        )


        # ====================================================
        # CASE SUMMARY
        # ====================================================

        case_summary = pd.DataFrame(
            summary_rows
        )


        case_summary.to_csv(
            (
                case_output_dir
                /
                "threshold_summary.csv"
            ),
            index=False,
        )


        case_metadata = {
            "case_submitter_id": (
                case_id
            ),

            "class_label": (
                class_label
            ),

            "group": (
                group_name
            ),

            "wsi_file_id": (
                file_id
            ),

            "slide_token": (
                slide_token
            ),

            "width": int(
                width
            ),

            "height": int(
                height
            ),

            "candidate_tiles": int(
                len(candidates)
            ),

            "candidates_checked": int(
                checked
            ),

            "scan_exhausted": bool(
                scan_exhausted
            ),

            "deterministic_seed": int(
                seed
            ),

            "artifact_rejected_seen": int(
                artifact_rejected_seen
            ),

            "artifact_reason_counts": (
                artifact_reason_counts
            ),

            "mask_preview": str(
                mask_path
            ),

            "threshold_overview": str(
                overview_path
            ),
        }


        metadata_path = (
            case_output_dir
            /
            "case_metadata.json"
        )


        metadata_path.write_text(
            json.dumps(
                case_metadata,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )


        # ====================================================
        # CLEAN IMAGE CACHE
        # ====================================================

        for image in image_cache.values():
            close_image_safely(
                image
            )


        return summary_rows


    finally:

        if slide is not None:
            slide.close()


        if not keep_wsi:

            base.remove_uuid_download(
                patient_temp_dir,
                file_id,
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


# ============================================================
# CREATE GLOBAL OVERVIEW
# ============================================================

def create_global_overview(
    case_ids,
):

    source_paths = []


    for case_id in case_ids:

        path = (
            FIGURE_ROOT
            /
            case_id
            /
            "threshold_comparison_overview.jpg"
        )


        if path.exists():

            source_paths.append(
                (
                    case_id,
                    path,
                )
            )


    if not source_paths:
        return None


    preview_rows = []


    for case_id, path in source_paths:

        with Image.open(path) as source:

            source = source.convert(
                "RGB"
            )


            ratio = (
                GLOBAL_OVERVIEW_WIDTH
                /
                source.width
            )


            resized_height = int(
                round(
                    source.height
                    *
                    ratio
                )
            )


            resized = source.resize(
                (
                    GLOBAL_OVERVIEW_WIDTH,
                    resized_height,
                ),
                Image.Resampling.LANCZOS,
            )


            preview_rows.append(
                (
                    case_id,
                    resized.copy(),
                )
            )


            resized.close()


    row_header_height = 48
    gap = 12


    total_height = 0


    for case_id, image in preview_rows:

        total_height += (
            row_header_height
            +
            image.height
        )


    if len(preview_rows) > 1:

        total_height += (
            gap
            *
            (
                len(preview_rows)
                - 1
            )
        )


    canvas = Image.new(
        "RGB",
        (
            GLOBAL_OVERVIEW_WIDTH,
            total_height,
        ),
        "white",
    )


    draw = ImageDraw.Draw(
        canvas
    )


    font = load_font(
        22
    )


    current_y = 0


    for case_id, image in preview_rows:

        draw.text(
            (
                10,
                current_y + 8,
            ),
            case_id,
            fill="black",
            font=font,
        )


        current_y += (
            row_header_height
        )


        canvas.paste(
            image,
            (
                0,
                current_y,
            ),
        )


        current_y += (
            image.height
        )


        current_y += (
            gap
        )


    output_path = (
        FIGURE_ROOT
        /
        "all_cases_threshold_overview.jpg"
    )


    canvas.save(
        output_path,
        quality=90,
        subsampling=0,
    )


    canvas.close()


    for case_id, image in preview_rows:
        close_image_safely(
            image
        )


    return output_path


# ============================================================
# BUILD PIVOT TABLE
# ============================================================

def build_selected_tiles_pivot(
    summary,
):

    pivot = summary.pivot_table(
        index=[
            "case_submitter_id",
            "class_label",
            "group",
        ],

        columns="threshold",

        values="selected_tiles",

        aggfunc="first",
    ).reset_index()


    rename_map = {}


    for column in pivot.columns:

        if isinstance(
            column,
            float,
        ):

            rename_map[
                column
            ] = (
                f"selected_"
                f"{column:.2f}"
            )


    pivot = pivot.rename(
        columns=rename_map
    )


    return pivot


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Audit FINAL_TISSUE_FRACTION "
            "0.90/0.85/0.80/0.75/0.70 while keeping "
            "all other V2 and artifact-QC rules frozen."
        )
    )


    parser.add_argument(
        "--failure-cases",
        nargs="+",
        default=DEFAULT_FAILURE_CASES,
    )


    parser.add_argument(
        "--control-cases",
        nargs="+",
        default=DEFAULT_CONTROL_CASES,
    )


    parser.add_argument(
        "--keep-wsi",
        action="store_true",
    )


    args = parser.parse_args()


    failure_cases = list(
        dict.fromkeys(
            [
                str(case_id)
                for case_id
                in args.failure_cases
            ]
        )
    )


    control_cases = list(
        dict.fromkeys(
            [
                str(case_id)
                for case_id
                in args.control_cases
            ]
        )
    )


    overlap = (
        set(
            failure_cases
        )
        &
        set(
            control_cases
        )
    )


    if overlap:

        raise RuntimeError(
            "Failure/control overlap: "
            f"{sorted(overlap)}"
        )


    write_or_verify_frozen_config(
        failure_cases=(
            failure_cases
        ),

        control_cases=(
            control_cases
        ),
    )


    if not COHORT_FILE.exists():

        raise FileNotFoundError(
            f"Cohort file not found: "
            f"{COHORT_FILE}"
        )


    if not WSI_CANDIDATES_FILE.exists():

        raise FileNotFoundError(
            f"WSI candidate file not found: "
            f"{WSI_CANDIDATES_FILE}"
        )


    cohort = pd.read_csv(
        COHORT_FILE
    )


    wsi_candidates = pd.read_csv(
        WSI_CANDIDATES_FILE
    )


    all_cases = (
        failure_cases
        +
        control_cases
    )


    available_cases = set(
        cohort[
            "case_submitter_id"
        ].astype(str)
    )


    missing_cases = [
        case_id
        for case_id
        in all_cases
        if case_id
        not in available_cases
    ]


    if missing_cases:

        raise RuntimeError(
            "Cases missing from cohort: "
            f"{missing_cases}"
        )


    # ========================================================
    # HEADER
    # ========================================================

    print()
    print("=" * 88)

    print(
        " FINAL TISSUE FRACTION AUDIT"
    )

    print("=" * 88)


    print(
        f"Protocol:           "
        f"{AUDIT_PROTOCOL}"
    )


    print(
        "Thresholds:         "
        +
        ", ".join(
            [
                f"{value:.2f}"
                for value
                in TISSUE_THRESHOLDS
            ]
        )
    )


    print(
        f"Failure cases:      "
        f"{len(failure_cases)}"
    )

    print(
        f"Control cases:      "
        f"{len(control_cases)}"
    )

    print(
        f"Keep WSI:           "
        f"{args.keep_wsi}"
    )

    print(
        f"Free disk:          "
        f"{base.free_disk_gb():.2f} GB"
    )


    print()

    print(
        "FROZEN:"
    )


    print(
        f"  candidate tissue >= "
        f"{base.CANDIDATE_TISSUE_FRACTION:.2f}"
    )


    print(
        f"  background <= "
        f"{base.MAX_BACKGROUND_FRACTION:.2f}"
    )


    print(
        f"  white <= "
        f"{base.MAX_WHITE_FRACTION:.2f}"
    )


    print(
        f"  artifact QC = "
        f"{pipeline.ARTIFACT_PROTOCOL_NAME}"
    )


    print()

    print(
        "ONLY FINAL_TISSUE_FRACTION VARIES."
    )

    print("=" * 88)


    global_summary_rows = []


    # ========================================================
    # RUN FAILURES
    # ========================================================

    for case_id in failure_cases:

        cohort_row = cohort[
            cohort[
                "case_submitter_id"
            ].astype(str)
            ==
            case_id
        ].iloc[0]


        class_label = str(
            cohort_row[
                "class_label"
            ]
        )


        candidate = resolve_wsi_candidate(
            wsi_candidates,
            case_id,
        )


        rows = audit_case(
            case_id=case_id,

            group_name="failure",

            class_label=class_label,

            candidate=candidate,

            keep_wsi=(
                args.keep_wsi
            ),
        )


        global_summary_rows.extend(
            rows
        )


    # ========================================================
    # RUN CONTROLS
    # ========================================================

    for case_id in control_cases:

        cohort_row = cohort[
            cohort[
                "case_submitter_id"
            ].astype(str)
            ==
            case_id
        ].iloc[0]


        class_label = str(
            cohort_row[
                "class_label"
            ]
        )


        candidate = resolve_wsi_candidate(
            wsi_candidates,
            case_id,
        )


        rows = audit_case(
            case_id=case_id,

            group_name="control",

            class_label=class_label,

            candidate=candidate,

            keep_wsi=(
                args.keep_wsi
            ),
        )


        global_summary_rows.extend(
            rows
        )


    # ========================================================
    # GLOBAL SUMMARY
    # ========================================================

    summary = pd.DataFrame(
        global_summary_rows
    )


    summary_path = (
        OUTPUT_ROOT
        /
        "tissue_threshold_summary.csv"
    )


    summary.to_csv(
        summary_path,
        index=False,
    )


    # ========================================================
    # PIVOT
    # ========================================================

    pivot = build_selected_tiles_pivot(
        summary
    )


    pivot_path = (
        OUTPUT_ROOT
        /
        "selected_tiles_by_threshold.csv"
    )


    pivot.to_csv(
        pivot_path,
        index=False,
    )


    # ========================================================
    # SUCCESS RATES
    # ========================================================

    threshold_success = summary.groupby(
        [
            "group",
            "threshold",
        ],
        as_index=False,
    ).agg(
        cases=(
            "case_submitter_id",
            "count",
        ),

        cases_reaching_64=(
            "reached_64",
            "sum",
        ),
    )


    threshold_success[
        "success_fraction"
    ] = (
        threshold_success[
            "cases_reaching_64"
        ]
        /
        threshold_success[
            "cases"
        ]
    )


    threshold_success_path = (
        OUTPUT_ROOT
        /
        "threshold_success_rates.csv"
    )


    threshold_success.to_csv(
        threshold_success_path,
        index=False,
    )


    # ========================================================
    # GLOBAL FIGURE
    # ========================================================

    global_overview_path = (
        create_global_overview(
            all_cases
        )
    )


    # ========================================================
    # FINAL PRINT
    # ========================================================

    print()
    print("=" * 88)

    print(
        " TISSUE THRESHOLD AUDIT COMPLETE"
    )

    print("=" * 88)


    print()
    print(
        "SELECTED TILES BY THRESHOLD"
    )

    print()


    print(
        pivot.to_string(
            index=False
        )
    )


    print()
    print(
        "SUCCESS RATE BY THRESHOLD"
    )

    print()


    print(
        threshold_success.to_string(
            index=False
        )
    )


    print()
    print(
        "Summary:"
    )

    print(
        f"  {summary_path}"
    )


    print(
        "Pivot:"
    )

    print(
        f"  {pivot_path}"
    )


    print(
        "Success rates:"
    )

    print(
        f"  {threshold_success_path}"
    )


    print(
        "Figures:"
    )

    print(
        f"  {FIGURE_ROOT}"
    )


    if global_overview_path is not None:

        print(
            "Global overview:"
        )

        print(
            f"  {global_overview_path}"
        )


    print()

    print(
        "NO PRODUCTION FILES WERE MODIFIED."
    )

    print("=" * 88)


if __name__ == "__main__":
    main()