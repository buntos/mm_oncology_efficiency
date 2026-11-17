from pathlib import Path
import argparse
import importlib.util
import json

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


# ============================================================
# ROOT / LOAD SCRIPT 17
# ============================================================

THIS_FILE = Path(__file__).resolve()
ROOT = THIS_FILE.parents[1]

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

AUDIT_NAME = "tissue_qc_failure_reason_audit_v1"

AUDIT_PROTOCOL = (
    "V2_PROD_TISSUE_QC_FAILURE_REASON_ANALYSIS"
    "+FROZEN_ARTIFACT_QC_CANDIDATE_V1"
)


# ============================================================
# CASES
# ============================================================

DEFAULT_CASES = [
    "TCGA-05-4396",
    "TCGA-05-4397",
    "TCGA-05-4434",
]


# ============================================================
# FROZEN / TESTED PARAMETERS
# ============================================================
#
# 0.80 is NOT yet production-final.
# It is the candidate identified by audit 18.
#
# Artifact-QC remains frozen.
# ============================================================

FINAL_TISSUE_THRESHOLD = 0.80

CURRENT_BACKGROUND_THRESHOLD = 0.10
CURRENT_WHITE_THRESHOLD = 0.10

BACKGROUND_SWEEP = [
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.40,
    0.50,
]

WHITE_SWEEP = [
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.40,
    0.50,
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
    / "temp_gdc_tissue_qc_failure_reason_audit"
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

TARGET_TILES = base.TARGET_TILES

PREVIEW_TILE_SIZE = 128
CONTACT_COLUMNS = 8
CONTACT_ROWS = 8
HEADER_HEIGHT = 58

JPEG_QUALITY = 94


# ============================================================
# HELPERS
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


def threshold_slug(value):
    return str(
        int(
            round(
                value * 100
            )
        )
    )


def safe_close(image):
    try:
        image.close()
    except Exception:
        pass


# ============================================================
# FREEZE CONFIG
# ============================================================

FROZEN_CONFIG_PATH = (
    METADATA_ROOT
    / "frozen_audit_configuration.json"
)


def write_or_verify_frozen_config(case_ids):
    payload = {
        "audit_name": AUDIT_NAME,
        "audit_protocol": AUDIT_PROTOCOL,
        "case_ids": list(case_ids),

        "final_tissue_threshold": (
            FINAL_TISSUE_THRESHOLD
        ),

        "current_background_threshold": (
            CURRENT_BACKGROUND_THRESHOLD
        ),

        "current_white_threshold": (
            CURRENT_WHITE_THRESHOLD
        ),

        "background_sweep": (
            BACKGROUND_SWEEP
        ),

        "white_sweep": (
            WHITE_SWEEP
        ),

        "candidate_tissue_fraction": float(
            base.CANDIDATE_TISSUE_FRACTION
        ),

        "artifact_qc_protocol": (
            pipeline.ARTIFACT_PROTOCOL_NAME
        ),

        "artifact_rules": (
            pipeline.ARTIFACT_RULES
        ),

        "note": (
            "Audit only. No production files are modified. "
            "FINAL_TISSUE_FRACTION=0.80 is evaluated as a "
            "candidate value identified by audit 18."
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
                "Existing frozen configuration differs "
                "from the current audit. "
                "Use a new AUDIT_NAME instead of overwriting it."
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
            f"No diagnostic WSI candidate for {case_id}"
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
# GENERATE V2 THUMBNAIL CANDIDATES
# ============================================================

def generate_candidates(
    slide,
    tissue_mask,
):
    width, height = slide.dimensions

    thumb_h, thumb_w = tissue_mask.shape

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
# CONTACT SHEET
# ============================================================

def create_contact_sheet(
    slide,
    selected_rows,
    output_path,
    title,
):
    width = (
        CONTACT_COLUMNS
        *
        PREVIEW_TILE_SIZE
    )

    height = (
        HEADER_HEIGHT
        +
        CONTACT_ROWS
        *
        PREVIEW_TILE_SIZE
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
        18
    )

    small_font = load_font(
        13
    )

    draw.text(
        (
            8,
            7,
        ),
        title,
        fill="black",
        font=title_font,
    )

    draw.text(
        (
            8,
            34,
        ),
        (
            f"selected="
            f"{len(selected_rows)}/"
            f"{TARGET_TILES}"
        ),
        fill="black",
        font=small_font,
    )

    for index in range(
        TARGET_TILES
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
            PREVIEW_TILE_SIZE
        )

        paste_y = (
            HEADER_HEIGHT
            +
            row
            *
            PREVIEW_TILE_SIZE
        )

        if index < len(selected_rows):
            record = selected_rows.iloc[
                index
            ]

            x = int(
                record[
                    "x_level0"
                ]
            )

            y = int(
                record[
                    "y_level0"
                ]
            )

            tile = slide.read_region(
                (
                    x,
                    y,
                ),
                0,
                (
                    base.TILE_SIZE,
                    base.TILE_SIZE,
                ),
            ).convert(
                "RGB"
            )

            preview = tile.resize(
                (
                    PREVIEW_TILE_SIZE,
                    PREVIEW_TILE_SIZE,
                ),
                Image.Resampling.LANCZOS,
            )

            sheet.paste(
                preview,
                (
                    paste_x,
                    paste_y,
                ),
            )

            preview.close()
            tile.close()

        else:
            blank = Image.new(
                "RGB",
                (
                    PREVIEW_TILE_SIZE,
                    PREVIEW_TILE_SIZE,
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
# QUANTILES
# ============================================================

def calculate_quantiles(
    values,
    metric_name,
    subset_name,
):
    values = pd.Series(
        values
    ).dropna()

    if values.empty:
        return []

    quantiles = [
        0.00,
        0.01,
        0.05,
        0.10,
        0.25,
        0.50,
        0.75,
        0.90,
        0.95,
        0.99,
        1.00,
    ]

    rows = []

    for q in quantiles:
        rows.append(
            {
                "subset": (
                    subset_name
                ),

                "metric": (
                    metric_name
                ),

                "quantile": float(
                    q
                ),

                "value": float(
                    values.quantile(
                        q
                    )
                ),
            }
        )

    return rows


# ============================================================
# ACCEPTANCE MASK
# ============================================================

def acceptance_mask(
    metrics,
    background_threshold,
    white_threshold,
):
    return (
        (
            metrics[
                "final_tissue_fraction"
            ]
            >=
            FINAL_TISSUE_THRESHOLD
        )
        &
        (
            metrics[
                "background_fraction"
            ]
            <=
            background_threshold
        )
        &
        (
            metrics[
                "white_fraction"
            ]
            <=
            white_threshold
        )
        &
        (
            ~metrics[
                "artifact_reject"
            ]
        )
    )


# ============================================================
# SELECT FIRST 64 IN DETERMINISTIC ORDER
# ============================================================

def select_first_64(
    metrics,
    background_threshold,
    white_threshold,
):
    mask = acceptance_mask(
        metrics,
        background_threshold,
        white_threshold,
    )

    accepted = metrics[
        mask
    ].sort_values(
        "candidate_order"
    )

    return accepted.head(
        TARGET_TILES
    ).copy()


# ============================================================
# AUDIT ONE CASE
# ============================================================

def audit_case(
    case_id,
    class_label,
    candidate,
    keep_wsi,
):
    print()
    print("=" * 92)

    print(
        f" {case_id} "
        f"({class_label})"
    )

    print("=" * 92)

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

    file_size_bytes = int(
        candidate[
            "file_size_bytes"
        ]
    )

    print(
        f"WSI UUID:     "
        f"{file_id}"
    )

    print(
        f"Slide:        "
        f"{slide_token}"
    )

    print(
        f"WSI size:     "
        f"{file_size_bytes / 1024**3:.3f} GB"
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

        file_name=str(
            candidate[
                "file_name"
            ]
        ),

        expected_size=(
            file_size_bytes
        ),

        expected_md5=str(
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
            len(candidates)
        )

        rng.shuffle(
            order
        )

        print(
            f"Thumbnail candidates: "
            f"{len(candidates)}"
        )

        print(
            "Reading all candidates and "
            "saving detailed metrics..."
        )

        metric_rows = []

        for position, candidate_index in enumerate(
            order,
            start=1,
        ):
            x, y, approx_fraction = (
                candidates[
                    candidate_index
                ]
            )

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

            metric_rows.append(
                {
                    "case_submitter_id": (
                        case_id
                    ),

                    "class_label": (
                        class_label
                    ),

                    "candidate_order": int(
                        position
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

                    "artifact_reject": bool(
                        artifact_reject
                    ),

                    "artifact_reasons": (
                        ";".join(
                            artifact_reasons
                        )
                    ),

                    "near_black_fraction": float(
                        artifact_metrics[
                            "near_black_fraction"
                        ]
                    ),

                    "gradient_mean": float(
                        artifact_metrics[
                            "gradient_mean"
                        ]
                    ),

                    "red_excess": float(
                        artifact_metrics[
                            "red_excess"
                        ]
                    ),

                    "cyan_excess": float(
                        artifact_metrics[
                            "cyan_excess"
                        ]
                    ),

                    "saturation_mean": float(
                        artifact_metrics[
                            "saturation_mean"
                        ]
                    ),
                }
            )

            tile.close()

            if (
                position % 2000
                ==
                0
            ):
                print(
                    f"  checked "
                    f"{position}/"
                    f"{len(candidates)}"
                )

        metrics = pd.DataFrame(
            metric_rows
        )

        metrics_path = (
            case_output_dir
            /
            "candidate_metrics.csv"
        )

        metrics.to_csv(
            metrics_path,
            index=False,
        )

        # ====================================================
        # FAILURE FLAGS
        # ====================================================

        metrics[
            "tissue_fail_080"
        ] = (
            metrics[
                "final_tissue_fraction"
            ]
            <
            FINAL_TISSUE_THRESHOLD
        )

        metrics[
            "background_fail_010"
        ] = (
            metrics[
                "background_fraction"
            ]
            >
            CURRENT_BACKGROUND_THRESHOLD
        )

        metrics[
            "white_fail_010"
        ] = (
            metrics[
                "white_fraction"
            ]
            >
            CURRENT_WHITE_THRESHOLD
        )

        tissue_pass = metrics[
            ~metrics[
                "tissue_fail_080"
            ]
        ].copy()

        # ====================================================
        # WATERFALL
        # ====================================================

        n_total = len(
            metrics
        )

        n_after_tissue = int(
            (
                metrics[
                    "final_tissue_fraction"
                ]
                >=
                FINAL_TISSUE_THRESHOLD
            ).sum()
        )

        after_tissue_bg_mask = (
            (
                metrics[
                    "final_tissue_fraction"
                ]
                >=
                FINAL_TISSUE_THRESHOLD
            )
            &
            (
                metrics[
                    "background_fraction"
                ]
                <=
                CURRENT_BACKGROUND_THRESHOLD
            )
        )

        n_after_background = int(
            after_tissue_bg_mask.sum()
        )

        after_white_mask = (
            after_tissue_bg_mask
            &
            (
                metrics[
                    "white_fraction"
                ]
                <=
                CURRENT_WHITE_THRESHOLD
            )
        )

        n_after_white = int(
            after_white_mask.sum()
        )

        final_current_mask = (
            after_white_mask
            &
            (
                ~metrics[
                    "artifact_reject"
                ]
            )
        )

        n_final = int(
            final_current_mask.sum()
        )

        waterfall = pd.DataFrame(
            [
                {
                    "stage": (
                        "thumbnail_candidates"
                    ),
                    "remaining": (
                        n_total
                    ),
                },
                {
                    "stage": (
                        "tissue_ge_0.80"
                    ),
                    "remaining": (
                        n_after_tissue
                    ),
                },
                {
                    "stage": (
                        "background_le_0.10"
                    ),
                    "remaining": (
                        n_after_background
                    ),
                },
                {
                    "stage": (
                        "white_le_0.10"
                    ),
                    "remaining": (
                        n_after_white
                    ),
                },
                {
                    "stage": (
                        "artifact_qc_pass"
                    ),
                    "remaining": (
                        n_final
                    ),
                },
            ]
        )

        waterfall.to_csv(
            (
                case_output_dir
                /
                "filter_waterfall.csv"
            ),
            index=False,
        )

        # ====================================================
        # OVERLAP OF FAILURE REASONS
        # ====================================================

        reason_rows = []

        for _, row in tissue_pass.iterrows():
            reasons = []

            if (
                row[
                    "background_fraction"
                ]
                >
                CURRENT_BACKGROUND_THRESHOLD
            ):
                reasons.append(
                    "BACKGROUND"
                )

            if (
                row[
                    "white_fraction"
                ]
                >
                CURRENT_WHITE_THRESHOLD
            ):
                reasons.append(
                    "WHITE"
                )

            if bool(
                row[
                    "artifact_reject"
                ]
            ):
                reasons.append(
                    "ARTIFACT"
                )

            if not reasons:
                reason_name = (
                    "PASS"
                )

            else:
                reason_name = (
                    "+".join(
                        reasons
                    )
                )

            reason_rows.append(
                reason_name
            )

        if reason_rows:
            reason_summary = (
                pd.Series(
                    reason_rows
                )
                .value_counts()
                .rename_axis(
                    "reason_combination"
                )
                .reset_index(
                    name="count"
                )
            )

        else:
            reason_summary = pd.DataFrame(
                columns=[
                    "reason_combination",
                    "count",
                ]
            )

        reason_summary.to_csv(
            (
                case_output_dir
                /
                "failure_reason_combinations.csv"
            ),
            index=False,
        )

        # ====================================================
        # QUANTILES
        # ====================================================

        quantile_rows = []

        for metric_name in [
            "final_tissue_fraction",
            "background_fraction",
            "white_fraction",
        ]:
            quantile_rows.extend(
                calculate_quantiles(
                    metrics[
                        metric_name
                    ],
                    metric_name,
                    "all_thumbnail_candidates",
                )
            )

            quantile_rows.extend(
                calculate_quantiles(
                    tissue_pass[
                        metric_name
                    ],
                    metric_name,
                    "tissue_ge_0.80",
                )
            )

        quantiles = pd.DataFrame(
            quantile_rows
        )

        quantiles.to_csv(
            (
                case_output_dir
                /
                "metric_quantiles.csv"
            ),
            index=False,
        )

        # ====================================================
        # BACKGROUND SWEEP
        # ====================================================

        background_rows = []

        for background_threshold in (
            BACKGROUND_SWEEP
        ):
            mask = acceptance_mask(
                metrics,
                background_threshold=(
                    background_threshold
                ),
                white_threshold=(
                    CURRENT_WHITE_THRESHOLD
                ),
            )

            accepted = metrics[
                mask
            ].sort_values(
                "candidate_order"
            )

            accepted_count = int(
                len(
                    accepted
                )
            )

            if (
                accepted_count
                >=
                TARGET_TILES
            ):
                checked_to_64 = int(
                    accepted.iloc[
                        TARGET_TILES - 1
                    ][
                        "candidate_order"
                    ]
                )

            else:
                checked_to_64 = None

            background_rows.append(
                {
                    "background_threshold": float(
                        background_threshold
                    ),

                    "white_threshold": float(
                        CURRENT_WHITE_THRESHOLD
                    ),

                    "tissue_threshold": float(
                        FINAL_TISSUE_THRESHOLD
                    ),

                    "accepted_tiles": (
                        accepted_count
                    ),

                    "reached_64": bool(
                        accepted_count
                        >=
                        TARGET_TILES
                    ),

                    "checked_to_64": (
                        checked_to_64
                    ),
                }
            )

            selected = accepted.head(
                TARGET_TILES
            )

            slug = threshold_slug(
                background_threshold
            )

            create_contact_sheet(
                slide=slide,

                selected_rows=selected,

                output_path=(
                    case_figure_dir
                    /
                    (
                        "background_"
                        f"{slug}_contact.jpg"
                    )
                ),

                title=(
                    f"{case_id} | "
                    f"tissue>=0.80 | "
                    f"background<="
                    f"{background_threshold:.2f} | "
                    f"white<=0.10"
                ),
            )

        background_sweep = pd.DataFrame(
            background_rows
        )

        background_sweep.to_csv(
            (
                case_output_dir
                /
                "background_sweep.csv"
            ),
            index=False,
        )

        # ====================================================
        # WHITE SWEEP
        # ====================================================

        white_rows = []

        for white_threshold in (
            WHITE_SWEEP
        ):
            mask = acceptance_mask(
                metrics,
                background_threshold=(
                    CURRENT_BACKGROUND_THRESHOLD
                ),
                white_threshold=(
                    white_threshold
                ),
            )

            accepted = metrics[
                mask
            ].sort_values(
                "candidate_order"
            )

            accepted_count = int(
                len(
                    accepted
                )
            )

            if (
                accepted_count
                >=
                TARGET_TILES
            ):
                checked_to_64 = int(
                    accepted.iloc[
                        TARGET_TILES - 1
                    ][
                        "candidate_order"
                    ]
                )

            else:
                checked_to_64 = None

            white_rows.append(
                {
                    "background_threshold": float(
                        CURRENT_BACKGROUND_THRESHOLD
                    ),

                    "white_threshold": float(
                        white_threshold
                    ),

                    "tissue_threshold": float(
                        FINAL_TISSUE_THRESHOLD
                    ),

                    "accepted_tiles": (
                        accepted_count
                    ),

                    "reached_64": bool(
                        accepted_count
                        >=
                        TARGET_TILES
                    ),

                    "checked_to_64": (
                        checked_to_64
                    ),
                }
            )

            selected = accepted.head(
                TARGET_TILES
            )

            slug = threshold_slug(
                white_threshold
            )

            create_contact_sheet(
                slide=slide,

                selected_rows=selected,

                output_path=(
                    case_figure_dir
                    /
                    (
                        "white_"
                        f"{slug}_contact.jpg"
                    )
                ),

                title=(
                    f"{case_id} | "
                    f"tissue>=0.80 | "
                    f"background<=0.10 | "
                    f"white<="
                    f"{white_threshold:.2f}"
                ),
            )

        white_sweep = pd.DataFrame(
            white_rows
        )

        white_sweep.to_csv(
            (
                case_output_dir
                /
                "white_sweep.csv"
            ),
            index=False,
        )

        # ====================================================
        # 2D BACKGROUND x WHITE GRID
        # ====================================================

        grid_rows = []

        for background_threshold in (
            BACKGROUND_SWEEP
        ):
            for white_threshold in (
                WHITE_SWEEP
            ):
                mask = acceptance_mask(
                    metrics,

                    background_threshold=(
                        background_threshold
                    ),

                    white_threshold=(
                        white_threshold
                    ),
                )

                accepted = metrics[
                    mask
                ].sort_values(
                    "candidate_order"
                )

                accepted_count = int(
                    len(
                        accepted
                    )
                )

                if (
                    accepted_count
                    >=
                    TARGET_TILES
                ):
                    checked_to_64 = int(
                        accepted.iloc[
                            TARGET_TILES - 1
                        ][
                            "candidate_order"
                        ]
                    )

                else:
                    checked_to_64 = None

                grid_rows.append(
                    {
                        "background_threshold": float(
                            background_threshold
                        ),

                        "white_threshold": float(
                            white_threshold
                        ),

                        "accepted_tiles": (
                            accepted_count
                        ),

                        "reached_64": bool(
                            accepted_count
                            >=
                            TARGET_TILES
                        ),

                        "checked_to_64": (
                            checked_to_64
                        ),
                    }
                )

        grid = pd.DataFrame(
            grid_rows
        )

        grid.to_csv(
            (
                case_output_dir
                /
                "background_white_grid_long.csv"
            ),
            index=False,
        )

        grid_pivot = grid.pivot(
            index="background_threshold",
            columns="white_threshold",
            values="accepted_tiles",
        )

        grid_pivot.to_csv(
            (
                case_output_dir
                /
                "background_white_grid.csv"
            )
        )

        # ====================================================
        # PRINT CASE SUMMARY
        # ====================================================

        print()
        print(
            "FILTER WATERFALL"
        )

        print(
            waterfall.to_string(
                index=False
            )
        )

        print()
        print(
            "FAILURE COMBINATIONS "
            "AMONG tissue>=0.80"
        )

        if reason_summary.empty:
            print(
                "  none"
            )

        else:
            print(
                reason_summary.to_string(
                    index=False
                )
            )

        print()
        print(
            "BACKGROUND SWEEP "
            "(white fixed at 0.10)"
        )

        print(
            background_sweep[
                [
                    "background_threshold",
                    "accepted_tiles",
                    "reached_64",
                    "checked_to_64",
                ]
            ].to_string(
                index=False
            )
        )

        print()
        print(
            "WHITE SWEEP "
            "(background fixed at 0.10)"
        )

        print(
            white_sweep[
                [
                    "white_threshold",
                    "accepted_tiles",
                    "reached_64",
                    "checked_to_64",
                ]
            ].to_string(
                index=False
            )
        )

        # ====================================================
        # RETURN GLOBAL SUMMARY
        # ====================================================

        return {
            "case_submitter_id": (
                case_id
            ),

            "class_label": (
                class_label
            ),

            "thumbnail_candidates": int(
                n_total
            ),

            "tissue_ge_080": int(
                n_after_tissue
            ),

            "after_background_010": int(
                n_after_background
            ),

            "after_white_010": int(
                n_after_white
            ),

            "after_artifact_qc": int(
                n_final
            ),

            "background_fail_among_tissue_pass": int(
                (
                    tissue_pass[
                        "background_fraction"
                    ]
                    >
                    CURRENT_BACKGROUND_THRESHOLD
                ).sum()
            ),

            "white_fail_among_tissue_pass": int(
                (
                    tissue_pass[
                        "white_fraction"
                    ]
                    >
                    CURRENT_WHITE_THRESHOLD
                ).sum()
            ),

            "artifact_fail_among_tissue_pass": int(
                tissue_pass[
                    "artifact_reject"
                ].sum()
            ),
        }

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
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Analyze why selected WSI fail V2 tissue QC. "
            "Only diagnostic analysis; production is not modified."
        )
    )

    parser.add_argument(
        "--cases",
        nargs="+",
        default=DEFAULT_CASES,
    )

    parser.add_argument(
        "--keep-wsi",
        action="store_true",
    )

    args = parser.parse_args()

    case_ids = list(
        dict.fromkeys(
            [
                str(case_id)
                for case_id
                in args.cases
            ]
        )
    )

    write_or_verify_frozen_config(
        case_ids
    )

    if not COHORT_FILE.exists():
        raise FileNotFoundError(
            f"Cohort file not found: {COHORT_FILE}"
        )

    if not WSI_CANDIDATES_FILE.exists():
        raise FileNotFoundError(
            f"WSI candidates file not found: "
            f"{WSI_CANDIDATES_FILE}"
        )

    cohort = pd.read_csv(
        COHORT_FILE
    )

    wsi_candidates = pd.read_csv(
        WSI_CANDIDATES_FILE
    )

    available_cases = set(
        cohort[
            "case_submitter_id"
        ].astype(str)
    )

    missing = [
        case_id
        for case_id
        in case_ids
        if case_id
        not in available_cases
    ]

    if missing:
        raise RuntimeError(
            f"Cases missing from cohort: {missing}"
        )

    print()
    print("=" * 92)
    print(
        " TISSUE QC FAILURE REASON AUDIT"
    )
    print("=" * 92)

    print(
        f"Protocol: "
        f"{AUDIT_PROTOCOL}"
    )

    print(
        f"Cases:    "
        f"{', '.join(case_ids)}"
    )

    print()
    print(
        "AUDIT CANDIDATE:"
    )

    print(
        f"  final tissue >= "
        f"{FINAL_TISSUE_THRESHOLD:.2f}"
    )

    print()
    print(
        "CURRENT OTHER RULES:"
    )

    print(
        f"  background <= "
        f"{CURRENT_BACKGROUND_THRESHOLD:.2f}"
    )

    print(
        f"  white <= "
        f"{CURRENT_WHITE_THRESHOLD:.2f}"
    )

    print(
        f"  artifact QC = "
        f"{pipeline.ARTIFACT_PROTOCOL_NAME}"
    )

    print()
    print(
        "BACKGROUND SWEEP:"
    )

    print(
        "  "
        +
        ", ".join(
            [
                f"{value:.2f}"
                for value
                in BACKGROUND_SWEEP
            ]
        )
    )

    print(
        "WHITE SWEEP:"
    )

    print(
        "  "
        +
        ", ".join(
            [
                f"{value:.2f}"
                for value
                in WHITE_SWEEP
            ]
        )
    )

    print()
    print(
        f"Keep WSI: {args.keep_wsi}"
    )

    print(
        f"Free disk: "
        f"{base.free_disk_gb():.2f} GB"
    )

    print("=" * 92)

    global_rows = []

    for case_id in case_ids:
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

        result = audit_case(
            case_id=case_id,

            class_label=class_label,

            candidate=candidate,

            keep_wsi=(
                args.keep_wsi
            ),
        )

        global_rows.append(
            result
        )

    global_summary = pd.DataFrame(
        global_rows
    )

    global_summary_path = (
        OUTPUT_ROOT
        /
        "global_failure_reason_summary.csv"
    )

    global_summary.to_csv(
        global_summary_path,
        index=False,
    )

    print()
    print("=" * 92)
    print(
        " GLOBAL FAILURE REASON SUMMARY"
    )
    print("=" * 92)

    print()
    print(
        global_summary.to_string(
            index=False
        )
    )

    print()
    print(
        "Global summary:"
    )

    print(
        f"  {global_summary_path}"
    )

    print()
    print(
        "Per-case candidate metrics:"
    )

    for case_id in case_ids:
        print(
            "  "
            +
            str(
                OUTPUT_ROOT
                /
                case_id
                /
                "candidate_metrics.csv"
            )
        )

    print()
    print(
        "Figures:"
    )

    print(
        f"  {FIGURE_ROOT}"
    )

    print()
    print(
        "NO PRODUCTION FILES WERE MODIFIED."
    )

    print("=" * 92)


if __name__ == "__main__":
    main()