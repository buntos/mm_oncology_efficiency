from pathlib import Path
import json

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from skimage.color import rgb2hsv


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

MANIFEST_DIR = ROOT / "data" / "manifests"
METADATA_DIR = ROOT / "data" / "metadata"

RESULTS_DIR = (
    ROOT
    / "results"
    / "processed"
)

FIGURE_DIR = (
    ROOT
    / "results"
    / "figures"
    / "pilot_texture_audit_v2"
)

TILE_MANIFEST = (
    MANIFEST_DIR
    / "pilot_tile_manifest_stage5_v2.csv"
)

TILING_QC = (
    METADATA_DIR
    / "pilot_tiling_qc_stage5_v2.csv"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# Number of most suspicious tiles to show.
TOP_N = 32

# Contact sheet settings.
GRID_COLUMNS = 4
GRID_ROWS = 8

TILE_PREVIEW_SIZE = 256
LABEL_HEIGHT = 42


# ============================================================
# HELPERS
# ============================================================

def calculate_metrics(image):
    """
    Calculate simple colour and texture metrics.

    No rejection threshold is applied here.
    This script is diagnostic only.
    """

    rgb = np.asarray(
        image.convert("RGB"),
        dtype=np.float32,
    )

    rgb_01 = rgb / 255.0


    # --------------------------------------------------------
    # GRAYSCALE
    # --------------------------------------------------------

    gray = (
        0.299 * rgb[:, :, 0]
        + 0.587 * rgb[:, :, 1]
        + 0.114 * rgb[:, :, 2]
    )


    gray_mean = float(
        gray.mean()
    )

    gray_std = float(
        gray.std()
    )

    gray_var = float(
        gray.var()
    )


    gray_p05 = float(
        np.percentile(
            gray,
            5,
        )
    )

    gray_p95 = float(
        np.percentile(
            gray,
            95,
        )
    )

    gray_p01 = float(
        np.percentile(
            gray,
            1,
        )
    )

    gray_p99 = float(
        np.percentile(
            gray,
            99,
        )
    )


    contrast_p95_p05 = (
        gray_p95 - gray_p05
    )

    contrast_p99_p01 = (
        gray_p99 - gray_p01
    )


    # --------------------------------------------------------
    # SIMPLE LOCAL GRADIENT
    #
    # Flat scanner/background areas should have very low
    # neighbour-to-neighbour intensity changes.
    # --------------------------------------------------------

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


    gradient_p95 = float(
        np.percentile(
            np.concatenate([
                grad_x.ravel(),
                grad_y.ravel(),
            ]),
            95,
        )
    )


    # --------------------------------------------------------
    # HSV
    # --------------------------------------------------------

    hsv = rgb2hsv(
        rgb_01
    )

    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]


    saturation_mean = float(
        saturation.mean()
    )

    saturation_std = float(
        saturation.std()
    )

    value_mean = float(
        value.mean()
    )

    value_std = float(
        value.std()
    )


    # --------------------------------------------------------
    # RGB CHANNEL VARIABILITY
    # --------------------------------------------------------

    rgb_std = float(
        rgb.std()
    )


    return {
        "gray_mean": gray_mean,
        "gray_std": gray_std,
        "gray_var": gray_var,

        "gray_p01": gray_p01,
        "gray_p05": gray_p05,
        "gray_p95": gray_p95,
        "gray_p99": gray_p99,

        "contrast_p95_p05": (
            contrast_p95_p05
        ),

        "contrast_p99_p01": (
            contrast_p99_p01
        ),

        "gradient_mean": (
            gradient_mean
        ),

        "gradient_p95": (
            gradient_p95
        ),

        "saturation_mean": (
            saturation_mean
        ),

        "saturation_std": (
            saturation_std
        ),

        "value_mean": (
            value_mean
        ),

        "value_std": (
            value_std
        ),

        "rgb_std": (
            rgb_std
        ),
    }


def print_quantiles(
    dataframe,
    column,
):
    """
    Print useful quantiles for one metric.
    """

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

    values = (
        dataframe[column]
        .quantile(
            quantiles
        )
    )

    print(
        f"\n{column}:"
    )

    for q, value in values.items():

        print(
            f"  q{int(q * 100):02d}: "
            f"{value:.4f}"
        )


def create_contact_sheet(
    dataframe,
    out_path,
    title_metric,
):
    """
    Create a labelled contact sheet of suspicious tiles.
    """

    width = (
        GRID_COLUMNS
        * TILE_PREVIEW_SIZE
    )

    height = (
        GRID_ROWS
        * (
            TILE_PREVIEW_SIZE
            +
            LABEL_HEIGHT
        )
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


    for i, (_, row) in enumerate(
        dataframe.head(TOP_N).iterrows()
    ):

        if i >= (
            GRID_COLUMNS
            * GRID_ROWS
        ):
            break


        image = Image.open(
            row["tile_path"]
        ).convert(
            "RGB"
        )


        image = image.resize(
            (
                TILE_PREVIEW_SIZE,
                TILE_PREVIEW_SIZE,
            )
        )


        col = (
            i % GRID_COLUMNS
        )

        grid_row = (
            i // GRID_COLUMNS
        )


        x = (
            col
            * TILE_PREVIEW_SIZE
        )

        y = (
            grid_row
            * (
                TILE_PREVIEW_SIZE
                +
                LABEL_HEIGHT
            )
        )


        sheet.paste(
            image,
            (
                x,
                y,
            ),
        )


        label_y = (
            y
            + TILE_PREVIEW_SIZE
        )


        patient = row[
            "case_submitter_id"
        ]

        tile_index = int(
            row["tile_index"]
        )


        metric_value = row[
            title_metric
        ]


        label1 = (
            f"{patient} "
            f"#{tile_index:03d}"
        )

        label2 = (
            f"{title_metric}="
            f"{metric_value:.2f}"
        )


        draw.text(
            (
                x + 3,
                label_y + 2,
            ),
            label1,
            fill="black",
        )


        draw.text(
            (
                x + 3,
                label_y + 20,
            ),
            label2,
            fill="black",
        )


        image.close()


    sheet.save(
        out_path,
        quality=95,
    )


# ============================================================
# LOAD DATA
# ============================================================

tiles = pd.read_csv(
    TILE_MANIFEST
)

qc = pd.read_csv(
    TILING_QC
)


pass_patients = set(
    qc.loc[
        qc["status"] == "PASS",
        "case_submitter_id",
    ]
)


tiles = tiles[
    tiles[
        "case_submitter_id"
    ].isin(
        pass_patients
    )
].copy()


print()
print("=" * 78)
print(
    " TCGA-LUAD / TCGA-LUSC — V2 TILE TEXTURE AUDIT"
)
print("=" * 78)

print(
    f"PASS patients:                 "
    f"{len(pass_patients)}"
)

print(
    f"Tiles to analyse:              "
    f"{len(tiles)}"
)

print()


# ============================================================
# CALCULATE METRICS
# ============================================================

metric_rows = []


for i, row in tiles.iterrows():

    tile_path = Path(
        row["tile_path"]
    )


    if not tile_path.exists():

        print(
            f"WARNING: missing tile: "
            f"{tile_path}"
        )

        continue


    image = Image.open(
        tile_path
    )


    metrics = calculate_metrics(
        image
    )


    image.close()


    result = row.to_dict()

    result.update(
        metrics
    )


    metric_rows.append(
        result
    )


    if (
        len(metric_rows)
        % 100
        == 0
    ):

        print(
            f"  analysed "
            f"{len(metric_rows)}/{len(tiles)}"
        )


metrics_df = pd.DataFrame(
    metric_rows
)


# ============================================================
# SAVE FULL TABLE
# ============================================================

metrics_path = (
    RESULTS_DIR
    / "pilot_tile_texture_metrics_v2.csv"
)


metrics_df.to_csv(
    metrics_path,
    index=False,
)


# ============================================================
# DISTRIBUTIONS
# ============================================================

print()
print("-" * 78)
print("GLOBAL DISTRIBUTIONS")


for metric in [
    "gray_std",
    "gradient_mean",
    "contrast_p95_p05",
    "saturation_mean",
    "gray_mean",
]:

    print_quantiles(
        metrics_df,
        metric,
    )


# ============================================================
# PATIENT-LEVEL SUMMARY
# ============================================================

patient_summary = (
    metrics_df
    .groupby(
        [
            "case_submitter_id",
            "class_label",
        ]
    )
    .agg(
        tiles=(
            "tile_index",
            "count",
        ),

        gray_std_min=(
            "gray_std",
            "min",
        ),

        gray_std_median=(
            "gray_std",
            "median",
        ),

        gradient_mean_min=(
            "gradient_mean",
            "min",
        ),

        gradient_mean_median=(
            "gradient_mean",
            "median",
        ),

        contrast_min=(
            "contrast_p95_p05",
            "min",
        ),

        saturation_min=(
            "saturation_mean",
            "min",
        ),

        saturation_median=(
            "saturation_mean",
            "median",
        ),
    )
    .reset_index()
)


patient_summary_path = (
    RESULTS_DIR
    / "pilot_tile_texture_patient_summary_v2.csv"
)


patient_summary.to_csv(
    patient_summary_path,
    index=False,
)


print()
print("-" * 78)
print("PATIENT-LEVEL MINIMUM TEXTURE")
print()


display = patient_summary[
    [
        "case_submitter_id",
        "class_label",
        "gray_std_min",
        "gray_std_median",
        "gradient_mean_min",
        "gradient_mean_median",
        "contrast_min",
    ]
].copy()


for column in [
    "gray_std_min",
    "gray_std_median",
    "gradient_mean_min",
    "gradient_mean_median",
    "contrast_min",
]:

    display[column] = (
        display[column]
        .round(3)
    )


print(
    display.to_string(
        index=False
    )
)


# ============================================================
# LOWEST-TEXTURE TILES
# ============================================================

lowest_std = (
    metrics_df
    .sort_values(
        [
            "gray_std",
            "gradient_mean",
        ],
        ascending=[
            True,
            True,
        ],
    )
    .head(
        TOP_N
    )
    .copy()
)


lowest_gradient = (
    metrics_df
    .sort_values(
        [
            "gradient_mean",
            "gray_std",
        ],
        ascending=[
            True,
            True,
        ],
    )
    .head(
        TOP_N
    )
    .copy()
)


lowest_contrast = (
    metrics_df
    .sort_values(
        [
            "contrast_p95_p05",
            "gray_std",
        ],
        ascending=[
            True,
            True,
        ],
    )
    .head(
        TOP_N
    )
    .copy()
)


lowest_std.to_csv(
    RESULTS_DIR
    / "pilot_lowest_gray_std_tiles_v2.csv",
    index=False,
)


lowest_gradient.to_csv(
    RESULTS_DIR
    / "pilot_lowest_gradient_tiles_v2.csv",
    index=False,
)


lowest_contrast.to_csv(
    RESULTS_DIR
    / "pilot_lowest_contrast_tiles_v2.csv",
    index=False,
)


# ============================================================
# PRINT TOP 20 LOWEST GRAY STD
# ============================================================

print()
print("-" * 78)

print(
    "20 LOWEST GRAYSCALE STANDARD DEVIATION TILES"
)

print()


columns = [
    "case_submitter_id",
    "tile_index",
    "gray_mean",
    "gray_std",
    "gradient_mean",
    "contrast_p95_p05",
    "saturation_mean",
    "final_tissue_fraction",
    "background_fraction",
]


display = (
    lowest_std[
        columns
    ]
    .head(20)
    .copy()
)


for column in [
    "gray_mean",
    "gray_std",
    "gradient_mean",
    "contrast_p95_p05",
    "saturation_mean",
    "final_tissue_fraction",
    "background_fraction",
]:

    display[column] = (
        display[column]
        .round(4)
    )


print(
    display.to_string(
        index=False
    )
)


# ============================================================
# CONTACT SHEETS
# ============================================================

create_contact_sheet(
    lowest_std,
    FIGURE_DIR
    / "lowest_32_gray_std.jpg",
    "gray_std",
)


create_contact_sheet(
    lowest_gradient,
    FIGURE_DIR
    / "lowest_32_gradient.jpg",
    "gradient_mean",
)


create_contact_sheet(
    lowest_contrast,
    FIGURE_DIR
    / "lowest_32_contrast.jpg",
    "contrast_p95_p05",
)


# ============================================================
# RANK INFORMATION
# ============================================================

metrics_df[
    "gray_std_rank"
] = (
    metrics_df[
        "gray_std"
    ]
    .rank(
        method="min",
        ascending=True,
    )
)


metrics_df[
    "gradient_rank"
] = (
    metrics_df[
        "gradient_mean"
    ]
    .rank(
        method="min",
        ascending=True,
    )
)


metrics_df[
    "contrast_rank"
] = (
    metrics_df[
        "contrast_p95_p05"
    ]
    .rank(
        method="min",
        ascending=True,
    )
)


# Combined diagnostic score.
# Low values indicate low texture by several independent metrics.
metrics_df[
    "texture_rank_score"
] = (
    metrics_df[
        "gray_std_rank"
    ]
    +
    metrics_df[
        "gradient_rank"
    ]
    +
    metrics_df[
        "contrast_rank"
    ]
)


combined = (
    metrics_df
    .sort_values(
        "texture_rank_score"
    )
    .head(
        TOP_N
    )
    .copy()
)


combined.to_csv(
    RESULTS_DIR
    / "pilot_low_texture_combined_v2.csv",
    index=False,
)


create_contact_sheet(
    combined,
    FIGURE_DIR
    / "lowest_32_combined_texture.jpg",
    "gray_std",
)


# ============================================================
# SUMMARY JSON
# ============================================================

summary = {
    "protocol": (
        "V2 tile texture diagnostic audit"
    ),

    "pass_patients": int(
        len(pass_patients)
    ),

    "tiles_analysed": int(
        len(metrics_df)
    ),

    "gray_std": {
        "min": float(
            metrics_df[
                "gray_std"
            ].min()
        ),

        "p01": float(
            metrics_df[
                "gray_std"
            ].quantile(
                0.01
            )
        ),

        "p05": float(
            metrics_df[
                "gray_std"
            ].quantile(
                0.05
            )
        ),

        "median": float(
            metrics_df[
                "gray_std"
            ].median()
        ),

        "max": float(
            metrics_df[
                "gray_std"
            ].max()
        ),
    },

    "gradient_mean": {
        "min": float(
            metrics_df[
                "gradient_mean"
            ].min()
        ),

        "p01": float(
            metrics_df[
                "gradient_mean"
            ].quantile(
                0.01
            )
        ),

        "p05": float(
            metrics_df[
                "gradient_mean"
            ].quantile(
                0.05
            )
        ),

        "median": float(
            metrics_df[
                "gradient_mean"
            ].median()
        ),

        "max": float(
            metrics_df[
                "gradient_mean"
            ].max()
        ),
    },
}


summary_path = (
    METADATA_DIR
    / "pilot_tile_texture_audit_v2.json"
)


summary_path.write_text(
    json.dumps(
        summary,
        indent=2,
    ),
    encoding="utf-8",
)


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 78)
print(
    " V2 TILE TEXTURE AUDIT COMPLETE"
)
print("=" * 78)


print(
    f"Tiles analysed:                 "
    f"{len(metrics_df)}"
)


print()
print("Saved:")

print(
    "  results/processed/"
    "pilot_tile_texture_metrics_v2.csv"
)

print(
    "  results/processed/"
    "pilot_tile_texture_patient_summary_v2.csv"
)

print(
    "  results/processed/"
    "pilot_lowest_gray_std_tiles_v2.csv"
)

print(
    "  results/processed/"
    "pilot_lowest_gradient_tiles_v2.csv"
)

print(
    "  results/processed/"
    "pilot_lowest_contrast_tiles_v2.csv"
)

print(
    "  results/processed/"
    "pilot_low_texture_combined_v2.csv"
)

print(
    "  results/figures/"
    "pilot_texture_audit_v2/"
)

print(
    "  data/metadata/"
    "pilot_tile_texture_audit_v2.json"
)

print()
print(
    "NOTE: No tile rejection threshold was applied."
)

print(
    "Use these results to define the V3 texture filter."
)

print("=" * 78)