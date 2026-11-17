from pathlib import Path
import argparse
import json
import math

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from skimage.color import rgb2hsv


# ============================================================
# ROOT / PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

MANIFEST_DIR = ROOT / "data" / "manifests"
METADATA_DIR = ROOT / "data" / "metadata"

PATIENT_TILE_MANIFEST_ROOT = (
    MANIFEST_DIR
    / "production_v2"
    / "patient_tiles"
)

RESULTS_DIR = (
    ROOT
    / "results"
    / "processed"
)

FIGURE_ROOT = (
    ROOT
    / "results"
    / "figures"
    / "validation30_tile_artifact_audit"
)

AUDIT_METADATA_DIR = (
    METADATA_DIR
    / "validation30_tile_artifact_audit"
)


for directory in [
    RESULTS_DIR,
    FIGURE_ROOT,
    AUDIT_METADATA_DIR,
]:
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# CONTACT SHEET CONFIG
# ============================================================

TOP_N_DEFAULT = 32

CONTACT_COLUMNS = 4

TILE_PREVIEW = 256
LABEL_HEIGHT = 54

JPEG_QUALITY = 95


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
# METRICS
# ============================================================

def calculate_tile_metrics(image):

    rgb = np.asarray(
        image.convert("RGB"),
        dtype=np.float32,
    )

    rgb01 = rgb / 255.0


    # --------------------------------------------------------
    # RGB
    # --------------------------------------------------------

    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]


    mean_r = float(
        r.mean()
    )

    mean_g = float(
        g.mean()
    )

    mean_b = float(
        b.mean()
    )


    std_r = float(
        r.std()
    )

    std_g = float(
        g.std()
    )

    std_b = float(
        b.std()
    )


    # --------------------------------------------------------
    # LUMINANCE / TEXTURE
    # --------------------------------------------------------

    gray = (
        0.299 * r
        +
        0.587 * g
        +
        0.114 * b
    )


    gray_mean = float(
        gray.mean()
    )

    gray_std = float(
        gray.std()
    )


    gray_p01 = float(
        np.percentile(
            gray,
            1,
        )
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

    gray_p99 = float(
        np.percentile(
            gray,
            99,
        )
    )


    contrast_p95_p05 = (
        gray_p95
        -
        gray_p05
    )


    # --------------------------------------------------------
    # LOCAL GRADIENT
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


    gradient_values = np.concatenate([
        grad_x.ravel(),
        grad_y.ravel(),
    ])


    gradient_p95 = float(
        np.percentile(
            gradient_values,
            95,
        )
    )


    # --------------------------------------------------------
    # HSV
    # --------------------------------------------------------

    hsv = rgb2hsv(
        rgb01
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
    # BLACK / WHITE
    # --------------------------------------------------------

    max_channel = rgb.max(
        axis=2
    )

    min_channel = rgb.min(
        axis=2
    )


    black_fraction = float(
        (
            max_channel
            < 25
        ).mean()
    )


    near_black_fraction = float(
        (
            max_channel
            < 50
        ).mean()
    )


    dark_fraction = float(
        (
            max_channel
            < 80
        ).mean()
    )


    near_white_fraction = float(
        (
            min_channel
            > 245
        ).mean()
    )


    # --------------------------------------------------------
    # COLOUR DOMINANCE
    # --------------------------------------------------------
    #
    # Signed metrics:
    #
    # red_excess > 0  -> unusually red
    # cyan_excess > 0 -> green/blue dominance over red
    # magenta_excess  -> red/blue dominance over green
    #

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


    magenta_excess_map = (
        (
            r + b
        )
        / 2.0
        -
        g
    )


    red_excess = float(
        red_excess_map.mean()
    )

    cyan_excess = float(
        cyan_excess_map.mean()
    )

    magenta_excess = float(
        magenta_excess_map.mean()
    )


    # Strong pixel-level dominance.
    strong_red_fraction = float(
        (
            red_excess_map
            > 50
        ).mean()
    )


    strong_cyan_fraction = float(
        (
            cyan_excess_map
            > 50
        ).mean()
    )


    # --------------------------------------------------------
    # NORMALIZED CHROMATICITY
    # --------------------------------------------------------

    rgb_sum = (
        mean_r
        +
        mean_g
        +
        mean_b
        +
        1e-8
    )


    chroma_r = float(
        mean_r
        / rgb_sum
    )

    chroma_g = float(
        mean_g
        / rgb_sum
    )

    chroma_b = float(
        mean_b
        / rgb_sum
    )


    # --------------------------------------------------------
    # SPATIAL RGB VARIABILITY
    # --------------------------------------------------------

    rgb_spatial_std = float(
        (
            std_r
            +
            std_g
            +
            std_b
        )
        / 3.0
    )


    return {
        "mean_r": mean_r,
        "mean_g": mean_g,
        "mean_b": mean_b,

        "std_r": std_r,
        "std_g": std_g,
        "std_b": std_b,

        "gray_mean": gray_mean,
        "gray_std": gray_std,

        "gray_p01": gray_p01,
        "gray_p05": gray_p05,
        "gray_p95": gray_p95,
        "gray_p99": gray_p99,

        "contrast_p95_p05": (
            contrast_p95_p05
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

        "black_fraction": (
            black_fraction
        ),

        "near_black_fraction": (
            near_black_fraction
        ),

        "dark_fraction": (
            dark_fraction
        ),

        "near_white_fraction": (
            near_white_fraction
        ),

        "red_excess": (
            red_excess
        ),

        "cyan_excess": (
            cyan_excess
        ),

        "magenta_excess": (
            magenta_excess
        ),

        "strong_red_fraction": (
            strong_red_fraction
        ),

        "strong_cyan_fraction": (
            strong_cyan_fraction
        ),

        "chroma_r": (
            chroma_r
        ),

        "chroma_g": (
            chroma_g
        ),

        "chroma_b": (
            chroma_b
        ),

        "rgb_spatial_std": (
            rgb_spatial_std
        ),
    }


# ============================================================
# ROBUST STANDARDIZATION
# ============================================================

def robust_z(series):

    values = pd.to_numeric(
        series,
        errors="coerce",
    ).astype(float)


    median = float(
        values.median()
    )


    mad = float(
        (
            values
            -
            median
        )
        .abs()
        .median()
    )


    scale = (
        1.4826
        * mad
    )


    # Some metrics such as black_fraction can contain
    # many exact zero values, giving MAD=0.
    if (
        not np.isfinite(
            scale
        )
        or
        scale
        < 1e-9
    ):

        q25 = float(
            values.quantile(
                0.25
            )
        )

        q75 = float(
            values.quantile(
                0.75
            )
        )

        iqr = (
            q75
            -
            q25
        )


        scale = (
            iqr
            / 1.349
        )


    if (
        not np.isfinite(
            scale
        )
        or
        scale
        < 1e-9
    ):

        scale = float(
            values.std()
        )


    if (
        not np.isfinite(
            scale
        )
        or
        scale
        < 1e-9
    ):

        scale = 1.0


    return (
        values
        -
        median
    ) / scale


# ============================================================
# CONTACT SHEET
# ============================================================

def create_contact_sheet(
    dataframe,
    output_path,
    title,
    metric,
    top_n,
):

    subset = (
        dataframe
        .head(
            top_n
        )
        .copy()
    )


    rows = math.ceil(
        len(subset)
        /
        CONTACT_COLUMNS
    )


    width = (
        CONTACT_COLUMNS
        *
        TILE_PREVIEW
    )


    header_height = 52


    height = (
        header_height
        +
        rows
        *
        (
            TILE_PREVIEW
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


    title_font = load_font(
        23
    )

    label_font = load_font(
        14
    )


    draw.text(
        (
            10,
            10,
        ),
        title,
        fill="black",
        font=title_font,
    )


    for index, (
        _,
        record,
    ) in enumerate(
        subset.iterrows()
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


        x = (
            column
            *
            TILE_PREVIEW
        )

        y = (
            header_height
            +
            row
            *
            (
                TILE_PREVIEW
                +
                LABEL_HEIGHT
            )
        )


        tile_path = Path(
            record[
                "tile_path"
            ]
        )


        with Image.open(
            tile_path
        ) as source:

            image = source.convert(
                "RGB"
            )

            if (
                image.size
                !=
                (
                    TILE_PREVIEW,
                    TILE_PREVIEW,
                )
            ):

                image = image.resize(
                    (
                        TILE_PREVIEW,
                        TILE_PREVIEW,
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


        case_id = str(
            record[
                "case_submitter_id"
            ]
        )

        tile_index = int(
            record[
                "tile_index"
            ]
        )


        metric_value = float(
            record[
                metric
            ]
        )


        line1 = (
            f"{case_id}  "
            f"#{tile_index:02d}"
        )


        line2 = (
            f"{metric}="
            f"{metric_value:.4f}"
        )


        draw.text(
            (
                x + 4,
                y
                +
                TILE_PREVIEW
                +
                3,
            ),
            line1,
            fill="black",
            font=label_font,
        )


        draw.text(
            (
                x + 4,
                y
                +
                TILE_PREVIEW
                +
                27,
            ),
            line2,
            fill="black",
            font=label_font,
        )


    sheet.save(
        output_path,
        quality=JPEG_QUALITY,
        subsampling=0,
    )


# ============================================================
# PRINT DISTRIBUTION
# ============================================================

def print_distribution(
    dataframe,
    metric,
):

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
        dataframe[
            metric
        ]
        .quantile(
            quantiles
        )
    )


    print()
    print(
        f"{metric}:"
    )


    for quantile, value in (
        values.items()
    ):

        print(
            f"  q{int(quantile * 100):02d}: "
            f"{value:.4f}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Diagnostic artifact audit for all "
            "V2_PROD tiles in a frozen validation batch."
        )
    )


    parser.add_argument(
        "--batch-name",
        default="validation30_v2prod",
    )


    parser.add_argument(
        "--top-n",
        type=int,
        default=TOP_N_DEFAULT,
    )


    args = parser.parse_args()


    batch_manifest = (
        MANIFEST_DIR
        /
        f"{args.batch_name}_patients.csv"
    )


    if not batch_manifest.exists():

        raise FileNotFoundError(
            f"Batch manifest missing: "
            f"{batch_manifest}"
        )


    batch = pd.read_csv(
        batch_manifest
    )


    expected_patients = len(
        batch
    )


    expected_tiles = (
        expected_patients
        *
        64
    )


    # ========================================================
    # LOAD ALL TILE MANIFESTS
    # ========================================================

    tile_frames = []

    missing_manifests = []


    print()
    print("=" * 80)
    print(
        " V2_PROD VALIDATION TILE ARTIFACT AUDIT"
    )
    print("=" * 80)

    print(
        f"Batch:                    "
        f"{args.batch_name}"
    )

    print(
        f"Patients expected:        "
        f"{expected_patients}"
    )

    print(
        f"Tiles expected:           "
        f"{expected_tiles}"
    )

    print()


    for _, patient in (
        batch.iterrows()
    ):

        case_id = str(
            patient[
                "case_submitter_id"
            ]
        )


        path = (
            PATIENT_TILE_MANIFEST_ROOT
            /
            f"{case_id}.csv"
        )


        if not path.exists():

            missing_manifests.append(
                case_id
            )

            continue


        patient_tiles = pd.read_csv(
            path
        )


        if (
            "class_label"
            not in patient_tiles.columns
        ):

            patient_tiles[
                "class_label"
            ] = (
                patient[
                    "class_label"
                ]
            )


        tile_frames.append(
            patient_tiles
        )


    if not tile_frames:

        raise RuntimeError(
            "No tile manifests found."
        )


    tiles = pd.concat(
        tile_frames,
        ignore_index=True,
    )


    print(
        f"Patient manifests found:  "
        f"{len(tile_frames)}"
    )

    print(
        f"Patient manifests missing:"
        f"  {len(missing_manifests)}"
    )

    print(
        f"Tiles loaded:             "
        f"{len(tiles)}"
    )


    if missing_manifests:

        print()

        print(
            "Missing patient manifests:"
        )

        for case_id in (
            missing_manifests
        ):

            print(
                f"  {case_id}"
            )


    # ========================================================
    # VERIFY TILE FILES
    # ========================================================

    missing_tiles = []


    for _, row in (
        tiles.iterrows()
    ):

        tile_path = Path(
            row[
                "tile_path"
            ]
        )


        if not tile_path.exists():

            missing_tiles.append(
                str(
                    tile_path
                )
            )


    print(
        f"Missing tile files:       "
        f"{len(missing_tiles)}"
    )


    if missing_tiles:

        raise RuntimeError(
            "Some tile files are missing. "
            "Audit aborted."
        )


    # ========================================================
    # CALCULATE METRICS
    # ========================================================

    print()
    print(
        "Calculating artifact metrics..."
    )


    records = []


    for number, (
        _,
        row,
    ) in enumerate(
        tiles.iterrows(),
        start=1,
    ):

        tile_path = Path(
            row[
                "tile_path"
            ]
        )


        with Image.open(
            tile_path
        ) as image:

            metrics = calculate_tile_metrics(
                image
            )


        record = row.to_dict()

        record.update(
            metrics
        )


        records.append(
            record
        )


        if (
            number
            % 200
            == 0
        ):

            print(
                f"  analysed "
                f"{number}/{len(tiles)}"
            )


    metrics_df = pd.DataFrame(
        records
    )


    # ========================================================
    # ROBUST OUTLIER FEATURES
    # ========================================================

    #
    # Positive values mean "more suspicious" for that
    # particular artifact type.
    #
    # These are diagnostic rankings only.
    # NO rejection threshold is applied.
    #

    metrics_df[
        "z_red"
    ] = robust_z(
        metrics_df[
            "red_excess"
        ]
    )


    metrics_df[
        "z_cyan"
    ] = robust_z(
        metrics_df[
            "cyan_excess"
        ]
    )


    metrics_df[
        "z_black"
    ] = robust_z(
        metrics_df[
            "near_black_fraction"
        ]
    )


    metrics_df[
        "z_dark"
    ] = -robust_z(
        metrics_df[
            "gray_mean"
        ]
    )


    metrics_df[
        "z_low_gray_std"
    ] = -robust_z(
        metrics_df[
            "gray_std"
        ]
    )


    metrics_df[
        "z_low_gradient"
    ] = -robust_z(
        metrics_df[
            "gradient_mean"
        ]
    )


    metrics_df[
        "z_high_saturation"
    ] = robust_z(
        metrics_df[
            "saturation_mean"
        ]
    )


    metrics_df[
        "z_low_value_std"
    ] = -robust_z(
        metrics_df[
            "value_std"
        ]
    )


    # --------------------------------------------------------
    # General multifeature distance from the centre of the
    # validation distribution.
    #
    # This does NOT define "bad tile".
    # It merely ranks unusual tiles for human review.
    # --------------------------------------------------------

    distance_features = [
        "gray_mean",
        "gray_std",
        "gradient_mean",
        "saturation_mean",
        "value_mean",
        "red_excess",
        "cyan_excess",
        "chroma_r",
        "chroma_g",
        "chroma_b",
    ]


    squared_distance = np.zeros(
        len(
            metrics_df
        ),
        dtype=np.float64,
    )


    for feature in (
        distance_features
    ):

        z = robust_z(
            metrics_df[
                feature
            ]
        )


        # Prevent one extreme value from dominating
        # numerical visualization completely.
        z = np.clip(
            z,
            -20,
            20,
        )


        squared_distance += (
            z.to_numpy()
            ** 2
        )


    metrics_df[
        "robust_multifeature_distance"
    ] = np.sqrt(
        squared_distance
    )


    # --------------------------------------------------------
    # "Any artifact" score:
    #
    # take strongest positive artifact signal.
    #
    # Still only a diagnostic ranking.
    # --------------------------------------------------------

    artifact_signals = metrics_df[
        [
            "z_red",
            "z_cyan",
            "z_black",
            "z_dark",
            "z_low_gray_std",
            "z_low_gradient",
            "z_high_saturation",
            "z_low_value_std",
        ]
    ].clip(
        lower=0
    )


    metrics_df[
        "max_artifact_signal"
    ] = artifact_signals.max(
        axis=1
    )


    # ========================================================
    # SAVE COMPLETE TABLE
    # ========================================================

    metrics_path = (
        RESULTS_DIR
        /
        f"{args.batch_name}_tile_artifact_metrics.csv"
    )


    metrics_df.to_csv(
        metrics_path,
        index=False,
    )


    # ========================================================
    # GLOBAL DISTRIBUTIONS
    # ========================================================

    print()
    print("-" * 80)
    print(
        "GLOBAL DISTRIBUTIONS"
    )


    for metric in [
        "near_black_fraction",
        "red_excess",
        "cyan_excess",
        "gray_std",
        "gradient_mean",
        "saturation_mean",
        "robust_multifeature_distance",
        "max_artifact_signal",
    ]:

        print_distribution(
            metrics_df,
            metric,
        )


    # ========================================================
    # RANKED TABLES
    # ========================================================

    darkest = (
        metrics_df
        .sort_values(
            [
                "near_black_fraction",
                "gray_mean",
            ],
            ascending=[
                False,
                True,
            ],
        )
        .head(
            args.top_n
        )
        .copy()
    )


    reddest = (
        metrics_df
        .sort_values(
            [
                "red_excess",
                "strong_red_fraction",
            ],
            ascending=[
                False,
                False,
            ],
        )
        .head(
            args.top_n
        )
        .copy()
    )


    cyanest = (
        metrics_df
        .sort_values(
            [
                "cyan_excess",
                "strong_cyan_fraction",
            ],
            ascending=[
                False,
                False,
            ],
        )
        .head(
            args.top_n
        )
        .copy()
    )


    lowest_texture = (
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
            args.top_n
        )
        .copy()
    )


    highest_saturation = (
        metrics_df
        .sort_values(
            [
                "saturation_mean",
                "gradient_mean",
            ],
            ascending=[
                False,
                True,
            ],
        )
        .head(
            args.top_n
        )
        .copy()
    )


    multifeature_outliers = (
        metrics_df
        .sort_values(
            "robust_multifeature_distance",
            ascending=False,
        )
        .head(
            args.top_n
        )
        .copy()
    )


    combined_outliers = (
        metrics_df
        .sort_values(
            "max_artifact_signal",
            ascending=False,
        )
        .head(
            args.top_n
        )
        .copy()
    )


    ranked_sets = {
        "darkest": darkest,
        "reddest": reddest,
        "cyanest": cyanest,
        "lowest_texture": lowest_texture,
        "highest_saturation": highest_saturation,
        "multifeature_outliers": (
            multifeature_outliers
        ),
        "combined_outliers": (
            combined_outliers
        ),
    }


    for name, dataframe in (
        ranked_sets.items()
    ):

        dataframe.to_csv(
            RESULTS_DIR
            /
            (
                f"{args.batch_name}_"
                f"{name}.csv"
            ),
            index=False,
        )


    # ========================================================
    # CONTACT SHEETS
    # ========================================================

    print()
    print("-" * 80)
    print(
        "CREATING REVIEW CONTACT SHEETS"
    )


    create_contact_sheet(
        darkest,
        FIGURE_ROOT
        / "01_darkest_tiles.jpg",
        (
            "Validation30 — darkest / "
            "near-black tiles"
        ),
        "near_black_fraction",
        args.top_n,
    )


    create_contact_sheet(
        reddest,
        FIGURE_ROOT
        / "02_strongest_red_tiles.jpg",
        (
            "Validation30 — strongest "
            "red dominance"
        ),
        "red_excess",
        args.top_n,
    )


    create_contact_sheet(
        cyanest,
        FIGURE_ROOT
        / "03_strongest_cyan_tiles.jpg",
        (
            "Validation30 — strongest "
            "cyan / green-blue dominance"
        ),
        "cyan_excess",
        args.top_n,
    )


    create_contact_sheet(
        lowest_texture,
        FIGURE_ROOT
        / "04_lowest_texture_tiles.jpg",
        (
            "Validation30 — lowest "
            "local texture"
        ),
        "gradient_mean",
        args.top_n,
    )


    create_contact_sheet(
        highest_saturation,
        FIGURE_ROOT
        / "05_highest_saturation_tiles.jpg",
        (
            "Validation30 — highest "
            "mean saturation"
        ),
        "saturation_mean",
        args.top_n,
    )


    create_contact_sheet(
        multifeature_outliers,
        FIGURE_ROOT
        / "06_multifeature_outliers.jpg",
        (
            "Validation30 — robust "
            "multifeature outliers"
        ),
        "robust_multifeature_distance",
        args.top_n,
    )


    create_contact_sheet(
        combined_outliers,
        FIGURE_ROOT
        / "07_combined_artifact_signals.jpg",
        (
            "Validation30 — strongest "
            "single artifact signal"
        ),
        "max_artifact_signal",
        args.top_n,
    )


    # ========================================================
    # PATIENT-LEVEL SUMMARY
    # ========================================================

    patient_summary = (
        metrics_df
        .groupby(
            [
                "case_submitter_id",
                "class_label",
            ],
            dropna=False,
        )
        .agg(
            tiles=(
                "tile_index",
                "count",
            ),

            max_near_black_fraction=(
                "near_black_fraction",
                "max",
            ),

            max_red_excess=(
                "red_excess",
                "max",
            ),

            max_cyan_excess=(
                "cyan_excess",
                "max",
            ),

            min_gray_std=(
                "gray_std",
                "min",
            ),

            min_gradient_mean=(
                "gradient_mean",
                "min",
            ),

            max_saturation_mean=(
                "saturation_mean",
                "max",
            ),

            max_multifeature_distance=(
                "robust_multifeature_distance",
                "max",
            ),

            max_artifact_signal=(
                "max_artifact_signal",
                "max",
            ),
        )
        .reset_index()
    )


    patient_summary = (
        patient_summary
        .sort_values(
            "max_multifeature_distance",
            ascending=False,
        )
    )


    patient_summary_path = (
        RESULTS_DIR
        /
        (
            f"{args.batch_name}_"
            "patient_artifact_summary.csv"
        )
    )


    patient_summary.to_csv(
        patient_summary_path,
        index=False,
    )


    print()
    print(
        "PATIENTS WITH HIGHEST "
        "MULTIFEATURE OUTLIER SCORE"
    )

    print()


    display = (
        patient_summary[
            [
                "case_submitter_id",
                "class_label",
                "max_near_black_fraction",
                "max_red_excess",
                "max_cyan_excess",
                "min_gradient_mean",
                "max_multifeature_distance",
                "max_artifact_signal",
            ]
        ]
        .head(15)
        .copy()
    )


    numeric_columns = [
        column
        for column
        in display.columns
        if column
        not in [
            "case_submitter_id",
            "class_label",
        ]
    ]


    for column in (
        numeric_columns
    ):

        display[
            column
        ] = (
            display[
                column
            ]
            .round(
                4
            )
        )


    print(
        display.to_string(
            index=False
        )
    )


    # ========================================================
    # SUMMARY JSON
    # ========================================================

    summary = {
        "protocol": (
            "V2_PROD validation tile "
            "artifact diagnostic audit"
        ),

        "batch_name": (
            args.batch_name
        ),

        "patients_expected": int(
            expected_patients
        ),

        "patients_loaded": int(
            len(
                tile_frames
            )
        ),

        "tiles_expected": int(
            expected_tiles
        ),

        "tiles_loaded": int(
            len(
                metrics_df
            )
        ),

        "missing_patient_manifests": (
            missing_manifests
        ),

        "missing_tile_files": int(
            len(
                missing_tiles
            )
        ),

        "important_note": (
            "Diagnostic ranking only. "
            "No artifact rejection threshold "
            "was applied."
        ),

        "top_multifeature_outliers": (
            multifeature_outliers[
                [
                    "case_submitter_id",
                    "tile_index",
                    "robust_multifeature_distance",
                    "tile_path",
                ]
            ]
            .head(
                10
            )
            .to_dict(
                orient="records"
            )
        ),

        "top_combined_artifact_signals": (
            combined_outliers[
                [
                    "case_submitter_id",
                    "tile_index",
                    "max_artifact_signal",
                    "tile_path",
                ]
            ]
            .head(
                10
            )
            .to_dict(
                orient="records"
            )
        ),
    }


    summary_path = (
        AUDIT_METADATA_DIR
        /
        f"{args.batch_name}_artifact_audit.json"
    )


    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


    # ========================================================
    # FINAL
    # ========================================================

    print()
    print("=" * 80)
    print(
        " VALIDATION TILE ARTIFACT AUDIT COMPLETE"
    )
    print("=" * 80)

    print(
        f"Patients analysed:        "
        f"{len(tile_frames)}"
    )

    print(
        f"Tiles analysed:           "
        f"{len(metrics_df)}"
    )

    print()
    print(
        "Full metrics:"
    )

    print(
        f"  {metrics_path}"
    )

    print(
        "Patient summary:"
    )

    print(
        f"  {patient_summary_path}"
    )

    print(
        "Review figures:"
    )

    print(
        f"  {FIGURE_ROOT}"
    )

    print(
        "Audit JSON:"
    )

    print(
        f"  {summary_path}"
    )

    print()
    print(
        "IMPORTANT: no rejection threshold "
        "was applied."
    )

    print("=" * 80)


if __name__ == "__main__":
    main()