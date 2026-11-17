from pathlib import Path
import argparse
import json
import math

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


# ============================================================
# ROOT / PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_METRICS_PATH = (
    ROOT
    / "results"
    / "processed"
    / "validation30_v2prod_tile_artifact_metrics.csv"
)

OUTPUT_ROOT = (
    ROOT
    / "results"
    / "artifact_qc_rule_simulation"
)

FIGURE_ROOT = (
    ROOT
    / "results"
    / "figures"
    / "artifact_qc_rule_simulation"
)

METADATA_ROOT = (
    ROOT
    / "data"
    / "metadata"
    / "artifact_qc_rule_simulation"
)


for directory in [
    OUTPUT_ROOT,
    FIGURE_ROOT,
    METADATA_ROOT,
]:
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# REFERENCE CASES
# ============================================================
#
# These are NOT used by the rules.
#
# They are only printed separately because visual audit already
# identified them before this rule simulation.
#
# This lets us answer:
#
#   "Does a candidate rule detect known technical artifacts?"
#
# without letting patient identity influence classification.
#
# ============================================================

REFERENCE_SUSPICIOUS_CASES = {
    "TCGA-86-8056",
    "TCGA-86-8278",
    "TCGA-78-8660",
}


# ============================================================
# CANDIDATE RULESETS
# ============================================================
#
# IMPORTANT:
#
# These are diagnostic candidate rules only.
#
# They are deliberately evaluated as THREE profiles so that we
# do not silently choose one threshold set based on validation30.
#
# Final thresholds must only be frozen after independent review.
#
# ------------------------------------------------------------
#
# near_black_fraction
#   proportion of pixels where max RGB < 50
#
# gradient_mean
#   mean local luminance gradient
#
# red_excess
#   R - (G+B)/2
#
# cyan_excess
#   (G+B)/2 - R
#
# saturation_mean
#   HSV saturation mean
#
# ============================================================

RULESETS = {

    # --------------------------------------------------------
    # CONSERVATIVE
    #
    # Intended to catch only very obvious corruption:
    #
    # - almost entirely black tiles
    # - extremely flat + extremely abnormal colour
    # - extremely red + highly saturated tiles
    # - extreme cyan/green-blue tiles
    #
    # --------------------------------------------------------

    "conservative": {

        "black_fraction": 0.70,

        "flat_gradient": 0.16,

        "flat_red_excess": 140.0,

        "flat_cyan_excess": 15.0,

        "hyper_red_excess": 160.0,

        "hyper_red_saturation": 0.85,

        "hyper_cyan_excess": 35.0,

        "hyper_cyan_saturation": 0.70,
    },


    # --------------------------------------------------------
    # BALANCED
    #
    # Slightly wider candidate region.
    #
    # --------------------------------------------------------

    "balanced": {

        "black_fraction": 0.50,

        "flat_gradient": 0.22,

        "flat_red_excess": 135.0,

        "flat_cyan_excess": 12.0,

        "hyper_red_excess": 150.0,

        "hyper_red_saturation": 0.80,

        "hyper_cyan_excess": 25.0,

        "hyper_cyan_saturation": 0.65,
    },


    # --------------------------------------------------------
    # SENSITIVE
    #
    # Deliberately broad.
    #
    # We do NOT expect to automatically adopt this one.
    # Its purpose is to show where false-positive rejection
    # begins to appear.
    #
    # --------------------------------------------------------

    "sensitive": {

        "black_fraction": 0.30,

        "flat_gradient": 0.30,

        "flat_red_excess": 125.0,

        "flat_cyan_excess": 8.0,

        "hyper_red_excess": 140.0,

        "hyper_red_saturation": 0.75,

        "hyper_cyan_excess": 18.0,

        "hyper_cyan_saturation": 0.60,
    },
}


# ============================================================
# CONTACT SHEET
# ============================================================

CONTACT_COLUMNS = 4
PREVIEW_SIZE = 256
LABEL_HEIGHT = 74
HEADER_HEIGHT = 64

DEFAULT_TOP_N = 48

JPEG_QUALITY = 95


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


def safe_float(value, default=0.0):

    try:

        value = float(value)

        if np.isfinite(value):
            return value

    except Exception:
        pass

    return default


# ============================================================
# RULE EVALUATION
# ============================================================

def evaluate_rules(
    dataframe,
    ruleset_name,
    thresholds,
):

    result = dataframe.copy()


    black = (
        result["near_black_fraction"]
        >= thresholds["black_fraction"]
    )


    flat_red = (
        (
            result["gradient_mean"]
            <= thresholds["flat_gradient"]
        )
        &
        (
            result["red_excess"]
            >= thresholds["flat_red_excess"]
        )
    )


    flat_cyan = (
        (
            result["gradient_mean"]
            <= thresholds["flat_gradient"]
        )
        &
        (
            result["cyan_excess"]
            >= thresholds["flat_cyan_excess"]
        )
    )


    hyper_red = (
        (
            result["red_excess"]
            >= thresholds["hyper_red_excess"]
        )
        &
        (
            result["saturation_mean"]
            >= thresholds["hyper_red_saturation"]
        )
    )


    hyper_cyan = (
        (
            result["cyan_excess"]
            >= thresholds["hyper_cyan_excess"]
        )
        &
        (
            result["saturation_mean"]
            >= thresholds["hyper_cyan_saturation"]
        )
    )


    result["rule_black"] = black

    result["rule_flat_red"] = flat_red

    result["rule_flat_cyan"] = flat_cyan

    result["rule_hyper_red"] = hyper_red

    result["rule_hyper_cyan"] = hyper_cyan


    result["would_reject"] = (
        black
        |
        flat_red
        |
        flat_cyan
        |
        hyper_red
        |
        hyper_cyan
    )


    # ========================================================
    # HUMAN-READABLE REASONS
    # ========================================================

    reasons = []


    for _, row in result.iterrows():

        tile_reasons = []


        if bool(row["rule_black"]):

            tile_reasons.append(
                "near_black"
            )


        if bool(row["rule_flat_red"]):

            tile_reasons.append(
                "flat_extreme_red"
            )


        if bool(row["rule_flat_cyan"]):

            tile_reasons.append(
                "flat_extreme_cyan"
            )


        if bool(row["rule_hyper_red"]):

            tile_reasons.append(
                "hyper_red_saturated"
            )


        if bool(row["rule_hyper_cyan"]):

            tile_reasons.append(
                "hyper_cyan_saturated"
            )


        reasons.append(
            ";".join(tile_reasons)
        )


    result["reject_reasons"] = reasons


    # ========================================================
    # RULE SEVERITY
    # ========================================================
    #
    # This is ONLY for sorting the review contact sheet.
    #
    # It is NOT another rejection rule.
    #
    # Score ~1 = around threshold.
    # Larger = more extreme.
    #
    # ========================================================

    severity = np.zeros(
        len(result),
        dtype=np.float64,
    )


    black_score = (
        result["near_black_fraction"]
        /
        thresholds["black_fraction"]
    )


    red_flat_score = np.minimum(
        result["red_excess"]
        /
        thresholds["flat_red_excess"],

        thresholds["flat_gradient"]
        /
        np.maximum(
            result["gradient_mean"],
            1e-6,
        ),
    )


    cyan_flat_score = np.minimum(
        result["cyan_excess"]
        /
        thresholds["flat_cyan_excess"],

        thresholds["flat_gradient"]
        /
        np.maximum(
            result["gradient_mean"],
            1e-6,
        ),
    )


    hyper_red_score = np.minimum(
        result["red_excess"]
        /
        thresholds["hyper_red_excess"],

        result["saturation_mean"]
        /
        thresholds["hyper_red_saturation"],
    )


    hyper_cyan_score = np.minimum(
        result["cyan_excess"]
        /
        thresholds["hyper_cyan_excess"],

        result["saturation_mean"]
        /
        thresholds["hyper_cyan_saturation"],
    )


    severity = np.maximum.reduce([
        black_score.to_numpy(),
        red_flat_score.to_numpy(),
        cyan_flat_score.to_numpy(),
        hyper_red_score.to_numpy(),
        hyper_cyan_score.to_numpy(),
    ])


    result["rule_severity"] = severity


    result["ruleset"] = (
        ruleset_name
    )


    return result


# ============================================================
# BORDERLINE RETAINED SCORE
# ============================================================

def calculate_borderline_score(
    dataframe,
    thresholds,
):

    result = dataframe.copy()


    components = []


    components.append(
        result["near_black_fraction"]
        /
        thresholds["black_fraction"]
    )


    components.append(
        result["red_excess"]
        /
        thresholds["flat_red_excess"]
    )


    components.append(
        result["cyan_excess"]
        /
        thresholds["flat_cyan_excess"]
    )


    components.append(
        thresholds["flat_gradient"]
        /
        np.maximum(
            result["gradient_mean"],
            1e-6,
        )
    )


    components.append(
        result["saturation_mean"]
        /
        thresholds["hyper_red_saturation"]
    )


    stacked = np.column_stack(
        [
            np.asarray(
                component,
                dtype=np.float64,
            )
            for component
            in components
        ]
    )


    result["borderline_score"] = (
        np.nanmax(
            stacked,
            axis=1,
        )
    )


    return result


# ============================================================
# CONTACT SHEET
# ============================================================

def create_contact_sheet(
    dataframe,
    output_path,
    title,
    value_column,
    top_n,
):

    subset = (
        dataframe
        .head(top_n)
        .copy()
    )


    if len(subset) == 0:

        image = Image.new(
            "RGB",
            (1000, 180),
            "white",
        )

        draw = ImageDraw.Draw(
            image
        )

        font = load_font(24)

        draw.text(
            (20, 20),
            title,
            fill="black",
            font=font,
        )

        draw.text(
            (20, 80),
            "No tiles in this category.",
            fill="black",
            font=font,
        )

        image.save(
            output_path,
            quality=JPEG_QUALITY,
        )

        return


    rows = math.ceil(
        len(subset)
        /
        CONTACT_COLUMNS
    )


    width = (
        CONTACT_COLUMNS
        *
        PREVIEW_SIZE
    )


    height = (
        HEADER_HEIGHT
        +
        rows
        *
        (
            PREVIEW_SIZE
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


    title_font = load_font(22)
    label_font = load_font(13)


    draw.text(
        (10, 12),
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
            PREVIEW_SIZE
        )

        y = (
            HEADER_HEIGHT
            +
            row
            *
            (
                PREVIEW_SIZE
                +
                LABEL_HEIGHT
            )
        )


        tile_path = Path(
            str(
                record["tile_path"]
            )
        )


        with Image.open(
            tile_path
        ) as source:

            image = source.convert(
                "RGB"
            )


            if image.size != (
                PREVIEW_SIZE,
                PREVIEW_SIZE,
            ):

                image = image.resize(
                    (
                        PREVIEW_SIZE,
                        PREVIEW_SIZE,
                    ),
                    Image.Resampling.LANCZOS,
                )


            sheet.paste(
                image,
                (x, y),
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


        value = safe_float(
            record[
                value_column
            ]
        )


        reason = str(
            record.get(
                "reject_reasons",
                "",
            )
        )


        line1 = (
            f"{case_id}  "
            f"#{tile_index:02d}"
        )


        line2 = (
            f"{value_column}="
            f"{value:.3f}"
        )


        line3 = (
            reason
            if reason
            else "RETAINED"
        )


        draw.text(
            (
                x + 3,
                y
                +
                PREVIEW_SIZE
                +
                3,
            ),
            line1,
            fill="black",
            font=label_font,
        )


        draw.text(
            (
                x + 3,
                y
                +
                PREVIEW_SIZE
                +
                25,
            ),
            line2,
            fill="black",
            font=label_font,
        )


        draw.text(
            (
                x + 3,
                y
                +
                PREVIEW_SIZE
                +
                47,
            ),
            line3,
            fill="black",
            font=label_font,
        )


    sheet.save(
        output_path,
        quality=JPEG_QUALITY,
        subsampling=0,
    )


# ============================================================
# PER-PATIENT SUMMARY
# ============================================================

def build_patient_summary(
    evaluated,
):

    summary = (
        evaluated
        .groupby(
            [
                "case_submitter_id",
                "class_label",
            ],
            dropna=False,
        )
        .agg(
            total_tiles=(
                "tile_index",
                "count",
            ),

            rejected_tiles=(
                "would_reject",
                "sum",
            ),

            black_tiles=(
                "rule_black",
                "sum",
            ),

            flat_red_tiles=(
                "rule_flat_red",
                "sum",
            ),

            flat_cyan_tiles=(
                "rule_flat_cyan",
                "sum",
            ),

            hyper_red_tiles=(
                "rule_hyper_red",
                "sum",
            ),

            hyper_cyan_tiles=(
                "rule_hyper_cyan",
                "sum",
            ),

            max_rule_severity=(
                "rule_severity",
                "max",
            ),
        )
        .reset_index()
    )


    summary["remaining_of_original_64"] = (
        summary["total_tiles"]
        -
        summary["rejected_tiles"]
    )


    summary["rejected_fraction"] = (
        summary["rejected_tiles"]
        /
        summary["total_tiles"]
    )


    summary["reference_suspicious"] = (
        summary[
            "case_submitter_id"
        ]
        .astype(str)
        .isin(
            REFERENCE_SUSPICIOUS_CASES
        )
    )


    summary = (
        summary
        .sort_values(
            [
                "rejected_tiles",
                "max_rule_severity",
            ],
            ascending=[
                False,
                False,
            ],
        )
        .reset_index(
            drop=True
        )
    )


    return summary


# ============================================================
# PRINT SUMMARY
# ============================================================

def print_ruleset_summary(
    ruleset_name,
    evaluated,
    patient_summary,
):

    rejected = evaluated[
        evaluated[
            "would_reject"
        ]
    ]


    total_tiles = len(
        evaluated
    )


    rejected_count = len(
        rejected
    )


    rejected_pct = (
        rejected_count
        /
        total_tiles
        *
        100.0
    )


    affected_patients = int(
        (
            patient_summary[
                "rejected_tiles"
            ]
            > 0
        )
        .sum()
    )


    print()
    print("=" * 88)
    print(
        f" RULESET: {ruleset_name.upper()}"
    )
    print("=" * 88)

    print(
        f"Rejected tiles:        "
        f"{rejected_count}/{total_tiles} "
        f"({rejected_pct:.2f}%)"
    )

    print(
        f"Patients affected:     "
        f"{affected_patients}/"
        f"{len(patient_summary)}"
    )


    print()
    print(
        "Rejection reason counts "
        "(rules may overlap):"
    )


    reason_columns = [
        (
            "near_black",
            "rule_black",
        ),
        (
            "flat_extreme_red",
            "rule_flat_red",
        ),
        (
            "flat_extreme_cyan",
            "rule_flat_cyan",
        ),
        (
            "hyper_red_saturated",
            "rule_hyper_red",
        ),
        (
            "hyper_cyan_saturated",
            "rule_hyper_cyan",
        ),
    ]


    for label, column in (
        reason_columns
    ):

        count = int(
            evaluated[
                column
            ].sum()
        )


        print(
            f"  {label:<24} "
            f"{count:4d}"
        )


    print()
    print(
        "REFERENCE SUSPICIOUS CASES"
    )


    reference = (
        patient_summary[
            patient_summary[
                "reference_suspicious"
            ]
        ]
        .copy()
    )


    if len(reference) > 0:

        print(
            reference[
                [
                    "case_submitter_id",
                    "total_tiles",
                    "rejected_tiles",
                    "remaining_of_original_64",
                    "rejected_fraction",
                ]
            ]
            .to_string(
                index=False
            )
        )


    print()
    print(
        "OTHER PATIENTS WITH MOST REJECTIONS"
    )


    others = (
        patient_summary[
            ~patient_summary[
                "reference_suspicious"
            ]
        ]
        .head(12)
        .copy()
    )


    print(
        others[
            [
                "case_submitter_id",
                "class_label",
                "rejected_tiles",
                "remaining_of_original_64",
                "rejected_fraction",
                "max_rule_severity",
            ]
        ]
        .to_string(
            index=False
        )
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Simulate conservative candidate artifact-QC "
            "rules on validation30 V2_PROD tile metrics. "
            "No files are deleted or modified."
        )
    )


    parser.add_argument(
        "--metrics",
        default=str(
            DEFAULT_METRICS_PATH
        ),
    )


    parser.add_argument(
        "--top-n",
        type=int,
        default=DEFAULT_TOP_N,
    )


    args = parser.parse_args()


    metrics_path = Path(
        args.metrics
    )


    if not metrics_path.exists():

        raise FileNotFoundError(
            f"Metrics file not found: "
            f"{metrics_path}"
        )


    metrics = pd.read_csv(
        metrics_path
    )


    required_columns = [
        "case_submitter_id",
        "class_label",
        "tile_index",
        "tile_path",
        "near_black_fraction",
        "gradient_mean",
        "red_excess",
        "cyan_excess",
        "saturation_mean",
    ]


    missing_columns = [
        column
        for column
        in required_columns
        if column
        not in metrics.columns
    ]


    if missing_columns:

        raise RuntimeError(
            "Required columns missing from metrics CSV: "
            f"{missing_columns}"
        )


    # ========================================================
    # VERIFY TILE FILES
    # ========================================================

    missing_tile_paths = []


    for tile_path in (
        metrics[
            "tile_path"
        ]
        .astype(str)
    ):

        if not Path(
            tile_path
        ).exists():

            missing_tile_paths.append(
                tile_path
            )


    if missing_tile_paths:

        raise RuntimeError(
            f"{len(missing_tile_paths)} "
            "tile files are missing."
        )


    patient_count = int(
        metrics[
            "case_submitter_id"
        ]
        .nunique()
    )


    print()
    print("=" * 88)
    print(
        " ARTIFACT QC RULE SIMULATION"
    )
    print("=" * 88)

    print(
        f"Input metrics:        "
        f"{metrics_path}"
    )

    print(
        f"Patients:             "
        f"{patient_count}"
    )

    print(
        f"Tiles:                "
        f"{len(metrics)}"
    )

    print(
        f"Candidate rulesets:   "
        f"{len(RULESETS)}"
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "  - diagnostic simulation only"
    )

    print(
        "  - no tile is deleted"
    )

    print(
        "  - no manifest is modified"
    )

    print(
        "  - remaining_of_original_64 does NOT mean "
        "the WSI has insufficient tissue"
    )

    print(
        "    because production would continue searching "
        "for replacement candidates."
    )

    print("=" * 88)


    master_comparison_rows = []

    all_rule_outputs = {}


    # ========================================================
    # EVALUATE RULESETS
    # ========================================================

    for (
        ruleset_name,
        thresholds,
    ) in RULESETS.items():

        evaluated = evaluate_rules(
            metrics,
            ruleset_name,
            thresholds,
        )


        patient_summary = (
            build_patient_summary(
                evaluated
            )
        )


        rejected = (
            evaluated[
                evaluated[
                    "would_reject"
                ]
            ]
            .sort_values(
                "rule_severity",
                ascending=False,
            )
            .copy()
        )


        retained = (
            evaluated[
                ~evaluated[
                    "would_reject"
                ]
            ]
            .copy()
        )


        retained = (
            calculate_borderline_score(
                retained,
                thresholds,
            )
        )


        retained = (
            retained
            .sort_values(
                "borderline_score",
                ascending=False,
            )
        )


        # ====================================================
        # OUTPUT DIRECTORIES
        # ====================================================

        ruleset_output = (
            OUTPUT_ROOT
            /
            ruleset_name
        )


        ruleset_figures = (
            FIGURE_ROOT
            /
            ruleset_name
        )


        ruleset_output.mkdir(
            parents=True,
            exist_ok=True,
        )


        ruleset_figures.mkdir(
            parents=True,
            exist_ok=True,
        )


        # ====================================================
        # SAVE TABLES
        # ====================================================

        evaluated_path = (
            ruleset_output
            /
            "all_tiles_with_simulated_qc.csv"
        )


        rejected_path = (
            ruleset_output
            /
            "rejected_tiles.csv"
        )


        retained_path = (
            ruleset_output
            /
            "highest_risk_retained_tiles.csv"
        )


        patient_path = (
            ruleset_output
            /
            "patient_summary.csv"
        )


        evaluated.to_csv(
            evaluated_path,
            index=False,
        )


        rejected.to_csv(
            rejected_path,
            index=False,
        )


        retained.head(
            args.top_n
        ).to_csv(
            retained_path,
            index=False,
        )


        patient_summary.to_csv(
            patient_path,
            index=False,
        )


        # ====================================================
        # CONTACT SHEETS
        # ====================================================

        create_contact_sheet(
            rejected,
            ruleset_figures
            /
            "01_would_reject.jpg",
            (
                f"{ruleset_name} — "
                "tiles candidate QC would reject"
            ),
            "rule_severity",
            args.top_n,
        )


        create_contact_sheet(
            retained,
            ruleset_figures
            /
            "02_highest_risk_retained.jpg",
            (
                f"{ruleset_name} — "
                "highest-risk tiles still retained"
            ),
            "borderline_score",
            args.top_n,
        )


        # ====================================================
        # PRINT
        # ====================================================

        print_ruleset_summary(
            ruleset_name,
            evaluated,
            patient_summary,
        )


        # ====================================================
        # MASTER SUMMARY
        # ====================================================

        rejected_count = int(
            evaluated[
                "would_reject"
            ].sum()
        )


        affected_patients = int(
            (
                patient_summary[
                    "rejected_tiles"
                ]
                > 0
            ).sum()
        )


        reference_summary = (
            patient_summary[
                patient_summary[
                    "reference_suspicious"
                ]
            ]
        )


        nonreference_summary = (
            patient_summary[
                ~patient_summary[
                    "reference_suspicious"
                ]
            ]
        )


        master_comparison_rows.append({

            "ruleset": (
                ruleset_name
            ),

            "tiles_total": int(
                len(evaluated)
            ),

            "tiles_rejected": (
                rejected_count
            ),

            "tiles_rejected_fraction": (
                rejected_count
                /
                len(evaluated)
            ),

            "patients_total": int(
                len(patient_summary)
            ),

            "patients_affected": (
                affected_patients
            ),

            "reference_cases_rejected_tiles": int(
                reference_summary[
                    "rejected_tiles"
                ].sum()
            ),

            "other_cases_rejected_tiles": int(
                nonreference_summary[
                    "rejected_tiles"
                ].sum()
            ),

            "other_cases_affected": int(
                (
                    nonreference_summary[
                        "rejected_tiles"
                    ]
                    > 0
                ).sum()
            ),
        })


        all_rule_outputs[
            ruleset_name
        ] = {
            "thresholds": thresholds,

            "tiles_rejected": (
                rejected_count
            ),

            "patients_affected": (
                affected_patients
            ),

            "rejected_csv": str(
                rejected_path
            ),

            "patient_summary_csv": str(
                patient_path
            ),

            "rejected_contact_sheet": str(
                ruleset_figures
                /
                "01_would_reject.jpg"
            ),

            "retained_contact_sheet": str(
                ruleset_figures
                /
                "02_highest_risk_retained.jpg"
            ),
        }


    # ========================================================
    # MASTER COMPARISON
    # ========================================================

    comparison = pd.DataFrame(
        master_comparison_rows
    )


    comparison_path = (
        OUTPUT_ROOT
        /
        "ruleset_comparison.csv"
    )


    comparison.to_csv(
        comparison_path,
        index=False,
    )


    print()
    print("=" * 88)
    print(
        " RULESET COMPARISON"
    )
    print("=" * 88)

    print(
        comparison.to_string(
            index=False
        )
    )


    # ========================================================
    # JSON METADATA
    # ========================================================

    metadata = {

        "purpose": (
            "Diagnostic simulation of candidate "
            "artifact rejection rules."
        ),

        "important_note": (
            "No threshold in this script is a frozen "
            "production threshold. Rules must be reviewed "
            "visually and validated on an independent batch."
        ),

        "input_metrics": str(
            metrics_path
        ),

        "patients": int(
            patient_count
        ),

        "tiles": int(
            len(metrics)
        ),

        "reference_suspicious_cases": sorted(
            REFERENCE_SUSPICIOUS_CASES
        ),

        "rulesets": (
            all_rule_outputs
        ),
    }


    metadata_path = (
        METADATA_ROOT
        /
        "artifact_qc_rule_simulation.json"
    )


    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


    # ========================================================
    # FINAL
    # ========================================================

    print()
    print("=" * 88)
    print(
        " ARTIFACT QC RULE SIMULATION COMPLETE"
    )
    print("=" * 88)

    print(
        "Comparison:"
    )

    print(
        f"  {comparison_path}"
    )

    print()
    print(
        "Per-ruleset tables:"
    )

    print(
        f"  {OUTPUT_ROOT}"
    )

    print()
    print(
        "Review figures:"
    )

    print(
        f"  {FIGURE_ROOT}"
    )

    print()
    print(
        "Metadata:"
    )

    print(
        f"  {metadata_path}"
    )

    print()
    print(
        "NO TILE FILES WERE MODIFIED."
    )

    print("=" * 88)


if __name__ == "__main__":
    main()