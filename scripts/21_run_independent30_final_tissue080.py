from pathlib import Path
import hashlib
import importlib.util
import json
import sys

import pandas as pd


# ============================================================
# PATHS
# ============================================================

THIS_FILE = Path(__file__).resolve()
ROOT = THIS_FILE.parents[1]

PIPELINE17_PATH = (
    THIS_FILE.parent
    / "17_process_cohort_v2prod_artifact_qc.py"
)


# ============================================================
# RUN IDENTITY
# ============================================================

BATCH_NAME = (
    "independent30_final_tissue080_v1"
)

BATCH_SIZE = 30

FINAL_TISSUE_FRACTION = 0.80


# ============================================================
# EXPECTED FROZEN CONFIG
# ============================================================

EXPECTED_CANDIDATE_TISSUE_FRACTION = 0.70
EXPECTED_MAX_BACKGROUND_FRACTION = 0.10
EXPECTED_MAX_WHITE_FRACTION = 0.10

EXPECTED_TILE_SIZE = 256
EXPECTED_TARGET_TILES = 64


EXPECTED_ARTIFACT_RULES = {
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
# OUTPUT METADATA
# ============================================================

RUN_METADATA_DIR = (
    ROOT
    / "data"
    / "metadata"
    / "independent30_final_tissue080_v1"
)

RUN_CONFIG_PATH = (
    RUN_METADATA_DIR
    / "frozen_run_configuration.json"
)


# ============================================================
# HELPERS
# ============================================================

def sha256_file(path):

    digest = hashlib.sha256()

    with open(
        path,
        "rb",
    ) as handle:

        while True:

            chunk = handle.read(
                1024 * 1024
            )

            if not chunk:
                break

            digest.update(
                chunk
            )

    return digest.hexdigest()


def assert_float_equal(
    name,
    actual,
    expected,
):

    actual = float(
        actual
    )

    expected = float(
        expected
    )

    if abs(
        actual
        -
        expected
    ) > 1e-12:

        raise RuntimeError(
            f"{name} changed unexpectedly: "
            f"{actual} != {expected}. "
            f"This independent run must vary "
            f"only FINAL_TISSUE_FRACTION."
        )


# ============================================================
# LOAD SCRIPT 17
# ============================================================

def load_pipeline17():

    if not PIPELINE17_PATH.exists():

        raise FileNotFoundError(
            f"Script 17 not found: "
            f"{PIPELINE17_PATH}"
        )


    spec = importlib.util.spec_from_file_location(
        "artifact_pipeline_v1",
        PIPELINE17_PATH,
    )


    if (
        spec is None
        or
        spec.loader is None
    ):

        raise RuntimeError(
            f"Unable to load "
            f"{PIPELINE17_PATH}"
        )


    module = (
        importlib.util.module_from_spec(
            spec
        )
    )


    spec.loader.exec_module(
        module
    )


    return module


# ============================================================
# VERIFY FROZEN RULES
# ============================================================

def verify_frozen_rules(
    pipeline,
):

    if not hasattr(
        pipeline,
        "base",
    ):

        raise RuntimeError(
            "Script 17 does not expose "
            "the imported base pipeline."
        )


    base = pipeline.base


    assert_float_equal(
        "CANDIDATE_TISSUE_FRACTION",
        base.CANDIDATE_TISSUE_FRACTION,
        EXPECTED_CANDIDATE_TISSUE_FRACTION,
    )


    assert_float_equal(
        "MAX_BACKGROUND_FRACTION",
        base.MAX_BACKGROUND_FRACTION,
        EXPECTED_MAX_BACKGROUND_FRACTION,
    )


    assert_float_equal(
        "MAX_WHITE_FRACTION",
        base.MAX_WHITE_FRACTION,
        EXPECTED_MAX_WHITE_FRACTION,
    )


    if int(
        base.TILE_SIZE
    ) != EXPECTED_TILE_SIZE:

        raise RuntimeError(
            f"TILE_SIZE changed: "
            f"{base.TILE_SIZE}"
        )


    if int(
        base.TARGET_TILES
    ) != EXPECTED_TARGET_TILES:

        raise RuntimeError(
            f"TARGET_TILES changed: "
            f"{base.TARGET_TILES}"
        )


    if not hasattr(
        pipeline,
        "ARTIFACT_RULES",
    ):

        raise RuntimeError(
            "ARTIFACT_RULES not found "
            "in script 17."
        )


    for key in (
        EXPECTED_ARTIFACT_RULES
    ):

        if (
            key
            not in
            pipeline.ARTIFACT_RULES
        ):

            raise RuntimeError(
                f"Missing artifact rule: "
                f"{key}"
            )


        assert_float_equal(
            f"ARTIFACT_RULES[{key}]",
            pipeline.ARTIFACT_RULES[
                key
            ],
            EXPECTED_ARTIFACT_RULES[
                key
            ],
        )


# ============================================================
# PATCH ONLY FINAL TISSUE THRESHOLD
# ============================================================

def apply_tissue_080(
    pipeline,
):

    base = pipeline.base


    old_value = float(
        base.FINAL_TISSUE_FRACTION
    )


    base.FINAL_TISSUE_FRACTION = (
        FINAL_TISSUE_FRACTION
    )


    # Safety:
    # if script 17 also has a local copy of the constant,
    # patch it too.
    if hasattr(
        pipeline,
        "FINAL_TISSUE_FRACTION",
    ):

        pipeline.FINAL_TISSUE_FRACTION = (
            FINAL_TISSUE_FRACTION
        )


    if float(
        base.FINAL_TISSUE_FRACTION
    ) != FINAL_TISSUE_FRACTION:

        raise RuntimeError(
            "Failed to set "
            "FINAL_TISSUE_FRACTION=0.80"
        )


    return old_value


# ============================================================
# FREEZE RUN CONFIG
# ============================================================

def write_or_verify_run_config(
    pipeline,
):

    RUN_METADATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    script11_path = Path(
        pipeline.base.__file__
    ).resolve()


    payload = {

        "batch_name": (
            BATCH_NAME
        ),

        "batch_size": (
            BATCH_SIZE
        ),

        "expected_balance": {
            "LUAD": 15,
            "LUSC": 15,
        },

        "candidate_tissue_fraction": (
            EXPECTED_CANDIDATE_TISSUE_FRACTION
        ),

        "final_tissue_fraction": (
            FINAL_TISSUE_FRACTION
        ),

        "max_background_fraction": (
            EXPECTED_MAX_BACKGROUND_FRACTION
        ),

        "max_white_fraction": (
            EXPECTED_MAX_WHITE_FRACTION
        ),

        "tile_size": (
            EXPECTED_TILE_SIZE
        ),

        "target_tiles": (
            EXPECTED_TARGET_TILES
        ),

        "artifact_rules": (
            EXPECTED_ARTIFACT_RULES
        ),

        "script17_sha256": (
            sha256_file(
                PIPELINE17_PATH
            )
        ),

        "script11_sha256": (
            sha256_file(
                script11_path
            )
        ),

        "rule": (
            "This is a fresh independent validation. "
            "Do not tune preprocessing thresholds "
            "using its results."
        ),
    }


    if RUN_CONFIG_PATH.exists():

        existing = json.loads(
            RUN_CONFIG_PATH.read_text(
                encoding="utf-8"
            )
        )


        if existing != payload:

            raise RuntimeError(
                "Existing frozen run configuration "
                "does not match the current one:\n"
                f"{RUN_CONFIG_PATH}"
            )


        print(
            "Verified frozen run configuration:"
        )

        print(
            f"  {RUN_CONFIG_PATH}"
        )


        return


    RUN_CONFIG_PATH.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


    print(
        "Saved frozen run configuration:"
    )

    print(
        f"  {RUN_CONFIG_PATH}"
    )


# ============================================================
# POST-RUN CHECK
# ============================================================

def post_run_check():

    patient_qc_path = (
        ROOT
        / "data"
        / "metadata"
        / "production_v2_artifact_qc_candidate_v1"
        / "batches"
        / f"{BATCH_NAME}_patient_qc.csv"
    )


    batch_path = (
        ROOT
        / "data"
        / "manifests"
        / "production_v2_artifact_qc_candidate_v1"
        / "batches"
        / f"{BATCH_NAME}_patients.csv"
    )


    print()
    print("=" * 88)
    print(
        " FINAL INDEPENDENT30 POST-RUN CHECK"
    )
    print("=" * 88)


    if batch_path.exists():

        batch = pd.read_csv(
            batch_path
        )


        print(
            f"Frozen batch patients: "
            f"{len(batch)}"
        )


        if (
            "class_label"
            in batch.columns
        ):

            print()
            print(
                "Batch class distribution:"
            )

            print(
                batch[
                    "class_label"
                ]
                .value_counts()
                .to_string()
            )


        if len(
            batch
        ) != BATCH_SIZE:

            print()
            print(
                "WARNING: expected 30 patients."
            )


    else:

        print(
            "Batch manifest was not found:"
        )

        print(
            f"  {batch_path}"
        )


    if patient_qc_path.exists():

        results = pd.read_csv(
            patient_qc_path
        )


        if (
            "status"
            in results.columns
        ):

            print()
            print(
                "Status counts:"
            )

            print(
                results[
                    "status"
                ]
                .value_counts()
                .to_string()
            )


            terminal = (
                results[
                    "status"
                ]
                .isin(
                    [
                        "PASS",
                        "EXCLUDED_WSI_QC",
                    ]
                )
                .sum()
            )


            check_required = (
                (
                    results[
                        "status"
                    ]
                    ==
                    "CHECK_REQUIRED"
                )
                .sum()
            )


            print()
            print(
                f"Terminal results: "
                f"{int(terminal)}/"
                f"{BATCH_SIZE}"
            )

            print(
                f"CHECK_REQUIRED: "
                f"{int(check_required)}"
            )


        print()
        print(
            "Patient QC:"
        )

        print(
            f"  {patient_qc_path}"
        )


    else:

        print()
        print(
            "Patient QC file not found yet:"
        )

        print(
            f"  {patient_qc_path}"
        )


    print("=" * 88)


# ============================================================
# MAIN
# ============================================================

def main():

    pipeline = load_pipeline17()


    # --------------------------------------------------------
    # Verify everything except the tissue threshold
    # --------------------------------------------------------

    verify_frozen_rules(
        pipeline
    )


    # --------------------------------------------------------
    # Change ONLY 0.90 -> 0.80
    # --------------------------------------------------------

    old_tissue_threshold = (
        apply_tissue_080(
            pipeline
        )
    )


    # --------------------------------------------------------
    # Freeze reproducibility info
    # --------------------------------------------------------

    write_or_verify_run_config(
        pipeline
    )


    # --------------------------------------------------------
    # Pre-flight
    # --------------------------------------------------------

    print()
    print("=" * 88)
    print(
        " FINAL INDEPENDENT30 — TISSUE 0.80"
    )
    print("=" * 88)

    print(
        f"Batch:                    "
        f"{BATCH_NAME}"
    )

    print(
        f"Patients requested:       "
        f"{BATCH_SIZE}"
    )

    print(
        f"Previous tissue threshold:"
        f" {old_tissue_threshold:.2f}"
    )

    print(
        f"Effective tissue threshold:"
        f" {pipeline.base.FINAL_TISSUE_FRACTION:.2f}"
    )

    print(
        f"Candidate tissue:         "
        f"{pipeline.base.CANDIDATE_TISSUE_FRACTION:.2f}"
    )

    print(
        f"Background max:           "
        f"{pipeline.base.MAX_BACKGROUND_FRACTION:.2f}"
    )

    print(
        f"White max:                "
        f"{pipeline.base.MAX_WHITE_FRACTION:.2f}"
    )

    print(
        f"Tiles:                    "
        f"{pipeline.base.TARGET_TILES}"
        f" x "
        f"{pipeline.base.TILE_SIZE}"
        f"x"
        f"{pipeline.base.TILE_SIZE}"
    )

    print(
        "Artifact QC:              "
        "SENSITIVE_CANDIDATE_V1"
    )

    print(
        "Contact sheets:           True"
    )

    print(
        "Save rejected artifacts: True"
    )

    print(
        "Force reprocess:          False"
    )

    print("=" * 88)

    print()
    print(
        "IMPORTANT: this batch must contain "
        "only patients not used in previous "
        "pilot/smoke/validation runs."
    )

    print(
        "Script 17 should exclude previous "
        "terminal and historical cases automatically."
    )

    print()


    # --------------------------------------------------------
    # Run script 17 with a NEW batch name
    # --------------------------------------------------------

    old_argv = sys.argv[:]


    try:

        sys.argv = [
            str(
                PIPELINE17_PATH
            ),

            "--batch-name",
            BATCH_NAME,

            "--limit",
            str(
                BATCH_SIZE
            ),

            "--contact-sheets",

            "--save-rejected-artifacts",
        ]


        pipeline.main()


    finally:

        sys.argv = old_argv


    # --------------------------------------------------------
    # Check output
    # --------------------------------------------------------

    post_run_check()


if __name__ == "__main__":
    main()