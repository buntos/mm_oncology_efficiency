from pathlib import Path
import argparse
import datetime
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import time

import pandas as pd
from PIL import Image


# ============================================================
# PROJECT
# ============================================================

THIS_FILE = Path(__file__).resolve()
ROOT = THIS_FILE.parents[1]

SCRIPT17_PATH = (
    THIS_FILE.parent
    / "17_process_cohort_v2prod_artifact_qc.py"
)


# ============================================================
# FINAL PROTOCOL IDENTITY
# ============================================================

FINAL_NAMESPACE = (
    "production_final_v2_tissue080"
)

FINAL_PROTOCOL_NAME = (
    "V2_FINAL_HSV_64x256"
    "+TISSUE080"
    "+ARTIFACT_QC_V1"
)

FINAL_BATCH_NAME = (
    "final_full_cohort_v1"
)


# ============================================================
# EXPECTED COHORT
# ============================================================

EXPECTED_PATIENTS = 950

EXPECTED_LUAD = 474
EXPECTED_LUSC = 476


# ============================================================
# FINAL SCIENTIFIC CONFIGURATION
# ============================================================

TILE_SIZE = 256
TARGET_TILES = 64

CANDIDATE_TISSUE_FRACTION = 0.70
FINAL_TISSUE_FRACTION = 0.80

MAX_BACKGROUND_FRACTION = 0.10
MAX_WHITE_FRACTION = 0.10


# ============================================================
# FROZEN ARTIFACT QC
# ============================================================

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
# OPERATIONAL HARDENING
# ============================================================

DOWNLOAD_RETRIES = 3

DOWNLOAD_RETRY_DELAY_SECONDS = 5

# Large TCGA WSI can take >10 minutes on a slow connection.
# 30 minutes is intentionally generous, but prevents an
# indefinitely stuck gdc-client process.
DOWNLOAD_TIMEOUT_SECONDS = 30 * 60

MIN_FREE_SPACE_BUFFER_GB = 8.0


# ============================================================
# FINAL OUTPUT PATHS
# ============================================================

FINAL_TILE_ROOT = (
    ROOT
    / "data"
    / "tiles"
    / FINAL_NAMESPACE
)

FINAL_RNA_ROOT = (
    ROOT
    / "data"
    / "rna"
    / FINAL_NAMESPACE
)

FINAL_TEMP_GDC_ROOT = (
    ROOT
    / "data"
    / "temp_gdc_final_v2_tissue080"
)

FINAL_TEMP_TILE_ROOT = (
    ROOT
    / "data"
    / "temp_tiles_final_v2_tissue080"
)

FINAL_PATIENT_TILE_MANIFEST_ROOT = (
    ROOT
    / "data"
    / "manifests"
    / FINAL_NAMESPACE
    / "patient_tiles"
)

FINAL_BATCH_MANIFEST_ROOT = (
    ROOT
    / "data"
    / "manifests"
    / FINAL_NAMESPACE
    / "batches"
)

FINAL_METADATA_ROOT = (
    ROOT
    / "data"
    / "metadata"
    / FINAL_NAMESPACE
)

FINAL_BATCH_METADATA_ROOT = (
    FINAL_METADATA_ROOT
    / "batches"
)

FINAL_WSI_ATTEMPT_ROOT = (
    FINAL_METADATA_ROOT
    / "wsi_attempts"
)

FINAL_MASK_ROOT = (
    ROOT
    / "results"
    / "figures"
    / "production_final_v2_tissue080_masks"
)

FINAL_CONTACT_ROOT = (
    ROOT
    / "results"
    / "figures"
    / "production_final_v2_tissue080_contact_sheets"
)

FINAL_REJECTED_ARTIFACT_ROOT = (
    ROOT
    / "results"
    / "figures"
    / "production_final_v2_tissue080_rejected_artifacts"
)

FINAL_LOG_ROOT = (
    ROOT
    / "logs"
    / FINAL_NAMESPACE
)

FINAL_CONFIG_PATH = (
    FINAL_METADATA_ROOT
    / "frozen_protocol_configuration.json"
)

FINAL_BATCH_PATH = (
    FINAL_BATCH_MANIFEST_ROOT
    / f"{FINAL_BATCH_NAME}_patients.csv"
)


# ============================================================
# INPUT
# ============================================================

COHORT_FILE = (
    ROOT
    / "data"
    / "manifests"
    / "paired_cohort_stage2.csv"
)


# ============================================================
# HELPERS
# ============================================================

def sha256_file(
    path,
    chunk_size=8 * 1024 * 1024,
):

    h = hashlib.sha256()

    with open(
        path,
        "rb",
    ) as handle:

        while True:

            chunk = handle.read(
                chunk_size
            )

            if not chunk:
                break

            h.update(
                chunk
            )

    return h.hexdigest()


def md5_file(
    path,
    chunk_size=8 * 1024 * 1024,
):

    h = hashlib.md5()

    with open(
        path,
        "rb",
    ) as handle:

        while True:

            chunk = handle.read(
                chunk_size
            )

            if not chunk:
                break

            h.update(
                chunk
            )

    return h.hexdigest()


def free_disk_gb():

    usage = shutil.disk_usage(
        ROOT
    )

    return (
        usage.free
        / 1024**3
    )


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
            f"{actual} != {expected}"
        )


# ============================================================
# LOAD SCRIPT 17
# ============================================================

def load_pipeline17():

    if not SCRIPT17_PATH.exists():

        raise FileNotFoundError(
            "Script 17 not found:\n"
            f"{SCRIPT17_PATH}"
        )


    spec = (
        importlib.util.spec_from_file_location(
            "final_artifact_pipeline_base",
            SCRIPT17_PATH,
        )
    )


    if (
        spec is None
        or
        spec.loader is None
    ):

        raise RuntimeError(
            "Unable to create import specification "
            "for script 17."
        )


    module = (
        importlib.util.module_from_spec(
            spec
        )
    )


    spec.loader.exec_module(
        module
    )


    if not hasattr(
        module,
        "base",
    ):

        raise RuntimeError(
            "Script 17 does not expose "
            "the imported V2 base module."
        )


    return module


# ============================================================
# REWRITE SCRIPT 17 FINAL NAMESPACE
# ============================================================

PATH_TEXT_REPLACEMENTS = [
    (
        "production_v2_artifact_qc_candidate_v1",
        FINAL_NAMESPACE,
    ),

    (
        "production_v2_artifact_qc_contact_sheets",
        "production_final_v2_tissue080_contact_sheets",
    ),

    (
        "production_v2_artifact_qc_masks",
        "production_final_v2_tissue080_masks",
    ),

    (
        "production_v2_artifact_qc_rejected",
        "production_final_v2_tissue080_rejected_artifacts",
    ),

    (
        "temp_gdc_v2prod_artifact_qc",
        "temp_gdc_final_v2_tissue080",
    ),

    (
        "temp_tiles_v2prod_artifact_qc",
        "temp_tiles_final_v2_tissue080",
    ),
]


TEXT_REPLACEMENTS = [
    (
        "V2_PROD_HSV_64x256"
        "+ARTIFACT_QC_SENSITIVE_CANDIDATE_V1",

        FINAL_PROTOCOL_NAME,
    ),

    (
        "production_v2_artifact_qc_candidate_v1",
        FINAL_NAMESPACE,
    ),
]


def replace_path_text(
    value,
):

    text = str(
        value
    )

    for old, new in PATH_TEXT_REPLACEMENTS:

        text = text.replace(
            old,
            new,
        )

    return Path(
        text
    )


def replace_string_text(
    value,
):

    result = value

    for old, new in TEXT_REPLACEMENTS:

        result = result.replace(
            old,
            new,
        )

    return result


def patch_pipeline_namespace(
    pipeline,
):

    # Rewrite Path globals created by script 17.
    for name, value in list(
        vars(
            pipeline
        ).items()
    ):

        if isinstance(
            value,
            Path,
        ):

            patched = (
                replace_path_text(
                    value
                )
            )

            setattr(
                pipeline,
                name,
                patched,
            )


        elif isinstance(
            value,
            str,
        ):

            patched = (
                replace_string_text(
                    value
                )
            )

            setattr(
                pipeline,
                name,
                patched,
            )


    # Common protocol constant names.
    for name in [
        "PROTOCOL_NAME",
        "ARTIFACT_PROTOCOL_NAME",
    ]:

        if hasattr(
            pipeline,
            name,
        ):

            setattr(
                pipeline,
                name,
                FINAL_PROTOCOL_NAME,
            )


    for name in [
        "PROTOCOL_NAMESPACE",
        "PROTOCOL_SLUG",
        "OUTPUT_NAMESPACE",
    ]:

        if hasattr(
            pipeline,
            name,
        ):

            setattr(
                pipeline,
                name,
                FINAL_NAMESPACE,
            )


# ============================================================
# PATCH BASE OUTPUT PATHS
# ============================================================

def patch_base_paths(
    pipeline,
):

    base = pipeline.base


    # Input file constants from script 11 are intentionally
    # left unchanged.
    #
    # Only OUTPUT locations are replaced.

    base.TILE_ROOT = (
        FINAL_TILE_ROOT
    )

    base.RNA_ROOT = (
        FINAL_RNA_ROOT
    )

    base.TEMP_GDC_ROOT = (
        FINAL_TEMP_GDC_ROOT
    )

    base.TEMP_TILE_ROOT = (
        FINAL_TEMP_TILE_ROOT
    )

    base.PATIENT_TILE_MANIFEST_ROOT = (
        FINAL_PATIENT_TILE_MANIFEST_ROOT
    )

    base.WSI_ATTEMPT_ROOT = (
        FINAL_WSI_ATTEMPT_ROOT
    )

    base.MASK_ROOT = (
        FINAL_MASK_ROOT
    )

    base.CONTACT_ROOT = (
        FINAL_CONTACT_ROOT
    )

    base.LOG_ROOT = (
        FINAL_LOG_ROOT
    )


    # In script 11 these variables are also used for
    # batch/output CSVs.
    #
    # Inputs such as COHORT_FILE and WSI_CANDIDATES_FILE are
    # already absolute constants and are therefore unaffected.

    base.MANIFEST_DIR = (
        FINAL_BATCH_MANIFEST_ROOT
    )

    base.METADATA_DIR = (
        FINAL_BATCH_METADATA_ROOT
    )


    for directory in [
        FINAL_TILE_ROOT,
        FINAL_RNA_ROOT,
        FINAL_TEMP_GDC_ROOT,
        FINAL_TEMP_TILE_ROOT,
        FINAL_PATIENT_TILE_MANIFEST_ROOT,
        FINAL_BATCH_MANIFEST_ROOT,
        FINAL_METADATA_ROOT,
        FINAL_BATCH_METADATA_ROOT,
        FINAL_WSI_ATTEMPT_ROOT,
        FINAL_MASK_ROOT,
        FINAL_CONTACT_ROOT,
        FINAL_REJECTED_ARTIFACT_ROOT,
        FINAL_LOG_ROOT,
    ]:

        directory.mkdir(
            parents=True,
            exist_ok=True,
        )


# ============================================================
# PATCH FINAL SCIENTIFIC THRESHOLDS
# ============================================================

def patch_final_thresholds(
    pipeline,
):

    base = pipeline.base


    # Verify all unchanged V2 parameters BEFORE patching
    # final tissue.

    assert_float_equal(
        "CANDIDATE_TISSUE_FRACTION",
        base.CANDIDATE_TISSUE_FRACTION,
        CANDIDATE_TISSUE_FRACTION,
    )

    assert_float_equal(
        "MAX_BACKGROUND_FRACTION",
        base.MAX_BACKGROUND_FRACTION,
        MAX_BACKGROUND_FRACTION,
    )

    assert_float_equal(
        "MAX_WHITE_FRACTION",
        base.MAX_WHITE_FRACTION,
        MAX_WHITE_FRACTION,
    )


    if int(
        base.TILE_SIZE
    ) != TILE_SIZE:

        raise RuntimeError(
            "Unexpected TILE_SIZE: "
            f"{base.TILE_SIZE}"
        )


    if int(
        base.TARGET_TILES
    ) != TARGET_TILES:

        raise RuntimeError(
            "Unexpected TARGET_TILES: "
            f"{base.TARGET_TILES}"
        )


    # Final scientific change selected by our audits:
    #
    # 0.90 -> 0.80

    base.FINAL_TISSUE_FRACTION = (
        FINAL_TISSUE_FRACTION
    )


    # Script 17 may also expose a local copy.
    if hasattr(
        pipeline,
        "FINAL_TISSUE_FRACTION",
    ):

        pipeline.FINAL_TISSUE_FRACTION = (
            FINAL_TISSUE_FRACTION
        )


    base.PROTOCOL_NAME = (
        FINAL_PROTOCOL_NAME
    )


# ============================================================
# VERIFY FROZEN ARTIFACT RULES
# ============================================================

def verify_artifact_rules(
    pipeline,
):

    if not hasattr(
        pipeline,
        "ARTIFACT_RULES",
    ):

        raise RuntimeError(
            "ARTIFACT_RULES not found "
            "in script 17."
        )


    rules = (
        pipeline.ARTIFACT_RULES
    )


    for key, expected in (
        EXPECTED_ARTIFACT_RULES.items()
    ):

        if key not in rules:

            raise RuntimeError(
                "Missing artifact rule: "
                f"{key}"
            )


        assert_float_equal(
            f"ARTIFACT_RULES[{key}]",
            rules[
                key
            ],
            expected,
        )


# ============================================================
# OPERATIONAL FIX #1
#
# STRIP PNG ICC / iCCP METADATA
# ============================================================

ORIGINAL_PIL_SAVE = (
    Image.Image.save
)


def install_clean_png_save_patch():

    def clean_png_save(
        self,
        fp,
        format=None,
        **params,
    ):

        effective_format = (
            format
        )


        if effective_format is None:

            try:

                suffix = (
                    Path(
                        fp
                    )
                    .suffix
                    .lower()
                )

            except (
                TypeError,
                ValueError,
            ):

                suffix = ""


            if suffix == ".png":

                effective_format = (
                    "PNG"
                )


        if (
            effective_format
            is not None
            and
            str(
                effective_format
            ).upper()
            ==
            "PNG"
        ):

            # Pixel-identical copy.
            #
            # No ICC -> sRGB transform is performed.
            # We remove ancillary metadata only.

            clean = self.copy()

            clean.info.clear()


            clean_params = dict(
                params
            )

            clean_params.pop(
                "icc_profile",
                None,
            )

            clean_params.pop(
                "pnginfo",
                None,
            )

            clean_params.pop(
                "exif",
                None,
            )


            try:

                return ORIGINAL_PIL_SAVE(
                    clean,
                    fp,
                    format=format,
                    **clean_params,
                )

            finally:

                clean.close()


        return ORIGINAL_PIL_SAVE(
            self,
            fp,
            format=format,
            **params,
        )


    Image.Image.save = (
        clean_png_save
    )


# ============================================================
# OPERATIONAL FIX #2
#
# CONTACT SHEET MUST NEVER FAIL THE PATIENT
# ============================================================

def install_safe_contact_sheet_patch(
    pipeline,
):

    base = pipeline.base


    def safe_create_contact_sheet(
        tile_paths,
        out_path,
    ):

        if (
            len(
                tile_paths
            )
            !=
            TARGET_TILES
        ):

            return


        sheet = None


        try:

            out_path = Path(
                out_path
            )

            out_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )


            sheet = Image.new(
                "RGB",
                (
                    8 * TILE_SIZE,
                    8 * TILE_SIZE,
                ),
                "white",
            )


            for i, path in enumerate(
                tile_paths
            ):

                with Image.open(
                    path
                ) as source:

                    image = (
                        source.convert(
                            "RGB"
                        )
                    )


                    try:

                        x = (
                            i % 8
                        ) * TILE_SIZE

                        y = (
                            i // 8
                        ) * TILE_SIZE


                        sheet.paste(
                            image,
                            (
                                x,
                                y,
                            ),
                        )

                    finally:

                        image.close()


            # JPEG has no need to inherit source tile metadata.
            ORIGINAL_PIL_SAVE(
                sheet,
                out_path,
                format="JPEG",
                quality=94,
            )


        except Exception as exc:

            # Diagnostic visualization is NOT allowed to
            # change dataset inclusion or crash a 950-patient
            # production run.

            print()

            print(
                "WARNING: contact sheet creation failed."
            )

            print(
                "This is NON-FATAL."
            )

            print(
                f"  {type(exc).__name__}: "
                f"{exc}"
            )


        finally:

            if sheet is not None:

                sheet.close()


    base.create_contact_sheet = (
        safe_create_contact_sheet
    )


    # Defensive patch if script 17 itself imported/aliased it.
    if hasattr(
        pipeline,
        "create_contact_sheet",
    ):

        pipeline.create_contact_sheet = (
            safe_create_contact_sheet
        )


# ============================================================
# OPERATIONAL FIX #3
#
# GDC DOWNLOAD TIMEOUT + RESUME + MD5
# ============================================================

def install_download_timeout_patch(
    pipeline,
):

    base = pipeline.base


    def hardened_download_gdc_file(
        file_id,
        file_name,
        expected_size,
        expected_md5,
        patient_temp_dir,
    ):

        required_gb = (
            int(
                expected_size
            )
            / 1024**3
        )


        if (
            base.free_disk_gb()
            <
            required_gb
            +
            MIN_FREE_SPACE_BUFFER_GB
        ):

            raise RuntimeError(
                "Insufficient free disk space. "
                f"Need file {required_gb:.2f} GB "
                f"+ {MIN_FREE_SPACE_BUFFER_GB:.1f} GB buffer."
            )


        existing = (
            base.locate_downloaded_file(
                patient_temp_dir,
                file_id,
                file_name,
            )
        )


        ok, reason = (
            base.verify_downloaded_file(
                existing,
                expected_size,
                expected_md5,
            )
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
                "      GDC download attempt "
                f"{attempt}/{DOWNLOAD_RETRIES}"
            )


            command = [
                str(
                    base.GDC_CLIENT
                ),

                "download",

                str(
                    file_id
                ),

                "-d",

                str(
                    patient_temp_dir
                ),

                "--no-related-files",
                "--no-annotations",
            ]


            timed_out = False


            try:

                result = subprocess.run(
                    command,
                    cwd=str(
                        ROOT
                    ),
                    timeout=(
                        DOWNLOAD_TIMEOUT_SECONDS
                    ),
                )


                if (
                    result.returncode
                    !=
                    0
                ):

                    print(
                        "      gdc-client returned "
                        f"exit code "
                        f"{result.returncode}"
                    )


            except subprocess.TimeoutExpired:

                timed_out = True

                print(
                    "      DOWNLOAD TIMEOUT after "
                    f"{DOWNLOAD_TIMEOUT_SECONDS // 60} min"
                )

                print(
                    "      gdc-client was terminated; "
                    "partial data will be preserved "
                    "for resume."
                )


            except Exception as exc:

                print(
                    "      gdc-client execution error: "
                    f"{repr(exc)}"
                )


            path = (
                base.locate_downloaded_file(
                    patient_temp_dir,
                    file_id,
                    file_name,
                )
            )


            ok, reason = (
                base.verify_downloaded_file(
                    path,
                    expected_size,
                    expected_md5,
                )
            )


            # The download may technically finish immediately
            # before timeout; always trust size+MD5.

            if ok:

                print(
                    "      download + MD5: PASS"
                )

                return path


            print(
                "      verification failed: "
                f"{reason}"
            )


            uuid_dir = (
                Path(
                    patient_temp_dir
                )
                / str(
                    file_id
                )
            )


            partials = []

            if uuid_dir.exists():

                partials = list(
                    uuid_dir.glob(
                        "*.partial"
                    )
                )


            if partials:

                print(
                    "      partial file detected; "
                    "next attempt will resume"
                )


            else:

                # If this was a timeout we avoid deleting
                # an incomplete UUID directory unnecessarily.
                #
                # Otherwise invalid complete downloads are
                # removed before retry.

                if (
                    not timed_out
                    and
                    path is not None
                ):

                    base.remove_uuid_download(
                        patient_temp_dir,
                        file_id,
                    )


            if (
                attempt
                <
                DOWNLOAD_RETRIES
            ):

                print(
                    "      retrying in "
                    f"{DOWNLOAD_RETRY_DELAY_SECONDS}s"
                )

                time.sleep(
                    DOWNLOAD_RETRY_DELAY_SECONDS
                )


        raise RuntimeError(
            "GDC download failed after retries: "
            f"{file_id} ({file_name})"
        )


    base.download_gdc_file = (
        hardened_download_gdc_file
    )


    # Defensive patch for a direct alias in script 17.
    if hasattr(
        pipeline,
        "download_gdc_file",
    ):

        pipeline.download_gdc_file = (
            hardened_download_gdc_file
        )


# ============================================================
# VERIFY / FREEZE FULL COHORT
# ============================================================

def prepare_full_cohort_manifest():

    if not COHORT_FILE.exists():

        raise FileNotFoundError(
            "Stage-2 cohort not found:\n"
            f"{COHORT_FILE}"
        )


    cohort = pd.read_csv(
        COHORT_FILE
    )


    required_columns = {
        "case_submitter_id",
        "class_label",
    }


    missing_columns = (
        required_columns
        -
        set(
            cohort.columns
        )
    )


    if missing_columns:

        raise RuntimeError(
            "Cohort missing required columns: "
            f"{sorted(missing_columns)}"
        )


    if (
        cohort[
            "case_submitter_id"
        ].duplicated().any()
    ):

        duplicates = (
            cohort.loc[
                cohort[
                    "case_submitter_id"
                ].duplicated(
                    keep=False
                ),
                "case_submitter_id",
            ]
            .tolist()
        )

        raise RuntimeError(
            "Duplicate patient IDs in Stage-2 cohort: "
            f"{duplicates[:20]}"
        )


    luad = int(
        (
            cohort[
                "class_label"
            ]
            ==
            "LUAD"
        ).sum()
    )


    lusc = int(
        (
            cohort[
                "class_label"
            ]
            ==
            "LUSC"
        ).sum()
    )


    if len(
        cohort
    ) != EXPECTED_PATIENTS:

        raise RuntimeError(
            "Unexpected Stage-2 cohort size: "
            f"{len(cohort)} != {EXPECTED_PATIENTS}"
        )


    if luad != EXPECTED_LUAD:

        raise RuntimeError(
            "Unexpected LUAD count: "
            f"{luad} != {EXPECTED_LUAD}"
        )


    if lusc != EXPECTED_LUSC:

        raise RuntimeError(
            "Unexpected LUSC count: "
            f"{lusc} != {EXPECTED_LUSC}"
        )


    batch = (
        cohort
        .sort_values(
            "case_submitter_id"
        )
        .reset_index(
            drop=True
        )
    )


    FINAL_BATCH_MANIFEST_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )


    if FINAL_BATCH_PATH.exists():

        existing = pd.read_csv(
            FINAL_BATCH_PATH
        )


        existing_ids = (
            existing[
                "case_submitter_id"
            ]
            .astype(
                str
            )
            .tolist()
        )


        expected_ids = (
            batch[
                "case_submitter_id"
            ]
            .astype(
                str
            )
            .tolist()
        )


        if existing_ids != expected_ids:

            raise RuntimeError(
                "Existing FINAL frozen batch differs "
                "from Stage-2 cohort."
            )


        print(
            "Verified existing FINAL frozen batch:"
        )

        print(
            f"  {FINAL_BATCH_PATH}"
        )


    else:

        batch.to_csv(
            FINAL_BATCH_PATH,
            index=False,
        )


        print(
            "Created FINAL frozen full-cohort batch:"
        )

        print(
            f"  {FINAL_BATCH_PATH}"
        )


    return (
        batch,
        luad,
        lusc,
    )


# ============================================================
# FREEZE FINAL CONFIGURATION
# ============================================================

def build_frozen_config(
    pipeline,
    cohort,
    luad,
    lusc,
):

    base = pipeline.base


    script11_path = Path(
        base.__file__
    ).resolve()


    config = {
        "protocol_name": (
            FINAL_PROTOCOL_NAME
        ),

        "namespace": (
            FINAL_NAMESPACE
        ),

        "batch_name": (
            FINAL_BATCH_NAME
        ),

        "cohort": {
            "patients": int(
                len(
                    cohort
                )
            ),

            "LUAD": int(
                luad
            ),

            "LUSC": int(
                lusc
            ),

            "stage2_cohort_file": str(
                COHORT_FILE.relative_to(
                    ROOT
                )
            ),

            "stage2_cohort_sha256": (
                sha256_file(
                    COHORT_FILE
                )
            ),
        },

        "tiling": {
            "tile_size": (
                TILE_SIZE
            ),

            "target_tiles": (
                TARGET_TILES
            ),

            "candidate_tissue_fraction": (
                CANDIDATE_TISSUE_FRACTION
            ),

            "final_tissue_fraction": (
                FINAL_TISSUE_FRACTION
            ),

            "max_background_fraction": (
                MAX_BACKGROUND_FRACTION
            ),

            "max_white_fraction": (
                MAX_WHITE_FRACTION
            ),

            "selection": (
                "deterministic candidate order "
                "per WSI"
            ),

            "diagnostic_slide_fallback": (
                "DX1 -> DX2 -> DX3/... "
                "until 64 valid tiles"
            ),
        },

        "artifact_qc": {
            "name": (
                "SENSITIVE_CANDIDATE_V1"
            ),

            "rules": (
                EXPECTED_ARTIFACT_RULES
            ),
        },

        "operational_hardening": {
            "png_metadata": (
                "Strip PNG ancillary metadata "
                "including ICC/iCCP without "
                "ICC-to-sRGB color conversion."
            ),

            "pixel_values_changed_by_png_fix": (
                False
            ),

            "contact_sheet_failure": (
                "NON_FATAL"
            ),

            "gdc_download_timeout_seconds": (
                DOWNLOAD_TIMEOUT_SECONDS
            ),

            "gdc_download_retries": (
                DOWNLOAD_RETRIES
            ),

            "gdc_resume_partial": (
                True
            ),

            "download_validation": (
                "expected size + MD5"
            ),

            "delete_raw_wsi_after_processing": (
                True
            ),

            "save_rejected_artifact_images": (
                False
            ),
        },

        "code": {
            "script24": str(
                THIS_FILE.relative_to(
                    ROOT
                )
            ),

            "script24_sha256": (
                sha256_file(
                    THIS_FILE
                )
            ),

            "script17": str(
                SCRIPT17_PATH.relative_to(
                    ROOT
                )
            ),

            "script17_sha256": (
                sha256_file(
                    SCRIPT17_PATH
                )
            ),

            "script11": str(
                script11_path.relative_to(
                    ROOT
                )
            ),

            "script11_sha256": (
                sha256_file(
                    script11_path
                )
            ),
        },

        "scientific_freeze": (
            "No preprocessing threshold may be "
            "changed after final full-cohort "
            "processing begins."
        ),
    }


    return config


def write_or_verify_frozen_config(
    config,
):

    FINAL_METADATA_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )


    if FINAL_CONFIG_PATH.exists():

        existing = json.loads(
            FINAL_CONFIG_PATH.read_text(
                encoding="utf-8"
            )
        )


        if existing != config:

            print()
            print(
                "Existing FINAL frozen configuration "
                "does not match this script."
            )

            print()
            print(
                "Existing:"
            )

            print(
                json.dumps(
                    existing,
                    indent=2,
                    ensure_ascii=False,
                )
            )

            print()
            print(
                "Current:"
            )

            print(
                json.dumps(
                    config,
                    indent=2,
                    ensure_ascii=False,
                )
            )


            raise RuntimeError(
                "FINAL protocol drift detected. "
                "Refusing to continue."
            )


        print(
            "Verified frozen FINAL protocol:"
        )

        print(
            f"  {FINAL_CONFIG_PATH}"
        )


    else:

        FINAL_CONFIG_PATH.write_text(
            json.dumps(
                config,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )


        print(
            "Saved frozen FINAL protocol:"
        )

        print(
            f"  {FINAL_CONFIG_PATH}"
        )


# ============================================================
# FINAL PRE-FLIGHT
# ============================================================

def print_preflight(
    cohort,
    luad,
    lusc,
):

    print()
    print(
        "=" * 92
    )

    print(
        " FINAL FULL-COHORT PREPROCESSING — PREFLIGHT"
    )

    print(
        "=" * 92
    )

    print(
        f"Protocol:                 "
        f"{FINAL_PROTOCOL_NAME}"
    )

    print(
        f"Namespace:                "
        f"{FINAL_NAMESPACE}"
    )

    print(
        f"Batch:                    "
        f"{FINAL_BATCH_NAME}"
    )

    print(
        f"Patients:                 "
        f"{len(cohort)}"
    )

    print(
        f"LUAD / LUSC:              "
        f"{luad} / {lusc}"
    )

    print()

    print(
        f"Tile size:                "
        f"{TILE_SIZE}x{TILE_SIZE}"
    )

    print(
        f"Tiles per patient:        "
        f"{TARGET_TILES}"
    )

    print(
        f"Candidate tissue:         "
        f">= {CANDIDATE_TISSUE_FRACTION:.2f}"
    )

    print(
        f"FINAL tissue:             "
        f">= {FINAL_TISSUE_FRACTION:.2f}"
    )

    print(
        f"Background:               "
        f"<= {MAX_BACKGROUND_FRACTION:.2f}"
    )

    print(
        f"White:                    "
        f"<= {MAX_WHITE_FRACTION:.2f}"
    )

    print(
        "Artifact QC:              "
        "SENSITIVE_CANDIDATE_V1"
    )

    print()

    print(
        "PNG ICC/iCCP metadata:    STRIPPED"
    )

    print(
        "ICC -> sRGB conversion:   NO"
    )

    print(
        "Contact-sheet error:      NON-FATAL"
    )

    print(
        "GDC timeout:              "
        f"{DOWNLOAD_TIMEOUT_SECONDS // 60} min / attempt"
    )

    print(
        "GDC retries:              "
        f"{DOWNLOAD_RETRIES}"
    )

    print(
        "Partial resume:           YES"
    )

    print(
        "Size + MD5 validation:    YES"
    )

    print(
        "Delete WSI after QC:      YES"
    )

    print()

    print(
        f"Free disk:                "
        f"{free_disk_gb():.2f} GB"
    )

    print()

    print(
        "Final tiles:"
    )

    print(
        f"  {FINAL_TILE_ROOT}"
    )

    print(
        "Final RNA:"
    )

    print(
        f"  {FINAL_RNA_ROOT}"
    )

    print(
        "Frozen batch:"
    )

    print(
        f"  {FINAL_BATCH_PATH}"
    )

    print(
        "Frozen protocol:"
    )

    print(
        f"  {FINAL_CONFIG_PATH}"
    )

    print(
        "=" * 92
    )


# ============================================================
# POST-RUN RESULT DISCOVERY
# ============================================================

def find_result_file(
    filename,
):

    candidates = list(
        (
            ROOT
            / "data"
            / "metadata"
        ).rglob(
            filename
        )
    )


    if not candidates:
        return None


    final_candidates = [
        path
        for path in candidates
        if FINAL_NAMESPACE
        in str(
            path
        )
    ]


    if final_candidates:

        return sorted(
            final_candidates,
            key=lambda item: str(
                item
            ),
        )[0]


    return sorted(
        candidates,
        key=lambda item: str(
            item
        ),
    )[0]


# ============================================================
# POST-RUN SUMMARY
# ============================================================

def post_run_summary():

    print()
    print(
        "=" * 92
    )

    print(
        " FINAL FULL-COHORT SUMMARY"
    )

    print(
        "=" * 92
    )


    result_name = (
        f"{FINAL_BATCH_NAME}"
        "_patient_qc.csv"
    )


    result_path = (
        find_result_file(
            result_name
        )
    )


    if (
        result_path is None
        or
        not result_path.exists()
    ):

        print(
            "Patient QC file was not found."
        )

        print(
            "The run may have terminated before "
            "writing the first result."
        )

        print(
            "=" * 92
        )

        return


    results = pd.read_csv(
        result_path
    )


    print(
        "Patient QC:"
    )

    print(
        f"  {result_path}"
    )

    print()


    if (
        "status"
        not in results.columns
    ):

        print(
            "status column missing."
        )

        return


    counts = (
        results[
            "status"
        ]
        .value_counts(
            dropna=False
        )
    )


    print(
        counts.to_string()
    )

    print()


    pass_count = int(
        (
            results[
                "status"
            ]
            ==
            "PASS"
        ).sum()
    )


    excluded_count = int(
        (
            results[
                "status"
            ]
            ==
            "EXCLUDED_WSI_QC"
        ).sum()
    )


    check_count = int(
        (
            results[
                "status"
            ]
            ==
            "CHECK_REQUIRED"
        ).sum()
    )


    terminal_count = (
        pass_count
        +
        excluded_count
    )


    print(
        f"PASS:                "
        f"{pass_count}"
    )

    print(
        f"EXCLUDED_WSI_QC:     "
        f"{excluded_count}"
    )

    print(
        f"CHECK_REQUIRED:       "
        f"{check_count}"
    )

    print(
        f"Terminal:             "
        f"{terminal_count}/{EXPECTED_PATIENTS}"
    )

    print(
        f"Recorded rows:        "
        f"{len(results)}/{EXPECTED_PATIENTS}"
    )

    print(
        f"Free disk:            "
        f"{free_disk_gb():.2f} GB"
    )


    if check_count > 0:

        print()

        print(
            "CHECK_REQUIRED patients remain retryable."
        )

        print(
            "Rerun the exact same script after "
            "transient technical issues are resolved."
        )


    if (
        terminal_count
        ==
        EXPECTED_PATIENTS
        and
        check_count
        ==
        0
    ):

        print()

        print(
            "FINAL PREPROCESSING COMPLETE."
        )


    print(
        "=" * 92
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Final full TCGA LUAD/LUSC multimodal "
            "preprocessing run."
        )
    )


    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help=(
            "Validate/freeze configuration and cohort "
            "without downloading or processing WSI."
        ),
    )


    args = parser.parse_args()


    # ========================================================
    # LOAD PROVEN ARTIFACT-QC PIPELINE
    # ========================================================

    pipeline = (
        load_pipeline17()
    )


    # ========================================================
    # FINAL NAMESPACE
    # ========================================================

    patch_pipeline_namespace(
        pipeline
    )

    patch_base_paths(
        pipeline
    )


    # ========================================================
    # FINAL SCIENTIFIC CONFIGURATION
    # ========================================================

    patch_final_thresholds(
        pipeline
    )

    verify_artifact_rules(
        pipeline
    )


    # ========================================================
    # OPERATIONAL HARDENING
    # ========================================================

    install_clean_png_save_patch()

    install_safe_contact_sheet_patch(
        pipeline
    )

    install_download_timeout_patch(
        pipeline
    )


    # ========================================================
    # FREEZE ALL 950 PATIENTS
    # ========================================================

    (
        cohort,
        luad,
        lusc,
    ) = prepare_full_cohort_manifest()


    # ========================================================
    # FREEZE CONFIG
    # ========================================================

    config = build_frozen_config(
        pipeline,
        cohort,
        luad,
        lusc,
    )


    write_or_verify_frozen_config(
        config
    )


    # ========================================================
    # DISPLAY FINAL SETTINGS
    # ========================================================

    print_preflight(
        cohort,
        luad,
        lusc,
    )


    if args.preflight_only:

        print()

        print(
            "PREFLIGHT ONLY — "
            "NO WSI/RNA PROCESSING STARTED."
        )

        return


    # ========================================================
    # RUN FULL COHORT
    # ========================================================

    print()
    print(
        "=" * 92
    )

    print(
        " STARTING FINAL FULL-COHORT RUN"
    )

    print(
        "=" * 92
    )

    print()

    print(
        "Scientific preprocessing is now FROZEN."
    )

    print(
        "Do not change thresholds based on "
        "individual full-cohort failures."
    )

    print()


    old_argv = (
        sys.argv[:]
    )


    try:

        # Script 17 was explicitly designed so --all
        # processes the complete cohort rather than the
        # historical independent/dev subset.
        #
        # The batch itself has also already been frozen above
        # as all 950 Stage-2 patients.

        sys.argv = [
            str(
                SCRIPT17_PATH
            ),

            "--batch-name",
            FINAL_BATCH_NAME,

            "--all",

            "--contact-sheets",
        ]


        pipeline.main()


    finally:

        sys.argv = (
            old_argv
        )


    # ========================================================
    # SUMMARY
    # ========================================================

    post_run_summary()


if __name__ == "__main__":
    main()