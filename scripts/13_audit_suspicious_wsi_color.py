from pathlib import Path
import argparse
import hashlib
import json
import shutil
import subprocess
import time

import numpy as np
import pandas as pd
from PIL import Image, ImageCms, ImageDraw, ImageFont
import openslide


# ============================================================
# ROOT / INPUTS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

MANIFEST_DIR = ROOT / "data" / "manifests"
METADATA_DIR = ROOT / "data" / "metadata"

WSI_CANDIDATES_FILE = (
    MANIFEST_DIR
    / "wsi_primary_diagnostic_candidates.csv"
)

PATIENT_TILE_MANIFEST_ROOT = (
    MANIFEST_DIR
    / "production_v2"
    / "patient_tiles"
)

GDC_CLIENT = (
    ROOT
    / "tools"
    / "gdc-client.exe"
)


# ============================================================
# DEFAULT SUSPICIOUS CASES
# ============================================================

DEFAULT_CASES = [
    "TCGA-86-8056",
    "TCGA-86-8278",
    "TCGA-78-8660",
]


# ============================================================
# OUTPUTS
# ============================================================

TEMP_ROOT = (
    ROOT
    / "data"
    / "temp_gdc_color_audit"
)

OUTPUT_ROOT = (
    ROOT
    / "results"
    / "figures"
    / "suspicious_wsi_color_audit"
)

AUDIT_METADATA_ROOT = (
    METADATA_DIR
    / "suspicious_wsi_color_audit"
)


for directory in [
    TEMP_ROOT,
    OUTPUT_ROOT,
    AUDIT_METADATA_ROOT,
]:
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# CONFIG
# ============================================================

TILE_SIZE = 256

DOWNLOAD_RETRIES = 5
DOWNLOAD_RETRY_DELAY_SECONDS = 5

CONTACT_COLUMNS = 8
CONTACT_ROWS = 8

LABEL_HEIGHT = 24

OVERVIEW_CONTACT_SIZE = 640

MIN_FREE_SPACE_BUFFER_GB = 8.0


# ============================================================
# BASIC HELPERS
# ============================================================

def md5_file(
    path,
    chunk_size=8 * 1024 * 1024,
):

    digest = hashlib.md5()

    with open(
        path,
        "rb",
    ) as file:

        while True:

            chunk = file.read(
                chunk_size
            )

            if not chunk:
                break

            digest.update(
                chunk
            )

    return digest.hexdigest()


def free_disk_gb():

    usage = shutil.disk_usage(
        ROOT
    )

    return (
        usage.free
        / 1024**3
    )


def remove_download(
    patient_temp_dir,
    file_id,
):

    directory = (
        patient_temp_dir
        / str(file_id)
    )

    if directory.exists():

        shutil.rmtree(
            directory,
            ignore_errors=True,
        )


def locate_download(
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


def verify_file(
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
        md5_file(
            path
        )
        .lower()
    )


    if (
        actual_md5
        !=
        str(
            expected_md5
        ).lower()
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
# DOWNLOAD
# ============================================================

def download_wsi(
    candidate,
    patient_temp_dir,
):

    file_id = str(
        candidate[
            "file_id"
        ]
    )

    file_name = str(
        candidate[
            "file_name"
        ]
    )

    expected_size = int(
        candidate[
            "file_size_bytes"
        ]
    )

    expected_md5 = str(
        candidate[
            "md5sum"
        ]
    )


    required_gb = (
        expected_size
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
            f"WSI requires {required_gb:.2f} GB "
            f"+ {MIN_FREE_SPACE_BUFFER_GB:.1f} GB buffer."
        )


    existing = locate_download(
        patient_temp_dir,
        file_id,
        file_name,
    )


    ok, reason = verify_file(
        existing,
        expected_size,
        expected_md5,
    )


    if ok:

        print(
            "    existing WSI verified"
        )

        return existing


    for attempt in range(
        1,
        DOWNLOAD_RETRIES + 1,
    ):

        print(
            f"    GDC download "
            f"{attempt}/{DOWNLOAD_RETRIES}"
        )


        command = [
            str(
                GDC_CLIENT
            ),

            "download",

            file_id,

            "-d",
            str(
                patient_temp_dir
            ),

            "--no-related-files",
            "--no-annotations",
        ]


        subprocess.run(
            command,
            cwd=str(
                ROOT
            ),
        )


        downloaded = locate_download(
            patient_temp_dir,
            file_id,
            file_name,
        )


        ok, reason = verify_file(
            downloaded,
            expected_size,
            expected_md5,
        )


        if ok:

            print(
                "    download + MD5: PASS"
            )

            return downloaded


        print(
            f"    verification failed: "
            f"{reason}"
        )


        partial_dir = (
            patient_temp_dir
            / file_id
        )


        partials = []

        if partial_dir.exists():

            partials = list(
                partial_dir.glob(
                    "*.partial"
                )
            )


        if partials:

            print(
                "    partial detected; "
                "next attempt will resume"
            )


        elif (
            downloaded is not None
        ):

            remove_download(
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
        "WSI download failed after retries: "
        f"{file_id}"
    )


# ============================================================
# FONT
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


# ============================================================
# IMAGE METRICS
# ============================================================

def image_difference_metrics(
    image_a,
    image_b,
):

    a = np.asarray(
        image_a.convert(
            "RGB"
        ),
        dtype=np.float32,
    )


    b = np.asarray(
        image_b.convert(
            "RGB"
        ),
        dtype=np.float32,
    )


    delta = np.abs(
        a - b
    )


    per_pixel = np.mean(
        delta,
        axis=2,
    )


    return {
        "rgb_mae": float(
            delta.mean()
        ),

        "rgb_pixel_p50": float(
            np.percentile(
                per_pixel,
                50,
            )
        ),

        "rgb_pixel_p95": float(
            np.percentile(
                per_pixel,
                95,
            )
        ),

        "rgb_pixel_p99": float(
            np.percentile(
                per_pixel,
                99,
            )
        ),

        "rgb_max_abs": float(
            delta.max()
        ),
    }


def image_rgb_summary(
    image,
):

    rgb = np.asarray(
        image.convert(
            "RGB"
        ),
        dtype=np.float32,
    )


    return {
        "mean_r": float(
            rgb[:, :, 0].mean()
        ),

        "mean_g": float(
            rgb[:, :, 1].mean()
        ),

        "mean_b": float(
            rgb[:, :, 2].mean()
        ),

        "std_r": float(
            rgb[:, :, 0].std()
        ),

        "std_g": float(
            rgb[:, :, 1].std()
        ),

        "std_b": float(
            rgb[:, :, 2].std()
        ),
    }


# ============================================================
# ICC
# ============================================================

def get_profile_text(
    profile,
):

    if profile is None:
        return None


    parts = []


    try:

        name = (
            ImageCms
            .getProfileName(
                profile
            )
            .strip()
        )

        if name:
            parts.append(
                name
            )

    except Exception:

        pass


    try:

        description = (
            ImageCms
            .getProfileDescription(
                profile
            )
            .strip()
        )

        if (
            description
            and
            description
            not in parts
        ):

            parts.append(
                description
            )

    except Exception:

        pass


    if not parts:

        return (
            "ICC profile present"
        )


    return " | ".join(
        parts
    )


def build_icc_transform(
    slide,
):

    if (
        slide.color_profile
        is None
    ):

        return (
            None,
            None,
            None,
        )


    srgb_profile = (
        ImageCms.createProfile(
            "sRGB"
        )
    )


    intent = (
        ImageCms.getDefaultIntent(
            slide.color_profile
        )
    )


    transform = (
        ImageCms.buildTransform(
            slide.color_profile,
            srgb_profile,
            "RGBA",
            "RGBA",
            intent,
            0,
        )
    )


    profile_text = (
        get_profile_text(
            slide.color_profile
        )
    )


    return (
        transform,
        intent,
        profile_text,
    )


def apply_icc_to_region(
    region_rgba,
    transform,
):

    corrected = (
        region_rgba.copy()
    )


    ImageCms.applyTransform(
        corrected,
        transform,
        True,
    )


    return corrected.convert(
        "RGB"
    )


# ============================================================
# CONTACT SHEET
# ============================================================

def create_numbered_contact_sheet(
    tile_records,
    path,
    title,
):

    font = load_font(
        17
    )

    title_font = load_font(
        24
    )


    width = (
        CONTACT_COLUMNS
        * TILE_SIZE
    )


    height = (
        52
        +
        CONTACT_ROWS
        * (
            TILE_SIZE
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


    draw.text(
        (
            10,
            10,
        ),
        title,
        fill="black",
        font=title_font,
    )


    for index, record in enumerate(
        tile_records
    ):

        row = (
            index
            // CONTACT_COLUMNS
        )

        column = (
            index
            % CONTACT_COLUMNS
        )


        x = (
            column
            * TILE_SIZE
        )

        y = (
            52
            +
            row
            * (
                TILE_SIZE
                +
                LABEL_HEIGHT
            )
        )


        image = record[
            "image"
        ]


        sheet.paste(
            image,
            (
                x,
                y,
            ),
        )


        label = (
            f"#{int(record['tile_index']):02d}"
        )


        draw.text(
            (
                x + 4,
                y + TILE_SIZE + 2,
            ),
            label,
            fill="black",
            font=font,
        )


    sheet.save(
        path,
        quality=95,
        subsampling=0,
    )


def make_preview(
    path,
    size,
):

    with Image.open(
        path
    ) as source:

        source = source.convert(
            "RGB"
        )

        source.thumbnail(
            (
                size,
                size,
            ),
            Image.Resampling.LANCZOS,
        )


        canvas = Image.new(
            "RGB",
            (
                size,
                size,
            ),
            "white",
        )


        x = (
            size
            - source.width
        ) // 2

        y = (
            size
            - source.height
        ) // 2


        canvas.paste(
            source,
            (
                x,
                y,
            ),
        )


    return canvas


def create_three_way_overview(
    saved_path,
    raw_path,
    srgb_path,
    out_path,
    case_id,
    icc_available,
    profile_text,
):

    titles = [
        "A. V2_PROD saved",
        "B. RAW reread",
        (
            "C. ICC -> sRGB"
            if icc_available
            else
            "C. ICC unavailable"
        ),
    ]


    paths = [
        saved_path,
        raw_path,
        srgb_path,
    ]


    title_font = load_font(
        26
    )

    label_font = load_font(
        21
    )

    small_font = load_font(
        16
    )


    margin = 20
    gap = 20

    header = 105
    footer = 60


    width = (
        margin * 2
        +
        3 * OVERVIEW_CONTACT_SIZE
        +
        2 * gap
    )


    height = (
        header
        +
        OVERVIEW_CONTACT_SIZE
        +
        footer
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


    draw.text(
        (
            margin,
            12,
        ),
        (
            f"{case_id} — WSI color audit"
        ),
        fill="black",
        font=title_font,
    )


    profile_line = (
        f"ICC: {profile_text}"
        if profile_text
        else
        "ICC: not available"
    )


    draw.text(
        (
            margin,
            52,
        ),
        profile_line,
        fill="black",
        font=small_font,
    )


    for index, (
        image_path,
        title,
    ) in enumerate(
        zip(
            paths,
            titles,
        )
    ):

        x = (
            margin
            +
            index
            * (
                OVERVIEW_CONTACT_SIZE
                +
                gap
            )
        )


        preview = make_preview(
            image_path,
            OVERVIEW_CONTACT_SIZE,
        )


        overview.paste(
            preview,
            (
                x,
                header,
            ),
        )


        draw.text(
            (
                x,
                header
                +
                OVERVIEW_CONTACT_SIZE
                +
                12,
            ),
            title,
            fill="black",
            font=label_font,
        )


    overview.save(
        out_path,
        quality=95,
        subsampling=0,
    )


# ============================================================
# CASE PROCESSING
# ============================================================

def process_case(
    case_id,
    wsi_candidates,
    keep_wsi,
):

    print()
    print("=" * 80)
    print(
        f" {case_id}"
    )
    print("=" * 80)


    patient_manifest_path = (
        PATIENT_TILE_MANIFEST_ROOT
        / f"{case_id}.csv"
    )


    if not patient_manifest_path.exists():

        raise FileNotFoundError(
            "Patient tile manifest missing: "
            f"{patient_manifest_path}"
        )


    tiles = pd.read_csv(
        patient_manifest_path
    )


    tiles = tiles.sort_values(
        "tile_index"
    ).copy()


    if len(tiles) != 64:

        raise RuntimeError(
            f"{case_id}: expected 64 tiles, "
            f"found {len(tiles)}"
        )


    selected_file_ids = (
        tiles[
            "selected_wsi_file_id"
        ]
        .astype(str)
        .unique()
    )


    if len(
        selected_file_ids
    ) != 1:

        raise RuntimeError(
            f"{case_id}: expected one WSI UUID "
            f"in tile manifest, found "
            f"{len(selected_file_ids)}"
        )


    selected_file_id = (
        selected_file_ids[0]
    )


    candidates = (
        wsi_candidates[
            wsi_candidates[
                "file_id"
            ].astype(str)
            ==
            selected_file_id
        ]
        .copy()
    )


    if len(candidates) != 1:

        raise RuntimeError(
            f"{case_id}: unable to uniquely "
            f"resolve WSI UUID {selected_file_id}"
        )


    candidate = (
        candidates.iloc[0]
    )


    print(
        f"WSI UUID:      "
        f"{selected_file_id}"
    )

    print(
        f"slide token:   "
        f"{candidate['slide_token']}"
    )

    print(
        f"WSI size:      "
        f"{candidate['file_size_bytes'] / 1024**3:.3f} GB"
    )


    patient_temp_dir = (
        TEMP_ROOT
        / case_id
    )


    patient_temp_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    wsi_path = download_wsi(
        candidate,
        patient_temp_dir,
    )


    case_output = (
        OUTPUT_ROOT
        / case_id
    )


    case_output.mkdir(
        parents=True,
        exist_ok=True,
    )


    saved_records = []
    raw_records = []
    srgb_records = []

    metric_rows = []


    with openslide.OpenSlide(
        str(
            wsi_path
        )
    ) as slide:

        (
            transform,
            rendering_intent,
            profile_text,
        ) = build_icc_transform(
            slide
        )


        icc_available = (
            transform
            is not None
        )


        print(
            f"ICC profile:   "
            f"{'YES' if icc_available else 'NO'}"
        )


        if profile_text:

            print(
                f"ICC info:      "
                f"{profile_text}"
            )


        if (
            rendering_intent
            is not None
        ):

            print(
                f"ICC intent:    "
                f"{rendering_intent}"
            )


        print(
            f"dimensions:     "
            f"{slide.dimensions[0]}"
            f"x"
            f"{slide.dimensions[1]}"
        )

        print(
            f"levels:         "
            f"{slide.level_count}"
        )


        for _, row in (
            tiles.iterrows()
        ):

            tile_index = int(
                row[
                    "tile_index"
                ]
            )

            x = int(
                row[
                    "x_level0"
                ]
            )

            y = int(
                row[
                    "y_level0"
                ]
            )


            saved_path = Path(
                row[
                    "tile_path"
                ]
            )


            if not saved_path.exists():

                raise FileNotFoundError(
                    f"Saved production tile missing: "
                    f"{saved_path}"
                )


            with Image.open(
                saved_path
            ) as saved_image_source:

                saved_image = (
                    saved_image_source
                    .convert(
                        "RGB"
                    )
                    .copy()
                )


            region_rgba = (
                slide.read_region(
                    (
                        x,
                        y,
                    ),
                    0,
                    (
                        TILE_SIZE,
                        TILE_SIZE,
                    ),
                )
            )


            raw_image = (
                region_rgba
                .convert(
                    "RGB"
                )
            )


            region_icc_bytes = (
                region_rgba.info.get(
                    "icc_profile"
                )
            )


            if icc_available:

                srgb_image = (
                    apply_icc_to_region(
                        region_rgba,
                        transform,
                    )
                )

            else:

                # Keep the same RAW image so the contact sheet
                # clearly shows that no transform was possible.
                srgb_image = (
                    raw_image.copy()
                )


            # --------------------------------------------
            # A vs B:
            # saved V2_PROD vs freshly reread RAW.
            #
            # Should normally be identical or essentially
            # identical if coordinates and decoding are stable.
            # --------------------------------------------

            saved_vs_raw = (
                image_difference_metrics(
                    saved_image,
                    raw_image,
                )
            )


            # --------------------------------------------
            # B vs C:
            # actual effect of ICC -> sRGB.
            # --------------------------------------------

            raw_vs_srgb = (
                image_difference_metrics(
                    raw_image,
                    srgb_image,
                )
            )


            raw_summary = (
                image_rgb_summary(
                    raw_image
                )
            )


            srgb_summary = (
                image_rgb_summary(
                    srgb_image
                )
            )


            metric_rows.append({
                "case_submitter_id": (
                    case_id
                ),

                "tile_index": (
                    tile_index
                ),

                "x_level0": (
                    x
                ),

                "y_level0": (
                    y
                ),

                "icc_available": (
                    icc_available
                ),

                "region_icc_bytes_present": (
                    region_icc_bytes
                    is not None
                ),

                "region_icc_byte_count": (
                    len(
                        region_icc_bytes
                    )
                    if region_icc_bytes
                    is not None
                    else 0
                ),

                "saved_vs_raw_mae": (
                    saved_vs_raw[
                        "rgb_mae"
                    ]
                ),

                "saved_vs_raw_p95": (
                    saved_vs_raw[
                        "rgb_pixel_p95"
                    ]
                ),

                "raw_vs_srgb_mae": (
                    raw_vs_srgb[
                        "rgb_mae"
                    ]
                ),

                "raw_vs_srgb_p50": (
                    raw_vs_srgb[
                        "rgb_pixel_p50"
                    ]
                ),

                "raw_vs_srgb_p95": (
                    raw_vs_srgb[
                        "rgb_pixel_p95"
                    ]
                ),

                "raw_vs_srgb_p99": (
                    raw_vs_srgb[
                        "rgb_pixel_p99"
                    ]
                ),

                "raw_vs_srgb_max": (
                    raw_vs_srgb[
                        "rgb_max_abs"
                    ]
                ),

                "raw_mean_r": (
                    raw_summary[
                        "mean_r"
                    ]
                ),

                "raw_mean_g": (
                    raw_summary[
                        "mean_g"
                    ]
                ),

                "raw_mean_b": (
                    raw_summary[
                        "mean_b"
                    ]
                ),

                "srgb_mean_r": (
                    srgb_summary[
                        "mean_r"
                    ]
                ),

                "srgb_mean_g": (
                    srgb_summary[
                        "mean_g"
                    ]
                ),

                "srgb_mean_b": (
                    srgb_summary[
                        "mean_b"
                    ]
                ),
            })


            saved_records.append({
                "tile_index": (
                    tile_index
                ),

                "image": (
                    saved_image
                ),
            })


            raw_records.append({
                "tile_index": (
                    tile_index
                ),

                "image": (
                    raw_image
                ),
            })


            srgb_records.append({
                "tile_index": (
                    tile_index
                ),

                "image": (
                    srgb_image
                ),
            })


    # ========================================================
    # SAVE CONTACT SHEETS
    # ========================================================

    saved_contact = (
        case_output
        / "A_v2prod_saved.jpg"
    )

    raw_contact = (
        case_output
        / "B_raw_reread.jpg"
    )

    srgb_contact = (
        case_output
        / "C_icc_to_srgb.jpg"
    )


    create_numbered_contact_sheet(
        saved_records,
        saved_contact,
        (
            f"{case_id} — "
            "A: original V2_PROD saved tiles"
        ),
    )


    create_numbered_contact_sheet(
        raw_records,
        raw_contact,
        (
            f"{case_id} — "
            "B: fresh RAW OpenSlide read_region"
        ),
    )


    create_numbered_contact_sheet(
    srgb_records,
    srgb_contact,
    (
        f"{case_id} — "
        + (
            "C: ICC -> sRGB"
            if icc_available
            else
            "C: no ICC profile; RAW copy"
        )
    ),
)


    overview_path = (
        case_output
        / "D_three_way_overview.jpg"
    )


    create_three_way_overview(
        saved_contact,
        raw_contact,
        srgb_contact,
        overview_path,
        case_id,
        icc_available,
        profile_text,
    )


    # ========================================================
    # METRICS
    # ========================================================

    metrics = pd.DataFrame(
        metric_rows
    )


    metrics_path = (
        case_output
        / "tile_color_metrics.csv"
    )


    metrics.to_csv(
        metrics_path,
        index=False,
    )


    summary = {
        "case_submitter_id": (
            case_id
        ),

        "selected_wsi_file_id": (
            selected_file_id
        ),

        "selected_slide_token": (
            str(
                candidate[
                    "slide_token"
                ]
            )
        ),

        "wsi_file_name": (
            str(
                candidate[
                    "file_name"
                ]
            )
        ),

        "icc_available": bool(
            icc_available
        ),

        "icc_profile_text": (
            profile_text
        ),

        "icc_rendering_intent": (
            int(
                rendering_intent
            )
            if rendering_intent
            is not None
            else None
        ),

        "tiles": int(
            len(
                metrics
            )
        ),

        "saved_vs_raw_mae_mean": float(
            metrics[
                "saved_vs_raw_mae"
            ].mean()
        ),

        "saved_vs_raw_mae_max": float(
            metrics[
                "saved_vs_raw_mae"
            ].max()
        ),

        "raw_vs_srgb_mae_mean": float(
            metrics[
                "raw_vs_srgb_mae"
            ].mean()
        ),

        "raw_vs_srgb_mae_median": float(
            metrics[
                "raw_vs_srgb_mae"
            ].median()
        ),

        "raw_vs_srgb_mae_max": float(
            metrics[
                "raw_vs_srgb_mae"
            ].max()
        ),

        "raw_vs_srgb_p95_mean": float(
            metrics[
                "raw_vs_srgb_p95"
            ].mean()
        ),
    }


    summary_path = (
        AUDIT_METADATA_ROOT
        / f"{case_id}.json"
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
    # PRINT RESULT
    # ========================================================

    print()
    print(
        f"saved vs RAW MAE mean: "
        f"{summary['saved_vs_raw_mae_mean']:.4f}"
    )

    print(
        f"saved vs RAW MAE max:  "
        f"{summary['saved_vs_raw_mae_max']:.4f}"
    )


    if icc_available:

        print(
            f"RAW vs sRGB MAE mean:  "
            f"{summary['raw_vs_srgb_mae_mean']:.4f}"
        )

        print(
            f"RAW vs sRGB MAE max:   "
            f"{summary['raw_vs_srgb_mae_max']:.4f}"
        )

    else:

        print(
            "RAW vs sRGB:            "
            "not applicable — no ICC"
        )


    print(
        f"overview:                "
        f"{overview_path}"
    )


    # ========================================================
    # RELEASE IMAGES
    # ========================================================

    for record_list in [
        saved_records,
        raw_records,
        srgb_records,
    ]:

        for record in record_list:

            try:

                record[
                    "image"
                ].close()

            except Exception:

                pass


    # ========================================================
    # CLEAN RAW WSI
    # ========================================================

    if not keep_wsi:

        remove_download(
            patient_temp_dir,
            selected_file_id,
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


    return summary


# ============================================================
# GLOBAL OVERVIEW
# ============================================================

def create_global_overview(
    summaries,
):

    if not summaries:
        return None


    case_count = len(
        summaries
    )


    preview_size = 600

    margin = 20
    gap = 20

    header = 70
    label_height = 90


    width = (
        margin * 2
        +
        case_count
        * preview_size
        +
        max(
            0,
            case_count - 1,
        )
        * gap
    )


    height = (
        header
        +
        preview_size
        +
        label_height
    )


    page = Image.new(
        "RGB",
        (
            width,
            height,
        ),
        "white",
    )


    draw = ImageDraw.Draw(
        page
    )


    title_font = load_font(
        27
    )

    label_font = load_font(
        18
    )


    draw.text(
        (
            margin,
            15,
        ),
        (
            "Suspicious WSI color audit "
            "— ICC/sRGB comparison"
        ),
        fill="black",
        font=title_font,
    )


    for index, summary in enumerate(
        summaries
    ):

        case_id = (
            summary[
                "case_submitter_id"
            ]
        )


        overview_path = (
            OUTPUT_ROOT
            / case_id
            / "D_three_way_overview.jpg"
        )


        preview = make_preview(
            overview_path,
            preview_size,
        )


        x = (
            margin
            +
            index
            * (
                preview_size
                +
                gap
            )
        )


        page.paste(
            preview,
            (
                x,
                header,
            ),
        )


        line_1 = (
            f"{case_id}"
        )


        line_2 = (
            f"ICC="
            f"{'YES' if summary['icc_available'] else 'NO'}"
        )


        line_3 = (
            "RAW->sRGB MAE="
            f"{summary['raw_vs_srgb_mae_mean']:.2f}"
        )


        draw.text(
            (
                x,
                header
                +
                preview_size
                +
                8,
            ),
            line_1,
            fill="black",
            font=label_font,
        )


        draw.text(
            (
                x,
                header
                +
                preview_size
                +
                34,
            ),
            line_2,
            fill="black",
            font=label_font,
        )


        draw.text(
            (
                x,
                header
                +
                preview_size
                +
                59,
            ),
            line_3,
            fill="black",
            font=label_font,
        )


    output_path = (
        OUTPUT_ROOT
        / "all_cases_overview.jpg"
    )


    page.save(
        output_path,
        quality=95,
        subsampling=0,
    )


    return output_path


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Audit suspicious V2_PROD WSI tiles "
            "for ICC color-management effects."
        )
    )


    parser.add_argument(
        "--cases",
        nargs="+",
        default=DEFAULT_CASES,
        help=(
            "Case IDs to audit."
        ),
    )


    parser.add_argument(
        "--keep-wsi",
        action="store_true",
        help=(
            "Keep downloaded WSI files after audit."
        ),
    )


    args = parser.parse_args()


    if not GDC_CLIENT.exists():

        raise FileNotFoundError(
            f"GDC client not found: "
            f"{GDC_CLIENT}"
        )


    if not WSI_CANDIDATES_FILE.exists():

        raise FileNotFoundError(
            f"WSI candidate manifest missing: "
            f"{WSI_CANDIDATES_FILE}"
        )


    wsi_candidates = pd.read_csv(
        WSI_CANDIDATES_FILE
    )


    print()
    print("=" * 80)
    print(
        " SUSPICIOUS WSI COLOR / ICC AUDIT"
    )
    print("=" * 80)

    print(
        f"Cases:       "
        f"{len(args.cases)}"
    )

    print(
        f"Keep WSI:    "
        f"{args.keep_wsi}"
    )

    print(
        f"Free disk:   "
        f"{free_disk_gb():.2f} GB"
    )

    print("=" * 80)


    summaries = []


    for case_id in (
        args.cases
    ):

        try:

            summary = process_case(
                case_id,
                wsi_candidates,
                args.keep_wsi,
            )


            summaries.append(
                summary
            )


        except Exception as exc:

            print()
            print(
                f"ERROR {case_id}: "
                f"{repr(exc)}"
            )


            summaries.append({
                "case_submitter_id": (
                    case_id
                ),

                "status": "ERROR",

                "error": repr(
                    exc
                ),
            })


    # ========================================================
    # SAVE GLOBAL CSV / JSON
    # ========================================================

    summary_df = pd.DataFrame(
        summaries
    )


    csv_path = (
        AUDIT_METADATA_ROOT
        / "suspicious_wsi_color_audit.csv"
    )


    summary_df.to_csv(
        csv_path,
        index=False,
    )


    json_path = (
        AUDIT_METADATA_ROOT
        / "suspicious_wsi_color_audit.json"
    )


    json_path.write_text(
        json.dumps(
            summaries,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


    valid_summaries = [
        summary
        for summary
        in summaries
        if (
            summary.get(
                "status"
            )
            != "ERROR"
        )
    ]


    global_overview = (
        create_global_overview(
            valid_summaries
        )
    )


    # ========================================================
    # FINAL
    # ========================================================

    print()
    print("=" * 80)
    print(
        " COLOR AUDIT COMPLETE"
    )
    print("=" * 80)


    for summary in summaries:

        case_id = (
            summary[
                "case_submitter_id"
            ]
        )


        if (
            summary.get(
                "status"
            )
            == "ERROR"
        ):

            print(
                f"{case_id}: ERROR"
            )

            continue


        print(
            f"{case_id}: "
            f"ICC="
            f"{'YES' if summary['icc_available'] else 'NO'}, "
            f"saved/raw MAE="
            f"{summary['saved_vs_raw_mae_mean']:.3f}, "
            f"raw/sRGB MAE="
            f"{summary['raw_vs_srgb_mae_mean']:.3f}"
        )


    print()
    print(
        "Summary CSV:"
    )

    print(
        f"  {csv_path}"
    )


    print(
        "Summary JSON:"
    )

    print(
        f"  {json_path}"
    )


    if global_overview:

        print(
            "Global overview:"
        )

        print(
            f"  {global_overview}"
        )


    print()
    print(
        "Per-case results:"
    )

    print(
        f"  {OUTPUT_ROOT}"
    )

    print("=" * 80)


if __name__ == "__main__":
    main()