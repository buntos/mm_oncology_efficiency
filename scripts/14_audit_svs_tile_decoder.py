from pathlib import Path
import argparse
import hashlib
import json
import math
import shutil
import subprocess
import time

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import openslide
import tifffile
import imagecodecs


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
# DEFAULT CASES
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
    / "temp_gdc_decoder_audit"
)

OUTPUT_ROOT = (
    ROOT
    / "results"
    / "figures"
    / "svs_decoder_audit"
)

AUDIT_METADATA_ROOT = (
    METADATA_DIR
    / "svs_decoder_audit"
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

PATCH_SIZE = 256

DOWNLOAD_RETRIES = 5
DOWNLOAD_RETRY_DELAY_SECONDS = 5

MIN_FREE_SPACE_BUFFER_GB = 8.0

CONTACT_COLUMNS = 8
CONTACT_ROWS = 8
LABEL_HEIGHT = 24

TOP_DIFFERENCE_TILES = 12

COMPARISON_TILE_SIZE = 256
COMPARISON_LABEL_HEIGHT = 54


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


    ok, _ = verify_file(
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


        uuid_dir = (
            patient_temp_dir
            / file_id
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
# IMAGE HELPERS
# ============================================================

def normalize_rgb_array(
    array,
):

    if array is None:

        raise RuntimeError(
            "Decoded TIFF segment is None"
        )


    arr = np.asarray(
        array
    )


    # tifffile segment decoder often returns:
    #
    # (1, H, W, 3)
    #
    # Remove singleton axes.
    arr = np.squeeze(
        arr
    )


    if arr.ndim == 2:

        arr = np.stack(
            [
                arr,
                arr,
                arr,
            ],
            axis=-1,
        )


    elif (
        arr.ndim == 3
        and
        arr.shape[0] in (
            3,
            4,
        )
        and
        arr.shape[-1] not in (
            3,
            4,
        )
    ):

        arr = np.moveaxis(
            arr,
            0,
            -1,
        )


    if arr.ndim != 3:

        raise RuntimeError(
            "Unexpected decoded segment shape: "
            f"{arr.shape}"
        )


    if arr.shape[-1] == 4:

        arr = arr[
            ...,
            :3,
        ]


    if arr.shape[-1] != 3:

        raise RuntimeError(
            "Expected RGB-like decoded segment, "
            f"got shape {arr.shape}"
        )


    if arr.dtype != np.uint8:

        if np.issubdtype(
            arr.dtype,
            np.floating,
        ):

            if (
                arr.max()
                <= 1.0
            ):

                arr = (
                    arr
                    * 255.0
                )


        arr = np.clip(
            arr,
            0,
            255,
        ).astype(
            np.uint8
        )


    return arr


def difference_metrics(
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


    pixel_mae = delta.mean(
        axis=2
    )


    return {
        "mae": float(
            delta.mean()
        ),

        "pixel_mae_p50": float(
            np.percentile(
                pixel_mae,
                50,
            )
        ),

        "pixel_mae_p95": float(
            np.percentile(
                pixel_mae,
                95,
            )
        ),

        "pixel_mae_p99": float(
            np.percentile(
                pixel_mae,
                99,
            )
        ),

        "max_abs": float(
            delta.max()
        ),

        "different_pixel_fraction_gt5": float(
            (
                pixel_mae
                > 5.0
            ).mean()
        ),

        "different_pixel_fraction_gt20": float(
            (
                pixel_mae
                > 20.0
            ).mean()
        ),
    }


def tile_visual_metrics(
    image,
):

    rgb = np.asarray(
        image.convert(
            "RGB"
        ),
        dtype=np.float32,
    )


    r = rgb[
        :,
        :,
        0
    ]

    g = rgb[
        :,
        :,
        1
    ]

    b = rgb[
        :,
        :,
        2
    ]


    gray = (
        0.299 * r
        +
        0.587 * g
        +
        0.114 * b
    )


    black_fraction = float(
        (
            rgb.max(
                axis=2
            )
            < 25
        ).mean()
    )


    near_black_fraction = float(
        (
            rgb.max(
                axis=2
            )
            < 50
        ).mean()
    )


    red_excess = float(
        (
            r
            -
            (
                g + b
            )
            / 2.0
        ).mean()
    )


    green_excess = float(
        (
            g
            -
            (
                r + b
            )
            / 2.0
        ).mean()
    )


    blue_excess = float(
        (
            b
            -
            (
                r + g
            )
            / 2.0
        ).mean()
    )


    return {
        "gray_mean": float(
            gray.mean()
        ),

        "gray_std": float(
            gray.std()
        ),

        "black_fraction": (
            black_fraction
        ),

        "near_black_fraction": (
            near_black_fraction
        ),

        "mean_r": float(
            r.mean()
        ),

        "mean_g": float(
            g.mean()
        ),

        "mean_b": float(
            b.mean()
        ),

        "red_excess": (
            red_excess
        ),

        "green_excess": (
            green_excess
        ),

        "blue_excess": (
            blue_excess
        ),
    }


# ============================================================
# TIFF METADATA
# ============================================================

def tag_value(
    page,
    name,
):

    tag = page.tags.get(
        name
    )

    if tag is None:
        return None

    value = tag.value

    if isinstance(
        value,
        np.ndarray,
    ):

        return value.tolist()

    if isinstance(
        value,
        tuple,
    ):

        return list(
            value
        )

    try:

        if hasattr(
            value,
            "value",
        ):

            return int(
                value.value
            )

    except Exception:

        pass

    try:

        if isinstance(
            value,
            np.generic,
        ):

            return value.item()

    except Exception:

        pass

    return value


def tag_text(
    page,
    name,
):

    tag = page.tags.get(
        name
    )

    if tag is None:
        return None

    return str(
        tag.value
    )


def find_level0_page(
    tif,
    expected_width,
    expected_height,
):

    matches = []


    for page_index, page in enumerate(
        tif.pages
    ):

        try:

            width = int(
                page.imagewidth
            )

            height = int(
                page.imagelength
            )

        except Exception:

            continue


        if (
            width
            == expected_width
            and
            height
            == expected_height
        ):

            matches.append(
                (
                    page_index,
                    page,
                )
            )


    tiled_matches = [
        pair
        for pair
        in matches
        if pair[1].is_tiled
    ]


    if tiled_matches:

        return tiled_matches[0]


    if matches:

        return matches[0]


    raise RuntimeError(
        "Unable to find TIFF page matching "
        f"OpenSlide level-0 dimensions "
        f"{expected_width}x{expected_height}"
    )


def build_tiff_metadata(
    tif,
    page_index,
    page,
):

    compression_tag = (
        page.tags.get(
            "Compression"
        )
    )

    photometric_tag = (
        page.tags.get(
            "PhotometricInterpretation"
        )
    )


    compression_value = None
    compression_text = None

    photometric_value = None
    photometric_text = None


    if compression_tag is not None:

        try:

            compression_value = int(
                compression_tag.value
            )

        except Exception:

            compression_value = str(
                compression_tag.value
            )

        compression_text = str(
            compression_tag.value
        )


    if photometric_tag is not None:

        try:

            photometric_value = int(
                photometric_tag.value
            )

        except Exception:

            photometric_value = str(
                photometric_tag.value
            )

        photometric_text = str(
            photometric_tag.value
        )


    metadata = {
        "tifffile_version": (
            tifffile.__version__
        ),

        "imagecodecs_version": (
            imagecodecs.__version__
        ),

        "openslide_python_version": (
            getattr(
                openslide,
                "__version__",
                None,
            )
        ),

        "openslide_library_version": (
            getattr(
                openslide,
                "__library_version__",
                None,
            )
        ),

        "tiff_page_index": int(
            page_index
        ),

        "image_width": int(
            page.imagewidth
        ),

        "image_height": int(
            page.imagelength
        ),

        "is_tiled": bool(
            page.is_tiled
        ),

        "compression_value": (
            compression_value
        ),

        "compression_text": (
            compression_text
        ),

        "photometric_value": (
            photometric_value
        ),

        "photometric_text": (
            photometric_text
        ),

        "tile_width": (
            int(
                page.tilewidth
            )
            if page.is_tiled
            else None
        ),

        "tile_length": (
            int(
                page.tilelength
            )
            if page.is_tiled
            else None
        ),

        "samples_per_pixel": (
            tag_value(
                page,
                "SamplesPerPixel",
            )
        ),

        "bits_per_sample": (
            tag_value(
                page,
                "BitsPerSample",
            )
        ),

        "planar_configuration": (
            tag_value(
                page,
                "PlanarConfiguration",
            )
        ),

        "planar_configuration_text": (
            tag_text(
                page,
                "PlanarConfiguration",
            )
        ),

        "ycbcr_subsampling": (
            tag_value(
                page,
                "YCbCrSubSampling",
            )
        ),

        "jpeg_tables_present": bool(
            page.jpegtables
            is not None
        ),

        "jpeg_tables_bytes": (
            len(
                page.jpegtables
            )
            if page.jpegtables
            is not None
            else 0
        ),

        "data_segments": int(
            len(
                page.dataoffsets
            )
        ),

        "description_prefix": (
            str(
                page.description
            )[:1000]
            if page.description
            else None
        ),
    }


    try:

        metadata[
            "jpeg2k_codec_version"
        ] = (
            imagecodecs.jpeg2k_version()
        )

    except Exception:

        metadata[
            "jpeg2k_codec_version"
        ] = None


    return metadata


# ============================================================
# LOW-LEVEL TIFF TILE DECODE
# ============================================================

def decode_tiff_segment(
    tif,
    page,
    segment_index,
):

    if (
        segment_index
        < 0
        or
        segment_index
        >= len(
            page.dataoffsets
        )
    ):

        raise IndexError(
            "TIFF segment index out of range: "
            f"{segment_index}"
        )


    offset = int(
        page.dataoffsets[
            segment_index
        ]
    )

    bytecount = int(
        page.databytecounts[
            segment_index
        ]
    )


    if (
        offset == 0
        or
        bytecount == 0
    ):

        raise RuntimeError(
            "Sparse/empty TIFF tile encountered: "
            f"segment={segment_index}, "
            f"offset={offset}, "
            f"bytecount={bytecount}"
        )


    fh = tif.filehandle

    fh.seek(
        offset
    )


    compressed_data = fh.read(
        bytecount
    )


    decoded, indices, shape = (
        page.decode(
            compressed_data,
            segment_index,
            jpegtables=page.jpegtables,
        )
    )


    rgb = normalize_rgb_array(
        decoded
    )


    return {
        "rgb": rgb,
        "offset": offset,
        "bytecount": bytecount,
        "indices": (
            list(indices)
            if indices
            is not None
            else None
        ),
        "decoded_shape": (
            list(
                np.asarray(
                    decoded
                ).shape
            )
        ),
        "reported_shape": (
            list(shape)
            if shape
            is not None
            else None
        ),
    }


def decode_patch_with_tifffile(
    tif,
    page,
    x,
    y,
    patch_size,
):

    if not page.is_tiled:

        raise RuntimeError(
            "Level-0 TIFF page is not tiled. "
            "This audit expects tiled Aperio SVS."
        )


    planar = tag_value(
        page,
        "PlanarConfiguration",
    )


    # Aperio SVS is normally contiguous RGB.
    # Separate planes would require a different
    # segment-index mapping.
    if (
        planar is not None
        and
        int(planar) != 1
    ):

        raise RuntimeError(
            "Unsupported PlanarConfiguration "
            f"for this audit: {planar}"
        )


    tile_width = int(
        page.tilewidth
    )

    tile_height = int(
        page.tilelength
    )

    image_width = int(
        page.imagewidth
    )

    image_height = int(
        page.imagelength
    )


    tiles_x = math.ceil(
        image_width
        / tile_width
    )


    start_tile_x = (
        x
        // tile_width
    )

    end_tile_x = (
        (
            x
            + patch_size
            - 1
        )
        // tile_width
    )


    start_tile_y = (
        y
        // tile_height
    )

    end_tile_y = (
        (
            y
            + patch_size
            - 1
        )
        // tile_height
    )


    patch = np.zeros(
        (
            patch_size,
            patch_size,
            3,
        ),
        dtype=np.uint8,
    )


    coverage = np.zeros(
        (
            patch_size,
            patch_size,
        ),
        dtype=bool,
    )


    used_segments = []


    for tile_y in range(
        start_tile_y,
        end_tile_y + 1,
    ):

        for tile_x in range(
            start_tile_x,
            end_tile_x + 1,
        ):

            segment_index = (
                tile_y
                * tiles_x
                +
                tile_x
            )


            segment = decode_tiff_segment(
                tif,
                page,
                segment_index,
            )


            tile_rgb = segment[
                "rgb"
            ]


            global_tile_x0 = (
                tile_x
                * tile_width
            )

            global_tile_y0 = (
                tile_y
                * tile_height
            )


            global_x0 = max(
                x,
                global_tile_x0,
            )

            global_y0 = max(
                y,
                global_tile_y0,
            )

            global_x1 = min(
                x + patch_size,
                global_tile_x0
                + tile_rgb.shape[1],
                image_width,
            )

            global_y1 = min(
                y + patch_size,
                global_tile_y0
                + tile_rgb.shape[0],
                image_height,
            )


            if (
                global_x1
                <= global_x0
                or
                global_y1
                <= global_y0
            ):

                continue


            src_x0 = (
                global_x0
                - global_tile_x0
            )

            src_y0 = (
                global_y0
                - global_tile_y0
            )

            src_x1 = (
                src_x0
                +
                (
                    global_x1
                    - global_x0
                )
            )

            src_y1 = (
                src_y0
                +
                (
                    global_y1
                    - global_y0
                )
            )


            dst_x0 = (
                global_x0
                - x
            )

            dst_y0 = (
                global_y0
                - y
            )

            dst_x1 = (
                dst_x0
                +
                (
                    global_x1
                    - global_x0
                )
            )

            dst_y1 = (
                dst_y0
                +
                (
                    global_y1
                    - global_y0
                )
            )


            patch[
                dst_y0:dst_y1,
                dst_x0:dst_x1,
            ] = tile_rgb[
                src_y0:src_y1,
                src_x0:src_x1,
            ]


            coverage[
                dst_y0:dst_y1,
                dst_x0:dst_x1,
            ] = True


            used_segments.append({
                "segment_index": int(
                    segment_index
                ),

                "tile_x_index": int(
                    tile_x
                ),

                "tile_y_index": int(
                    tile_y
                ),

                "offset": int(
                    segment[
                        "offset"
                    ]
                ),

                "bytecount": int(
                    segment[
                        "bytecount"
                    ]
                ),

                "decoded_shape": (
                    segment[
                        "decoded_shape"
                    ]
                ),
            })


    coverage_fraction = float(
        coverage.mean()
    )


    if coverage_fraction < 1.0:

        raise RuntimeError(
            "TIFF reconstruction did not cover "
            f"the complete patch: "
            f"{coverage_fraction:.6f}"
        )


    return (
        Image.fromarray(
            patch,
            mode="RGB",
        ),
        used_segments,
    )


# ============================================================
# CONTACT SHEETS
# ============================================================

def create_numbered_contact_sheet(
    records,
    image_key,
    output_path,
    title,
):

    font = load_font(
        16
    )

    title_font = load_font(
        24
    )


    width = (
        CONTACT_COLUMNS
        * PATCH_SIZE
    )


    height = (
        52
        +
        CONTACT_ROWS
        * (
            PATCH_SIZE
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
        records
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
            * PATCH_SIZE
        )

        y = (
            52
            +
            row
            * (
                PATCH_SIZE
                +
                LABEL_HEIGHT
            )
        )


        image = record[
            image_key
        ]


        sheet.paste(
            image,
            (
                x,
                y,
            ),
        )


        draw.text(
            (
                x + 4,
                y + PATCH_SIZE + 2,
            ),
            (
                f"#{int(record['tile_index']):02d}"
            ),
            fill="black",
            font=font,
        )


    sheet.save(
        output_path,
        quality=95,
        subsampling=0,
    )


def create_top_difference_sheet(
    records,
    output_path,
    case_id,
):

    ranked = sorted(
        records,
        key=lambda item: (
            item[
                "openslide_vs_tiff_mae"
            ]
        ),
        reverse=True,
    )[:TOP_DIFFERENCE_TILES]


    rows = len(
        ranked
    )


    width = (
        3
        * COMPARISON_TILE_SIZE
    )


    header_height = 62


    height = (
        header_height
        +
        rows
        * (
            COMPARISON_TILE_SIZE
            +
            COMPARISON_LABEL_HEIGHT
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
        24
    )

    label_font = load_font(
        17
    )


    draw.text(
        (
            10,
            10,
        ),
        (
            f"{case_id} — largest "
            "OpenSlide vs tifffile differences"
        ),
        fill="black",
        font=title_font,
    )


    column_titles = [
        "V2 saved",
        "OpenSlide reread",
        "tifffile direct",
    ]


    for row_index, record in enumerate(
        ranked
    ):

        y = (
            header_height
            +
            row_index
            * (
                COMPARISON_TILE_SIZE
                +
                COMPARISON_LABEL_HEIGHT
            )
        )


        images = [
            record[
                "saved_image"
            ],
            record[
                "openslide_image"
            ],
            record[
                "tiff_image"
            ],
        ]


        for column_index, image in enumerate(
            images
        ):

            x = (
                column_index
                * COMPARISON_TILE_SIZE
            )


            sheet.paste(
                image,
                (
                    x,
                    y,
                ),
            )


            label = (
                f"{column_titles[column_index]} "
                f"#{record['tile_index']:02d}"
            )


            if column_index == 2:

                label += (
                    " | MAE="
                    f"{record['openslide_vs_tiff_mae']:.2f}"
                )


            draw.text(
                (
                    x + 4,
                    y
                    + COMPARISON_TILE_SIZE
                    + 3,
                ),
                label,
                fill="black",
                font=label_font,
            )


    sheet.save(
        output_path,
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
    openslide_path,
    tiff_path,
    output_path,
    case_id,
    compression_text,
    photometric_text,
):

    preview_size = 640

    margin = 20
    gap = 20

    header_height = 105
    footer_height = 55


    width = (
        margin * 2
        +
        3 * preview_size
        +
        2 * gap
    )


    height = (
        header_height
        +
        preview_size
        +
        footer_height
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
        26
    )

    small_font = load_font(
        16
    )

    label_font = load_font(
        20
    )


    draw.text(
        (
            margin,
            12,
        ),
        (
            f"{case_id} — SVS decoder audit"
        ),
        fill="black",
        font=title_font,
    )


    draw.text(
        (
            margin,
            52,
        ),
        (
            f"Compression={compression_text} | "
            f"Photometric={photometric_text}"
        ),
        fill="black",
        font=small_font,
    )


    paths = [
        saved_path,
        openslide_path,
        tiff_path,
    ]


    titles = [
        "A. V2_PROD saved",
        "B. OpenSlide reread",
        "C. tifffile direct decode",
    ]


    for index, (
        path,
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
                preview_size
                +
                gap
            )
        )


        preview = make_preview(
            path,
            preview_size,
        )


        page.paste(
            preview,
            (
                x,
                header_height,
            ),
        )


        draw.text(
            (
                x,
                header_height
                +
                preview_size
                +
                12,
            ),
            title,
            fill="black",
            font=label_font,
        )


    page.save(
        output_path,
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


    tiles = (
        tiles
        .sort_values(
            "tile_index"
        )
        .copy()
    )


    if len(tiles) != 64:

        raise RuntimeError(
            f"{case_id}: expected 64 tiles, "
            f"found {len(tiles)}"
        )


    file_ids = (
        tiles[
            "selected_wsi_file_id"
        ]
        .astype(str)
        .unique()
    )


    if len(file_ids) != 1:

        raise RuntimeError(
            f"{case_id}: expected one selected WSI"
        )


    file_id = (
        file_ids[0]
    )


    candidate_rows = (
        wsi_candidates[
            wsi_candidates[
                "file_id"
            ].astype(str)
            ==
            file_id
        ]
    )


    if len(candidate_rows) != 1:

        raise RuntimeError(
            f"{case_id}: cannot resolve WSI UUID "
            f"{file_id}"
        )


    candidate = (
        candidate_rows.iloc[0]
    )


    print(
        f"WSI UUID:      {file_id}"
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


    records = []
    metric_rows = []


    with openslide.OpenSlide(
        str(
            wsi_path
        )
    ) as slide:

        expected_width = int(
            slide.dimensions[0]
        )

        expected_height = int(
            slide.dimensions[1]
        )


        with tifffile.TiffFile(
            str(
                wsi_path
            )
        ) as tif:

            (
                page_index,
                page,
            ) = find_level0_page(
                tif,
                expected_width,
                expected_height,
            )


            metadata = build_tiff_metadata(
                tif,
                page_index,
                page,
            )


            print()
            print(
                "TIFF level-0 metadata:"
            )

            print(
                f"  page index:       "
                f"{metadata['tiff_page_index']}"
            )

            print(
                f"  dimensions:       "
                f"{metadata['image_width']}"
                f"x"
                f"{metadata['image_height']}"
            )

            print(
                f"  tiled:            "
                f"{metadata['is_tiled']}"
            )

            print(
                f"  tile size:        "
                f"{metadata['tile_width']}"
                f"x"
                f"{metadata['tile_length']}"
            )

            print(
                f"  compression:      "
                f"{metadata['compression_value']} "
                f"({metadata['compression_text']})"
            )

            print(
                f"  photometric:      "
                f"{metadata['photometric_value']} "
                f"({metadata['photometric_text']})"
            )

            print(
                f"  planar config:    "
                f"{metadata['planar_configuration']} "
                f"({metadata['planar_configuration_text']})"
            )

            print(
                f"  YCbCr subsample:  "
                f"{metadata['ycbcr_subsampling']}"
            )

            print(
                f"  JPEG tables:      "
                f"{metadata['jpeg_tables_present']}"
            )

            print(
                f"  TIFF segments:    "
                f"{metadata['data_segments']}"
            )

            print(
                f"  tifffile:         "
                f"{metadata['tifffile_version']}"
            )

            print(
                f"  imagecodecs:      "
                f"{metadata['imagecodecs_version']}"
            )

            print(
                f"  JPEG2K codec:     "
                f"{metadata['jpeg2k_codec_version']}"
            )


            metadata.update({
                "case_submitter_id": (
                    case_id
                ),

                "wsi_file_id": (
                    file_id
                ),

                "wsi_file_name": (
                    str(
                        candidate[
                            "file_name"
                        ]
                    )
                ),

                "slide_token": (
                    str(
                        candidate[
                            "slide_token"
                        ]
                    )
                ),
            })


            metadata_path = (
                case_output
                / "tiff_metadata.json"
            )


            metadata_path.write_text(
                json.dumps(
                    metadata,
                    indent=2,
                    ensure_ascii=False,
                    default=str,
                ),
                encoding="utf-8",
            )


            print()
            print(
                "Decoding the same 64 patches "
                "through both decoder paths..."
            )


            for row_number, (
                _,
                row,
            ) in enumerate(
                tiles.iterrows(),
                start=1,
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
                        f"Saved tile missing: "
                        f"{saved_path}"
                    )


                with Image.open(
                    saved_path
                ) as source:

                    saved_image = (
                        source
                        .convert(
                            "RGB"
                        )
                        .copy()
                    )


                openslide_image = (
                    slide.read_region(
                        (
                            x,
                            y,
                        ),
                        0,
                        (
                            PATCH_SIZE,
                            PATCH_SIZE,
                        ),
                    )
                    .convert(
                        "RGB"
                    )
                )


                (
                    tiff_image,
                    used_segments,
                ) = decode_patch_with_tifffile(
                    tif,
                    page,
                    x,
                    y,
                    PATCH_SIZE,
                )


                saved_vs_openslide = (
                    difference_metrics(
                        saved_image,
                        openslide_image,
                    )
                )


                openslide_vs_tiff = (
                    difference_metrics(
                        openslide_image,
                        tiff_image,
                    )
                )


                saved_vs_tiff = (
                    difference_metrics(
                        saved_image,
                        tiff_image,
                    )
                )


                visual = tile_visual_metrics(
                    saved_image
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

                    "source_tiff_segments": (
                        ";".join(
                            str(
                                segment[
                                    "segment_index"
                                ]
                            )
                            for segment
                            in used_segments
                        )
                    ),

                    "source_segment_count": int(
                        len(
                            used_segments
                        )
                    ),

                    "saved_vs_openslide_mae": (
                        saved_vs_openslide[
                            "mae"
                        ]
                    ),

                    "saved_vs_openslide_p95": (
                        saved_vs_openslide[
                            "pixel_mae_p95"
                        ]
                    ),

                    "openslide_vs_tiff_mae": (
                        openslide_vs_tiff[
                            "mae"
                        ]
                    ),

                    "openslide_vs_tiff_p50": (
                        openslide_vs_tiff[
                            "pixel_mae_p50"
                        ]
                    ),

                    "openslide_vs_tiff_p95": (
                        openslide_vs_tiff[
                            "pixel_mae_p95"
                        ]
                    ),

                    "openslide_vs_tiff_p99": (
                        openslide_vs_tiff[
                            "pixel_mae_p99"
                        ]
                    ),

                    "openslide_vs_tiff_max": (
                        openslide_vs_tiff[
                            "max_abs"
                        ]
                    ),

                    "openslide_vs_tiff_fraction_gt5": (
                        openslide_vs_tiff[
                            "different_pixel_fraction_gt5"
                        ]
                    ),

                    "openslide_vs_tiff_fraction_gt20": (
                        openslide_vs_tiff[
                            "different_pixel_fraction_gt20"
                        ]
                    ),

                    "saved_vs_tiff_mae": (
                        saved_vs_tiff[
                            "mae"
                        ]
                    ),

                    **visual,
                })


                records.append({
                    "tile_index": (
                        tile_index
                    ),

                    "saved_image": (
                        saved_image
                    ),

                    "openslide_image": (
                        openslide_image
                    ),

                    "tiff_image": (
                        tiff_image
                    ),

                    "openslide_vs_tiff_mae": (
                        openslide_vs_tiff[
                            "mae"
                        ]
                    ),
                })


                if (
                    row_number % 8
                    == 0
                ):

                    print(
                        f"  decoded "
                        f"{row_number}/64"
                    )


    # ========================================================
    # SAVE METRICS
    # ========================================================

    metrics = pd.DataFrame(
        metric_rows
    )


    metrics = metrics.sort_values(
        "openslide_vs_tiff_mae",
        ascending=False,
    )


    metrics_path = (
        case_output
        / "decoder_metrics.csv"
    )


    metrics.to_csv(
        metrics_path,
        index=False,
    )


    # ========================================================
    # SAVE CONTACT SHEETS
    # ========================================================

    saved_contact = (
        case_output
        / "A_v2prod_saved.jpg"
    )

    openslide_contact = (
        case_output
        / "B_openslide_reread.jpg"
    )

    tiff_contact = (
        case_output
        / "C_tifffile_direct.jpg"
    )


    create_numbered_contact_sheet(
        records,
        "saved_image",
        saved_contact,
        (
            f"{case_id} — "
            "A: V2_PROD saved"
        ),
    )


    create_numbered_contact_sheet(
        records,
        "openslide_image",
        openslide_contact,
        (
            f"{case_id} — "
            "B: OpenSlide reread"
        ),
    )


    create_numbered_contact_sheet(
        records,
        "tiff_image",
        tiff_contact,
        (
            f"{case_id} — "
            "C: tifffile direct decode"
        ),
    )


    overview_path = (
        case_output
        / "D_three_way_overview.jpg"
    )


    create_three_way_overview(
        saved_contact,
        openslide_contact,
        tiff_contact,
        overview_path,
        case_id,
        metadata[
            "compression_text"
        ],
        metadata[
            "photometric_text"
        ],
    )


    top_difference_path = (
        case_output
        / "E_top_decoder_differences.jpg"
    )


    create_top_difference_sheet(
        records,
        top_difference_path,
        case_id,
    )


    # ========================================================
    # SUMMARY
    # ========================================================

    summary = {
        "case_submitter_id": (
            case_id
        ),

        "wsi_file_id": (
            file_id
        ),

        "compression_value": (
            metadata[
                "compression_value"
            ]
        ),

        "compression_text": (
            metadata[
                "compression_text"
            ]
        ),

        "photometric_value": (
            metadata[
                "photometric_value"
            ]
        ),

        "photometric_text": (
            metadata[
                "photometric_text"
            ]
        ),

        "tile_width": (
            metadata[
                "tile_width"
            ]
        ),

        "tile_length": (
            metadata[
                "tile_length"
            ]
        ),

        "ycbcr_subsampling": (
            metadata[
                "ycbcr_subsampling"
            ]
        ),

        "tifffile_version": (
            metadata[
                "tifffile_version"
            ]
        ),

        "imagecodecs_version": (
            metadata[
                "imagecodecs_version"
            ]
        ),

        "jpeg2k_codec_version": (
            metadata[
                "jpeg2k_codec_version"
            ]
        ),

        "tiles_compared": int(
            len(
                metrics
            )
        ),

        "saved_vs_openslide_mae_mean": float(
            metrics[
                "saved_vs_openslide_mae"
            ].mean()
        ),

        "saved_vs_openslide_mae_max": float(
            metrics[
                "saved_vs_openslide_mae"
            ].max()
        ),

        "openslide_vs_tiff_mae_mean": float(
            metrics[
                "openslide_vs_tiff_mae"
            ].mean()
        ),

        "openslide_vs_tiff_mae_median": float(
            metrics[
                "openslide_vs_tiff_mae"
            ].median()
        ),

        "openslide_vs_tiff_mae_p95": float(
            metrics[
                "openslide_vs_tiff_mae"
            ].quantile(
                0.95
            )
        ),

        "openslide_vs_tiff_mae_max": float(
            metrics[
                "openslide_vs_tiff_mae"
            ].max()
        ),

        "tiles_mae_gt_1": int(
            (
                metrics[
                    "openslide_vs_tiff_mae"
                ]
                > 1.0
            ).sum()
        ),

        "tiles_mae_gt_5": int(
            (
                metrics[
                    "openslide_vs_tiff_mae"
                ]
                > 5.0
            ).sum()
        ),

        "tiles_mae_gt_20": int(
            (
                metrics[
                    "openslide_vs_tiff_mae"
                ]
                > 20.0
            ).sum()
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
            default=str,
        ),
        encoding="utf-8",
    )


    print()
    print(
        "Decoder comparison:"
    )

    print(
        "  saved/OpenSlide MAE mean: "
        f"{summary['saved_vs_openslide_mae_mean']:.4f}"
    )

    print(
        "  saved/OpenSlide MAE max:  "
        f"{summary['saved_vs_openslide_mae_max']:.4f}"
    )

    print(
        "  OpenSlide/tiff MAE mean:  "
        f"{summary['openslide_vs_tiff_mae_mean']:.4f}"
    )

    print(
        "  OpenSlide/tiff MAE p95:   "
        f"{summary['openslide_vs_tiff_mae_p95']:.4f}"
    )

    print(
        "  OpenSlide/tiff MAE max:   "
        f"{summary['openslide_vs_tiff_mae_max']:.4f}"
    )

    print(
        "  tiles MAE > 1:            "
        f"{summary['tiles_mae_gt_1']}/64"
    )

    print(
        "  tiles MAE > 5:            "
        f"{summary['tiles_mae_gt_5']}/64"
    )

    print(
        "  tiles MAE > 20:           "
        f"{summary['tiles_mae_gt_20']}/64"
    )

    print(
        f"  overview:                 "
        f"{overview_path}"
    )


    # ========================================================
    # CLOSE PIL IMAGES
    # ========================================================

    for record in records:

        for key in [
            "saved_image",
            "openslide_image",
            "tiff_image",
        ]:

            try:

                record[
                    key
                ].close()

            except Exception:

                pass


    # ========================================================
    # CLEAN WSI
    # ========================================================

    if not keep_wsi:

        remove_download(
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


    return summary


# ============================================================
# GLOBAL SUMMARY
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Compare OpenSlide decoding of selected "
            "Aperio SVS patches with direct TIFF "
            "segment decoding using tifffile/imagecodecs."
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


    if not GDC_CLIENT.exists():

        raise FileNotFoundError(
            f"GDC client not found: "
            f"{GDC_CLIENT}"
        )


    if not WSI_CANDIDATES_FILE.exists():

        raise FileNotFoundError(
            f"WSI manifest not found: "
            f"{WSI_CANDIDATES_FILE}"
        )


    wsi_candidates = pd.read_csv(
        WSI_CANDIDATES_FILE
    )


    print()
    print("=" * 80)
    print(
        " APERIO SVS DECODER AUDIT"
    )
    print("=" * 80)

    print(
        f"Cases:             "
        f"{len(args.cases)}"
    )

    print(
        f"tifffile:          "
        f"{tifffile.__version__}"
    )

    print(
        f"imagecodecs:        "
        f"{imagecodecs.__version__}"
    )

    try:

        print(
            f"JPEG2000 backend:   "
            f"{imagecodecs.jpeg2k_version()}"
        )

    except Exception:

        print(
            "JPEG2000 backend:   unavailable"
        )

    print(
        f"Keep WSI:          "
        f"{args.keep_wsi}"
    )

    print(
        f"Free disk:         "
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


            summary[
                "status"
            ] = "PASS"


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


    summary_df = pd.DataFrame(
        summaries
    )


    csv_path = (
        AUDIT_METADATA_ROOT
        / "svs_decoder_audit.csv"
    )


    summary_df.to_csv(
        csv_path,
        index=False,
    )


    json_path = (
        AUDIT_METADATA_ROOT
        / "svs_decoder_audit.json"
    )


    json_path.write_text(
        json.dumps(
            summaries,
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )


    print()
    print("=" * 80)
    print(
        " DECODER AUDIT COMPLETE"
    )
    print("=" * 80)


    for summary in summaries:

        case_id = (
            summary[
                "case_submitter_id"
            ]
        )


        if (
            summary[
                "status"
            ]
            == "ERROR"
        ):

            print(
                f"{case_id}: ERROR — "
                f"{summary['error']}"
            )

            continue


        print(
            f"{case_id}: "
            f"compression="
            f"{summary['compression_value']}, "
            f"OpenSlide/tiff MAE="
            f"{summary['openslide_vs_tiff_mae_mean']:.3f}, "
            f"max="
            f"{summary['openslide_vs_tiff_mae_max']:.3f}"
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

    print(
        "Figures:"
    )

    print(
        f"  {OUTPUT_ROOT}"
    )

    print("=" * 80)


if __name__ == "__main__":
    main()