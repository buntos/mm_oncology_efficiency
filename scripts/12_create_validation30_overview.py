from pathlib import Path
import argparse
import math

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


# ============================================================
# ROOT
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

MANIFEST_DIR = ROOT / "data" / "manifests"
METADATA_DIR = ROOT / "data" / "metadata"

CONTACT_ROOT = (
    ROOT
    / "results"
    / "figures"
    / "production_v2_contact_sheets"
)

OUTPUT_ROOT = (
    ROOT
    / "results"
    / "figures"
)


# ============================================================
# DEFAULT LAYOUT
# ============================================================

DEFAULT_COLUMNS = 3
DEFAULT_ROWS = 2

DEFAULT_PREVIEW_SIZE = 640

PAGE_MARGIN = 24
CELL_GAP = 18

LABEL_HEIGHT = 64
HEADER_HEIGHT = 72

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
# HELPERS
# ============================================================

def safe_text(value):

    if pd.isna(value):
        return ""

    return str(value)


def build_patient_table(
    batch_manifest,
    patient_qc,
):

    batch = pd.read_csv(
        batch_manifest
    )

    qc = pd.read_csv(
        patient_qc
    )


    # Keep the exact frozen batch membership/order.
    patient_table = batch[
        [
            "case_submitter_id",
            "class_label",
        ]
    ].copy()


    qc_columns = [
        "case_submitter_id",
        "status",
        "selected_slide_token",
        "valid_tiles",
    ]


    existing_qc_columns = [
        column
        for column
        in qc_columns
        if column in qc.columns
    ]


    qc_small = qc[
        existing_qc_columns
    ].copy()


    patient_table = patient_table.merge(
        qc_small,
        on="case_submitter_id",
        how="left",
    )


    return patient_table


def find_contact_sheet(
    contact_dir,
    case_id,
):

    expected = (
        contact_dir
        / f"{case_id}_64_tiles.jpg"
    )

    if expected.exists():
        return expected


    # Defensive fallback in case extension differs.
    candidates = list(
        contact_dir.glob(
            f"{case_id}_64_tiles.*"
        )
    )

    if len(candidates) == 1:
        return candidates[0]

    return None


def create_page(
    page_df,
    page_number,
    total_pages,
    contact_dir,
    output_path,
    columns,
    rows,
    preview_size,
):

    cell_width = preview_size

    cell_height = (
        preview_size
        + LABEL_HEIGHT
    )


    page_width = (
        PAGE_MARGIN * 2
        +
        columns * cell_width
        +
        (columns - 1) * CELL_GAP
    )


    page_height = (
        HEADER_HEIGHT
        +
        PAGE_MARGIN
        +
        rows * cell_height
        +
        (rows - 1) * CELL_GAP
        +
        PAGE_MARGIN
    )


    page = Image.new(
        "RGB",
        (
            page_width,
            page_height,
        ),
        "white",
    )


    draw = ImageDraw.Draw(
        page
    )


    header_font = load_font(
        28
    )

    patient_font = load_font(
        22
    )

    detail_font = load_font(
        18
    )


    title = (
        "V2_PROD validation30 visual QC "
        f"— page {page_number}/{total_pages}"
    )


    draw.text(
        (
            PAGE_MARGIN,
            18,
        ),
        title,
        fill="black",
        font=header_font,
    )


    for local_index, (
        _,
        patient,
    ) in enumerate(
        page_df.iterrows()
    ):

        row_index = (
            local_index
            // columns
        )

        column_index = (
            local_index
            % columns
        )


        x = (
            PAGE_MARGIN
            +
            column_index
            * (
                cell_width
                +
                CELL_GAP
            )
        )


        y = (
            HEADER_HEIGHT
            +
            PAGE_MARGIN
            +
            row_index
            * (
                cell_height
                +
                CELL_GAP
            )
        )


        case_id = safe_text(
            patient[
                "case_submitter_id"
            ]
        )

        class_label = safe_text(
            patient[
                "class_label"
            ]
        )

        status = safe_text(
            patient.get(
                "status",
                "",
            )
        )

        selected_slide = safe_text(
            patient.get(
                "selected_slide_token",
                "",
            )
        )

        valid_tiles = safe_text(
            patient.get(
                "valid_tiles",
                "",
            )
        )


        contact_path = (
            find_contact_sheet(
                contact_dir,
                case_id,
            )
        )


        if contact_path is not None:

            with Image.open(
                contact_path
            ) as image:

                image = image.convert(
                    "RGB"
                )

                image.thumbnail(
                    (
                        preview_size,
                        preview_size,
                    ),
                    Image.Resampling.LANCZOS,
                )


                # White backing in case aspect ratio changes.
                preview = Image.new(
                    "RGB",
                    (
                        preview_size,
                        preview_size,
                    ),
                    "white",
                )


                paste_x = (
                    preview_size
                    - image.width
                ) // 2

                paste_y = (
                    preview_size
                    - image.height
                ) // 2


                preview.paste(
                    image,
                    (
                        paste_x,
                        paste_y,
                    ),
                )


                page.paste(
                    preview,
                    (
                        x,
                        y,
                    ),
                )


        else:

            # Make missing contact sheets very obvious.
            draw.rectangle(
                (
                    x,
                    y,
                    x + preview_size,
                    y + preview_size,
                ),
                outline="black",
                width=3,
            )

            draw.text(
                (
                    x + 20,
                    y + 20,
                ),
                "CONTACT SHEET MISSING",
                fill="black",
                font=patient_font,
            )


        label_y = (
            y
            + preview_size
        )


        draw.rectangle(
            (
                x,
                label_y,
                x + preview_size,
                label_y + LABEL_HEIGHT,
            ),
            fill="white",
            outline="black",
            width=1,
        )


        line_1 = (
            f"{case_id}  |  {class_label}"
        )


        details = [
            f"status={status}",
            f"slide={selected_slide}",
        ]


        if valid_tiles:

            details.append(
                f"tiles={valid_tiles}"
            )


        line_2 = "  |  ".join(
            details
        )


        draw.text(
            (
                x + 7,
                label_y + 5,
            ),
            line_1,
            fill="black",
            font=patient_font,
        )


        draw.text(
            (
                x + 7,
                label_y + 36,
            ),
            line_2,
            fill="black",
            font=detail_font,
        )


    page.save(
        output_path,
        format="JPEG",
        quality=JPEG_QUALITY,
        subsampling=0,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Create overview pages from V2_PROD "
            "patient contact sheets."
        )
    )


    parser.add_argument(
        "--batch-name",
        default="validation30_v2prod",
    )


    parser.add_argument(
        "--columns",
        type=int,
        default=DEFAULT_COLUMNS,
    )


    parser.add_argument(
        "--rows",
        type=int,
        default=DEFAULT_ROWS,
    )


    parser.add_argument(
        "--preview-size",
        type=int,
        default=DEFAULT_PREVIEW_SIZE,
    )


    args = parser.parse_args()


    batch_manifest = (
        MANIFEST_DIR
        / f"{args.batch_name}_patients.csv"
    )


    patient_qc = (
        METADATA_DIR
        / f"{args.batch_name}_patient_qc.csv"
    )


    contact_dir = (
        CONTACT_ROOT
        / args.batch_name
    )


    output_dir = (
        OUTPUT_ROOT
        / f"{args.batch_name}_overview"
    )


    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    # --------------------------------------------------------
    # VALIDATE INPUTS
    # --------------------------------------------------------

    if not batch_manifest.exists():

        raise FileNotFoundError(
            f"Batch manifest not found: "
            f"{batch_manifest}"
        )


    if not patient_qc.exists():

        raise FileNotFoundError(
            f"Patient QC file not found: "
            f"{patient_qc}"
        )


    if not contact_dir.exists():

        raise FileNotFoundError(
            f"Contact sheet directory not found: "
            f"{contact_dir}"
        )


    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    patients = build_patient_table(
        batch_manifest,
        patient_qc,
    )


    patients_per_page = (
        args.columns
        * args.rows
    )


    total_pages = math.ceil(
        len(patients)
        / patients_per_page
    )


    # --------------------------------------------------------
    # AUDIT CONTACT SHEETS
    # --------------------------------------------------------

    audit_rows = []


    for _, patient in (
        patients.iterrows()
    ):

        case_id = (
            patient[
                "case_submitter_id"
            ]
        )


        contact_path = (
            find_contact_sheet(
                contact_dir,
                case_id,
            )
        )


        audit_rows.append({
            "case_submitter_id": (
                case_id
            ),

            "class_label": (
                patient[
                    "class_label"
                ]
            ),

            "status": (
                patient.get(
                    "status",
                    None,
                )
            ),

            "selected_slide_token": (
                patient.get(
                    "selected_slide_token",
                    None,
                )
            ),

            "contact_sheet_found": (
                contact_path
                is not None
            ),

            "contact_sheet_path": (
                str(contact_path)
                if contact_path
                is not None
                else None
            ),
        })


    audit_df = pd.DataFrame(
        audit_rows
    )


    audit_path = (
        output_dir
        / "overview_contact_sheet_audit.csv"
    )


    audit_df.to_csv(
        audit_path,
        index=False,
    )


    found_count = int(
        audit_df[
            "contact_sheet_found"
        ].sum()
    )


    missing_count = int(
        (
            ~audit_df[
                "contact_sheet_found"
            ]
        ).sum()
    )


    # --------------------------------------------------------
    # CREATE PAGES
    # --------------------------------------------------------

    print()
    print("=" * 78)
    print(
        " V2_PROD VISUAL QC OVERVIEW"
    )
    print("=" * 78)

    print(
        f"Batch:                    "
        f"{args.batch_name}"
    )

    print(
        f"Patients:                 "
        f"{len(patients)}"
    )

    print(
        f"Contact sheets found:     "
        f"{found_count}"
    )

    print(
        f"Contact sheets missing:   "
        f"{missing_count}"
    )

    print(
        f"Patients per page:        "
        f"{patients_per_page}"
    )

    print(
        f"Overview pages:           "
        f"{total_pages}"
    )

    print()


    for page_index in range(
        total_pages
    ):

        start = (
            page_index
            * patients_per_page
        )

        end = min(
            start
            + patients_per_page,
            len(patients),
        )


        page_df = (
            patients.iloc[
                start:end
            ]
            .copy()
        )


        page_number = (
            page_index + 1
        )


        output_path = (
            output_dir
            / (
                f"{args.batch_name}"
                f"_overview_"
                f"{page_number:02d}.jpg"
            )
        )


        create_page(
            page_df=page_df,
            page_number=page_number,
            total_pages=total_pages,
            contact_dir=contact_dir,
            output_path=output_path,
            columns=args.columns,
            rows=args.rows,
            preview_size=args.preview_size,
        )


        print(
            f"Page {page_number}/{total_pages}: "
            f"{output_path.name}"
        )


    # --------------------------------------------------------
    # FINAL
    # --------------------------------------------------------

    print()
    print("=" * 78)
    print(
        " OVERVIEW COMPLETE"
    )
    print("=" * 78)

    print(
        f"Output directory:"
    )

    print(
        f"  {output_dir}"
    )

    print()
    print(
        f"Audit:"
    )

    print(
        f"  {audit_path}"
    )

    print("=" * 78)


if __name__ == "__main__":
    main()