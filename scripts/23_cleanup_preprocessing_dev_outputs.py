from pathlib import Path
import argparse
import datetime
import json
import shutil
import zipfile


# ============================================================
# PROJECT ROOT
# ============================================================

THIS_FILE = Path(__file__).resolve()
ROOT = THIS_FILE.parents[1]


# ============================================================
# REQUIRED PROJECT STRUCTURE
# ============================================================

REQUIRED_PROJECT_DIRS = [
    ROOT / "scripts",
    ROOT / "data",
    ROOT / "results",
]


# ============================================================
# IMPORTANT EVIDENCE
# ============================================================
#
# Cleanup stops if one of these files is missing,
# unless --force is explicitly supplied.
#
# These files represent the key evidence required to reproduce
# and explain the preprocessing decisions made before the
# final full-cohort run.
# ============================================================

IMPORTANT_EVIDENCE = [
    ROOT
    / "data"
    / "manifests"
    / "paired_cohort_stage2.csv",

    ROOT
    / "data"
    / "manifests"
    / "wsi_primary_diagnostic_candidates.csv",

    ROOT
    / "data"
    / "manifests"
    / "rna_primary_tumor_candidates.csv",

    ROOT
    / "results"
    / "final_tissue_threshold_audit_v1"
    / "selected_tiles_by_threshold.csv",

    ROOT
    / "results"
    / "tissue_qc_failure_reason_audit_v1"
    / "global_failure_reason_summary.csv",

    ROOT
    / "results"
    / "final_tissue_distribution_audit_v1"
    / "global_scenario_summary.csv",

    ROOT
    / "data"
    / "metadata"
    / "independent30_final_tissue080_v1"
    / "frozen_run_configuration.json",
]


# ============================================================
# KNOWN DEVELOPMENT TILE OUTPUTS
# ============================================================
#
# These are old development/preprocessing namespaces.
#
# The upcoming final full-cohort run should use a new clean
# namespace, so these outputs can safely be removed after
# evidence has been archived.
# ============================================================

KNOWN_TILE_DIRS = [
    ROOT
    / "data"
    / "tiles"
    / "production_v2",

    ROOT
    / "data"
    / "tiles"
    / "production_v2_artifact_qc_candidate_v1",

    ROOT
    / "data"
    / "tiles"
    / "pilot_v2",

    ROOT
    / "data"
    / "tiles"
    / "pilot",

    ROOT
    / "data"
    / "tiles"
    / "validation30_v2prod",
]


# ============================================================
# RAW PILOT DOWNLOAD
# ============================================================
#
# This directory contains the original pilot WSI/RNA downloads.
#
# The important pilot manifests and metadata are preserved.
# The raw files themselves can be downloaded again from GDC
# if ever required.
# ============================================================

PILOT_DOWNLOAD_DIR = (
    ROOT
    / "data"
    / "pilot_download"
)


# ============================================================
# TEMPORARY DOWNLOAD DIRECTORIES
# ============================================================

TEMP_PATTERNS = [
    "temp_gdc*",
    "tmp_gdc*",
]


# ============================================================
# DEVELOPMENT FIGURE DIRECTORIES
# ============================================================

FULLY_DELETABLE_FIGURE_DIR_NAMES = {
    "production_v2_contact_sheets",
    "production_v2_masks",
    "production_v2_artifact_qc_contact_sheets",
    "production_v2_artifact_qc_masks",
    "production_v2_artifact_qc_rejected",
    "validation30_v2prod",
    "validation30_v2prod_overview",
    "validation30_tile_artifact_audit",
    "pilot_v2",
}


# ============================================================
# IMPORTANT FIGURES TO PRESERVE IN EVIDENCE ARCHIVE
# ============================================================

IMPORTANT_FIGURE_FILES = [
    ROOT
    / "results"
    / "figures"
    / "final_tissue_threshold_audit_v1"
    / "all_cases_threshold_overview.jpg",
]


# ============================================================
# HELPERS
# ============================================================

def path_size_bytes(path):

    if not path.exists():
        return 0

    if path.is_file():

        try:
            return path.stat().st_size

        except OSError:
            return 0

    total = 0

    try:

        iterator = path.rglob("*")

        for item in iterator:

            try:

                if (
                    item.is_file()
                    and
                    not item.is_symlink()
                ):

                    total += (
                        item.stat().st_size
                    )

            except OSError:
                pass

    except OSError:
        pass

    return total


def human_size(size):

    value = float(
        size
    )

    units = [
        "B",
        "KB",
        "MB",
        "GB",
        "TB",
    ]

    for unit in units:

        if (
            value < 1024.0
            or
            unit == units[-1]
        ):

            return (
                f"{value:.2f} {unit}"
            )

        value /= 1024.0

    return (
        f"{size} B"
    )


def safe_relative(path):

    try:

        return (
            path.resolve()
            .relative_to(
                ROOT.resolve()
            )
        )

    except Exception:

        return path


def validate_delete_path(path):

    path = path.resolve()
    root = ROOT.resolve()

    try:

        relative = (
            path.relative_to(
                root
            )
        )

    except ValueError:

        raise RuntimeError(
            "REFUSING TO DELETE PATH "
            "OUTSIDE PROJECT:\n"
            f"{path}"
        )

    if path == root:

        raise RuntimeError(
            "REFUSING TO DELETE "
            "PROJECT ROOT"
        )

    # Require at least:
    #
    # data/foo
    # scripts/__pycache__
    # results/foo
    #
    # This avoids accidental broad deletion such as data/.
    if len(relative.parts) < 2:

        raise RuntimeError(
            "REFUSING SUSPICIOUS "
            "DELETION TARGET:\n"
            f"{path}"
        )

    # Never touch the virtual environment.
    if (
        relative.parts
        and
        relative.parts[0].lower()
        ==
        ".venv"
    ):

        raise RuntimeError(
            "REFUSING TO DELETE "
            "ANYTHING INSIDE .venv:\n"
            f"{path}"
        )

    return True


def delete_path(path):

    validate_delete_path(
        path
    )

    if not path.exists():
        return

    if path.is_symlink():

        path.unlink()
        return

    if path.is_file():

        path.unlink()
        return

    shutil.rmtree(
        path
    )


# ============================================================
# VERIFY PROJECT
# ============================================================

def verify_project():

    print()
    print(
        "=" * 88
    )

    print(
        " PROJECT SAFETY CHECK"
    )

    print(
        "=" * 88
    )

    print(
        "Project root:"
    )

    print(
        f"  {ROOT}"
    )

    missing_dirs = [
        path
        for path
        in REQUIRED_PROJECT_DIRS
        if not path.exists()
    ]

    if missing_dirs:

        text = "\n".join(
            str(path)
            for path
            in missing_dirs
        )

        raise RuntimeError(
            "Project structure check failed. "
            "Missing:\n"
            f"{text}"
        )

    if ROOT.resolve() == Path(
        ROOT.anchor
    ).resolve():

        raise RuntimeError(
            "Invalid project root."
        )

    print()
    print(
        "Project structure: PASS"
    )


# ============================================================
# VERIFY EVIDENCE
# ============================================================

def verify_evidence():

    print()
    print(
        "=" * 88
    )

    print(
        " EVIDENCE CHECK"
    )

    print(
        "=" * 88
    )

    missing = []

    for path in IMPORTANT_EVIDENCE:

        relative = safe_relative(
            path
        )

        if path.exists():

            print(
                f"[KEEP] {relative}"
            )

        else:

            print(
                f"[MISSING] {relative}"
            )

            missing.append(
                path
            )

    return missing


# ============================================================
# COLLECT EVIDENCE FILES
# ============================================================

def collect_evidence_files():

    files = set()

    # --------------------------------------------------------
    # Core manifests
    # --------------------------------------------------------

    manifest_root = (
        ROOT
        / "data"
        / "manifests"
    )

    if manifest_root.exists():

        for path in manifest_root.rglob(
            "*"
        ):

            if (
                path.is_file()
                and
                path.suffix.lower()
                in {
                    ".csv",
                    ".json",
                    ".txt",
                    ".tsv",
                }
            ):

                files.add(
                    path
                )

    # --------------------------------------------------------
    # Metadata / QC / frozen configurations
    # --------------------------------------------------------

    metadata_root = (
        ROOT
        / "data"
        / "metadata"
    )

    if metadata_root.exists():

        for path in metadata_root.rglob(
            "*"
        ):

            if (
                path.is_file()
                and
                path.suffix.lower()
                in {
                    ".csv",
                    ".json",
                    ".txt",
                    ".log",
                    ".tsv",
                }
            ):

                files.add(
                    path
                )

    # --------------------------------------------------------
    # Audit results
    # --------------------------------------------------------

    audit_dirs = [
        ROOT
        / "results"
        / "final_tissue_threshold_audit_v1",

        ROOT
        / "results"
        / "tissue_qc_failure_reason_audit_v1",

        ROOT
        / "results"
        / "final_tissue_distribution_audit_v1",
    ]

    for audit_dir in audit_dirs:

        if not audit_dir.exists():
            continue

        for path in audit_dir.rglob(
            "*"
        ):

            if (
                path.is_file()
                and
                path.suffix.lower()
                in {
                    ".csv",
                    ".json",
                    ".txt",
                    ".tsv",
                }
            ):

                files.add(
                    path
                )

    # --------------------------------------------------------
    # Important overview figures
    # --------------------------------------------------------

    for path in IMPORTANT_FIGURE_FILES:

        if path.exists():

            files.add(
                path
            )

    return sorted(
        files,
        key=lambda item: str(
            item
        ),
    )


# ============================================================
# CREATE EVIDENCE ZIP
# ============================================================

def create_evidence_archive():

    archive_dir = (
        ROOT
        / "artifacts"
        / "evidence"
    )

    archive_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = (
        datetime.datetime.now()
        .strftime(
            "%Y%m%d_%H%M%S"
        )
    )

    archive_path = (
        archive_dir
        /
        (
            "preprocessing_dev_evidence_"
            f"{timestamp}.zip"
        )
    )

    files = (
        collect_evidence_files()
    )

    print()
    print(
        "=" * 88
    )

    print(
        " CREATE EVIDENCE ARCHIVE"
    )

    print(
        "=" * 88
    )

    print(
        f"Files: {len(files)}"
    )

    print(
        "Archive:"
    )

    print(
        f"  {archive_path}"
    )

    with zipfile.ZipFile(
        archive_path,
        mode="w",
        compression=(
            zipfile.ZIP_DEFLATED
        ),
        compresslevel=6,
    ) as archive:

        for path in files:

            relative = (
                path.relative_to(
                    ROOT
                )
            )

            archive.write(
                path,
                arcname=str(
                    relative
                ),
            )

    print(
        "Archive size: "
        f"{human_size(path_size_bytes(archive_path))}"
    )

    return archive_path


# ============================================================
# BUILD DELETE PLAN
# ============================================================

def build_delete_plan(
    drop_rna_cache,
):

    targets = set()

    # --------------------------------------------------------
    # Known development tile directories
    # --------------------------------------------------------

    for path in KNOWN_TILE_DIRS:

        if path.exists():

            targets.add(
                path
            )

    # --------------------------------------------------------
    # Detect other obvious development tile namespaces
    # --------------------------------------------------------

    tiles_root = (
        ROOT
        / "data"
        / "tiles"
    )

    if tiles_root.exists():

        for child in tiles_root.iterdir():

            if not child.is_dir():
                continue

            name = (
                child.name.lower()
            )

            if (
                "pilot" in name
                or
                "validation" in name
                or
                "smoke" in name
                or
                "artifact_qc_candidate"
                in name
                or
                name == "production_v2"
            ):

                targets.add(
                    child
                )

    # --------------------------------------------------------
    # Raw pilot downloads
    # --------------------------------------------------------

    if PILOT_DOWNLOAD_DIR.exists():

        targets.add(
            PILOT_DOWNLOAD_DIR
        )

    # --------------------------------------------------------
    # Temporary GDC directories
    # --------------------------------------------------------

    data_root = (
        ROOT
        / "data"
    )

    if data_root.exists():

        for pattern in TEMP_PATTERNS:

            for path in data_root.glob(
                pattern
            ):

                if path.exists():

                    targets.add(
                        path
                    )

    # --------------------------------------------------------
    # Development figures
    # --------------------------------------------------------

    figures_root = (
        ROOT
        / "results"
        / "figures"
    )

    if figures_root.exists():

        for child in figures_root.iterdir():

            if not child.is_dir():
                continue

            name = (
                child.name.lower()
            )

            if (
                child.name
                in
                FULLY_DELETABLE_FIGURE_DIR_NAMES
                or
                "production_v2_artifact_qc_contact"
                in name
                or
                "production_v2_artifact_qc_rejected"
                in name
                or
                "validation30"
                in name
                or
                "smoke"
                in name
            ):

                targets.add(
                    child
                )

    # --------------------------------------------------------
    # RNA caches — OPTIONAL
    # --------------------------------------------------------

    if drop_rna_cache:

        rna_candidates = [
            ROOT
            / "data"
            / "rna"
            / "production_v2",

            ROOT
            / "data"
            / "rna"
            / "production_v2_artifact_qc_candidate_v1",

            ROOT
            / "data"
            / "rna"
            / "pilot",

            ROOT
            / "data"
            / "rna"
            / "pilot_v2",

            ROOT
            / "data"
            / "rna"
            / "validation30_v2prod",
        ]

        for path in rna_candidates:

            if path.exists():

                targets.add(
                    path
                )

    # --------------------------------------------------------
    # Python cache ONLY in project code
    #
    # IMPORTANT:
    # .venv is deliberately excluded.
    # --------------------------------------------------------

    project_cache_roots = [
        ROOT / "scripts",
        ROOT / "src",
    ]

    for cache_root in project_cache_roots:

        if not cache_root.exists():
            continue

        for path in cache_root.rglob(
            "__pycache__"
        ):

            if path.is_dir():

                targets.add(
                    path
                )

    # --------------------------------------------------------
    # Remove nested duplicate deletion targets
    # --------------------------------------------------------

    ordered = sorted(
        targets,
        key=lambda path: len(
            path.parts
        ),
    )

    final_targets = []

    for path in ordered:

        path_resolved = (
            path.resolve()
        )

        # Extra protection against .venv.
        try:

            relative = (
                path_resolved
                .relative_to(
                    ROOT.resolve()
                )
            )

        except ValueError:

            continue

        if (
            relative.parts
            and
            relative.parts[0].lower()
            ==
            ".venv"
        ):

            continue

        already_covered = False

        for parent in final_targets:

            try:

                path_resolved.relative_to(
                    parent.resolve()
                )

                already_covered = True
                break

            except ValueError:
                pass

        if not already_covered:

            final_targets.append(
                path
            )

    return final_targets


# ============================================================
# OPTIONAL AUDIT FIGURE PRUNING
# ============================================================

def build_audit_figure_prune_plan():

    files = []

    # --------------------------------------------------------
    # Final tissue threshold audit
    #
    # Keep:
    # - global overview
    # - masks
    # - per-case comparison overviews
    #
    # Remove:
    # - individual threshold contact sheets
    # --------------------------------------------------------

    threshold_figures = (
        ROOT
        / "results"
        / "figures"
        / "final_tissue_threshold_audit_v1"
    )

    if threshold_figures.exists():

        for path in threshold_figures.rglob(
            "*"
        ):

            if not path.is_file():
                continue

            name = (
                path.name.lower()
            )

            if (
                name
                ==
                "all_cases_threshold_overview.jpg"
            ):
                continue

            if (
                "thumbnail_tissue_mask"
                in name
            ):
                continue

            if (
                "threshold_comparison_overview"
                in name
            ):
                continue

            if "_contact." in name:

                files.append(
                    path
                )

    # --------------------------------------------------------
    # Tissue QC failure reason audit
    #
    # Keep CSV metrics.
    # Individual contact sheets are reproducible.
    # --------------------------------------------------------

    failure_figures = (
        ROOT
        / "results"
        / "figures"
        / "tissue_qc_failure_reason_audit_v1"
    )

    if failure_figures.exists():

        for path in failure_figures.rglob(
            "*"
        ):

            if not path.is_file():
                continue

            name = (
                path.name.lower()
            )

            if "_contact." in name:

                files.append(
                    path
                )

    return files


# ============================================================
# PRINT LARGEST FILES
# ============================================================

def print_largest_files(
    limit=25,
):

    print()
    print(
        "=" * 88
    )

    print(
        " LARGEST FILES CURRENTLY IN PROJECT"
    )

    print(
        "=" * 88
    )

    entries = []

    for path in ROOT.rglob(
        "*"
    ):

        try:

            if (
                path.is_file()
                and
                not path.is_symlink()
            ):

                entries.append(
                    (
                        path.stat().st_size,
                        path,
                    )
                )

        except OSError:
            pass

    entries.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    for size, path in entries[
        :limit
    ]:

        relative = safe_relative(
            path
        )

        print(
            f"{human_size(size):>12}  "
            f"{relative}"
        )


# ============================================================
# PRINT CLEANUP PLAN
# ============================================================

def print_cleanup_plan(
    targets,
    prune_files,
):

    print()
    print(
        "=" * 88
    )

    print(
        " CLEANUP PLAN"
    )

    print(
        "=" * 88
    )

    total_bytes = 0

    if (
        not targets
        and
        not prune_files
    ):

        print(
            "Nothing matched cleanup rules."
        )

    for path in targets:

        size = (
            path_size_bytes(
                path
            )
        )

        total_bytes += size

        print(
            "[DELETE DIR] "
            f"{human_size(size):>12}  "
            f"{safe_relative(path)}"
        )

    for path in prune_files:

        size = (
            path_size_bytes(
                path
            )
        )

        total_bytes += size

        print(
            "[DELETE FILE] "
            f"{human_size(size):>12}  "
            f"{safe_relative(path)}"
        )

    print()

    print(
        "Potential space recovered: "
        f"{human_size(total_bytes)}"
    )

    return total_bytes


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Safely inspect and remove heavy "
            "development preprocessing outputs "
            "before the final full-cohort run."
        )
    )

    parser.add_argument(
        "--apply",
        action="store_true",
        help=(
            "Actually delete files. "
            "Without this flag the script "
            "performs dry-run only."
        ),
    )

    parser.add_argument(
        "--drop-rna-cache",
        action="store_true",
        help=(
            "Also remove old production/pilot "
            "RNA caches. "
            "By default RNA caches are preserved."
        ),
    )

    parser.add_argument(
        "--prune-audit-contact-sheets",
        action="store_true",
        help=(
            "Delete large per-threshold audit "
            "contact sheets while preserving "
            "CSV/JSON evidence, masks and overviews."
        ),
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Allow cleanup even if an expected "
            "evidence file is missing."
        ),
    )

    parser.add_argument(
        "--skip-evidence-archive",
        action="store_true",
        help=(
            "Do not create the safety evidence ZIP. "
            "Not recommended."
        ),
    )

    args = parser.parse_args()


    # ========================================================
    # VERIFY PROJECT
    # ========================================================

    verify_project()


    # ========================================================
    # VERIFY EVIDENCE
    # ========================================================

    missing_evidence = (
        verify_evidence()
    )

    if (
        missing_evidence
        and
        not args.force
    ):

        print()

        print(
            "Cleanup stopped because "
            "important evidence files "
            "are missing."
        )

        print()

        print(
            "Inspect the list above."
        )

        print(
            "If the missing files are "
            "intentional, rerun with --force."
        )

        return


    # ========================================================
    # CREATE SAFETY ARCHIVE
    # ========================================================

    archive_path = None

    if not args.skip_evidence_archive:

        archive_path = (
            create_evidence_archive()
        )


    # ========================================================
    # BUILD DELETE PLAN
    # ========================================================

    targets = (
        build_delete_plan(
            drop_rna_cache=(
                args.drop_rna_cache
            )
        )
    )

    prune_files = []

    if (
        args.prune_audit_contact_sheets
    ):

        prune_files = (
            build_audit_figure_prune_plan()
        )


    # ========================================================
    # SHOW PLAN
    # ========================================================

    print_cleanup_plan(
        targets=targets,
        prune_files=prune_files,
    )

    if archive_path is not None:

        print()

        print(
            "Evidence archive preserved:"
        )

        print(
            f"  {archive_path}"
        )


    # ========================================================
    # DRY RUN
    # ========================================================

    if not args.apply:

        print()
        print(
            "=" * 88
        )

        print(
            " DRY RUN ONLY — "
            "NOTHING WAS DELETED"
        )

        print(
            "=" * 88
        )

        print()

        print(
            "Review the cleanup plan above."
        )

        print()

        print(
            "To apply this cleanup:"
        )

        command = (
            "python "
            "scripts\\23_cleanup_preprocessing_dev_outputs.py "
            "--apply"
        )

        if args.drop_rna_cache:

            command += (
                " --drop-rna-cache"
            )

        if (
            args.prune_audit_contact_sheets
        ):

            command += (
                " --prune-audit-contact-sheets"
            )

        if args.force:

            command += (
                " --force"
            )

        print()

        print(
            f"  {command}"
        )

        print_largest_files()

        return


    # ========================================================
    # APPLY CLEANUP
    # ========================================================

    print()
    print(
        "=" * 88
    )

    print(
        " APPLY CLEANUP"
    )

    print(
        "=" * 88
    )

    deleted_bytes = 0


    # --------------------------------------------------------
    # Delete directories
    # --------------------------------------------------------

    for path in targets:

        if not path.exists():
            continue

        size = (
            path_size_bytes(
                path
            )
        )

        print(
            "Deleting directory: "
            f"{safe_relative(path)}"
        )

        delete_path(
            path
        )

        deleted_bytes += size


    # --------------------------------------------------------
    # Delete optional audit contact-sheet files
    # --------------------------------------------------------

    for path in prune_files:

        if not path.exists():
            continue

        validate_delete_path(
            path
        )

        size = (
            path_size_bytes(
                path
            )
        )

        print(
            "Deleting file: "
            f"{safe_relative(path)}"
        )

        path.unlink()

        deleted_bytes += size


    # ========================================================
    # RESULT
    # ========================================================

    print()
    print(
        "=" * 88
    )

    print(
        " CLEANUP COMPLETE"
    )

    print(
        "=" * 88
    )

    print(
        "Recovered approximately: "
        f"{human_size(deleted_bytes)}"
    )

    print()

    print(
        "Preserved:"
    )

    print(
        "  .venv/"
    )

    print(
        "  scripts/"
    )

    print(
        "  data/manifests/"
    )

    print(
        "  data/metadata/"
    )

    print(
        "  audit CSV/JSON results"
    )

    print(
        "  frozen configurations"
    )

    if not args.drop_rna_cache:

        print(
            "  RNA caches"
        )

    if archive_path is not None:

        print(
            "  evidence ZIP:"
        )

        print(
            f"    {archive_path}"
        )

    print_largest_files()


if __name__ == "__main__":
    main()