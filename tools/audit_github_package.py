#!/usr/bin/env python3
r"""Inventory a research project before selecting files for GitHub.

Python 3.9+; standard library only; Windows/PowerShell supported.
Run from the project root:
  .\.venv\Scripts\python.exe -B tools\audit_github_package.py

Reads directory entries, file sizes and Git's tracked-file list only.
Does NOT read file contents, hash datasets, change Git, delete/copy source
files, train models, download data, or upload anything. Reports are written
to a NEW sibling directory outside the project. report.zip contains ONLY
the reports, never original project files. Classification is a proposal,
not an approved publication manifest or a secrets/license audit.
"""

import argparse
import csv
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
import uuid
import zipfile

VERSION = "1.0.0"
MIB = 1024 ** 2
LOCAL_DIRS = {".venv", "venv", "env", "__pycache__", ".pytest_cache",
              ".mypy_cache", ".ruff_cache", "node_modules", "site-packages"}
CODE_TEXT = {".py", ".ps1", ".bat", ".cmd", ".sh", ".r", ".ipynb",
             ".md", ".txt", ".rst", ".json", ".jsonl", ".csv", ".tsv",
             ".yaml", ".yml", ".toml", ".ini", ".cfg", ".xml", ".lock"}
WEIGHTS = {".pt", ".pth", ".ckpt", ".onnx", ".engine", ".safetensors"}
WSI = {".svs", ".ndpi", ".mrxs", ".scn", ".vms", ".vmu", ".bif"}
RASTER = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
ARCHIVES = {".zip", ".7z", ".rar", ".tar", ".gz", ".bz2", ".xz"}
BINARIES = {".exe", ".dll", ".so", ".dylib", ".whl", ".msi", ".deb"}


def classify(relative, size):
    p = Path(relative)
    parts = [x.lower() for x in p.parts]
    name, suffix = p.name.lower(), p.suffix.lower()
    if ".git" in parts:
        return "git_internal", "LOCAL_ONLY"
    if any(x in LOCAL_DIRS for x in parts) or suffix in {".pyc", ".pyo"}:
        return "environment_or_cache", "LOCAL_ONLY"
    if (name == ".env" or name.startswith(".env.") and name not in {".env.example", ".env.template"}
            or suffix in {".pem", ".key", ".p12", ".pfx"}
            or name in {"credentials", "credentials.json", "credentials.ini"}
            or name.startswith("gdc-user-token")):
        return "private_configuration_candidate", "KEEP_LOCAL_REVIEW_NAME"
    if name.startswith("best_model") and suffix in WEIGHTS:
        return "selected_model_weights", "EXTERNAL_ASSET_KEEP_METADATA_IN_GIT"
    if suffix in WEIGHTS:
        return "other_weights_or_checkpoints", "REVIEW_EXTERNAL_OR_REGENERABLE"
    if suffix in WSI or (suffix in RASTER and "data" in parts):
        return "source_images_or_tiles", "EXTERNAL_DATA_WITH_MANIFEST"
    if suffix in {".npy", ".npz", ".h5", ".hdf5", ".h5ad", ".parquet", ".arrow"}:
        return "numeric_or_feature_data", "REVIEW_CONTENT_AND_SIZE"
    if suffix in ARCHIVES or suffix in BINARIES:
        return "archive_or_binary_dependency", "REVIEW_CONTENT_AND_SIZE"
    if suffix in {".log", ".tmp", ".bak"} or name.endswith("~"):
        return "log_or_temporary", "REVIEW_RESEARCH_RELEVANCE"
    if suffix in CODE_TEXT or name in {".gitignore", ".gitattributes", "license", "dockerfile"}:
        action = "GIT_CANDIDATE_REVIEW" if size <= 50 * MIB else "REVIEW_LARGE_TEXT_FILE"
        return "code_config_metadata_results", action
    if suffix in RASTER or suffix in {".svg", ".pdf", ".docx", ".xlsx", ".pptx"}:
        return "figure_or_document", "GIT_CANDIDATE_REVIEW"
    return "other", "REVIEW_CONTENT_AND_SIZE"


def tracked_files(root):
    """No Git changes; return paths only if root is the repository root."""
    env = dict(os.environ, GIT_OPTIONAL_LOCKS="0")
    def git(*args):
        return subprocess.run(["git", "-C", str(root), *args], stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, timeout=30, env=env, check=False)
    try:
        top = git("rev-parse", "--show-toplevel")
        if top.returncode:
            return set(), "not_a_git_worktree"
        repo = Path(os.fsdecode(top.stdout).strip()).resolve()
        if repo != root:
            return set(), "project_is_inside_a_parent_repository_not_a_root"
        result = git("ls-files", "-z", "--cached")
        if result.returncode:
            return set(), "tracked_file_listing_failed"
        return {os.fsdecode(x).replace("\\", "/") for x in result.stdout.split(b"\0") if x}, "listed"
    except (OSError, subprocess.TimeoutExpired):
        return set(), "git_unavailable_or_timed_out"


def write_csv(path, columns, rows):
    with path.open("w", encoding="utf-8-sig", errors="replace", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-parent", type=Path, help="Parent for a NEW report directory, outside the project")
    args = parser.parse_args()
    root = args.project_root.resolve()
    if not root.is_dir():
        parser.error("Project root is not a directory")
    parent = (args.output_parent or root.parent).resolve()
    if parent == root or root in parent.parents:
        parser.error("Report directory must be outside the project")
    if not parent.is_dir():
        parser.error("Output parent must already exist")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = parent / (root.name + "_github_audit_" + stamp + "_" + uuid.uuid4().hex[:6])
    out.mkdir(exist_ok=False)
    tracked, git_status = tracked_files(root)
    print("Metadata-only inventory:", root, flush=True)
    rows, errors, skipped = [], [], []
    stack = [root]
    while stack:
        folder = stack.pop()
        try:
            with os.scandir(folder) as entries:
                for entry in entries:
                    path = Path(entry.path)
                    rel = path.relative_to(root).as_posix()
                    try:
                        info = entry.stat(follow_symlinks=False)
                        reparse = bool(getattr(info, "st_file_attributes", 0) & 0x400)
                        if entry.is_symlink() or reparse:
                            skipped.append({"path": rel, "reason": "link_or_reparse_point_not_followed"})
                            continue
                        if stat.S_ISDIR(info.st_mode):
                            stack.append(path)
                        elif stat.S_ISREG(info.st_mode):
                            category, suggestion = classify(rel, info.st_size)
                            rows.append({"path": rel, "bytes": info.st_size,
                                         "mib": round(info.st_size / MIB, 4),
                                         "extension": path.suffix.lower() or "[none]",
                                         "category": category, "suggestion": suggestion,
                                         "git_tracked": "yes" if rel in tracked else "no" if git_status == "listed" else "unknown",
                                         "over_50_mib": int(info.st_size > 50 * MIB),
                                         "over_100_mib": int(info.st_size > 100 * MIB)})
                            if len(rows) % 10000 == 0:
                                print("  Files listed:", len(rows), flush=True)
                        else:
                            skipped.append({"path": rel, "reason": "not_a_regular_file"})
                    except OSError as exc:
                        errors.append({"path": rel, "error": type(exc).__name__, "errno": exc.errno})
        except OSError as exc:
            errors.append({"path": folder.relative_to(root).as_posix(), "error": type(exc).__name__, "errno": exc.errno})
    rows.sort(key=lambda r: r["path"])
    cols = ["path", "bytes", "mib", "extension", "category", "suggestion", "git_tracked", "over_50_mib", "over_100_mib"]
    write_csv(out / "files.csv", cols, rows)
    write_csv(out / "largest_files.csv", cols, sorted(rows, key=lambda r: r["bytes"], reverse=True)[:100])
    groups = defaultdict(lambda: [0, 0])
    for row in rows:
        parts = row["path"].split("/")
        top = parts[0] if len(parts) > 1 else "[root files]"
        depth2 = "/".join(parts[:2]) if len(parts) > 2 else top
        for kind, key in [("top_folder", top), ("folder_depth_2", depth2),
                          ("extension", row["extension"]), ("category", row["category"])]:
            groups[kind, key][0] += 1
            groups[kind, key][1] += row["bytes"]
    totals = [{"group_type": kind, "group": key, "file_count": count,
               "bytes": size, "gib": round(size / 1024 ** 3, 6)}
              for (kind, key), (count, size) in groups.items()]
    totals.sort(key=lambda r: (r["group_type"], -r["bytes"], r["group"]))
    write_csv(out / "totals.csv", ["group_type", "group", "file_count", "bytes", "gib"], totals)
    write_csv(out / "errors.csv", ["path", "error", "errno"], errors)
    write_csv(out / "skipped_links.csv", ["path", "reason"], skipped)
    present = {r["path"] for r in rows}
    summary = {"script_version": VERSION, "created_utc": stamp, "project_name": root.name,
               "status": "INCOMPLETE_READ_ERRORS" if errors else "COMPLETE_WITH_SKIPPED_ENTRIES" if skipped else "COMPLETE",
               "files": len(rows), "logical_bytes": sum(r["bytes"] for r in rows),
               "over_50_mib": sum(r["over_50_mib"] for r in rows),
               "over_100_mib": sum(r["over_100_mib"] for r in rows),
               "git_status": git_status, "git_tracked_paths": len(tracked),
               "tracked_paths_not_in_inventory": sorted(tracked - present),
               "read_errors": len(errors), "skipped_entries": len(skipped),
               "category_counts": dict(Counter(r["category"] for r in rows)),
               "limitations": ["Paths and sizes only; no file content inspection or hashing.",
                               "Size is logical file size, not allocated disk usage; hard links counted by path.",
                               "Symlinks and Windows reparse points are not followed.",
                               "Does not analyze Git history or decide whether files are safe to publish.",
                               "The classification is provisional; missing large assets may still be required inputs."]}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    lines = ["# GitHub package inventory", "", f"Status: {summary['status']}",
             f"Files: {len(rows):,}; logical size: {summary['logical_bytes'] / 1024 ** 3:.3f} GiB.",
             f"Files >50 MiB: {summary['over_50_mib']}; >100 MiB: {summary['over_100_mib']}.",
             "", "## Largest top-level groups", "", "| Group | Files | GiB |", "|---|---:|---:|"]
    for t in [t for t in totals if t["group_type"] == "top_folder"][:25]:
        label = t["group"].replace("|", "\\|")
        lines.append(f"| {label} | {t['file_count']} | {t['gib']:.3f} |")
    lines += ["", "## How to use this report", "",
              "Review files.csv before making an explicit publication manifest. Do not delete source files based on the category column.",
              "Keep code, protocols, split manifests, predictions and resource measurements together with their dependencies.",
              "Consider selected model weights as separate downloadable assets; raw data need download/reconstruction instructions.",
              "Never treat a small file size as proof of publication eligibility.",
              "", "## Limitations", ""] + ["- " + x for x in summary["limitations"]]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    archive = out / "report.zip"
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as z:
        for name in ("REPORT.md", "summary.json", "files.csv", "largest_files.csv", "totals.csv", "errors.csv", "skipped_links.csv"):
            z.write(out / name, arcname=name)
    print(summary["status"], f"{len(rows):,} files; {summary['logical_bytes'] / 1024 ** 3:.3f} GiB", flush=True)
    print("Reports:", out, flush=True)
    print("Send this report archive:", archive, flush=True)
    if errors:
        print("Some entries could not be listed; errors.csv records the gaps.", flush=True)
    return 1 if errors else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("Interrupted; project files were not modified.", file=sys.stderr)
        sys.exit(130)
