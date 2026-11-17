#!/usr/bin/env python3
r"""Stage 28: audit production PNGs and freeze patient-level splits (no training).

Windows / Python 3.12:
  .\.venv\Scripts\python.exe scripts\28_prepare_patch_budget_experiment.py --preflight-only
  .\.venv\Scripts\python.exe scripts\28_prepare_patch_budget_experiment.py --prepare

Requires Pillow and, for a READY training environment, numpy, torch, torchvision
and a working CUDA device. Does not install packages or download model weights.
Default mode is read-only --preflight-only. --prepare repeats the FULL audit;
it publishes a new directory only after all checks pass. Existing prepared
files are verified and never overwritten. Close output CSVs in Excel before
future scripts write to them. This script itself never replaces their contents.

--self-test uses synthetic images in temporary directories; it does not inspect
the real dataset or require torch/CUDA. Exact pixel duplicates across patients
block preparation for review; patients are never silently removed or replaced.
The data/split protocol is frozen here; training settings remain a DRAFT until
the separate GPU forward/backward pilot. No accuracy or resource result is
produced by this script.
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import argparse
import collections
import csv
import hashlib
import importlib.metadata
import io
import json
import math
import os
import platform
import re
import shutil
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath

VERSION = "1.0.2"
NAMESPACE = "production_final_v2_tissue080"
PRODUCTION_PROTOCOL = "V2_FINAL_HSV_64x256+TISSUE080+ARTIFACT_QC_V1"
OUTPUT_NAME = "wsi_patch_budget_v1"
SPLIT_SEED = 20261112
TRAINING_SEEDS = [20261113, 20261114, 20261115]
CLASSES = ("LUAD", "LUSC")
PARTITIONS = ("train", "validation", "test")
PARTITION_ALIASES = {"train": "train", "training": "train", "val": "validation",
                     "valid": "validation", "validation": "validation", "test": "test"}
PATIENT_ID_COLUMNS = {"case_submitter_id", "patient_id"}
PARTITION_COLUMNS = {"split", "partition", "subset"}
FOLD_COLUMNS = {"fold", "fold_id", "fold_index"}
EXPECTED_HASHES = {
    "cohort": "0b62f5aa3b1dc09f9e96c411edb04b4456004a188eaa882072e4339e2d1611fa",
    "tiles": "9aef5106e4907ad846039be0a4c0bdf84d824a82daeb856355703ddacc2665e0",
    "qc": "a7043c92c3e99c99c326544f0c29bad3a3a66788de21e5b250d07dcd0bd4cdef",
    "config": "a8b34c4ab05e987a7ac07c62228d20c2fc75f7fe247d3241ba3890e8c6fe4484",
    "script11": "3af2ff8ce9d085a3723eb29542f10fdbc97aa449422fafd3815683d5e5857b60",
    "script17": "89edd15f150e30f424e945290ed22ba23cebc75d679d2ad70db3c3669d805b86",
    "script24": "c97ada4ce9b5f8a3f669b63035f90f072ab13233906ce42e87ebd005848cd664",
}
OWNED_FILES = {
    "input_audit.json", "environment.json", "source_inputs.json",
    "patient_splits.csv", "pilot_manifest.csv", "patch_inventory.csv",
    "protocol.json", "run_status.json",
}
ID_RE = re.compile(r"TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}\Z")


class AuditError(Exception):
    pass


def require(condition, message):
    if not condition:
        raise AuditError(message)


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("utf-8")


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def csv_rows(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        require(reader.fieldnames is not None, f"Empty CSV: {path}")
        require(len(set(reader.fieldnames)) == len(reader.fieldnames),
                f"Duplicate CSV column names: {path}")
        rows = list(reader)
    require(all(None not in r and all(v is not None for v in r.values()) for r in rows),
            f"Malformed CSV rows: {path}")
    return rows


def keyed(rows, field, description):
    require(all(field in r and r[field] for r in rows), f"Missing {field}: {description}")
    result = {r[field]: r for r in rows}
    require(len(result) == len(rows), f"Duplicate {field}: {description}")
    return result


def integer(value, description):
    try:
        n = float(value)
        require(math.isfinite(n) and n.is_integer(), f"Non-integer {description}: {value}")
        return int(n)
    except (TypeError, ValueError, OverflowError):
        raise AuditError(f"Invalid integer {description}: {value}") from None


def bounded(value, lo, hi, description):
    try:
        n = float(value)
    except (TypeError, ValueError):
        raise AuditError(f"Invalid numeric {description}: {value}") from None
    require(math.isfinite(n) and lo <= n <= hi,
            f"Out-of-range {description}: {value}; expected [{lo}, {hi}]")
    return n


def source_paths(root):
    manifests = root / "data" / "manifests"
    metadata = root / "data" / "metadata" / NAMESPACE
    qc_candidates = [metadata / "batches" / "final_full_cohort_v1_patient_qc.csv",
                     manifests / NAMESPACE / "batches" / "final_full_cohort_v1_patient_qc.csv"]
    existing = [p for p in qc_candidates if p.is_file()]
    require(existing, "Patient QC CSV missing. Expected one of: " + "; ".join(map(str, qc_candidates)))
    require(len({sha_file(p) for p in existing}) == 1,
            "Conflicting QC copies; restore the original production files before proceeding.")
    paths = {
        "cohort": manifests / "paired_cohort_stage2.csv",
        "tiles": manifests / NAMESPACE / "batches" / "final_full_cohort_v1_tile_manifest.csv",
        "qc": existing[0], "config": metadata / "frozen_protocol_configuration.json",
        "script11": root / "scripts" / "11_process_cohort_batch_v2_prod.py",
        "script17": root / "scripts" / "17_process_cohort_v2prod_artifact_qc.py",
        "script24": root / "scripts" / "24_process_final_full_cohort.py",
    }
    return paths, existing


def verify_sources(paths):
    records = {}
    for name, path in paths.items():
        require(path.is_file(), f"Required original source missing: {path}")
        actual = sha_file(path)
        require(actual == EXPECTED_HASHES[name],
                f"Frozen SHA256 mismatch for {name}: {path}\n"
                f"  expected {EXPECTED_HASHES[name]}\n  actual   {actual}\n"
                "Do not bypass this check; review the source/version first.")
        records[name] = {"path": str(path.resolve()), "sha256": actual,
                         "bytes": path.stat().st_size}
    return records


def tile_location(row, tiles_root):
    pid, name = row["case_submitter_id"], row["tile_name"]
    require(ID_RE.fullmatch(pid), f"Invalid patient ID: {pid}")
    idx = integer(row["tile_index"], "tile_index")
    x, y = integer(row["x_level0"], "x_level0"), integer(row["y_level0"], "y_level0")
    require(1 <= idx <= 64 and x >= 0 and y >= 0, f"Invalid tile coordinates/index: {pid}/{name}")
    require(name == f"tile_{idx:03d}_x{x}_y{y}.png", f"Unexpected tile filename: {pid}/{name}")
    stored = PureWindowsPath(row["tile_path"])
    require(".." not in stored.parts and tuple(stored.parts[-5:]) ==
            ("data", "tiles", NAMESPACE, pid, name), f"Unexpected stored tile_path: {row['tile_path']}")
    actual = (tiles_root / pid / name).resolve()
    require(actual.is_relative_to(tiles_root.resolve()), f"Tile resolves outside tiles root: {actual}")
    return actual, idx, x, y


def validate_manifests(paths, tiles_root):
    with paths["config"].open(encoding="utf-8-sig") as f:
        config = json.load(f)
    require(config["protocol_name"] == PRODUCTION_PROTOCOL and config["namespace"] == NAMESPACE,
            "Unexpected production protocol.")
    cohort = keyed(csv_rows(paths["cohort"]), "case_submitter_id", "cohort")
    qc = keyed(csv_rows(paths["qc"]), "case_submitter_id", "QC")
    require(len(cohort) == 950 and set(cohort) == set(qc), "Cohort/QC patient set must match all 950 patients.")
    require(collections.Counter(r["class_label"] for r in cohort.values()) == {"LUAD": 474, "LUSC": 476},
            "Unexpected original cohort class counts.")
    patients, excluded = {}, []
    for pid, row in qc.items():
        require(ID_RE.fullmatch(pid), f"Invalid patient ID: {pid}")
        label = cohort[pid]["class_label"]
        require(row["class_label"] == label and cohort[pid]["project_id"] == "TCGA-" + label,
                f"Class mismatch: {pid}")
        if row["status"] == "PASS":
            require(integer(row["valid_tiles"], "valid_tiles") == 64, f"PASS count is not 64: {pid}")
            require(row["selected_slide_token"] == "DX1" and
                    row["selected_wsi_file_id"] == cohort[pid]["wsi_file_id"], f"Unexpected WSI selection: {pid}")
            patients[pid] = {"case_submitter_id": pid, "class_label": label,
                             "wsi_file_id": row["selected_wsi_file_id"], "tile_count": 64}
        else:
            require(row["status"] == "EXCLUDED_WSI_QC", f"Unexpected QC status: {pid}: {row['status']}")
            excluded.append(pid)
    require(len(patients) == 929 and len(excluded) == 21, "Expected 929 PASS and 21 excluded patients.")
    require(collections.Counter(p["class_label"] for p in patients.values()) == {"LUAD": 455, "LUSC": 474},
            "Unexpected PASS class counts.")
    tiles = csv_rows(paths["tiles"])
    require(len(tiles) == 59456, "Expected 59,456 production tile records.")
    seen_indices, seen_coords = collections.defaultdict(set), collections.defaultdict(set)
    actual_paths = set()
    for r in tiles:
        pid = r["case_submitter_id"]
        require(pid in patients, f"Tile belongs to a non-PASS patient: {pid}")
        p = patients[pid]
        require(r["class_label"] == p["class_label"] and r["selected_wsi_file_id"] == p["wsi_file_id"]
                and r["selected_wsi_slide_token"] == "DX1", f"Tile label/WSI mismatch: {pid}")
        require(integer(r["tile_size"], "tile_size") == 256, f"Unexpected tile size: {pid}")
        require(r["artifact_qc_status"] == "PASS" and r["artifact_qc_protocol"] == PRODUCTION_PROTOCOL,
                f"Unexpected artifact QC: {pid}")
        bounded(r["final_tissue_fraction"], .8, 1, "final_tissue_fraction")
        bounded(r["approx_tissue_fraction"], .7, 1, "approx_tissue_fraction")
        bounded(r["background_fraction"], 0, .1, "background_fraction")
        bounded(r["white_fraction"], 0, .1, "white_fraction")
        path, idx, x, y = tile_location(r, tiles_root)
        require(idx not in seen_indices[pid] and (x, y) not in seen_coords[pid], f"Repeated tile/coordinates: {pid}")
        require(path not in actual_paths, f"Repeated actual path: {path}")
        seen_indices[pid].add(idx)
        seen_coords[pid].add((x, y))
        actual_paths.add(path)
        r.update(_path=str(path), _index=idx, _x=x, _y=y)
    require(all(seen_indices[p] == set(range(1, 65)) for p in patients), "Each patient must have indices 1..64.")
    return patients, sorted(excluded), sorted(tiles, key=lambda r: (r["case_submitter_id"], r["_index"]))


def read_png(path):
    from PIL import Image
    before = path.stat()
    require(0 < before.st_size <= 16 * 1024 * 1024, f"Unexpected PNG file size: {path}")
    raw = path.read_bytes()
    after = path.stat()
    require((before.st_size, before.st_mtime_ns, before.st_ino) ==
            (after.st_size, after.st_mtime_ns, after.st_ino) and len(raw) == before.st_size,
            f"File changed during read: {path}")
    with Image.open(io.BytesIO(raw)) as im:
        require(im.format == "PNG" and im.size == (256, 256) and im.mode == "RGB" and
                getattr(im, "n_frames", 1) == 1, f"Expected a single RGB 256x256 PNG: {path}")
        im.verify()
    with Image.open(io.BytesIO(raw)) as im:
        im.load()  # Force real decompression; Image.open alone is not a readability test.
        pixel_hash = hashlib.sha256(im.tobytes()).hexdigest()
    return {"bytes": len(raw), "mtime_ns": after.st_mtime_ns,
            "file_sha256": hashlib.sha256(raw).hexdigest(), "pixel_sha256": pixel_hash}


def scan_images(tiles, progress=True):
    inventory, first_pixel, within, errors, cross = [], {}, 0, [], 0
    failures, last_print = 0, time.monotonic()
    for i, row in enumerate(tiles, 1):
        pid, path = row["case_submitter_id"], Path(row["_path"])
        try:
            info = read_png(path)
            record = {"case_submitter_id": pid, "class_label": row["class_label"],
                      "tile_index": row["_index"], "x_level0": row["_x"], "y_level0": row["_y"],
                      "relative_path": f"{pid}/{row['tile_name']}", "actual_path": str(path), **info}
            prev = first_pixel.get(info["pixel_sha256"])
            if prev is not None:
                if prev[0] != pid:
                    cross += 1
                    if len(errors) < 20:
                        errors.append(f"CROSS-PATIENT pixel duplicate: {prev[0]}/{prev[1]} == {pid}/{row['tile_name']}")
                else:
                    within += 1
            else:
                first_pixel[info["pixel_sha256"]] = (pid, row["tile_name"])
            inventory.append(record)
        except (OSError, ValueError, AuditError, SyntaxError) as e:
            failures += 1
            if len(errors) < 20:
                errors.append(f"{path}: {type(e).__name__}: {e}")
        if progress and (i % 1000 == 0 or i == len(tiles) or time.monotonic() - last_print >= 10):
            print(f"  PNG {i}/{len(tiles)}; unreadable/invalid={failures}; cross-patient duplicates={cross}", flush=True)
            last_print = time.monotonic()
    require(failures == 0 and cross == 0,
            f"PNG AUDIT FAILED: {failures} invalid/missing files; {cross} cross-patient duplicate occurrences.\n"
            + "\n".join(errors) + "\nNo patients were dropped. Review/repair inputs before preparing a split.")
    stable = [{k: r[k] for k in ("case_submitter_id", "class_label", "tile_index", "x_level0", "y_level0",
                                "relative_path", "file_sha256", "pixel_sha256")} for r in inventory]
    return inventory, {"decoded_pngs": len(inventory), "total_png_bytes": sum(r["bytes"] for r in inventory),
                       "exact_pixel_duplicates_within_patient": within, "cross_patient_duplicates": cross,
                       "dataset_fingerprint_sha256": digest(stable)}


def rank(tag, pid):
    return hashlib.sha256(f"{tag}|{SPLIT_SEED}|{pid}".encode("ascii")).hexdigest()


def new_split(patients):
    result = {}
    for label in CLASSES:
        ids = sorted((p for p in patients if patients[p]["class_label"] == label),
                     key=lambda p: (rank(f"split-v1|{label}", p), p))
        n = int(len(ids) * .2 + .5)
        for i, pid in enumerate(ids):
            result[pid] = "validation" if i < n else "test" if i < 2 * n else "train"
    return result


def read_split(path, patients, excluded):
    rows = csv_rows(path)
    require(rows, f"Empty split file: {path}")
    id_keys = [k for k in ("case_submitter_id", "patient_id") if k in rows[0]]
    split_keys = [k for k in ("split", "partition", "subset") if k in rows[0]]
    require(len(id_keys) == 1 and len(split_keys) == 1,
            f"Split CSV needs one patient ID column and one split/partition/subset column: {path}")
    selected, seen = {}, set()
    known = set(patients) | set(excluded)
    for r in rows:
        pid = r[id_keys[0]].strip()
        require(pid in known and pid not in seen, f"Unknown/duplicate patient in split {path}: {pid}")
        seen.add(pid)
        require(r[split_keys[0]].strip().lower() in PARTITION_ALIASES,
                f"Unsupported partition for {pid}: {split_keys[0]}={r[split_keys[0]]!r} in {path}")
        if pid in patients:
            if "class_label" in r:
                require(r["class_label"].strip() == patients[pid]["class_label"], f"Split class mismatch: {pid}")
            selected[pid] = PARTITION_ALIASES[r[split_keys[0]].strip().lower()]
    require(set(selected) == set(patients), f"Split does not cover all 929 PASS patients: {path}")
    require(set(selected.values()) == set(PARTITIONS), f"Split must contain train, validation and test: {path}")
    return selected


def json_has_patient_split(value):
    """Detect partition assignments, not incidental 'test' text in QC metadata."""
    partition_keys = set(PARTITION_ALIASES)

    def has_patient_id(item):
        if isinstance(item, str):
            return bool(ID_RE.fullmatch(item))
        if isinstance(item, dict):
            return any(has_patient_id(k) or has_patient_id(v) for k, v in item.items())
        if isinstance(item, list):
            return any(has_patient_id(v) for v in item)
        return False

    pending = [value]
    while pending:
        item = pending.pop()
        if isinstance(item, dict):
            normalized = {str(k).strip().lower(): v for k, v in item.items()}
            keys = set(normalized)
            if keys & PATIENT_ID_COLUMNS:
                if keys & FOLD_COLUMNS:
                    return True
                if any(isinstance(normalized[k], str) and
                       normalized[k].strip().lower() in PARTITION_ALIASES
                       for k in keys & PARTITION_COLUMNS):
                    return True
            for k, v in item.items():
                key = str(k).strip().lower()
                # Support train_ids/test_patients as well as train/test lists.
                for suffix in ("_patient_ids", "_patients", "_ids", "_cases"):
                    if key.endswith(suffix):
                        key = key[:-len(suffix)]
                        break
                if key in partition_keys | FOLD_COLUMNS and has_patient_id(v):
                    return True
                if ID_RE.fullmatch(str(k)) and isinstance(v, str) and v.strip().lower() in partition_keys:
                    return True
                if isinstance(v, (dict, list)):
                    pending.append(v)
        elif isinstance(item, list):
            pending.extend(v for v in item if isinstance(v, (dict, list)))
    return False


def split_evidence(path, root, non_split_tables=None):
    """Use explicit split intent or patient partition columns/content as evidence.

    A QC batch called artifact_qc3_replacement_test is not a test-set manifest.
    Explicit split/fold names and standalone train/test patient lists remain
    candidates even when incomplete or malformed; they must still be reviewed.
    Generic subset/partition/split column names alone are insufficient: e.g.
    subset=all/candidate/accepted describes QC groups, not model partitions.
    One recognized partition value anywhere in the file is enough to retain
    the entire file for strict validation (including all its invalid rows).
    """
    strong = re.compile(r"(^|[_\-.])(splits?|folds?|partitions?)([_\-.]|$)", re.I)
    dedicated = re.compile(
        r"(?:(?:train|training|val|valid|validation|test)(?:[_-](?:patients?|patient_ids|ids|cases))?"
        r"|(?:patients?|patient_ids|ids|cases)[_-](?:train|training|val|valid|validation|test))\Z", re.I)
    relative = path.relative_to(root)
    if any(strong.search(part) for part in relative.parts):
        return "explicit split/fold/partition path"
    if dedicated.fullmatch(path.stem) or any(dedicated.fullmatch(part) for part in relative.parts[:-1]):
        return "dedicated train/validation/test list or directory"
    # Inspect every CSV header regardless of filename. Stream rows only when
    # ambiguous role columns exist; a late train/test value must not be missed.
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.reader(f)
            header = [str(k).strip().lower() for k in next(reader, [])]
            fields = set(header)
            if fields & PATIENT_ID_COLUMNS and fields & FOLD_COLUMNS:
                return "CSV patient and fold columns"
            if fields & PATIENT_ID_COLUMNS and fields & PARTITION_COLUMNS:
                columns = [i for i, name in enumerate(header) if name in PARTITION_COLUMNS]
                examples, rows_read = set(), 0
                for row in reader:
                    rows_read += 1
                    for i in columns:
                        label = row[i].strip().lower() if i < len(row) else ""
                        if label in PARTITION_ALIASES:
                            return "CSV patient role column contains train/validation/test values"
                        if len(examples) < 20:
                            examples.add(label[:160])
                if non_split_tables is not None:
                    non_split_tables.append({"path": str(path.resolve()),
                        "reason": "Generic role column has no train/validation/test values; no explicit split path",
                        "columns": [header[i] for i in columns], "rows_checked": rows_read,
                        "observed_values_sample": sorted(examples)})
    elif path.suffix.lower() == ".json":
        try:
            with path.open(encoding="utf-8-sig") as f:
                value = json.load(f)
        except (ValueError, RecursionError):
            # An unrelated malformed JSON is not proof of an existing split.
            # Explicit split/fold names have already been retained above.
            return None
        if json_has_patient_split(value):
            return "JSON patient partition assignments"
    return None


def discover_history(root, output):
    candidates, training_scripts, non_split_tests, non_split_tables = [], [], [], []
    weak = re.compile(r"(^|[_\-.])(train|training|val|valid|validation|test)([_\-.]|$)", re.I)
    # Inspect names and partition schemas; arbitrary project scripts are never executed.
    for base in (root / "data" / "manifests", root / "data" / "metadata", root / "data" / "splits",
                 root / "splits", root / "configs", root / "results", root / "artifacts"):
        if not base.is_dir():
            continue
        for dirpath, dirs, files in os.walk(base, followlinks=False):
            parent = Path(dirpath)
            dirs[:] = sorted(d for d in dirs if not d.startswith(".") and d not in
                             {"tiles", "images", "logs", "checkpoints", "cache", "raw"}
                             and not (parent / d).is_symlink() and (parent / d).resolve() != output.resolve())
            for name in sorted(files):
                p = parent / name
                if p.suffix.lower() not in {".csv", ".json", ".yaml", ".yml", ".txt"}:
                    continue
                evidence = split_evidence(p, root, non_split_tables)
                if evidence:
                    candidates.append(p.resolve())
                elif any(weak.search(part) for part in p.relative_to(root).parts):
                    non_split_tests.append({"path": str(p.resolve()),
                                            "reason": "Incidental train/test text; no explicit split path or partition schema"})
    scripts = root / "scripts"
    if scripts.is_dir():
        for p in sorted(scripts.glob("*.py")):
            if re.search(r"train|split|fold|mil|model", p.stem, re.I):
                training_scripts.append({"path": str(p.resolve()), "sha256": sha_file(p)})
    return sorted(set(candidates)), training_scripts, non_split_tests, non_split_tables


def select_split(root, output, explicit, patients, excluded):
    candidates, scripts, non_split_tests, non_split_tables = discover_history(root, output)
    history = {"split_candidates": [str(p) for p in candidates], "training_scripts": scripts,
               "non_split_test_files": non_split_tests, "non_split_role_tables": non_split_tables}
    if explicit is not None:
        require(not explicit.resolve().is_relative_to(output.resolve()),
                "--split-file must be an external existing split; omit it when rechecking this prepared output.")
        selected = read_split(explicit, patients, excluded)
        return selected, {"kind": "explicit_existing", "path": str(explicit.resolve()),
                          "sha256": sha_file(explicit), **history}
    if output.exists():
        require(output.is_dir(), f"Output path exists and is not a directory: {output}")
        verify_bundle(output)
        selected = read_split(output / "patient_splits.csv", patients, excluded)
        old = json.loads((output / "protocol.json").read_text(encoding="utf-8"))
        return selected, {**old["split_origin"], "current_history": history}
    valid, ambiguous = [], []
    for path in candidates:
        try:
            require(path.suffix.lower() == ".csv", "Non-CSV split candidate requires review/export to patient CSV.")
            valid.append((path, read_split(path, patients, excluded)))
        except (AuditError, OSError, UnicodeError, csv.Error) as e:
            ambiguous.append(f"{path}: {e}")
    require(not ambiguous,
            "Possible existing split(s) need review; refusing to silently create another split.\n"
            + "\n".join(ambiguous[:15]) + "\nUse --split-file <complete_patient_split.csv> to explicitly reuse a reviewed split.")
    if valid:
        require(all(mapping == valid[0][1] for _, mapping in valid),
                "Conflicting existing splits. Select the intended one with --split-file <path>.")
        path, selected = valid[0]
        return selected, {"kind": "auto_existing", "path": str(path), "sha256": sha_file(path), **history}
    return new_split(patients), {"kind": "new_deterministic", "seed": SPLIT_SEED,
                               "rule": "SHA256 ASCII split-v1|CLASS|20261112|PATIENT; ascending; first rounded 20% validation, next test, rest train",
                               **history}


def pilot_rows(patients, split):
    rows = []
    for part, per_class in (("train", 30), ("validation", 10)):
        for label in CLASSES:
            ids = [p for p in patients if split[p] == part and patients[p]["class_label"] == label]
            require(len(ids) >= per_class, f"Existing split too small for planned pilot: {part}/{label}")
            ids.sort(key=lambda p: (rank(f"pilot-v1|{part}|{label}", p), p))
            for pid in ids[:per_class]:
                rows.append({"case_submitter_id": pid, "class_label": label, "split": part,
                             "pilot_order_within_class": ids.index(pid) + 1})
    return rows


ENV_PROBE = r'''
import json, sys
r = {"ready": False, "errors": []}
try:
    import torch
    import torchvision
    from torchvision.models import resnet18, ResNet18_Weights
    r.update(torch=torch.__version__, torchvision=torchvision.__version__,
             cuda_runtime=torch.version.cuda, cuda_available=torch.cuda.is_available(),
             resnet18_available=callable(resnet18),
             weights_enum="ResNet18_Weights.IMAGENET1K_V1",
             weights_url=ResNet18_Weights.IMAGENET1K_V1.url)
    if not r["cuda_available"]:
        r["errors"].append("CUDA is not available to this Python/PyTorch installation.")
    else:
        r["devices"] = []
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            d = {"index": i, "name": p.name, "total_memory_bytes": p.total_memory,
                 "compute_capability": [p.major, p.minor]}
            try:
                d["free_memory_bytes"], d["reported_total_memory_bytes"] = torch.cuda.mem_get_info(i)
            except Exception as e:
                d["free_memory_query_warning"] = str(e)
            r["devices"].append(d)
        r["ready"] = bool(r["devices"] and r["resnet18_available"])
except Exception as e:
    r["errors"].append(type(e).__name__ + ": " + str(e))
print("STAGE28_ENV_JSON=" + json.dumps(r))
'''


def environment_probe(root):
    result = {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
              "processor": platform.processor(),
              "logical_cpu_count": os.cpu_count(), "versions": {}, "warnings": [],
              "gpu_forward_backward_tested": False}
    for package in ("Pillow", "numpy", "torch", "torchvision", "psutil"):
        try:
            result["versions"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result["versions"][package] = None
    try:
        import psutil
        ram = psutil.virtual_memory()
        result["ram"] = {"total_bytes": ram.total, "available_bytes": ram.available}
        result["physical_cpu_count"] = psutil.cpu_count(logical=False)
    except ImportError:
        if os.name == "nt":
            import ctypes
            class MemoryStatus(ctypes.Structure):
                _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
                    (name, ctypes.c_ulonglong) for name in ("total", "available", "page_total", "page_available",
                                                          "virtual_total", "virtual_available", "extended_available")]
            memory = MemoryStatus()
            memory.length = ctypes.sizeof(memory)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)):
                result["ram"] = {"total_bytes": memory.total, "available_bytes": memory.available}
            else:
                result["warnings"].append("Windows RAM query failed.")
        else:
            result["warnings"].append("psutil absent: RAM availability not measured; no package installed automatically.")
    result["disk_free_bytes"] = shutil.disk_usage(root).free
    expected_python = root / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if expected_python.is_file() and Path(sys.executable).absolute() != expected_python.absolute():
        result["warnings"].append(f"Interpreter is not the expected project .venv: {expected_python}")
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    try:
        proc = subprocess.run([sys.executable, "-B", "-c", ENV_PROBE], capture_output=True, text=True,
                              timeout=55, env=env, check=False)
        lines = [line.partition("=")[2] for line in proc.stdout.splitlines() if line.startswith("STAGE28_ENV_JSON=")]
        require(proc.returncode == 0 and len(lines) == 1, "Torch probe produced no valid report: " + proc.stderr[-1500:])
        result["gpu_probe"] = json.loads(lines[0])
    except (OSError, subprocess.TimeoutExpired, AuditError, ValueError) as e:
        result["gpu_probe"] = {"ready": False, "errors": [str(e)]}
    smi = shutil.which("nvidia-smi")
    if smi:
        try:
            proc = subprocess.run([smi, "--query-gpu=name,driver_version,memory.total,memory.free", "--format=csv,noheader"],
                                  capture_output=True, text=True, timeout=10, check=False)
            result["nvidia_smi"] = proc.stdout.strip() if proc.returncode == 0 else proc.stderr.strip()
        except (OSError, subprocess.TimeoutExpired) as e:
            result["warnings"].append(f"nvidia-smi unavailable: {e}")
    else:
        result["warnings"].append("nvidia-smi not on PATH; driver version not recorded.")
    result["ready"] = bool(result["gpu_probe"]["ready"] and result["versions"]["Pillow"] and result["versions"]["numpy"])
    return result


def write_json(path, data):
    with path.open("x", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())


def write_csv(path, rows):
    require(rows, f"Refusing to write empty CSV: {path}")
    with path.open("x", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        f.flush()
        os.fsync(f.fileno())


def verify_bundle(output):
    require((output / "file_hashes.json").is_file(),
            f"Output already exists but is not a verified prepared bundle: {output}. No files overwritten.")
    with (output / "file_hashes.json").open(encoding="utf-8") as f:
        record = json.load(f)
    require(record.get("schema_version") == 1 and set(record.get("files", {})) == OWNED_FILES,
            "Prepared file hash manifest has an unexpected schema/file list.")
    for name, expected in record["files"].items():
        path = output / name
        require(path.is_file() and not path.is_symlink() and sha_file(path) == expected,
                f"Prepared file missing/changed: {path}. No files overwritten.")
    return record


def publish_bundle(output, json_files, csv_files, identity):
    if output.exists():
        record = verify_bundle(output)
        require(record["identity"] == identity,
                "Prepared identity differs from current data, script, roots or split. No overwrite; review the change.")
        return "ALREADY_PREPARED"
    require(set(json_files) | set(csv_files) == OWNED_FILES, "Internal output file set mismatch.")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{OUTPUT_NAME}.prepare-", dir=output.parent))
    try:
        for name, data in json_files.items():
            write_json(staging / name, data)
        for name, rows in csv_files.items():
            write_csv(staging / name, rows)
        checks = {name: sha_file(staging / name) for name in sorted(OWNED_FILES)}
        write_json(staging / "file_hashes.json", {"schema_version": 1, "identity": identity, "files": checks})
        verify_bundle(staging)
        require(not output.exists(), f"Output appeared during preparation: {output}; refusing to replace it.")
        os.rename(staging, output)
    except BaseException:
        # Delete only this process's newly created temporary bundle; no user input/output is removed.
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return "PREPARED"


def check_inputs_unchanged(paths, source_records, split_origin, inventory):
    for name, path in paths.items():
        require(sha_file(path) == source_records[name]["sha256"], f"Source changed during audit: {path}")
    if split_origin.get("path"):
        path = Path(split_origin["path"])
        require(path.is_file() and sha_file(path) == split_origin["sha256"], f"Existing source split changed/missing: {path}")
    for r in inventory:
        s = Path(r["actual_path"]).stat()
        require(s.st_size == r["bytes"] and s.st_mtime_ns == r["mtime_ns"],
                f"PNG changed since decoding: {r['actual_path']}")


def run(args):
    started = time.monotonic()
    root = args.root.expanduser().resolve()
    require(root.is_dir(), f"Project root does not exist: {root}")
    output = root / "results" / OUTPUT_NAME
    tiles_root = (args.tiles_root.expanduser() if args.tiles_root else root / "data" / "tiles" / NAMESPACE).resolve()
    print(f"STAGE 28 v{VERSION}; {'PREPARE' if args.prepare else 'READ-ONLY PREFLIGHT'}", flush=True)
    print(f"Project: {root}\nPython: {sys.executable}\nTiles: {tiles_root}", flush=True)
    paths, qc_copies = source_paths(root)
    sources = verify_sources(paths)
    patients, excluded, tiles = validate_manifests(paths, tiles_root)
    print("MANIFEST PASS: 950 original / 929 PASS / 21 excluded; LUAD=455, LUSC=474; 59456 tiles", flush=True)
    split, origin = select_split(root, output, args.split_file, patients, excluded)
    pilots = pilot_rows(patients, split)
    print(f"Split source: {origin['kind']}")
    history = origin.get("current_history", origin)
    print(f"History: {len(history.get('split_candidates', []))} split candidate(s); "
          f"{len(history.get('non_split_test_files', []))} unrelated train/test-named files recorded as non-splits.")
    print(f"  Non-split tables with generic subset/partition columns: {len(history.get('non_split_role_tables', []))}")
    counts = {part: {label: sum(split[p] == part and patients[p]['class_label'] == label for p in patients)
                     for label in CLASSES} for part in PARTITIONS}
    for part, values in counts.items():
        print(f"  {part:10s} LUAD={values['LUAD']:3d} LUSC={values['LUSC']:3d} total={sum(values.values())}")
    print("Checking environment (CUDA availability only; no forward/backward or weight downloads)...", flush=True)
    environment = environment_probe(root)
    print(json.dumps(environment, ensure_ascii=True, indent=2), flush=True)
    require(environment["versions"]["Pillow"], "Pillow is missing in this interpreter; cannot decode PNGs.")
    print("Decoding and hashing ALL PNGs; do not modify the source data while this runs...", flush=True)
    inventory, image_summary = scan_images(tiles)
    check_inputs_unchanged(paths, sources, origin, inventory)
    for q in qc_copies:
        require(sha_file(q) == EXPECTED_HASHES["qc"], f"QC copy changed during audit: {q}")
    identity = {"stage28_version": VERSION, "script_sha256": sha_file(Path(__file__).resolve()),
                "source_sha256": {k: v["sha256"] for k, v in sources.items()},
                "dataset_fingerprint_sha256": image_summary["dataset_fingerprint_sha256"],
                "split_sha256": digest(split), "project_root": str(root), "tiles_root": str(tiles_root)}
    if output.exists():
        require(verify_bundle(output)["identity"] == identity,
                "Existing prepared bundle differs from current data/script/paths/split. No overwrite.")
    elapsed = time.monotonic() - started
    print(f"DATA AUDIT PASS: {len(inventory)} decoded PNGs; {image_summary['total_png_bytes'] / 1024**3:.3f} GiB")
    if image_summary["exact_pixel_duplicates_within_patient"]:
        print(f"WARNING: within-patient exact duplicate occurrences={image_summary['exact_pixel_duplicates_within_patient']}; preserved and recorded.")
    if not environment["ready"]:
        print("ENVIRONMENT BLOCKED: data audit passed, but torch/torchvision/CUDA or required packages are not ready.")
        print("No output written. Send the environment report above before installing/changing packages.")
        return 2
    print(f"PREFLIGHT PASS; audit elapsed={elapsed:.1f}s; output={output}")
    if not args.prepare:
        print("Read-only audit complete. No output writes, downloads or training. Next: repeat with --prepare.")
        return 0
    required_disk = max(128 * 1024**2, sum(len(canonical(r)) for r in inventory) * 3)
    require(shutil.disk_usage(root).free >= required_disk,
            f"Insufficient free disk for audit publication (required buffer {required_disk / 1024**2:.0f} MiB).")
    now = utc_now()
    protocol = {
        "schema_version": 1, "experiment": "WSI patch budget: fixed K before adaptive development",
        "prepared_utc": now, "data_status": "FROZEN", "training_status": "DRAFT_PENDING_GPU_SMOKE",
        "production_protocol": PRODUCTION_PROTOCOL, "namespace": NAMESPACE, "cohort": counts,
        "pass_patients": 929, "excluded_patients": excluded, "patches_per_patient": 64,
        "tile_size": [256, 256], "tissue_threshold": .8, "split_origin": origin,
        "split_seed": SPLIT_SEED if origin["kind"] == "new_deterministic" else None,
        "pilot": {"train": 60, "validation": 20, "test": 0,
                  "ranking": "SHA256 ASCII pilot-v1|SPLIT|CLASS|20261112|PATIENT"},
        "training_draft": {
            "modalities": ["WSI"], "architecture": "ResNet18 + gated-attention MIL (hidden=128) + 2 logits",
            "weights": "ResNet18_Weights.IMAGENET1K_V1", "encoder_trainable": True,
            "training_patch_budgets": [16, 32, 64], "primary_comparison": "32 versus 64",
            "validation_and_test_patch_budget": 64, "training_seeds": TRAINING_SEEDS,
            "epochs_provisional": 30, "precision": "AMP FP16", "patient_batch_size": 1,
            "gradient_accumulation_patients": 4, "optimizer": "AdamW", "encoder_lr": .0001,
            "mil_lr": .001, "weight_decay": .0001,
            "batchnorm": "frozen running statistics; affine parameters trainable",
            "test_policy": "No test scoring during development; freeze methods and checkpoints before final test",
            "gpu_feasibility_verified": False,
        },
        "identity": identity,
        "limitations": ["No learning, performance, energy or novelty claim follows from this audit.",
                        "Exact-pixel duplicate checks do not detect near-duplicates or all biological relatedness.",
                        "A patient split does not constitute external or center-held-out validation.",
                        "Training configuration must be finalized separately after the GPU pilot."],
    }
    audit = {"status": "PASS", "utc": now, "elapsed_seconds": elapsed, **image_summary,
             "patients": 929, "original_patients": 950, "excluded_patients": 21, "split_counts": counts,
             "qc_source_copies": [str(p) for p in qc_copies], "image_validation": "PNG verify + decode; RGB; 256x256; one frame",
             "training_not_started": True}
    patient_rows = [{**patients[p], "split": split[p], "tss": p.split("-")[1]} for p in sorted(patients)]
    json_files = {"input_audit.json": audit, "environment.json": environment, "source_inputs.json": sources,
                  "protocol.json": protocol, "run_status.json": {"status": "PREPARED", "utc": now,
                    "training_status": "NOT_STARTED", "next_step": "Stage 29 GPU feasibility smoke/pilot; no test evaluation"}}
    csv_files = {"patient_splits.csv": patient_rows, "pilot_manifest.csv": pilots, "patch_inventory.csv": inventory}
    status = publish_bundle(output, json_files, csv_files, identity)
    print(f"{status}: {output}\nData and splits frozen; training remains DRAFT_PENDING_GPU_SMOKE.")
    print("Outputs: patient_splits.csv (929), pilot_manifest.csv (80), patch_inventory.csv (59456), JSON audit/protocol.")
    return 0


def self_test():
    """Temporary synthetic tests; never uses the production root or CUDA."""
    import unittest
    from unittest.mock import patch
    from contextlib import redirect_stdout
    from PIL import Image, PngImagePlugin

    class Tests(unittest.TestCase):
        def setUp(self):
            self.temp = tempfile.TemporaryDirectory(prefix="stage28-selftest-")
            self.root = Path(self.temp.name)
        def tearDown(self):
            self.temp.cleanup()
        def patients(self):
            return {f"TCGA-{code}-{i:04d}": {"class_label": label}
                    for label, code, count in (("LUAD", "AA", 455), ("LUSC", "BB", 474)) for i in range(count)}
        def test_stratification_and_order_invariance(self):
            p = self.patients()
            split = new_split(p)
            self.assertEqual(split, new_split(dict(reversed(list(p.items())))))
            self.assertEqual(collections.Counter(split.values()), {"train": 557, "validation": 186, "test": 186})
            self.assertEqual(collections.Counter(p[x]["class_label"] for x in p if split[x] == "train"), {"LUAD": 273, "LUSC": 284})
            pilot = pilot_rows(p, split)
            self.assertEqual(collections.Counter(r["split"] for r in pilot), {"train": 60, "validation": 20})
            self.assertEqual(len({r['case_submitter_id'] for r in pilot}), 80)
        def test_existing_split_and_conflicts(self):
            patients = self.patients()
            split = new_split(patients)
            path = self.root / "split.csv"
            rows = [{"patient_id": p, "split": "val" if s == "validation" else s} for p, s in split.items()]
            write_csv(path, rows)
            self.assertEqual(read_split(path, patients, []), split)
            path2 = self.root / "duplicate.csv"
            write_csv(path2, rows + [rows[0]])
            with self.assertRaises(AuditError):
                read_split(path2, patients, [])
            path3 = self.root / "missing.csv"
            write_csv(path3, rows[:-1])
            with self.assertRaises(AuditError):
                read_split(path3, patients, [])
        def test_windows_paths_and_traversal(self):
            r = {"case_submitter_id": "TCGA-AA-0001", "tile_name": "tile_001_x0_y256.png",
                 "tile_index": "1", "x_level0": "0", "y_level0": "256"}
            r["tile_path"] = "C:\\\\research\\\\mm_oncology_efficiency\\\\data\\\\tiles\\\\" + NAMESPACE + "\\\\" + r["case_submitter_id"] + "\\\\" + r["tile_name"]
            path, idx, x, y = tile_location(r, self.root)
            self.assertEqual(path, self.root / r["case_submitter_id"] / r["tile_name"])
            r["tile_path"] = r["tile_path"].replace("data", "..\\data")
            with self.assertRaises(AuditError):
                tile_location(r, self.root)
        def test_decoding_size_mode_and_truncation(self):
            path = self.root / "good.png"
            Image.new("RGB", (256, 256), (100, 30, 60)).save(path)
            self.assertEqual(len(read_png(path)["pixel_sha256"]), 64)
            for mode, size, name in (("RGBA", (256, 256), "rgba.png"), ("RGB", (128, 256), "small.png")):
                bad = self.root / name
                Image.new(mode, size).save(bad)
                with self.assertRaises(AuditError):
                    read_png(bad)
            broken = self.root / "broken.png"
            broken.write_bytes(path.read_bytes()[:100])
            with self.assertRaises((AuditError, OSError, SyntaxError)):
                read_png(broken)
        def test_pixel_duplicate_metadata_independence_and_readonly(self):
            a, b = self.root / "a.png", self.root / "b.png"
            im = Image.new("RGB", (256, 256), (50, 60, 70))
            im.save(a)
            metadata = PngImagePlugin.PngInfo()
            metadata.add_text("note", "different encoded file, identical pixels")
            im.save(b, pnginfo=metadata)
            self.assertNotEqual(read_png(a)["file_sha256"], read_png(b)["file_sha256"])
            rows = [{"case_submitter_id": pid, "class_label": "LUAD", "_path": str(path), "_index": i,
                     "_x": i * 256, "_y": 0, "tile_name": path.name}
                    for i, (pid, path) in enumerate((("TCGA-AA-0001", a), ("TCGA-AA-0002", b)), 1)]
            before = {p.name: sha_file(p) for p in self.root.iterdir()}
            with self.assertRaisesRegex(AuditError, "cross-patient"):
                scan_images(rows, progress=False)
            rows[1]["case_submitter_id"] = rows[0]["case_submitter_id"]
            inventory, summary = scan_images(rows, progress=False)
            self.assertEqual(summary["exact_pixel_duplicates_within_patient"], 1)
            self.assertEqual(before, {p.name: sha_file(p) for p in self.root.iterdir()})
        def test_publish_idempotence_and_tamper(self):
            output = self.root / "results" / OUTPUT_NAME
            json_files = {n: {"test": 1} for n in OWNED_FILES if n.endswith(".json")}
            csv_files = {n: [{"x": "1"}] for n in OWNED_FILES if n.endswith(".csv")}
            self.assertEqual(publish_bundle(output, json_files, csv_files, {"test": "same"}), "PREPARED")
            before = {p.name: (sha_file(p), p.stat().st_mtime_ns) for p in output.iterdir()}
            self.assertEqual(publish_bundle(output, json_files, csv_files, {"test": "same"}), "ALREADY_PREPARED")
            self.assertEqual(before, {p.name: (sha_file(p), p.stat().st_mtime_ns) for p in output.iterdir()})
            with self.assertRaises(AuditError):
                publish_bundle(output, json_files, csv_files, {"test": "changed"})
            (output / "protocol.json").write_text("{}", encoding="utf-8")
            with self.assertRaises(AuditError):
                verify_bundle(output)
        def test_existing_unprepared_output_not_overwritten(self):
            out = self.root / "existing"
            out.mkdir()
            (out / "user.txt").write_text("preserve", encoding="utf-8")
            with self.assertRaises(AuditError):
                publish_bundle(out, {}, {}, {})
            self.assertEqual((out / "user.txt").read_text(), "preserve")
        def test_split_discovery_and_ambiguity(self):
            p = self.patients()
            base = self.root / "data" / "manifests"
            base.mkdir(parents=True)
            out = self.root / "results" / OUTPUT_NAME
            original = new_split(p)
            write_csv(base / "patient_split.csv", [{"patient_id": pid, "split": s} for pid, s in original.items()])
            selected, origin = select_split(self.root, out, None, p, [])
            self.assertEqual(selected, original)
            self.assertEqual(origin["kind"], "auto_existing")
            (base / "old_folds.json").write_text("{}", encoding="utf-8")
            with self.assertRaises(AuditError):
                select_split(self.root, out, None, p, [])
            selected, origin = select_split(self.root, out, base / "patient_split.csv", p, [])
            self.assertEqual(selected, original)
        def test_old_qc_replacement_test_is_not_a_dataset_split(self):
            ns = "production_v2_artifact_qc_candidate_v1"
            batch = "artifact_qc3_replacement_test"
            csv_paths = [
                f"data/manifests/{ns}/batches/{batch}_patients.csv",
                f"data/manifests/{ns}/batches/{batch}_tile_manifest.csv",
                f"data/metadata/{ns}/batches/{batch}_patient_qc.csv",
                *[f"data/metadata/{ns}/artifact_rejections/{batch}/{pid}_wsi.csv"
                  for pid in ("TCGA-78-8660", "TCGA-86-8056", "TCGA-86-8278")],
            ]
            for relative in csv_paths:
                path = self.root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                write_csv(path, [{"case_submitter_id": "TCGA-78-8660", "class_label": "LUAD", "status": "PASS"}])
            metadata_path = self.root / f"data/manifests/{ns}/batches/{batch}_patients.json"
            write_json(metadata_path, {"batch_name": batch, "patients": [
                {"case_submitter_id": "TCGA-78-8660", "class_label": "LUAD"}]})
            before = {str(p): sha_file(p) for p in self.root.rglob("*") if p.is_file()}
            patients = self.patients()
            selected, origin = select_split(self.root, self.root / "results" / OUTPUT_NAME, None, patients, [])
            self.assertEqual(origin["kind"], "new_deterministic")
            self.assertEqual(selected, new_split(patients))
            self.assertEqual(origin["split_candidates"], [])
            self.assertEqual(len(origin["non_split_test_files"]), 7)
            self.assertEqual(before, {str(p): sha_file(p) for p in self.root.rglob("*") if p.is_file()})
        def test_real_split_detected_by_content_even_with_qc_filename(self):
            patients = self.patients()
            original = new_split(patients)
            path = self.root / "data/manifests/artifact_qc3_replacement_test_patients.csv"
            path.parent.mkdir(parents=True)
            write_csv(path, [{"case_submitter_id": p, "split": s} for p, s in original.items()])
            selected, origin = select_split(self.root, self.root / "results" / OUTPUT_NAME, None, patients, [])
            self.assertEqual(selected, original)
            self.assertEqual(origin["kind"], "auto_existing")
            path.rename(path.with_name("assignments.csv"))
            selected, origin = select_split(self.root, self.root / "results" / OUTPUT_NAME, None, patients, [])
            self.assertEqual(origin["kind"], "auto_existing")
            self.assertEqual(selected, original)
        def test_quantile_qc_role_columns_are_not_patient_partitions(self):
            patients = self.patients()
            for pid, role_column in zip(("TCGA-05-4396", "TCGA-05-4397", "TCGA-05-4434"),
                                        ("subset", "partition", "split")):
                path = self.root / "results/final_tissue_distribution_audit_v1" / pid / "final_tissue_quantiles.csv"
                path.parent.mkdir(parents=True)
                # Representative QC groups, not a copy of the user's unavailable CSV.
                write_csv(path, [{"case_submitter_id": pid, role_column: group, "q50": "0.81"}
                                 for group in ("all_candidates", "artifact_pass", "selected")])
            before = {str(p): sha_file(p) for p in self.root.rglob("*") if p.is_file()}
            selected, origin = select_split(self.root, self.root / "results" / OUTPUT_NAME, None, patients, [])
            self.assertEqual(selected, new_split(patients))
            self.assertEqual(origin["kind"], "new_deterministic")
            self.assertEqual(origin["split_candidates"], [])
            self.assertEqual(len(origin["non_split_role_tables"]), 3)
            self.assertTrue(all(r["rows_checked"] == 3 for r in origin["non_split_role_tables"]))
            self.assertEqual(before, {str(p): sha_file(p) for p in self.root.rglob("*") if p.is_file()})
        def test_real_subset_split_under_audit_name_is_still_reused(self):
            patients = self.patients()
            original = new_split(patients)
            path = self.root / "results/final_tissue_distribution_audit_v1/final_tissue_quantiles.csv"
            path.parent.mkdir(parents=True)
            write_csv(path, [{"patient_id": p, "subset": " TRAINING " if s == "train" else s.upper()}
                             for p, s in original.items()])
            selected, origin = select_split(self.root, self.root / "results" / OUTPUT_NAME, None, patients, [])
            self.assertEqual(selected, original)
            self.assertEqual(origin["kind"], "auto_existing")
        def test_mixed_invalid_and_valid_roles_do_not_get_ignored(self):
            patients = self.patients()
            path = self.root / "data/manifests/assignments.csv"
            path.parent.mkdir(parents=True)
            rows = [{"patient_id": p, "subset": "unrecognized"} for p in patients]
            # Must inspect past an arbitrary preview limit; valid label is last.
            rows[-1]["subset"] = "test"
            write_csv(path, rows)
            self.assertIsNotNone(split_evidence(path, self.root))
            with self.assertRaisesRegex(AuditError, "Unsupported partition"):
                select_split(self.root, self.root / "results" / OUTPUT_NAME, None, patients, [])
        def test_explicit_split_with_only_invalid_values_still_blocks(self):
            patients = self.patients()
            path = self.root / "data/manifests/patient_split.csv"
            path.parent.mkdir(parents=True)
            write_csv(path, [{"patient_id": p, "subset": "unknown"} for p in patients])
            with self.assertRaisesRegex(AuditError, "Unsupported partition"):
                select_split(self.root, self.root / "results" / OUTPUT_NAME, None, patients, [])
        def test_json_qc_subsets_versus_actual_patient_roles(self):
            pid = "TCGA-AA-0001"
            self.assertFalse(json_has_patient_split({"patients": [{"patient_id": pid, "subset": "all_candidates"}]}))
            self.assertFalse(json_has_patient_split({"subset": {"all_candidates": [pid]}}))
            for role in ("split", "partition", "subset"):
                self.assertTrue(json_has_patient_split({"patient_id": pid, role: " TEST "}))
            self.assertTrue(json_has_patient_split({"partition": {"train_ids": [pid]}}))
            self.assertTrue(json_has_patient_split({pid: " TRAINING "}))
            self.assertTrue(json_has_patient_split({"patient_id": pid, "fold": 0}))
        def test_generic_fold_assignments_still_require_review(self):
            patients = self.patients()
            path = self.root / "data/manifests/assignments.csv"
            path.parent.mkdir(parents=True)
            write_csv(path, [{"patient_id": p, "fold_id": i % 5} for i, p in enumerate(patients)])
            with self.assertRaisesRegex(AuditError, "need review"):
                select_split(self.root, self.root / "results" / OUTPUT_NAME, None, patients, [])
        def test_partial_test_list_and_json_split_still_require_review(self):
            base = self.root / "data/manifests"
            base.mkdir(parents=True)
            patients = self.patients()
            out = self.root / "results" / OUTPUT_NAME
            partial = base / "test_patients.csv"
            write_csv(partial, [{"patient_id": next(iter(patients))}])
            with self.assertRaisesRegex(AuditError, "need review"):
                select_split(self.root, out, None, patients, [])
            partial.unlink()
            generic_json = base / "assignments.json"
            write_json(generic_json, {"train_ids": [next(iter(patients))]})
            with self.assertRaisesRegex(AuditError, "need review"):
                select_split(self.root, out, None, patients, [])
        def test_conflicting_real_splits_still_block(self):
            patients = self.patients()
            original = new_split(patients)
            base = self.root / "data/manifests"
            base.mkdir(parents=True)
            write_csv(base / "first.csv", [{"patient_id": p, "split": s} for p, s in original.items()])
            changed = dict(original)
            changed[next(p for p in changed if changed[p] == "train")] = "test"
            write_csv(base / "second.csv", [{"patient_id": p, "split": s} for p, s in changed.items()])
            with self.assertRaisesRegex(AuditError, "Conflicting existing splits"):
                select_split(self.root, self.root / "results" / OUTPUT_NAME, None, patients, [])
        def test_readonly_prepare_resume_and_environment_gate(self):
            # End-to-end orchestration on a SMALL synthetic fixture. Production
            # manifest validation is intentionally outside this injected fixture.
            patients, tiles = {}, []
            for i in range(200):
                pid = f"TCGA-AA-{i:04d}"
                label = "LUAD" if i < 100 else "LUSC"
                patients[pid] = {"case_submitter_id": pid, "class_label": label}
                path = self.root / f"image{i}.png"
                Image.new("RGB", (256, 256), (i, 80, 90)).save(path)
                tiles.append({"case_submitter_id": pid, "class_label": label, "_path": str(path),
                              "_index": 1, "_x": 0, "_y": 0, "tile_name": path.name})
            args = argparse.Namespace(root=self.root, tiles_root=None, split_file=None, prepare=False)
            fake_env = {"ready": True, "versions": {"Pillow": "synthetic-fixture"}}
            overrides = {"source_paths": lambda root: ({}, []), "verify_sources": lambda paths: {},
                         "validate_manifests": lambda paths, root: (patients, [], tiles),
                         "environment_probe": lambda root: fake_env}
            before = {str(p): (sha_file(p), p.stat().st_mtime_ns) for p in self.root.rglob("*") if p.is_file()}
            with patch.dict(globals(), overrides), redirect_stdout(io.StringIO()):
                self.assertEqual(run(args), 0)
                self.assertFalse((self.root / "results").exists())
                fake_env["ready"] = False
                args.prepare = True
                self.assertEqual(run(args), 2)
                self.assertFalse((self.root / "results").exists())
                self.assertEqual(before, {str(p): (sha_file(p), p.stat().st_mtime_ns)
                                         for p in self.root.rglob("*") if p.is_file()})
                fake_env["ready"] = True
                self.assertEqual(run(args), 0)
                out = self.root / "results" / OUTPUT_NAME
                first = {p.name: (sha_file(p), p.stat().st_mtime_ns) for p in out.iterdir()}
                self.assertEqual(run(args), 0)
                self.assertEqual(first, {p.name: (sha_file(p), p.stat().st_mtime_ns) for p in out.iterdir()})

    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    return 0 if result.wasSuccessful() else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preflight-only", action="store_true", help="Full read-only audit (default).")
    mode.add_argument("--prepare", action="store_true", help="Full audit, then atomically freeze data/splits; no training.")
    mode.add_argument("--self-test", action="store_true", help="Synthetic temporary tests; no real dataset/GPU required.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent, help="Project root (default: script parent's parent).")
    parser.add_argument("--tiles-root", type=Path, help="Relocated directory containing the production patient PNG folders.")
    parser.add_argument("--split-file", type=Path, help="Explicit existing complete patient split CSV to reuse.")
    args = parser.parse_args()
    try:
        return self_test() if args.self_test else run(args)
    except KeyboardInterrupt:
        print("\nINTERRUPTED. Source files unchanged; rerun the same command to repeat the audit.", file=sys.stderr)
        return 130
    except (AuditError, OSError, ValueError, KeyError, csv.Error, ImportError) as e:
        print(f"ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
