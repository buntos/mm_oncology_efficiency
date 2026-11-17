#!/usr/bin/env python3
"""Paired 60-patient WSI tissue-threshold experiment, version 1.0.0.

Save as PROJECT/scripts/26_run_paired60_tissue_thresholds.py.
Uses the ORIGINAL research venv (Python 3.12, numpy, Pillow, scikit-image,
openslide-python/native OpenSlide) and PROJECT/tools/gdc-client.exe.
No import or modification of scripts 11/17/24/25 or production output.

Commands (run from the project root with .venv activated):
  python scripts/26_run_paired60_tissue_thresholds.py --preflight-only
  python scripts/26_run_paired60_tissue_thresholds.py --prepare
  python scripts/26_run_paired60_tissue_thresholds.py --run --limit 1
  python scripts/26_run_paired60_tissue_thresholds.py --run
  python scripts/26_run_paired60_tissue_thresholds.py --summarize-only
  python scripts/26_run_paired60_tissue_thresholds.py --self-test

Preflight is read-only, no network; prepare saves the fixed protocol/sample,
no WSI downloads. Run freezes if not prepared and resumes patient checkpoints.
Outputs: PROJECT/results/tissue_threshold_paired60_v1.
No seed/size overrides: do not search for a more favorable sample.

Primary endpoint: patient-level fraction of coordinates replaced at 0.90 versus
0.80, conditional on BOTH 64-patch sets being complete on the same fixed DX1.
Availability/attrition is reported separately across all 60 selected patients.
Other contrasts are secondary/exploratory. Technical errors are never counted
as QC failures or replaced with another patient. No alternate-slide fallback.
The 60-patient balanced sample is not the original 950-patient cohort and is
not an independent external validation set (prior study overlap is reported).

First 64 accepted patches at each threshold; candidate mask threshold 0.70,
level-0 grid 256x256, stride 256 and ALL other QC rules remain frozen. All five
thresholds share the same deterministic candidate order and one WSI traversal.
A scan stops when all thresholds have 64 patches or the candidate pool ends.
Thus counters are per-threshold stopping counts, not whole-slide prevalence.
Total scan_seconds measures the SHARED scan, not five independent runtimes.
Native MPP is recorded, not normalized. Tissue fraction is not tumor fraction
and patch replacement is not evidence of lost diagnostic information.

95% intervals: Wilson for complete-set proportions; deterministic percentile
bootstrap (10000 patient resamples) for paired metric means. Overall bootstrap
is stratified by class within eligible pairs; class-specific bootstrap resamples
patients in that class. No patch-level independent observations, p-value search,
threshold optimization, or automatic claim of publication sufficiency.
Final summaries appear only when all 60 checkpoints and input hashes validate.

Downloads have 3 attempts, 30-minute attempt timeout, size+MD5 validation and
an 8 GiB free-space buffer. --wsi-root reads existing UUID/name or flat/name
paths without changing them. Only this experiment's owned downloaded WSI are
deleted after a committed valid result, unless --keep-wsi is set. Candidate logs
and selected coordinates/metrics are kept; images/RNA/models are not generated.
An interrupted patient restarts its scan; completed patients are not repeated.

Exit codes: 0 success (including deliberate --limit partial run), 1 technical
or validation error, 2 baseline-control mismatch, 130 interruption. A stale
run.lock after forced termination may be removed ONLY after checking that no
copy of this script is still running. Do not edit code/dependencies mid-run.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import re
from contextlib import contextmanager
from types import SimpleNamespace
from datetime import datetime, timezone

VERSION = '1.0.0'
EXPERIMENT = 'tissue_threshold_paired60_v1'
SAMPLE_SEED = '20261111'
SAMPLE_SALT = 'WSI_PAIRED60_V1'
PER_CLASS = 30
BOOTSTRAP_REPLICATES = 10000
PRIMARY_THRESHOLD = '0.90'
BASELINE_THRESHOLD = '0.80'


NAMESPACE = 'production_final_v2_tissue080'
THRESHOLDS = (0.70, 0.75, 0.80, 0.85, 0.90)
TARGET_TILES = 64
TILE_SIZE = STRIDE = 256
THUMBNAIL_MAX_SIZE = 4096
CANDIDATE_TISSUE_FRACTION = 0.70
SATURATION_FLOOR = 0.03
MAX_TISSUE_VALUE = 0.95
BACKGROUND_SATURATION_MAX = 0.04
BACKGROUND_VALUE_MIN = 0.55
MAX_BACKGROUND_FRACTION = 0.10
WHITE_VALUE_MIN = 0.97
MAX_WHITE_FRACTION = 0.10
MORPH_MAX_SIZE = 31
ARTIFACT_RULES = {
    'near_black_fraction_min': 0.30, 'flat_gradient_max': 0.30,
    'flat_red_excess_min': 125.0, 'flat_cyan_excess_min': 8.0,
    'hyper_red_excess_min': 140.0, 'hyper_red_saturation_min': 0.75,
    'hyper_cyan_excess_min': 18.0, 'hyper_cyan_saturation_min': 0.60,
}
SOURCE_HASHES = {
    'script11': '3af2ff8ce9d085a3723eb29542f10fdbc97aa449422fafd3815683d5e5857b60',
    'script17': '89edd15f150e30f424e945290ed22ba23cebc75d679d2ad70db3c3669d805b86',
    'script24': 'c97ada4ce9b5f8a3f669b63035f90f072ab13233906ce42e87ebd005848cd664',
}
COHORT_HASH = '0b62f5aa3b1dc09f9e96c411edb04b4456004a188eaa882072e4339e2d1611fa'


def require(condition, message):
    if not condition:
        raise RuntimeError(message)

def digest(path, kind='sha256'):
    h = hashlib.new(kind)
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)

def utc_now():
    return datetime.now(timezone.utc).isoformat()

def load_dependencies():
    global np, Image, openslide, rgb2hsv, threshold_otsu
    global remove_small_objects, remove_small_holes, base
    import numpy as np
    from PIL import Image
    import openslide
    from skimage.color import rgb2hsv
    from skimage.filters import threshold_otsu
    from skimage.morphology import remove_small_objects, remove_small_holes
    require('max_size' in inspect.signature(remove_small_objects).parameters,
            'This protocol needs the original skimage morphology max_size API. '
            'Use the existing research venv; do not change its dependencies blindly.')
    require('max_size' in inspect.signature(remove_small_holes).parameters,
            'remove_small_holes lacks max_size; use the original research venv.')
    base = SimpleNamespace(**{k: v for k, v in globals().items()
                             if k.isupper() or k in ('rgb_to_hsv_arrays', 'integral_image', 'rect_sum')})
    versions = {'python': platform.python_version(), 'platform': platform.platform(),
                'openslide_native': openslide.__library_version__}
    for package in ('numpy', 'Pillow', 'scikit-image', 'openslide-python'):
        versions[package] = importlib.metadata.version(package)
    return versions

def deterministic_seed(text):
    digest = hashlib.sha256(text.encode('utf-8')).digest()
    return int.from_bytes(digest[:8], byteorder='little', signed=False)

def integral_image(mask):
    return mask.astype(np.uint32).cumsum(axis=0).cumsum(axis=1)

def rect_sum(ii, x1, y1, x2, y2):
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(ii.shape[1], x2)
    y2 = min(ii.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return 0
    result = int(ii[y2 - 1, x2 - 1])
    if x1 > 0:
        result -= int(ii[y2 - 1, x1 - 1])
    if y1 > 0:
        result -= int(ii[y1 - 1, x2 - 1])
    if x1 > 0 and y1 > 0:
        result += int(ii[y1 - 1, x1 - 1])
    return result

def rgb_to_hsv_arrays(rgb_uint8):
    rgb_float = rgb_uint8.astype(np.float32) / 255.0
    hsv = rgb2hsv(rgb_float)
    return (hsv[:, :, 1], hsv[:, :, 2])

def create_tissue_mask(thumbnail):
    rgb = np.asarray(thumbnail)
    saturation, value = rgb_to_hsv_arrays(rgb)
    raw_otsu = float(threshold_otsu(saturation))
    saturation_threshold = max(raw_otsu, SATURATION_FLOOR)
    tissue_mask = (saturation >= saturation_threshold) & (value <= MAX_TISSUE_VALUE)
    tissue_mask = remove_small_objects(tissue_mask, max_size=MORPH_MAX_SIZE)
    tissue_mask = remove_small_holes(tissue_mask, max_size=MORPH_MAX_SIZE)
    return (tissue_mask, raw_otsu, saturation_threshold)

def evaluate_tile(tile, saturation_threshold):
    rgb = np.asarray(tile)
    saturation, value = rgb_to_hsv_arrays(rgb)
    tissue_pixels = (saturation >= saturation_threshold) & (value <= MAX_TISSUE_VALUE)
    background_pixels = (saturation <= BACKGROUND_SATURATION_MAX) & (value >= BACKGROUND_VALUE_MIN)
    white_pixels = value >= WHITE_VALUE_MIN
    return (float(tissue_pixels.mean()), float(background_pixels.mean()), float(white_pixels.mean()))

def calculate_artifact_metrics(tile):
    rgb = np.asarray(tile.convert('RGB'), dtype=np.float32)
    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]
    max_channel = rgb.max(axis=2)
    near_black_fraction = float((max_channel < 50).mean())
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    grad_x = np.abs(np.diff(gray, axis=1))
    grad_y = np.abs(np.diff(gray, axis=0))
    gradient_mean = float((grad_x.mean() + grad_y.mean()) / 2.0)
    red_excess_map = r - (g + b) / 2.0
    cyan_excess_map = (g + b) / 2.0 - r
    red_excess = float(red_excess_map.mean())
    cyan_excess = float(cyan_excess_map.mean())
    saturation, _ = base.rgb_to_hsv_arrays(rgb.astype(np.uint8))
    saturation_mean = float(saturation.mean())
    return {'near_black_fraction': near_black_fraction, 'gradient_mean': gradient_mean, 'red_excess': red_excess, 'cyan_excess': cyan_excess, 'saturation_mean': saturation_mean}

def evaluate_artifact_qc(metrics):
    reasons = []
    if metrics['near_black_fraction'] >= ARTIFACT_RULES['near_black_fraction_min']:
        reasons.append('near_black')
    if metrics['gradient_mean'] <= ARTIFACT_RULES['flat_gradient_max'] and metrics['red_excess'] >= ARTIFACT_RULES['flat_red_excess_min']:
        reasons.append('flat_extreme_red')
    if metrics['gradient_mean'] <= ARTIFACT_RULES['flat_gradient_max'] and metrics['cyan_excess'] >= ARTIFACT_RULES['flat_cyan_excess_min']:
        reasons.append('flat_extreme_cyan')
    if metrics['red_excess'] >= ARTIFACT_RULES['hyper_red_excess_min'] and metrics['saturation_mean'] >= ARTIFACT_RULES['hyper_red_saturation_min']:
        reasons.append('hyper_red_saturated')
    if metrics['cyan_excess'] >= ARTIFACT_RULES['hyper_cyan_excess_min'] and metrics['saturation_mean'] >= ARTIFACT_RULES['hyper_cyan_saturation_min']:
        reasons.append('hyper_cyan_saturated')
    return (len(reasons) > 0, reasons)

def generate_candidates(slide, tissue_mask):
    width, height = slide.dimensions
    thumb_h, thumb_w = tissue_mask.shape
    scale_x = width / thumb_w
    scale_y = height / thumb_h
    ii = base.integral_image(tissue_mask)
    candidates = []
    for y in range(0, height - base.TILE_SIZE + 1, base.STRIDE):
        ty1 = int(np.floor(y / scale_y))
        ty2 = int(np.ceil((y + base.TILE_SIZE) / scale_y))
        for x in range(0, width - base.TILE_SIZE + 1, base.STRIDE):
            tx1 = int(np.floor(x / scale_x))
            tx2 = int(np.ceil((x + base.TILE_SIZE) / scale_x))
            area = max(1, tx2 - tx1) * max(1, ty2 - ty1)
            tissue_pixels = base.rect_sum(ii, tx1, ty1, tx2, ty2)
            approx_fraction = tissue_pixels / area
            if approx_fraction >= base.CANDIDATE_TISSUE_FRACTION:
                candidates.append((int(x), int(y), float(approx_fraction)))
    return candidates

def new_states():
    return {format(t, '.2f'): {'threshold': t, 'checked': 0, 'accepted': [],
            'rejected_low_tissue': 0, 'rejected_background': 0, 'rejected_white': 0,
            'rejected_artifact': 0, 'checked_to_64': None} for t in THRESHOLDS}

def update_states(states, record):
    """Counters use the original sequential rejection order, separately per threshold."""
    decisions = {}
    for key, s in states.items():
        if len(s['accepted']) == TARGET_TILES:
            decisions[key] = 'already_complete'
            continue
        s['checked'] += 1
        if record['final_tissue_fraction'] < s['threshold']:
            reason = 'low_tissue'
        elif record['background_fraction'] > MAX_BACKGROUND_FRACTION:
            reason = 'background'
        elif record['white_fraction'] > MAX_WHITE_FRACTION:
            reason = 'white'
        elif record['artifact_reject']:
            reason = 'artifact'
        else:
            reason = 'accepted'
        decisions[key] = reason
        if reason == 'accepted':
            s['accepted'].append(dict(record))
            if len(s['accepted']) == TARGET_TILES:
                s['checked_to_64'] = s['checked']
        else:
            s['rejected_' + reason] += 1
    return decisions

def verified_file(path, item):
    return (path.is_file() and path.stat().st_size == item['file_size_bytes']
            and digest(path, 'md5').lower() == item['md5sum'].lower())

def obtain_wsi(item, args, output):
    # Never delete/rewrite a user-supplied source.
    if args.wsi_root:
        for path in [args.wsi_root / item['file_id'] / item['file_name'], args.wsi_root / item['file_name']]:
            if path.exists():
                require(verified_file(path, item), f'Local WSI failed size/MD5 check: {path}')
                return path, False
    cache = output / 'downloads'
    path = cache / item['file_id'] / item['file_name']
    if verified_file(path, item):
        return path, True
    require(args.gdc_client.is_file(), f'GDC client not found: {args.gdc_client}')
    cache.mkdir(parents=True, exist_ok=True)
    require(shutil.disk_usage(cache).free >= item['file_size_bytes'] + 8 * 1024**3,
            'Insufficient disk space: need WSI size plus 8 GiB free buffer')
    logpath = output / 'logs' / (item['case_submitter_id'] + '_download.log')
    logpath.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, 4):
        # Only discard an invalid completed file owned by this audit, never .partial data.
        if path.is_file():
            path.unlink()
        print(f'  Download {attempt}/3; log: {logpath}', flush=True)
        with logpath.open('ab') as log:
            log.write(f'\n{utc_now()} attempt={attempt}\n'.encode())
            try:
                subprocess.run([str(args.gdc_client), 'download', item['file_id'], '-d', str(cache),
                                '--no-related-files', '--no-annotations'], cwd=args.root,
                               stdout=log, stderr=subprocess.STDOUT, timeout=1800, check=False)
            except subprocess.TimeoutExpired:
                log.write(b'\nDownload timeout after 1800 seconds\n')
        if verified_file(path, item):
            return path, True
        if attempt < 3:
            time.sleep(5)
    raise RuntimeError(f'Download failed size/MD5 validation: {item["file_id"]}; see {logpath}')

def scan_wsi(path, item, patient_dir):
    states = new_states()
    seed = deterministic_seed(item['case_submitter_id'] + '|' + item['file_id'])
    logpath = patient_dir / 'candidates.csv.gz'
    partial = patient_dir / 'candidates.partial.csv.gz'
    started = time.monotonic()
    with openslide.OpenSlide(str(path)) as slide:
        thumbnail = slide.get_thumbnail((THUMBNAIL_MAX_SIZE, THUMBNAIL_MAX_SIZE)).convert('RGB')
        tissue_mask, raw_otsu, saturation = create_tissue_mask(thumbnail)
        candidates = generate_candidates(slide, tissue_mask)
        order = np.arange(len(candidates))
        np.random.default_rng(seed).shuffle(order)
        metadata = {'width': slide.dimensions[0], 'height': slide.dimensions[1],
                    'objective': slide.properties.get(openslide.PROPERTY_NAME_OBJECTIVE_POWER),
                    'mpp_x': slide.properties.get(openslide.PROPERTY_NAME_MPP_X),
                    'mpp_y': slide.properties.get(openslide.PROPERTY_NAME_MPP_Y),
                    'raw_otsu_saturation': float(raw_otsu), 'saturation_threshold': float(saturation),
                    'tissue_mask_fraction': float(tissue_mask.mean()), 'deterministic_seed': str(seed),
                    'candidates_total': len(candidates)}
        thumbnail.close()
        checked = 0
        last_report = time.monotonic()
        columns = ['candidate_checked_index', 'x_level0', 'y_level0', 'approx_tissue_fraction',
                   'final_tissue_fraction', 'background_fraction', 'white_fraction',
                   'artifact_evaluated', 'artifact_reject', 'artifact_reasons',
                   'near_black_fraction', 'gradient_mean', 'red_excess', 'cyan_excess', 'saturation_mean',
                   ] + ['decision_' + format(t, '.2f') for t in THRESHOLDS]
        with gzip.open(partial, 'wt', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            for ix in order:
                x, y, approximate = candidates[ix]
                checked += 1
                with slide.read_region((int(x), int(y)), 0, (TILE_SIZE, TILE_SIZE)) as region:
                    tile = region.convert('RGB')
                try:
                    tissue, background, white = evaluate_tile(tile, saturation)
                    metrics = {k: None for k in ('near_black_fraction', 'gradient_mean', 'red_excess',
                                                'cyan_excess', 'saturation_mean')}
                    evaluated = tissue >= min(THRESHOLDS) and background <= .1 and white <= .1
                    rejected, reasons = False, []
                    if evaluated:
                        metrics = calculate_artifact_metrics(tile)
                        rejected, reasons = evaluate_artifact_qc(metrics)
                finally:
                    tile.close()
                record = {'candidate_checked_index': checked, 'x_level0': int(x), 'y_level0': int(y),
                          'approx_tissue_fraction': float(approximate), 'final_tissue_fraction': tissue,
                          'background_fraction': background, 'white_fraction': white,
                          'artifact_evaluated': evaluated, 'artifact_reject': rejected,
                          'artifact_reasons': ';'.join(reasons), **metrics}
                decisions = update_states(states, record)
                writer.writerow({**record, **{'decision_' + k: v for k, v in decisions.items()}})
                if time.monotonic() - last_report >= 20:
                    print(f'  Checked {checked}/{len(candidates)}; accepted ' +
                          ', '.join(f'{k}: {len(s["accepted"])}' for k, s in states.items()), flush=True)
                    last_report = time.monotonic()
                if all(len(s['accepted']) == TARGET_TILES for s in states.values()):
                    break
        metadata.update({'candidates_checked': checked, 'scan_exhausted': checked == len(candidates),
                         'scan_seconds': time.monotonic() - started})
    for s in states.values():
        require(s['checked'] == len(s['accepted']) + sum(s[k] for k in
                ('rejected_low_tissue', 'rejected_background', 'rejected_white', 'rejected_artifact')),
                'Internal counter reconciliation failed')
        require(len(s['accepted']) == 64 or metadata['scan_exhausted'], 'Incomplete scan cannot imply QC failure')
    os.replace(partial, logpath)
    return states, metadata, digest(logpath)


def validate_inputs(root):
    """Validate all original evidence without importing or mutating old pipelines."""
    manifests = root / 'data' / 'manifests'
    metadata = root / 'data' / 'metadata' / NAMESPACE
    paths = {
        'config': metadata / 'frozen_protocol_configuration.json',
        'cohort': manifests / 'paired_cohort_stage2.csv',
        'candidates': manifests / 'wsi_primary_diagnostic_candidates.csv',
        'tiles': manifests / NAMESPACE / 'batches' / 'final_full_cohort_v1_tile_manifest.csv',
    }
    # Both historical output locations are supported; ambiguous copies must agree.
    options = [metadata / 'batches' / 'final_full_cohort_v1_patient_qc.csv',
               manifests / NAMESPACE / 'batches' / 'final_full_cohort_v1_patient_qc.csv']
    found = [p for p in options if p.is_file()]
    require(found, 'Missing patient QC: expected ' + ' OR '.join(map(str, options)))
    require(len({digest(p) for p in found}) == 1, 'Conflicting copies of patient QC')
    paths['qc'] = found[0]
    for p in paths.values():
        require(p.is_file(), f'Missing required input: {p}')
    config = json.loads(paths['config'].read_text(encoding='utf-8-sig'))
    require(config['namespace'] == NAMESPACE, 'Unexpected frozen namespace')
    require(config['cohort']['stage2_cohort_sha256'] == COHORT_HASH, 'Unknown frozen cohort')
    require(digest(paths['cohort']) == COHORT_HASH, 'Cohort bytes differ from frozen protocol')
    require(config['artifact_qc']['rules'] == ARTIFACT_RULES, 'Artifact rules differ')
    expected = {'tile_size': 256, 'target_tiles': 64, 'candidate_tissue_fraction': .7,
                'final_tissue_fraction': .8, 'max_background_fraction': .1, 'max_white_fraction': .1}
    for key, value in expected.items():
        require(config['tiling'][key] == value, f'Frozen tiling mismatch: {key}')
    for key, expected_hash in SOURCE_HASHES.items():
        path = root / Path(config['code'][key].replace('\\', '/'))
        require(path.is_file(), f'Missing original script: {path}')
        require(config['code'][key + '_sha256'] == expected_hash, f'Unknown {key} version')
        require(digest(path) == expected_hash, f'Original {key} changed; restore frozen version: {path}')
        paths[key] = path
    cohort = read_csv(paths['cohort'])
    qc = read_csv(paths['qc'])
    candidates = read_csv(paths['candidates'])
    tiles = read_csv(paths['tiles'])
    ci = {r['case_submitter_id']: r for r in cohort}
    qi = {r['case_submitter_id']: r for r in qc}
    require(len(ci) == len(cohort) == len(qi) == len(qc) == 950, 'Expected 950 unique cohort/QC patients')
    require(set(ci) == set(qi), 'QC and cohort patients differ')
    for cl, total, passed in [('LUAD', 474, 455), ('LUSC', 476, 474)]:
        require(sum(r['class_label'] == cl for r in cohort) == total, f'Unexpected {cl} cohort count')
        require(sum(r['class_label'] == cl and r['status'] == 'PASS' for r in qc) == passed,
                f'Unexpected {cl} PASS count')
    by_tiles, by_candidates = {}, {}
    for r in candidates:
        if r['case_submitter_id'] in ci:
            by_candidates.setdefault(r['case_submitter_id'], []).append(r)
    seen = set()
    for r in tiles:
        case = r['case_submitter_id']
        require(case in qi and qi[case]['status'] == 'PASS', f'Tile for non-PASS case: {case}')
        require(r['class_label'] == qi[case]['class_label'] == ci[case]['class_label'], 'Tile class mismatch')
        require(r['selected_wsi_file_id'] == qi[case]['selected_wsi_file_id'], 'Tile WSI mismatch')
        require(r['artifact_qc_protocol'] == config['protocol_name'] and r['artifact_qc_status'] == 'PASS',
                'Tile protocol/status mismatch')
        for name, low, high in [('final_tissue_fraction', .8, 1), ('background_fraction', 0, .1),
                                ('white_fraction', 0, .1)]:
            require(low <= float(r[name]) <= high, f'Invalid {name} for {case}')
        key = (case, r['selected_wsi_file_id'], int(r['x_level0']), int(r['y_level0']))
        require(key not in seen, f'Duplicate tile coordinates: {case}')
        seen.add(key)
        by_tiles.setdefault(case, []).append(r)
    require(len(tiles) == 59456 and len(by_tiles) == 929, 'Expected 59456 tiles / 929 patients')
    require(all(len(rows) == 64 for rows in by_tiles.values()), 'Expected 64 tiles per PASS patient')
    excluded = []
    for case, r in qi.items():
        require(r['class_label'] == ci[case]['class_label'], f'QC class mismatch: {case}')
        require(r['status'] in ('PASS', 'EXCLUDED_WSI_QC'), 'Unexpected QC status')
        if r['status'] == 'EXCLUDED_WSI_QC':
            require(r['reason'] == 'all_diagnostic_slides_failed_wsi_qc', 'Unknown exclusion reason')
            require(float(r['wsi_technical_error_count']) == 0, 'Technical failures in exclusion cohort')
            require(float(r['diagnostic_slides_attempted']) == 1, 'Unexpected fallback in excluded cohort')
            excluded.append(case)
    require(len(excluded) == 21, 'Expected 21 excluded patients')
    hashes = {k: digest(p) for k, p in paths.items()}
    return {'hashes': hashes, 'paths': paths, 'tiles': by_tiles, 'qc': qi,
            'cohort': ci, 'candidates': by_candidates}


def atomic_csv(path, rows, columns=None):
    """Write a header even for an empty result; never leave stale previous rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    require(columns or rows, f'No schema for {path}')
    columns = columns or list(rows[0])
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def sample_key(case):
    return hashlib.sha256(f'{SAMPLE_SALT}|{SAMPLE_SEED}|{case}'.encode('utf-8')).hexdigest()


def make_plan(evidence):
    """One fixed SHA256-ranked draw per class, from ALL 950 before any new scan."""
    cohort, qc, candidates = (evidence[k] for k in ('cohort', 'qc', 'candidates'))
    prior_controls = set()
    for cl in ('LUAD', 'LUSC'):
        eligible = [case for case, r in qc.items() if r['status'] == 'PASS' and r['class_label'] == cl
                    and len(candidates.get(case, [])) == 1
                    and candidates[case][0]['file_id'] == r['selected_wsi_file_id']]
        require(eligible, f'No prior control candidate for {cl}')
        prior_controls.add(min(eligible, key=lambda case: (int(candidates[case][0]['file_size_bytes']), case)))
    selected = {cl: sorted([case for case, row in cohort.items() if row['class_label'] == cl],
                           key=lambda case: (sample_key(case), case))[:PER_CLASS]
                for cl in ('LUAD', 'LUSC')}
    plan = []
    # Alternating classes is execution order only; sample membership is fixed above.
    for rank in range(PER_CLASS):
        for cl in ('LUAD', 'LUSC'):
            case = selected[cl][rank]
            rows = candidates.get(case, [])
            require(rows, f'No WSI candidates for selected patient {case}; do not replace this patient')
            rows = sorted(rows, key=lambda r: (int(float(r['dx_index'])), r['associated_barcode'], r['file_id']))
            row = rows[0]
            require(row['slide_token'] == 'DX1' and int(float(row['dx_index'])) == 1, f'Expected DX1 for {case}')
            require(row['file_id'] == cohort[case]['wsi_file_id'], f'Frozen primary WSI differs for {case}')
            require(row['access'] == 'open', f'{case}: open access required')
            require(re.fullmatch(r'[0-9a-fA-F-]{36}', row['file_id']) is not None, 'Invalid WSI UUID')
            require(Path(row['file_name']).name == row['file_name'] and '\\' not in row['file_name'],
                    'Invalid WSI filename')
            require(re.fullmatch(r'[0-9a-fA-F]{32}', row['md5sum']) is not None, 'Invalid WSI MD5')
            require(row['md5sum'].lower() == cohort[case]['wsi_md5sum'].lower()
                    and row['file_name'] == cohort[case]['wsi_file_name'],
                    f'WSI name/MD5 differs from the frozen cohort: {case}')
            require(int(row['file_size_bytes']) == int(cohort[case]['wsi_file_size_bytes']) > 0,
                    f'WSI file-size metadata mismatch: {case}')
            if qc[case]['status'] == 'PASS':
                require(qc[case]['selected_wsi_file_id'] == row['file_id'],
                        f'{case}: original PASS used another slide; review design, do not substitute')
                original = evidence['tiles'][case]
                require(len({int(r['tile_index']) for r in original}) == 64, f'Duplicate tile index: {case}')
                expected_seed = str(deterministic_seed(case + '|' + row['file_id']))
                # The historical combined CSV serialized uint64 seeds as float64
                # scientific notation. NEVER recover the RNG seed from that lossy
                # field: derive the exact uint64 from the original SHA256 formula.
                # This check only verifies the old numeric representation; the
                # decisive reproducibility check is all 64 ordered coordinates.
                for r in original:
                    stored_seed = r['deterministic_seed']
                    if any(ch in stored_seed for ch in '.eE'):
                        require(float(stored_seed) == float(expected_seed), f'Legacy seed metadata mismatch: {case}')
                    else:
                        require(stored_seed == expected_seed, f'Seed mismatch: {case}')
            else:
                require(len(rows) == 1, f'{case}: prior exclusion had fallback; fixed-DX1 control needs review')
            plan.append({'execution_order': len(plan) + 1, 'case_submitter_id': case, 'class_label': cl,
                         'class_sample_rank': rank + 1, 'sample_key_sha256': sample_key(case),
                         'tss_code': case.split('-')[1], 'baseline_status': qc[case]['status'],
                         'prior_stage1_role': 'excluded' if qc[case]['status'] != 'PASS'
                         else 'positive_control' if case in prior_controls else 'not_in_stage1',
                         'file_id': row['file_id'], 'file_name': row['file_name'],
                         'file_size_bytes': int(row['file_size_bytes']), 'md5sum': row['md5sum'],
                         'slide_token': row['slide_token'], 'available_diagnostic_slides': len(rows)})
    require(len(plan) == len({r['case_submitter_id'] for r in plan}) == 60, 'Sample is not 60 unique patients')
    return plan


PROTOCOL_TEXT = """# Paired tissue-threshold study: fixed protocol v1

## Population and sampling
Source: frozen 950 paired TCGA LUAD/LUSC patients (474/476). Draw 30 per class
using ascending SHA256('WSI_PAIRED60_V1|20261111|' + case_submitter_id), with case
ID as tie-breaker. This draw includes all baseline QC statuses. Do not redraw,
replace failed patients, enlarge the sample after examining significance, or
present this balanced subsample as a new independent cohort. Prior stage-1
membership and TSS codes are recorded. The sample size is resource-constrained,
not a prospective power calculation. Descriptive/exploratory single-cohort study.

## Intervention
One fixed original primary DX1 per patient; no alternate-slide fallback at any
threshold. Thresholds 0.70, 0.75, 0.80, 0.85, 0.90. Native level 0, 256x256 pixels,
stride 256, candidate mask threshold 0.70, 64 patches. Original HSV/morphology,
background <=0.10, white <=0.10 and artifact rules are unchanged. First 64 valid
patches in the original deterministic order. Scan until all five thresholds
reach 64 or the candidate pool is exhausted. Every original PASS case must
reproduce 64 exact ordered coordinates and numerical QC metrics at 0.80; an
original excluded case must remain below 64 at 0.80. A mismatch blocks results.

## Primary endpoint and estimand
Primary contrast: 0.90 versus 0.80. For patients with BOTH full sets, calculate
replacement_fraction = 1 - |coordinates_080 intersection coordinates_090| / 64.
The primary estimate is the mean of this patient-level fraction in eligible
sampled patients. This is conditional on full-set availability; it does not
assign zero to patients who cannot supply a full set. Report the eligible n
and separate losses/gains/neither-complete counts alongside the endpoint.

## Secondary outcomes
Other threshold contrasts against 0.80 are exploratory. Report full-set
availability across the complete sample, coordinate Jaccard, patient-level
mean tissue/background/white fractions, paired differences in these means,
checked_to_64 difference and ratio. Per-threshold quality summaries condition
on a full 64-patch set; their populations can differ. Paired differences use
only patients complete at BOTH thresholds. Do not infer a quality change from
an unpaired comparison of different retained groups.

## Uncertainty
Unit of analysis: patient, never patch. 95% Wilson intervals for complete-set
proportions; 10000 deterministic percentile bootstrap patient resamples for
paired means (numpy quantile method='linear'). Overall bootstrap is stratified
by class among eligible pairs, preserving each stratum's eligible n. Class
analyses resample patients in that class. n<2 in any represented stratum:
no bootstrap interval; constant
observations produce a degenerate empirical interval, labelled explicitly.
No confirmatory p-value claims or multiple-comparison discovery. The ALL group
is the observed balanced subsample, NOT a prevalence-weighted 950-patient estimate.
Original cohort retention at 0.70-0.80 comes from the separate stage-1 analysis.

## Confounding and scope
Record native MPP, objective and TSS. Within-patient contrasts hold the slide
and resolution fixed; between-class differences can still reflect source/site,
scale or staining. TCGA-05 deserves explicit sensitivity discussion; no causal
histology claim from class contrasts. The tissue detector is not a tumor-cell
measurement or independent expert quality label. Coordinate replacement is not
proof of changed model embeddings or diagnostic accuracy. No classification or
clinical-quality claims without additional validation.

## Computation and missingness
One shared scan is used for five virtual stopping rules. checked_to_64 is a
candidate-count cost measure; shared scan_seconds is NOT a per-threshold time
benchmark. Failed thresholds have no checked_to_64, not zero. Candidate logs
contain sequential rejection reasons (first applicable filter), not independent
artifact/white/background prevalence. Completed insufficient sets are scientific
observations; interrupted/download/read failures are technical missingness.
Do not replace such patients. Final estimates/intervals require all 60 complete,
valid baseline controls and unchanged input hashes; partial files are labelled.
"""


def freeze_payload(evidence, plan, versions):
    return {'version': VERSION, 'experiment': EXPERIMENT,
            'audit_script_sha256': digest(Path(__file__)), 'input_sha256': evidence['hashes'],
            'sample_seed': SAMPLE_SEED, 'sample_salt': SAMPLE_SALT, 'per_class': PER_CLASS,
            'sampling': 'ascending SHA256 rank within class, from all 950; no replacement/redraw',
            'sample': plan, 'thresholds': list(THRESHOLDS), 'baseline': BASELINE_THRESHOLD,
            'primary_contrast': PRIMARY_THRESHOLD, 'primary_metric': 'replacement_fraction',
            'bootstrap_replicates': BOOTSTRAP_REPLICATES, 'bootstrap_unit': 'patient',
            'protocol_sha256': hashlib.sha256(PROTOCOL_TEXT.encode('utf-8')).hexdigest(),
            'environment': versions}


def run_signature(frozen):
    return hashlib.sha256(json.dumps(frozen, sort_keys=True).encode('utf-8')).hexdigest()


def verify_existing_freeze(output, frozen):
    path = output / 'experiment_configuration.json'
    if path.exists():
        require(json.loads(path.read_text(encoding='utf-8')) == frozen,
                'Frozen sample/code/inputs/environment differ. Keep this run intact; do not edit to bypass.')
    else:
        require(not (output / 'patients').exists(), 'Patient results exist without a frozen configuration')


@contextmanager
def run_lock(output):
    output.mkdir(parents=True, exist_ok=True)
    path = output / 'run.lock'
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    except FileExistsError:
        raise RuntimeError(f'Active run or stale lock: {path}. Remove only after checking no audit is running.')
    try:
        with os.fdopen(fd, 'w') as f:
            f.write(f'pid={os.getpid()} started={utc_now()}\n')
        yield
    finally:
        path.unlink(missing_ok=True)


def save_freeze(output, frozen):
    verify_existing_freeze(output, frozen)
    path = output / 'experiment_configuration.json'
    if not path.exists():
        atomic_json(path, frozen)
    # Derivatives of the authoritative frozen JSON, safe to regenerate unchanged.
    atomic_csv(output / 'sample_manifest.csv', frozen['sample'])
    temp = output / 'protocol.md.tmp'
    temp.write_text(PROTOCOL_TEXT, encoding='utf-8')
    os.replace(temp, output / 'protocol.md')


def verify_inputs_unchanged(evidence):
    for key, path in evidence['paths'].items():
        require(digest(path) == evidence['hashes'][key], f'Input changed during experiment: {path}')


def verify_baseline(item, states, metadata, originals):
    selected = states[BASELINE_THRESHOLD]['accepted']
    if item['baseline_status'] != 'PASS':
        return (len(selected) < 64 and metadata['scan_exhausted'],
                'Expected exhausted candidates and fewer than 64 at 0.80')
    if len(selected) != 64:
        return False, 'Original PASS did not reach 64 at 0.80'
    originals = sorted(originals, key=lambda r: int(r['tile_index']))
    for new, old in zip(selected, originals):
        for key in ('x_level0', 'y_level0'):
            if int(new[key]) != int(old[key]):
                return False, f'Coordinate mismatch at 0.80: {key}'
        for key in ('approx_tissue_fraction', 'final_tissue_fraction', 'background_fraction', 'white_fraction',
                    'near_black_fraction', 'gradient_mean', 'red_excess', 'cyan_excess', 'saturation_mean'):
            if not math.isclose(float(new[key]), float(old[key]), rel_tol=0, abs_tol=1e-9):
                return False, f'Metric mismatch at 0.80: {key}'
    return True, 'Original 64 ordered coordinates and QC metrics reproduced at 0.80'


def check_result(result, item, signature):
    require(result['run_signature'] == signature and result['patient'] == item,
            'Checkpoint experiment/patient mismatch')
    require(result['status'] == 'COMPLETE' and result['control_ok'], 'Invalid checkpoint status/control')
    states = result['states']
    require(set(states) == {f'{t:.2f}' for t in THRESHOLDS}, 'Checkpoint threshold set differs')
    previous_count = 65
    for key in (f'{t:.2f}' for t in THRESHOLDS):
        s = states[key]
        n = len(s['accepted'])
        require(n <= 64 and n <= previous_count, 'Threshold monotonicity/cap violated')
        previous_count = n
        require(s['threshold'] == float(key), 'Checkpoint threshold value differs')
        require(s['checked'] == n + sum(s[k] for k in ('rejected_low_tissue', 'rejected_background',
                'rejected_white', 'rejected_artifact')), 'Checkpoint counters do not reconcile')
        coords = {(r['x_level0'], r['y_level0']) for r in s['accepted']}
        require(len(coords) == n, 'Duplicate selected coordinates')
        if n == 64:
            require(s['checked_to_64'] == s['checked'] >= 64, 'Invalid full-set stopping count')
        else:
            require(s['checked_to_64'] is None and result['metadata']['scan_exhausted'],
                    'Incomplete scan treated as insufficient tissue')
            require(s['checked'] == result['metadata']['candidates_total'], 'Failed threshold did not exhaust pool')
        for r in s['accepted']:
            require(float(key) <= r['final_tissue_fraction'] <= 1 and 0 <= r['background_fraction'] <= .1
                    and 0 <= r['white_fraction'] <= .1 and r['artifact_evaluated'] and not r['artifact_reject'],
                    'Selected patch violates QC')
    require(result['metadata']['candidates_checked'] <= result['metadata']['candidates_total'], 'Invalid scan size')


def read_checkpoint(patient_dir, item, signature):
    path = patient_dir / 'result.json'
    if not path.exists():
        return None
    result = json.loads(path.read_text(encoding='utf-8'))
    require(result['run_signature'] == signature and result['patient'] == item, f'Foreign checkpoint: {path}')
    if result['status'] == 'CONTROL_MISMATCH':
        raise RuntimeError(f'Baseline mismatch recorded for {item["case_submitter_id"]}; review before continuing')
    check_result(result, item, signature)
    evidence = patient_dir / 'candidates.csv.gz'
    require(evidence.is_file() and digest(evidence) == result['candidate_log_sha256'],
            f'Candidate evidence missing/changed: {evidence}')
    return result


def wilson_interval(successes, n):
    if n == 0:
        return None, None
    z = 1.959963984540054
    p = successes / n
    denominator = 1 + z*z/n
    center = (p + z*z/(2*n)) / denominator
    half = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / denominator
    return max(0., center-half), min(1., center+half)


def boot_mean_ci(values, labels, key):
    """Percentile CI; class-stratified patient resampling, never resample patches."""
    require(len(values) == len(labels), 'Bootstrap labels do not align')
    if len(values) < 2:
        return None, None, 'n_lt_2'
    if any(labels.count(label) < 2 for label in set(labels)):
        return None, None, 'stratum_n_lt_2'
    if all(v == values[0] for v in values):
        return values[0], values[0], 'constant_empirical_distribution'
    rng = np.random.default_rng(deterministic_seed('PAIRED60_BOOTSTRAP_V1|' + key))
    sums = np.zeros(BOOTSTRAP_REPLICATES, dtype=np.float64)
    for label in sorted(set(labels)):
        a = np.array([v for v, lab in zip(values, labels) if lab == label], dtype=np.float64)
        indices = rng.integers(0, len(a), size=(BOOTSTRAP_REPLICATES, len(a)))
        sums += a[indices].sum(axis=1)
    means = sums / len(values)
    lo, hi = np.quantile(means, [.025, .975], method='linear')
    return float(lo), float(hi), 'stratified_patient_percentile_bootstrap'


def mean_or_none(values):
    return sum(values) / len(values) if values else None


def patch_means(state):
    return {key: mean_or_none([r[key] for r in state['accepted']]) for key in
            ('final_tissue_fraction', 'background_fraction', 'white_fraction')}


def pair_metrics(baseline, target):
    """Never compare a full set against a short set as though sizes were equal."""
    if len(baseline['accepted']) != 64 or len(target['accepted']) != 64:
        return None
    a = {(r['x_level0'], r['y_level0']) for r in baseline['accepted']}
    b = {(r['x_level0'], r['y_level0']) for r in target['accepted']}
    require(len(a) == len(b) == 64, 'Duplicate coordinates in pair')
    intersection = len(a & b)
    am, bm = patch_means(baseline), patch_means(target)
    return {'overlap_n': intersection, 'replacement_fraction': 1 - intersection / 64,
            'jaccard': intersection / len(a | b),
            'delta_mean_tissue_fraction': bm['final_tissue_fraction'] - am['final_tissue_fraction'],
            'delta_mean_background_fraction': bm['background_fraction'] - am['background_fraction'],
            'delta_mean_white_fraction': bm['white_fraction'] - am['white_fraction'],
            'delta_checked_to_64': target['checked_to_64'] - baseline['checked_to_64'],
            'checked_to_64_ratio': target['checked_to_64'] / baseline['checked_to_64']}


PAIR_METRICS = ('replacement_fraction', 'jaccard', 'delta_mean_tissue_fraction',
                'delta_mean_background_fraction', 'delta_mean_white_fraction',
                'delta_checked_to_64', 'checked_to_64_ratio')
SELECTED_COLUMNS = ['case_submitter_id', 'class_label', 'file_id', 'threshold', 'selection_index',
                    'candidate_checked_index', 'x_level0', 'y_level0', 'approx_tissue_fraction',
                    'final_tissue_fraction', 'background_fraction', 'white_fraction', 'artifact_evaluated',
                    'artifact_reject', 'artifact_reasons', 'near_black_fraction', 'gradient_mean',
                    'red_excess', 'cyan_excess', 'saturation_mean']


def build_tables(plan, results, errors, final):
    patients, pairs = [], []
    for item in plan:
        case = item['case_submitter_id']
        result = results.get(case)
        absent_status = errors[case]['kind'] if case in errors else 'NOT_PROCESSED'
        metadata = result['metadata'] if result else {}
        for t in THRESHOLDS:
            key = f'{t:.2f}'
            s = result['states'][key] if result else None
            means = patch_means(s) if s else {}
            n = len(s['accepted']) if s else None
            patients.append({'case_submitter_id': case, 'class_label': item['class_label'],
                'tss_code': item['tss_code'], 'baseline_status': item['baseline_status'],
                'prior_stage1_role': item['prior_stage1_role'], 'file_id': item['file_id'], 'threshold': t,
                'status': ('FULL_64' if n == 64 else 'INSUFFICIENT_PATCHES') if s else absent_status,
                'selected_tiles': n, 'reached_64': n == 64 if s else None,
                'checked': s['checked'] if s else None, 'checked_to_64': s['checked_to_64'] if s else None,
                'candidates_total': metadata.get('candidates_total'),
                'shared_scan_checked': metadata.get('candidates_checked'),
                'shared_scan_seconds': metadata.get('scan_seconds'),
                'scan_exhausted': metadata.get('scan_exhausted'),
                'control_080_ok': result['control_ok'] if result else None,
                'native_mpp_x': metadata.get('mpp_x'), 'native_mpp_y': metadata.get('mpp_y'),
                'objective': metadata.get('objective'),
                'mean_tissue_fraction_selected': means.get('final_tissue_fraction'),
                'mean_background_fraction_selected': means.get('background_fraction'),
                'mean_white_fraction_selected': means.get('white_fraction'),
                **{k: s[k] if s else None for k in ('rejected_low_tissue', 'rejected_background',
                                                   'rejected_white', 'rejected_artifact')}})
            if key == BASELINE_THRESHOLD:
                continue
            b = result['states'][BASELINE_THRESHOLD] if result else None
            bc, tc = (len(b['accepted']) == 64, n == 64) if result else (None, None)
            category = ('BOTH_COMPLETE' if bc and tc else 'BASELINE_ONLY' if bc else
                        'TARGET_ONLY' if tc else 'NEITHER_COMPLETE') if result else absent_status
            metrics = pair_metrics(b, s) if result else None
            pairs.append({'case_submitter_id': case, 'class_label': item['class_label'],
                'tss_code': item['tss_code'], 'baseline_threshold': .80, 'target_threshold': t,
                'primary_contrast': key == PRIMARY_THRESHOLD, 'pair_status': category,
                'baseline_selected_n': len(b['accepted']) if b else None, 'target_selected_n': n,
                **{k: metrics[k] if metrics else None for k in ('overlap_n',) + PAIR_METRICS}})
    threshold_summary, effects = [], []
    for group in ('ALL', 'LUAD', 'LUSC'):
        planned = [i for i in plan if group == 'ALL' or i['class_label'] == group]
        for t in THRESHOLDS:
            subset = [r for r in patients if r['threshold'] == t and (group == 'ALL' or r['class_label'] == group)]
            completed = [r for r in subset if r['status'] in ('FULL_64', 'INSUFFICIENT_PATCHES')]
            full = [r for r in subset if r['status'] == 'FULL_64']
            lo, hi = wilson_interval(len(full), len(planned)) if final else (None, None)
            row = {'group': group, 'threshold': t, 'final_analysis': final, 'n_planned': len(planned),
                'n_processed': len(completed), 'n_full_64': len(full),
                'n_insufficient': len(completed)-len(full),
                'n_technical_error': sum(r['status'] == 'TECHNICAL_ERROR' for r in subset),
                'n_control_mismatch': sum(r['status'] == 'CONTROL_MISMATCH' for r in subset),
                'n_pending': sum(r['status'] == 'NOT_PROCESSED' for r in subset),
                'full_set_rate_final': len(full)/len(planned) if final else None,
                'rate_ci95_low': lo, 'rate_ci95_high': hi,
                'rate_ci_method': 'Wilson' if final else 'pending_complete_sample'}
            for metric in ('mean_tissue_fraction_selected', 'mean_background_fraction_selected',
                           'mean_white_fraction_selected', 'checked_to_64'):
                row[metric + '_fullsets_mean_final'] = mean_or_none([r[metric] for r in full]) if final else None
            threshold_summary.append(row)
        for t in THRESHOLDS:
            if f'{t:.2f}' == BASELINE_THRESHOLD:
                continue
            subset = [r for r in pairs if r['target_threshold'] == t and (group == 'ALL' or r['class_label'] == group)]
            eligible = [r for r in subset if r['pair_status'] == 'BOTH_COMPLETE']
            for metric in PAIR_METRICS:
                values = [r[metric] for r in eligible]
                labels = [r['class_label'] for r in eligible]
                lo, hi, method = boot_mean_ci(values, labels, f'{group}|{t:.2f}|{metric}') if final else (None, None, 'pending_complete_sample')
                effects.append({'group': group, 'baseline_threshold': .8, 'target_threshold': t,
                    'metric': metric, 'primary_endpoint': group == 'ALL' and f'{t:.2f}' == PRIMARY_THRESHOLD
                    and metric == 'replacement_fraction', 'final_analysis': final,
                    'n_planned': len(planned), 'n_both_complete': len(eligible),
                    'n_baseline_only': sum(r['pair_status'] == 'BASELINE_ONLY' for r in subset),
                    'n_target_only': sum(r['pair_status'] == 'TARGET_ONLY' for r in subset),
                    'n_neither_complete': sum(r['pair_status'] == 'NEITHER_COMPLETE' for r in subset),
                    'n_missing': sum(r['pair_status'] in ('NOT_PROCESSED', 'TECHNICAL_ERROR', 'CONTROL_MISMATCH') for r in subset),
                    'paired_mean_final': mean_or_none(values) if final else None,
                    'ci95_low': lo, 'ci95_high': hi, 'ci_method': method})
    return patients, pairs, threshold_summary, effects


def publish_summary(output, plan, results, errors, integrity_verified):
    """Invalidate first, then atomically update tables; publish final status last."""
    complete = len(results) == len(plan) and not errors and integrity_verified
    status = {'complete': False, 'state': 'UPDATING_SUMMARIES', 'expected_patients': len(plan),
              'completed_patients': len(results), 'input_integrity_verified': integrity_verified,
              'errors': errors, 'updated_utc': utc_now()}
    atomic_json(output / 'run_status.json', status)
    tables = build_tables(plan, results, errors, complete)
    for name, rows in zip(('patient_threshold_summary.csv', 'paired_comparisons.csv',
                          'threshold_summary.csv', 'paired_effect_summary.csv'), tables):
        atomic_csv(output / name, rows)
    path = output / 'selected_tiles.csv.gz'
    tmp = output / 'selected_tiles.partial.csv.gz'
    with gzip.open(tmp, 'wt', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=SELECTED_COLUMNS)
        writer.writeheader()
        for item in plan:
            case = item['case_submitter_id']
            if case not in results:
                continue
            for key, state in results[case]['states'].items():
                for j, record in enumerate(state['accepted'], 1):
                    writer.writerow({'case_submitter_id': case, 'class_label': item['class_label'],
                                     'file_id': item['file_id'], 'threshold': float(key), 'selection_index': j,
                                     **record})
    os.replace(tmp, path)
    status.update({'complete': complete, 'state': 'COMPLETE' if complete else 'PARTIAL',
                   'controls_passed': sum(r['control_ok'] for r in results.values()),
                   'completed_by_class': {cl: sum(r['patient']['class_label'] == cl for r in results.values())
                                          for cl in ('LUAD', 'LUSC')},
                   'pending_patients': [i['case_submitter_id'] for i in plan
                                        if i['case_submitter_id'] not in results and i['case_submitter_id'] not in errors],
                   'primary_contrast': '0.90 vs 0.80',
                   'primary_metric': 'mean patient-level replacement_fraction among both-complete pairs',
                   'note': 'Final means/rates/intervals are blank until all 60 valid results are available.',
                   'updated_utc': utc_now()})
    atomic_json(output / 'run_status.json', status)
    return complete


def load_results(output, plan, signature):
    results, errors = {}, {}
    for item in plan:
        case = item['case_submitter_id']
        patient_dir = output / 'patients' / case
        result = read_checkpoint(patient_dir, item, signature)
        if result:
            results[case] = result
        else:
            path = patient_dir / 'last_error.json'
            if path.exists():
                err = json.loads(path.read_text(encoding='utf-8'))
                require(err['run_signature'] == signature, f'Foreign error checkpoint: {path}')
                errors[case] = err
    return results, errors


def clean_owned_download(output, item):
    path = output / 'downloads' / item['file_id']
    # UUID was validated at preflight; never delete a local --wsi-root file.
    if path.is_dir():
        require(path.resolve().parent == (output / 'downloads').resolve(), 'Unsafe cache directory')
        shutil.rmtree(path)


def execute(args, evidence, plan, frozen):
    output = args.root / 'results' / EXPERIMENT
    signature = run_signature(frozen)
    with run_lock(output):
        verify_existing_freeze(output, frozen)
        if args.summarize_only:
            require((output / 'experiment_configuration.json').is_file(), 'No prepared experiment to summarize')
        save_freeze(output, frozen)
        atomic_json(output / 'run_status.json', {'complete': False, 'state': 'VALIDATING_CHECKPOINTS',
                                               'updated_utc': utc_now()})
        results, errors = {}, {}
        try:
            results, errors = load_results(output, plan, signature)
            if args.prepare or args.summarize_only:
                verify_inputs_unchanged(evidence)
                complete = publish_summary(output, plan, results, errors, True)
                print(f'{"SUMMARY COMPLETE" if complete else "PREPARED / PARTIAL"}: {output}')
                print('No WSI downloaded or scanned by this command.')
                return 0
            publish_summary(output, plan, results, errors, False)
            processed = 0
            for item in plan:
                case = item['case_submitter_id']
                if case in results:
                    print(f'SKIP {case}: verified completed checkpoint', flush=True)
                    if not args.keep_wsi:
                        clean_owned_download(output, item)
                    continue
                if args.limit is not None and processed >= args.limit:
                    break
                patient_dir = output / 'patients' / case
                patient_dir.mkdir(parents=True, exist_ok=True)
                print(f'PROCESS {item["execution_order"]}/60 {case} ({item["class_label"]}, '
                      f'baseline {item["baseline_status"]})', flush=True)
                try:
                    path, owned = obtain_wsi(item, args, output)
                    states, metadata, loghash = scan_wsi(path, item, patient_dir)
                    ok, detail = verify_baseline(item, states, metadata, evidence['tiles'].get(case, []))
                    result = {'run_signature': signature, 'patient': item, 'states': states,
                              'metadata': metadata, 'candidate_log_sha256': loghash,
                              'control_ok': ok, 'control_detail': detail, 'finished_utc': utc_now(),
                              'status': 'COMPLETE' if ok else 'CONTROL_MISMATCH'}
                    if ok:
                        check_result(result, item, signature)
                    atomic_json(patient_dir / 'result.json', result)
                    if not ok:
                        errors[case] = {'kind': 'CONTROL_MISMATCH', 'error': detail,
                                        'run_signature': signature, 'time': utc_now()}
                        atomic_json(patient_dir / 'last_error.json', errors[case])
                        publish_summary(output, plan, results, errors, False)
                        print(f'CONTROL MISMATCH: {case}: {detail}. Experiment stopped.', flush=True)
                        return 2
                    results[case] = result
                    errors.pop(case, None)
                    processed += 1
                    print('  COMPLETE, baseline reproduced: ' + ', '.join(
                        f'{key}={len(state["accepted"])}' for key, state in states.items()), flush=True)
                    publish_summary(output, plan, results, errors, False)
                    if owned and not args.keep_wsi:
                        clean_owned_download(output, item)
                except (Exception, KeyboardInterrupt) as exc:
                    if case not in results:
                        errors[case] = {'kind': 'TECHNICAL_ERROR', 'type': type(exc).__name__,
                                        'error': str(exc) or 'Interrupted', 'run_signature': signature,
                                        'time': utc_now(), 'note': 'Not a QC exclusion; rerun same patient.'}
                        atomic_json(patient_dir / 'last_error.json', errors[case])
                    publish_summary(output, plan, results, errors, False)
                    raise
            verify_inputs_unchanged(evidence)
            complete = publish_summary(output, plan, results, errors, True)
            print(f'Experiment {"COMPLETE" if complete else "PARTIAL (rerun --run to continue)"}: {output}')
            if complete:
                print('Primary result: paired_effect_summary.csv, primary_endpoint=True. '
                      'Review attrition counts and eligible n alongside the interval.')
            return 0
        except (Exception, KeyboardInterrupt) as exc:
            atomic_json(output / 'last_run_error.json', {'run_signature': signature,
                'error': str(exc) or 'Interrupted', 'type': type(exc).__name__, 'time': utc_now()})
            atomic_json(output / 'run_status.json', {'complete': False, 'state': 'STOPPED',
                'known_completed_patients': len(results), 'expected_patients': 60,
                'error': str(exc) or 'Interrupted', 'updated_utc': utc_now(),
                'note': 'Do not use final estimates until a successful resume/summarize validates the experiment.'})
            raise


def print_preflight(plan, versions, args):
    total = sum(i['file_size_bytes'] for i in plan) / 1024**3
    peak = max(i['file_size_bytes'] for i in plan) / 1024**3
    print('PREFLIGHT PASS: original hashes, 950-patient cohort, 59456 original patch records')
    print(f'Fixed sample: 30 LUAD + 30 LUSC; seed {SAMPLE_SEED}; SHA256 ranking; no replacements')
    print('Thresholds: 0.70 / 0.75 / 0.80 / 0.85 / 0.90; 64 patches; fixed DX1; no fallback')
    print('Primary: replacement fraction 0.90 versus 0.80 among BOTH-complete patient pairs')
    for item in plan:
        print(f'  {item["execution_order"]:02d} {item["case_submitter_id"]} {item["class_label"]} '
              f'{item["baseline_status"]} {item["file_size_bytes"]/1024**3:.3f} GiB '
              f'prior_stage1={item["prior_stage1_role"]}')
    print('Baseline PASS / excluded:', sum(i['baseline_status']=='PASS' for i in plan), '/',
          sum(i['baseline_status']!='PASS' for i in plan))
    print(f'Total WSI download if all absent: {total:.2f} GiB; largest WSI: {peak:.2f} GiB')
    print(f'Without --keep-wsi: at least largest WSI + 8 GiB buffer ({peak+8:.2f} GiB), '
          'plus accumulating CSV logs. With --keep-wsi budget the total download plus buffer/logs.')
    print('Dependencies:', json.dumps(versions))
    print('Results:', args.root / 'results' / EXPERIMENT)


def self_test():
    """Offline scientific invariants; no project, WSI, network or OpenSlide needed."""
    global np
    import numpy as np
    import random
    rng = random.Random(2601)
    records = []
    for i in range(6000):
        records.append({'x_level0': i, 'y_level0': 0, 'final_tissue_fraction': rng.choice(
            [.699, .70, .749, .75, .799, .80, .849, .85, .899, .90, .99]),
            'background_fraction': rng.choice([0., .1, .1001]),
            'white_fraction': rng.choice([0., .1, .1001]),
            'artifact_reject': rng.choice([False, False, True])})
    states = new_states()
    for record in records:
        update_states(states, record)
    # Independent sequential reference for each threshold, including exact boundaries.
    for key, state in states.items():
        accepted, checked = [], 0
        rejects = dict(low_tissue=0, background=0, white=0, artifact=0)
        for row in records:
            checked += 1
            if row['final_tissue_fraction'] < float(key):
                rejects['low_tissue'] += 1
            elif row['background_fraction'] > .1:
                rejects['background'] += 1
            elif row['white_fraction'] > .1:
                rejects['white'] += 1
            elif row['artifact_reject']:
                rejects['artifact'] += 1
            else:
                accepted.append(row)
            if len(accepted) == 64:
                break
        require(state['accepted'] == accepted and state['checked'] == checked, 'Independent threshold oracle failed')
        require(all(state['rejected_'+k] == v for k,v in rejects.items()), 'Rejection counters failed')
    def state(start, tissue, checked):
        return {'accepted': [{'x_level0': i, 'y_level0': 0, 'final_tissue_fraction': tissue,
                            'background_fraction': .02, 'white_fraction': .01}
                            for i in range(start,start+64)], 'checked_to_64': checked}
    a, b = state(0,.85,80), state(32,.95,160)
    metrics = pair_metrics(a,b)
    require(metrics['overlap_n']==32 and metrics['replacement_fraction']==.5, 'Replacement calculation failed')
    require(math.isclose(metrics['jaccard'],1/3) and metrics['checked_to_64_ratio']==2, 'Pair metrics failed')
    require(math.isclose(metrics['delta_mean_tissue_fraction'],.1), 'Paired quality delta failed')
    require(pair_metrics(a, {'accepted':[]}) is None, 'Short set improperly compared to full set')
    require(pair_metrics(a,a)['replacement_fraction']==0, 'Identical-set comparison failed')
    low,high = wilson_interval(0,30)
    require(low==0 and .11 < high < .12, 'Wilson zero-success interval failed')
    low,high = wilson_interval(30,30)
    require(.88 < low < .89 and math.isclose(high,1), 'Wilson full-success interval failed')
    require(boot_mean_ci([.5],['LUAD'],'one')[:2]==(None,None), 'n=1 interval must be unavailable')
    v=boot_mean_ci([0.,.25,.75,1.],['LUAD','LUAD','LUSC','LUSC'],'deterministic')
    require(v==boot_mean_ci([0.,.25,.75,1.],['LUAD','LUAD','LUSC','LUSC'],'deterministic'), 'Bootstrap not deterministic')
    require(boot_mean_ci([.5,.5],['LUAD','LUAD'],'constant')[:2]==(.5,.5), 'Constant bootstrap failed')
    require(boot_mean_ci([0.,1.],['LUAD','LUSC'],'singletons')[:2]==(None,None), 'Singleton-stratum CI guard failed')
    require(sample_key('TCGA-TEST-0001')==sample_key('TCGA-TEST-0001'), 'Sample key not deterministic')
    with tempfile.TemporaryDirectory() as td:
        output=Path(td)
        planned=[{'case_submitter_id':'TCGA-TEST-0001','class_label':'LUAD','tss_code':'TEST',
                  'baseline_status':'PASS','prior_stage1_role':'not_in_stage1','file_id':'test'}]
        require(not publish_summary(output,planned,{}, {},False), 'Empty run incorrectly complete')
        for row in read_csv(output/'paired_effect_summary.csv'):
            require(row['paired_mean_final']=='' and row['ci95_low']=='', 'Partial run emitted final estimate')
        with run_lock(output):
            try:
                with run_lock(output):
                    pass
            except RuntimeError:
                pass
            else:
                raise RuntimeError('Concurrent-run lock failed')
        require(not (output/'run.lock').exists(), 'Lock not released')
    print('SELF-TEST PASS: five-threshold oracle, boundary inclusion, counters, paired metrics, '
          'missing pairs, Wilson/bootstrap intervals, incomplete-summary guard and run lock')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--preflight-only', action='store_true', help='Read-only validation (default); no downloads/writes')
    mode.add_argument('--prepare', action='store_true', help='Freeze 60 patients/protocol; no WSI download')
    mode.add_argument('--run', action='store_true', help='Execute or resume the frozen experiment')
    mode.add_argument('--summarize-only', action='store_true', help='Validate checkpoints and rebuild tables, no WSI')
    mode.add_argument('--self-test', action='store_true', help='Offline logic/statistics tests, only numpy needed')
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--gdc-client', type=Path, help='Default: PROJECT/tools/gdc-client.exe')
    parser.add_argument('--wsi-root', type=Path, help='Read-only raw files at UUID/name or name; missing files downloaded')
    parser.add_argument('--keep-wsi', action='store_true', help='Keep this experiment\'s downloaded raw files')
    parser.add_argument('--limit', type=int, help='Maximum NEW patients this invocation; completed ones are skipped')
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    require(args.limit is None or (args.run and args.limit>0), '--limit requires --run and a positive integer')
    args.root=args.root.resolve()
    args.gdc_client=(args.gdc_client or args.root/'tools/gdc-client.exe').resolve()
    if args.wsi_root:
        args.wsi_root=args.wsi_root.resolve()
        require(args.wsi_root.is_dir(), f'WSI folder missing: {args.wsi_root}')
    evidence=validate_inputs(args.root)
    plan=make_plan(evidence)
    versions=load_dependencies()
    if not args.wsi_root and not args.summarize_only:
        require(args.gdc_client.is_file(), f'GDC client missing: {args.gdc_client}')
    frozen=freeze_payload(evidence,plan,versions)
    verify_existing_freeze(args.root/'results'/EXPERIMENT,frozen)
    print_preflight(plan,versions,args)
    if not (args.run or args.prepare or args.summarize_only):
        print('Read-only preflight complete. Use --prepare to save protocol/sample, then --run --limit 1.')
        return 0
    return execute(args,evidence,plan,frozen)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print('\nInterrupted. Completed patients are saved; rerun --run to resume.',file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f'ERROR [{type(exc).__name__}]: {exc}',file=sys.stderr)
        raise SystemExit(1)
