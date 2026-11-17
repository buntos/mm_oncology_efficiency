#!/usr/bin/env python3
"""Independent tissue-threshold audit for the frozen 950-patient TCGA cohort.

Save in PROJECT/scripts. Python 3.12, the EXISTING project venv and GDC client.
  python scripts/25_audit_excluded_tissue_thresholds.py --preflight-only
  python scripts/25_audit_excluded_tissue_thresholds.py --run --limit 1
  python scripts/25_audit_excluded_tissue_thresholds.py --run
  python scripts/25_audit_excluded_tissue_thresholds.py --self-test

Default action is preflight (no downloads, no writes). The run audits all 21
EXCLUDED_WSI_QC patients plus two positive controls (smallest PASS WSI per class,
ties by case ID). Controls verify the original 64 coordinates AND QC metrics at
0.80; they are not a representative research sample. Each current audit patient
has exactly one diagnostic WSI; preflight refuses another candidate structure.

Only final tissue threshold changes: 0.70 / 0.75 / 0.80. Candidate mask threshold,
HSV conversion, artifact rules, level-0 grid and deterministic order stay fixed.
The 929 original PASS cases certify RETENTION at lower thresholds, not their
newly selected patch composition. No claim about diagnostic accuracy is made.

Results: PROJECT/results/tissue_threshold_rescue_v1 (separate from production).
Successful patient checkpoints resume automatically. Interrupted patients restart
their scan; verified downloads are reused. Technical failures are not QC failures.
Each patient writes one JSON checkpoint and a gzip CSV of scanned candidates.
Only this audit's downloaded WSI are removed AFTER a valid checkpoint is saved;
--keep-wsi retains them. Files supplied via --wsi-root are always read-only.
No RNA download, model training, image export or production-script import occurs.

Exit codes: 0 success/preflight; 1 operational/validation failure; 2 control
mismatch; 130 interrupted. After a killed process, remove run.lock only after
confirming no other copy is running. Do not edit a script mid-run or mix versions.
"""
from __future__ import annotations

import argparse
import ast
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
from types import SimpleNamespace
from datetime import datetime, timezone

VERSION = '1.0.0'
NAMESPACE = 'production_final_v2_tissue080'
THRESHOLDS = (0.70, 0.75, 0.80)
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


def atomic_csv(path, rows):
    require(bool(rows), f'No rows for {path.name}')
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
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

# Frozen numerical functions copied from scripts 11 and 17 (AST-equivalent).
# generate_candidates is from script 18; its grid is equivalent to script 17.
# They are embedded to avoid import-time output-directory mutations in old scripts.


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


def preflight(root):
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
    controls = []
    for cl in ('LUAD', 'LUSC'):
        eligible = [case for case, r in qi.items() if r['status'] == 'PASS' and r['class_label'] == cl
                    and len(by_candidates.get(case, [])) == 1
                    and by_candidates[case][0]['file_id'] == r['selected_wsi_file_id']]
        require(eligible, f'No eligible {cl} positive control')
        controls.append(min(eligible, key=lambda case: (int(by_candidates[case][0]['file_size_bytes']), case)))
    plan = []
    for case in controls + sorted(excluded):
        rows = by_candidates.get(case, [])
        require(len(rows) == 1, f'{case}: expected exactly one diagnostic WSI; fallback needs review')
        row = rows[0]
        require(row['slide_token'] == 'DX1' and int(float(row['dx_index'])) == 1, 'Expected DX1')
        require(row['access'] == 'open', 'Only open-access WSI supported')
        require(row['file_id'] == ci[case]['wsi_file_id'], f'Stage2 WSI differs for {case}')
        require(Path(row['file_name']).name == row['file_name'] and '\\' not in row['file_name'],
                'Invalid WSI filename')
        require(len(row['md5sum']) == 32 and int(row['file_size_bytes']) > 0, 'Missing size/MD5')
        plan.append({'case_submitter_id': case, 'class_label': qi[case]['class_label'],
                     'role': 'positive_control' if case in controls else 'excluded',
                     'file_id': row['file_id'], 'file_name': row['file_name'],
                     'file_size_bytes': int(row['file_size_bytes']), 'md5sum': row['md5sum'],
                     'slide_token': row['slide_token']})
    hashes = {k: digest(p) for k, p in paths.items()}
    return {'plan': plan, 'hashes': hashes, 'paths': paths, 'tiles': by_tiles, 'qc': qi}


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


def verify_control(plan, states, originals):
    selected = states['0.80']['accepted']
    if plan['role'] == 'excluded':
        return len(selected) < 64, 'Expected fewer than 64 patches at 0.80'
    if len(selected) != 64:
        return False, 'Positive control did not reach 64 at 0.80'
    originals = sorted(originals, key=lambda r: int(r['tile_index']))
    metric_names = ('approx_tissue_fraction', 'final_tissue_fraction', 'background_fraction',
                    'white_fraction', 'near_black_fraction', 'gradient_mean', 'red_excess',
                    'cyan_excess', 'saturation_mean')
    for new, old in zip(selected, originals):
        for k in ('x_level0', 'y_level0'):
            if int(new[k]) != int(old[k]):
                return False, f'Positive control coordinate mismatch: {k}'
        for k in metric_names:
            if not math.isclose(float(new[k]), float(old[k]), rel_tol=0, abs_tol=1e-9):
                return False, f'Positive control metric mismatch: {k}'
    return True, 'All 64 original coordinates and QC metrics match at 0.80'


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
                   'decision_0.70', 'decision_0.75', 'decision_0.80']
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


def read_checkpoint(path, signature):
    if not path.exists():
        return None
    result = json.loads(path.read_text(encoding='utf-8'))
    require(result['run_signature'] == signature, f'Checkpoint from another experiment: {path}')
    if result['status'] != 'COMPLETE':
        return None
    evidence = path.parent / 'candidates.csv.gz'
    require(evidence.is_file() and digest(evidence) == result['candidate_log_sha256'],
            f'Checkpoint evidence missing or changed: {evidence}')
    require(result['control_ok'], 'Invalid control in completed checkpoint')
    return result


def summarize(output, plan, signature, qc):
    rows, per_patient = [], []
    complete_excluded = {cl: 0 for cl in ('LUAD', 'LUSC')}
    rescued = {(cl, key): 0 for cl in complete_excluded for key in ('0.70', '0.75', '0.80')}
    controls_complete = 0
    for item in plan:
        result = read_checkpoint(output / 'patients' / item['case_submitter_id'] / 'result.json', signature)
        if result is None:
            continue
        if item['role'] == 'positive_control':
            controls_complete += 1
        else:
            complete_excluded[item['class_label']] += 1
        for key, state in result['states'].items():
            n = len(state['accepted'])
            if item['role'] == 'excluded' and n == 64:
                rescued[item['class_label'], key] += 1
            per_patient.append({'case_submitter_id': item['case_submitter_id'], 'class_label': item['class_label'],
                                'role': item['role'], 'threshold': state['threshold'], 'selected_tiles': n,
                                'reached_64': n == 64, 'checked': state['checked'],
                                'checked_to_64': state['checked_to_64'], 'control_080_ok': True,
                                **{k: state[k] for k in ('rejected_low_tissue', 'rejected_background',
                                                       'rejected_white', 'rejected_artifact')}})
    all_complete = sum(complete_excluded.values()) == 21 and controls_complete == 2
    for key in ('0.70', '0.75', '0.80'):
        for cl in ('LUAD', 'LUSC'):
            n = sum(r['class_label'] == cl for r in qc.values())
            original_pass = sum(r['class_label'] == cl and r['status'] == 'PASS' for r in qc.values())
            observed = rescued[cl, key]
            rows.append({'threshold': float(key), 'class_label': cl, 'cohort_n': n,
                         'original_pass_certified': original_pass, 'excluded_retested': complete_excluded[cl],
                         'rescued_observed': observed, 'complete_experiment': all_complete,
                         'retained_lower_bound': original_pass + observed,
                         'retained_final': original_pass + observed if all_complete else '',
                         'excluded_final': n - original_pass - observed if all_complete else '',
                         'retention_rate_final': (original_pass + observed) / n if all_complete else '',
                         'evidence': '929 original PASS certified; excluded cases remeasured'})
    atomic_csv(output / 'cohort_retention_summary.csv', rows)
    if per_patient:
        atomic_csv(output / 'patient_threshold_summary.csv', per_patient)
    atomic_json(output / 'run_status.json', {'complete': all_complete, 'controls_complete': controls_complete,
                'excluded_complete': complete_excluded, 'updated_utc': utc_now(),
                'note': 'Blank final counts mean incomplete; lower bounds are not final estimates.'})
    return all_complete


def run(args, evidence, versions):
    output = args.root / 'results' / 'tissue_threshold_rescue_v1'
    output.mkdir(parents=True, exist_ok=True)
    lock = output / 'run.lock'
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise RuntimeError(f'Another run or stale lock: {lock}. Remove only when no audit is running.')
    with os.fdopen(fd, 'w') as f:
        f.write(f'pid={os.getpid()} started={utc_now()}\n')
    try:
        frozen = {'version': VERSION, 'audit_script_sha256': digest(Path(__file__)),
                  'input_sha256': evidence['hashes'], 'thresholds': THRESHOLDS,
                  'plan': evidence['plan'], 'environment': versions,
                  'retention_inference': 'Original PASS at 0.80 implies feasible 64 patches at 0.70/0.75',
                  'selection': 'First 64 accepted in unchanged deterministic order; one DX1 per audit patient'}
        # Normalize tuples to JSON lists before comparing a resumed run.
        frozen = json.loads(json.dumps(frozen))
        signature = hashlib.sha256(json.dumps(frozen, sort_keys=True).encode()).hexdigest()
        configpath = output / 'audit_configuration.json'
        if configpath.exists():
            require(json.loads(configpath.read_text(encoding='utf-8')) == frozen,
                    'Audit inputs/code/environment changed. Keep this run intact; use a reviewed new version.')
        else:
            require(not (output / 'patients').exists(), 'Patient results exist without their configuration')
            atomic_json(configpath, frozen)
        atomic_csv(output / 'audit_patient_plan.csv', evidence['plan'])
        summarize(output, evidence['plan'], signature, evidence['qc'])
        processed = 0
        for item in evidence['plan']:
            case = item['case_submitter_id']
            patient_dir = output / 'patients' / case
            resultpath = patient_dir / 'result.json'
            if read_checkpoint(resultpath, signature):
                print(f'SKIP {case}: verified completed checkpoint', flush=True)
                continue
            if args.limit is not None and processed >= args.limit:
                break
            patient_dir.mkdir(parents=True, exist_ok=True)
            print(f'PROCESS {case} ({item["class_label"]}, {item["role"]})', flush=True)
            try:
                path, owned = obtain_wsi(item, args, output)
                states, metadata, loghash = scan_wsi(path, item, patient_dir)
                ok, detail = verify_control(item, states, evidence['tiles'].get(case, []))
                result = {'run_signature': signature, 'patient': item, 'states': states,
                          'metadata': metadata, 'candidate_log_sha256': loghash,
                          'control_ok': ok, 'control_detail': detail, 'finished_utc': utc_now(),
                          'status': 'COMPLETE' if ok else 'CONTROL_MISMATCH'}
                atomic_json(resultpath, result)
                if not ok:
                    print(f'CONTROL MISMATCH: {case}: {detail}. Stopped; inspect result.json.', flush=True)
                    summarize(output, evidence['plan'], signature, evidence['qc'])
                    return 2
                processed += 1
                if owned and not args.keep_wsi:
                    # Confined to this audit's single UUID directory; never original raw data.
                    shutil.rmtree(output / 'downloads' / item['file_id'])
                print('  COMPLETE: ' + ', '.join(f'{k}={len(s["accepted"])}' for k, s in states.items()), flush=True)
                summarize(output, evidence['plan'], signature, evidence['qc'])
            except Exception as exc:
                # If cleanup fails after completion, preserve the already valid checkpoint.
                atomic_json(patient_dir / 'last_error.json', {'type': type(exc).__name__, 'error': str(exc),
                            'time': utc_now(), 'note': 'Technical failure is not a QC exclusion. Rerun to resume.'})
                raise
        for key, path in evidence['paths'].items():
            require(digest(path) == evidence['hashes'][key], f'Input changed during run: {path}')
        complete = summarize(output, evidence['plan'], signature, evidence['qc'])
        print(f'Experiment {"COMPLETE" if complete else "PARTIAL (rerun without --limit to continue)"}: {output}')
        return 0
    finally:
        lock.unlink(missing_ok=True)


def self_test():
    """Offline tests of nesting, boundary inclusion, counters, control and checkpoint guards."""
    states = new_states()
    def record(t=.8, bg=0., white=0., artifact=False, x=0):
        return {'final_tissue_fraction': t, 'background_fraction': bg, 'white_fraction': white,
                'artifact_reject': artifact, 'x_level0': x, 'y_level0': 0}
    update_states(states, record(.69))
    update_states(states, record(.70))
    update_states(states, record(.75))
    update_states(states, record(.80, .1001))
    update_states(states, record(.80, white=.1001))
    update_states(states, record(.80, artifact=True))
    for i in range(64):
        update_states(states, record(.80, .10, .10, x=i))
    require([len(s['accepted']) for s in states.values()] == [64, 64, 64], 'Cap/nesting failed')
    require([s['checked'] for s in states.values()] == [68, 69, 70], 'Independent stopping failed')
    for s in states.values():
        require(s['rejected_background'] == s['rejected_white'] == s['rejected_artifact'] == 1,
                'Sequential rejection counters failed')
    require(not verify_control({'role': 'excluded'}, states, [])[0], 'Mismatch not detected')
    empty = new_states()
    require(verify_control({'role': 'excluded'}, empty, [])[0], 'Exclusion control failed')
    metrics = dict(approx_tissue_fraction=.9, final_tissue_fraction=.9, background_fraction=.1,
                   white_fraction=.1, near_black_fraction=0., gradient_mean=1., red_excess=0.,
                   cyan_excess=0., saturation_mean=.5)
    original = [dict(metrics, tile_index=i, x_level0=i, y_level0=0) for i in range(64)]
    empty['0.80']['accepted'] = [dict(r) for r in original]
    require(verify_control({'role': 'positive_control'}, empty, original)[0], 'Positive match failed')
    empty['0.80']['accepted'][0]['x_level0'] = -1
    require(not verify_control({'role': 'positive_control'}, empty, original)[0], 'Coordinate change missed')
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / 'result.json'
        log = Path(tmp) / 'candidates.csv.gz'
        log.write_bytes(b'test evidence')
        atomic_json(p, {'run_signature': 'test', 'status': 'COMPLETE', 'control_ok': True,
                        'candidate_log_sha256': digest(log)})
        require(read_checkpoint(p, 'test') is not None, 'Resume failed')
        log.write_bytes(b'changed')
        try:
            read_checkpoint(p, 'test')
        except RuntimeError:
            pass
        else:
            raise RuntimeError('Corrupt checkpoint evidence accepted')
    print('SELF-TEST PASS: threshold boundaries, counters, caps, controls, atomic checkpoint and corruption guard')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--preflight-only', action='store_true', help='Read-only validation (default)')
    mode.add_argument('--run', action='store_true', help='Download/reuse WSI and execute/resume the audit')
    mode.add_argument('--self-test', action='store_true', help='Offline core tests, no project files needed')
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--gdc-client', type=Path, help='Default: PROJECT/tools/gdc-client.exe')
    parser.add_argument('--wsi-root', type=Path, help='Optional read-only raw WSI folder (UUID/name or name)')
    parser.add_argument('--keep-wsi', action='store_true', help='Retain only this audit\'s own WSI cache')
    parser.add_argument('--limit', type=int, help='Maximum NEW patients this invocation; resume skips completed ones')
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    require(args.limit is None or args.limit > 0, '--limit must be positive')
    args.root = args.root.resolve()
    args.gdc_client = (args.gdc_client or args.root / 'tools' / 'gdc-client.exe').resolve()
    if args.wsi_root:
        args.wsi_root = args.wsi_root.resolve()
        require(args.wsi_root.is_dir(), f'WSI folder missing: {args.wsi_root}')
    evidence = preflight(args.root)
    versions = load_dependencies()
    if not args.wsi_root:
        require(args.gdc_client.is_file(), f'Missing GDC client: {args.gdc_client}')
    plan = evidence['plan']
    print('PREFLIGHT PASS: frozen source hashes, 950 patients, 929 PASS / 21 excluded, 59456 patches')
    print('Thresholds: 0.70 / 0.75 / 0.80; 64 patches, 256x256, native level 0')
    print('Audit: 21 exclusions + 2 reproducibility controls; controls processed first')
    for item in plan:
        print(f'  {item["case_submitter_id"]} {item["class_label"]} {item["role"]} '
              f'{item["file_size_bytes"] / 1024**3:.3f} GiB')
    print(f'Total raw WSI if all downloaded: {sum(i["file_size_bytes"] for i in plan) / 1024**3:.2f} GiB')
    print('Dependencies:', json.dumps(versions))
    print('Results:', args.root / 'results' / 'tissue_threshold_rescue_v1')
    if not args.run:
        print('Read-only preflight complete. No downloads or output writes. Use --run to execute.')
        return 0
    return run(args, evidence, versions)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print('\nInterrupted. Completed patients are saved; rerun the same command to resume.', file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f'ERROR [{type(exc).__name__}]: {exc}', file=sys.stderr)
        raise SystemExit(1)
