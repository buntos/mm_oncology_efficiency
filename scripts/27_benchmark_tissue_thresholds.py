#!/usr/bin/env python3
"""WSI resource benchmark v1.0.0 (Windows/Linux, Python 3.12).

Place next to the UNMODIFIED 26_run_paired60_tissue_thresholds.py in scripts/.
Use the original research venv. No new packages. Stage 26 must be COMPLETE.

  python scripts/27_benchmark_tissue_thresholds.py --preflight-only
  python scripts/27_benchmark_tissue_thresholds.py --prepare
  python scripts/27_benchmark_tissue_thresholds.py --run --limit 1
  python scripts/27_benchmark_tissue_thresholds.py --run
  python scripts/27_benchmark_tissue_thresholds.py --summarize-only
  python scripts/27_benchmark_tissue_thresholds.py --self-test

Default: read-only preflight. Output: results/tissue_threshold_benchmark20_v1.
20 fixed patients (first 10/class from stage26), 3 thresholds, 3 repetitions.
Every measured run has a separate preceding warm-up subprocess of the SAME
threshold. No cache flushing: results are warm-cache-oriented, not cold-disk.
Each measured run uses a fresh process/OpenSlide handle. Initialization of the
Python interpreter and imports, WSI download/MD5, correctness validation, JSON
serialization and warm-up are OUTSIDE wall/CPU timing. Timing INCLUDES slide
open/close, thumbnail/mask, full candidate generation/shuffle and selection.
No candidate CSV logging, image export or model inference is timed.

Peak memory is OS process-lifetime peak RSS/working set through scan completion,
INCLUDING interpreter/imports. It is not an interval-only peak, private bytes,
VRAM or system RAM. Peak before scan is also recorded. RAM reduction is not
assumed. Native numerical thread pools are fixed to 1 before child imports.

Partial patients restart all 9 paired measurements after interruption; completed
patients are verified and skipped. No fastest-run selection, outlier removal or
replacement of QC failures. Only owned WSI downloads are removed after valid
patient checkpoint; --wsi-root is read-only. Default download cache is separate
from stage26; stage26 --keep-wsi downloads can be passed as --wsi-root.

Exit: 0 success/intentional partial, 1 error, 130 interruption. A stale run.lock
may be removed only after checking no parent/worker remains. Do not modify
scripts or environment mid-run. This post-stage26 benchmark is a separate
follow-up, not a retrospectively preregistered stage26 endpoint.
"""
from __future__ import annotations
import argparse
import ctypes
import hashlib
import importlib.util
import itertools
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

VERSION = '1.0.0'
EXPERIMENT = 'tissue_threshold_benchmark20_v1'
SOURCE26 = '26_run_paired60_tissue_thresholds.py'
SOURCE26_SHA256 = '2af99ec3c0b2d038ea3688dd688cbcb1d69e8d77a06c3b570c199c4eb3972858'
THRESHOLDS = ('0.70', '0.80', '0.90')
REPEATS = 3
PER_CLASS = 10
THREAD_ENV = {k: '1' for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
    'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS')}
THREAD_ENV.update(OMP_DYNAMIC='FALSE', MKL_DYNAMIC='FALSE')
COUNTERS = ('checked', 'checked_to_64', 'rejected_low_tissue', 'rejected_background',
            'rejected_white', 'rejected_artifact')
METRICS = ('wall_seconds', 'cpu_seconds', 'peak_rss_mib')
RECORD_FLOATS = ('approx_tissue_fraction', 'final_tissue_fraction',
    'background_fraction', 'white_fraction', 'near_black_fraction', 'gradient_mean',
    'red_excess', 'cyan_excess', 'saturation_mean')
PROTOCOL = '''# Resource benchmark: frozen follow-up protocol v1.0.0

## Scope and selection
Post-stage26 follow-up, exploratory, not external validation. First 10 patients
per class by previously frozen class_sample_rank from the stage26 30+30 draw.
No replacement or selection by observed effect. Includes original QC failures.
The resource-constrained sample size is not a power calculation.
Thresholds 0.70/0.80/0.90, 64 patches, same DX1, native level0 256x256,
original candidate mask/order and all other QC unchanged. Tissue is not tumor.
Existing stage26 primary endpoint remains replacement_fraction 0.90 vs 0.80.

## Benchmark endpoints
Main benchmark endpoint: mean patient-level wall-time ratio 0.90/0.80 among
patients reaching 64 at BOTH thresholds. Prespecified additional contrast:
0.70/0.80. First take the median of three repeats at each threshold per patient,
then form patient ratios; do not pool repeat observations as independent data.
Secondary: process CPU time and OS peak resident memory ratio/difference.
Also report all-patient attempt costs including candidate exhaustion and
counts of both complete, baseline only, target only and neither complete.
An unsuccessful attempt is not time-to-64 and is not a technical error.
Overall percentile 95% CI: 10000 class-stratified patient bootstrap resamples,
fixed seed. No p-value search or arbitrary outlier removal. CI is conditional
on sample and benchmark environment; three repeats describe runtime variability.

## Execution and cache
Three repeats. Per patient, a cyclic Latin schedule gives each threshold each
position once. Base permutation cycles over six permutations across patients.
Each threshold/round: one untimed warm-up in a fresh subprocess, followed by
one measured fresh subprocess, same WSI/threshold. Warm-up is validated too.
OS caches are not flushed; actual OS residency cannot be guaranteed. This is
warm-cache-oriented local processing, not cold-start or network latency.
One worker at a time, native numerical thread pools set to one before imports.
Use same machine, disk, power plan and low background load throughout.

## Measurement boundary
perf_counter: from before OpenSlide open through close. process_time: same
interval, all CPU threads of this process. Includes thumbnail, tissue mask,
candidate list/order, region decoding, QC and first-64 selection/exhaustion.
Excludes startup/imports, downloads/MD5, warm-up, reference comparisons and
JSON/log output. No candidate CSV writes, patch saving, RNA or model inference.
Native artifact metrics run only after tissue/background/white pass for the
single current threshold, preserving original sequential filtering semantics.
Compare ordered coordinates, accepted QC metrics, stopping/rejection counts,
candidate pool size and seed with stage26 for EVERY warm-up and measurement.

## Memory interpretation
Windows GetProcessMemoryInfo PeakWorkingSetSize; Linux ru_maxrss converted
from KiB (macOS bytes). Process-lifetime peak through scan completion includes
Python and imports; also retain pre-scan peak. Fresh processes exclude prior
warm-up allocations. No claims about interval peak, energy or GPU memory.

## Recovery and provenance
Code/dependency/machine identity, source inputs, stage26 configuration and
selected patient checkpoint/candidate evidence hashes are frozen. Technical
failures stop and remain distinct from QC. Complete patients resume unchanged;
partial patients restart ALL rounds, never selecting fastest repeats. Writes
only to benchmark output; deletes only owned downloaded WSI after checkpoint.
Final aggregate estimates are suppressed until all 20 patients validate.
'''


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(b)
    return h.hexdigest()


def load_core():
    path = Path(__file__).resolve().with_name(SOURCE26)
    require(path.is_file(), f'Missing {path}; put script27 beside the original script26')
    require(sha(path) == SOURCE26_SHA256,
            'Script26 hash differs from the released version. Do not bypass: review file/version.')
    spec = importlib.util.spec_from_file_location('frozen_core26', path)
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)  # stage26 has no import-time filesystem mutations
    return core


def peak_rss_bytes():
    if sys.platform == 'win32':
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD)] + [
                (k, ctypes.c_size_t) for k in ('PeakWorkingSetSize', 'WorkingSetSize',
                'QuotaPeakPagedPoolUsage', 'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage',
                'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage')]
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        psapi = ctypes.WinDLL('psapi', use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        fn = psapi.GetProcessMemoryInfo
        fn.argtypes = (wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD)
        fn.restype = wintypes.BOOL
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        if not fn(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        return int(counters.PeakWorkingSetSize)
    import resource
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == 'darwin' else value * 1024)


def schedule(index):
    base = list(itertools.permutations(THRESHOLDS))[index % 6]
    return [{'repeat': r + 1, 'position': j + 1, 'threshold': base[(j + r) % 3]}
            for r in range(REPEATS) for j in range(3)]


def scan_one(core, path, item, threshold):
    """Independent single-threshold scan, no file logging in the timed interval."""
    key = f'{float(threshold):.2f}'
    states = {key: core.new_states()[key]}
    state = states[key]
    seed = core.deterministic_seed(item['case_submitter_id'] + '|' + item['file_id'])
    with core.openslide.OpenSlide(str(path)) as slide:
        thumbnail = slide.get_thumbnail((core.THUMBNAIL_MAX_SIZE, core.THUMBNAIL_MAX_SIZE)).convert('RGB')
        try:
            mask, raw_otsu, saturation = core.create_tissue_mask(thumbnail)
            candidates = core.generate_candidates(slide, mask)
        finally:
            thumbnail.close()
        order = core.np.arange(len(candidates))
        core.np.random.default_rng(seed).shuffle(order)
        metadata = dict(candidates_total=len(candidates), deterministic_seed=str(seed),
                        saturation_threshold=float(saturation), raw_otsu_saturation=float(raw_otsu))
        for ix in order:
            x, y, approximate = candidates[ix]
            with slide.read_region((int(x), int(y)), 0, (core.TILE_SIZE, core.TILE_SIZE)) as region:
                tile = region.convert('RGB')
            try:
                tissue, background, white = core.evaluate_tile(tile, saturation)
                metrics = {k: None for k in RECORD_FLOATS[4:]}
                evaluated = (tissue >= float(key) and background <= core.MAX_BACKGROUND_FRACTION
                             and white <= core.MAX_WHITE_FRACTION)
                rejected, reasons = False, []
                if evaluated:
                    metrics = core.calculate_artifact_metrics(tile)
                    rejected, reasons = core.evaluate_artifact_qc(metrics)
            finally:
                tile.close()
            record = dict(candidate_checked_index=state['checked'] + 1,
                x_level0=int(x), y_level0=int(y), approx_tissue_fraction=float(approximate),
                final_tissue_fraction=tissue, background_fraction=background, white_fraction=white,
                artifact_evaluated=evaluated, artifact_reject=rejected,
                artifact_reasons=';'.join(reasons), **metrics)
            core.update_states(states, record)
            if len(state['accepted']) == core.TARGET_TILES:
                break
        metadata['scan_exhausted'] = state['checked'] == len(candidates)
    return state, metadata


def compare_reference(state, meta, ref, threshold):
    expected = ref['states'][threshold]
    for key in COUNTERS:
        require(state[key] == expected[key], f'Stage26 counter mismatch: {threshold} {key}')
    require(meta['candidates_total'] == ref['metadata']['candidates_total'], 'Candidate pool mismatch')
    require(meta['deterministic_seed'] == ref['metadata']['deterministic_seed'], 'Seed mismatch')
    for k in ('saturation_threshold', 'raw_otsu_saturation'):
        require(math.isclose(meta[k], ref['metadata'][k], rel_tol=0, abs_tol=1e-9), f'Mask mismatch: {k}')
    require(len(state['accepted']) == len(expected['accepted']), 'Selected count mismatch')
    if len(state['accepted']) < 64:
        require(meta['scan_exhausted'], 'Incomplete scan must exhaust candidate pool')
    for a, b in zip(state['accepted'], expected['accepted']):
        for key in ('x_level0', 'y_level0', 'candidate_checked_index', 'artifact_reject', 'artifact_evaluated'):
            require(a[key] == b[key], f'Stage26 patch mismatch: {key}')
        for key in RECORD_FLOATS:
            require(math.isclose(a[key], b[key], rel_tol=0, abs_tol=1e-9), f'Stage26 metric mismatch: {key}')


def worker(jobpath):
    os.environ.update(THREAD_ENV)
    job = json.loads(Path(jobpath).read_text(encoding='utf-8'))
    require(job['script_sha256'] == sha(__file__), 'Worker script hash changed')
    core = load_core()
    versions = core.load_dependencies()
    require(versions == job['environment'], 'Worker dependency versions differ')
    refpath = Path(job['reference_path'])
    require(sha(refpath) == job['reference_sha256'], 'Stage26 reference changed')
    ref = json.loads(refpath.read_text(encoding='utf-8'))
    require(ref['patient'] == job['patient'], 'Worker patient differs from reference')
    peak_before = peak_rss_bytes()
    cpu0 = time.process_time()
    wall0 = time.perf_counter()
    state, metadata = scan_one(core, Path(job['wsi']), job['patient'], job['threshold'])
    wall = time.perf_counter() - wall0
    cpu = time.process_time() - cpu0
    peak = peak_rss_bytes()
    compare_reference(state, metadata, ref, job['threshold'])
    core.atomic_json(Path(job['result_path']), dict(
        run_signature=job['run_signature'], case_submitter_id=job['patient']['case_submitter_id'],
        threshold=job['threshold'], repeat=job['repeat'], position=job['position'], phase=job['phase'],
        control_ok=True, wsi_source_path=job['wsi'], state=state, metadata=metadata, wall_seconds=wall, cpu_seconds=cpu,
        peak_rss_mib=peak / 1024**2, pre_scan_peak_rss_mib=peak_before / 1024**2,
        finished_utc=core.utc_now()))


def preflight(core, args):
    versions = core.load_dependencies()
    require(peak_rss_bytes() > 0, 'OS peak-memory measurement unavailable')
    stage = args.stage26
    require(not (stage / 'run.lock').exists(), 'Stage26 has a run.lock; finish it before benchmarking')
    status = json.loads((stage / 'run_status.json').read_text(encoding='utf-8'))
    require(status.get('complete') is True and status.get('completed_patients') == 60
            and status.get('controls_passed') == 60 and not status.get('errors'),
            'Stage26 must be COMPLETE with all 60 controls passed')
    evidence = core.validate_inputs(args.root)
    full_plan = core.make_plan(evidence)
    expected = core.freeze_payload(evidence, full_plan, versions)
    frozen26 = json.loads((stage / 'experiment_configuration.json').read_text(encoding='utf-8'))
    require(frozen26 == expected, 'Stage26 frozen code/inputs/environment differ from current preflight')
    plan = [p for p in full_plan if p['class_sample_rank'] <= PER_CLASS]
    require(len(plan) == 20, 'Expected 20 fixed patients')
    refs, source_hashes = {}, {}
    for path in (stage / 'experiment_configuration.json', stage / 'run_status.json'):
        source_hashes[str(path)] = sha(path)
    for item in plan:
        case = item['case_submitter_id']
        folder = stage / 'patients' / case
        ref = core.read_checkpoint(folder, item, core.run_signature(frozen26))
        require(ref is not None, f'Missing stage26 checkpoint for {case}')
        refs[case] = ref
        source_hashes[str(folder / 'result.json')] = sha(folder / 'result.json')
        source_hashes[str(folder / 'candidates.csv.gz')] = ref['candidate_log_sha256']
    frozen = dict(version=VERSION, experiment=EXPERIMENT, script_sha256=sha(__file__),
        source26_sha256=SOURCE26_SHA256, environment=versions, threads=THREAD_ENV,
        machine=dict(host=platform.node(), processor=platform.processor(), logical_cpus=os.cpu_count(),
                     python_executable=str(Path(sys.executable).resolve())),
        project_root=str(args.root), stage26=str(stage),
        external_wsi_root=str(args.wsi_root) if args.wsi_root else None,
        original_inputs=evidence['hashes'],
        source_files=source_hashes, sample=plan, thresholds=THRESHOLDS, repeats=REPEATS,
        protocol_sha256=hashlib.sha256(PROTOCOL.encode()).hexdigest(),
        schedules={p['case_submitter_id']: schedule(i) for i, p in enumerate(plan)})
    # Canonical JSON roundtrip (tuples -> lists) for saved/resumed equality.
    frozen = json.loads(json.dumps(frozen))
    out = args.root / 'results' / EXPERIMENT
    check_freeze(out, frozen)
    print('PREFLIGHT PASS: stage26 COMPLETE, original hashes and 20 source checkpoints verified')
    print('Fixed first 10 LUAD + 10 LUSC; thresholds 0.70 / 0.80 / 0.90; 3 repeats each')
    print('180 measured + 180 warm-up scans; isolated processes; 1 numerical thread; no patient replacements')
    for i, p in enumerate(plan):
        counts = ', '.join(f'{t}={len(refs[p["case_submitter_id"]]["states"][t]["accepted"])}' for t in THRESHOLDS)
        print(f'  {i+1:02d} {p["case_submitter_id"]} {p["class_label"]} '
              f'{p["file_size_bytes"]/1024**3:.3f} GiB; stage26 {counts}')
    total = sum(p['file_size_bytes'] for p in plan) / 1024**3
    largest = max(p['file_size_bytes'] for p in plan) / 1024**3
    print(f'Download if all absent: {total:.2f} GiB; largest WSI {largest:.2f} GiB')
    print(f'Free-space budget: {total+8 if args.keep_wsi else largest+8:.2f} GiB plus logs')
    print(f'Results: {out}')
    return evidence, plan, refs, frozen


def check_freeze(out, frozen):
    path = out / 'experiment_configuration.json'
    if path.exists():
        require(json.loads(path.read_text(encoding='utf-8')) == frozen,
                'Frozen benchmark differs (code, inputs, machine or environment). Do not overwrite/bypass.')
    else:
        require(not (out / 'patients').exists(), 'Benchmark patients exist without frozen protocol')


def verify_sources(core, evidence, frozen):
    core.verify_inputs_unchanged(evidence)
    require(sha(__file__) == frozen['script_sha256'], 'Script27 changed during run')
    require(sha(Path(__file__).with_name(SOURCE26)) == frozen['source26_sha256'], 'Script26 changed')
    for path, expected in frozen['source_files'].items():
        require(sha(path) == expected, f'Stage26 source changed: {path}')


def launch_worker(core, job, folder, stem):
    jobfile = folder / (stem + '_job.json')
    resultfile = folder / (stem + '.json')
    resultfile.unlink(missing_ok=True)  # never accept stale output after a failed child
    job = dict(job, result_path=str(resultfile))
    core.atomic_json(jobfile, job)
    with (folder / (stem + '.log')).open('wb') as log:
        process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--worker', str(jobfile)],
            env={**os.environ, **THREAD_ENV}, stdout=log, stderr=subprocess.STDOUT)
        started, last_report = time.monotonic(), time.monotonic()
        try:
            while True:
                try:
                    code = process.wait(timeout=1)
                    break
                except subprocess.TimeoutExpired:
                    if time.monotonic() - last_report >= 30:
                        print(f'    {stem}: still running ({time.monotonic()-started:.0f}s)', flush=True)
                        last_report = time.monotonic()
        except BaseException:
            process.kill()
            process.wait()
            raise
    require(code == 0 and resultfile.exists(), f'Worker failed: {folder / (stem + ".log")}')
    return json.loads(resultfile.read_text(encoding='utf-8')), resultfile


def validate_run(row, ref, spec, phase, signature):
    require(row['run_signature'] == signature and row['case_submitter_id'] == ref['patient']['case_submitter_id'],
            'Worker/result identity mismatch')
    require(row['phase'] == phase and row['control_ok'] is True, 'Invalid worker phase/control')
    require(all(row[k] == spec[k] for k in ('threshold', 'repeat', 'position')), 'Schedule mismatch')
    compare_reference(row['state'], row['metadata'], ref, spec['threshold'])
    for k in METRICS + ('pre_scan_peak_rss_mib',):
        require(math.isfinite(row[k]) and row[k] >= 0, f'Invalid measurement: {k}')
    require(row['wall_seconds'] > 0 and row['peak_rss_mib'] > 0, 'Nonpositive wall time/memory')
    require(row['peak_rss_mib'] >= row['pre_scan_peak_rss_mib'], 'Peak RSS decreased')


def read_patient(core, folder, item, ref, signature, specs):
    path = folder / 'result.json'
    if not path.exists():
        return None
    result = json.loads(path.read_text(encoding='utf-8'))
    require(result['run_signature'] == signature and result['patient'] == item, 'Foreign patient checkpoint')
    require(result['status'] == 'COMPLETE' and len(result['runs']) == 9, 'Incomplete checkpoint')
    require(len(result['evidence_sha256']) == 18, 'Missing measured/warm-up evidence')
    for spec, row in zip(specs, result['runs']):
        validate_run(row, ref, spec, 'measure', signature)
        for phase in ('warmup', 'measure'):
            name = f'r{spec["repeat"]}_p{spec["position"]}_t{spec["threshold"]}_{phase}.json'
            child = folder / name
            require(sha(child) == result['evidence_sha256'][name], f'Changed worker evidence: {child}')
            stored = json.loads(child.read_text(encoding='utf-8'))
            validate_run(stored, ref, spec, phase, signature)
            if phase == 'measure':
                require(stored == row, 'Patient/worker results differ')
    return result


def median_rows(plan, results):
    rows = []
    for item in plan:
        case = item['case_submitter_id']
        if case not in results:
            continue
        for t in THRESHOLDS:
            runs = [r for r in results[case]['runs'] if r['threshold'] == t]
            first = runs[0]
            row = dict(case_submitter_id=case, class_label=item['class_label'], threshold=t,
                selected_n=len(first['state']['accepted']), checked=first['state']['checked'], repeats=len(runs))
            row['full_64'] = row['selected_n'] == 64
            for k in METRICS:
                values = [r[k] for r in runs]
                row[k] = statistics.median(values)
                row[k + '_min'] = min(values)
                row[k + '_max'] = max(values)
            rows.append(row)
    return rows


def publish(core, out, plan, results, final=False):
    complete = final and len(results) == len(plan)
    # Mark incomplete before rewriting derivative tables, then commit final status.
    core.atomic_json(out / 'run_status.json', dict(complete=False, state='UPDATING_SUMMARIES'))
    medians = median_rows(plan, results)
    raw = []
    for item in plan:
        for row in results.get(item['case_submitter_id'], {}).get('runs', []):
            raw.append({**{k: row[k] for k in ('case_submitter_id', 'threshold', 'repeat', 'position',
                'wall_seconds', 'cpu_seconds', 'peak_rss_mib', 'pre_scan_peak_rss_mib', 'wsi_source_path')},
                'class_label': item['class_label'], 'selected_n': len(row['state']['accepted']),
                'checked': row['state']['checked'], 'control_ok': row['control_ok']})
    core.atomic_csv(out / 'measurements.csv', raw, ['case_submitter_id','class_label','threshold','repeat','position',
        *METRICS,'pre_scan_peak_rss_mib','wsi_source_path','selected_n','checked','control_ok'])
    core.atomic_csv(out / 'patient_medians.csv', medians, ['case_submitter_id','class_label','threshold',
        'selected_n','checked','repeats','full_64',*[v for k in METRICS for v in (k,k+'_min',k+'_max')]])
    index = {(r['case_submitter_id'], r['threshold']): r for r in medians}
    pairs, effects, thresholds = [], [], []
    for item in plan:
        case = item['case_submitter_id']
        if case not in results:
            continue
        a = index[case, '0.80']
        for t in ('0.70', '0.90'):
            b = index[case, t]
            status = ('BOTH_COMPLETE' if a['full_64'] and b['full_64'] else
                'BASELINE_ONLY' if a['full_64'] else 'TARGET_ONLY' if b['full_64'] else 'NEITHER_COMPLETE')
            r = dict(case_submitter_id=case, class_label=item['class_label'], target_threshold=t,
                     baseline_threshold='0.80', pair_status=status)
            for k in METRICS:
                r[k+'_attempt_ratio'] = b[k] / a[k] if a[k] else None
                r[k+'_attempt_difference'] = b[k] - a[k]
            pairs.append(r)
    for group in ('ALL', 'LUAD', 'LUSC'):
        gp = [r for r in pairs if group == 'ALL' or r['class_label'] == group]
        gm = [r for r in medians if group == 'ALL' or r['class_label'] == group]
        for t in THRESHOLDS:
            z = [r for r in gm if r['threshold'] == t]
            entry = dict(group=group, threshold=t, n_processed=len(z), n_full_64=sum(r['full_64'] for r in z),
                         final_analysis=complete)
            for k in METRICS:
                entry['mean_attempt_'+k] = statistics.mean(r[k] for r in z) if complete and z else None
            thresholds.append(entry)
        for t in ('0.70', '0.90'):
            z = [r for r in gp if r['target_threshold'] == t]
            for population in ('BOTH_COMPLETE', 'ALL_ATTEMPTS'):
                eligible = [r for r in z if population == 'ALL_ATTEMPTS' or r['pair_status'] == 'BOTH_COMPLETE']
                for k in METRICS:
                    for stat in ('ratio', 'difference'):
                        field = k+'_attempt_'+stat
                        valid = [r for r in eligible if r[field] is not None]
                        values = [r[field] for r in valid]
                        lo, hi, method = (None, None, 'partial')
                        if complete:
                            lo, hi, method = core.boot_mean_ci(values, [r['class_label'] for r in valid],
                                f'BENCHMARK20_V1|{group}|{t}|{population}|{field}')
                        effects.append(dict(group=group, target_threshold=t, baseline_threshold='0.80',
                            population=population, metric=k, statistic=stat,
                            primary_benchmark_endpoint=(group=='ALL' and t=='0.90' and population=='BOTH_COMPLETE'
                                                        and k=='wall_seconds' and stat=='ratio'),
                            final_analysis=complete, n_pairs=len(valid), n_zero_denominator=len(eligible)-len(valid),
                            n_both_complete=sum(r['pair_status']=='BOTH_COMPLETE' for r in z),
                            n_baseline_only=sum(r['pair_status']=='BASELINE_ONLY' for r in z),
                            n_target_only=sum(r['pair_status']=='TARGET_ONLY' for r in z),
                            n_neither_complete=sum(r['pair_status']=='NEITHER_COMPLETE' for r in z),
                            mean=statistics.mean(values) if complete and values else None,
                            ci95_low=lo, ci95_high=hi, ci_method=method))
    core.atomic_csv(out / 'paired_comparisons.csv', pairs, ['case_submitter_id','class_label',
        'target_threshold','baseline_threshold','pair_status',*[k+'_attempt_'+s for k in METRICS for s in ('ratio','difference')]])
    core.atomic_csv(out / 'threshold_summary.csv', thresholds)
    core.atomic_csv(out / 'paired_effect_summary.csv', effects)
    core.atomic_json(out / 'run_status.json', dict(complete=complete,
        state='COMPLETE' if complete else 'PARTIAL', expected_patients=len(plan), completed_patients=len(results),
        measured_runs=len(raw), pending_patients=[p['case_submitter_id'] for p in plan if p['case_submitter_id'] not in results],
        updated_utc=core.utc_now(), note='Final aggregate estimates require all 20 validated patients; raw partial rows are available.'))
    return complete


def execute(core, args, evidence, plan, refs, frozen):
    out = args.root / 'results' / EXPERIMENT
    signature = core.run_signature(frozen)
    with core.run_lock(out):
        results = {}
        try:
            check_freeze(out, frozen)
            if args.summarize_only:
                require((out / 'experiment_configuration.json').exists(), 'Prepare/run first')
            core.atomic_json(out / 'experiment_configuration.json', frozen)
            (out / 'protocol.md').write_text(PROTOCOL, encoding='utf-8')
            core.atomic_csv(out / 'sample_manifest.csv', plan)
            core.atomic_csv(out / 'schedule.csv', [dict(case_submitter_id=p['case_submitter_id'], **s)
                for p in plan for s in frozen['schedules'][p['case_submitter_id']]])
            core.atomic_json(out / 'run_status.json', dict(complete=False, state='VALIDATING'))
            for item in plan:
                case = item['case_submitter_id']
                result = read_patient(core, out/'patients'/case, item, refs[case], signature, frozen['schedules'][case])
                if result is not None:
                    results[case] = result
            if not args.prepare and not args.summarize_only:
                processed = 0
                for item in plan:
                    case = item['case_submitter_id']
                    if case in results:
                        print(f'SKIP {case}: verified completed checkpoint', flush=True)
                        if not args.keep_wsi:
                            core.clean_owned_download(out, item)
                        continue
                    if args.limit is not None and processed >= args.limit:
                        break
                    folder = out/'patients'/case
                    folder.mkdir(parents=True, exist_ok=True)
                    print(f'PROCESS {case} ({item["class_label"]}); all 9 measurements plus warm-ups', flush=True)
                    path, owned = core.obtain_wsi(item, args, out)
                    runs, hashes = [], {}
                    for spec in frozen['schedules'][case]:
                        for phase in ('warmup', 'measure'):
                            stem = f'r{spec["repeat"]}_p{spec["position"]}_t{spec["threshold"]}_{phase}'
                            print(f'  {stem}', flush=True)
                            refpath = args.stage26/'patients'/case/'result.json'
                            job = dict(**spec, phase=phase, patient=item, wsi=str(path),
                                run_signature=signature, script_sha256=frozen['script_sha256'],
                                environment=frozen['environment'], reference_path=str(refpath),
                                reference_sha256=frozen['source_files'][str(refpath)])
                            row, resultfile = launch_worker(core, job, folder, stem)
                            require(row['wsi_source_path'] == str(path), 'Worker WSI path mismatch')
                            validate_run(row, refs[case], spec, phase, signature)
                            hashes[resultfile.name] = sha(resultfile)
                            if phase == 'measure':
                                runs.append(row)
                                print(f'    wall={row["wall_seconds"]:.3f}s CPU={row["cpu_seconds"]:.3f}s '
                                      f'peak RSS={row["peak_rss_mib"]:.1f} MiB; '
                                      f'selected={len(row["state"]["accepted"])}; stage26 reproduced', flush=True)
                    core.atomic_json(folder/'result.json', dict(status='COMPLETE', run_signature=signature,
                        patient=item, runs=runs, evidence_sha256=hashes, finished_utc=core.utc_now()))
                    results[case] = read_patient(core, folder, item, refs[case], signature, frozen['schedules'][case])
                    processed += 1
                    publish(core, out, plan, results)
                    if owned and not args.keep_wsi:
                        core.clean_owned_download(out, item)
            verify_sources(core, evidence, frozen)
            complete = publish(core, out, plan, results, final=True)
            print(f'Benchmark {"COMPLETE" if complete else "PARTIAL"}: {out}')
            if args.prepare:
                print('Protocol/sample saved. No WSI downloads or scans.')
            elif not complete:
                print('Use --run without --limit to continue. Partial patients restart all repeats.')
            return 0
        except BaseException as exc:
            core.atomic_json(out/'run_status.json', dict(complete=False, state='STOPPED',
                known_completed_patients=len(results), error=str(exc) or 'Interrupted',
                updated_utc=core.utc_now(), note='Technical/validation failure, not a QC exclusion.'))
            raise


def self_test():
    for i in range(20):
        specs = schedule(i)
        require(len(specs) == 9, 'Schedule size')
        for t in THRESHOLDS:
            require(sorted(s['position'] for s in specs if s['threshold']==t)==[1,2,3], 'Unbalanced position')
        for r in (1,2,3):
            require(sorted(s['threshold'] for s in specs if s['repeat']==r)==list(THRESHOLDS), 'Unbalanced round')
    require(peak_rss_bytes() > 0, 'Peak RSS unavailable')
    core = load_core()
    core.np = __import__('numpy')
    low, high, _ = core.boot_mean_ci([1,2,3,4], ['LUAD','LUAD','LUSC','LUSC'], 'BENCHMARK20_TEST')
    require(1 <= low <= 2.5 <= high <= 4, 'Bootstrap invalid')
    print('SELF-TEST PASS: core26 hash, balanced schedule, OS peak RSS, patient bootstrap; no WSI touched')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group()
    for flag in ('preflight-only','prepare','run','summarize-only','self-test'):
        mode.add_argument('--'+flag, action='store_true')
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument('--stage26', type=Path)
    parser.add_argument('--wsi-root', type=Path)
    parser.add_argument('--gdc-client', type=Path)
    parser.add_argument('--keep-wsi', action='store_true')
    parser.add_argument('--limit', type=int, help='Max NEW patients this invocation; completed patients skipped')
    parser.add_argument('--worker', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
        return 0
    if args.self_test:
        self_test()
        return 0
    require(args.limit is None or (args.run and args.limit > 0), '--limit needs --run and a positive integer')
    args.root = args.root.resolve()
    args.stage26 = (args.stage26 or args.root/'results'/'tissue_threshold_paired60_v1').resolve()
    args.gdc_client = (args.gdc_client or args.root/'tools'/'gdc-client.exe').resolve()
    if args.wsi_root:
        args.wsi_root = args.wsi_root.resolve()
        require(args.wsi_root.is_dir(), '--wsi-root directory not found')
    os.environ.update(THREAD_ENV)
    core = load_core()
    evidence, plan, refs, frozen = preflight(core, args)
    if not (args.prepare or args.run or args.summarize_only):
        print('Read-only preflight complete. Use --prepare, then --run --limit 1.')
        return 0
    return execute(core, args, evidence, plan, refs, frozen)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print('\nInterrupted. Resume with --run; partial patient restarts all repeats.', file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f'ERROR: {type(exc).__name__}: {exc}', file=sys.stderr)
        raise SystemExit(1)
