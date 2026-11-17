#!/usr/bin/env python3
r"""Stage 32 v1.0.0 -- frozen nine-model, patient-level internal test evaluation.

Install as scripts\32_evaluate_patch_budget_test.py in mm_oncology_efficiency.
Run from the project root with the EXISTING .venv, without editing Stages 28--31:
  .\.venv\Scripts\python.exe -B scripts\32_evaluate_patch_budget_test.py --self-test
  .\.venv\Scripts\python.exe -B scripts\32_evaluate_patch_budget_test.py --preflight-only
  .\.venv\Scripts\python.exe -B scripts\32_evaluate_patch_budget_test.py --prepare
  .\.venv\Scripts\python.exe -B scripts\32_evaluate_patch_budget_test.py --smoke
  .\.venv\Scripts\python.exe -B scripts\32_evaluate_patch_budget_test.py --run
  .\.venv\Scripts\python.exe -B scripts\32_evaluate_patch_budget_test.py --analyze

--prepare freezes the plan and all nine previously selected checkpoints.
--smoke reproduces ALL 186 validation predictions for EACH selected checkpoint.
Only --run can read test PNGs. It scores all nine models, never trains or selects.
Repeat the SAME command after a power failure. Each patient prediction is committed
atomically with its own checksum/identity; only uncommitted work is repeated.
No partial test metrics are printed. --analyze requires all 9 x 186 predictions.
If a committed record is corrupt, STOP for review: do not delete or replace results.
Close output CSVs in Excel before rebuilding summaries. There is no --force/reset.

Output: results\stage32_patch_budget_test_v1\ (old results are read only).
Main files: protocol.json, preparation.json, smoke_report.json, test_predictions.csv,
metrics_by_run.csv, metrics_by_budget.csv, paired_auroc_comparisons.csv,
training_resources_by_run.csv, training_resources_by_budget.csv, analysis.json,
REPORT.md, completion.json. Individual records under records/{validation,test}/.
No weights, source images, packages or network downloads are created by this script.

Analysis fixed BEFORE test: primary mean-of-three-seeds AUROC(K16)-AUROC(K64),
secondary K32-K64; 10,000 paired class-stratified patient bootstrap replicates,
PCG64 seed 20261116. Same resampled patients for all nine models. Percentile 95%
intervals condition on these three trained seeds; they do NOT estimate uncertainty
across the population of training seeds. No seed stacking, seed resampling, ensemble,
p-value, noninferiority/equivalence margin, threshold tuning or best-seed selection.
K is the TRAINING budget. Every evaluation uses all 64 patches, threshold 0.5.
Resource tables reuse measured Stage31 training costs; evaluation timings are not
an inference benchmark or evidence of energy savings. Frozen plan is local, not an
independently timestamped public preregistration. Internal patient split, not external
or site-held-out validation. Keep TSS/MPP and validation-reuse limitations in the paper.

Local CPU self-tests do not establish CUDA compatibility: --smoke must pass locally.
"""
import sys
sys.dont_write_bytecode = True
import argparse
import collections
import csv
import datetime
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import statistics
import tempfile
import uuid

VERSION = '1.0.0'
ROOT = Path(__file__).resolve().parent.parent
OUTPUT_NAME = 'stage32_patch_budget_test_v1'
SOURCE_NAME = 'stage31_patch_budget_v1_1'
STAGE31_PROTOCOL_SHA = '22914f0aa57fea450eb8704e9823a74f0b75a1c048cf8c603cc029da2edce015'
CODE_HASHES = {
    '28_prepare_patch_budget_experiment.py': '9d818c3734ff760cba8eaa07f36c098c9ad388a86479c7e5d999e7a37224759f',
    '29_train_patch_budget_baselines.py': 'a8a7409a32c748c3529544138b5e20b8f6294b233a8fb22c20585dcd522ab09d',
    '30_diagnose_balanced_patient_batches.py': '20c8a3af5c3bb9f1c25dd3a322ece6fb43fb8f902a05644c211262b45b2d5ca7',
    '31_train_patch_budget_baselines.py': '53432b4f5af2a8266ae717563de27dda03c412e98f4f737335bd5e905f19160b',
}
BUDGETS = (16, 32, 64)
SEEDS = (20261113, 20261114, 20261115)
MATRIX = [(k, s) for k in (64, 32, 16) for s in SEEDS]
# These model exports/epochs were selected before Stage32, audited from long_run.zip.
REVIEWED_MODELS = {'K64_seed20261113': {'sha256': 'cb3fd0198f103a7320a36e1961b85655cb947cc864c89353bbc228fbd934487b', 'epoch': 7, 'val_ce': 0.37373382181034176}, 'K64_seed20261114': {'sha256': '2f625a948f8aacae063ef95500c02df22d53603ee9b715ae8608721c7dc887cc', 'epoch': 9, 'val_ce': 0.4135463819457268}, 'K64_seed20261115': {'sha256': 'd844a88ce63a26121fcf050f56afc6c771d725b460e9a73c3a83a55e92d265d5', 'epoch': 6, 'val_ce': 0.43747724879593136}, 'K32_seed20261113': {'sha256': 'a03d1fdaa8f16bec6efdb17a30a7c4e70d9d155ca7aeae5130e6a80aac2db291', 'epoch': 16, 'val_ce': 0.32561006862153735}, 'K32_seed20261114': {'sha256': 'fe29b571796add8f2bfd4bea40a4831213773fb3c8b0c79d9d39ac055e8ccde1', 'epoch': 14, 'val_ce': 0.34303345336542485}, 'K32_seed20261115': {'sha256': '7728c5b65bb209be66a0c9d8fb6c6c7731f6bb8b4fdf76be79b114e4817c5b5c', 'epoch': 21, 'val_ce': 0.3131757344047898}, 'K16_seed20261113': {'sha256': 'e25a8f6a898599be917cf757150c52c8019cf0a33b7561bc9945beaf4b43fb3d', 'epoch': 14, 'val_ce': 0.35082940329808204}, 'K16_seed20261114': {'sha256': '685f2bc6d40ab5deceb64aecd400dddf5bbe5c08d4acd3fce3d7b4bc657b17eb', 'epoch': 17, 'val_ce': 0.30381849011979695}, 'K16_seed20261115': {'sha256': '50938ec79c272ff9f85518b13d0539f291ee5f45ebbbe9313dcdc516b72b449b', 'epoch': 29, 'val_ce': 0.3419997044354072}}
SOURCE_FILES = ('best_model.json', 'epoch_metrics.csv', 'validation_predictions.csv',
                'status.json', 'run_summary.json', 'resource_summary.json',
                'resources_by_epoch.csv', 'resource_sessions.csv', 'best_model.pt')
PLAN = {
    'evaluation': 'all 64 patches in frozen tile-index order; no augmentation; no crop',
    'precision': 'FP16 autocast encoder; FP32 gated attention, pooling and classifier',
    'normalization': 'RGB/255; ImageNet mean (0.485,0.456,0.406), std (0.229,0.224,0.225)',
    'positive_class': 'LUSC=1; LUAD=0', 'threshold': 0.5,
    'models': 'all nine Stage31 minimum-validation-CE selected exports; no reselection',
    'primary': 'mean across paired seeds of test AUROC(K16)-AUROC(K64)',
    'secondary': 'mean across paired seeds of test AUROC(K32)-AUROC(K64)',
    'bootstrap': {'replicates': 10000, 'seed': 20261116, 'rng': 'numpy PCG64',
                  'method': 'paired patient bootstrap within each class; same multiplicities across all nine models',
                  'interval': '95% percentile, numpy quantile method=linear',
                  'scope': 'conditional on three fitted seeds and this internal class-stratified test sample'},
    'descriptive_metrics': ['auroc', 'balanced_accuracy', 'macro_f1', 'accuracy',
                            'sensitivity', 'specificity', 'ce', 'brier'],
    'seed_summary': 'arithmetic mean and sample SD (ddof=1); not an ensemble',
    'inference_policy': 'no p-values, no noninferiority/equivalence claims, no threshold tuning',
    'secondary_intervals': 'pointwise exploratory; no family-wise coverage claim',
    'validation_reproduction': {'patients_per_model': 186, 'probability_atol': 1e-6,
                                'logit_atol': 1e-5, 'rtol': 0.0},
    'limitations': ['three seeds; internal patient split, not a held-out-site evaluation',
                   'TSS/class confounding and heterogeneous physical patch scale',
                   '20 pilot-validation patients used in earlier development',
                   'test has not informed this analysis plan; plan follows validation inspection',
                   'training resource savings do not establish inference or energy savings'],
}
S29 = None


class AuditError(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise AuditError(message)


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def encoded(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def sha(path):
    require(path.is_file() and not path.is_symlink(), f'Missing/nonregular/symlink file: {path}')
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def atomic_bytes(path, raw):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp-' + uuid.uuid4().hex)
    try:
        with tmp.open('xb') as f:
            f.write(raw)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        if os.name != 'nt':
            fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
    finally:
        tmp.unlink(missing_ok=True)


def atomic_json(path, value):
    atomic_bytes(path, encoded(value) + b'\n')


def table(path, rows):
    require(bool(rows), f'Empty table: {path}')
    out = io.StringIO(newline='')
    w = csv.DictWriter(out, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
    atomic_bytes(path, out.getvalue().encode('utf-8'))


def load_stage29():
    global S29
    if S29 is None:
        p = ROOT / 'scripts' / '29_train_patch_budget_baselines.py'
        require(sha(p) == CODE_HASHES[p.name], 'Stage29 hash mismatch.')
        spec = importlib.util.spec_from_file_location('stage29_for_stage32', p)
        S29 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(S29)
    return S29


def run_name(k, s):
    return f'K{k}_seed{s}'


def expected_identity(k, s):
    return dict(budget=k, seed=s, max_epochs=30, patience=5, subset='full',
                train_n=557, validation_n=186, protocol_sha256=STAGE31_PROTOCOL_SHA)


def prediction_map(rows, ids, labels):
    result = {}
    for r in rows:
        pid = r['case_submitter_id']
        require(pid in labels and pid not in result, f'Duplicate/unknown prediction ID: {pid}')
        y, p, z = int(r['label']), float(r['p_LUSC']), float(r['logit_diff_LUSC_minus_LUAD'])
        require(y == labels[pid] and math.isfinite(p) and 0 <= p <= 1 and math.isfinite(z),
                f'Invalid prediction/label: {pid}')
        sigmoid = math.exp(-max(-z, 0)) / (1 + math.exp(-abs(z)))
        require(abs(p - sigmoid) <= 2e-6, f'Probability/logit disagreement: {pid}')
        result[pid] = dict(case_submitter_id=pid, label=y, p_LUSC=p, logit_diff_LUSC_minus_LUAD=z)
    require(set(result) == set(ids), 'Prediction IDs do not match the required split.')
    return result


def metrics(y, p, z):
    import numpy as np
    y, p, z = np.asarray(y), np.asarray(p, dtype=float), np.asarray(z, dtype=float)
    require(y.ndim == p.ndim == z.ndim == 1 and len(y) == len(p) == len(z) and set(y) == {0, 1},
            'Metrics need aligned vectors with both classes.')
    require(np.isfinite(p).all() and np.isfinite(z).all() and ((p >= 0) & (p <= 1)).all(), 'Nonfinite scores.')
    pos, neg = p[y == 1], p[y == 0]
    auc = float(((pos[:, None] > neg) + .5 * (pos[:, None] == neg)).mean())
    pred = p >= .5
    tp = int(((y == 1) & pred).sum()); tn = int(((y == 0) & ~pred).sum())
    fp, fn = len(neg) - tn, len(pos) - tp
    sensitivity, specificity = tp / len(pos), tn / len(neg)
    return dict(auroc=auc, balanced_accuracy=(sensitivity + specificity) / 2,
                macro_f1=(2 * tp / (2 * tp + fp + fn) + 2 * tn / (2 * tn + fp + fn)) / 2,
                accuracy=(tp + tn) / len(y), sensitivity=sensitivity, specificity=specificity,
                ce=float(np.mean(np.logaddexp(0, z) - y * z)), brier=float(np.mean((p - y) ** 2)),
                tn=tn, fp=fp, fn=fn, tp=tp, patients=len(y))


def audit_run(path, k, seed, val_ids, labels):
    best = read_json(path / 'best_model.json')
    expected = REVIEWED_MODELS[run_name(k, seed)]
    require(best['sha256'] == expected['sha256'] and best['epoch'] == expected['epoch']
            and best['val_ce'] == expected['val_ce'], f'Selected model differs from reviewed series: {path.name}')
    require(sha(path / 'best_model.pt') == best['sha256'], f'Model byte hash mismatch: {path.name}')
    identity = expected_identity(k, seed)
    status, summary = read_json(path / 'status.json'), read_json(path / 'run_summary.json')
    require(best['identity'] == status['identity'] == identity and best['budget'] == k and best['seed'] == seed,
            f'Run identity mismatch: {path.name}')
    require(status['status'] == 'COMPLETE' and summary['status'] == status['outcome']
            and summary['status'] in ('EARLY_STOP_PATIENCE', 'MAX_EPOCHS') and summary['test_scored'] is False,
            f'Incomplete or previously test-scored source: {path.name}')
    rows = [r for r in read_csv(path / 'epoch_metrics.csv') if int(r['epoch']) > 0]
    n = summary['completed_epochs']
    require([int(r['epoch']) for r in rows] == list(range(1, n + 1)) and 1 <= n <= 30, 'Epoch coverage mismatch.')
    winner = min(rows, key=lambda r: (float(r['val_ce']), int(r['epoch'])))
    require(int(winner['epoch']) == best['epoch'] == summary['selection']['best_epoch']
            and float(winner['val_ce']) == best['val_ce'] == summary['selection']['best_ce'], 'Wrong selected epoch.')
    bce, wait = math.inf, 0
    for i, row in enumerate(rows, 1):
        require(int(row['budget']) == k and int(row['seed']) == seed and int(row['train_patients']) == 557
                and int(row['train_patches']) == 557 * k and int(row['skipped_steps']) == 0, 'Epoch identity/AMP mismatch.')
        ce = float(row['val_ce']); require(math.isfinite(ce), 'Nonfinite validation CE.')
        if ce < bce:
            bce, wait = ce, 0
        else:
            wait += 1
        require(wait < 5 or i == n, 'Training continued past the frozen patience rule.')
    require((summary['status'] == 'EARLY_STOP_PATIENCE' and wait == 5) or
            (summary['status'] == 'MAX_EPOCHS' and n == 30), 'Stopping rule mismatch.')
    saved = [r for r in read_csv(path / 'validation_predictions.csv') if int(r['epoch']) == best['epoch']]
    reference = prediction_map(saved, val_ids, labels)
    m = metrics([reference[p]['label'] for p in val_ids], [reference[p]['p_LUSC'] for p in val_ids],
                [reference[p]['logit_diff_LUSC_minus_LUAD'] for p in val_ids])
    for key in ('auroc', 'balanced_accuracy', 'macro_f1', 'accuracy', 'ce'):
        require(abs(m[key] - float(winner['val_' + key])) <= (2e-6 if key == 'ce' else 1e-12),
                f'Independent validation metric mismatch: {path.name}/{key}')
    return best, reference


def context(check_environment):
    s29 = load_stage29()
    source = ROOT / 'results' / SOURCE_NAME
    require(sha(source / 'protocol.json') == STAGE31_PROTOCOL_SHA, 'Wrong Stage31 protocol/version.')
    old = read_json(source / 'protocol.json')
    require(importlib.metadata.version('numpy') == old['environment']['versions']['numpy'],
            'NumPy version differs from the frozen analysis environment.')
    require(read_json(source / 'preparation.json')['protocol_sha256'] == STAGE31_PROTOCOL_SHA, 'Unprepared Stage31.')
    for name, expected in CODE_HASHES.items():
        require(sha(ROOT / 'scripts' / name) == expected, f'Frozen code changed: {name}')
    bundle = s29.load_bundle(ROOT)  # metadata/hashes only; no image reads
    counts = collections.Counter((r['split'], r['class_label']) for r in bundle['patients'].values())
    require(counts == {('train', 'LUAD'): 273, ('train', 'LUSC'): 284, ('validation', 'LUAD'): 91,
                       ('validation', 'LUSC'): 95, ('test', 'LUAD'): 91, ('test', 'LUSC'): 95}, 'Unexpected splits.')
    require(bundle['manifest_sha256'] == old['stage28_manifest_sha256'], 'Stage28/31 mismatch.')
    ids = {part: sorted(p for p, r in bundle['patients'].items() if r['split'] == part)
           for part in ('train', 'validation', 'test')}
    labels = {p: int(r['class_label'] == 'LUSC') for p, r in bundle['patients'].items()}
    models, refs, files = {}, {}, {}
    for k, seed in MATRIX:
        name = run_name(k, seed); path = source / 'runs' / name
        best, reference = audit_run(path, k, seed, ids['validation'], labels)
        models[name] = best; refs[name] = reference
        for filename in SOURCE_FILES:
            files[str((path / filename).relative_to(ROOT)).replace('\\', '/')] = sha(path / filename)
        print(f'  VERIFIED {name}: selected epoch {best["epoch"]}, complete; 186 validation references', flush=True)
    device = None
    if check_environment:
        device = s29.initialize_torch(cuda=True)
        require(s29.environment_record(bundle) == old['environment'], 'Environment differs from Stage31; review before proceeding.')
    record = dict(schema_version=1, stage32_version=VERSION, script32_sha256=sha(Path(__file__)),
                  dependent_scripts_sha256=CODE_HASHES, stage31_protocol_sha256=STAGE31_PROTOCOL_SHA,
                  stage28_manifest_sha256=bundle['manifest_sha256'], environment=old['environment'],
                  patient_ids=ids, labels=labels, models=models, source_files_sha256=files, plan=PLAN)
    return dict(s29=s29, bundle=bundle, ids=ids, labels=labels, refs=refs, models=models,
                source=source, output=ROOT / 'results' / OUTPUT_NAME, record=record,
                identity=digest(record), device=device)


def verify_prepared(ctx):
    out = ctx['output']
    require((out / 'protocol.json').is_file() and (out / 'preparation.json').is_file(), 'Run --prepare first.')
    require(read_json(out / 'protocol.json') == ctx['record'], 'Frozen Stage32 plan/inputs/code changed. Do not overwrite it.')
    prep = read_json(out / 'preparation.json')
    require(prep['identity'] == ctx['identity'] and prep['protocol_sha256'] == sha(out / 'protocol.json'), 'Preparation mismatch.')


def prepare(ctx):
    out = ctx['output']
    if out.exists():
        verify_prepared(ctx)
        print('ALREADY PREPARED: verified; no changes.'); return
    # Sibling lock prevents concurrent preparation; no existing output is overwritten.
    with ctx['s29'].execution_lock(out.parent / (out.name + '.lock')):
        if out.exists():
            verify_prepared(ctx); return
        staging = out.with_name(out.name + '.preparing-' + uuid.uuid4().hex)
        staging.mkdir()
        atomic_json(staging / 'protocol.json', ctx['record'])
        atomic_json(staging / 'preparation.json', dict(identity=ctx['identity'], utc=utc(),
                    protocol_sha256=sha(staging / 'protocol.json'), status='PREPARED_BEFORE_TEST'))
        os.rename(staging, out)
    print(f'PREPARED: {out}. No test images opened; no model inference.')


class SplitReader:
    """Independent inference-only loader. Test requires an explicit --run capability."""
    def __init__(self, ctx, part, test_authorized=False):
        require(part in ('validation', 'test'), 'Training data are not supported by this loader.')
        require(part != 'test' or test_authorized, 'Test reading is only enabled by --run.')
        self.ctx, self.part = ctx, part
        self.allowed = set(ctx['ids'][part])

    def load(self, pid):
        require(pid in self.allowed and self.ctx['bundle']['patients'][pid]['split'] == self.part,
                f'Patient outside authorized split: {pid}')  # guard BEFORE any byte read
        import numpy as np
        from PIL import Image
        torch = self.ctx['s29'].torch
        images = []
        rows = self.ctx['bundle']['patches'][pid]
        require([r['index'] for r in rows] == list(range(1, 65)), 'Patch order/count changed.')
        for row in rows:
            raw = self.ctx['s29'].verified_png_bytes(row)
            with Image.open(io.BytesIO(raw)) as im:
                require(im.format == 'PNG' and im.mode == 'RGB' and im.size == (256, 256), 'PNG shape/mode changed.')
                im.load()
                array = np.array(im, copy=True, order='C')
            images.append(torch.from_numpy(array).permute(2, 0, 1))
        x = torch.stack(images).float().div_(255.)
        x.sub_(torch.tensor([.485, .456, .406]).view(1, 3, 1, 1))
        x.div_(torch.tensor([.229, .224, .225]).view(1, 3, 1, 1))
        return x.to(self.ctx['device'])


def load_model(ctx, name):
    torch = ctx['s29'].torch
    path = ctx['source'] / 'runs' / name / 'best_model.pt'
    best = ctx['models'][name]
    require(sha(path) == best['sha256'], f'Model changed during execution: {name}')
    state = torch.load(path, map_location='cpu', weights_only=True)
    require(set(state) == {'model', 'meta'}, 'Unexpected export payload.')
    meta = {key: best[key] for key in ('epoch', 'val_ce', 'budget', 'seed', 'identity', 'checkpoint_sequence')}
    require(state['meta'] == meta, 'Embedded checkpoint metadata disagree with best_model.json.')
    model = ctx['s29'].build_model(None, seed=best['seed'], device=torch.device('cpu'))
    model.load_state_dict(state['model'], strict=True)
    require(all(torch.isfinite(v).all().item() for v in model.state_dict().values() if v.is_floating_point()),
            'Nonfinite model parameters/buffers.')
    model.to(ctx['device']).eval()
    model.requires_grad_(False)
    return model


def predict(ctx, model, x):
    torch = ctx['s29'].torch
    device = ctx['device']
    with torch.inference_mode():
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == 'cuda'):
            h = model.encoder(x)
        h = h.float()
        scores = model.attention_w(torch.tanh(model.attention_v(h)) * torch.sigmoid(model.attention_u(h)))
        attention = torch.softmax(scores.squeeze(-1), dim=0)
        pooled = torch.sum(attention[:, None] * h, dim=0, keepdim=True)
        logits = model.classifier(pooled).float()
        require(torch.isfinite(logits).all().item(), 'Nonfinite inference logits.')
        p = float(torch.softmax(logits, dim=1)[0, 1].item())
        z = float((logits[0, 1] - logits[0, 0]).item())
    return p, z


def record_path(ctx, part, name, pid):
    require(part in ('validation', 'test') and name in ctx['models'] and pid in ctx['ids'][part], 'Unsafe record key.')
    return ctx['output'] / 'records' / part / name / (pid + '.json')


def record_value(ctx, part, name, pid, p, z):
    return dict(identity=ctx['identity'], split=part, model=name, case_submitter_id=pid,
                label=ctx['labels'][pid], p_LUSC=p, logit_diff_LUSC_minus_LUAD=z,
                model_sha256=ctx['models'][name]['sha256'], evaluation_patches=64)


def read_record(ctx, part, name, pid):
    path = record_path(ctx, part, name, pid)
    if not path.exists():
        return None
    require(not path.is_symlink(), 'Prediction record is a symlink.')
    envelope = read_json(path)
    r = envelope['record']
    require(envelope['sha256'] == digest(r), f'Corrupt committed prediction: {path}. Stop for review.')
    expected = record_value(ctx, part, name, pid, r['p_LUSC'], r['logit_diff_LUSC_minus_LUAD'])
    require(r == expected, f'Prediction identity mismatch: {path}')
    prediction_map([r], [pid], ctx['labels'])
    return r


def commit_record(ctx, part, name, pid, p, z):
    path = record_path(ctx, part, name, pid)
    value = record_value(ctx, part, name, pid, p, z)
    prediction_map([value], [pid], ctx['labels'])
    require(not path.exists(), f'Refusing to overwrite a committed prediction: {path}')
    atomic_json(path, dict(record=value, sha256=digest(value)))
    return value


def compare_validation(ctx, name, row):
    ref = ctx['refs'][name][row['case_submitter_id']]
    dp = abs(row['p_LUSC'] - ref['p_LUSC'])
    dz = abs(row['logit_diff_LUSC_minus_LUAD'] - ref['logit_diff_LUSC_minus_LUAD'])
    require(dp <= PLAN['validation_reproduction']['probability_atol'] and
            dz <= PLAN['validation_reproduction']['logit_atol'],
            f'Validation reproduction FAILED: {name}/{row["case_submitter_id"]}: dp={dp}, dz={dz}. Test remains blocked.')
    return dp, dz


def verify_smoke(ctx):
    path = ctx['output'] / 'smoke_report.json'
    require(path.is_file(), 'Run --smoke first. No test image has been opened by this command.')
    report = read_json(path)
    require(report['identity'] == ctx['identity'] and report['status'] == 'PASS', 'Invalid smoke report.')
    for k, seed in MATRIX:
        name = run_name(k, seed)
        for pid in ctx['ids']['validation']:
            row = read_record(ctx, 'validation', name, pid)
            require(row is not None, f'Missing validation reproduction record: {name}/{pid}')
            compare_validation(ctx, name, row)


def evaluate(ctx, part):
    if part == 'test':
        verify_smoke(ctx)  # all checks precede loader creation and test reads
    reader = SplitReader(ctx, part, test_authorized=part == 'test')
    results = []
    for k, seed in MATRIX:
        name = run_name(k, seed); model = None; dp_max = dz_max = 0.
        for i, pid in enumerate(ctx['ids'][part], 1):
            row = read_record(ctx, part, name, pid)
            if row is None:
                if model is None:
                    model = load_model(ctx, name)
                p, z = predict(ctx, model, reader.load(pid))
                proposed = record_value(ctx, part, name, pid, p, z)
                if part == 'validation':
                    compare_validation(ctx, name, proposed)  # do not commit a failed reproduction
                row = commit_record(ctx, part, name, pid, p, z)
            if part == 'validation':
                dp, dz = compare_validation(ctx, name, row)
                dp_max, dz_max = max(dp_max, dp), max(dz_max, dz)
            if i % 30 == 0 or i == len(ctx['ids'][part]):
                print(f'  {part} {name}: {i}/{len(ctx["ids"][part])} committed (including resume)', flush=True)
        del model
        ctx['s29'].torch.cuda.empty_cache()
        results.append(dict(model=name, patients=len(ctx['ids'][part]), max_probability_difference=dp_max,
                            max_logit_difference=dz_max))
    if part == 'validation':
        atomic_json(ctx['output'] / 'smoke_report.json', dict(status='PASS', identity=ctx['identity'],
                    utc=utc(), comparisons=results, test_images_opened=False))
        print('SMOKE PASS: all 9 x 186 validation predictions reproduced. Next: --run.')
    else:
        analyze(ctx)


def complete_test_matrix(ctx):
    import numpy as np
    ids = ctx['ids']['test']
    y = np.array([ctx['labels'][p] for p in ids])
    predictions, arrays, logits = [], {}, {}
    for k, seed in MATRIX:
        name = run_name(k, seed); rows = []
        for pid in ids:
            row = read_record(ctx, 'test', name, pid)
            require(row is not None, f'Test matrix incomplete: {name}/{pid}. Continue --run; no partial analysis.')
            rows.append(row)
        indexed = prediction_map(rows, ids, ctx['labels'])
        arrays[(k, seed)] = np.array([indexed[p]['p_LUSC'] for p in ids])
        logits[(k, seed)] = np.array([indexed[p]['logit_diff_LUSC_minus_LUAD'] for p in ids])
        predictions.extend(dict(budget=k, seed=seed, selected_epoch=ctx['models'][name]['epoch'],
                                **indexed[p]) for p in ids)
    return y, arrays, logits, predictions


def bootstrap_auroc(y, scores, repeats, seed):
    """Multinomial multiplicities are equivalent to resampling patients with replacement.
    Retain identical patient multiplicities across every K and seed in a replicate.
    AUC is the fraction of concordant positive-negative pairs, with half-credit ties.
    """
    import numpy as np
    y = np.asarray(y)
    require(y.ndim == 1 and set(y) == {0, 1} and repeats > 0, 'Invalid bootstrap inputs.')
    npos, nneg = int((y == 1).sum()), int((y == 0).sum())
    pairs = {}
    for key, p in scores.items():
        p = np.asarray(p)
        require(p.shape == y.shape and np.isfinite(p).all(), 'Bootstrap alignment/score failure.')
        pos, neg = p[y == 1], p[y == 0]
        pairs[key] = (pos[:, None] > neg).astype(float) + .5 * (pos[:, None] == neg)
    rng = np.random.Generator(np.random.PCG64(seed))
    # Generate once, reuse across all nine models. About 15 MB for this dataset.
    wp = rng.multinomial(npos, np.full(npos, 1 / npos), size=repeats).astype(float)
    wn = rng.multinomial(nneg, np.full(nneg, 1 / nneg), size=repeats).astype(float)
    return {key: np.einsum('bi,ij,bj->b', wp, a, wn, optimize=True) / (npos * nneg)
            for key, a in pairs.items()}


def resource_rows(ctx):
    rows = []
    for k, seed in MATRIX:
        name = run_name(k, seed); path = ctx['source'] / 'runs' / name
        summary = read_json(path / 'run_summary.json'); r = read_json(path / 'resource_summary.json')
        epochs = [x for x in read_csv(path / 'resources_by_epoch.csv') if int(x['epoch']) > 0]
        n = summary['completed_epochs']
        require(len(epochs) == n and [int(x['epoch']) for x in epochs] == list(range(1, n + 1)), 'Resource epoch mismatch.')
        total = sum(float(x['train_wall_seconds']) for x in epochs)
        require(abs(total - r['total_train_wall_seconds']) < 1e-5, 'Resource total mismatch.')
        rows.append(dict(budget=k, seed=seed, trained_epochs=n, selected_epoch=ctx['models'][name]['epoch'],
                         train_wall_seconds=total, train_seconds_per_epoch=total / n,
                         full_training_session_seconds=r['active_seconds_training_sessions'],
                         time_is_lower_bound=r['active_seconds_is_lower_bound'],
                         train_peak_allocated_GiB=max(float(x['train_peak_allocated_bytes']) for x in epochs) / 2**30,
                         train_peak_reserved_GiB=max(float(x['train_peak_reserved_bytes']) for x in epochs) / 2**30,
                         train_sampled_peak_RSS_GiB=max(float(x['train_sampled_peak_rss_bytes']) for x in epochs) / 2**30,
                         replayed_train_attempts=r['total_train_attempts_replayed'],
                         unmeasured_checkpoints=r['total_checkpoints_unmeasured']))
    return rows


def summarize(rows, fields):
    result = []
    for k in (64, 32, 16):
        selected = [r for r in rows if r['budget'] == k]
        require(len(selected) == 3 and {r['seed'] for r in selected} == set(SEEDS), 'Incomplete seed group.')
        row = dict(budget=k, seeds=3)
        for field in fields:
            values = [r[field] for r in selected]
            row[field + '_mean'] = statistics.mean(values)
            row[field + '_sd'] = statistics.stdev(values)
        result.append(row)
    return result


def analyze(ctx):
    import numpy as np
    verify_smoke(ctx)
    y, arrays, logits, predictions = complete_test_matrix(ctx)
    rows = [dict(budget=k, seed=s, selected_epoch=ctx['models'][run_name(k, s)]['epoch'],
                 **metrics(y, arrays[(k, s)], logits[(k, s)])) for k, s in MATRIX]
    grouped = summarize(rows, PLAN['descriptive_metrics'])
    bp = PLAN['bootstrap']
    print(f'Complete test matrix verified: 9 x {len(y)}. Computing {bp["replicates"]} paired bootstrap replicates...', flush=True)
    boot = bootstrap_auroc(y, arrays, bp['replicates'], bp['seed'])
    actual = {(r['budget'], r['seed']): r['auroc'] for r in rows}
    comparisons, distributions = [], {}
    for k, role in ((16, 'primary'), (32, 'secondary_exploratory')):
        delta = np.mean([boot[(k, s)] - boot[(64, s)] for s in SEEDS], axis=0)
        lower, upper = np.quantile(delta, [.025, .975], method='linear')
        distributions[f'K{k}_minus_K64'] = delta
        row = dict(comparison=f'K{k}-K64', role=role,
                   mean_paired_auroc_difference=statistics.mean(actual[(k, s)] - actual[(64, s)] for s in SEEDS),
                   ci95_lower=float(lower), ci95_upper=float(upper), patient_n=len(y), fixed_training_seeds=3,
                   bootstrap_replicates=bp['replicates'])
        for s in SEEDS:
            row[f'difference_seed{s}'] = actual[(k, s)] - actual[(64, s)]
        comparisons.append(row)
    for row in grouped:
        dist = np.mean([boot[(row['budget'], s)] for s in SEEDS], axis=0)
        lo, hi = np.quantile(dist, [.025, .975], method='linear')
        row.update(auroc_mean_patient_bootstrap_ci95_lower=float(lo), auroc_mean_patient_bootstrap_ci95_upper=float(hi))
    resources = resource_rows(ctx)
    resource_grouped = summarize(resources, ['trained_epochs', 'train_wall_seconds', 'train_seconds_per_epoch',
                                             'full_training_session_seconds', 'train_peak_allocated_GiB',
                                             'train_peak_reserved_GiB', 'train_sampled_peak_RSS_GiB'])
    base = next(r for r in resource_grouped if r['budget'] == 64)
    for r in resource_grouped:
        for f in ('train_seconds_per_epoch', 'full_training_session_seconds', 'train_peak_allocated_GiB'):
            r[f + '_saving_percent_ratio_of_means_vs_K64'] = 100 * (1 - r[f + '_mean'] / base[f + '_mean'])
        r['any_time_lower_bound'] = any(x['time_is_lower_bound'] for x in resources if x['budget'] == r['budget'])
    out = ctx['output']
    outputs = {
        'test_predictions.csv': predictions, 'metrics_by_run.csv': rows, 'metrics_by_budget.csv': grouped,
        'paired_auroc_comparisons.csv': comparisons, 'training_resources_by_run.csv': resources,
        'training_resources_by_budget.csv': resource_grouped,
    }
    for filename, values in outputs.items():
        table(out / filename, values)
    buf = io.BytesIO()
    np.savez_compressed(buf, **distributions)
    atomic_bytes(out / 'paired_auroc_bootstrap.npz', buf.getvalue())
    analysis = dict(identity=ctx['identity'], status='COMPLETE', plan=PLAN, metrics_by_run=rows,
                    metrics_by_budget=grouped, comparisons=comparisons, resources_by_budget=resource_grouped)
    atomic_json(out / 'analysis.json', analysis)
    report = ['# Stage 32: internal test evaluation', '',
              'All nine previously selected checkpoints; the same 186 test patients and 64 patches per evaluation.',
              'Selection used validation CE. No test-based selection or threshold tuning. LUAD=0, LUSC=1; threshold=0.5.', '',
              '| Training K | Test AUROC mean ± seed SD | BA mean ± seed SD | CE mean |',
              '|---|---|---|---|']
    for r in grouped:
        report.append(f'| {r["budget"]} | {r["auroc_mean"]:.4f} ± {r["auroc_sd"]:.4f} | '
                      f'{r["balanced_accuracy_mean"]:.4f} ± {r["balanced_accuracy_sd"]:.4f} | {r["ce_mean"]:.4f} |')
    report += ['', '## Paired differences', '']
    for r in comparisons:
        report.append(f'- {r["comparison"]} ({r["role"]}): ΔAUROC={r["mean_paired_auroc_difference"]:.4f}; '
                      f'95% patient-bootstrap CI [{r["ci95_lower"]:.4f}, {r["ci95_upper"]:.4f}].')
    report += ['', 'Intervals condition on these three fitted seeds. They do not quantify the population of training seeds. '
               'Secondary intervals are pointwise exploratory. A CI containing zero does not establish equivalence or noninferiority.', '',
               '## Measured training resources', '',
               '| K | Training seconds/epoch (mean) | Full training session minutes (mean) | Peak allocated training GPU GiB (mean) |',
               '|---|---|---|---|']
    for r in resource_grouped:
        report.append(f'| {r["budget"]} | {r["train_seconds_per_epoch_mean"]:.2f} | '
                      f'{r["full_training_session_seconds_mean"]/60:.2f} | {r["train_peak_allocated_GiB_mean"]:.3f} |')
    report += ['', 'Full training cost includes epochs after the selected epoch until the fixed stopping rule. '
               'Source lower-bound and replay flags are preserved in CSVs. GPU allocated, GPU reserved and process RSS are separate metrics. '
               'All evaluation budgets are 64: these results do not demonstrate inference-time savings or energy savings.', '',
               '## Scope and limitations', ''] + ['- ' + x for x in PLAN['limitations']]
    atomic_bytes(out / 'REPORT.md', ('\n'.join(report) + '\n').encode('utf-8'))
    artifact_names = list(outputs) + ['paired_auroc_bootstrap.npz', 'analysis.json', 'REPORT.md']
    atomic_json(out / 'completion.json', dict(status='COMPLETE', identity=ctx['identity'],
                test_patients=186, models=9, prediction_rows=1674,
                files_sha256={f: sha(out / f) for f in artifact_names}))
    print(f'COMPLETE: {out}\nSee REPORT.md, metrics_by_budget.csv and paired_auroc_comparisons.csv.')


def self_test():
    import numpy as np
    import unittest
    class Tests(unittest.TestCase):
        def test_auc_and_confusion(self):
            m = metrics([0, 0, 1, 1], [.1, .2, .8, .9], [-2., -1., 1., 2.])
            self.assertEqual((m['auroc'], m['balanced_accuracy'], m['macro_f1']), (1., 1., 1.))
            self.assertEqual((m['tn'], m['tp'], m['fp'], m['fn']), (2, 2, 0, 0))
        def test_ties_and_threshold(self):
            m = metrics([0, 1], [.5, .5], [0., 0.])
            self.assertEqual(m['auroc'], .5); self.assertEqual(m['balanced_accuracy'], .5)
            self.assertEqual(m['tp'], 1); self.assertAlmostEqual(m['ce'], math.log(2))
        def test_extreme_ce(self):
            self.assertTrue(math.isfinite(metrics([0, 1], [1., 0.], [1000., -1000.])['ce']))
        def test_invalid_scores(self):
            with self.assertRaises(AuditError): metrics([0, 1], [.2, float('nan')], [0, 0])
        def test_missing_class(self):
            with self.assertRaises(AuditError): metrics([1, 1], [.2, .3], [0, 0])
        def test_patient_alignment(self):
            r = dict(case_submitter_id='a', label=0, p_LUSC=.5, logit_diff_LUSC_minus_LUAD=0.)
            with self.assertRaises(AuditError): prediction_map([r, r], ['a', 'b'], {'a': 0, 'b': 1})
            with self.assertRaises(AuditError): prediction_map([r], ['b'], {'a': 0, 'b': 1})
            with self.assertRaises(AuditError): prediction_map([r], ['a'], {'a': 1})
        def test_probability_logit_check(self):
            r = dict(case_submitter_id='a', label=0, p_LUSC=.8, logit_diff_LUSC_minus_LUAD=0.)
            with self.assertRaises(AuditError): prediction_map([r], ['a'], {'a': 0})
        def test_bootstrap_pairing(self):
            y = np.array([0, 0, 1, 1]); p = np.array([.1, .7, .4, .9])
            b = bootstrap_auroc(y, {'a': p, 'b': p.copy()}, 100, 17)
            np.testing.assert_array_equal(b['a'], b['b'])
            np.testing.assert_array_equal(b['a'], bootstrap_auroc(y, {'a': p}, 100, 17)['a'])
        def test_bootstrap_exact_pair_count(self):
            y = np.array([0, 0, 1, 1]); p = np.array([.5, .8, .5, .9]); count = 25
            b = bootstrap_auroc(y, {'a': p}, count, 17)['a']
            rng = np.random.Generator(np.random.PCG64(17))
            wp = rng.multinomial(2, [.5, .5], size=count)
            wn = rng.multinomial(2, [.5, .5], size=count)
            for i in range(count):
                pos = np.repeat(p[y == 1], wp[i]); neg = np.repeat(p[y == 0], wn[i])
                expected = sum((a > c) + .5 * (a == c) for a in pos for c in neg) / 4
                self.assertEqual(b[i], expected)
        def test_split_guard_before_io(self):
            ctx = dict(ids={'validation': ['v'], 'test': ['t']}, bundle={'patients': {'v': {'split': 'validation'}, 't': {'split': 'test'}}})
            with self.assertRaises(AuditError): SplitReader(ctx, 'test')
            reader = SplitReader(ctx, 'validation')
            with self.assertRaises(AuditError): reader.load('t')
            with self.assertRaises(AuditError): SplitReader(ctx, 'train')
        def test_resume_and_corruption(self):
            with tempfile.TemporaryDirectory() as tmp:
                ctx = dict(output=Path(tmp), identity='one', ids={'test': ['a'], 'validation': []},
                           labels={'a': 0}, models={'m': {'sha256': 'model'}})
                self.assertIsNone(read_record(ctx, 'test', 'm', 'a'))
                r = commit_record(ctx, 'test', 'm', 'a', .5, 0.)
                self.assertEqual(r, read_record(ctx, 'test', 'm', 'a'))
                with self.assertRaises(AuditError): commit_record(ctx, 'test', 'm', 'a', .5, 0.)
                ctx['identity'] = 'changed'
                with self.assertRaises(AuditError): read_record(ctx, 'test', 'm', 'a')
                ctx['identity'] = 'one'
                path = record_path(ctx, 'test', 'm', 'a')
                e = read_json(path); e['record']['p_LUSC'] = .7; atomic_json(path, e)
                with self.assertRaises(AuditError): read_record(ctx, 'test', 'm', 'a')
        def test_no_partial_analysis(self):
            ctx = dict(output=Path('/nonexistent-stage32-test'), identity='x', ids={'test': ['a']}, labels={'a': 0},
                       models={run_name(k, s): {'sha256': 'x'} for k, s in MATRIX})
            with self.assertRaises(AuditError): complete_test_matrix(ctx)
        def test_strict_smoke_gate(self):
            with tempfile.TemporaryDirectory() as tmp:
                with self.assertRaises(AuditError): verify_smoke({'output': Path(tmp)})
        def test_smoke_tolerance(self):
            row = dict(case_submitter_id='v', p_LUSC=.5, logit_diff_LUSC_minus_LUAD=0.)
            ctx = dict(refs={'m': {'v': row.copy()}})
            self.assertEqual(compare_validation(ctx, 'm', row), (0., 0.))
            with self.assertRaises(AuditError): compare_validation(ctx, 'm', dict(row, p_LUSC=.51))
        def test_protocol_identity(self):
            self.assertNotEqual(digest({'code': 'a', 'plan': PLAN}), digest({'code': 'b', 'plan': PLAN}))
        def test_atomic_failed_replace_preserves_previous(self):
            from unittest.mock import patch
            with tempfile.TemporaryDirectory() as tmp:
                p = Path(tmp) / 'metrics.csv'; atomic_bytes(p, b'old')
                with patch('os.replace', side_effect=PermissionError('Excel')):
                    with self.assertRaises(PermissionError): atomic_bytes(p, b'new')
                self.assertEqual(p.read_bytes(), b'old')
                self.assertEqual(len(list(Path(tmp).iterdir())), 1)
        def test_seed_sd(self):
            rows = [dict(budget=k, seed=s, x=i) for k in BUDGETS for i, s in enumerate(SEEDS)]
            for r in summarize(rows, ['x']):
                self.assertEqual(r['x_mean'], 1); self.assertEqual(r['x_sd'], 1)
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(Tests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    require(result.wasSuccessful(), 'CPU SELF-TEST FAILED')
    print(f'CPU SELF-TEST PASS: {result.testsRun}; no project data, CUDA or test PNGs used.')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    actions = parser.add_mutually_exclusive_group(required=True)
    for flag in ('self-test', 'preflight-only', 'prepare', 'smoke', 'run', 'analyze'):
        actions.add_argument('--' + flag, action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test(); return 0
    print(f'STAGE 32 v{VERSION}: fixed 9 models; no training; test access only with --run.', flush=True)
    ctx = context(check_environment=not args.analyze)
    if args.preflight_only:
        if ctx['output'].exists(): verify_prepared(ctx)
        print('PREFLIGHT PASS: frozen sources, nine model hashes, selection rules and validation references verified.\n'
              'No PNGs opened, no inference or output writes. Next: --prepare, then --smoke.')
        return 0
    if args.prepare:
        prepare(ctx); return 0
    verify_prepared(ctx)
    with ctx['s29'].execution_lock(ctx['output'].parent / (OUTPUT_NAME + '.lock')):
        if args.smoke:
            evaluate(ctx, 'validation')
        elif args.run:
            evaluate(ctx, 'test')
        elif args.analyze:
            analyze(ctx)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print('\nINTERRUPTED: repeat the same command; committed predictions remain intact.', file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f'ERROR: {type(exc).__name__}: {exc}\nNo source data/models were modified. '
              'If CSV is locked, close Excel and repeat the command. Otherwise retain files for review.', file=sys.stderr)
        raise SystemExit(1)
