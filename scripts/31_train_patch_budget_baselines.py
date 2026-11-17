#!/usr/bin/env python3
r"""Stage 31 v1.0.1: full-train patch-budget baselines (K=16/32/64; seeds 20261113/4/5) -- runner + checks.

v1.0.1 changes ONLY resource accounting (v1.0.0 archived in results\stage31_v1_0_0_archive): actual work cost is
written to an append-only, fsync'd resource journal (resource_journal.jsonl) as each measured block finishes, and
all resource tables are rebuilt from that journal; the checkpoint holds only logical training state. Training
numerics (model, optimizer, AMP, BN, order, patches, augmentation, stopping, selection) are unchanged.

Prepares a correct, resumable run on the frozen Stage 28 split (557 train / 186 validation; 186 test CLOSED).
It does NOT claim that model quality is fixed; quality is to be measured by the runs.

Configuration (selected after results\stage29_learning_investigation_v1 D01-D08 and Stage 30):
  ResNet18 (official ImageNet weights already prepared by Stage 29), trainable encoder, gated attention MIL,
  2-logit classifier with the ORIGINAL head initialisation; encoder LR 1e-5, head LR 1e-4, AdamW wd 1e-4;
  frozen BN running stats (affine trainable); AMP FP16 encoder, FP32 attention/pool/head; one patient per
  forward; gradient accumulation over 4 patients (last partial group kept, loss / actual group size);
  Stage 29 'original' deterministic patient shuffling, D4 augmentation, ImageNet normalisation, nested patch
  selection; validation = all 64 patches, no augmentation; 4 CPU threads, workers=0, deterministic algorithms.
  Selection: max 30 epochs; criterion mean patient-level validation CE (186); improvement = strictly lower CE;
  patience 5; ties keep the earlier checkpoint; selected model = minimal validation CE. Threshold 0.5.

The frozen Stage 29 module is imported read-only after its SHA256 is verified, for: Stage 28 bundle validation,
weights verification, environment record, deterministic setup, model construction, optimizer/scaler factories,
SHA-keyed patient order / patch selection / augmentation (with the RUN seed passed explicitly), metrics,
atomic checkpoints and process lock (v1.0.1 replaces the Stage 29 session ledger with its own resource journal). Stage 29's pilot-only loader is NOT used or patched; this
script has its own loader restricted to the frozen train/validation IDs (test IDs are refused before any read).
Ported (not imported) with line references to Stage 29 v1.0.0: train_group (L502-550), load_bag transform
(L420-438), validate_model (L854-874); equivalence is verified numerically against D08-B in --smoke.

Commands (from C:\research\mm_oncology_efficiency):
  .\.venv\Scripts\python.exe -B scripts\31_train_patch_budget_baselines.py --self-test
  .\.venv\Scripts\python.exe -B scripts\31_train_patch_budget_baselines.py --preflight-only
  .\.venv\Scripts\python.exe -B scripts\31_train_patch_budget_baselines.py --smoke
  .\.venv\Scripts\python.exe -B scripts\31_train_patch_budget_baselines.py --prepare
  .\.venv\Scripts\python.exe -B scripts\31_train_patch_budget_baselines.py --run --budget 64 --seed 20261113
Resume after interruption / power loss: repeat exactly the same --run command.
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True
import argparse
import collections
import contextlib
import csv
import hashlib
import importlib.util
import io
import json
import math
import os
import random
import shutil
import subprocess
import time
import uuid
from pathlib import Path

PROCESS_T0, PROCESS_C0 = time.perf_counter(), time.process_time()  # session accounting starts at script start
VERSION = "1.0.1"
ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = {"stage28": ROOT / "scripts" / "28_prepare_patch_budget_experiment.py",
           "stage29": ROOT / "scripts" / "29_train_patch_budget_baselines.py",
           "stage30": ROOT / "scripts" / "30_diagnose_balanced_patient_batches.py"}
S29_SHA256 = "a8a7409a32c748c3529544138b5e20b8f6294b233a8fb22c20585dcd522ab09d"
MAIN_OUTPUT = ROOT / "results" / "stage31_patch_budget_v1_1"
SMOKE_OUTPUT = ROOT / "results" / "stage31_patch_budget_v1_1_smoke"
D08B = ROOT / "results" / "stage29_learning_investigation_v1" / "d08_head_lr_B_B64_20ep"
BUDGETS = (16, 32, 64)
SEEDS = (20261113, 20261114, 20261115)
EXPECTED_COUNTS = {("train", "LUAD"): 273, ("train", "LUSC"): 284, ("validation", "LUAD"): 91,
                   ("validation", "LUSC"): 95, ("test", "LUAD"): 91, ("test", "LUSC"): 95}
SETTINGS = {
    "architecture": "torchvision ResNet18 (fc=Identity) + gated attention (V/U 512->128, w 128->1) + Linear(512,2)",
    "weights": "official ImageNet ResNet18 resnet18-f37072fd.pth prepared and verified by Stage 29",
    "head_init": "original (Stage 29 build_model with the run seed); not zero",
    "encoder_trainable": True, "encoder_lr": 1e-5, "head_lr": 1e-4, "optimizer": "AdamW foreach=False (Stage 29 optimizer_for)",
    "weight_decay": 1e-4, "betas_eps_amsgrad": "torch defaults (as D08-B)",
    "batchnorm": "frozen running statistics, trainable affine", "precision": "AMP FP16 encoder; FP32 attention/pool/classifier",
    "grad_scaler": "torch.amp.GradScaler('cuda', init_scale=1024) (Stage 29 scaler_for)",
    "patients_per_forward": 1, "accumulation_patients": 4,
    "partial_last_group": "kept; loss divided by actual number of patients in the group; no drop, no duplication",
    "patient_order": "Stage 29 patient_order(ids, epoch, seed) ('original'); identical across K for a seed",
    "patch_selection": "Stage 29 patch_indices(pid, epoch, K, seed): nested K16 c K32 c K64",
    "augmentation": "Stage 29 augmentation_code(pid, epoch, tile, seed): D4, identical across K for shared patches",
    "normalisation": "ImageNet mean/std after /255; no crop", "validation": "all 64 patches, no augmentation",
    "cpu_threads": 4, "interop_threads": 1, "workers": 0, "deterministic": "Stage 29 initialize_torch",
    "scheduler": None, "grad_clipping": None, "sampler": "none (all patients once per epoch)", "threshold": 0.5,
    "checkpoint_every_attempts": 5, "checkpoint_phase_boundaries": ["train_done", "validation_done", "epoch_committed"],
    "max_consecutive_amp_skips": 3, "amp_skip_policy": "after 3 consecutive skipped steps: save state, stop with status AMP_SKIPS, no auto-resume",
    "train_eval_subset": "10 LUAD + 10 LUSC train patients, first by SHA256('stage31-train-eval-v1|'+ID) within class",
    "diagnostics_frequency": "once per epoch (and epoch 0 for the train-eval subset); identical for all K",
    "step_update_norms": "not computed in main runs (no per-step parameter copies); epoch-level weight change only",
    "resource_accounting": "v1.0.1: append-only fsync'd resource_journal.jsonl (one event per measured block: setup, each "
                           "train attempt, validation, diagnostics, each checkpoint write, auxiliary I/O); tables rebuilt "
                           "from the journal; replayed work counted as actual cost; unknown crash tails reported as lower bounds",
}
STOPPING = {"max_epochs": 30, "criterion": "mean patient-level validation cross-entropy over all validation patients (186)",
            "improvement": "strictly lower CE (min_delta = 0)", "patience_epochs": 5,
            "tie_rule": "equal CE keeps the earlier best checkpoint",
            "selected_model": "checkpoint with minimal validation CE (best_model.pt; authoritative copy = best checkpoint generation)",
            "no_threshold_tuning": True}
RUN_MATRIX = {"budgets": list(BUDGETS), "seeds": list(SEEDS), "launch": "each run explicitly by --budget/--seed; never automatically",
              "first_planned": {"budget": 64, "seed": 20261113}}
NOTES = {"validation_reuse": "20 pilot-validation patients (Stage 28 pilot_manifest) were used during Stage 29/30 tuning; "
                             "validation is for model selection, NOT an independent final estimate.",
         "test": "186 test patients remain closed: metadata only (counts/labels/split); no PNG opened, no features, no predictions; no test CLI.",
         "not_a_quality_fix": "Configuration chosen from 60-patient diagnostics; quality on the full train set is to be measured."}

s29 = torch = np = None


# --------------------------------------------------------------------------------------------- utilities
def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(1 << 20), b""):
            h.update(part)
    return h.hexdigest()


def load_stage29():
    global s29
    if s29 is None:
        if sha_file(SCRIPTS["stage29"]) != S29_SHA256:
            raise SystemExit("ERROR: Stage 29 script hash differs from the verified version; refusing to run.")
        spec = importlib.util.spec_from_file_location("stage29_frozen", SCRIPTS["stage29"])
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        s29 = mod
    return s29


def require(cond, msg):
    if not cond:
        raise load_stage29().ExperimentError(msg)


def init_torch(cuda):
    global torch, np
    device = s29.initialize_torch(cuda=cuda)
    torch, np = s29.torch, s29.np
    return device


def write_table(path, rows):
    """Atomic CSV from in-memory (checkpointed) rows. A file locked by Excel only produces a warning."""
    if not rows:
        return True
    keys = list(dict.fromkeys(k for r in rows for k in r))
    text = io.StringIO(newline="")
    w = csv.DictWriter(text, fieldnames=keys)
    w.writeheader()
    w.writerows([{k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in r.items()} for r in rows])
    try:
        s29.atomic_bytes(path, text.getvalue().encode("utf-8"))
        return True
    except PermissionError:
        print(f"WARNING: {path} is locked (Excel?). Close it; tables are rebuilt from the checkpoint on the next commit/run.", flush=True)
        return False


def read_rows(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


# --------------------------------------------------------------------------------------------- data access
class Data:
    """Frozen Stage 28 split; only explicitly allowed train/validation IDs can be loaded. Test is never readable."""

    def __init__(self, bundle, train_ids, val_ids):
        pats = bundle["patients"]
        self.bundle, self.patches = bundle, bundle["patches"]
        self.labels = {p: r["class_label"] for p, r in pats.items()}
        require(all(pats[p]["split"] == "train" for p in train_ids), "Non-train patient in train list.")
        require(all(pats[p]["split"] == "validation" for p in val_ids), "Non-validation patient in validation list.")
        require(not set(train_ids) & set(val_ids), "Train/validation overlap.")
        require(len(set(train_ids)) == len(train_ids) and len(set(val_ids)) == len(val_ids), "Duplicate IDs.")
        self.allowed = {"train": frozenset(train_ids), "validation": frozenset(val_ids)}
        self.train_ids, self.val_ids = sorted(train_ids), sorted(val_ids)

    def label(self, pid):
        return 0 if self.labels[pid] == "LUAD" else 1

    def read_png(self, pid, row):
        require(self.bundle["patients"][pid]["split"] != "test", f"Refusing to read a TEST PNG: {pid}")
        require(pid in self.allowed["train"] or pid in self.allowed["validation"], f"Patient not allowed in this run: {pid}")
        return s29.verified_png_bytes(row)

    def load_bag(self, pid, part, epoch, budget, seed, augment, device):
        """Port of Stage 29 load_bag (L414-438) with explicit seed and split authorisation."""
        from PIL import Image
        require(part in ("train", "validation"), "Only train/validation can be loaded.")
        require(self.bundle["patients"][pid]["split"] == part and pid in self.allowed[part], f"{pid} not allowed as {part}.")
        if augment:
            require(part == "train", "Augmentation only for training bags.")
            indices = s29.patch_indices(pid, epoch, budget, seed=seed)
        else:
            require(budget == 64, "Non-augmented evaluation uses all 64 patches.")
            indices = list(range(64))
        images = []
        for i in indices:
            row = self.patches[pid][i]
            raw = self.read_png(pid, row)
            with Image.open(io.BytesIO(raw)) as im:
                require(im.format == "PNG" and im.mode == "RGB" and im.size == (256, 256), "Unexpected PNG format.")
                im.load()
                array = np.asarray(im)
            if augment:
                code = s29.augmentation_code(pid, epoch, row["index"], seed=seed)
                array = np.rot90(array, code % 4)
                if code // 4:
                    array = np.flip(array, axis=1)
            images.append(torch.from_numpy(np.array(array, copy=True, order="C")).permute(2, 0, 1))
        x = torch.stack(images).float().div_(255.)
        x.sub_(torch.tensor([.485, .456, .406]).view(1, 3, 1, 1))
        x.div_(torch.tensor([.229, .224, .225]).view(1, 3, 1, 1))
        return x.to(device), torch.tensor([self.label(pid)], device=device, dtype=torch.long)


def select_train_eval(train_ids, labels, per_class):
    out = []
    for c in ("LUAD", "LUSC"):
        ids = sorted((p for p in train_ids if labels[p] == c),
                     key=lambda p: (hashlib.sha256(f"stage31-train-eval-v1|{p}".encode("ascii")).hexdigest(), p))
        require(len(ids) >= per_class, "Not enough patients for the train-eval subset.")
        out += ids[:per_class]
    return out


def make_groups(train_ids, epoch, seed, accumulation=4):
    order = s29.patient_order(train_ids, epoch, seed=seed)
    groups = [order[a:b] for a, b in s29.group_slices(len(order), 0, accumulation)]
    flat = [p for g in groups for p in g]
    require(sorted(flat) == sorted(train_ids) and len(flat) == len(set(flat)), "Every train patient exactly once per epoch.")
    return groups


# --------------------------------------------------------------------------------------------- model steps
def train_group(model, optimizer, scaler, ids, loader, device, step_norms=False):
    """One update ATTEMPT on one accumulation group (port of Stage 29 L502-550).
    Returns attempted/executed/skipped explicitly; update norms only when requested AND executed (else None)."""
    require(1 <= len(ids) <= 4, "Invalid accumulation group.")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss_sum, patches = 0., 0
    for pid in ids:
        x, target = loader(pid)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(logits, target)
        require(bool(torch.isfinite(loss).item()), f"Non-finite loss for {pid}")
        scaler.scale(loss / len(ids)).backward()
        loss_sum += float(loss.detach().item())
        patches += len(x)
        del x, target, logits, loss
    scaler.unscale_(optimizer)
    grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    require(grads, "No model gradients.")
    finite = bool(torch.stack([torch.isfinite(g).all() for g in grads]).all().item())
    before = None
    if step_norms:
        before = [p.detach().clone() for p in model.parameters()]
    old_scale = float(scaler.get_scale())
    scaler.step(optimizer)
    scaler.update()
    new_scale = float(scaler.get_scale())
    skipped = int(new_scale < old_scale)
    require(finite or skipped, "Non-finite gradient reached the optimizer without an AMP skip.")
    enc_upd = head_upd = None
    if step_norms and not skipped:
        names = [n for n, _ in model.named_parameters()]
        with torch.no_grad():
            d = [(n, (p - q).float()) for n, p, q in zip(names, model.parameters(), before)]
            enc_upd = math.sqrt(sum(float((t ** 2).sum()) for n, t in d if n.startswith("encoder.")))
            head_upd = math.sqrt(sum(float((t ** 2).sum()) for n, t in d if not n.startswith("encoder.")))
    del before
    optimizer.zero_grad(set_to_none=True)
    return {"patients": len(ids), "patches": patches, "loss_sum": loss_sum, "attempted": 1, "executed": 1 - skipped,
            "skipped": skipped, "finite_gradients": finite, "scale_before": old_scale, "scale_after": new_scale,
            "enc_update_norm": enc_upd, "head_update_norm": head_upd}


def forward_pool(model, x, device):
    """Same operations as Stage 29 GatedMIL.forward (L389-397), additionally returning the pooled vector."""
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        h = model.encoder(x)
    h = h.float()
    scores = model.attention_w(torch.tanh(model.attention_v(h)) * torch.sigmoid(model.attention_u(h)))
    attention = torch.softmax(scores.squeeze(-1), dim=0)
    pooled = torch.sum(attention[:, None] * h, dim=0, keepdim=True)
    return model.classifier(pooled), pooled[0]


def evaluate(model, data, ids, part, device, epoch_label):
    """Port of Stage 29 validate_model (L854-874): mean patient CE, p_LUSC = softmax[:,1]; plus pooled-feature stats."""
    model.eval()
    preds, ces, Z = [], [], []
    with torch.inference_mode():
        for pid in ids:
            x, target = data.load_bag(pid, part, 0, 64, 0, False, device)
            logits, z = forward_pool(model, x, device)
            loss = torch.nn.functional.cross_entropy(logits, target)
            p = float(torch.softmax(logits.float(), dim=1)[0, 1].item())
            require(math.isfinite(float(loss.item())) and math.isfinite(p), "Non-finite evaluation output.")
            ces.append(float(loss.item()))
            preds.append({"epoch": epoch_label, "case_submitter_id": pid, "label": int(target.item()), "p_LUSC": p,
                          "logit_diff_LUSC_minus_LUAD": float(logits[0, 1] - logits[0, 0])})
            Z.append(z.cpu().numpy())
            del x, target, logits
    model.train()
    y = np.array([r["label"] for r in preds]); pr = np.array([r["p_LUSC"] for r in preds])
    ld = np.array([r["logit_diff_LUSC_minus_LUAD"] for r in preds]); Z = np.stack(Z)
    m = s29.binary_metrics([int(v) for v in y], [float(v) for v in pr])  # python types: checkpoints load with weights_only=True
    out = {"ce": float(np.mean(ces)), **{k: float(v) for k, v in m.items()}, "p_mean": float(pr.mean()), "p_sd": float(pr.std()), "p_min": float(pr.min()),
           "p_max": float(pr.max()), "n_pred_LUAD": int((pr < .5).sum()), "n_pred_LUSC": int((pr >= .5).sum()),
           "mean_p_diff_LUSC_minus_LUAD": float(pr[y == 1].mean() - pr[y == 0].mean()),
           "logit_diff_mean": float(ld.mean()), "logit_diff_sd": float(ld.std()),
           "pooled_abs_mean": float(np.abs(Z).mean()), "pooled_between_patient_sd": float(Z.std(0).mean())}
    return native(out), native(preds)


def native(v):
    """Convert numpy scalars to python types (checkpoint state must load with torch.load(weights_only=True))."""
    if isinstance(v, dict):
        return {k: native(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return type(v)(native(x) for x in v)
    if np is not None and isinstance(v, np.generic):
        return v.item()
    return v


# --------------------------------------------------------------------------------------------- resource journal (v1.0.1)
JOURNAL_NAME = "resource_journal.jsonl"
MEASURED_KINDS = ("setup", "train_attempt", "validation", "diagnostics", "checkpoint_io", "aux_io")


class ResourceJournal:
    """Append-only, fsync'd JSONL of ACTUAL work, written as each measured block ends. It is never rewritten.
    Includes work that a crash later rolls back and that is executed again (replay). Logical training progress
    lives only in checkpoints; nothing here is summed into checkpoint state, so rebuilding tables cannot double count."""

    def __init__(self, path, clock=time.perf_counter, cpu_clock=time.process_time, t0=None, c0=None, session_id=None, utc=None):
        self.path, self.clock, self.cpu_clock = Path(path), clock, cpu_clock
        self.t0 = PROCESS_T0 if t0 is None else t0
        self.c0 = PROCESS_C0 if c0 is None else c0
        self.session_id, self.seq, self.utc = session_id or uuid.uuid4().hex, 0, utc or s29.now
        self.path.parent.mkdir(parents=True, exist_ok=True)
        repaired = False
        if self.path.exists() and self.path.stat().st_size:
            with self.path.open("rb") as f:          # a crash can leave a partial last line without '\n'
                f.seek(-1, os.SEEK_END)
                repaired = f.read(1) != b"\n"
        self.f = self.path.open("a", encoding="utf-8", newline="\n")
        if repaired:
            self.f.write("\n"); self._sync()
        self.closed = False
        self.event("session_start", repaired_truncated_tail=repaired, pid=os.getpid())

    def _sync(self):
        self.f.flush(); os.fsync(self.f.fileno())

    def event(self, kind, **fields):
        require(not self.closed, "Journal already closed.")
        self.seq += 1
        row = {"session_id": self.session_id, "seq": self.seq, "kind": kind, "utc": self.utc(),
               "active_seconds": self.clock() - self.t0, "active_cpu_seconds": self.cpu_clock() - self.c0, **fields}
        self.f.write(json.dumps(native(row), allow_nan=False) + "\n")
        self._sync()
        return row

    def close(self, outcome):
        self.event("session_end", outcome=outcome)
        self.f.close(); self.closed = True

    def abandon(self):
        """Emulated crash: no session_end, so the session tail stays unknown (lower bound)."""
        self.f.close(); self.closed = True


class Meter:
    """Measures one block (wall, CPU, CUDA peak allocated/reserved, sampled peak RSS) and journals it ONLY when the
    block completes; an exception inside the block (crash/stop) leaves no event (unknown tail)."""

    def __init__(self, journal, monitor_factory, clock=time.perf_counter, cpu_clock=time.process_time):
        self.journal, self.monitor_factory, self.clock, self.cpu_clock = journal, monitor_factory, clock, cpu_clock

    @contextlib.contextmanager
    def measure(self, kind, **tags):
        extra = {}
        w0, c0 = self.clock(), self.cpu_clock()
        with self.monitor_factory(kind) as mon:
            yield extra
            vals = mon.values()
        wall, cpu = self.clock() - w0, self.cpu_clock() - c0
        require(wall >= 0 and cpu >= 0, f"Negative duration in {kind}.")
        self.journal.event(kind, wall_seconds=wall, cpu_seconds=cpu,
                           peak_allocated_bytes=int(vals["peak_allocated_vram_bytes"]),
                           peak_reserved_bytes=int(vals["peak_reserved_vram_bytes"]),
                           sampled_peak_rss_bytes=int(vals["sampled_peak_rss_bytes"]), **tags, **extra)


def parse_journal(path):
    """Tolerant reader: skips unparsable (e.g. truncated) lines and duplicate (session_id, seq) keys."""
    events, bad, seen = [], 0, set()
    path = Path(path)
    if not path.exists():
        return events, bad
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                e = json.loads(line)
                key = (e["session_id"], e["seq"])
            except (ValueError, KeyError, TypeError):
                bad += 1
                continue
            if key not in seen:
                seen.add(key); events.append(e)
    return events, bad


def build_resource_tables(events, bad_lines):
    """Pure function of the journal: per-epoch actual cost, per-session accounting, run summary."""
    blocks = ("train", "validation", "diagnostics", "checkpoint_io", "aux_io")
    kind_block = {"train_attempt": "train", "validation": "validation", "diagnostics": "diagnostics",
                  "checkpoint_io": "checkpoint_io", "aux_io": "aux_io"}
    epochs = sorted({e["epoch"] for e in events if e.get("epoch") is not None and e["kind"] in kind_block or
                     (e["kind"] == "checkpoint_io_unmeasured" and e.get("epoch") is not None)})
    measured_seq = {e["checkpoint_seq"] for e in events if e["kind"] == "checkpoint_io"}
    rows = []
    for ep in epochs:
        evs = [e for e in events if e.get("epoch") == ep]
        row = {"epoch": ep}
        for b in blocks:
            be = [e for e in evs if kind_block.get(e["kind"]) == b]
            row[f"{b}_wall_seconds"] = sum(e["wall_seconds"] for e in be)
            row[f"{b}_cpu_seconds"] = sum(e["cpu_seconds"] for e in be)
            row[f"{b}_peak_allocated_bytes"] = max((e["peak_allocated_bytes"] for e in be), default=0)
            row[f"{b}_peak_reserved_bytes"] = max((e["peak_reserved_bytes"] for e in be), default=0)
            row[f"{b}_sampled_peak_rss_bytes"] = max((e["sampled_peak_rss_bytes"] for e in be), default=0)
            row[f"{b}_blocks"] = len(be)
        att = [e for e in evs if e["kind"] == "train_attempt"]
        logical = {e["attempt_total"] for e in att}
        row["train_attempts_executed_work"] = len(att)          # actual work incl. replays after rollback
        row["train_attempts_logical"] = len(logical)            # distinct logical attempts
        row["train_attempts_replayed"] = len(att) - len(logical)
        row["validation_replayed"] = max(0, row["validation_blocks"] - 1)
        row["diagnostics_replayed"] = max(0, row["diagnostics_blocks"] - 1)
        unmeasured = {e["checkpoint_seq"] for e in evs if e["kind"] == "checkpoint_io_unmeasured"} - measured_seq
        row["checkpoints_measured"] = row["checkpoint_io_blocks"]
        row["checkpoints_unmeasured"] = len(unmeasured)
        row["checkpoint_io_wall_is_lower_bound"] = bool(unmeasured)
        rows.append(row)
    sessions = []
    order = list(dict.fromkeys(e["session_id"] for e in events))
    for sid in order:
        se = [e for e in events if e["session_id"] == sid]
        end = [e for e in se if e["kind"] == "session_end"]
        ctx = [e for e in se if e["kind"] == "session_context"]
        comp = {b: sum(e["wall_seconds"] for e in se if kind_block.get(e["kind"]) == b) for b in blocks}
        setup = sum(e["wall_seconds"] for e in se if e["kind"] == "setup")
        active = max(e["active_seconds"] for e in se)
        sessions.append({"session_id": sid, "start_utc": se[0]["utc"], "last_event_utc": se[-1]["utc"], "events": len(se),
                         "clean_end": bool(end), "outcome": end[-1]["outcome"] if end else "UNKNOWN (no session_end: crash/kill/power loss)",
                         "active_seconds": active, "active_seconds_is_lower_bound": not end,
                         "active_cpu_seconds": max(e["active_cpu_seconds"] for e in se),
                         "setup_wall_seconds": setup, **{f"{b}_wall_seconds": comp[b] for b in blocks},
                         "other_wall_seconds": active - setup - sum(comp.values()),
                         "status_at_open": ctx[0].get("status_at_open") if ctx else None,
                         "post_completion_session": bool(ctx) and ctx[0].get("status_at_open") is not None,
                         "repaired_truncated_tail_at_start": any(e.get("repaired_truncated_tail") for e in se if e["kind"] == "session_start")})
    training = [s for s in sessions if not s["post_completion_session"]]
    post = [s for s in sessions if s["post_completion_session"]]
    summary = {"sessions": len(sessions), "unclean_sessions": [s["session_id"] for s in sessions if not s["clean_end"]],
               "journal_unparsable_lines": bad_lines,
               "active_seconds_training_sessions": sum(s["active_seconds"] for s in training),
               "active_seconds_post_completion_sessions": sum(s["active_seconds"] for s in post),
               "active_seconds_is_lower_bound": any(not s["clean_end"] for s in sessions) or bad_lines > 0,
               **{f"total_{b}_wall_seconds": sum(r[f"{b}_wall_seconds"] for r in rows) for b in blocks},
               "total_setup_wall_seconds": sum(s["setup_wall_seconds"] for s in sessions),
               "total_train_attempts_executed_work": sum(r["train_attempts_executed_work"] for r in rows),
               "total_train_attempts_replayed": sum(r["train_attempts_replayed"] for r in rows),
               "total_checkpoints_measured": sum(r["checkpoints_measured"] for r in rows),
               "total_checkpoints_unmeasured": sum(r["checkpoints_unmeasured"] for r in rows),
               "note": ("Actual work from resource_journal.jsonl (includes replay after rollback). Wall/CPU time, not 'GPU time'. "
                        "Sessions without session_end have an unknown tail: their active time is a lower bound. "
                        "Unmeasured checkpoints were written but their write time was not journaled before a crash.")}
    return rows, sessions, summary


# --------------------------------------------------------------------------------------------- selection rule
def new_selection():
    return {"best_ce": None, "best_epoch": None, "epochs_without_improvement": 0}


def update_selection(sel, epoch, ce):
    require(math.isfinite(ce), "Non-finite validation CE.")
    improved = sel["best_ce"] is None or ce < sel["best_ce"]
    if improved:
        sel.update(best_ce=ce, best_epoch=epoch, epochs_without_improvement=0)
    else:
        sel["epochs_without_improvement"] += 1
    return improved


def stop_reason(sel, epoch, max_epochs, patience):
    if sel["epochs_without_improvement"] >= patience:
        return "EARLY_STOP_PATIENCE"
    if epoch >= max_epochs:
        return "MAX_EPOCHS"
    return None


# --------------------------------------------------------------------------------------------- runner
class StopRequested(Exception):
    pass


class Runner:
    """Epoch loop with phase checkpoints. All callables injectable for CPU self-tests.
    v1.0.1: the checkpoint holds only LOGICAL state; every measured block of actual work goes to the resource journal
    (via self.meter) as soon as it completes. Epoch tags: 0 = initial diagnostics/checkpoint; k = work for epoch k
    (its training attempts, validation, diagnostics, its checkpoints incl. the commit, epoch-start/best exports)."""

    def __init__(self, *, run_dir, identity, model, optimizer, scaler, device, train_ids, labels, seed, budget,
                 max_epochs, patience, loader, validate_fn, train_eval_fn, meter, checkpoint_every=5, step_norms=False,
                 stop_after_attempts=None, stop_at_phase=None, crash_after_save_seq=None, max_skips=3):
        self.__dict__.update(locals()); del self.__dict__["self"]
        self.checkpoints = s29.Checkpoints(run_dir / "checkpoints", identity)
        self.init_params = [p.detach().clone() for p in model.parameters()]
        self.names = [n for n, _ in model.named_parameters()]

    # ---- helpers
    def _norms(self, ref):
        with torch.no_grad():
            d = [(n, (p.detach() - q.to(p.device)).float()) for n, p, q in zip(self.names, self.model.parameters(), ref)]
            enc = math.sqrt(sum(float((t ** 2).sum()) for n, t in d if n.startswith("encoder.")))
            head = math.sqrt(sum(float((t ** 2).sum()) for n, t in d if not n.startswith("encoder.")))
        return enc, head

    def _bias(self):
        b = self.model.classifier.bias.detach().double().cpu().tolist()
        return {"bias_LUAD": repr(b[0]), "bias_LUSC": repr(b[1]), "bias_diff_LUSC_minus_LUAD": repr(b[1] - b[0])}

    def _rng_fp(self):
        n = np.random.get_state()
        cuda = tuple(hashlib.sha256(s.numpy().tobytes()).hexdigest() for s in torch.cuda.get_rng_state_all()) \
            if self.device.type == "cuda" else ()
        return (hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(), cuda,
                hash(random.getstate()), hashlib.sha256(n[1].tobytes()).hexdigest() + str(n[2]))

    def _guarded(self, fn, *a):
        rng, bn = self._rng_fp(), s29.bn_state(self.model)
        out = fn(*a)
        self.model.train()
        require(self._rng_fp() == rng, "Diagnostic/evaluation changed RNG state.")
        require(s29.tree_equal(bn, s29.bn_state(self.model)), "Diagnostic/evaluation changed BN buffers.")
        return out

    def _save(self, epoch_tag, label, mark_best=False):
        """Checkpoint write measured as its own block; the event is journaled AFTER the checkpoint is durable, so a
        crash in between leaves a checkpoint without cost (detected and reported as unmeasured at the next start)."""
        converted = native(self.state)   # in place: the loop keeps a reference to self.state
        self.state.clear(); self.state.update(converted)
        self.state["_save_tag"] = {"epoch": epoch_tag, "label": label}
        with self.meter.measure("checkpoint_io", epoch=epoch_tag, label=label, phase=self.state["phase"],
                                group=self.state["group"], attempts_total=self.state["attempts_total"]) as ev:
            final = self.checkpoints.save(self.model, self.optimizer, self.scaler, self.state, mark_best=mark_best)
            ev["checkpoint_seq"] = int(final.name.split("_")[-1])
            if self.crash_after_save_seq is not None and ev["checkpoint_seq"] == self.crash_after_save_seq:
                raise StopRequested(f"emulated crash after writing checkpoint {ev['checkpoint_seq']} before its cost event")

    def _maybe_stop(self, kind, value):
        if kind == "attempt" and self.stop_after_attempts is not None and value == self.stop_after_attempts:
            raise StopRequested(f"after attempt {value}")
        if kind == "phase" and self.stop_at_phase is not None and value == self.stop_at_phase:
            raise StopRequested(f"at phase {value}")

    def _epoch_start_file(self):
        return self.run_dir / "epoch_start_params.pt"

    def _write_epoch_start(self, epoch):
        sha = s29.save_tensor_file(self._epoch_start_file(), {"epoch": epoch, "params": [p.detach().cpu().clone() for p in self.model.parameters()]})
        self.state["epoch_start_sha256"] = sha

    def _read_epoch_start(self, epoch):
        f = self._epoch_start_file()
        if not f.is_file() or sha_file(f) != self.state.get("epoch_start_sha256"):
            return None
        d = torch.load(f, map_location="cpu", weights_only=True)
        return d["params"] if d["epoch"] == epoch else None

    def _record_unmeasured_checkpoints(self):
        """A checkpoint whose write completed but whose cost event was not journaled (crash in between) is reported once
        as 'checkpoint_io_unmeasured' with its epoch tag: counted, cost unknown (never estimated)."""
        events, _ = parse_journal(self.meter.journal.path)
        known = {e["checkpoint_seq"] for e in events if e["kind"] in ("checkpoint_io", "checkpoint_io_unmeasured")}
        for path in self.checkpoints.generations():
            seq = int(path.name.split("_")[-1])
            if seq in known:
                continue
            tag = {"epoch": None, "label": "unreadable"}
            try:
                meta = s29.read_json(path / "commit.json")
                if meta["identity"] == self.identity and sha_file(path / "state.pt") == meta["sha256"]:
                    tag = torch.load(path / "state.pt", map_location="cpu", weights_only=True)["progress"].get("_save_tag", tag)
            except (OSError, ValueError, KeyError, RuntimeError):
                pass
            self.meter.journal.event("checkpoint_io_unmeasured", checkpoint_seq=seq, epoch=tag["epoch"], label=tag["label"],
                                     note="checkpoint present on disk; its write cost was not journaled before a crash")

    # ---- main
    def start(self):
        with self.meter.measure("aux_io", epoch=None, label="checkpoint_load_or_init") as ev:
            saved, location = self.checkpoints.load()
            ev["resumed_from"] = location.name if location is not None else None
        if saved is None:
            self.state = {"epoch": 0, "phase": "train", "group": 0, "attempts_total": 0, "executed_total": 0, "skipped_total": 0,
                          "consecutive_skips": 0, "epoch_acc": {"loss_sum": 0., "patients": 0, "patches": 0, "attempts": 0,
                                                                "executed": 0, "skipped": 0},
                          "pending_validation": None, "pending_predictions": [],
                          "selection": new_selection(), "status": None, "metrics": [], "groups": [], "predictions": [],
                          "steps": [], "epoch_start_sha256": None}
            self.meter.journal.event("session_context", status_at_open=None, epoch_completed=0, phase="train", group=0)
            with self.meter.measure("diagnostics", epoch=0, label="train_eval_epoch0"):
                diag = self._guarded(self.train_eval_fn, self.model)
            self.state["metrics"].append({"epoch": 0, **{f"train_eval_{k}": v for k, v in diag.items()}, **self._bias()})
            self._save(0, "initial")
            return "START"
        self.state = s29.restore(saved, self.model, self.optimizer, self.scaler)
        self._record_unmeasured_checkpoints()
        self.meter.journal.event("session_context", status_at_open=self.state["status"], epoch_completed=self.state["epoch"],
                                 phase=self.state["phase"], group=self.state["group"], resumed_from=location.name)
        return f"RESUME epoch {self.state['epoch'] + 1} phase {self.state['phase']} group {self.state['group']} ({location.name})"

    def run(self):
        st = self.state
        if st["status"] in ("AMP_SKIPS",):
            raise s29.ExperimentError("Run stopped after consecutive AMP skips; review before any continuation.")
        while st["status"] is None:
            e = st["epoch"]
            groups = make_groups(self.train_ids, e, self.seed)
            if st["phase"] == "train":
                if st["group"] == 0 and self._read_epoch_start(e) is None:
                    with self.meter.measure("aux_io", epoch=e + 1, label="epoch_start_params"):
                        self._write_epoch_start(e)
                loader = lambda pid: self.loader(pid, e)
                while st["group"] < len(groups):
                    gi = st["group"]; ids = groups[gi]
                    # one journal event per update attempt: durable as soon as the attempt finishes (before any checkpoint)
                    with self.meter.measure("train_attempt", epoch=e + 1, group=gi + 1) as ev:
                        g = train_group(self.model, self.optimizer, self.scaler, ids, loader, self.device, self.step_norms)
                        acc = st["epoch_acc"]
                        for k_src, k_dst in (("loss_sum", "loss_sum"), ("patients", "patients"), ("patches", "patches"),
                                             ("attempted", "attempts"), ("executed", "executed"), ("skipped", "skipped")):
                            acc[k_dst] += g[k_src]
                        st["attempts_total"] += 1; st["executed_total"] += g["executed"]; st["skipped_total"] += g["skipped"]
                        st["consecutive_skips"] = st["consecutive_skips"] + 1 if g["skipped"] else 0
                        st["group"] += 1
                        ev.update(attempt_total=st["attempts_total"], patients=g["patients"], patches=g["patches"],
                                  executed=g["executed"], skipped=g["skipped"])
                    labs = [self.labels[p] for p in ids]
                    st["groups"].append({"epoch": e + 1, "group": gi + 1, "attempt_total": st["attempts_total"],
                                         "group_size": len(ids), "patients": "|".join(ids), "labels": "|".join(labs),
                                         "group_mean_loss": g["loss_sum"] / g["patients"], "executed": g["executed"],
                                         "skipped": g["skipped"], "scale_before": g["scale_before"],
                                         "scale_after": g["scale_after"], **self._bias()})
                    if self.step_norms:
                        st["steps"].append({"epoch": e + 1, "group": gi + 1, "attempt_total": st["attempts_total"],
                                            "executed": g["executed"], "enc_update_norm": g["enc_update_norm"],
                                            "head_update_norm": g["head_update_norm"]})
                    if st["consecutive_skips"] >= self.max_skips:
                        st["status"] = "AMP_SKIPS"
                        break
                    if st["group"] == len(groups):
                        st["phase"] = "trained"
                    if st["attempts_total"] % self.checkpoint_every == 0 or st["group"] == len(groups):
                        self._save(e + 1, "train_done" if st["group"] == len(groups) else "attempt")
                    self._maybe_stop("attempt", st["attempts_total"])
                if st["status"] == "AMP_SKIPS":
                    self._save(e + 1, "amp_skips_stop")
                    return "AMP_SKIPS"
                self._maybe_stop("phase", (e + 1, "trained"))
            if st["phase"] == "trained":
                require(s29.tree_equal(self.frozen_bn, s29.bn_state(self.model)), "BatchNorm running stats changed.")
                with self.meter.measure("validation", epoch=e + 1):
                    val, preds = self._guarded(self.validate_fn, self.model, e + 1)
                st.update(phase="validated", pending_validation=val, pending_predictions=preds)
                self._save(e + 1, "validation_done")
                self._maybe_stop("phase", (e + 1, "validated"))
            if st["phase"] == "validated":
                with self.meter.measure("diagnostics", epoch=e + 1, label="train_eval_and_weight_change"):
                    diag = self._guarded(self.train_eval_fn, self.model)
                    start_params = self._read_epoch_start(e)
                    ep_enc, ep_head = self._norms(start_params) if start_params is not None else (None, None)
                    init_enc, init_head = self._norms(self.init_params)
                val, acc = st["pending_validation"], st["epoch_acc"]
                require(acc["patients"] == len(self.train_ids) and acc["attempts"] == len(groups), "Epoch accounting mismatch.")
                improved = update_selection(st["selection"], e + 1, val["ce"])
                row = {"epoch": e + 1, "budget": self.budget, "seed": self.seed,
                       "train_loss_aug_running": acc["loss_sum"] / acc["patients"], "train_patients": acc["patients"],
                       "train_patches": acc["patches"], "attempted_steps": acc["attempts"], "executed_steps": acc["executed"],
                       "skipped_steps": acc["skipped"], **{f"val_{k}": v for k, v in val.items()},
                       **{f"train_eval_{k}": v for k, v in diag.items()},
                       "epoch_weight_change_encoder": ep_enc, "epoch_weight_change_head": ep_head,
                       "weight_change_from_init_encoder": init_enc, "weight_change_from_init_head": init_head,
                       "improved": improved, "best_epoch": st["selection"]["best_epoch"], "best_val_ce": st["selection"]["best_ce"],
                       "epochs_without_improvement": st["selection"]["epochs_without_improvement"], **self._bias()}
                st["metrics"].append(row)
                st["predictions"].extend(st["pending_predictions"])
                st.update(epoch=e + 1, phase="train", group=0, pending_validation=None, pending_predictions=[],
                          epoch_acc={k: (0. if k == "loss_sum" else 0) for k in st["epoch_acc"]})
                st["status"] = stop_reason(st["selection"], e + 1, self.max_epochs, self.patience)
                self._save(e + 1, "epoch_committed", mark_best=improved)
                if improved:
                    with self.meter.measure("aux_io", epoch=e + 1, label="best_model_export"):
                        self.export_best()
                self.on_epoch(row)
                self._maybe_stop("phase", (e + 1, "committed"))
        return st["status"]

    def on_epoch(self, row):  # overridden by CLI runner for printing/emitting
        pass

    def export_best(self):
        st = self.state
        meta = {"epoch": st["selection"]["best_epoch"], "val_ce": st["selection"]["best_ce"], "budget": self.budget,
                "seed": self.seed, "identity": self.identity, "checkpoint_sequence": st.get("best_sequence")}
        sha = s29.save_tensor_file(self.run_dir / "best_model.pt", {"model": s29.cpu_tree(self.model.state_dict()), "meta": meta})
        s29.atomic_json(self.run_dir / "best_model.json", {**meta, "sha256": sha, "utc": s29.now()})

    def verify_or_rebuild_best(self):
        st = self.state
        if st["selection"]["best_epoch"] is None:
            return "none"
        js, pt = self.run_dir / "best_model.json", self.run_dir / "best_model.pt"
        ok = js.is_file() and pt.is_file() and s29.read_json(js).get("sha256") == sha_file(pt) and \
            s29.read_json(js).get("epoch") == st["selection"]["best_epoch"]
        if ok:
            return "verified"
        seq = st.get("best_sequence")
        require(seq is not None, "Best checkpoint sequence unknown.")
        path = self.run_dir / "checkpoints" / f"checkpoint_{seq:08d}"
        meta = s29.read_json(path / "commit.json")
        require(meta["identity"] == self.identity and sha_file(path / "state.pt") == meta["sha256"], "Best checkpoint corrupt.")
        data = torch.load(path / "state.pt", map_location="cpu", weights_only=True)
        meta2 = {"epoch": st["selection"]["best_epoch"], "val_ce": st["selection"]["best_ce"], "budget": self.budget,
                 "seed": self.seed, "identity": self.identity, "checkpoint_sequence": seq}
        sha = s29.save_tensor_file(pt, {"model": data["model"], "meta": meta2})
        s29.atomic_json(js, {**meta2, "sha256": sha, "utc": s29.now(), "rebuilt_from_checkpoint": True})
        return "rebuilt"


def emit(run_dir, state, extra=None):
    """Logical tables from the checkpoint state; resource tables ONLY from the resource journal (pure rebuild,
    no accumulation, so re-running it cannot double count or lower stored values). Locked CSVs only warn."""
    write_table(run_dir / "epoch_metrics.csv", state["metrics"])
    write_table(run_dir / "groups.csv", state["groups"])
    write_table(run_dir / "validation_predictions.csv", state["predictions"])
    if state["steps"]:
        write_table(run_dir / "steps.csv", state["steps"])
    events, bad = parse_journal(run_dir / JOURNAL_NAME)
    rows, sessions, rsum = build_resource_tables(events, bad)
    write_table(run_dir / "resources_by_epoch.csv", rows)
    write_table(run_dir / "resource_sessions.csv", sessions)
    summary = {"completed_epochs": state["epoch"], "status": state["status"] or "IN_PROGRESS", "selection": state["selection"],
               "attempts_total_logical": state["attempts_total"], "executed_total_logical": state["executed_total"],
               "skipped_total_logical": state["skipped_total"], "best_checkpoint_sequence": state.get("best_sequence"),
               "test_scored": False, "resources": rsum, **(extra or {})}
    for name, data in (("resource_summary.json", rsum), ("run_summary.json", summary)):
        try:
            s29.atomic_json(run_dir / name, data)
        except PermissionError:
            print(f"WARNING: {name} locked; rebuilt from the journal next time.", flush=True)
    return rows, sessions, rsum


# --------------------------------------------------------------------------------------------- bundle / protocol
def full_bundle():
    bundle = s29.load_bundle(ROOT)  # verifies frozen Stage 28 files, IDs, labels, paths, 64 patches/patient
    counts = collections.Counter((r["split"], r["class_label"]) for r in bundle["patients"].values())
    require(dict(counts) == EXPECTED_COUNTS, f"Unexpected split/class counts: {dict(counts)}")
    ids = {s: sorted(p for p, r in bundle["patients"].items() if r["split"] == s) for s in ("train", "validation", "test")}
    require(not (set(ids["train"]) & set(ids["validation"])) and not (set(ids["train"]) & set(ids["test"]))
            and not (set(ids["validation"]) & set(ids["test"])), "Split overlap.")
    require(sum(len(v) for v in ids.values()) == 929, "Expected 929 patients.")
    for p in ids["train"] + ids["validation"] + ids["test"]:  # metadata only for test
        require(len(bundle["patches"][p]) == 64, f"Expected 64 patches: {p}")
    return bundle, ids


def protocol_record(kind, bundle, ids, weights, environment, extra):
    return {"schema_version": 1, "stage31_version": VERSION, "kind": kind,
            "script31_sha256": sha_file(Path(__file__)),
            "dependent_scripts_sha256": {k: sha_file(v) for k, v in SCRIPTS.items()},
            "stage29_imported_sha256": S29_SHA256, "stage28_manifest_sha256": bundle["manifest_sha256"],
            "stage28_files": s29.EXPECTED_FILES, "dataset_fingerprint": s29.EXPECTED_DATASET,
            "split_fingerprint": s29.EXPECTED_SPLIT, "weights_sha256": sha_file(weights),
            "weights_path": str(Path(weights).relative_to(ROOT)), "environment": environment,
            "settings": SETTINGS, "settings_sha256": s29.digest(SETTINGS), "stopping": STOPPING,
            "run_matrix": RUN_MATRIX, "notes": NOTES,
            "split_counts": {f"{s}/{c}": n for (s, c), n in EXPECTED_COUNTS.items()},
            "train_patients_sha256": s29.digest(ids["train"]), "validation_patients_sha256": s29.digest(ids["validation"]),
            "pilot_validation_patients": sorted(p for p, r in bundle["pilot"].items() if r["split"] == "validation"),
            **extra}


def prepare_folder(output, kind, extra):
    bundle, ids = full_bundle()
    _, _, weights = s29.read_prepared(bundle, None)
    environment = s29.environment_record(bundle)
    if output.exists():
        verify_protocol(output, bundle, environment)
        print(f"ALREADY PREPARED (verified): {output}")
        return
    require(shutil.disk_usage(ROOT).free >= 5 * 1024 ** 3, "Need >= 5 GiB free disk.")
    output.mkdir(parents=True)
    rec = protocol_record(kind, bundle, ids, weights, environment, extra)
    s29.atomic_json(output / "protocol.json", rec)
    s29.atomic_json(output / "preparation.json", {"status": "PREPARED", "utc": s29.now(), "protocol_sha256": sha_file(output / "protocol.json"),
                                                  "code_frozen_note": "Do not modify scripts/31 after this point; a change makes --run refuse."})
    print(f"PREPARED ({kind}): {output}. No training has run.")


def verify_protocol(output, bundle, environment):
    p, prep = output / "protocol.json", output / "preparation.json"
    require(p.is_file() and prep.is_file(), f"Not prepared: {output}")
    protocol, record = s29.read_json(p), s29.read_json(prep)
    require(record["protocol_sha256"] == sha_file(p), "protocol.json changed after preparation.")
    require(protocol["script31_sha256"] == sha_file(Path(__file__)), "Stage 31 code changed after preparation; refusing (version a new output).")
    require(protocol["dependent_scripts_sha256"] == {k: sha_file(v) for k, v in SCRIPTS.items()}, "A dependent script changed.")
    require(protocol["stage28_manifest_sha256"] == bundle["manifest_sha256"], "Stage 28 snapshot differs.")
    require(protocol["settings"] == json.loads(json.dumps(SETTINGS)) and protocol["stopping"] == STOPPING, "Settings/stopping differ.")
    if environment is not None:
        require(protocol["environment"] == environment, "Execution environment differs from preparation.")
    return protocol, record["protocol_sha256"]


def preflight():
    bundle, ids = full_bundle()
    _, _, weights = s29.read_prepared(bundle, None)
    t0, done = time.perf_counter(), 0
    for part in ("train", "validation"):  # test PNGs are never opened
        for p in ids[part]:
            for row in bundle["patches"][p]:
                s29.verified_png_bytes(row); done += 1
            if done % 6400 == 0:
                print(f"  PNG hashes verified: {done}/{(len(ids['train']) + len(ids['validation'])) * 64}", flush=True)
    device = init_torch(True)
    env = s29.environment_record(bundle)
    labels = {p: bundle["patients"][p]["class_label"] for p in ids["train"]}
    for seed in SEEDS:
        for e in range(STOPPING["max_epochs"]):
            g = make_groups(ids["train"], e, seed)
            require(len(g) == 140 and len(g[-1]) == 1, "Expected 139x4 + 1 groups.")
    sub = select_train_eval(ids["train"], labels, 10)
    print(f"PREFLIGHT PASS: Stage 28 bundle/IDs/labels/paths; counts {dict(collections.Counter((bundle['patients'][p]['split'], bundle['patients'][p]['class_label']) for p in bundle['patients']))}; "
          f"{done} train+validation PNG hashes in {time.perf_counter() - t0:.0f}s; test PNGs not opened; weights {sha_file(weights)[:12]}; "
          f"torch {env['versions']['torch']} on {env['gpu']}; 140 groups/epoch (last=1) for all seeds; train-eval subset {len(sub)}. No writes.")


# --------------------------------------------------------------------------------------------- CLI run
SMOKE_SUBSETS = {
    "compat_pilot80": "pilot train 60 / pilot validation 20 (Stage 28 pilot_manifest), train-eval 10+10 from these 60",
    "resume_19": "pilot train first 7 LUAD + 6 LUSC by pilot_order_within_class (13 -> groups 4,4,4,1); "
                 "pilot validation first 3 + 3; train-eval 3+3",
}


def subset_ids(bundle, ids, name):
    if name == "full":
        return ids["train"], ids["validation"], 10
    pil = bundle["pilot"]
    by = lambda split, c: [r["case_submitter_id"] for r in sorted((r for r in pil.values() if r["split"] == split and r["class_label"] == c),
                                                                    key=lambda r: int(r["pilot_order_within_class"]))]
    if name == "compat_pilot80":
        return sorted(by("train", "LUAD") + by("train", "LUSC")), sorted(by("validation", "LUAD") + by("validation", "LUSC")), 10
    if name == "resume_19":
        return sorted(by("train", "LUAD")[:7] + by("train", "LUSC")[:6]), sorted(by("validation", "LUAD")[:3] + by("validation", "LUSC")[:3]), 3
    raise ValueError(name)


def cli_run(output, budget, seed, run_name=None, subset="full", max_epochs=None, stop_after_attempts=None, stop_at_phase=None):
    require(budget in BUDGETS and seed in SEEDS, "Budget/seed must be one of the protocol values.")
    bundle, ids = full_bundle()
    _, _, weights = s29.read_prepared(bundle, None)
    device = init_torch(True)
    environment = s29.environment_record(bundle)
    protocol, protocol_sha = verify_protocol(output, bundle, environment)
    if output.resolve() == MAIN_OUTPUT.resolve():
        require(subset == "full" and max_epochs is None and run_name is None and stop_after_attempts is None and stop_at_phase is None,
                "Main runs use the frozen protocol only (no test hooks).")
    train_ids, val_ids, per_class = subset_ids(bundle, ids, subset)
    max_epochs = max_epochs or STOPPING["max_epochs"]
    run_dir = output / "runs" / (run_name or f"K{budget}_seed{seed}")
    run_dir.mkdir(parents=True, exist_ok=True)
    identity = {"protocol_sha256": protocol_sha, "budget": budget, "seed": seed, "subset": subset, "max_epochs": max_epochs,
                "patience": STOPPING["patience_epochs"], "train_n": len(train_ids), "validation_n": len(val_ids)}
    with s29.execution_lock(run_dir / "execution.lock"):
        marker = run_dir / "run_identity.json"
        if marker.exists():
            require(s29.read_json(marker) == identity, "Run folder belongs to different settings; refusing to continue.")
        else:
            s29.atomic_json(marker, identity)
        journal = ResourceJournal(run_dir / JOURNAL_NAME)
        meter = Meter(journal, lambda kind: s29.MemoryMonitor(device))
        outcome, runner, sub, crashed = "FAILED", None, None, False

        def do_emit():
            with meter.measure("aux_io", epoch=None, label="emit_tables"):
                emit(run_dir, runner.state, {"train_eval_subset": sub})
        try:
            data = Data(bundle, train_ids, val_ids)
            labels = {p: data.labels[p] for p in train_ids}
            sub = select_train_eval(train_ids, labels, per_class)
            model = s29.build_model(weights, seed=seed, device=device)
            optimizer, scaler = s29.optimizer_for(model), s29.scaler_for(device)
            optimizer.param_groups[0]["lr"] = SETTINGS["encoder_lr"]; optimizer.param_groups[1]["lr"] = SETTINGS["head_lr"]
            loader = lambda pid, e: data.load_bag(pid, "train", e, budget, seed, True, device)
            validate_fn = lambda m, ep: evaluate(m, data, data.val_ids, "validation", device, ep)
            def train_eval_fn(m):
                out, _ = evaluate(m, data, sub, "train", device, None)
                return {k: v for k, v in out.items() if k not in ("balanced_accuracy", "macro_f1", "accuracy")}
            runner = Runner(run_dir=run_dir, identity=identity, model=model, optimizer=optimizer, scaler=scaler, device=device,
                            train_ids=train_ids, labels=labels, seed=seed, budget=budget, max_epochs=max_epochs,
                            patience=STOPPING["patience_epochs"], loader=loader, validate_fn=validate_fn, train_eval_fn=train_eval_fn,
                            meter=meter, checkpoint_every=SETTINGS["checkpoint_every_attempts"], stop_after_attempts=stop_after_attempts,
                            stop_at_phase=stop_at_phase, max_skips=SETTINGS["max_consecutive_amp_skips"])
            runner.frozen_bn = s29.bn_state(model)
            # setup = script start -> model built (imports, bundle/hash/protocol checks, data index, model construction)
            journal.event("setup", epoch=None, wall_seconds=time.perf_counter() - PROCESS_T0, cpu_seconds=time.process_time() - PROCESS_C0)
            def on_epoch(row):
                do_emit()
                print(f"K{budget} s{seed} ep{row['epoch']:2d}: trainCE={row['train_loss_aug_running']:.4f} valCE={row['val_ce']:.4f} "
                      f"AUROC={row['val_auroc']:.3f} BA={row['val_balanced_accuracy']:.3f} best={row['best_epoch']}({row['best_val_ce']:.4f}) "
                      f"wait={row['epochs_without_improvement']} skips={row['skipped_steps']} status={runner.state['status']}", flush=True)
            runner.on_epoch = on_epoch
            s29.atomic_json(run_dir / "status.json", {"status": "RUNNING", "identity": identity, "utc": s29.now()})
            msg = runner.start()
            print(f"{msg}: {run_dir}", flush=True)
            if runner.state["epoch"] > 0 or runner.state["status"]:
                with meter.measure("aux_io", epoch=None, label="best_model_verify"):
                    print(f"best_model: {runner.verify_or_rebuild_best()}", flush=True)
            do_emit()
            try:
                status = runner.run()
            except StopRequested as s:
                # test hook emulating a crash: no session_end, no further events, tables not rebuilt now
                outcome, crashed = f"EMULATED_CRASH {s}", True
                journal.abandon()
                print(f"STOP REQUESTED ({s}); emulated crash, checkpoint state is resumable.", flush=True)
                return 3
            do_emit()
            outcome = status
            print(f"FINISHED {status}: best epoch {runner.state['selection']['best_epoch']} "
                  f"(val CE {runner.state['selection']['best_ce']}); {run_dir}", flush=True)
            return 4 if status == "AMP_SKIPS" else 0
        except KeyboardInterrupt:
            outcome = "INTERRUPTED"
            raise
        except Exception as e:
            outcome = f"FAILED: {type(e).__name__}: {e}"
            raise
        finally:
            if not crashed:
                if not journal.closed:
                    journal.close(outcome)
                rsum = None
                if runner is not None and getattr(runner, "state", None) is not None:
                    # final rebuild after session_end (this rebuild itself is not measured: the journal is closed)
                    rsum = emit(run_dir, runner.state, {"train_eval_subset": sub})[2]
                final = {"EARLY_STOP_PATIENCE": "COMPLETE", "MAX_EPOCHS": "COMPLETE", "AMP_SKIPS": "STOPPED_AMP_SKIPS"}.get(outcome, "PARTIAL")
                if outcome.startswith("FAILED"):
                    final = "FAILED"
                try:
                    s29.atomic_json(run_dir / "status.json", {"status": final, "outcome": outcome, "identity": identity, "utc": s29.now(),
                                                              "resume": "repeat the same --run command", "resources": rsum})
                except OSError as e:
                    print(f"WARNING: status.json not written ({e}); checkpoints and the journal are authoritative.", flush=True)


# --------------------------------------------------------------------------------------------- smoke (GPU, separate folder)
TIMING_SUFFIXES = ("_seconds", "_bytes", "checkpoint_count", "sessions_contributing")


def strip_timing(v):
    if isinstance(v, dict):
        return {k: strip_timing(x) for k, x in v.items() if not k.endswith(TIMING_SUFFIXES)}
    if isinstance(v, list):
        return [strip_timing(x) for x in v]
    return v


def child(args):
    cmd = [sys.executable, "-B", str(Path(__file__).resolve())] + args
    return subprocess.run(cmd, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1")).returncode


def smoke():
    out = SMOKE_OUTPUT
    report = {"utc": s29.now(), "folder": str(out.relative_to(ROOT)), "checks": {}}
    require(child(["--prepare-smoke"]) == 0, "smoke prepare failed")
    W = ["--worker", "--output", str(out)]
    t = time.perf_counter()
    # 1. compatibility with Stage 29 / D08-B: pilot 60/20, K=64, seed 20261113, one epoch
    require(child(W + ["--run-name", "compat_d08b", "--subset", "compat_pilot80", "--budget", "64", "--seed", "20261113", "--max-epochs", "1"]) == 0,
            "compat run failed")
    report["compat_wall_seconds"] = time.perf_counter() - t
    report["checks"]["compat_d08b"] = compare_d08b(out / "runs" / "compat_d08b")
    # 2. resume: 13 train (groups 4,4,4,1), 6 validation, K=16, seed 20261114, 2 epochs
    t = time.perf_counter()
    base = W + ["--subset", "resume_19", "--budget", "16", "--seed", "20261114", "--max-epochs", "2"]
    require(child(base + ["--run-name", "resume_continuous"]) == 0, "continuous run failed")
    t1 = time.perf_counter()
    cdir = out / "runs" / "resume_continuous"
    before = {n: (cdir / n).read_bytes() for n in ("resources_by_epoch.csv", "epoch_metrics.csv", "groups.csv", "validation_predictions.csv")}
    sess_before = len(read_rows(cdir / "resource_sessions.csv"))
    require(child(base + ["--run-name", "resume_continuous"]) == 0, "reopen of completed run failed")   # reopen finished run
    sess_after = read_rows(cdir / "resource_sessions.csv")
    report["checks"]["reopen_completed"] = {
        "status": "PASS" if all((cdir / n).read_bytes() == b for n, b in before.items()) and len(sess_after) == sess_before + 1
                  and sess_after[-1]["post_completion_session"] == "True" else "FAIL",
        "historical_tables_byte_identical": {n: (cdir / n).read_bytes() == b for n, b in before.items()},
        "sessions_before_after": [sess_before, len(sess_after)], "new_session_post_completion": sess_after[-1]["post_completion_session"]}
    t1 = time.perf_counter()
    rc = [child(base + ["--run-name", "resume_interrupted", "--stop-after-attempts", "3"]),     # mid epoch 1 (before 5-attempt checkpoint)
          child(base + ["--run-name", "resume_interrupted", "--stop-at-phase", "1:trained"]),  # after training, before validation
          child(base + ["--run-name", "resume_interrupted", "--stop-at-phase", "2:validated"]),  # after validation, before commit
          child(base + ["--run-name", "resume_interrupted"])]
    require(rc == [3, 3, 3, 0], f"unexpected exit codes {rc}")
    report["resume_wall_seconds"] = {"continuous": t1 - t, "interrupted_total": time.perf_counter() - t1}
    report["checks"]["resume"] = compare_runs(out / "runs" / "resume_continuous", out / "runs" / "resume_interrupted")
    report["status"] = "PASS" if all(c["status"] == "PASS" for c in report["checks"].values()) else "FAIL"
    report["script31_sha256"] = sha_file(Path(__file__))
    s29.atomic_json(out / "smoke_report.json", report)
    print(json.dumps(report, indent=1, default=str))
    return 0 if report["status"] == "PASS" else 1


def final_checkpoint(run_dir):
    import torch as T
    last = sorted((run_dir / "checkpoints").glob("checkpoint_*"))[-1]
    return T.load(last / "state.pt", map_location="cpu", weights_only=True)


def compare_d08b(run_dir):
    import torch as T
    s29.torch = s29.torch or T
    st = final_checkpoint(run_dir)["progress"]
    row = st["metrics"][1]
    ref = {r["epoch"]: r for r in json.loads((D08B / "epochs.json").read_text())}[1]
    keys = {"val_ce": "val_ce", "val_auroc": "val_auroc", "val_balanced_accuracy": "val_balanced_accuracy",
            "val_macro_f1": "val_macro_f1", "val_p_mean": "val_p_mean", "val_p_sd": "val_p_sd"}
    diffs = {k: abs(row[k] - ref[v]) for k, v in keys.items()}
    pred = [(r["case_submitter_id"], r["label"], r["p_LUSC"]) for r in st["predictions"]]
    refp = [(r["case_submitter_id"], int(r["label"]), float(r["p_LUSC"])) for r in read_rows(D08B / "validation_predictions.csv") if r["epoch"] == "1"]
    groups = [(g["patients"], g["bias_diff_LUSC_minus_LUAD"]) for g in st["groups"]]
    refg = [(g["patients"], g["bias_diff_LUSC_minus_LUAD"]) for g in read_rows(D08B / "groups.csv") if g["epoch"] == "1"]
    ok = max(diffs.values()) == 0.0 and pred == refp and groups == refg and st["executed_total"] == 15 and st["skipped_total"] == 0
    return {"status": "PASS" if ok else "MISMATCH", "max_abs_diff_epoch1": max(diffs.values()), "diffs": diffs,
            "validation_predictions_identical": pred == refp, "groups_and_bias_trajectory_identical": groups == refg,
            "attempted_executed_skipped": [st["attempts_total"], st["executed_total"], st["skipped_total"]],
            "note": "Same inputs/initial weights as D08-B (pilot 60/20, K=64, seed 20261113); Stage 31 loader/train_group/evaluate are ports."}


def compare_runs(a_dir, b_dir):
    a, b = final_checkpoint(a_dir), final_checkpoint(b_dir)
    import torch as T
    s29.torch = s29.torch or T
    checks = {k: s29.tree_equal(a[k], b[k]) for k in ("model", "optimizer", "scaler", "rng")}
    pa, pb = a["progress"], b["progress"]
    for k in ("metrics", "groups", "predictions", "selection", "status", "attempts_total", "executed_total", "skipped_total", "epoch"):
        checks[f"progress_{k}_identical"] = s29.tree_equal(strip_timing(pa[k]), strip_timing(pb[k]))
    checks["no_duplicate_metric_rows"] = [r["epoch"] for r in pb["metrics"]] == [0, 1, 2]
    checks["no_duplicate_group_rows"] = [(g["epoch"], g["group"]) for g in pb["groups"]] == [(e, g) for e in (1, 2) for g in (1, 2, 3, 4)]
    checks["partial_group_present"] = [g["group_size"] for g in pb["groups"] if g["epoch"] == 1] == [4, 4, 4, 1]
    checks["csv_identical"] = all((a_dir / n).read_bytes() == (b_dir / n).read_bytes() for n in ("epoch_metrics.csv", "groups.csv", "validation_predictions.csv"))
    checks["best_model_verified"] = s29.read_json(b_dir / "best_model.json")["sha256"] == sha_file(b_dir / "best_model.pt") and \
        s29.read_json(b_dir / "best_model.json")["epoch"] == pb["selection"]["best_epoch"]
    # resource accounting from the journals (actual work), checked against the known interruption pattern
    ra, rb = (build_resource_tables(*parse_journal(d / JOURNAL_NAME)) for d in (a_dir, b_dir))
    rows_a, rows_b = {r["epoch"]: r for r in ra[0]}, {r["epoch"]: r for r in rb[0]}
    checks["resource_epochs_0_1_2"] = sorted(rows_a) == sorted(rows_b) == [0, 1, 2]
    checks["resource_logical_attempts_equal"] = all(rows_a[e]["train_attempts_logical"] == rows_b[e]["train_attempts_logical"] == 4 for e in (1, 2))
    checks["resource_replay_counted_epoch1"] = rows_b[1]["train_attempts_replayed"] == 3 and rows_b[1]["train_attempts_executed_work"] == 7         and rows_b[2]["train_attempts_replayed"] == 0 and all(rows_a[e]["train_attempts_replayed"] == 0 for e in (1, 2))
    checks["resource_no_validation_or_diag_replay"] = all(rows_b[e]["validation_blocks"] == 1 and rows_b[e]["diagnostics_blocks"] == 1 for e in (1, 2))
    checks["resource_checkpoint_counts_equal"] = all(rows_a[e]["checkpoints_measured"] == rows_b[e]["checkpoints_measured"] for e in (0, 1, 2))         and all(rows_b[e]["checkpoints_unmeasured"] == 0 for e in (0, 1, 2))
    checks["resource_no_negative_values"] = all(v >= 0 for t in (ra, rb) for r in t[0] for v in r.values() if isinstance(v, (int, float)) and not isinstance(v, bool))         and all(s["other_wall_seconds"] >= 0 for t in (ra, rb) for s in t[1])
    checks["resource_train_time_present_every_epoch"] = all(rows_b[e]["train_wall_seconds"] > 0 and rows_b[e]["train_sampled_peak_rss_bytes"] > 0 for e in (1, 2))
    checks["resource_unclean_sessions_flagged"] = len(rb[2]["unclean_sessions"]) == 3 and rb[2]["active_seconds_is_lower_bound"]
    timing = {"continuous": [{k: v for k, v in r.items() if k.endswith("_wall_seconds")} for r in ra[0]],
              "interrupted": [{k: v for k, v in r.items() if k.endswith("_wall_seconds")} for r in rb[0]]}
    return {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks, "timing_excluded_from_equality": timing,
            "interrupted_resource_summary": rb[2], "attempts": [pb["attempts_total"], pb["executed_total"], pb["skipped_total"]]}


# --------------------------------------------------------------------------------------------- CPU self-tests
def self_test():
    import unittest
    import tempfile
    from unittest.mock import patch
    init_torch(False)
    bundle, ids = full_bundle()

    class FakeScaler:
        """Scripted AMP: True in pattern = skip that attempt (no optimizer.step, scale halves)."""
        def __init__(self, pattern):
            self.pattern, self.i, self.s = list(pattern), 0, 1024.
        def scale(self, loss): return loss
        def unscale_(self, opt): pass
        def step(self, opt):
            self._skip = self.pattern[self.i] if self.i < len(self.pattern) else False
            self.i += 1
            if not self._skip:
                opt.step()
        def update(self):
            if self._skip:
                self.s /= 2
        def get_scale(self): return self.s
        def state_dict(self): return {"s": self.s, "i": self.i}
        def load_state_dict(self, d): self.s, self.i = d["s"], d["i"]

    def synthetic_loader(pid, epoch=0):
        i = int(pid[-4:])
        g = torch.Generator().manual_seed(1000 * epoch + i)
        return torch.rand((2, 3, 8, 8), generator=g) + (i % 2) * .5, torch.tensor([i % 2])

    class Clock:
        """Controlled wall/CPU clocks: synthetic work advances them by fixed amounts (expected sums are exact)."""
        def __init__(self):
            self.t, self.c = 1000., 500.
        def wall(self): return self.t
        def cpu(self): return self.c
        def adv(self, w, c): self.t += w; self.c += c

    PEAK = {"train_attempt": 100, "validation": 200, "diagnostics": 150, "checkpoint_io": 300, "aux_io": 10}

    class FakeMon:
        def __init__(self, kind): self.v = PEAK[kind]
        def __enter__(self): return self
        def __exit__(self, *e): return False
        def values(self): return {"peak_allocated_vram_bytes": self.v, "peak_reserved_vram_bytes": 2 * self.v, "sampled_peak_rss_bytes": 3 * self.v}

    COST = {"patient": (1.0, .25), "validation": (2.0, 1.0), "diagnostics": (3.0, 1.5), "checkpoint": (.5, .1)}

    class Tests(unittest.TestCase):
        def setUp(self):
            self.tmp = tempfile.TemporaryDirectory(prefix="stage31-test-")
            self.dir = Path(self.tmp.name)
            self.clock, self.journals = Clock(), []
        def tearDown(self):
            for j in self.journals:
                if not j.closed:
                    j.abandon()
            self.tmp.cleanup()

        def tiny(self, seed=20261113):
            m = s29.build_model(None, seed, device=torch.device("cpu"), tiny=True)
            o = s29.optimizer_for(m); o.param_groups[0]["lr"] = 1e-2; o.param_groups[1]["lr"] = 1e-2
            return m, o

        def test_557_groups_cover_once_last_is_one(self):
            for seed in SEEDS:
                for e in (0, 1, 7, 29):
                    g = make_groups(ids["train"], e, seed)
                    self.assertEqual(len(g), 140)
                    self.assertEqual([len(x) for x in g[:-1]], [4] * 139)
                    self.assertEqual(len(g[-1]), 1)
                    self.assertEqual(sorted(p for x in g for p in x), ids["train"])

        def test_gradient_average_groups_4_and_1(self):
            for group in (["TCGA-AA-0001", "TCGA-AA-0002", "TCGA-AA-0003", "TCGA-AA-0004"], ["TCGA-AA-0005"]):
                ma, oa = self.tiny(); mb, ob = self.tiny()
                train_group(ma, oa, s29.scaler_for(torch.device("cpu")), group, synthetic_loader, torch.device("cpu"))
                mb.train()
                losses = [torch.nn.functional.cross_entropy(mb(synthetic_loader(p)[0]), synthetic_loader(p)[1]) for p in group]
                torch.stack(losses).mean().backward(); ob.step()
                for a, b in zip(ma.parameters(), mb.parameters()):
                    torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-7)

        def test_amp_skip_first_and_later_no_stale_norms(self):
            m, o = self.tiny(); sc = FakeScaler([True, False, True, False])
            ids4 = ["TCGA-AA-0001", "TCGA-AA-0002"]
            before = [p.detach().clone() for p in m.parameters()]
            r1 = train_group(m, o, sc, ids4, synthetic_loader, torch.device("cpu"), step_norms=True)
            self.assertEqual((r1["attempted"], r1["executed"], r1["skipped"]), (1, 0, 1))
            self.assertIsNone(r1["enc_update_norm"]); self.assertIsNone(r1["head_update_norm"])
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(before, m.parameters())))
            r2 = train_group(m, o, sc, ids4, synthetic_loader, torch.device("cpu"), step_norms=True)
            self.assertEqual(r2["executed"], 1); self.assertGreater(r2["head_update_norm"], 0)
            r3 = train_group(m, o, sc, ids4, synthetic_loader, torch.device("cpu"), step_norms=True)
            self.assertEqual(r3["skipped"], 1); self.assertIsNone(r3["head_update_norm"])  # not the stale r2 value

        def runner(self, run_dir, validate_ces, scaler=None, max_epochs=6, patience=2, n_train=9, **kw):
            m, o = self.tiny()
            tr = [f"TCGA-AA-{i:04d}" for i in range(1, n_train + 1)]  # 9 -> 4,4,1 ; 13 -> 4,4,4,1
            labels = {p: ("LUAD" if int(p[-4:]) % 2 == 0 else "LUSC") for p in tr}
            ck = self.clock
            def loader(pid, epoch):
                ck.adv(*COST["patient"]); return synthetic_loader(pid, epoch)
            def validate_fn(model, ep):
                ck.adv(*COST["validation"])
                return ({"ce": validate_ces[ep - 1], "auroc": .5, "balanced_accuracy": .5, "macro_f1": .5},
                        [{"epoch": ep, "case_submitter_id": "X", "label": 0, "p_LUSC": .5}])
            def train_eval_fn(model):
                ck.adv(*COST["diagnostics"]); return {"ce": float(sum(float(p.sum()) for p in model.parameters()))}
            j = ResourceJournal(run_dir / JOURNAL_NAME, clock=ck.wall, cpu_clock=ck.cpu, t0=ck.wall(), c0=ck.cpu(),
                                session_id=f"s{len(self.journals)}", utc=lambda: "T")
            self.journals.append(j)
            r = Runner(run_dir=run_dir, identity={"t": 1}, model=m, optimizer=o, scaler=scaler or s29.scaler_for(torch.device("cpu")),
                       device=torch.device("cpu"), train_ids=tr, labels=labels, seed=20261113, budget=16, max_epochs=max_epochs,
                       patience=patience, loader=loader, validate_fn=validate_fn, train_eval_fn=train_eval_fn,
                       meter=Meter(j, FakeMon, clock=ck.wall, cpu_clock=ck.cpu), checkpoint_every=2, **kw)
            real_save = r.checkpoints.save
            def timed_save(*a, **k):
                out = real_save(*a, **k); ck.adv(*COST["checkpoint"]); return out
            r.checkpoints.save = timed_save
            r.frozen_bn = s29.bn_state(m)
            return r

        def sessions(self, d, ces, stops, n_train=13, **common):
            """Successive 'processes'; StopRequested = emulated crash (journal abandoned, no session_end)."""
            snaps, status, r = [], None, None
            for kw in stops:
                r = self.runner(d, ces, n_train=n_train, **common, **kw)
                try:
                    r.start(); status = r.run(); r.meter.journal.close(str(status))
                except StopRequested:
                    r.meter.journal.abandon()
                snaps.append(build_resource_tables(*parse_journal(d / JOURNAL_NAME)))
            return r, status, snaps

        CES = [.7, .6, .6, .65]   # patience 2 -> stops after epoch 4; best epoch 2 (tie at 3 keeps 2)

        def expected_epoch(self, k, replay_patients=0, replay_attempts=0):
            return {"train_wall_seconds": 13 * 1.0 + replay_patients * 1.0, "train_cpu_seconds": 13 * .25 + replay_patients * .25,
                    "train_attempts_logical": 4, "train_attempts_executed_work": 4 + replay_attempts, "train_attempts_replayed": replay_attempts,
                    "validation_wall_seconds": 2.0, "validation_cpu_seconds": 1.0, "validation_blocks": 1,
                    "diagnostics_wall_seconds": 3.0, "diagnostics_cpu_seconds": 1.5, "diagnostics_blocks": 1,
                    "checkpoints_measured": 4, "checkpoint_io_wall_seconds": 2.0, "checkpoints_unmeasured": 0,
                    "train_peak_allocated_bytes": 100, "train_sampled_peak_rss_bytes": 300, "validation_peak_reserved_bytes": 400,
                    "diagnostics_peak_allocated_bytes": 150, "checkpoint_io_sampled_peak_rss_bytes": 900}

        def assertRows(self, rows, overrides=None):
            rows = {r["epoch"]: r for r in rows}
            self.assertEqual(sorted(rows), [0, 1, 2, 3, 4])
            self.assertEqual((rows[0]["diagnostics_wall_seconds"], rows[0]["checkpoints_measured"], rows[0]["train_wall_seconds"]), (3.0, 1, 0))
            for k in (1, 2, 3, 4):
                exp = self.expected_epoch(k, **(overrides or {}).get(k, {}))
                for f, v in exp.items():
                    self.assertAlmostEqual(rows[k][f], v, places=9, msg=f"epoch {k} {f}")

        def assertSane(self, snaps):
            prev = None
            for rows, sessions, summ in snaps:
                for r in rows:
                    for f, v in r.items():
                        if isinstance(v, (int, float)) and not isinstance(v, bool):
                            self.assertGreaterEqual(v, 0, f)
                for sres in sessions:
                    self.assertGreaterEqual(sres["other_wall_seconds"], -1e-9)
                if prev is not None:   # already stored values never decrease across sessions
                    cur = {r["epoch"]: r for r in rows}
                    for r in prev:
                        for f, v in r.items():
                            if isinstance(v, (int, float)) and not isinstance(v, bool):
                                self.assertGreaterEqual(cur[r["epoch"]][f], v - 1e-12, f"{f} decreased")
                prev = rows

        def test_selection_rule_pure(self):
            s = new_selection()
            out = [update_selection(s, e, ce) for e, ce in enumerate([.7, .6, .6, .65, .61], 1)]
            self.assertEqual(out, [True, True, False, False, False])
            self.assertEqual((s["best_epoch"], s["best_ce"], s["epochs_without_improvement"]), (2, .6, 3))
            self.assertEqual(stop_reason(s, 5, 30, 3), "EARLY_STOP_PATIENCE")
            self.assertEqual(stop_reason(s, 5, 30, 5), None)
            self.assertEqual(stop_reason(new_selection() | {"epochs_without_improvement": 0}, 30, 30, 5), "MAX_EPOCHS")

        def test_patience_tie_best_and_resume_in_new_objects(self):
            ces = [.7, .6, .6, .65, .62, .5]
            a = self.runner(self.dir / "a", ces); a.start(); self.assertEqual(a.run(), "EARLY_STOP_PATIENCE")
            self.assertEqual(a.state["epoch"], 4); self.assertEqual(a.state["selection"]["best_epoch"], 2)
            best = torch.load(self.dir / "a" / "best_model.pt", weights_only=True)
            self.assertEqual(best["meta"]["epoch"], 2)
            # interrupted at several points, each continuation with fresh model/optimizer objects ("new process")
            stops = [dict(stop_after_attempts=2), dict(stop_at_phase=(1, "trained")), dict(stop_at_phase=(2, "validated")),
                     dict(stop_at_phase=(3, "committed")), {}]
            for kw in stops:
                b = self.runner(self.dir / "b", ces, **kw); b.start()
                try:
                    status = b.run()
                except StopRequested:
                    continue
            self.assertEqual(status, "EARLY_STOP_PATIENCE")
            ca, cb = a.checkpoints.load()[0], b.checkpoints.load()[0]
            for k in ("model", "optimizer", "scaler", "rng"):
                self.assertTrue(s29.tree_equal(ca[k], cb[k]), k)
            for k in ("metrics", "groups", "predictions", "selection", "status", "attempts_total"):
                self.assertTrue(s29.tree_equal(strip_timing(ca["progress"][k]), strip_timing(cb["progress"][k])), k)
            self.assertEqual([r["epoch"] for r in cb["progress"]["metrics"]], [0, 1, 2, 3, 4])
            self.assertEqual(len(cb["progress"]["groups"]), 4 * 3)
            self.assertTrue(s29.tree_equal(best["model"], torch.load(self.dir / "b" / "best_model.pt", weights_only=True)["model"]))
            # best checkpoint generation retained and equals exported best model
            seq = cb["progress"]["best_sequence"]
            gen = torch.load(self.dir / "b" / "checkpoints" / f"checkpoint_{seq:08d}" / "state.pt", weights_only=True)
            self.assertTrue(s29.tree_equal(gen["model"], best["model"]))
            # resume after completion keeps stop state
            c = self.runner(self.dir / "b", ces); c.start()
            self.assertEqual(c.state["status"], "EARLY_STOP_PATIENCE"); self.assertEqual(c.run(), "EARLY_STOP_PATIENCE")
            # corrupted export is rebuilt from the best generation
            (self.dir / "b" / "best_model.pt").write_bytes(b"corrupt")
            self.assertEqual(c.verify_or_rebuild_best(), "rebuilt")
            self.assertTrue(s29.tree_equal(torch.load(self.dir / "b" / "best_model.pt", weights_only=True)["model"], best["model"]))

        def test_three_consecutive_amp_skips_stop(self):
            r = self.runner(self.dir / "s", [.7] * 6, scaler=FakeScaler([False, True, True, True]))
            r.start()
            self.assertEqual(r.run(), "AMP_SKIPS")
            saved = r.checkpoints.load()[0]["progress"]
            self.assertEqual((saved["attempts_total"], saved["executed_total"], saved["skipped_total"], saved["status"]), (4, 1, 3, "AMP_SKIPS"))
            r2 = self.runner(self.dir / "s", [.7] * 6, scaler=FakeScaler([])); r2.start()
            with self.assertRaises(s29.ExperimentError):
                r2.run()

        def test_patch_nesting_and_seed_consistency(self):
            pid = ids["train"][0]
            for seed in SEEDS:
                for e in range(8):
                    k = {b: s29.patch_indices(pid, e, b, seed=seed) for b in BUDGETS}
                    self.assertEqual(k[16], k[32][:16]); self.assertEqual(k[32], k[64][:32])
            self.assertNotEqual(s29.patient_order(ids["train"], 0, seed=SEEDS[0]), s29.patient_order(ids["train"], 0, seed=SEEDS[1]))
            self.assertNotEqual([s29.augmentation_code(pid, 0, t, seed=SEEDS[0]) for t in range(1, 65)],
                                [s29.augmentation_code(pid, 0, t, seed=SEEDS[1]) for t in range(1, 65)])
            self.assertNotEqual(s29.patch_indices(pid, 0, 16, seed=SEEDS[0]), s29.patch_indices(pid, 0, 16, seed=SEEDS[1]))
            # model init depends on the run seed
            a = s29.build_model(None, SEEDS[0], device=torch.device("cpu"), tiny=True)
            b = s29.build_model(None, SEEDS[1], device=torch.device("cpu"), tiny=True)
            self.assertFalse(torch.equal(a.classifier.weight, b.classifier.weight))

        def test_test_png_never_loaded(self):
            d = Data(bundle, ids["train"][:4], ids["validation"][:2])
            tpid = ids["test"][0]
            with patch.object(s29, "verified_png_bytes", side_effect=AssertionError("must not read")):
                for part in ("train", "validation", "test"):
                    with self.assertRaises(s29.ExperimentError):
                        d.load_bag(tpid, part, 0, 64, SEEDS[0], False, torch.device("cpu"))
                with self.assertRaises(s29.ExperimentError):
                    d.read_png(tpid, bundle["patches"][tpid][0])
                with self.assertRaises(s29.ExperimentError):  # validation patient requested as train
                    d.load_bag(ids["validation"][0], "train", 0, 16, SEEDS[0], True, torch.device("cpu"))
                with self.assertRaises(s29.ExperimentError):  # train patient outside this run
                    d.load_bag(ids["train"][10], "train", 0, 16, SEEDS[0], True, torch.device("cpu"))
            with self.assertRaises(s29.ExperimentError):
                Data(bundle, ids["train"][:2] + ids["test"][:1], ids["validation"][:2])

        def test_loader_equals_stage29_on_pilot_patient(self):
            pid = sorted(p for p, r in bundle["pilot"].items() if r["split"] == "train")[0]
            d = Data(bundle, [pid], [])
            mine, y1 = d.load_bag(pid, "train", 3, 16, 20261113, True, torch.device("cpu"))
            ref, y2 = s29.load_bag(bundle, pid, 3, 16, True, torch.device("cpu"))
            self.assertTrue(torch.equal(mine, ref)); self.assertTrue(torch.equal(y1, y2))

        def test_train_eval_subset(self):
            labels = {p: bundle["patients"][p]["class_label"] for p in ids["train"]}
            a = select_train_eval(ids["train"], labels, 10)
            self.assertEqual(a, select_train_eval(list(reversed(ids["train"])), labels, 10))
            self.assertEqual(collections.Counter(labels[p] for p in a), {"LUAD": 10, "LUSC": 10})
            self.assertTrue(set(a) <= set(ids["train"]))

        def test_numpy_metrics_checkpoint_loads_weights_only(self):
            # regression for smoke attempt 1: Stage 29 binary_metrics returned a numpy scalar for numpy labels
            ces = [.7, .6]
            r = self.runner(self.dir / "n", ces, max_epochs=2)
            orig = r.validate_fn
            def numpy_validate(model, ep):
                m, p = orig(model, ep)
                y = np.array([0, 1, 0, 1]); pr = np.array([.2, .7, .4, .6])
                m["auroc"] = s29.binary_metrics(list(y), list(pr))["auroc"]  # numpy scalar on purpose
                m["ce"] = np.float64(m["ce"])
                return m, p
            r.validate_fn = numpy_validate
            r.start(); r.run()
            st = r.checkpoints.load()[0]  # torch.load(weights_only=True) inside Stage 29 Checkpoints
            self.assertIsInstance(st["progress"]["metrics"][1]["val_auroc"], float)
            self.assertEqual([x["epoch"] for x in st["progress"]["metrics"]], [0, 1, 2])  # loop kept writing to the saved state

        # ---------------- v1.0.1 resource-accounting regression tests (controlled clocks + fake memory peaks)
        def continuous(self, d):
            return self.sessions(d, self.CES, [{}])

        def test_res_continuous_expected_sums_and_peaks(self):
            r, status, snaps = self.continuous(self.dir / "c")
            self.assertEqual(status, "EARLY_STOP_PATIENCE")
            rows, sessions, summ = snaps[-1]
            self.assertRows(rows)
            self.assertEqual(len(sessions), 1); self.assertTrue(sessions[0]["clean_end"])
            self.assertFalse(summ["active_seconds_is_lower_bound"]); self.assertEqual(summ["total_train_attempts_replayed"], 0)
            self.assertAlmostEqual(summ["total_train_wall_seconds"], 4 * 13.0)
            self.assertSane(snaps)

        def test_res_stop_after_intermediate_checkpoint(self):
            _, status, snaps = self.sessions(self.dir / "a", self.CES, [dict(stop_after_attempts=2), {}])
            self.assertEqual(status, "EARLY_STOP_PATIENCE")
            self.assertRows(snaps[-1][0]); self.assertSane(snaps)
            self.assertTrue(snaps[-1][2]["active_seconds_is_lower_bound"]); self.assertEqual(len(snaps[-1][2]["unclean_sessions"]), 1)
            first = {r["epoch"]: r for r in snaps[0][0]}   # what the crashed session had made durable: 2 attempts + 1 checkpoint
            self.assertEqual((first[1]["train_attempts_executed_work"], first[1]["train_wall_seconds"], first[1]["checkpoints_measured"]), (2, 8.0, 1))

        def test_res_unsaved_attempts_replayed_and_counted(self):
            # checkpoint at attempt 2; attempt 3 (4 patients) is journaled, then crash -> rollback -> replayed
            _, status, snaps = self.sessions(self.dir / "u", self.CES, [dict(stop_after_attempts=3), {}])
            self.assertEqual(status, "EARLY_STOP_PATIENCE")
            self.assertRows(snaps[-1][0], {1: {"replay_patients": 4, "replay_attempts": 1}}); self.assertSane(snaps)

        def test_res_stop_after_training_before_validation(self):
            _, _, snaps = self.sessions(self.dir / "t", self.CES, [dict(stop_at_phase=(1, "trained")), {}])
            self.assertRows(snaps[-1][0]); self.assertSane(snaps)   # v1.0.0 lost epoch-1 training time here

        def test_res_stop_after_validation_before_commit(self):
            _, _, snaps = self.sessions(self.dir / "v", self.CES, [dict(stop_at_phase=(2, "validated")), {}])
            self.assertRows(snaps[-1][0]); self.assertSane(snaps)

        def test_res_crash_between_checkpoint_write_and_cost_event(self):
            # sequences: initial 0; epoch 1: attempt2=1, train_done=2, validated=3, commit=4
            _, status, snaps = self.sessions(self.dir / "x", self.CES, [dict(crash_after_save_seq=4), {}])
            self.assertEqual(status, "EARLY_STOP_PATIENCE")
            rows = {r["epoch"]: r for r in snaps[-1][0]}
            self.assertEqual((rows[1]["checkpoints_measured"], rows[1]["checkpoints_unmeasured"]), (3, 1))
            self.assertAlmostEqual(rows[1]["checkpoint_io_wall_seconds"], 1.5)          # unknown cost is not substituted
            self.assertTrue(rows[1]["checkpoint_io_wall_is_lower_bound"])
            self.assertEqual(rows[1]["train_attempts_replayed"], 0)                     # commit checkpoint was durable
            for k in (2, 3, 4):
                self.assertEqual((rows[k]["checkpoints_measured"], rows[k]["checkpoints_unmeasured"]), (4, 0))
            self.assertSane(snaps)
            again = self.sessions(self.dir / "x", self.CES, [{}])[2]   # reported once only, not re-added on reopen
            self.assertEqual({r["epoch"]: r for r in again[-1][0]}[1]["checkpoints_unmeasured"], 1)

        def test_res_truncated_journal_tail(self):
            d = self.dir / "j"
            self.sessions(d, self.CES, [dict(stop_after_attempts=2)])
            with (d / JOURNAL_NAME).open("ab") as f:
                f.write(b'{"session_id": "s0", "seq": 999, "kind": "train_att')       # torn last write, no newline
            _, status, snaps = self.sessions(d, self.CES, [{}])
            self.assertEqual(status, "EARLY_STOP_PATIENCE")
            rows, sessions, summ = snaps[-1]
            self.assertRows(rows)
            self.assertEqual(summ["journal_unparsable_lines"], 1); self.assertTrue(summ["active_seconds_is_lower_bound"])
            self.assertTrue(sessions[-1]["repaired_truncated_tail_at_start"])

        def test_res_reopen_completed_run_keeps_history(self):
            d = self.dir / "r"
            r, _, _ = self.continuous(d)
            emit(d, r.state)
            before = {n: (d / n).read_bytes() for n in ("resources_by_epoch.csv", "epoch_metrics.csv", "groups.csv", "validation_predictions.csv")}
            r2, status, snaps = self.sessions(d, self.CES, [{}])
            self.assertEqual(status, "EARLY_STOP_PATIENCE")
            emit(d, r2.state)
            for n, b in before.items():
                self.assertEqual((d / n).read_bytes(), b, n)
            sess = snaps[-1][1]
            self.assertEqual(len(sess), 2); self.assertTrue(sess[-1]["post_completion_session"]); self.assertFalse(sess[0]["post_completion_session"])

        def test_res_tables_rebuilt_after_locked_csv(self):
            d = self.dir / "l"
            r, _, _ = self.continuous(d)
            emit(d, r.state)
            good = (d / "resources_by_epoch.csv").read_bytes()
            (d / "resources_by_epoch.csv").write_bytes(b"stale")
            real = s29.atomic_bytes
            def locked(path, data):
                if Path(path).name == "resources_by_epoch.csv":
                    raise PermissionError("locked by Excel")
                return real(path, data)
            with patch.object(s29, "atomic_bytes", side_effect=locked), contextlib.redirect_stdout(io.StringIO()):
                emit(d, r.state)                                  # no exception
            self.assertEqual((d / "resources_by_epoch.csv").read_bytes(), b"stale")
            emit(d, r.state)                                      # rebuilt from the journal; no re-summing
            emit(d, r.state)
            self.assertEqual((d / "resources_by_epoch.csv").read_bytes(), good)

        def test_res_training_trajectory_unchanged_and_equal_to_v1_0_0(self):
            arch = ROOT / "results" / "stage31_v1_0_0_archive" / "31_train_patch_budget_baselines_v1.0.0.py"
            outs = {}
            for name, stops in (("cont", [{}]), ("unsaved", [dict(stop_after_attempts=3), {}]),
                                ("ckpt", [dict(crash_after_save_seq=4), {}]), ("trained", [dict(stop_at_phase=(1, "trained")), {}])):
                r, _, _ = self.sessions(self.dir / name, self.CES, stops)
                outs[name] = r.checkpoints.load()[0]
            for name in ("unsaved", "ckpt", "trained"):
                for k in ("model", "optimizer", "scaler", "rng"):
                    self.assertTrue(s29.tree_equal(outs["cont"][k], outs[name][k]), f"{name} {k}")
                for k in ("metrics", "groups", "predictions", "selection", "status"):
                    self.assertTrue(s29.tree_equal(outs["cont"]["progress"][k], outs[name]["progress"][k]), f"{name} {k}")
            if not arch.is_file():
                self.skipTest("v1.0.0 archive not present")
            self.assertEqual(sha_file(arch), "776a66428592079bfe882a385ee09f10c7a3541a765438ebc01c6cb4b41e6b4d")
            spec = importlib.util.spec_from_file_location("stage31_v100_ref", arch)
            old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
            old.s29, old.torch, old.np = s29, torch, np
            m, o = self.tiny()
            tr = [f"TCGA-AA-{i:04d}" for i in range(1, 14)]
            labels = {p: ("LUAD" if int(p[-4:]) % 2 == 0 else "LUSC") for p in tr}
            ces = self.CES
            ro = old.Runner(run_dir=self.dir / "old", identity={"t": 1}, model=m, optimizer=o, scaler=s29.scaler_for(torch.device("cpu")),
                            device=torch.device("cpu"), train_ids=tr, labels=labels, seed=20261113, budget=16, max_epochs=6, patience=2,
                            loader=synthetic_loader,
                            validate_fn=lambda model, ep: ({"ce": ces[ep - 1], "auroc": .5, "balanced_accuracy": .5, "macro_f1": .5},
                                                           [{"epoch": ep, "case_submitter_id": "X", "label": 0, "p_LUSC": .5}]),
                            train_eval_fn=lambda model: {"ce": float(sum(float(p.sum()) for p in model.parameters()))}, checkpoint_every=2)
            ro.frozen_bn = s29.bn_state(m)
            ro.start(); self.assertEqual(ro.run(), "EARLY_STOP_PATIENCE")
            ref = ro.checkpoints.load()[0]
            for k in ("model", "optimizer", "scaler"):
                self.assertTrue(s29.tree_equal(outs["cont"][k], ref[k]), f"v1.0.0 {k}")
            for k in ("metrics", "groups", "predictions", "selection", "status", "attempts_total"):
                self.assertTrue(s29.tree_equal(outs["cont"]["progress"][k], ref["progress"][k]), f"v1.0.0 {k}")

        def test_locked_csv_does_not_raise(self):
            with patch.object(s29, "atomic_bytes", side_effect=PermissionError("locked")), contextlib.redirect_stdout(io.StringIO()):
                self.assertFalse(write_table(self.dir / "x.csv", [{"a": 1}]))

    r = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    return 0 if r.wasSuccessful() else 1


# --------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    m = ap.add_mutually_exclusive_group(required=True)
    for f in ("--self-test", "--preflight-only", "--prepare", "--smoke", "--run"):
        m.add_argument(f, action="store_true")
    m.add_argument("--prepare-smoke", action="store_true", help=argparse.SUPPRESS)
    m.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--budget", type=int, choices=BUDGETS)
    ap.add_argument("--seed", type=int, choices=SEEDS)
    ap.add_argument("--output", type=Path, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--run-name", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--subset", default="full", choices=("full",) + tuple(SMOKE_SUBSETS), help=argparse.SUPPRESS)
    ap.add_argument("--max-epochs", type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--stop-after-attempts", type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--stop-at-phase", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()
    load_stage29()
    try:
        if args.self_test:
            return self_test()
        if args.preflight_only:
            preflight(); return 0
        if args.prepare:
            init_torch(True)
            prepare_folder(MAIN_OUTPUT, "MAIN", {"status": "FROZEN_FOR_MAIN_RUNS"}); return 0
        if args.prepare_smoke:
            init_torch(True)
            prepare_folder(SMOKE_OUTPUT, "SMOKE_TEST_ONLY", {"status": "TEST_ONLY_NOT_FOR_RESULTS", "smoke_subsets": SMOKE_SUBSETS}); return 0
        if args.smoke:
            return smoke()
        if args.run:
            require(args.budget and args.seed, "--run needs --budget and --seed")
            require(args.output is None and args.run_name is None and args.subset == "full" and args.max_epochs is None
                    and args.stop_after_attempts is None and args.stop_at_phase is None, "Main runs accept only --budget and --seed.")
            return cli_run(MAIN_OUTPUT, args.budget, args.seed)
        if args.worker:
            require(args.output is not None and args.output.resolve() == SMOKE_OUTPUT.resolve(), "Workers run only in the smoke folder.")
            phase = None
            if args.stop_at_phase:
                e, p = args.stop_at_phase.split(":"); phase = (int(e), p)
            return cli_run(SMOKE_OUTPUT, args.budget, args.seed, args.run_name, args.subset, args.max_epochs, args.stop_after_attempts, phase)
    except s29.ExperimentError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nINTERRUPTED. Repeat the same command to resume from the last verified checkpoint.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
