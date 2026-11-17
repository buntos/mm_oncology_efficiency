#!/usr/bin/env python3
r"""Stage 29 v1.0.0: GPU smoke checks and a DEVELOPMENT pilot, not the main study.

Run from C:\research\mm_oncology_efficiency with the project .venv Python:
  python scripts\29_train_patch_budget_baselines.py --preflight-only
  python scripts\29_train_patch_budget_baselines.py --prepare
  python scripts\29_train_patch_budget_baselines.py --smoke
  python scripts\29_train_patch_budget_baselines.py --pilot --budget 64
  python scripts\29_train_patch_budget_baselines.py --pilot

--prepare downloads official ResNet18 ImageNet weights only if needed (~45 MB).
No SVS/RNA downloads. No test scoring. Original Stage 28 files stay immutable.
Pilot: 60 train + 20 validation, K=64/32/16, 5 epochs, seed 20261113.
Each budget starts in an isolated process from the same pretrained encoder/head
initialization. Resume uses verified atomic checkpoints at optimizer boundaries.
Repeat --pilot to continue after an interruption. Completed budgets are skipped.
This version deliberately has no command to launch the main 929-patient series;
training settings must be reviewed after the pilot and frozen separately.

Tests: --self-test (stdlib/numpy/Pillow); --self-test-torch (CPU, no downloads).
Sources: torchvision ResNet18 docs; PyTorch 2.12 AMP, torch.load and randomness
documentation. Runtime uses the user's installed versions without pip changes.
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True
import argparse
import collections
import contextlib
import csv
import gc
import hashlib
import importlib.metadata
import io
import json
import math
import os
import platform
import random
import re
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

VERSION = "1.0.0"
STAGE28_NAME = "wsi_patch_budget_v1"
STAGE29_NAME = "stage29_pilot_v1"
NAMESPACE = "production_final_v2_tissue080"
SEED = 20261113
EPOCHS = 5
BUDGETS = (64, 32, 16)
ACCUMULATION = 4
CPU_THREADS = 4
CHECKPOINT_STEPS = 5
WEIGHTS_NAME = "resnet18-f37072fd.pth"
WEIGHTS_URL = "https://download.pytorch.org/models/" + WEIGHTS_NAME
EXPECTED_DATASET = "78d439493d4d882f49afc1027eff5f6c92033fab52560989e447fb73ca5039c9"
EXPECTED_SPLIT = "8dbca5ef9368b8e29d50f7183d93d9e893fefa2b01eaf58b2812c055b0a00a8b"
EXPECTED_FILES = {
    "environment.json": "07fe41ee36f3b8d1ceb04b5cca9ee9466149ec580130a50f9cdcd3207e2fb1ec",
    "input_audit.json": "77feb67ff60e65b41da56ed249de34c43a5655fe329a525024ca5e3bf8bb4f87",
    "patch_inventory.csv": "4b992911cf23a57a958634a92911d51a5c60f087973c3697ff59319bb5e2c7f0",
    "patient_splits.csv": "a554ccbb514113dcd2ac5ad7cdb175c9d8e03341b4030657b6cdd8104765dc42",
    "pilot_manifest.csv": "52e049fe25a0bdb3adb38df2ad004b0b7b4d89b03df373bafd3d954071d0f783",
    "protocol.json": "fca0e10b1136b2eaa26d4a7fd4eca572e6a567ccd15927b912e79c63486bb7a5",
    "run_status.json": "c1de13b9df3394bfb4cbea30791c628817b607b312466533d696eadc1fa6bbd0",
    "source_inputs.json": "f8d925f51a30a31f36d8a98a02a718e258343c0a4908243ccb3e834d25fb6e62",
}
ID_RE = re.compile(r"TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}\Z")
torch = None
np = None
tv = None


class ExperimentError(Exception):
    pass


def require(condition, message):
    if not condition:
        raise ExperimentError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def read_json(path):
    with Path(path).open(encoding="utf-8-sig") as f:
        return json.load(f)


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        require(r.fieldnames and len(set(r.fieldnames)) == len(r.fieldnames), f"Invalid CSV header: {path}")
        rows = list(r)
    require(all(None not in x and all(v is not None for v in x.values()) for x in rows), f"Malformed CSV: {path}")
    return rows


def atomic_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    try:
        with tmp.open("xb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def atomic_json(path, value):
    atomic_bytes(path, json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False).encode() + b"\n")


def export_csv(path, rows):
    if not rows:
        return
    text = io.StringIO(newline="")
    writer = csv.DictWriter(text, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    try:
        atomic_bytes(path, text.getvalue().encode("utf-8"))
    except PermissionError:
        print(f"WARNING: close CSV in Excel and rerun to rebuild it: {path}. Checkpoints are preserved.", flush=True)


@contextlib.contextmanager
def execution_lock(path):
    """OS lock is released even after power loss; no stale-lock deletion needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    f = path.open("a+b")
    acquired = False
    try:
        if path.stat().st_size == 0:
            f.write(b"0")
            f.flush()
        f.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            acquired = True
        except OSError:
            raise ExperimentError("Another Stage 29 run is active in this output directory.") from None
        yield
    finally:
        if acquired:
            f.seek(0)
            if os.name == "nt":
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        f.close()


def keyed(rows):
    result = {r["case_submitter_id"]: r for r in rows}
    require(len(result) == len(rows), "Duplicate patient ID in manifest.")
    return result


def load_bundle(root):
    """Validate the received Stage 28 snapshot; never import/execute old scripts."""
    base = root / "results" / STAGE28_NAME
    hashes = read_json(base / "file_hashes.json")
    require(hashes.get("schema_version") == 1 and hashes.get("files") == EXPECTED_FILES,
            "Stage 28 differs from the reviewed frozen snapshot. Do not regenerate splits; send its protocol for review.")
    for name, expected in EXPECTED_FILES.items():
        path = base / name
        require(path.is_file() and not path.is_symlink() and sha_file(path) == expected, f"Frozen Stage 28 file changed/missing: {path}")
    protocol, audit = read_json(base / "protocol.json"), read_json(base / "input_audit.json")
    require(protocol["identity"] == hashes["identity"], "Stage 28 identity disagreement.")
    require(protocol["identity"]["dataset_fingerprint_sha256"] == EXPECTED_DATASET and
            protocol["identity"]["split_sha256"] == EXPECTED_SPLIT, "Unexpected data/split fingerprint.")
    require(protocol["data_status"] == "FROZEN" and audit["status"] == "PASS" and
            read_json(base / "run_status.json")["status"] == "PREPARED", "Stage 28 is not prepared.")
    require(protocol["namespace"] == NAMESPACE and protocol["patches_per_patient"] == 64 and
            protocol["tile_size"] == [256, 256] and protocol["tissue_threshold"] == .8, "Production protocol mismatch.")
    patients = keyed(read_csv(base / "patient_splits.csv"))
    pilot = keyed(read_csv(base / "pilot_manifest.csv"))
    require(len(patients) == 929 and len(pilot) == 80, "Unexpected patient/pilot counts.")
    split = {p: r["split"] for p, r in patients.items()}
    require(digest(split) == EXPECTED_SPLIT, "Patient assignment fingerprint mismatch.")
    expected_counts = {("train", "LUAD"): 30, ("train", "LUSC"): 30,
                       ("validation", "LUAD"): 10, ("validation", "LUSC"): 10}
    require(collections.Counter((r["split"], r["class_label"]) for r in pilot.values()) == expected_counts,
            "Pilot must contain 60 train + 20 validation and no test patients.")
    for pid, r in pilot.items():
        require(ID_RE.fullmatch(pid) and pid in patients and r["split"] == patients[pid]["split"]
                and r["class_label"] == patients[pid]["class_label"] and r["split"] != "test", f"Pilot leakage/mismatch: {pid}")
    by_patient = collections.defaultdict(list)
    tiles_root = (root / "data" / "tiles" / NAMESPACE).resolve()
    for r in read_csv(base / "patch_inventory.csv"):
        pid, index = r["case_submitter_id"], int(r["tile_index"])
        require(pid in patients and r["class_label"] == patients[pid]["class_label"], "Inventory label/patient mismatch.")
        relative = PurePosixPath(r["relative_path"])
        require(len(relative.parts) == 2 and relative.parts[0] == pid and ".." not in relative.parts
                and not relative.is_absolute(), "Unsafe inventory relative path.")
        x, y = int(r["x_level0"]), int(r["y_level0"])
        require(relative.name == f"tile_{index:03d}_x{x}_y{y}.png", "Inventory name/coordinate mismatch.")
        path = (tiles_root / relative.parts[0] / relative.parts[1]).resolve()
        require(path.is_relative_to(tiles_root), f"PNG path escapes production root: {path}")
        r.update(index=index, path=path, bytes=int(r["bytes"]))
        by_patient[pid].append(r)
    require(set(by_patient) == set(patients), "Inventory patient set mismatch.")
    for pid, rows in by_patient.items():
        rows.sort(key=lambda r: r["index"])
        require([r["index"] for r in rows] == list(range(1, 65)), f"Expected indices 1..64: {pid}")
    return {"root": root, "base": base, "output": base / STAGE29_NAME, "protocol": protocol,
            "audit": audit, "patients": patients, "pilot": pilot, "patches": dict(by_patient),
            "hashes": hashes, "manifest_sha256": sha_file(base / "file_hashes.json")}


def pilot_ids(bundle, part):
    require(part in ("train", "validation"), "Test patients cannot be requested by Stage 29.")
    return sorted(p for p, r in bundle["pilot"].items() if r["split"] == part)


def verified_png_bytes(row):
    raw = row["path"].read_bytes()
    require(len(raw) == row["bytes"] and hashlib.sha256(raw).hexdigest() == row["file_sha256"],
            f"PNG changed since Stage 28: {row['path']}")
    return raw


def check_pilot_files(bundle):
    done = 0
    for pid in sorted(bundle["pilot"]):
        for row in bundle["patches"][pid]:
            verified_png_bytes(row)
            done += 1
        if done % 1024 == 0 or done == 5120:
            print(f"  Pilot PNG hashes: {done}/5120", flush=True)


def hash_key(*parts):
    return hashlib.sha256("|".join(map(str, parts)).encode("ascii")).digest()


def patch_indices(pid, epoch, budget, seed=SEED):
    require(budget in BUDGETS and epoch >= 0, "Invalid patch budget/epoch.")
    permutation = sorted(range(64), key=lambda i: (hash_key("patch-cycle-v1", seed, pid, epoch // 4, i), i))
    start = (epoch % 4) * 16
    rotated = permutation[start:] + permutation[:start]
    return rotated[:budget]


def patient_order(ids, epoch, seed=SEED):
    return sorted(ids, key=lambda p: (hash_key("patient-order-v1", seed, epoch, p), p))


def augmentation_code(pid, epoch, tile_index, seed=SEED):
    return hash_key("augment-d4-v1", seed, pid, epoch, tile_index)[0] % 8


def group_slices(count, start=0, accumulation=ACCUMULATION):
    require(0 <= start <= count and (start % accumulation == 0 or start == count), "Resume position is inside accumulated gradients.")
    return [(i, min(i + accumulation, count)) for i in range(start, count, accumulation)]


def binary_metrics(labels, scores):
    require(len(labels) == len(scores) and len(labels) > 0 and set(labels) == {0, 1}, "Metrics require both classes.")
    require(all(math.isfinite(s) and 0 <= s <= 1 for s in scores), "Non-finite probability.")
    npos, nneg = sum(labels), len(labels) - sum(labels)
    ordered = sorted(zip(scores, labels))
    rank_sum, i = 0.0, 0
    while i < len(ordered):
        j = i + 1
        while j < len(ordered) and ordered[j][0] == ordered[i][0]:
            j += 1
        rank_sum += ((i + 1 + j) / 2) * sum(label for _, label in ordered[i:j])
        i = j
    auroc = (rank_sum - npos * (npos + 1) / 2) / (npos * nneg)
    pred = [int(s >= .5) for s in scores]
    tp = sum(y == 1 and p == 1 for y, p in zip(labels, pred))
    tn = sum(y == 0 and p == 0 for y, p in zip(labels, pred))
    fp, fn = nneg - tn, npos - tp
    f1_pos = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0
    f1_neg = 2 * tn / (2 * tn + fp + fn) if 2 * tn + fp + fn else 0
    return {"auroc": auroc, "balanced_accuracy": .5 * (tp / npos + tn / nneg),
            "macro_f1": .5 * (f1_pos + f1_neg), "accuracy": (tp + tn) / len(labels)}


def initialize_torch(cuda=True):
    global torch, np, tv
    # Set BEFORE importing/initializing CUDA. Do not silently alter an incompatible value.
    current = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    require(current in (None, ":4096:8", ":16:8"), "Unsupported existing CUBLAS_WORKSPACE_CONFIG.")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch as torch_module
    import numpy as numpy_module
    import torchvision as vision_module
    torch, np, tv = torch_module, numpy_module, vision_module
    torch.set_num_threads(CPU_THREADS)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if cuda:
        require(torch.cuda.is_available(), "CUDA unavailable. Use the existing project .venv; do not change packages blindly.")
        torch.cuda.set_device(0)
    return torch.device("cuda:0" if cuda else "cpu")


def environment_record(bundle=None):
    versions = {p: importlib.metadata.version(p) for p in ("torch", "torchvision", "numpy", "Pillow", "psutil")}
    if bundle is not None:
        old = read_json(bundle["base"] / "environment.json")["versions"]
        require(all(versions[p] == old[p] for p in versions), "Package versions changed since Stage 28; review the environment first.")
    p = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None
    return {"python": platform.python_version(), "platform": platform.platform(), "versions": versions,
            "gpu": p.name if p else None, "gpu_memory_bytes": p.total_memory if p else None,
            "cuda_runtime": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "cpu_threads": CPU_THREADS, "interop_threads": 1, "workers": 0,
            "deterministic_algorithms": True, "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG")}


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(weights_file, seed=SEED, device=None, tiny=False):
    seed_everything(seed)
    nn = torch.nn

    class GatedMIL(nn.Module):
        def __init__(self):
            super().__init__()
            if tiny:
                self.encoder = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU(),
                                             nn.AdaptiveAvgPool2d(1), nn.Flatten())
                features = 8
            else:
                encoder = tv.models.resnet18(weights=None)
                if weights_file is not None:
                    encoder.load_state_dict(torch.load(weights_file, map_location="cpu", weights_only=True), strict=True)
                encoder.fc = nn.Identity()
                self.encoder, features = encoder, 512
            self.attention_v = nn.Linear(features, 128)
            self.attention_u = nn.Linear(features, 128)
            self.attention_w = nn.Linear(128, 1)
            self.classifier = nn.Linear(features, 2)
            self.train(True)

        def train(self, mode=True):
            super().train(mode)
            for module in self.encoder.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.eval()  # Running stats fixed; affine parameters still trainable.
            return self

        def forward(self, x):
            h = self.encoder(x)
            # CNN is autocast; attention, pooling and classifier stay FP32.
            with torch.autocast(device_type=h.device.type, enabled=False):
                h = h.float()
                scores = self.attention_w(torch.tanh(self.attention_v(h)) * torch.sigmoid(self.attention_u(h)))
                attention = torch.softmax(scores.squeeze(-1), dim=0)
                pooled = torch.sum(attention[:, None] * h, dim=0, keepdim=True)
                return self.classifier(pooled)

    model = GatedMIL().to(device or torch.device("cpu"))
    require(all(p.requires_grad for p in model.encoder.parameters()), "Encoder is unexpectedly frozen.")
    return model


def optimizer_for(model):
    head = [p for name, p in model.named_parameters() if not name.startswith("encoder.")]
    return torch.optim.AdamW([{"params": model.encoder.parameters(), "lr": .0001},
                             {"params": head, "lr": .001}], weight_decay=.0001, foreach=False)


def scaler_for(device):
    return torch.amp.GradScaler("cuda", init_scale=1024.0, enabled=device.type == "cuda")


def load_bag(bundle, pid, epoch, budget, training, device):
    from PIL import Image
    require(pid in bundle["pilot"] and bundle["pilot"][pid]["split"] == ("train" if training else "validation"),
            f"Attempt to load a patient outside the permitted pilot split: {pid}")
    indices = patch_indices(pid, epoch, budget) if training else list(range(64))
    require(training or budget == 64, "Validation must use all 64 patches.")
    images = []
    for i in indices:
        row = bundle["patches"][pid][i]
        raw = verified_png_bytes(row)
        with Image.open(io.BytesIO(raw)) as im:
            require(im.format == "PNG" and im.mode == "RGB" and im.size == (256, 256), "Unexpected PNG format.")
            im.load()
            array = np.asarray(im)
        if training:
            code = augmentation_code(pid, epoch, row["index"])
            array = np.rot90(array, code % 4)
            if code // 4:
                array = np.flip(array, axis=1)
        images.append(torch.from_numpy(np.array(array, copy=True, order="C")).permute(2, 0, 1))
    x = torch.stack(images).float().div_(255.)
    x.sub_(torch.tensor([.485, .456, .406]).view(1, 3, 1, 1))
    x.div_(torch.tensor([.229, .224, .225]).view(1, 3, 1, 1))
    label = 0 if bundle["patients"][pid]["class_label"] == "LUAD" else 1
    return x.to(device), torch.tensor([label], device=device, dtype=torch.long)


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def bn_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            if k.endswith(("running_mean", "running_var", "num_batches_tracked"))}


def tree_equal(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a.cpu(), b.cpu())
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(tree_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(tree_equal(x, y) for x, y in zip(a, b))
    return a == b


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(v) for v in value)
    return value


def capture_rng():
    n = np.random.get_state()
    return {"python": random.getstate(), "numpy": [n[0], n[1].tolist(), int(n[2]), int(n[3]), float(n[4])],
            "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def snapshot(model, optimizer, scaler, state):
    return {"model": cpu_tree(model.state_dict()), "optimizer": cpu_tree(optimizer.state_dict()),
            "scaler": scaler.state_dict(), "rng": capture_rng(), "progress": state}


def restore(snapshot_data, model, optimizer, scaler):
    model.load_state_dict(snapshot_data["model"], strict=True)
    optimizer.load_state_dict(snapshot_data["optimizer"])
    scaler.load_state_dict(snapshot_data["scaler"])
    optimizer.zero_grad(set_to_none=True)
    restore_rng(snapshot_data["rng"])
    return snapshot_data["progress"]


def train_group(model, optimizer, scaler, ids, loader, device, strict=False):
    """One optimizer update, mean PATIENT loss, including a partial final group."""
    require(1 <= len(ids) <= ACCUMULATION, "Invalid accumulation group.")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    synchronize(device)
    begin, cpu_begin = time.perf_counter(), time.process_time()
    load_seconds, loss_sum, patch_count, cuda_ms = 0., 0., 0, 0.
    for pid in ids:
        t = time.perf_counter()
        x, target = loader(pid)
        synchronize(device)
        load_seconds += time.perf_counter() - t
        events = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) if device.type == "cuda" else None
        if events:
            events[0].record()
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(logits, target)
        require(bool(torch.isfinite(loss).item()), f"Non-finite loss for {pid}")
        scaler.scale(loss / len(ids)).backward()
        if events:
            events[1].record()
        synchronize(device)
        if events:
            cuda_ms += events[0].elapsed_time(events[1])
        loss_sum += float(loss.detach().item())
        patch_count += len(x)
        del x, target, logits, loss
    scaler.unscale_(optimizer)
    grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    require(grads, "No model gradients.")
    finite = bool(torch.stack([torch.isfinite(g).all() for g in grads]).all().item())
    encoder_grad = next(model.encoder.parameters()).grad
    encoder_nonzero = encoder_grad is not None and bool(torch.count_nonzero(encoder_grad).item())
    if strict:
        require(finite and encoder_nonzero, "Smoke: non-finite or zero encoder gradient.")
    old_scale = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    skipped = int(scaler.get_scale() < old_scale)
    require(finite or skipped, "Non-finite gradient reached optimizer without an AMP skip.")
    optimizer.zero_grad(set_to_none=True)
    synchronize(device)
    return {"patients": len(ids), "patches": patch_count, "loss_sum": loss_sum,
            "wall_seconds": time.perf_counter() - begin, "cpu_seconds": time.process_time() - cpu_begin,
            "load_and_transfer_seconds": load_seconds, "forward_backward_cuda_ms": cuda_ms,
            "attempted_optimizer_steps": 1, "optimizer_steps": 1 - skipped, "amp_skips": skipped,
            "finite_gradients": finite, "encoder_gradient_nonzero": encoder_nonzero}


class MemoryMonitor:
    def __init__(self, device):
        self.device, self.peak_rss, self.stop = device, 0, threading.Event()
        self.thread = None
    def __enter__(self):
        import psutil
        self.process = psutil.Process()
        if self.device.type == "cuda":
            synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        def sample():
            while not self.stop.wait(.05):
                self.peak_rss = max(self.peak_rss, self.process.memory_info().rss)
        self.peak_rss = self.process.memory_info().rss
        self.thread = threading.Thread(target=sample, daemon=True)
        self.thread.start()
        return self
    def values(self):
        return {"sampled_peak_rss_bytes": max(self.peak_rss, self.process.memory_info().rss),
                "peak_allocated_vram_bytes": torch.cuda.max_memory_allocated(self.device) if self.device.type == "cuda" else 0,
                "peak_reserved_vram_bytes": torch.cuda.max_memory_reserved(self.device) if self.device.type == "cuda" else 0}
    def __exit__(self, *exc):
        synchronize(self.device)
        self.stop.set()
        self.thread.join()


class SessionLedger:
    """Separate real cost ledger: replayed work counts; an unclean tail is unknown."""
    def __init__(self, directory):
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / (now().replace(":", "-") + "_" + uuid.uuid4().hex + ".jsonl")
        self.started, self.cpu_started = time.perf_counter(), time.process_time()
        self.f = self.path.open("x", encoding="utf-8")
        self.record("START")
    def record(self, event, **detail):
        row = {"event": event, "utc": now(), "active_seconds": time.perf_counter() - self.started,
               "cpu_seconds": time.process_time() - self.cpu_started, **detail}
        self.f.write(json.dumps(row, allow_nan=False) + "\n")
        self.f.flush()
        os.fsync(self.f.fileno())
    def close(self, outcome):
        self.record("END", outcome=outcome)
        self.f.close()


def ledger_summary(directory):
    total, cpu, unclean = 0., 0., []
    files = sorted(directory.glob("*.jsonl")) if directory.exists() else []
    for path in files:
        last = None
        with path.open(encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                except ValueError:
                    break  # Interrupted trailing write: use last durable heartbeat.
                last = row
        if last:
            total += last["active_seconds"]
            cpu += last["cpu_seconds"]
        if not last or last["event"] != "END":
            unclean.append(path.name)
    return {"recorded_active_seconds_including_replay": total, "recorded_cpu_seconds": cpu,
            "sessions": len(files), "unclean_sessions": unclean, "time_is_lower_bound": bool(unclean),
            "note": "Power-off pauses excluded. Unclean tail after last durable heartbeat is unknown, not zero."}


class Checkpoints:
    """Immutable atomic generations; latest+previous+best retained."""
    def __init__(self, directory, identity):
        self.directory, self.identity = directory, identity
        self.last_valid = None
    def generations(self):
        return sorted(self.directory.glob("checkpoint_[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]")) if self.directory.exists() else []
    def save(self, model, optimizer, scaler, state, mark_best=False):
        self.directory.mkdir(parents=True, exist_ok=True)
        existing = self.generations()
        require(not existing or self.last_valid is not None, "Load existing checkpoints before saving a new generation.")
        sequence = int(existing[-1].name.split("_")[-1]) + 1 if existing else 0
        progress = dict(state)
        if mark_best:
            progress["best_sequence"] = sequence
        tmp = Path(tempfile.mkdtemp(prefix=".checkpoint-", dir=self.directory))
        final = self.directory / f"checkpoint_{sequence:08d}"
        try:
            data = snapshot(model, optimizer, scaler, progress)
            with (tmp / "state.pt").open("xb") as f:
                torch.save(data, f)
                f.flush()
                os.fsync(f.fileno())
            del data
            atomic_json(tmp / "commit.json", {"identity": self.identity, "sequence": sequence,
                        "sha256": sha_file(tmp / "state.pt"), "epoch": progress.get("epoch"),
                        "position": progress.get("position"), "best_sequence": progress.get("best_sequence"), "utc": now()})
            os.rename(tmp, final)
        except BaseException:
            if tmp.exists():
                shutil.rmtree(tmp)
            raise
        state.update(progress)
        # Preserve the last VERIFIED generation, not a corrupt newer directory
        # from which load() may have just fallen back.
        keep = {final.name}
        if self.last_valid is not None:
            keep.add(self.last_valid.name)
        if state.get("best_sequence") is not None:
            keep.add(f"checkpoint_{state['best_sequence']:08d}")
        for old in self.generations():
            if old.name not in keep:
                shutil.rmtree(old)
        self.last_valid = final
        return final
    def load(self):
        problems = []
        for path in reversed(self.generations()):
            try:
                meta = read_json(path / "commit.json")
                require(meta["identity"] == self.identity, "Checkpoint configuration identity mismatch.")
                require(sha_file(path / "state.pt") == meta["sha256"], "Checkpoint checksum mismatch.")
                state = torch.load(path / "state.pt", map_location="cpu", weights_only=True)
                if problems:
                    print("WARNING: using previous verified checkpoint; latest invalid: " + "; ".join(problems), flush=True)
                self.last_valid = path
                return state, path
            except (OSError, ValueError, KeyError, RuntimeError, ExperimentError) as e:
                problems.append(f"{path.name}: {e}")
        require(not problems, "No valid checkpoint remains: " + "; ".join(problems))
        return None, None


def prepare(bundle, environment):
    output = bundle["output"]
    if output.exists():
        read_prepared(bundle, environment)
        print(f"ALREADY PREPARED: {output}")
        return
    require(shutil.disk_usage(bundle["root"]).free >= 3 * 1024**3, "Need at least 3 GiB free for pilot checkpoints and temporary files.")
    require(tv.models.ResNet18_Weights.IMAGENET1K_V1.url == WEIGHTS_URL, "Unexpected torchvision weights URL.")
    staging = Path(tempfile.mkdtemp(prefix=".stage29-prepare-", dir=bundle["base"]))
    try:
        weights = staging / "weights" / WEIGHTS_NAME
        weights.parent.mkdir()
        cache = Path(torch.hub.get_dir()) / "checkpoints" / WEIGHTS_NAME
        if cache.is_file():
            require(sha_file(cache).startswith("f37072fd"), "Cached ResNet18 weights failed the official hash prefix.")
            shutil.copyfile(cache, weights)
            print("Using verified cached ImageNet weights.", flush=True)
        else:
            print(f"Downloading official ResNet18 weights (~45 MB): {WEIGHTS_URL}", flush=True)
            torch.hub.download_url_to_file(WEIGHTS_URL, str(weights), hash_prefix="f37072fd", progress=True)
        weights_hash = sha_file(weights)
        require(weights_hash.startswith("f37072fd"), "Weights hash mismatch.")
        # CPU architecture/state compatibility check; never a training forward.
        temporary_model = build_model(weights, device=torch.device("cpu"))
        del temporary_model
        config = {
            "schema_version": 1, "stage29_version": VERSION, "purpose": "DEVELOPMENT_PILOT_ONLY",
            "script_sha256": sha_file(Path(__file__)), "stage28_files": EXPECTED_FILES,
            "stage28_manifest_sha256": bundle["manifest_sha256"], "dataset_fingerprint": EXPECTED_DATASET,
            "split_fingerprint": EXPECTED_SPLIT, "weights_sha256": weights_hash, "weights_url": WEIGHTS_URL,
            "environment": environment, "seed": SEED, "epochs": EPOCHS, "budgets": list(BUDGETS),
            "train_patients": pilot_ids(bundle, "train"), "validation_patients": pilot_ids(bundle, "validation"),
            "test_used": False, "validation_patch_budget": 64,
            "model": "ResNet18 trainable encoder + gated attention 128 + patient classifier 2",
            "batchnorm": "Frozen running stats; trainable affine parameters", "encoder_precision": "AMP FP16",
            "attention_pool_classifier_precision": "FP32", "grad_scaler_initial_scale": 1024.,
            "patient_batch_size": 1, "gradient_accumulation_patients": ACCUMULATION,
            "optimizer": "AdamW foreach=False", "encoder_lr": .0001, "head_lr": .001,
            "weight_decay": .0001, "input_size": [256, 256], "input_normalization": "ImageNet mean/std; no crop",
            "augmentation": "SHA256 keyed per patient/epoch/tile D4 rotations/reflection; independent of K",
            "sampling": "SHA256 ranked 64 per four-epoch cycle; rotating quarter; B16 subset of B32 subset of B64",
            "workers": 0, "pin_memory": False, "cpu_threads": CPU_THREADS,
            "checkpoint_every_optimizer_attempts": CHECKPOINT_STEPS,
            "cost_scope": "Model setup, PNG read/hash/decode, train, validation, checkpoint I/O; preflight/weight download reported separately",
            "checkpoint_rule": "Atomic generations at optimizer boundaries; preserve latest, previous and best",
            "quality_warning": "Pilot scores are development diagnostics; no publication-level effectiveness conclusion",
        }
        atomic_json(staging / "pilot_protocol.json", config)
        atomic_json(staging / "preparation.json", {"status": "PREPARED", "utc": now(),
                    "pilot_protocol_sha256": sha_file(staging / "pilot_protocol.json"), "weights_sha256": weights_hash})
        require(not output.exists(), "Stage 29 output appeared during preparation; refusing to replace it.")
        os.rename(staging, output)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    print(f"PREPARED: {output}\nNext: --smoke. No model training has run yet.")


def read_prepared(bundle, environment):
    output = bundle["output"]
    require((output / "preparation.json").is_file(), "Stage 29 not prepared. Run --prepare first.")
    record, protocol = read_json(output / "preparation.json"), read_json(output / "pilot_protocol.json")
    require(record["status"] == "PREPARED" and sha_file(output / "pilot_protocol.json") == record["pilot_protocol_sha256"],
            "Pilot protocol changed since preparation.")
    require(protocol["script_sha256"] == sha_file(Path(__file__)), "Stage 29 code changed after preparation. Review/version the pilot; no automatic reset.")
    require(protocol["stage28_manifest_sha256"] == bundle["manifest_sha256"] and protocol["stage28_files"] == EXPECTED_FILES,
            "Prepared pilot refers to another Stage 28 snapshot.")
    if environment is not None:
        require(protocol["environment"] == environment, "Pilot execution environment differs from preparation.")
    weights = output / "weights" / WEIGHTS_NAME
    require(sha_file(weights) == record["weights_sha256"] == protocol["weights_sha256"], "Prepared weights changed.")
    return protocol, record["pilot_protocol_sha256"], weights


def smoke_ids(bundle):
    classes = {label: sorted((r for r in bundle["pilot"].values() if r["split"] == "train" and r["class_label"] == label),
                            key=lambda r: int(r["pilot_order_within_class"])) for label in ("LUAD", "LUSC")}
    return [classes["LUAD"][0]["case_submitter_id"], classes["LUSC"][0]["case_submitter_id"],
            classes["LUAD"][1]["case_submitter_id"], classes["LUSC"][1]["case_submitter_id"],
            classes["LUAD"][2]["case_submitter_id"]]


def save_tensor_file(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    try:
        with tmp.open("xb") as f:
            torch.save(value, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return sha_file(path)


def smoke_worker(bundle, identity, weights, device, budget, resume_check=False):
    output = bundle["output"] / "smoke"
    output.mkdir(exist_ok=True)
    ids = smoke_ids(bundle)
    loader = lambda pid: load_bag(bundle, pid, 0, budget, True, device)
    model = build_model(weights, device=device)
    optimizer, scaler = optimizer_for(model), scaler_for(device)
    proof_meta = output / "resume_proof.json"
    if resume_check:
        require(budget == 64, "Process resume proof is performed for K=64.")
        meta = read_json(proof_meta)
        require(meta["identity"] == identity, "Resume proof configuration mismatch.")
        for name in ("boundary.pt", "continuous.pt"):
            require(sha_file(output / name) == meta["hashes"][name], "Smoke proof checkpoint changed.")
        boundary = torch.load(output / "boundary.pt", map_location="cpu", weights_only=True)
        restore(boundary, model, optimizer, scaler)
        del boundary
        result = train_group(model, optimizer, scaler, ids[4:], loader, device, strict=True)
        resumed = snapshot(model, optimizer, scaler, {})
        expected = torch.load(output / "continuous.pt", map_location="cpu", weights_only=True)
        checks = {key: tree_equal(resumed[key], expected[key]) for key in ("model", "optimizer", "scaler", "rng")}
        require(all(checks.values()), f"Fresh-process resume did not reproduce the continuous run: {checks}")
        report = {"status": "PASS", "identity": identity, "utc": now(), "budget": 64,
                  "fresh_process_resume_equal": checks, "tail_group_size": 1, "loss_sum": result["loss_sum"]}
        atomic_json(output / "resume_check.json", report)
        print("RESUME PASS: model/optimizer/scaler/RNG exactly match after a fresh-process continuation.", flush=True)
        return
    first_weight = next(model.encoder.parameters())
    initial = first_weight.detach().cpu().clone()
    initial_head = model.classifier.weight.detach().cpu().clone()
    bn_before = bn_state(model)
    reports = []
    with MemoryMonitor(device) as memory:
        for start, end in group_slices(len(ids)):
            reports.append(train_group(model, optimizer, scaler, ids[start:end], loader, device, strict=True))
            require(reports[-1]["amp_skips"] == 0, "Smoke AMP step was skipped. Stop and review numerical settings.")
            if budget == 64 and end == 4:
                save_tensor_file(output / "boundary.pt", snapshot(model, optimizer, scaler, {"position": 4}))
        resources = memory.values()
    require(not torch.equal(first_weight.detach().cpu(), initial), "CNN encoder weights did not change.")
    require(not torch.equal(model.classifier.weight.detach().cpu(), initial_head), "MIL classifier weights did not change.")
    require(tree_equal(bn_before, bn_state(model)), "BatchNorm running statistics changed.")
    model.eval()
    model.train()
    require(all(not m.training for m in model.encoder.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)),
            "BatchNorm mode switched incorrectly.")
    if budget == 64:
        save_tensor_file(output / "continuous.pt", snapshot(model, optimizer, scaler, {}))
        atomic_json(proof_meta, {"identity": identity, "hashes": {
            name: sha_file(output / name) for name in ("boundary.pt", "continuous.pt")}})
    report = {"status": "PASS", "identity": identity, "utc": now(), "budget": budget,
              "patient_ids": ids, "groups": [4, 1], "encoder_updated": True, "classifier_updated": True,
              "batchnorm_buffers_unchanged": True, "steps": reports, "resources": resources,
              "note": "Smoke includes diagnostics and is not an unbiased timing benchmark."}
    atomic_json(output / f"smoke_B{budget}.json", report)
    print(f"SMOKE B{budget} PASS: finite gradients, CNN update, frozen BN; peak allocated VRAM={resources['peak_allocated_vram_bytes']/1024**3:.3f} GiB", flush=True)


def require_smoke(bundle, identity, budget):
    folder = bundle["output"] / "smoke"
    for name in (f"smoke_B{budget}.json", "resume_check.json"):
        require((folder / name).is_file(), "Run --smoke successfully before the pilot.")
        report = read_json(folder / name)
        require(report["status"] == "PASS" and report["identity"] == identity, "Smoke result does not match this pilot.")


def empty_accumulator():
    return {"patients": 0, "patches": 0, "loss_sum": 0., "wall_seconds": 0., "cpu_seconds": 0.,
            "load_and_transfer_seconds": 0., "forward_backward_cuda_ms": 0.,
            "attempted_optimizer_steps": 0, "optimizer_steps": 0, "amp_skips": 0,
            "peak_allocated_vram_bytes": 0, "peak_reserved_vram_bytes": 0, "sampled_peak_rss_bytes": 0}


def validate_model(bundle, model, epoch, device):
    model.eval()
    ids = pilot_ids(bundle, "validation")
    predictions, loss_sum = [], 0.
    synchronize(device)
    begin = time.perf_counter()
    with MemoryMonitor(device) as memory, torch.inference_mode():
        for pid in ids:
            x, target = load_bag(bundle, pid, epoch, 64, False, device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(x)
                loss = torch.nn.functional.cross_entropy(logits, target)
            score = float(torch.softmax(logits.float(), dim=1)[0, 1].item())
            require(math.isfinite(float(loss.item())) and math.isfinite(score), "Non-finite validation output.")
            loss_sum += float(loss.item())
            predictions.append({"epoch": epoch + 1, "case_submitter_id": pid, "label": int(target.item()), "p_LUSC": score})
            del x, target, logits, loss
        synchronize(device)
        resources = memory.values()
    metrics = binary_metrics([r["label"] for r in predictions], [r["p_LUSC"] for r in predictions])
    return {"loss": loss_sum / len(ids), "wall_seconds": time.perf_counter() - begin, **metrics, **resources}, predictions


def emit_pilot_tables(run_dir, state, budget):
    export_csv(run_dir / "epoch_metrics.csv", state["metrics"])
    export_csv(run_dir / "validation_predictions.csv", state["predictions"])
    summary = {"budget": budget, "seed": SEED, "completed_epochs": state["epoch"],
               "status": "COMPLETE" if state["epoch"] == EPOCHS else "PARTIAL",
               "best_validation": state["best"], **ledger_summary(run_dir / "sessions"),
               "main_study_started": False, "test_scored": False}
    if state["metrics"]:
        rows = state["metrics"]
        per_epoch = [(r["train_wall_seconds"] / 60) * 557 + (r["validation_wall_seconds"] / 20) * 186 for r in rows]
        summary["rough_main_30_epoch_core_seconds"] = 30 * sum(per_epoch) / len(per_epoch)
        summary["extrapolation_note"] = "Rough train+validation scaling only; excludes startup/checkpoint overhead and population-dependent I/O. Not a runtime guarantee."
    atomic_json(run_dir / "resource_summary.json", summary)


def pilot_worker(bundle, identity, weights, device, budget):
    require_smoke(bundle, identity, budget)
    run_dir = bundle["output"] / "runs" / f"B{budget}_seed{SEED}"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_identity = {"pilot_protocol_sha256": identity, "budget": budget, "seed": SEED}
    checkpoints = Checkpoints(run_dir / "checkpoints", checkpoint_identity)
    saved, location = checkpoints.load()
    if saved is not None and saved["progress"]["epoch"] == EPOCHS:
        emit_pilot_tables(run_dir, saved["progress"], budget)
        print(f"SKIP COMPLETE B{budget}: {run_dir}", flush=True)
        return
    ledger = SessionLedger(run_dir / "sessions")
    outcome = "FAILED"
    try:
        atomic_json(run_dir / "status.json", {"status": "RUNNING", "identity": checkpoint_identity, "utc": now()})
        model = build_model(weights, device=device)
        optimizer, scaler = optimizer_for(model), scaler_for(device)
        frozen_bn = bn_state(model)
        if saved is None:
            state = {"epoch": 0, "position": 0, "global_attempts": 0, "accumulator": empty_accumulator(),
                     "metrics": [], "predictions": [], "best": None, "best_sequence": None,
                     "consecutive_amp_skips": 0}
            checkpoints.save(model, optimizer, scaler, state)
        else:
            state = restore(saved, model, optimizer, scaler)
            print(f"RESUME B{budget}: epoch {state['epoch']+1}, next patient {state['position']+1}; {location.name}", flush=True)
            del saved
        ledger.record("MODEL_READY", epoch=state["epoch"], position=state["position"])
        while state["epoch"] < EPOCHS:
            epoch = state["epoch"]
            ids = patient_order(pilot_ids(bundle, "train"), epoch)
            loader = lambda pid: load_bag(bundle, pid, epoch, budget, True, device)
            with MemoryMonitor(device) as memory:
                for start, end in group_slices(len(ids), state["position"]):
                    group = train_group(model, optimizer, scaler, ids[start:end], loader, device)
                    acc = state["accumulator"]
                    for key in acc:
                        if key in group:
                            acc[key] += group[key]
                    for key, value in memory.values().items():
                        acc[key] = max(acc[key], value)
                    state["consecutive_amp_skips"] = state["consecutive_amp_skips"] + 1 if group["amp_skips"] else 0
                    require(state["consecutive_amp_skips"] < 3, "Three consecutive AMP skips; review numerical stability.")
                    state["position"] = end
                    state["global_attempts"] += 1
                    ledger.record("TRAIN_GROUP", epoch=epoch+1, next_position=end, patches=group["patches"], amp_skips=group["amp_skips"])
                    if state["global_attempts"] % CHECKPOINT_STEPS == 0 or end == len(ids):
                        checkpoints.save(model, optimizer, scaler, state)
                        ledger.record("CHECKPOINT", epoch=epoch+1, next_position=end)
                    if end % 20 == 0 or end == len(ids):
                        print(f"  B{budget} epoch {epoch+1}/{EPOCHS}: train {end}/60; loss={acc['loss_sum']/acc['patients']:.4f}", flush=True)
            require(tree_equal(frozen_bn, bn_state(model)), "BatchNorm running stats changed during training.")
            val, predictions = validate_model(bundle, model, epoch, device)
            ledger.record("VALIDATION", epoch=epoch+1)
            acc = state["accumulator"]
            require(acc["patients"] == 60 and acc["patches"] == 60 * budget, "Epoch patient/patch accounting mismatch.")
            row = {"budget": budget, "seed": SEED, "epoch": epoch+1,
                   "train_loss": acc["loss_sum"]/60, "validation_loss": val["loss"],
                   "validation_auroc": val["auroc"], "validation_balanced_accuracy": val["balanced_accuracy"],
                   "validation_macro_f1": val["macro_f1"], "train_patients": 60, "validation_patients": 20,
                   "train_patches": acc["patches"], "validation_patches": 1280,
                   "train_wall_seconds": acc["wall_seconds"], "train_cpu_seconds": acc["cpu_seconds"],
                   "load_and_transfer_seconds": acc["load_and_transfer_seconds"],
                   "forward_backward_cuda_ms": acc["forward_backward_cuda_ms"],
                   "validation_wall_seconds": val["wall_seconds"], "optimizer_steps": acc["optimizer_steps"],
                   "amp_skips": acc["amp_skips"], "train_peak_allocated_vram_bytes": acc["peak_allocated_vram_bytes"],
                   "train_peak_reserved_vram_bytes": acc["peak_reserved_vram_bytes"],
                   "train_sampled_peak_rss_bytes": acc["sampled_peak_rss_bytes"],
                   "validation_peak_allocated_vram_bytes": val["peak_allocated_vram_bytes"],
                   "validation_peak_reserved_vram_bytes": val["peak_reserved_vram_bytes"],
                   "validation_sampled_peak_rss_bytes": val["sampled_peak_rss_bytes"]}
            state["metrics"].append(row)
            state["predictions"].extend(predictions)
            best = state["best"]
            improved = best is None or val["auroc"] > best["auroc"] or (val["auroc"] == best["auroc"] and val["loss"] < best["loss"])
            if improved:
                state["best"] = {"epoch": epoch+1, "auroc": val["auroc"], "loss": val["loss"]}
            state.update(epoch=epoch+1, position=0, accumulator=empty_accumulator())
            checkpoints.save(model, optimizer, scaler, state, mark_best=improved)
            ledger.record("EPOCH_COMMITTED", epoch=epoch+1)
            emit_pilot_tables(run_dir, state, budget)
            print(f"  B{budget} epoch {epoch+1}: val AUROC={val['auroc']:.3f}, loss={val['loss']:.4f}; checkpoint committed", flush=True)
        outcome = "COMPLETE"
    except KeyboardInterrupt:
        outcome = "INTERRUPTED"
        raise
    finally:
        ledger.close(outcome)
        try:
            atomic_json(run_dir / "status.json", {"status": "PARTIAL" if outcome == "INTERRUPTED" else outcome,
                        "identity": checkpoint_identity, "utc": now(), "resume_from": "latest verified checkpoint"})
        except OSError as e:
            print(f"WARNING: status file could not be updated ({e}); verified checkpoints remain authoritative.", flush=True)
    emit_pilot_tables(run_dir, state, budget)
    atomic_json(run_dir / "status.json", {"status": "COMPLETE", "identity": checkpoint_identity, "utc": now(), "epochs": EPOCHS})
    print(f"PILOT B{budget} COMPLETE: {run_dir}", flush=True)


def launch_child(args, worker, budget):
    command = [sys.executable, "-B", str(Path(__file__).resolve()), "--root", str(args.root.resolve()),
               "--worker", worker, "--budget", str(budget)]
    result = subprocess.run(command, check=False, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
    require(result.returncode == 0, f"{worker} B{budget} stopped with exit code {result.returncode}. Completed checkpoints are preserved.")


def orchestrate(args, bundle, identity):
    if args.smoke:
        # All modes must pass the same engineering checks. A fresh process
        # replays the last K=64 update from the first process's saved boundary.
        for budget in BUDGETS:
            launch_child(args, "smoke", budget)
        launch_child(args, "resume-check", 64)
        atomic_json(bundle["output"] / "smoke_summary.json", {"status": "PASS", "identity": identity,
                    "budgets": list(BUDGETS), "fresh_process_resume_passed": True, "utc": now()})
        print("ALL SMOKE CHECKS PASS. Next: --pilot --budget 64.")
    elif args.pilot:
        budgets = [args.budget] if args.budget else BUDGETS
        for budget in budgets:
            require_smoke(bundle, identity, budget)
            launch_child(args, "pilot", budget)
        print("Requested DEVELOPMENT pilot runs complete. Review metrics/resources before freezing the main study.")
    return 0


def run(args):
    root = args.root.expanduser().resolve()
    print(f"STAGE 29 v{VERSION}: DEVELOPMENT PILOT; test evaluation disabled", flush=True)
    started = time.perf_counter()
    bundle = load_bundle(root)
    print("Stage 28 hashes verified; 929 patients / 59456 patches; fixed pilot 60 train + 20 validation.", flush=True)
    # Child workers verify each selected PNG as loaded. The orchestration parent
    # never imports torch/creates a competing CUDA context during GPU runs.
    if not args.worker:
        check_pilot_files(bundle)
    if args.smoke or args.pilot:
        _, identity, _ = read_prepared(bundle, None)
        with execution_lock(bundle["output"] / "execution.lock"):
            return orchestrate(args, bundle, identity)
    device = initialize_torch(cuda=True)
    environment = environment_record(bundle)
    print(f"Environment: torch={environment['versions']['torch']}; GPU={environment['gpu']}; threads={CPU_THREADS}; workers=0", flush=True)
    print(f"Preflight elapsed={time.perf_counter()-started:.1f}s; no test PNGs opened.", flush=True)
    if args.prepare:
        prepare(bundle, environment)
        return 0
    if not args.worker:
        print("PREFLIGHT PASS. No output writes/training/downloads. Next: --prepare, then --smoke.")
        return 0
    _, identity, weights = read_prepared(bundle, environment)
    if args.worker in ("smoke", "resume-check"):
        smoke_worker(bundle, identity, weights, device, args.budget, args.worker == "resume-check")
    else:
        pilot_worker(bundle, identity, weights, device, args.budget)
    return 0


def self_test(with_torch=False):
    import unittest
    from unittest.mock import patch

    class Tests(unittest.TestCase):
        def setUp(self):
            self.temp = tempfile.TemporaryDirectory(prefix="stage29-test-")
            self.root = Path(self.temp.name)
        def tearDown(self):
            self.temp.cleanup()
        def test_nested_budgets_and_four_epoch_coverage(self):
            for pid in ("TCGA-AA-0001", "TCGA-BB-0002"):
                for cycle in range(3):
                    counts = {k: collections.Counter() for k in BUDGETS}
                    for epoch in range(cycle*4, cycle*4+4):
                        choices = {k: patch_indices(pid, epoch, k) for k in BUDGETS}
                        self.assertEqual(choices[16], choices[32][:16])
                        self.assertEqual(choices[32], choices[64][:32])
                        for k in BUDGETS:
                            self.assertEqual(len(set(choices[k])), k)
                            counts[k].update(choices[k])
                    for k in BUDGETS:
                        self.assertEqual(set(counts[k]), set(range(64)))
                        self.assertEqual(set(counts[k].values()), {k//16})
        def test_rng_stream_independence(self):
            ids = [f"TCGA-AA-{i:04d}" for i in range(10)]
            before = patient_order(ids, 3)
            augment = [augmentation_code(ids[0], 3, i) for i in range(1, 65)]
            for k in BUDGETS:
                patch_indices(ids[0], 3, k)
            self.assertEqual(before, patient_order(list(reversed(ids)), 3))
            self.assertEqual(augment, [augmentation_code(ids[0], 3, i) for i in range(1, 65)])
        def test_partial_group_and_resume_boundary(self):
            self.assertEqual(group_slices(5), [(0, 4), (4, 5)])
            self.assertEqual(group_slices(5, 4), [(4, 5)])
            self.assertEqual(group_slices(5, 5), [])
            with self.assertRaises(ExperimentError):
                group_slices(5, 3)
        def test_metrics_ties_perfect_and_reversed(self):
            self.assertEqual(binary_metrics([0, 1], [.1, .9])["auroc"], 1.)
            self.assertEqual(binary_metrics([0, 1], [.9, .1])["auroc"], 0.)
            self.assertEqual(binary_metrics([0, 1, 0, 1], [.5]*4)["auroc"], .5)
            for labels, scores in (([0, 0], [.1, .2]), ([0, 1], [float("nan"), .2])):
                with self.assertRaises(ExperimentError):
                    binary_metrics(labels, scores)
        def test_test_patient_loader_guard(self):
            bundle = {"pilot": {"p": {"split": "test"}}}
            with self.assertRaises(ExperimentError):
                load_bag(bundle, "p", 0, 64, False, None)
            with self.assertRaises(ExperimentError):
                pilot_ids(bundle, "test")
        def test_png_hash_change_is_fatal(self):
            p = self.root / "patch.png"
            p.write_bytes(b"original")
            row = {"path": p, "bytes": p.stat().st_size, "file_sha256": sha_file(p)}
            self.assertEqual(verified_png_bytes(row), b"original")
            p.write_bytes(b"modified")
            with self.assertRaises(ExperimentError):
                verified_png_bytes(row)
        def test_cost_ledger_counts_replay_and_unknown_tail(self):
            folder = self.root / "sessions"
            folder.mkdir()
            (folder / "one.jsonl").write_text(json.dumps({"event": "END", "active_seconds": 10., "cpu_seconds": 3.})+"\n")
            (folder / "two.jsonl").write_text(json.dumps({"event": "TRAIN_GROUP", "active_seconds": 4., "cpu_seconds": 2.})+'\n{"truncated":')
            result = ledger_summary(folder)
            self.assertEqual(result["recorded_active_seconds_including_replay"], 14.)
            self.assertTrue(result["time_is_lower_bound"])
            self.assertEqual(result["unclean_sessions"], ["two.jsonl"])
        def test_atomic_and_csv_locked_does_not_touch_checkpoint(self):
            p = self.root / "record.json"
            atomic_json(p, {"a": 1})
            atomic_json(p, {"a": 2})
            self.assertEqual(read_json(p), {"a": 2})
            with patch.dict(globals(), {"atomic_bytes": lambda *a: (_ for _ in ()).throw(PermissionError("locked"))}), contextlib.redirect_stdout(io.StringIO()):
                export_csv(self.root / "metrics.csv", [{"x": 1}])
            self.assertEqual(read_json(p), {"a": 2})
        def test_process_lock_released_and_reusable(self):
            path = self.root / "execution.lock"
            with execution_lock(path):
                with self.assertRaises(ExperimentError):
                    with execution_lock(path):
                        pass
            with execution_lock(path):
                self.assertTrue(path.is_file())

    class TorchTests(Tests):
        def components(self, seed=SEED):
            model = build_model(None, seed, device=torch.device("cpu"), tiny=True)
            return model, optimizer_for(model), scaler_for(torch.device("cpu"))
        def loader(self, pid):
            g = torch.Generator().manual_seed(100 + int(pid))
            return torch.rand((2, 3, 12, 12), generator=g), torch.tensor([int(pid) % 2])
        def test_encoder_updates_and_bn_does_not(self):
            model, optimizer, scaler = self.components()
            initial = next(model.encoder.parameters()).detach().clone()
            bn = bn_state(model)
            result = train_group(model, optimizer, scaler, [0, 1, 2, 3], self.loader, torch.device("cpu"), strict=True)
            self.assertFalse(torch.equal(initial, next(model.encoder.parameters())))
            self.assertTrue(tree_equal(bn, bn_state(model)))
            self.assertEqual(result["optimizer_steps"], 1)
        def test_gradient_normalization_for_four_and_one(self):
            for ids in ([0, 1, 2, 3], [4]):
                model_a, opt_a, scale_a = self.components()
                model_b, opt_b, _ = self.components()
                train_group(model_a, opt_a, scale_a, ids, self.loader, torch.device("cpu"), strict=True)
                model_b.train()
                losses = [torch.nn.functional.cross_entropy(model_b(self.loader(i)[0]), self.loader(i)[1]) for i in ids]
                torch.stack(losses).mean().backward()
                opt_b.step()
                for a, b in zip(model_a.parameters(), model_b.parameters()):
                    torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-7)
        def test_checkpoint_resume_exact_and_fallback(self):
            device = torch.device("cpu")
            model, optimizer, scaler = self.components()
            checkpoints = Checkpoints(self.root / "checkpoints", {"test": 1})
            train_group(model, optimizer, scaler, [0, 1, 2, 3], self.loader, device, strict=True)
            state = {"epoch": 0, "position": 4, "best_sequence": None}
            first = checkpoints.save(model, optimizer, scaler, state)
            train_group(model, optimizer, scaler, [4], self.loader, device, strict=True)
            expected = snapshot(model, optimizer, scaler, {})
            saved, _ = checkpoints.load()
            other, opt2, scaler2 = self.components(seed=4)
            self.assertEqual(restore(saved, other, opt2, scaler2)["position"], 4)
            train_group(other, opt2, scaler2, [4], self.loader, device, strict=True)
            actual = snapshot(other, opt2, scaler2, {})
            for key in ("model", "optimizer", "scaler", "rng"):
                self.assertTrue(tree_equal(expected[key], actual[key]), key)
            second = checkpoints.save(other, opt2, scaler2, {"epoch": 1, "position": 0, "best_sequence": None})
            (second / "state.pt").write_bytes(b"simulated corruption")
            with contextlib.redirect_stdout(io.StringIO()):
                saved, chosen = checkpoints.load()
            self.assertEqual(chosen, first)
            self.assertEqual(saved["progress"]["position"], 4)
            third = checkpoints.save(other, opt2, scaler2, {"epoch": 1, "position": 0, "best_sequence": None})
            self.assertTrue(first.is_dir())
            self.assertFalse(second.exists())
            self.assertTrue(third.is_dir())
        def test_png_loader_uses_real_shape_and_same_nested_augmentations(self):
            from PIL import Image
            path = self.root / "patch.png"
            array = np.zeros((256, 256, 3), dtype=np.uint8)
            array[:90, :, 0] = 255
            array[:, :50, 1] = 150
            Image.fromarray(array).save(path)
            row = {"path": path, "bytes": path.stat().st_size, "file_sha256": sha_file(path)}
            pid = "TCGA-AA-0001"
            bundle = {"pilot": {pid: {"split": "train"}}, "patients": {pid: {"class_label": "LUAD"}},
                      "patches": {pid: [{**row, "index": i+1} for i in range(64)]}}
            x16, y = load_bag(bundle, pid, 2, 16, True, torch.device("cpu"))
            x32, _ = load_bag(bundle, pid, 2, 32, True, torch.device("cpu"))
            self.assertEqual(tuple(x16.shape), (16, 3, 256, 256))
            self.assertTrue(torch.equal(x16, x32[:16]))
            self.assertEqual(y.item(), 0)
        def test_pilot_interruption_resume_no_duplicate_epochs(self):
            patients, pilot = {}, {}
            for i in range(80):
                pid = f"TCGA-AA-{i:04d}"
                patients[pid] = {"class_label": "LUAD" if i % 2 == 0 else "LUSC"}
                pilot[pid] = {"split": "train" if i < 60 else "validation"}
            original_build, original_group = build_model, train_group
            def synthetic_bag(bundle, pid, epoch, budget, training, device):
                i = int(pid[-4:])
                g = torch.Generator().manual_seed(i + epoch * 100)
                x = torch.rand((budget, 3, 8, 8), generator=g) * .1 + (i % 2) * .8
                return x, torch.tensor([i % 2])
            def tiny_build(weights, seed=SEED, device=None, tiny=False):
                return original_build(None, seed, device, tiny=True)
            bundles = [{"output": self.root / name, "patients": patients, "pilot": pilot}
                       for name in ("continuous", "interrupted")]
            overrides = {"build_model": tiny_build, "load_bag": synthetic_bag,
                         "require_smoke": lambda *a: None, "EPOCHS": 2}
            with patch.dict(globals(), overrides), contextlib.redirect_stdout(io.StringIO()):
                pilot_worker(bundles[0], "synthetic", None, torch.device("cpu"), 16)
                calls = [0]
                def interrupt_on_seventh(*args, **kwargs):
                    calls[0] += 1
                    if calls[0] == 7:
                        raise KeyboardInterrupt()
                    return original_group(*args, **kwargs)
                with patch.dict(globals(), {"train_group": interrupt_on_seventh}):
                    with self.assertRaises(KeyboardInterrupt):
                        pilot_worker(bundles[1], "synthetic", None, torch.device("cpu"), 16)
                pilot_worker(bundles[1], "synthetic", None, torch.device("cpu"), 16)
                snapshots = []
                for b in bundles:
                    run_dir = b["output"] / "runs" / f"B16_seed{SEED}"
                    cp = Checkpoints(run_dir / "checkpoints", {"pilot_protocol_sha256": "synthetic", "budget": 16, "seed": SEED})
                    data, _ = cp.load()
                    snapshots.append(data)
                    self.assertEqual(len(read_csv(run_dir / "epoch_metrics.csv")), 2)
                    self.assertEqual(len(read_csv(run_dir / "validation_predictions.csv")), 40)
                    self.assertEqual(data["progress"]["epoch"], 2)
                for key in ("model", "optimizer", "scaler", "rng"):
                    self.assertTrue(tree_equal(snapshots[0][key], snapshots[1][key]), key)
        def test_mil_permutation_invariance_and_real_resnet_backward(self):
            model = build_model(None, device=torch.device("cpu"))
            model.train()
            x = torch.rand(2, 3, 64, 64)
            output = model(x)
            torch.testing.assert_close(output, model(x.flip(0)), rtol=1e-5, atol=1e-6)
            torch.nn.functional.cross_entropy(output, torch.tensor([1])).backward()
            self.assertTrue(bool(torch.isfinite(model.encoder.conv1.weight.grad).all()))
            self.assertGreater(float(model.encoder.conv1.weight.grad.abs().sum()), 0.)

    if with_torch:
        initialize_torch(cuda=False)
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TorchTests if with_torch else Tests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--preflight-only", action="store_true", help="Read-only input/pilot PNG/environment audit (default).")
    modes.add_argument("--prepare", action="store_true", help="Prepare pilot protocol and official weights; no training.")
    modes.add_argument("--smoke", action="store_true", help="GPU update checks for all K and fresh-process resume proof.")
    modes.add_argument("--pilot", action="store_true", help="Run/resume five DEVELOPMENT epochs for selected budget(s).")
    modes.add_argument("--self-test", action="store_true", help="Temporary logic tests; no torch/GPU/downloads.")
    modes.add_argument("--self-test-torch", action="store_true", help="CPU model/gradient/checkpoint tests; no data or downloads.")
    modes.add_argument("--worker", choices=("smoke", "resume-check", "pilot"), help=argparse.SUPPRESS)
    parser.add_argument("--budget", type=int, choices=BUDGETS, help="Pilot budget; omit to run all (64,32,16).")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
    args = parser.parse_args()
    require(not args.worker or args.budget is not None, "Internal worker requires --budget.")
    require(args.budget is None or args.pilot or args.worker, "--budget applies only to --pilot.")
    try:
        if args.self_test or args.self_test_torch:
            return self_test(args.self_test_torch)
        return run(args)
    except KeyboardInterrupt:
        print("\nINTERRUPTED. Repeat the same command; the last committed optimizer boundary will be restored.", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        if "out of memory" in str(e).lower():
            print("GPU OOM: stop here. No silent change of K, precision or batch size was made.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
