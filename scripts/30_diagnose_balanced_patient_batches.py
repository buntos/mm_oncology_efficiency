#!/usr/bin/env python3
r"""Stage 30 v1.0.0: DIAGNOSTIC test of label-balanced gradient-accumulation groups (not the main study).

Background: results\stage29_learning_investigation_v1\ (D01-D08). No implementation bug has been found; this
is a diagnostic intervention, not a bug fix. Reference configuration = D08-B:
  encoder LR 1e-5, head LR 1e-4, K=64, seed 20261113, same 60 pilot-train / 20 pilot-validation patients,
  Stage 29 architecture / weights / original head init / AMP / frozen BN / AdamW (wd 1e-4) / D4 augmentation.
The ONLY experimental factor is how accumulation groups (4 patients, one optimizer step) are formed:
  original : consecutive slices of the Stage 29 per-epoch patient order (identical to Stage 29 / D08-B);
  balanced : 2 LUAD + 2 LUSC per group. Each class keeps its Stage 29 per-epoch order; group g takes LUAD
             2g, 2g+1 and LUSC 2g, 2g+1; within a group patients are ordered by their position in the
             Stage 29 per-epoch order. Every patient exactly once per epoch; 15 groups.
Patch selection, augmentation, model seed and everything else are untouched (same Stage 29 functions).

The frozen Stage 29 module is imported read-only after its SHA256 is verified; its integrity checks
(Stage 28 hashes, prepared weights, environment) are used, never bypassed.

Run from C:\research\mm_oncology_efficiency with the project .venv:
  python scripts\30_diagnose_balanced_patient_batches.py --self-test                 (CPU, logic only)
  python scripts\30_diagnose_balanced_patient_batches.py --preflight-only            (read-only checks)
  python scripts\30_diagnose_balanced_patient_batches.py --prepare                   (protocol, no training)
  python scripts\30_diagnose_balanced_patient_batches.py --run --mode balanced       (run or RESUME)
  python scripts\30_diagnose_balanced_patient_batches.py --run --mode original --stop-after-epochs 1
  python scripts\30_diagnose_balanced_patient_batches.py --check-original-vs-d08b    (CPU comparison)
  python scripts\30_diagnose_balanced_patient_batches.py --resume-selftest           (separate test folder)
  python scripts\30_diagnose_balanced_patient_batches.py --analyze                   (CPU comparison report)
Resume after interruption / power loss: repeat the same --run command. The last verified checkpoint
(optimizer-step boundary; model, optimizer, scaler, RNG, epoch, next group, accumulated metrics) is restored;
metric rows are rebuilt from the checkpoint, so nothing is duplicated.
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True
import argparse
import csv
import hashlib
import importlib.util
import io
import json
import math
import os
import random
import subprocess
import time
from pathlib import Path

VERSION = "1.0.0"
ROOT = Path(__file__).resolve().parent.parent
S29_PATH = ROOT / "scripts" / "29_train_patch_budget_baselines.py"
S29_SHA256 = "a8a7409a32c748c3529544138b5e20b8f6294b233a8fb22c20585dcd522ab09d"
MAIN_OUTPUT = ROOT / "results" / "stage30_balanced_batches_v1"
SELFTEST_OUTPUT = ROOT / "results" / "stage30_balanced_batches_v1_resume_selftest"
D08B = ROOT / "results" / "stage29_learning_investigation_v1" / "d08_head_lr_B_B64_20ep"
MODES = ("original", "balanced")
MAIN_EPOCHS = 10
SETTINGS = {
    "encoder_lr": 1e-5, "head_lr": 1e-4, "budget": 64, "seed": 20261113, "accumulation_patients": 4,
    "weight_decay": 1e-4, "optimizer": "Stage 29 optimizer_for (AdamW foreach=False), LRs set on param_groups",
    "classifier_init": "original Stage 29 build_model(seed) initialisation", "amp": "Stage 29 (FP16 encoder, FP32 head)",
    "batchnorm": "Stage 29 (frozen running stats, trainable affine)", "augmentation": "Stage 29 SHA-keyed D4",
    "patch_sampling": "Stage 29 patch_indices", "validation_patch_budget": 64, "threshold": 0.5,
    "scheduler": None, "grad_clipping": None, "checkpoint_every_optimizer_steps": 5,
    "group_modes": {"original": "consecutive slices of Stage 29 patient_order(epoch)",
                    "balanced": "2 LUAD + 2 LUSC; per-class Stage 29 order; group g = LUAD[2g:2g+2] + LUSC[2g:2g+2]; "
                                "within-group order = position in Stage 29 patient_order(epoch)"},
}
PRIMARY_METRIC = {
    "name": "mean absolute epoch-to-epoch change of the mean logit difference (LUSC - LUAD)",
    "population": "the same 60 pilot-train patients, all 64 patches, no augmentation, eval mode, after each epoch",
    "transitions": "epochs 1->2 ... 9->10 (9 transitions)",
    "comparison": "balanced (this stage) vs D08-B epochs 1-10 (original grouping, same configuration)",
    "success_requires": "lower instability WITHOUT loss of class separation (train-eval and validation "
                        "mean p difference / AUROC must not collapse); validation is descriptive only",
}

s29 = torch = np = None


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(1 << 20), b""):
            h.update(part)
    return h.hexdigest()


def load_stage29():
    global s29
    if s29 is None:
        if sha_file(S29_PATH) != S29_SHA256:
            raise SystemExit("ERROR: Stage 29 script hash differs from the investigated version; refusing to run.")
        spec = importlib.util.spec_from_file_location("stage29_frozen", S29_PATH)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        s29 = module
    return s29


def require(cond, msg):
    load_stage29().require(cond, msg)


# ----------------------------------------------------------------------------- grouping (the single factor)
def make_groups(mode, train_ids, labels, epoch):
    order = s29.patient_order(train_ids, epoch)
    if mode == "original":
        groups = [order[a:b] for a, b in s29.group_slices(len(order))]
    elif mode == "balanced":
        pos = {p: i for i, p in enumerate(order)}
        luad = [p for p in order if labels[p] == "LUAD"]
        lusc = [p for p in order if labels[p] == "LUSC"]
        require(len(luad) == len(lusc) and len(luad) % 2 == 0, "Balanced groups need equal, even class counts.")
        groups = [sorted(luad[2 * g:2 * g + 2] + lusc[2 * g:2 * g + 2], key=pos.__getitem__) for g in range(len(luad) // 2)]
    else:
        raise ValueError(mode)
    flat = [p for g in groups for p in g]
    require(sorted(flat) == sorted(train_ids) and len(flat) == len(set(flat)), "Each patient must appear exactly once per epoch.")
    require(all(len(g) == 4 for g in groups) and len(groups) == 15, "Expected 15 groups of 4 patients.")
    if mode == "balanced":
        require(all(sum(labels[p] == "LUSC" for p in g) == 2 for g in groups), "Balanced group is not 2+2.")
    return groups


# ----------------------------------------------------------------------------- protocol
def protocol_record(bundle, weights, environment, epochs):
    return {"schema_version": 1, "stage30_version": VERSION, "purpose": "DIAGNOSTIC_BALANCED_GROUPS_ONLY",
            "not_a_bug_fix": "No implementation error found in D01-D08; this tests one optimisation factor.",
            "script30_sha256": sha_file(Path(__file__)), "stage29_script_sha256": S29_SHA256,
            "stage28_manifest_sha256": bundle["manifest_sha256"], "stage28_files": s29.EXPECTED_FILES,
            "dataset_fingerprint": s29.EXPECTED_DATASET, "split_fingerprint": s29.EXPECTED_SPLIT,
            "weights_sha256": sha_file(weights), "weights_source": "Stage 29 prepared weights (verified)",
            "train_patients": s29.pilot_ids(bundle, "train"), "validation_patients": s29.pilot_ids(bundle, "validation"),
            "test_used": False, "epochs": epochs, "settings": SETTINGS, "settings_sha256": s29.digest(SETTINGS),
            "primary_metric": PRIMARY_METRIC, "reference_run": str(D08B.relative_to(ROOT)),
            "environment": environment}


def read_protocol(output, bundle, environment):
    p, prep = output / "protocol.json", output / "preparation.json"
    require(p.is_file() and prep.is_file(), f"Not prepared: run --prepare first ({output}).")
    protocol, record = s29.read_json(p), s29.read_json(prep)
    require(record["protocol_sha256"] == sha_file(p), "protocol.json changed after preparation.")
    require(protocol["script30_sha256"] == sha_file(Path(__file__)),
            "Stage 30 code changed after preparation; refusing to continue (version a new output folder).")
    require(protocol["stage29_script_sha256"] == S29_SHA256 and protocol["stage28_manifest_sha256"] == bundle["manifest_sha256"],
            "Protocol refers to another Stage 28/29 snapshot.")
    require(protocol["settings"] == json.loads(json.dumps(SETTINGS)) and protocol["settings_sha256"] == s29.digest(SETTINGS),
            "Settings differ from the prepared protocol; refusing to continue.")
    if environment is not None:
        require(protocol["environment"] == environment, "Execution environment differs from preparation.")
    return protocol, record["protocol_sha256"]


def prepare(output, epochs):
    bundle = s29.load_bundle(ROOT)
    _, _, weights = s29.read_prepared(bundle, None)
    s29.initialize_torch(cuda=True)
    environment = s29.environment_record(bundle)
    if output.exists():
        read_protocol(output, bundle, environment)
        print(f"ALREADY PREPARED (verified): {output}")
        return
    output.mkdir(parents=True)
    rec = protocol_record(bundle, weights, environment, epochs)
    s29.atomic_json(output / "protocol.json", rec)
    s29.atomic_json(output / "preparation.json", {"status": "PREPARED", "utc": s29.now(),
                                                  "protocol_sha256": sha_file(output / "protocol.json")})
    print(f"PREPARED: {output} (epochs={epochs}). No training has run.")


def preflight():
    bundle = s29.load_bundle(ROOT)
    s29.read_prepared(bundle, None)
    s29.check_pilot_files(bundle)
    s29.initialize_torch(cuda=True)
    env = s29.environment_record(bundle)
    labels = {p: bundle["patients"][p]["class_label"] for p in s29.pilot_ids(bundle, "train")}
    for mode in MODES:
        for e in range(MAIN_EPOCHS):
            make_groups(mode, s29.pilot_ids(bundle, "train"), labels, e)
    require(D08B.is_dir(), "D08-B reference results missing.")
    print(f"PREFLIGHT PASS: Stage 29 hash, Stage 28 files, 5120 pilot PNG hashes, weights, environment "
          f"(torch {env['versions']['torch']}, {env['gpu']}), groups valid for both modes x {MAIN_EPOCHS} epochs. No writes.")


# ----------------------------------------------------------------------------- run
def write_table(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    text = io.StringIO(newline="")
    w = csv.DictWriter(text, fieldnames=keys)
    w.writeheader()
    w.writerows([{k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in r.items()} for r in rows])
    try:
        s29.atomic_bytes(path, text.getvalue().encode("utf-8"))
    except PermissionError:
        print(f"WARNING: close {path} (Excel?) and rerun to rebuild it; checkpoints are authoritative.", flush=True)


def emit(run_dir, state):
    write_table(run_dir / "epoch_metrics.csv", state["metrics"])
    write_table(run_dir / "groups.csv", state["groups"])
    write_table(run_dir / "steps.csv", state["steps"])
    write_table(run_dir / "validation_predictions.csv", state["predictions"])


def run(output, mode, run_name, stop_after_steps=None, stop_after_epochs=None):
    global torch, np
    bundle = s29.load_bundle(ROOT)
    _, _, weights = s29.read_prepared(bundle, None)
    device = s29.initialize_torch(cuda=True)
    torch, np = s29.torch, s29.np
    environment = s29.environment_record(bundle)
    protocol, protocol_sha = read_protocol(output, bundle, environment)
    epochs = protocol["epochs"]
    run_dir = output / "runs" / (run_name or f"{mode}_seed{SETTINGS['seed']}")
    run_dir.mkdir(parents=True, exist_ok=True)
    with s29.execution_lock(run_dir / "execution.lock"):
        return _run_locked(bundle, weights, device, protocol_sha, epochs, mode, run_dir, stop_after_steps, stop_after_epochs)


def _run_locked(bundle, weights, device, protocol_sha, epochs, mode, run_dir, stop_after_steps, stop_after_epochs):
    from PIL import Image
    from sklearn.metrics import roc_auc_score
    identity = {"protocol_sha256": protocol_sha, "mode": mode, "seed": SETTINGS["seed"], "epochs": epochs}
    marker = run_dir / "run_identity.json"
    if marker.exists():
        require(s29.read_json(marker) == identity, "Run folder belongs to a different mode/protocol; refusing to continue.")
    else:
        s29.atomic_json(marker, identity)
    checkpoints = s29.Checkpoints(run_dir / "checkpoints", identity)
    saved, location = checkpoints.load()
    if saved is not None and saved["progress"]["epoch"] == epochs:
        emit(run_dir, saved["progress"])
        print(f"SKIP COMPLETE: {run_dir}", flush=True)
        return 0

    model = s29.build_model(weights, device=device)
    init_enc = [p.detach().clone() for p in model.encoder.parameters()]
    init_head = [p.detach().clone() for n, p in model.named_parameters() if not n.startswith("encoder.")]
    optimizer, scaler = s29.optimizer_for(model), s29.scaler_for(device)
    optimizer.param_groups[0]["lr"] = SETTINGS["encoder_lr"]
    optimizer.param_groups[1]["lr"] = SETTINGS["head_lr"]
    frozen_bn = s29.bn_state(model)
    enc = list(model.encoder.parameters())
    head = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    nrm = lambda ts: math.sqrt(sum(float((t.float() ** 2).sum()) for t in ts))
    init_enc_norm, init_head_norm = nrm(init_enc), nrm(init_head)
    last_step, prev = {}, {}
    optimizer.register_step_pre_hook(lambda o, a, k: prev.update(e=[p.detach().clone() for p in enc], h=[p.detach().clone() for p in head]))
    def post(o, a, k):
        with torch.no_grad():
            last_step.update(enc_update=nrm([p - q for p, q in zip(enc, prev["e"])]),
                             head_update=nrm([p - q for p, q in zip(head, prev["h"])]), lr=[g["lr"] for g in o.param_groups])
    optimizer.register_step_post_hook(post)

    train_ids = s29.pilot_ids(bundle, "train")
    labels = {p: bundle["patients"][p]["class_label"] for p in train_ids}
    train_y = np.array([int(labels[p] == "LUSC") for p in train_ids])
    mean = torch.tensor([.485, .456, .406]).view(1, 3, 1, 1); std = torch.tensor([.229, .224, .225]).view(1, 3, 1, 1)
    print("Caching 60 train bags for eval-mode diagnostics (verified bytes, no augmentation)...", flush=True)
    train_bags = {}
    for p in train_ids:
        x = torch.stack([torch.from_numpy(np.array(Image.open(io.BytesIO(s29.verified_png_bytes(r))).convert("RGB"))).permute(2, 0, 1)
                         for r in bundle["patches"][p]]).float() / 255.
        train_bags[p] = ((x - mean) / std).half()

    def rng_fp():
        n = np.random.get_state()
        return (hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
                tuple(hashlib.sha256(s.numpy().tobytes()).hexdigest() for s in torch.cuda.get_rng_state_all()),
                hash(random.getstate()), hashlib.sha256(n[1].tobytes()).hexdigest() + str(n[2]))

    def guarded(fn):
        def wrapper(*a, **k):
            before = rng_fp()
            out = fn(*a, **k)
            model.train()
            require(rng_fp() == before, "Diagnostic changed RNG state.")
            require(s29.tree_equal(frozen_bn, s29.bn_state(model)), "Diagnostic changed BN buffers.")
            return out
        return wrapper

    def forward_eval(x):
        with torch.autocast("cuda", dtype=torch.float16):
            h = model.encoder(x)
        h = h.float()
        s = model.attention_w(torch.tanh(model.attention_v(h)) * torch.sigmoid(model.attention_u(h))).squeeze(-1)
        a = torch.softmax(s, 0); z = (a[:, None] * h).sum(0, keepdim=True)
        return model.classifier(z).float(), z[0]

    def summary(prefix, y, p, ld, Z):
        y, p, ld = np.asarray(y), np.asarray(p), np.asarray(ld); pred = (p >= SETTINGS["threshold"]).astype(int)
        return {f"{prefix}_p_mean": float(p.mean()), f"{prefix}_p_sd": float(p.std()), f"{prefix}_p_min": float(p.min()),
                f"{prefix}_p_max": float(p.max()), f"{prefix}_n_pred_LUAD": int((pred == 0).sum()), f"{prefix}_n_pred_LUSC": int(pred.sum()),
                f"{prefix}_mean_p_diff_LUSC_minus_LUAD": float(p[y == 1].mean() - p[y == 0].mean()),
                f"{prefix}_logit_diff_mean": float(ld.mean()), f"{prefix}_logit_diff_sd": float(ld.std()),
                f"{prefix}_feat_abs_mean": float(np.abs(Z).mean()), f"{prefix}_feat_between_patient_sd": float(Z.std(0).mean())}

    @guarded
    def eval_train():
        model.eval(); ps, ces, lds, Z = [], [], [], []
        with torch.inference_mode():
            for p, yy in zip(train_ids, train_y):
                lg, z = forward_eval(train_bags[p].float().to(device))
                ces.append(float(torch.nn.functional.cross_entropy(lg, torch.tensor([int(yy)], device=device))))
                ps.append(float(torch.softmax(lg, 1)[0, 1])); lds.append(float(lg[0, 1] - lg[0, 0])); Z.append(z.cpu().numpy())
        out = {"train_eval_ce": float(np.mean(ces)), "train_eval_auroc": float(roc_auc_score(train_y, ps))}
        out.update(summary("train_eval", train_y, ps, lds, np.stack(Z)))
        return out

    @guarded
    def eval_validation(epoch):
        val, pr = s29.validate_model(bundle, model, epoch, device)
        model.eval(); Z, lds = [], []
        with torch.inference_mode():
            for p in s29.pilot_ids(bundle, "validation"):
                x, _ = s29.load_bag(bundle, p, 0, 64, False, device)
                lg, z = forward_eval(x); Z.append(z.cpu().numpy()); lds.append(float(lg[0, 1] - lg[0, 0]))
        out = {"val_ce": val["loss"], "val_auroc": val["auroc"], "val_balanced_accuracy": val["balanced_accuracy"],
               "val_macro_f1": val["macro_f1"], "val_accuracy": val["accuracy"]}
        out.update(summary("val", [r["label"] for r in pr], [r["p_LUSC"] for r in pr], lds, np.stack(Z)))
        return out, pr

    def bias():
        b = model.classifier.bias.detach().double().cpu().tolist()
        return {"bias_LUAD": repr(b[0]), "bias_LUSC": repr(b[1]), "bias_diff_LUSC_minus_LUAD": repr(b[1] - b[0])}

    ledger = s29.SessionLedger(run_dir / "sessions")
    outcome = "FAILED"
    try:
        s29.atomic_json(run_dir / "status.json", {"status": "RUNNING", "identity": identity, "utc": s29.now()})
        if saved is None:
            state = {"epoch": 0, "group": 0, "global_steps": 0, "metrics": [{"epoch": 0, **eval_train(), **bias()}],
                     "groups": [], "steps": [], "predictions": [], "epoch_acc": {"loss_sum": 0., "amp_skips": 0, "train_seconds": 0.}}
            checkpoints.save(model, optimizer, scaler, state)
            print(f"START {mode}: epoch-0 diagnostics committed", flush=True)
        else:
            state = s29.restore(saved, model, optimizer, scaler)
            del saved
            require([g["lr"] for g in optimizer.param_groups] == [SETTINGS["encoder_lr"], SETTINGS["head_lr"]], "Restored LRs differ.")
            print(f"RESUME {mode}: epoch {state['epoch'] + 1}, next group {state['group'] + 1} ({location.name})", flush=True)
        ledger.record("MODEL_READY", epoch=state["epoch"], group=state["group"])
        while state["epoch"] < epochs:
            epoch = state["epoch"]
            groups = make_groups(mode, train_ids, labels, epoch)
            loader = lambda pid: s29.load_bag(bundle, pid, epoch, SETTINGS["budget"], True, device)
            while state["group"] < len(groups):
                gi = state["group"]; ids = groups[gi]
                t0 = time.perf_counter()
                g = s29.train_group(model, optimizer, scaler, ids, loader, device)
                acc = state["epoch_acc"]
                acc["loss_sum"] += g["loss_sum"]; acc["amp_skips"] += g["amp_skips"]; acc["train_seconds"] += time.perf_counter() - t0
                state["group"] += 1; state["global_steps"] += 1
                labs = [labels[p] for p in ids]
                row = {"epoch": epoch + 1, "group": gi + 1, "global_step": state["global_steps"], "patients": "|".join(ids),
                       "labels": "|".join(labs), "n_LUSC": labs.count("LUSC"), "n_LUAD": labs.count("LUAD"),
                       "group_mean_loss": g["loss_sum"] / g["patients"], "optimizer_steps": g["optimizer_steps"],
                       "amp_skips": g["amp_skips"], "enc_update": last_step["enc_update"], "head_update": last_step["head_update"], **bias()}
                state["groups"].append(row)
                state["steps"].append({"global_step": state["global_steps"], "epoch": epoch + 1, "group": gi + 1,
                                       "enc_update": last_step["enc_update"], "head_update": last_step["head_update"], "lr": last_step["lr"]})
                ledger.record("TRAIN_GROUP", epoch=epoch + 1, group=gi + 1)
                stop_now = stop_after_steps is not None and state["global_steps"] == stop_after_steps
                if state["global_steps"] % SETTINGS["checkpoint_every_optimizer_steps"] == 0 or state["group"] == len(groups) or stop_now:
                    checkpoints.save(model, optimizer, scaler, state)
                    ledger.record("CHECKPOINT", epoch=epoch + 1, group=state["group"])
                if stop_now:
                    outcome = "SIMULATED_INTERRUPTION"
                    print(f"SIMULATED INTERRUPTION after optimizer step {state['global_steps']} (checkpoint committed).", flush=True)
                    return 3
            require(s29.tree_equal(frozen_bn, s29.bn_state(model)), "BatchNorm running stats changed during training.")
            val, preds = eval_validation(epoch)
            st = [s for s in state["steps"] if s["epoch"] == epoch + 1]
            with torch.no_grad():
                enc_disp = nrm([p - q for p, q in zip(enc, init_enc)]) / init_enc_norm
                head_disp = nrm([p - q for p, q in zip(head, init_head)]) / init_head_norm
            acc = state["epoch_acc"]
            mrow = {"epoch": epoch + 1, "mode": mode, "train_aug_ce_running": acc["loss_sum"] / len(train_ids),
                    "train_seconds": acc["train_seconds"], "optimizer_steps": len(st), "amp_skips": acc["amp_skips"],
                    "lr_actual": st[-1]["lr"], "enc_update_mean": float(np.mean([s["enc_update"] for s in st])),
                    "enc_update_sum": float(np.sum([s["enc_update"] for s in st])),
                    "head_update_mean": float(np.mean([s["head_update"] for s in st])),
                    "head_update_sum": float(np.sum([s["head_update"] for s in st])),
                    "enc_rel_displacement_from_init": enc_disp, "head_rel_displacement_from_init": head_disp,
                    **val, **eval_train(), **bias()}
            state["metrics"].append(mrow)
            state["predictions"].extend(preds)
            state.update(epoch=epoch + 1, group=0, epoch_acc={"loss_sum": 0., "amp_skips": 0, "train_seconds": 0.})
            checkpoints.save(model, optimizer, scaler, state)
            ledger.record("EPOCH_COMMITTED", epoch=epoch + 1)
            emit(run_dir, state)
            print(f"{mode} ep{epoch + 1:2d}: trEvalCE={mrow['train_eval_ce']:.4f} trAUROC={mrow['train_eval_auroc']:.3f} "
                  f"trLD={mrow['train_eval_logit_diff_mean']:+.3f} trDp={mrow['train_eval_mean_p_diff_LUSC_minus_LUAD']:+.3f} | "
                  f"valCE={mrow['val_ce']:.4f} AUROC={mrow['val_auroc']:.3f} BA={mrow['val_balanced_accuracy']:.2f} "
                  f"F1={mrow['val_macro_f1']:.3f} pmean={mrow['val_p_mean']:.3f} LUAD/LUSC={mrow['val_n_pred_LUAD']}/{mrow['val_n_pred_LUSC']} | "
                  f"|h|={mrow['train_eval_feat_abs_mean']:.3f} skips={mrow['amp_skips']}", flush=True)
            if stop_after_epochs is not None and state["epoch"] == stop_after_epochs and state["epoch"] < epochs:
                outcome = "STOPPED_AFTER_EPOCH"
                print(f"STOPPED after epoch {state['epoch']} by request; resumable with the same --run command.", flush=True)
                return 0
        outcome = "COMPLETE"
        emit(run_dir, state)
        print(f"COMPLETE {mode}: {run_dir}", flush=True)
        return 0
    except KeyboardInterrupt:
        outcome = "INTERRUPTED"
        raise
    finally:
        ledger.close(outcome)
        status = {"COMPLETE": "COMPLETE", "FAILED": "FAILED"}.get(outcome, "PARTIAL")
        try:
            s29.atomic_json(run_dir / "status.json", {"status": status, "outcome": outcome, "identity": identity, "utc": s29.now(),
                                                      "resume": "repeat the same --run command", **s29.ledger_summary(run_dir / "sessions")})
        except OSError as e:
            print(f"WARNING: status not written ({e}); checkpoints remain authoritative.", flush=True)


# ----------------------------------------------------------------------------- checks / analysis (CPU)
def read_rows(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def check_original_vs_d08b():
    run_dir = MAIN_OUTPUT / "runs" / f"original_seed{SETTINGS['seed']}"
    mine = {int(r["epoch"]): r for r in read_rows(run_dir / "epoch_metrics.csv")}
    ref = {r["epoch"]: r for r in json.loads((D08B / "epochs.json").read_text())}
    keys = ["val_ce", "val_auroc", "val_balanced_accuracy", "val_macro_f1", "val_p_mean", "val_p_sd",
            "train_eval_ce", "train_eval_auroc", "train_eval_feat_abs_mean", "enc_update_mean", "head_update_mean"]
    epochs = sorted(e for e in mine if e >= 1)
    require(epochs, "original run has no completed epoch yet.")
    diffs = {e: {k: abs(float(mine[e][k]) - float(ref[e][k])) for k in keys if k in mine[e] and mine[e][k] != "" and k in ref[e]}
             for e in [0] + epochs}
    pm = [r for r in read_rows(run_dir / "validation_predictions.csv")]
    pr = [r for r in read_rows(D08B / "validation_predictions.csv") if int(r["epoch"]) in epochs]
    pred_equal = [(a["case_submitter_id"], a["label"], float(a["p_LUSC"])) for a in pm] == \
                 [(b["case_submitter_id"], b["label"], float(b["p_LUSC"])) for b in pr]
    groups = read_rows(run_dir / "groups.csv"); refg = [r for r in read_rows(D08B / "groups.csv") if int(r["epoch"]) in epochs]
    group_equal = [(g["patients"], g["bias_diff_LUSC_minus_LUAD"]) for g in groups] == [(g["patients"], g["bias_diff_LUSC_minus_LUAD"]) for g in refg]
    max_diff = max(v for d in diffs.values() for v in d.values())
    result = {"utc": s29.now() if s29 else None, "epochs_compared": [0] + epochs, "max_abs_diff": max_diff, "per_epoch_abs_diff": diffs,
              "validation_predictions_identical": pred_equal, "groups_and_bias_trajectory_identical": group_equal,
              "status": "PASS" if (max_diff == 0.0 and pred_equal and group_equal) else "MISMATCH"}
    s29.atomic_json(MAIN_OUTPUT / "check_original_vs_d08b.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "per_epoch_abs_diff"}, indent=1))
    return 0 if result["status"] == "PASS" else 1


def analyze():
    run_dir = MAIN_OUTPUT / "runs" / f"balanced_seed{SETTINGS['seed']}"
    bal = {int(r["epoch"]): {k: (float(v) if _num(v) else v) for k, v in r.items()} for r in read_rows(run_dir / "epoch_metrics.csv")}
    ref = {r["epoch"]: r for r in json.loads((D08B / "epochs.json").read_text()) if r["epoch"] <= MAIN_EPOCHS}
    require(sorted(bal) == list(range(MAIN_EPOCHS + 1)), "balanced run is not complete (epochs 0..10).")
    def primary(rows):
        ld = [rows[e]["train_eval_logit_diff_mean"] for e in range(1, MAIN_EPOCHS + 1)]
        ch = [abs(b - a) for a, b in zip(ld, ld[1:])]
        return {"value": float(np.mean(ch)), "transitions": [round(c, 4) for c in ch], "mean_logit_diff_by_epoch": [round(v, 4) for v in ld]}
    keys = ["train_eval_ce", "train_eval_auroc", "train_eval_mean_p_diff_LUSC_minus_LUAD", "train_eval_logit_diff_mean",
            "train_eval_p_mean", "train_eval_p_sd", "train_eval_n_pred_LUSC",
            "val_ce", "val_auroc", "val_balanced_accuracy", "val_macro_f1", "val_p_mean", "val_p_sd",
            "val_n_pred_LUAD", "val_n_pred_LUSC", "val_mean_p_diff_LUSC_minus_LUAD",
            "train_eval_feat_abs_mean", "train_eval_feat_between_patient_sd", "val_feat_abs_mean", "val_feat_between_patient_sd",
            "amp_skips", "enc_update_mean", "head_update_mean", "enc_rel_displacement_from_init", "head_rel_displacement_from_init"]
    table = []
    for e in range(0, MAIN_EPOCHS + 1):
        row = {"epoch": e}
        for k in keys:
            if k in bal[e] and k in ref[e]:
                row[f"{k}__balanced"] = bal[e][k]; row[f"{k}__d08b"] = ref[e][k]
        table.append(row)
    last5 = {k: {"balanced": float(np.mean([bal[e][k] for e in range(6, 11)])), "d08b": float(np.mean([ref[e][k] for e in range(6, 11)]))}
             for k in keys if all(k in bal[e] and k in ref[e] for e in range(6, 11))}
    result = {"primary_metric": PRIMARY_METRIC, "primary_balanced": primary(bal), "primary_d08b": primary(ref),
              "epochs_6_10_mean": last5, "note": "validation n=20, one seed: descriptive only"}
    s29.atomic_json(MAIN_OUTPUT / "comparison_balanced_vs_d08b.json", result)
    write_table(MAIN_OUTPUT / "comparison_balanced_vs_d08b_by_epoch.csv", table)
    print(json.dumps({k: v for k, v in result.items() if k != "epochs_6_10_mean"}, indent=1))
    for k, v in last5.items():
        print(f"  ep6-10 mean {k}: balanced={v['balanced']:.4f} d08b={v['d08b']:.4f}")
    return 0


def _num(v):
    try:
        float(v); return True
    except (TypeError, ValueError):
        return False


# ----------------------------------------------------------------------------- resume self-test (subprocesses)
def child(args_list):
    cmd = [sys.executable, "-B", str(Path(__file__).resolve())] + args_list
    r = subprocess.run(cmd, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
    return r.returncode


def resume_selftest():
    out = SELFTEST_OUTPUT
    require(child(["--prepare", "--output", str(out), "--epochs", "2"]) == 0, "selftest prepare failed")
    t0 = time.perf_counter()
    require(child(["--run", "--mode", "balanced", "--output", str(out), "--run-name", "continuous"]) == 0, "continuous run failed")
    t1 = time.perf_counter()
    # interruption 1: mid-epoch-1 (step 7); interruption 2: epoch 2 after epoch-1 metrics were committed (step 18)
    rc = [child(["--run", "--mode", "balanced", "--output", str(out), "--run-name", "interrupted", "--stop-after-steps", "7"]),
          child(["--run", "--mode", "balanced", "--output", str(out), "--run-name", "interrupted", "--stop-after-steps", "18"]),
          child(["--run", "--mode", "balanced", "--output", str(out), "--run-name", "interrupted"])]
    require(rc == [3, 3, 0], f"unexpected exit codes {rc}")
    t2 = time.perf_counter()
    return compare_selftest({"continuous": t1 - t0, "interrupted_total": t2 - t1})


TIMING_KEYS = ("train_seconds",)


def _strip_timing(rows):
    return [{k: v for k, v in r.items() if k not in TIMING_KEYS} for r in rows]


def compare_selftest(wall=None):
    """CPU comparison of the continuous vs interrupted self-test runs. Wall-clock fields are excluded from the
    equality check (they legitimately differ) and reported separately."""
    out = SELFTEST_OUTPUT
    import torch as T
    def final_state(name):
        d = out / "runs" / name / "checkpoints"
        last = sorted(d.glob("checkpoint_*"))[-1]
        return T.load(last / "state.pt", map_location="cpu", weights_only=True)
    a, b = final_state("continuous"), final_state("interrupted")
    load_stage29()
    s29.torch = T
    checks = {k: s29.tree_equal(a[k], b[k]) for k in ("model", "optimizer", "scaler", "rng")}
    checks["progress_metrics_identical_excluding_wall_time"] = s29.tree_equal(_strip_timing(a["progress"]["metrics"]), _strip_timing(b["progress"]["metrics"]))
    checks["progress_groups_identical"] = s29.tree_equal(a["progress"]["groups"], b["progress"]["groups"])
    checks["progress_steps_identical"] = s29.tree_equal(a["progress"]["steps"], b["progress"]["steps"])
    checks["progress_predictions_identical"] = s29.tree_equal(a["progress"]["predictions"], b["progress"]["predictions"])
    checks["metric_rows_no_duplicates"] = [r["epoch"] for r in b["progress"]["metrics"]] == [0, 1, 2]
    checks["group_rows_no_duplicates"] = [(r["epoch"], r["group"]) for r in b["progress"]["groups"]] == [(e, g) for e in (1, 2) for g in range(1, 16)]
    checks["csv_tables_identical"] = all((out / "runs" / "continuous" / n).read_bytes() == (out / "runs" / "interrupted" / n).read_bytes()
                                         for n in ("groups.csv", "steps.csv", "validation_predictions.csv"))
    checks["epoch_metrics_csv_identical_excluding_wall_time"] = _strip_timing(read_rows(out / "runs" / "continuous" / "epoch_metrics.csv")) == \
        _strip_timing(read_rows(out / "runs" / "interrupted" / "epoch_metrics.csv"))
    timing = {r["epoch"]: [r.get("train_seconds"), q.get("train_seconds")] for r, q in zip(a["progress"]["metrics"], b["progress"]["metrics"])}
    sessions = len(list((out / "runs" / "interrupted" / "sessions").glob("*.jsonl")))
    report = {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks, "interrupted_sessions": sessions,
              "interruptions": ["after optimizer step 7 (mid epoch 1)", "after optimizer step 18 (epoch 2, after epoch-1 commit)"],
              "excluded_from_equality": {"fields": list(TIMING_KEYS), "continuous_vs_interrupted_per_epoch": timing},
              "wall_seconds": wall, "epochs": 2, "mode": "balanced",
              "comparison_script_sha256": sha_file(Path(__file__))}
    previous = out / "resume_selftest.json"
    if previous.exists() and wall is None:
        report["wall_seconds"] = json.loads(previous.read_text()).get("wall_seconds")
        os.replace(previous, out / "resume_selftest_v0_timing_included.json")  # keep the first (FAIL-on-timing) report
    (out / "resume_selftest.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=1))
    return 0 if report["status"] == "PASS" else 1


# ----------------------------------------------------------------------------- CPU logic self-test
def self_test():
    import unittest
    load_stage29()

    class Tests(unittest.TestCase):
        def setUp(self):
            self.ids = [f"TCGA-AA-{i:04d}" for i in range(60)]
            self.labels = {p: ("LUAD" if i < 30 else "LUSC") for i, p in enumerate(self.ids)}
        def test_balanced_is_2_plus_2_each_once(self):
            for e in range(12):
                gs = make_groups("balanced", self.ids, self.labels, e)
                self.assertEqual(len(gs), 15)
                self.assertEqual(sorted(p for g in gs for p in g), sorted(self.ids))
                for g in gs:
                    self.assertEqual(sum(self.labels[p] == "LUSC" for p in g), 2)
        def test_balanced_preserves_class_order_and_within_group_order(self):
            order = s29.patient_order(self.ids, 3); pos = {p: i for i, p in enumerate(order)}
            gs = make_groups("balanced", self.ids, self.labels, 3)
            luad = [p for g in gs for p in g if self.labels[p] == "LUAD"]
            self.assertEqual(luad, [p for p in order if self.labels[p] == "LUAD"])
            for g in gs:
                self.assertEqual(g, sorted(g, key=pos.__getitem__))
        def test_original_equals_stage29_slices(self):
            order = s29.patient_order(self.ids, 5)
            self.assertEqual(make_groups("original", self.ids, self.labels, 5), [order[i:i + 4] for i in range(0, 60, 4)])
        def test_deterministic(self):
            self.assertEqual(make_groups("balanced", self.ids, self.labels, 7), make_groups("balanced", list(reversed(self.ids)), self.labels, 7))

    r = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    return 0 if r.wasSuccessful() else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    m = ap.add_mutually_exclusive_group(required=True)
    for flag in ("--self-test", "--preflight-only", "--prepare", "--run", "--check-original-vs-d08b", "--resume-selftest",
                 "--resume-selftest-compare", "--analyze"):
        m.add_argument(flag, action="store_true")
    ap.add_argument("--mode", choices=MODES)
    ap.add_argument("--output", type=Path, default=MAIN_OUTPUT, help=argparse.SUPPRESS)
    ap.add_argument("--epochs", type=int, default=MAIN_EPOCHS, help=argparse.SUPPRESS)
    ap.add_argument("--run-name", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--stop-after-steps", type=int, default=None, help="test hook: commit checkpoint and exit(3) after N optimizer steps")
    ap.add_argument("--stop-after-epochs", type=int, default=None, help="commit and stop after N epochs (resumable)")
    args = ap.parse_args()
    output = args.output.resolve()
    if output == MAIN_OUTPUT.resolve():
        if args.epochs != MAIN_EPOCHS or args.run_name is not None or args.stop_after_steps is not None:
            raise SystemExit("ERROR: the main output uses fixed settings (10 epochs, default run names, no step-stop test hook).")
    elif output != SELFTEST_OUTPUT.resolve():
        raise SystemExit("ERROR: only the main output or the resume-selftest folder are allowed.")
    global torch, np
    load_stage29()
    try:
        if args.self_test:
            return self_test()
        if args.check_original_vs_d08b:
            return check_original_vs_d08b()
        if args.analyze:
            import numpy as _np
            np = _np
            return analyze()
        if args.resume_selftest:
            return resume_selftest()
        if args.resume_selftest_compare:
            return compare_selftest()
        if args.preflight_only:
            preflight(); return 0
        if args.prepare:
            prepare(output, args.epochs); return 0
        if args.run:
            require(args.mode in MODES, "--run needs --mode original|balanced")
            return run(output, args.mode, args.run_name, args.stop_after_steps, args.stop_after_epochs)
    except s29.ExperimentError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nINTERRUPTED. Repeat the same command to resume from the last verified checkpoint.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
