#!/usr/bin/env python3
r"""Diagnostic D08: D07 head-LR A/B extended to a pre-declared 20-epoch horizon (rerun from scratch).

D07 saved no checkpoints, so epoch-5 continuation is impossible; both configs restart from the identical initial
state in new folders. Epochs 1-5 must reproduce D07 (checked after the run; recomputation counted separately).
  A: encoder LR 1e-5, head LR 1e-3      B: encoder LR 1e-5, head LR 1e-4
All else = frozen Stage 29 (hash checked): K=64, 60 pilot-train / 20 pilot-validation, seed 20261113, original
classifier init, Stage 29 patient order / D4 augmentation / sampling, AMP, frozen BN, accumulation 4, AdamW.
No LR scheduler, no threshold change (0.5), nothing adapted to interim results.

Added vs D07 (observation only; training path unchanged):
  * per accumulation group: patient IDs, label sequence, group mean loss, encoder/head update norms,
    classifier.bias (both, float64 repr) and b_LUSC - b_LUAD;
  * at every group boundary of epochs 4 and 5: fixed balanced TRAIN probe (first 4 LUAD + first 4 LUSC by
    pilot_order_within_class, all 64 patches, no augmentation, eval mode): mean/SD of logit diff LUSC-LUAD, SD p;
  * per epoch train-eval mean/SD logit diff (shift analysis uses logit differences, not single logits);
  * every diagnostic pass asserts torch/CUDA/python/numpy RNG states and BN buffers are unchanged and that
    model.train() is restored.

  .\.venv\Scripts\python.exe -B results\stage29_learning_investigation_v1\code\d08_head_lr_ab_20ep.py --config A
  .\.venv\Scripts\python.exe -B results\stage29_learning_investigation_v1\code\d08_head_lr_ab_20ep.py --config B
"""
import argparse, csv, hashlib, importlib.util, io, json, math, random, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results" / "stage29_learning_investigation_v1"
S29_PATH = ROOT / "scripts" / "29_train_patch_budget_baselines.py"
CONFIGS = {"A": {"encoder_lr": 1e-5, "head_lr": 1e-3}, "B": {"encoder_lr": 1e-5, "head_lr": 1e-4}}
BUDGET, EPOCHS, PROBE_EPOCHS = 64, 20, (4, 5)

PLAN = {
    "horizon": "20 epochs total, fixed before launch; no early stopping; no LR/scheduler/sampling/normalisation/threshold changes",
    "restart_reason": "D07 saved no checkpoints (model/optimizer/scaler/RNG/data position unavailable) -> rerun from scratch",
    "reproduction_check": "epochs 1-5 val CE/AUROC/p_mean and train-eval CE must equal D07 (same config); recomputed epochs 1-5 reported separately in cost",
    "outputs": ["full train-eval / validation CE, AUROC, balanced accuracy, macro-F1 per epoch",
                "mean p, SD, range, n predicted per class per epoch (train-eval and validation), threshold 0.5",
                "pooled-feature |h| and between-patient SD per epoch (train-eval and validation)",
                "last-five-epoch (16-20) means/SD per config, not best checkpoint",
                "per-group: IDs, labels, loss, update norms, bias values and b_LUSC-b_LUAD",
                "epochs 4-5 group-boundary train probe: mean logit diff LUSC-LUAD, SD p"],
    "analysis": ["shift = change of mean logit diff (LUSC-LUAD); a common shift of both logits is ignored",
                 "group label composition vs probe shift reported as association only, not cause",
                 "equal update norms are not taken as equal weight changes",
                 "validation differences are descriptive (n=20, one seed, no formal test); no equivalence claims",
                 "conclusion category: continued learning / overfitting / persistent instability, from train-eval vs validation CE and separation trends over epochs 6-20"],
    "probe_patients": "first 4 LUAD + first 4 LUSC pilot-train by pilot_order_within_class (same as D03/D04)",
}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", choices=tuple(CONFIGS), required=True)
    args = ap.parse_args(); cfg = CONFIGS[args.config]
    run_dir = OUT / f"d08_head_lr_{args.config}_B{BUDGET}_20ep"
    run_dir.mkdir(parents=True, exist_ok=False)
    spec = importlib.util.spec_from_file_location("stage29_frozen", S29_PATH)
    s29 = importlib.util.module_from_spec(spec); spec.loader.exec_module(s29)
    bundle = s29.load_bundle(ROOT)
    protocol, _, weights = s29.read_prepared(bundle, None)
    assert protocol["script_sha256"] == hashlib.sha256(S29_PATH.read_bytes()).hexdigest()
    (run_dir / "prereg.json").write_text(json.dumps({
        "utc": s29.now(), "config": args.config,
        "settings": {**cfg, "budget": BUDGET, "seed": s29.SEED, "epochs": EPOCHS, "weight_decay": 1e-4,
                     "accumulation": s29.ACCUMULATION, "classifier_init": "original (build_model, seed 20261113)",
                     "threshold": 0.5},
        **PLAN, "command": f".\\.venv\\Scripts\\python.exe -B results\\stage29_learning_investigation_v1\\code\\d08_head_lr_ab_20ep.py --config {args.config}",
        "diagnostic_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "stage29_script_sha256": protocol["script_sha256"]}, indent=2))
    device = s29.initialize_torch(cuda=True); torch, np = s29.torch, s29.np
    from PIL import Image
    from sklearn.metrics import roc_auc_score

    model = s29.build_model(weights, device=device)
    optimizer, scaler = s29.optimizer_for(model), s29.scaler_for(device)
    optimizer.param_groups[0]["lr"] = cfg["encoder_lr"]; optimizer.param_groups[1]["lr"] = cfg["head_lr"]
    frozen_bn = s29.bn_state(model)
    enc = list(model.encoder.parameters()); head = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    init_enc = [p.detach().clone() for p in enc]; init_head = [p.detach().clone() for p in head]
    nrm = lambda ts: math.sqrt(sum(float((t.float() ** 2).sum()) for t in ts))
    init_enc_norm, init_head_norm = nrm(init_enc), nrm(init_head)
    steps, prev = [], {}
    optimizer.register_step_pre_hook(lambda o, a, k: prev.update(e=[p.detach().clone() for p in enc], h=[p.detach().clone() for p in head]))
    def post(o, a, k):
        with torch.no_grad():
            steps.append({"enc_update": nrm([p - q for p, q in zip(enc, prev["e"])]),
                          "head_update": nrm([p - q for p, q in zip(head, prev["h"])]),
                          "lr": [g["lr"] for g in o.param_groups]})
    optimizer.register_step_post_hook(post)

    def rng_fingerprint():
        n = np.random.get_state()
        return (hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
                tuple(hashlib.sha256(s.numpy().tobytes()).hexdigest() for s in torch.cuda.get_rng_state_all()),
                hash(random.getstate()), hashlib.sha256(n[1].tobytes()).hexdigest() + str(n[2]))

    def guarded(fn):
        """Diagnostic must not change RNG, BN buffers or train mode."""
        def wrapper(*a, **k):
            before = rng_fingerprint()
            out = fn(*a, **k)
            assert rng_fingerprint() == before, "diagnostic changed RNG state"
            assert s29.tree_equal(frozen_bn, s29.bn_state(model)), "diagnostic changed BN buffers"
            assert model.training and all(not m.training for m in model.encoder.modules()
                                          if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)), "train/BN mode not restored"
            return out
        return wrapper

    mean = torch.tensor([.485, .456, .406]).view(1, 3, 1, 1); std = torch.tensor([.229, .224, .225]).view(1, 3, 1, 1)
    train_ids = s29.pilot_ids(bundle, "train")
    train_y = np.array([int(bundle["patients"][p]["class_label"] == "LUSC") for p in train_ids])
    train_bags = {}
    for p in train_ids:
        x = torch.stack([torch.from_numpy(np.array(Image.open(io.BytesIO(s29.verified_png_bytes(r))).convert("RGB"))).permute(2, 0, 1)
                         for r in bundle["patches"][p]]).float() / 255.
        train_bags[p] = ((x - mean) / std).half()
    probe_ids = []
    for label in ("LUAD", "LUSC"):
        rows = sorted((r for r in bundle["pilot"].values() if r["split"] == "train" and r["class_label"] == label),
                      key=lambda r: int(r["pilot_order_within_class"]))
        probe_ids += [r["case_submitter_id"] for r in rows[:4]]
    probe_y = np.array([int(bundle["patients"][p]["class_label"] == "LUSC") for p in probe_ids])

    def forward_eval(x):
        with torch.autocast("cuda", dtype=torch.float16):
            h = model.encoder(x)
        h = h.float()
        s = model.attention_w(torch.tanh(model.attention_v(h)) * torch.sigmoid(model.attention_u(h))).squeeze(-1)
        a = torch.softmax(s, 0); z = (a[:, None] * h).sum(0, keepdim=True)
        return model.classifier(z), z[0]

    def summarise(prefix, y, p, ld=None, Z=None):
        y = np.asarray(y); p = np.asarray(p); pred = (p >= .5).astype(int)
        out = {f"{prefix}_p_mean": float(p.mean()), f"{prefix}_p_sd": float(p.std()), f"{prefix}_p_min": float(p.min()),
               f"{prefix}_p_max": float(p.max()), f"{prefix}_n_pred_LUAD": int((pred == 0).sum()), f"{prefix}_n_pred_LUSC": int(pred.sum()),
               f"{prefix}_mean_p_diff_LUSC_minus_LUAD": float(p[y == 1].mean() - p[y == 0].mean()),
               f"{prefix}_balanced_accuracy": float(.5 * (((pred == 1) & (y == 1)).sum() / (y == 1).sum() + ((pred == 0) & (y == 0)).sum() / (y == 0).sum()))}
        if ld is not None:
            ld = np.asarray(ld)
            out.update({f"{prefix}_logit_diff_mean": float(ld.mean()), f"{prefix}_logit_diff_sd": float(ld.std()),
                        f"{prefix}_logit_diff_class_gap": float(ld[y == 1].mean() - ld[y == 0].mean())})
        if Z is not None:
            out.update({f"{prefix}_feat_abs_mean": float(np.abs(Z).mean()), f"{prefix}_feat_between_patient_sd": float(Z.std(0).mean())})
        return out

    @guarded
    def eval_train():
        model.eval(); ps, ces, lds, Z = [], [], [], []
        with torch.inference_mode():
            for p, yy in zip(train_ids, train_y):
                lg, z = forward_eval(train_bags[p].float().to(device)); lg = lg.float()
                ces.append(float(torch.nn.functional.cross_entropy(lg, torch.tensor([int(yy)], device=device))))
                ps.append(float(torch.softmax(lg, 1)[0, 1])); lds.append(float(lg[0, 1] - lg[0, 0])); Z.append(z.cpu().numpy())
        model.train()
        m = {"train_eval_ce": float(np.mean(ces)), "train_eval_auroc": float(roc_auc_score(train_y, ps))}
        m.update(summarise("train_eval", train_y, ps, lds, np.stack(Z)))
        return m

    @guarded
    def eval_val_features():
        model.eval(); Z, lds = [], []
        with torch.inference_mode():
            for p in s29.pilot_ids(bundle, "validation"):
                x, _ = s29.load_bag(bundle, p, 0, 64, False, device)
                lg, z = forward_eval(x); Z.append(z.cpu().numpy()); lds.append(float(lg[0, 1] - lg[0, 0]))
        model.train(); Z = np.stack(Z)
        return {"val_feat_abs_mean": float(np.abs(Z).mean()), "val_feat_between_patient_sd": float(Z.std(0).mean()),
                "val_logit_diff_mean": float(np.mean(lds)), "val_logit_diff_sd": float(np.std(lds))}

    @guarded
    def probe():
        model.eval(); lds, ps = [], []
        with torch.inference_mode():
            for p in probe_ids:
                lg = forward_eval(train_bags[p].float().to(device))[0].float()
                lds.append(float(lg[0, 1] - lg[0, 0])); ps.append(float(torch.softmax(lg, 1)[0, 1]))
        model.train(); lds = np.array(lds)
        return {"probe_logit_diff_mean": float(lds.mean()), "probe_logit_diff_sd": float(lds.std()),
                "probe_logit_diff_class_gap": float(lds[probe_y == 1].mean() - lds[probe_y == 0].mean()),
                "probe_p_sd": float(np.std(ps)), "probe_p_mean": float(np.mean(ps))}

    def _validate(e):  # Stage 29 validate_model leaves eval mode; Stage 29 train_group re-enters train() itself
        out = s29.validate_model(bundle, model, e, device); model.train(); return out
    validate = guarded(_validate)
    def bias():
        b = model.classifier.bias.detach().double().cpu().tolist()
        return {"bias_LUAD": repr(b[0]), "bias_LUSC": repr(b[1]), "bias_diff_LUSC_minus_LUAD": repr(b[1] - b[0])}

    rows, preds, groups, probes = [{"epoch": 0, **eval_train(), **eval_val_features(), **bias()}], [], [], []
    print("ep0", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in rows[0].items() if k != "epoch"}, flush=True)
    t_all = time.perf_counter(); epoch_times = []
    for epoch in range(EPOCHS):
        te = time.perf_counter()
        ids = s29.patient_order(train_ids, epoch)
        loader = lambda pid: s29.load_bag(bundle, pid, epoch, BUDGET, True, device)
        n0, loss_sum, skips, opt_steps, train_s = len(steps), 0., 0, 0, 0.
        do_probe = (epoch + 1) in PROBE_EPOCHS
        if do_probe:
            probes.append({"epoch": epoch + 1, "after_group": 0, **probe(), **bias()})
        for gi, (a, b) in enumerate(s29.group_slices(len(ids))):
            t0 = time.perf_counter()
            g = s29.train_group(model, optimizer, scaler, ids[a:b], loader, device)
            train_s += time.perf_counter() - t0
            loss_sum += g["loss_sum"]; skips += g["amp_skips"]; opt_steps += g["optimizer_steps"]
            labs = ["LUSC" if bundle["patients"][p]["class_label"] == "LUSC" else "LUAD" for p in ids[a:b]]
            groups.append({"epoch": epoch + 1, "group": gi + 1, "patients": "|".join(ids[a:b]), "labels": "|".join(labs),
                           "n_LUSC": labs.count("LUSC"), "n_LUAD": labs.count("LUAD"), "group_mean_loss": g["loss_sum"] / g["patients"],
                           "enc_update": steps[-1]["enc_update"], "head_update": steps[-1]["head_update"],
                           "amp_skips": g["amp_skips"], **bias()})
            if do_probe:
                probes.append({"epoch": epoch + 1, "after_group": gi + 1, "n_LUSC_in_group": labs.count("LUSC"), **probe(), **bias()})
        assert s29.tree_equal(frozen_bn, s29.bn_state(model)), "BN running stats changed"
        val, pr = validate(epoch)
        vy = [r["label"] for r in pr]; vp = [r["p_LUSC"] for r in pr]
        st = steps[n0:]
        with torch.no_grad():
            enc_disp = nrm([p - q for p, q in zip(enc, init_enc)]); head_disp = nrm([p - q for p, q in zip(head, init_head)])
        row = {"epoch": epoch + 1, "train_aug_ce_running": loss_sum / 60, "train_seconds": train_s,
               "optimizer_steps": opt_steps, "amp_skips": skips, "lr_actual": st[-1]["lr"],
               "enc_update_sum": sum(s["enc_update"] for s in st), "enc_update_mean": float(np.mean([s["enc_update"] for s in st])),
               "head_update_sum": sum(s["head_update"] for s in st), "head_update_mean": float(np.mean([s["head_update"] for s in st])),
               "enc_rel_displacement_from_init": enc_disp / init_enc_norm, "head_rel_displacement_from_init": head_disp / init_head_norm,
               "val_ce": val["loss"], "val_auroc": val["auroc"], "val_balanced_accuracy": val["balanced_accuracy"],
               "val_macro_f1": val["macro_f1"], "val_accuracy": val["accuracy"], **bias()}
        row.update(summarise("val", vy, vp)); row.pop("val_balanced_accuracy", None)
        row["val_balanced_accuracy"] = val["balanced_accuracy"]
        row.update(eval_val_features()); row.update(eval_train())
        rows.append(row); preds += pr; epoch_times.append(time.perf_counter() - te)
        print(f"{args.config} ep{epoch+1:2d}: trEvalCE={row['train_eval_ce']:.4f} trAUROC={row['train_eval_auroc']:.3f} "
              f"trBA={row['train_eval_balanced_accuracy']:.2f} trLD={row['train_eval_logit_diff_mean']:+.3f} | valCE={row['val_ce']:.4f} "
              f"AUROC={row['val_auroc']:.3f} BA={row['val_balanced_accuracy']:.2f} F1={row['val_macro_f1']:.3f} pmean={row['val_p_mean']:.3f} "
              f"psd={row['val_p_sd']:.4f} LUAD/LUSC={row['val_n_pred_LUAD']}/{row['val_n_pred_LUSC']} dP={row['val_mean_p_diff_LUSC_minus_LUAD']:+.4f} | "
              f"|h|={row['train_eval_feat_abs_mean']:.3f} sdh={row['train_eval_feat_between_patient_sd']:.3f} skips={skips} t={epoch_times[-1]:.0f}s", flush=True)
        # persist incrementally (power-loss safety; files are new and owned by this run)
        (run_dir / "epochs.json").write_text(json.dumps(rows, indent=1))
    (run_dir / "steps.json").write_text(json.dumps(steps, indent=1))
    for name, data in (("validation_predictions.csv", preds), ("groups.csv", groups), ("probe_epochs4_5.csv", probes)):
        keys = list(dict.fromkeys(k for r in data for k in r))
        with open(run_dir / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(data)
    flat = [{k: (json.dumps(v) if isinstance(v, list) else v) for k, v in r.items()} for r in rows]
    keys = list(dict.fromkeys(k for r in flat for k in r))
    with open(run_dir / "epoch_metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(flat)
    (run_dir / "summary.json").write_text(json.dumps({
        "config": args.config, **cfg, "epochs": EPOCHS, "wall_seconds_after_setup": time.perf_counter() - t_all,
        "epoch_wall_seconds_incl_diagnostics": epoch_times,
        "recomputed_epochs_1_5_wall_seconds": sum(epoch_times[:5]),
        "total_optimizer_steps": len(steps)}, indent=2))
    print("DONE", args.config, flush=True)


if __name__ == "__main__":
    sys.exit(main())
