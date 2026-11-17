#!/usr/bin/env python3
r"""Diagnostic D07: one-factor test of HEAD LR after removing feature collapse (encoder LR 1e-5).

  A: encoder LR 1e-5, head LR 1e-3      B: encoder LR 1e-5, head LR 1e-4
Everything else identical to the Stage 29 pilot loop (frozen module, hash checked): K=64, same 60 pilot-train /
20 pilot-validation patients, seed 20261113, 5 epochs, same initial weights incl. the ORIGINAL random classifier
init (no zero-init), patient order, SHA-keyed D4 augmentation, AMP FP16 encoder / FP32 head, frozen BN stats,
accumulation 4, AdamW (wd 1e-4, foreach=False). LRs are set on the optimizer param_groups after optimizer_for().
Training uses no torch RNG (sampling/augmentation are SHA-keyed), so the extra eval passes cannot alter the run.

  .\.venv\Scripts\python.exe -B results\stage29_learning_investigation_v1\code\d07_head_lr_ab.py --config A
  .\.venv\Scripts\python.exe -B results\stage29_learning_investigation_v1\code\d07_head_lr_ab.py --config B
Writes only to results\stage29_learning_investigation_v1\d07_head_lr_<config>_B64\. No test patients.
"""
import argparse, csv, hashlib, importlib.util, io, json, math, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results" / "stage29_learning_investigation_v1"
S29_PATH = ROOT / "scripts" / "29_train_patch_budget_baselines.py"
CONFIGS = {"A": {"encoder_lr": 1e-5, "head_lr": 1e-3}, "B": {"encoder_lr": 1e-5, "head_lr": 1e-4}}
BUDGET = 64

PREREG = {
    "question": "Does lowering head LR (1e-3 -> 1e-4) reduce prediction oscillation once feature collapse is removed, "
                "WITH evidence of learning and patient separation (not merely stable ~0.5 outputs)?",
    "single_factor": "head LR (param_groups[1])",
    "fixed": "K=64; 60 pilot train / 20 pilot validation; seed 20261113; 5 epochs; same initial weights incl. original "
             "random classifier init; Stage 29 patient order, D4 augmentation, AMP, frozen BN, accumulation 4, AdamW",
    "metrics_per_epoch": [
        "train CE in eval mode on identical inputs (60 train patients, all 64 patches, no augmentation) + train AUROC",
        "validation CE, AUROC, balanced accuracy, macro-F1 (Stage 29 validate_model, 64 patches)",
        "p_LUSC mean, SD, min, max; number predicted LUAD / LUSC (threshold 0.5) -- train-eval and validation",
        "difference of mean p_LUSC between true LUSC and true LUAD -- train-eval and validation",
        "pooled-feature |h| mean and between-patient SD (train-eval and validation)",
        "encoder/head update norms per step (sum, mean) and cumulative relative displacement from init; AMP skips",
        "mean |logit_diff| step-to-step common-mode shift: per-epoch change of train-eval mean logit difference"],
    "interpretation_rules": [
        "Reduced oscillation of mean p_LUSC counts only if accompanied by (a) decreasing train-eval CE and "
        "(b) growing train-eval class separation (mean p_LUSC difference / train AUROC) with non-collapsed features",
        "No configuration is chosen by the best single-epoch validation AUROC (n=20 is descriptive only)",
        "Validation is evaluated only; never used for training or selection of patients"],
}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", choices=tuple(CONFIGS), required=True)
    args = ap.parse_args(); cfg = CONFIGS[args.config]
    run_dir = OUT / f"d07_head_lr_{args.config}_B{BUDGET}"
    run_dir.mkdir(parents=True, exist_ok=False)
    spec = importlib.util.spec_from_file_location("stage29_frozen", S29_PATH)
    s29 = importlib.util.module_from_spec(spec); spec.loader.exec_module(s29)
    bundle = s29.load_bundle(ROOT)
    protocol, _, weights = s29.read_prepared(bundle, None)
    assert protocol["script_sha256"] == hashlib.sha256(S29_PATH.read_bytes()).hexdigest()
    (run_dir / "prereg.json").write_text(json.dumps({
        "utc": s29.now(), "config": args.config, "settings": {**cfg, "budget": BUDGET, "seed": s29.SEED, "epochs": s29.EPOCHS,
        "weight_decay": 1e-4, "accumulation": s29.ACCUMULATION, "classifier_init": "original (build_model, seed 20261113)"},
        **PREREG, "command": f".\\.venv\\Scripts\\python.exe -B results\\stage29_learning_investigation_v1\\code\\d07_head_lr_ab.py --config {args.config}",
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

    mean = torch.tensor([.485, .456, .406]).view(1, 3, 1, 1); std = torch.tensor([.229, .224, .225]).view(1, 3, 1, 1)
    train_ids = s29.pilot_ids(bundle, "train")
    train_y = np.array([int(bundle["patients"][p]["class_label"] == "LUSC") for p in train_ids])
    train_bags = {}
    for p in train_ids:  # identical eval inputs every epoch: verified bytes, all 64 patches, no augmentation
        x = torch.stack([torch.from_numpy(np.array(Image.open(io.BytesIO(s29.verified_png_bytes(r))).convert("RGB"))).permute(2, 0, 1)
                         for r in bundle["patches"][p]]).float() / 255.
        train_bags[p] = ((x - mean) / std).half()

    def forward_eval(x):
        with torch.autocast("cuda", dtype=torch.float16):
            h = model.encoder(x)
        h = h.float()
        s = model.attention_w(torch.tanh(model.attention_v(h)) * torch.sigmoid(model.attention_u(h))).squeeze(-1)
        a = torch.softmax(s, 0); z = (a[:, None] * h).sum(0, keepdim=True)
        return model.classifier(z), z[0]

    def summarise(prefix, y, p, Z):
        y = np.asarray(y); p = np.asarray(p); pred = (p >= .5).astype(int)
        return {f"{prefix}_p_mean": float(p.mean()), f"{prefix}_p_sd": float(p.std()), f"{prefix}_p_min": float(p.min()),
                f"{prefix}_p_max": float(p.max()), f"{prefix}_n_pred_LUAD": int((pred == 0).sum()), f"{prefix}_n_pred_LUSC": int(pred.sum()),
                f"{prefix}_mean_p_diff_LUSC_minus_LUAD": float(p[y == 1].mean() - p[y == 0].mean()),
                f"{prefix}_feat_abs_mean": float(np.abs(Z).mean()), f"{prefix}_feat_between_patient_sd": float(Z.std(0).mean())}

    def eval_train():
        model.eval(); ps, ces, Z = [], [], []
        with torch.inference_mode():
            for p, yy in zip(train_ids, train_y):
                lg, z = forward_eval(train_bags[p].float().to(device))
                ces.append(float(torch.nn.functional.cross_entropy(lg.float(), torch.tensor([int(yy)], device=device))))
                ps.append(float(torch.softmax(lg.float(), 1)[0, 1])); Z.append(z.cpu().numpy())
        model.train()
        m = {"train_eval_ce": float(np.mean(ces)), "train_eval_auroc": float(roc_auc_score(train_y, ps))}
        m.update(summarise("train_eval", train_y, ps, np.stack(Z)))
        return m

    def eval_val_features():
        model.eval(); Z = []
        with torch.inference_mode():
            for p in s29.pilot_ids(bundle, "validation"):
                x, _ = s29.load_bag(bundle, p, 0, 64, False, device)
                Z.append(forward_eval(x)[1].cpu().numpy())
        model.train(); Z = np.stack(Z)
        return {"val_feat_abs_mean": float(np.abs(Z).mean()), "val_feat_between_patient_sd": float(Z.std(0).mean())}

    rows, preds = [{"epoch": 0, **eval_train()}], []
    print("ep0", {k: round(v, 4) for k, v in rows[0].items() if k != "epoch"}, flush=True)
    t_all = time.perf_counter()
    for epoch in range(s29.EPOCHS):
        ids = s29.patient_order(train_ids, epoch)
        loader = lambda pid: s29.load_bag(bundle, pid, epoch, BUDGET, True, device)
        n0, loss_sum, skips, opt_steps, t0 = len(steps), 0., 0, 0, time.perf_counter()
        for a, b in s29.group_slices(len(ids)):
            g = s29.train_group(model, optimizer, scaler, ids[a:b], loader, device)
            loss_sum += g["loss_sum"]; skips += g["amp_skips"]; opt_steps += g["optimizer_steps"]
        train_s = time.perf_counter() - t0
        assert s29.tree_equal(frozen_bn, s29.bn_state(model)), "BN running stats changed"
        val, pr = s29.validate_model(bundle, model, epoch, device)
        vy = [r["label"] for r in pr]; vp = [r["p_LUSC"] for r in pr]
        st = steps[n0:]
        with torch.no_grad():
            enc_disp = nrm([p - q for p, q in zip(enc, init_enc)]); head_disp = nrm([p - q for p, q in zip(head, init_head)])
        row = {"epoch": epoch + 1, "train_aug_ce_running": loss_sum / 60, "train_seconds": train_s,
               "optimizer_steps": opt_steps, "amp_skips": skips, "lr_actual": st[-1]["lr"],
               "enc_update_sum": sum(s["enc_update"] for s in st), "enc_update_mean": float(np.mean([s["enc_update"] for s in st])),
               "head_update_sum": sum(s["head_update"] for s in st), "head_update_mean": float(np.mean([s["head_update"] for s in st])),
               "enc_rel_displacement_from_init": enc_disp / init_enc_norm, "head_rel_displacement_from_init": head_disp / init_head_norm,
               "classifier_bias": model.classifier.bias.detach().cpu().tolist(),
               "val_ce": val["loss"], "val_auroc": val["auroc"], "val_balanced_accuracy": val["balanced_accuracy"],
               "val_macro_f1": val["macro_f1"], "val_accuracy": val["accuracy"]}
        row.update(summarise("val", vy, vp, np.zeros((1, 1))))
        row.pop("val_feat_abs_mean"); row.pop("val_feat_between_patient_sd")
        row.update(eval_val_features()); row.update(eval_train())
        rows.append(row); preds += pr
        print(f"{args.config} ep{epoch+1}: trainEvalCE={row['train_eval_ce']:.4f} trainAUROC={row['train_eval_auroc']:.3f} "
              f"trainDiff={row['train_eval_mean_p_diff_LUSC_minus_LUAD']:+.4f} trainPmean={row['train_eval_p_mean']:.3f} | "
              f"valCE={row['val_ce']:.4f} AUROC={row['val_auroc']:.3f} BA={row['val_balanced_accuracy']:.2f} F1={row['val_macro_f1']:.3f} "
              f"pmean={row['val_p_mean']:.3f} psd={row['val_p_sd']:.4f} pred LUAD/LUSC={row['val_n_pred_LUAD']}/{row['val_n_pred_LUSC']} "
              f"diff={row['val_mean_p_diff_LUSC_minus_LUAD']:+.4f} | |h|tr={row['train_eval_feat_abs_mean']:.3f} "
              f"sdh={row['train_eval_feat_between_patient_sd']:.3f} encDisp={row['enc_rel_displacement_from_init']:.4f} "
              f"headDisp={row['head_rel_displacement_from_init']:.4f} skips={skips}", flush=True)
    (run_dir / "epochs.json").write_text(json.dumps(rows, indent=1))
    (run_dir / "steps.json").write_text(json.dumps(steps, indent=1))
    with open(run_dir / "validation_predictions.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(preds[0])); w.writeheader(); w.writerows(preds)
    flat = [{k: (json.dumps(v) if isinstance(v, list) else v) for k, v in r.items()} for r in rows]
    keys = sorted({k for r in flat for k in r}, key=lambda k: (k != "epoch", k))
    with open(run_dir / "epoch_metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(flat)
    (run_dir / "summary.json").write_text(json.dumps({"config": args.config, **cfg, "wall_seconds_after_setup": time.perf_counter() - t_all,
                                                      "total_optimizer_steps": len(steps)}, indent=2))


if __name__ == "__main__":
    sys.exit(main())
