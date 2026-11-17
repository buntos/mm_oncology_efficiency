#!/usr/bin/env python3
r"""Diagnostic D06: does the candidate fix remove the ORIGINAL pilot symptom?

Re-runs the Stage 29 pilot loop (same 60 pilot-train patients, K, patch sampling, D4 aug, patient order,
accumulation, AMP, optimizer, 5 epochs, seed, validation on the same 20 pilot-validation patients with all
64 patches) using the frozen Stage 29 functions, with exactly ONE change:
  --variant zeroinit_head : classifier weight/bias = 0 after build_model (pre-declared minimal fix)
  --variant enc_lr_1e-5   : encoder LR 1e-5 (alternative, run for comparison only)
No checkpoints, no writes into Stage 29 folders. Validation is only evaluated (never trained on / selected from);
the fix was declared before this run (see prereg.json). Test patients are never loaded (Stage 29 guard).
  .\.venv\Scripts\python.exe -B results\stage29_learning_investigation_v1\code\d06_pilot_symptom_check.py --variant zeroinit_head --budget 16
"""
import argparse, csv, hashlib, importlib.util, io, json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results" / "stage29_learning_investigation_v1"
S29_PATH = ROOT / "scripts" / "29_train_patch_budget_baselines.py"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=("zeroinit_head", "enc_lr_1e-5"), required=True)
    ap.add_argument("--budget", type=int, choices=(64, 32, 16), required=True)
    args = ap.parse_args()
    run_dir = OUT / f"d06_pilot_{args.variant}_B{args.budget}"
    run_dir.mkdir(parents=True, exist_ok=False)
    spec = importlib.util.spec_from_file_location("stage29_frozen", S29_PATH)
    s29 = importlib.util.module_from_spec(spec); spec.loader.exec_module(s29)
    bundle = s29.load_bundle(ROOT)
    protocol, _, weights = s29.read_prepared(bundle, None)
    assert protocol["script_sha256"] == hashlib.sha256(S29_PATH.read_bytes()).hexdigest()
    (run_dir / "prereg.json").write_text(json.dumps({
        "utc": s29.now(), "variant": args.variant, "budget": args.budget, "seed": s29.SEED, "epochs": s29.EPOCHS,
        "single_change": {"zeroinit_head": "classifier.weight=0, classifier.bias=0",
                          "enc_lr_1e-5": "optimizer param_groups[0].lr=1e-5"}[args.variant],
        "primary_fix_declared_before_run": "zeroinit_head",
        "symptom_metrics": ["validation p_LUSC SD per epoch", "fraction of validation patients flipping class together",
                            "validation AUROC (n=20, descriptive only)", "train pooled-feature magnitude after epoch 5"],
        "diagnostic_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "stage29_script_sha256": protocol["script_sha256"]}, indent=2))
    device = s29.initialize_torch(cuda=True); torch, np = s29.torch, s29.np
    model = s29.build_model(weights, device=device)
    if args.variant == "zeroinit_head":
        with torch.no_grad():
            model.classifier.weight.zero_(); model.classifier.bias.zero_()
    optimizer, scaler = s29.optimizer_for(model), s29.scaler_for(device)
    if args.variant == "enc_lr_1e-5":
        optimizer.param_groups[0]["lr"] = 1e-5
    frozen_bn = s29.bn_state(model)
    metrics, preds = [], []
    for epoch in range(s29.EPOCHS):
        ids = s29.patient_order(s29.pilot_ids(bundle, "train"), epoch)
        loader = lambda pid: s29.load_bag(bundle, pid, epoch, args.budget, True, device)
        t0, loss_sum, skips, steps = time.perf_counter(), 0., 0, 0
        for a, b in s29.group_slices(len(ids)):
            g = s29.train_group(model, optimizer, scaler, ids[a:b], loader, device)
            loss_sum += g["loss_sum"]; skips += g["amp_skips"]; steps += g["optimizer_steps"]
        train_s = time.perf_counter() - t0
        assert s29.tree_equal(frozen_bn, s29.bn_state(model))
        val, pr = s29.validate_model(bundle, model, epoch, device)
        p = np.array([r["p_LUSC"] for r in pr])
        metrics.append({"epoch": epoch + 1, "train_loss": loss_sum / 60, "train_seconds": train_s, "optimizer_steps": steps,
                        "amp_skips": skips, "val_loss": val["loss"], "val_auroc": val["auroc"],
                        "val_balanced_accuracy": val["balanced_accuracy"], "val_p_LUSC_mean": float(p.mean()),
                        "val_p_LUSC_sd": float(p.std()), "val_n_pred_LUSC": int((p >= .5).sum())})
        preds += pr
        m = metrics[-1]
        print(f"B{args.budget} {args.variant} ep{epoch+1}: trainCE={m['train_loss']:.4f} valCE={m['val_loss']:.4f} "
              f"AUROC={m['val_auroc']:.3f} BA={m['val_balanced_accuracy']:.2f} p_mean={m['val_p_LUSC_mean']:.3f} "
              f"p_sd={m['val_p_LUSC_sd']:.4f} predLUSC={m['val_n_pred_LUSC']}/20 skips={skips}", flush=True)
    # final train-patient feature magnitude (no aug, 64 patches), comparable with D05
    from PIL import Image
    mean = torch.tensor([.485, .456, .406]).view(1, 3, 1, 1); std = torch.tensor([.229, .224, .225]).view(1, 3, 1, 1)
    model.eval(); P = []
    with torch.inference_mode():
        for pid in s29.pilot_ids(bundle, "train"):
            x = torch.stack([torch.from_numpy(np.array(Image.open(io.BytesIO(s29.verified_png_bytes(r))).convert("RGB"))).permute(2, 0, 1)
                             for r in bundle["patches"][pid]]).float() / 255.
            x = ((x - mean) / std).to(device)
            with torch.autocast("cuda", dtype=torch.float16):
                h = model.encoder(x)
            h = h.float()
            s = model.attention_w(torch.tanh(model.attention_v(h)) * torch.sigmoid(model.attention_u(h))).squeeze(-1)
            a = torch.softmax(s, 0); P.append((a[:, None] * h).sum(0).cpu().numpy())
    P = np.stack(P)
    summary = {"variant": args.variant, "budget": args.budget, "epochs": metrics,
               "train_pooled_abs_mean_after_ep5": float(np.abs(P).mean()),
               "train_pooled_between_patient_sd_after_ep5": float(P.std(0).mean())}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    with open(run_dir / "validation_predictions.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(preds[0])); w.writeheader(); w.writerows(preds)
    print(json.dumps({k: v for k, v in summary.items() if k != "epochs"}))


if __name__ == "__main__":
    sys.exit(main())
