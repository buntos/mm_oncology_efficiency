#!/usr/bin/env python3
r"""Diagnostic D04: controlled memorisation test, 8 pilot-train patients x 16 fixed patches.

  .\.venv\Scripts\python.exe -B results\stage29_learning_investigation_v1\code\d04_overfit8.py --tag baseline
  (variants: --head-lr X | --fp32 | --encoder-lr X ; ONE factor changed vs baseline per run)

Reuses the frozen Stage 29 module (hash checked) for build_model / optimizer_for / scaler_for / train_group,
so the update path is byte-identical to the pilot. Differences vs pilot, by design of the test:
  * 8 patients (first 4 LUAD + first 4 LUSC train by pilot_order_within_class), not 60;
  * 16 fixed patches = Stage 29 patch_indices(pid, epoch=0, K=16), identical every epoch;
  * no augmentation (D4 code 0 = identity), same ImageNet normalisation;
  * patient order per epoch = Stage 29 patient_order(ids, epoch) (2 groups of 4 -> 2 optimizer steps/epoch).
Eval after every epoch: model.eval(), same 8 patients, same 16 patches, AMP as in Stage 29 validation.
Pre-registered success criterion (written to prereg json before training):
  8/8 correct AND mean eval CE < 0.1 for 3 consecutive epochs, within 50 epochs.
"""
import argparse, hashlib, importlib.util, json, math, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results" / "stage29_learning_investigation_v1"
S29_PATH = ROOT / "scripts" / "29_train_patch_budget_baselines.py"
EPOCHS, K, SEED = 50, 16, 20261113


def load_stage29():
    spec = importlib.util.spec_from_file_location("stage29_frozen", S29_PATH)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--head-lr", type=float, default=None)
    ap.add_argument("--encoder-lr", type=float, default=None)
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--zero-init-head", action="store_true", help="classifier weight=0, bias=0 (initial p=0.5 for all)")
    args = ap.parse_args()
    run_dir = OUT / f"d04_overfit8_{args.tag}"
    run_dir.mkdir(parents=True, exist_ok=False)  # never overwrite

    s29 = load_stage29()
    bundle = s29.load_bundle(ROOT)
    protocol, _, weights = s29.read_prepared(bundle, None)
    assert protocol["script_sha256"] == hashlib.sha256(S29_PATH.read_bytes()).hexdigest()
    device = s29.initialize_torch(cuda=True)
    torch, np = s29.torch, s29.np
    from PIL import Image

    ids = []
    for label in ("LUAD", "LUSC"):
        rows = sorted((r for r in bundle["pilot"].values() if r["split"] == "train" and r["class_label"] == label),
                      key=lambda r: int(r["pilot_order_within_class"]))
        ids += [r["case_submitter_id"] for r in rows[:4]]
    patches = {p: s29.patch_indices(p, 0, K) for p in ids}
    prereg = {"utc": s29.now(), "tag": args.tag, "patients": ids,
              "labels": {p: bundle["patients"][p]["class_label"] for p in ids},
              "patch_tile_indices": {p: [bundle["patches"][p][i]["index"] for i in patches[p]] for p in ids},
              "seed": SEED, "epochs_max": EPOCHS, "augmentation": "none",
              "changes_vs_stage29": {"head_lr": args.head_lr, "encoder_lr": args.encoder_lr, "fp32": args.fp32,
                                     "zero_init_head": args.zero_init_head},
              "diagnostic_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "success_criterion": "8/8 correct and mean eval CE < 0.1 for 3 consecutive epochs (eval mode, same patches)",
              "stage29_script_sha256": protocol["script_sha256"]}
    (run_dir / "prereg.json").write_text(json.dumps(prereg, indent=2))

    mean = torch.tensor([.485, .456, .406]).view(1, 3, 1, 1); std = torch.tensor([.229, .224, .225]).view(1, 3, 1, 1)
    cache = {}
    for p in ids:  # verified bytes, no augmentation, fixed patches (cached on CPU)
        imgs = []
        for i in patches[p]:
            raw = s29.verified_png_bytes(bundle["patches"][p][i])
            imgs.append(torch.from_numpy(np.array(Image.open(__import__("io").BytesIO(raw)).convert("RGB"))).permute(2, 0, 1))
        x = (torch.stack(imgs).float().div_(255.) - mean) / std
        cache[p] = (x, torch.tensor([0 if bundle["patients"][p]["class_label"] == "LUAD" else 1]))
    loader = lambda p: (cache[p][0].to(device), cache[p][1].to(device))

    model = s29.build_model(weights, device=device)
    if args.zero_init_head:
        with torch.no_grad():
            model.classifier.weight.zero_(); model.classifier.bias.zero_()
    optimizer = s29.optimizer_for(model)
    if args.encoder_lr is not None:
        optimizer.param_groups[0]["lr"] = args.encoder_lr
    if args.head_lr is not None:
        optimizer.param_groups[1]["lr"] = args.head_lr
    scaler = s29.scaler_for(device)
    if args.fp32:
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        real_autocast = torch.autocast
        class NoAutocast:  # disable autocast inside the frozen train_group/forward only for this process
            def __init__(self, *a, **k): self.cm = real_autocast(device_type="cuda", enabled=False)
            def __enter__(self): return self.cm.__enter__()
            def __exit__(self, *e): return self.cm.__exit__(*e)
        torch.autocast = NoAutocast
    enc_params = list(model.encoder.parameters()); head_params = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    step_log, prev = [], {}

    def pre_hook(opt, a, k):
        g_enc = torch.sqrt(sum((p.grad.float() ** 2).sum() for p in enc_params if p.grad is not None))
        g_head = torch.sqrt(sum((p.grad.float() ** 2).sum() for p in head_params if p.grad is not None))
        g_cls_w = model.classifier.weight.grad
        prev["enc"] = [p.detach().clone() for p in enc_params]; prev["head"] = [p.detach().clone() for p in head_params]
        prev["W"] = model.classifier.weight.detach().clone(); prev["b"] = model.classifier.bias.detach().clone()
        prev["g"] = {"grad_norm_encoder": float(g_enc), "grad_norm_head": float(g_head),
                     "grad_cls_w_same_sign_frac": float(((g_cls_w[1] * g_cls_w[1].sum().sign()) > 0).float().mean())}

    def post_hook(opt, a, k):
        d_enc = math.sqrt(sum(float(((p - q) ** 2).sum()) for p, q in zip(enc_params, prev["enc"])))
        n_enc = math.sqrt(sum(float((q ** 2).sum()) for q in prev["enc"]))
        d_head = math.sqrt(sum(float(((p - q) ** 2).sum()) for p, q in zip(head_params, prev["head"])))
        dW = (model.classifier.weight - prev["W"]); ddiff = dW[1] - dW[0]
        step_log.append({**prev["g"], "update_norm_encoder": d_enc, "rel_update_encoder": d_enc / n_enc,
                         "update_norm_head": d_head,
                         "cls_dWdiff_same_sign_frac": float(((ddiff * ddiff.sum().sign()) > 0).float().mean()),
                         "cls_dWdiff_abs_mean": float(ddiff.abs().mean()),
                         "cls_dbias_diff": float((model.classifier.bias - prev["b"])[1] - (model.classifier.bias - prev["b"])[0]),
                         "lr": [g["lr"] for g in opt.param_groups]})

    optimizer.register_step_pre_hook(pre_hook); optimizer.register_step_post_hook(post_hook)

    def evaluate(epoch):
        model.eval(); out = []
        with torch.inference_mode():
            for p in ids:
                x, y = loader(p)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=not args.fp32):
                    h = model.encoder(x)
                h = h.float()
                s = model.attention_w(torch.tanh(model.attention_v(h)) * torch.sigmoid(model.attention_u(h))).squeeze(-1)
                a = torch.softmax(s, 0); pooled = (a[:, None] * h).sum(0, keepdim=True)
                logits = model.classifier(pooled)
                ce = float(torch.nn.functional.cross_entropy(logits, y))
                out.append({"epoch": epoch, "patient": p, "label": int(y), "p_LUSC": float(torch.softmax(logits, 1)[0, 1]),
                            "ce": ce, "logit_diff": float(logits[0, 1] - logits[0, 0]),
                            "att_norm_entropy": float(-(a * a.clamp_min(1e-12).log()).sum() / math.log(len(a))),
                            "att_max": float(a.max()), "pooled": pooled[0].cpu()})
        P = torch.stack([o.pop("pooled") for o in out])
        feat = {"pooled_between_patient_sd_mean": float(P.std(0).mean()), "pooled_mean_abs": float(P.abs().mean()),
                "pooled_cosine_mean": float((torch.nn.functional.normalize(P, dim=1) @ torch.nn.functional.normalize(P, dim=1).T)[
                    torch.triu_indices(8, 8, 1).unbind()].mean())}
        model.train(); return out, feat

    per_patient, epochs, streak, success_epoch = [], [], 0, None
    rows0, f0 = evaluate(0); per_patient += rows0
    epochs.append({"epoch": 0, "eval_ce": sum(r["ce"] for r in rows0) / 8, "eval_correct": sum((r["p_LUSC"] >= .5) == r["label"] for r in rows0), **f0})
    t0 = time.perf_counter()
    for epoch in range(EPOCHS):
        order = s29.patient_order(ids, epoch)
        n_before = len(step_log); tl, skips = 0., 0
        for a, b in s29.group_slices(len(order)):
            g = s29.train_group(model, optimizer, scaler, order[a:b], loader, device)
            tl += g["loss_sum"]; skips += g["amp_skips"]
        rows, f = evaluate(epoch + 1); per_patient += rows
        ce = sum(r["ce"] for r in rows) / 8; corr = sum((r["p_LUSC"] >= .5) == r["label"] for r in rows)
        streak = streak + 1 if (corr == 8 and ce < .1) else 0
        if streak == 3 and success_epoch is None:
            success_epoch = epoch + 1
        st = step_log[n_before:]
        epochs.append({"epoch": epoch + 1, "train_ce": tl / 8, "eval_ce": ce, "eval_correct": corr, "amp_skips": skips,
                       "p_LUSC_sd": float(np.std([r["p_LUSC"] for r in rows])),
                       "logit_diff_sd": float(np.std([r["logit_diff"] for r in rows])),
                       "logit_diff_mean": float(np.mean([r["logit_diff"] for r in rows])),
                       "att_entropy_mean": float(np.mean([r["att_norm_entropy"] for r in rows])),
                       "grad_norm_encoder": float(np.mean([s["grad_norm_encoder"] for s in st])),
                       "grad_norm_head": float(np.mean([s["grad_norm_head"] for s in st])),
                       "rel_update_encoder": float(np.mean([s["rel_update_encoder"] for s in st])),
                       "update_norm_head": float(np.mean([s["update_norm_head"] for s in st])),
                       "cls_dWdiff_same_sign_frac": float(np.mean([s["cls_dWdiff_same_sign_frac"] for s in st])),
                       "lr": st[-1]["lr"], **f})
        e = epochs[-1]
        print(f"ep {epoch+1:2d} trainCE={e['train_ce']:.3f} evalCE={ce:.3f} correct={corr}/8 meanLD={e['logit_diff_mean']:+.2f} "
              f"sdLD={e['logit_diff_sd']:.3f} gEnc={e['grad_norm_encoder']:.3f} sameSign={e['cls_dWdiff_same_sign_frac']:.2f} skips={skips}", flush=True)
        if success_epoch is not None and epoch + 1 >= success_epoch + 2:  # 2 extra epochs to show stability, then stop
            break
    summary = {"tag": args.tag, "success": success_epoch is not None, "success_epoch": success_epoch,
               "epochs_run": len(epochs) - 1, "gpu_train_seconds": time.perf_counter() - t0,
               "final": epochs[-1], "initial": epochs[0]}
    (run_dir / "epochs.json").write_text(json.dumps(epochs, indent=1))
    (run_dir / "steps.json").write_text(json.dumps(step_log, indent=1))
    import csv
    with open(run_dir / "per_patient_eval.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(per_patient[0])); w.writeheader(); w.writerows(per_patient)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k not in ("final", "initial")}))


if __name__ == "__main__":
    sys.exit(main())
