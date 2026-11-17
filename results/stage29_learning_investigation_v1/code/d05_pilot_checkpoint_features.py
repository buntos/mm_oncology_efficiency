#!/usr/bin/env python3
r"""Diagnostic D05: did the ACTUAL Stage 29 pilot collapse encoder feature magnitude / between-patient spread?

Loads initial weights and the existing (read-only) Stage 29 checkpoints, and measures on the 60 PILOT-TRAIN
patients only (all 64 patches, no augmentation, eval mode, AMP as Stage 29 validation):
  pooled feature |h| mean, between-patient SD of pooled features, cosine, logit_diff mean/SD, p_LUSC SD.
  .\.venv\Scripts\python.exe -B results\stage29_learning_investigation_v1\code\d05_pilot_checkpoint_features.py
"""
import hashlib, importlib.util, io, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results" / "stage29_learning_investigation_v1"
S29_PATH = ROOT / "scripts" / "29_train_patch_budget_baselines.py"


def main():
    spec = importlib.util.spec_from_file_location("stage29_frozen", S29_PATH)
    s29 = importlib.util.module_from_spec(spec); spec.loader.exec_module(s29)
    bundle = s29.load_bundle(ROOT)
    protocol, _, weights = s29.read_prepared(bundle, None)
    assert protocol["script_sha256"] == hashlib.sha256(S29_PATH.read_bytes()).hexdigest()
    device = s29.initialize_torch(cuda=True); torch, np = s29.torch, s29.np
    from PIL import Image
    ids = s29.pilot_ids(bundle, "train")
    y = np.array([int(bundle["patients"][p]["class_label"] == "LUSC") for p in ids])
    mean = torch.tensor([.485, .456, .406]).view(1, 3, 1, 1); std = torch.tensor([.229, .224, .225]).view(1, 3, 1, 1)
    bags = {}
    for p in ids:
        imgs = [torch.from_numpy(np.array(Image.open(io.BytesIO(s29.verified_png_bytes(r))).convert("RGB"))).permute(2, 0, 1)
                for r in bundle["patches"][p]]
        bags[p] = ((torch.stack(imgs).float() / 255. - mean) / std).half()  # CPU cache, fp16 to fit RAM
    runs = S29_PATH.parents[1] / "results" / "wsi_patch_budget_v1" / "stage29_pilot_v1" / "runs"
    targets = [("init", None)]
    for b in (64, 16):
        for ck in sorted((runs / f"B{b}_seed20261113" / "checkpoints").glob("checkpoint_*")):
            targets.append((f"B{b}/{ck.name}", ck / "state.pt"))
    from sklearn.metrics import roc_auc_score
    report = {}
    model = s29.build_model(weights, device=device)
    init_state = {k: v.clone() for k, v in model.state_dict().items()}
    for name, path in targets:
        if path is None:
            model.load_state_dict(init_state); ep = 0
        else:
            st = torch.load(path, map_location="cpu", weights_only=True)
            model.load_state_dict(st["model"]); ep = st["progress"]["epoch"]; pos = st["progress"]["position"]
            del st
        model.eval(); pooled, ld = [], []
        with torch.inference_mode():
            for p in ids:
                x = bags[p].float().to(device)
                with torch.autocast("cuda", dtype=torch.float16):
                    h = model.encoder(x)
                h = h.float()
                s = model.attention_w(torch.tanh(model.attention_v(h)) * torch.sigmoid(model.attention_u(h))).squeeze(-1)
                a = torch.softmax(s, 0); z = (a[:, None] * h).sum(0, keepdim=True)
                lg = model.classifier(z)[0]
                pooled.append(z[0].cpu().numpy()); ld.append(float(lg[1] - lg[0]))
        P = np.stack(pooled); ld = np.array(ld); pr = 1 / (1 + np.exp(-ld))
        Pn = P / np.linalg.norm(P, axis=1, keepdims=True)
        report[name] = {"epoch_completed": ep, "pooled_abs_mean": float(np.abs(P).mean()),
                        "pooled_between_patient_sd_mean": float(P.std(0).mean()),
                        "pooled_cosine_mean": float((Pn @ Pn.T)[np.triu_indices(len(P), 1)].mean()),
                        "logit_diff_mean": float(ld.mean()), "logit_diff_sd": float(ld.std()),
                        "p_LUSC_sd": float(pr.std()), "train_auroc_no_aug_64": float(roc_auc_score(y, ld))}
        if path is not None:
            report[name]["position"] = pos
        print(name, {k: round(v, 4) for k, v in report[name].items()}, flush=True)
    (OUT / "d05_pilot_checkpoint_features.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    sys.exit(main())
