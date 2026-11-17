#!/usr/bin/env python3
r"""Diagnostic D02: input-path check + frozen ImageNet feature probe on PILOT TRAIN only.

Run from project root:
  .\.venv\Scripts\python.exe -B results\stage29_learning_investigation_v1\code\d02_frozen_feature_probe.py

Questions:
  (a) Does Stage 29 load_bag produce exactly the pixels an independent PIL read gives (incl. D4 code)?
  (b) What do initial (untrained) features / logits / attention look like?
  (c) Do frozen ImageNet ResNet18 features carry patient-level LUAD/LUSC signal on the 60 pilot-train
      patients (leave-one-out, train only; validation and test never opened)?
  (d) How large is the common (patient-shared) feature component vs patient-specific spread?
      Pre-declared probe: StandardScaler + L2 LogisticRegression(C=1.0), LOO over 60 patients.
Imports the frozen Stage 29 module read-only (its hash is checked against the pilot protocol).
"""
import hashlib, importlib.util, io, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results" / "stage29_learning_investigation_v1"
S29_PATH = ROOT / "scripts" / "29_train_patch_budget_baselines.py"


def load_stage29():
    spec = importlib.util.spec_from_file_location("stage29_frozen", S29_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    s29 = load_stage29()
    bundle = s29.load_bundle(ROOT)
    protocol, _, weights = s29.read_prepared(bundle, None)
    assert protocol["script_sha256"] == hashlib.sha256(S29_PATH.read_bytes()).hexdigest()
    device = s29.initialize_torch(cuda=True)
    torch, np = s29.torch, s29.np
    from PIL import Image
    train_ids = s29.pilot_ids(bundle, "train")
    labels = np.array([0 if bundle["patients"][p]["class_label"] == "LUAD" else 1 for p in train_ids])
    report = {"n_train": len(train_ids), "n_LUSC": int(labels.sum())}

    # (a) independent pixel path check on first 2 train patients, epoch 0, K=16
    mean = np.array([.485, .456, .406]); std = np.array([.229, .224, .225])
    max_err = 0.
    for pid in train_ids[:2]:
        x, y = s29.load_bag(bundle, pid, 0, 16, True, torch.device("cpu"))
        idx = s29.patch_indices(pid, 0, 16)
        for j, i in enumerate(idx):
            row = bundle["patches"][pid][i]
            a = np.asarray(Image.open(row["path"]).convert("RGB")).astype(np.float64) / 255.
            code = s29.augmentation_code(pid, 0, row["index"])
            a = np.rot90(a, code % 4)
            if code // 4:
                a = a[:, ::-1]
            ref = ((a - mean) / std).transpose(2, 0, 1)
            max_err = max(max_err, float(np.abs(x[j].double().numpy() - ref).max()))
        assert int(y) == int(bundle["patients"][pid]["class_label"] == "LUSC")
    report["load_bag_vs_independent_max_abs_err"] = max_err
    report["input_example_stats"] = {"dtype": str(x.dtype), "shape": list(x.shape),
                                     "min": float(x.min()), "max": float(x.max()), "mean": float(x.mean())}

    # (b) initial model, eval mode (frozen BN), no aug, all 64 patches, same AMP as Stage 29 validation
    model = s29.build_model(weights, device=device)
    model.eval()
    feats, pooled_att, init_p, att_entropy, att_max = [], [], [], [], []
    patch_feats = []
    with torch.inference_mode():
        for pid in train_ids:
            rows = bundle["patches"][pid]
            imgs = [torch.from_numpy(np.array(Image.open(r["path"]).convert("RGB"))).permute(2, 0, 1) for r in rows]
            x = torch.stack(imgs).float().div_(255.)
            x = ((x - torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)) / torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1)).to(device)
            with torch.autocast("cuda", dtype=torch.float16):
                h = model.encoder(x).float()
                logits = model(x)
            s = model.attention_w(torch.tanh(model.attention_v(h)) * torch.sigmoid(model.attention_u(h))).squeeze(-1)
            a = torch.softmax(s, 0)
            att_entropy.append(float(-(a * a.log()).sum() / np.log(len(a))))
            att_max.append(float(a.max()))
            feats.append(h.mean(0).cpu().numpy()); patch_feats.append(h.cpu().numpy())
            pooled_att.append((a[:, None] * h).sum(0).cpu().numpy())
            init_p.append(float(torch.softmax(logits.float(), 1)[0, 1]))
    F = np.stack(feats); P = np.stack(patch_feats)  # [60,512], [60,64,512]
    report["features"] = {
        "nonneg_fraction": float((P >= 0).mean()),
        "patch_L1_mean": float(np.abs(P).sum(-1).mean()),
        "patient_mean_L1_mean": float(np.abs(F).sum(-1).mean()),
        "per_dim_mean_over_patients_mean": float(F.mean()),
        # variance decomposition per dim, averaged over dims
        "between_patient_var": float(F.var(0).mean()),
        "within_patient_var": float(P.var(1).mean()),
        "common_mode_energy_ratio": float((F.mean(0) ** 2).sum() / (F ** 2).sum(-1).mean()),
        "cosine_between_patient_means_mean": float(((F / np.linalg.norm(F, axis=1, keepdims=True)) @
                                                     (F / np.linalg.norm(F, axis=1, keepdims=True)).T)[np.triu_indices(60, 1)].mean()),
    }
    report["initial_model"] = {"p_LUSC_mean": float(np.mean(init_p)), "p_LUSC_sd": float(np.std(init_p)),
                               "attention_norm_entropy_mean": float(np.mean(att_entropy)),
                               "attention_max_mean": float(np.mean(att_max)),
                               "initial_loss_mean": float(np.mean([-np.log(p if y else 1 - p) for p, y in zip(init_p, labels)]))}
    # Adam common-mode estimate: one Adam step (|dW_cj| ~ lr) aligned with the shared non-negative feature
    # moves logit difference by ~ 2*lr*||h||_1, compared with between-patient spread of a random projection.
    W = model.classifier.weight.detach().cpu().numpy()
    proj = F @ (W[1] - W[0])
    report["adam_common_mode_estimate"] = {
        "head_lr": .001, "approx_logit_diff_shift_per_aligned_step": float(2 * .001 * np.abs(F).sum(-1).mean()),
        "initial_logit_diff_between_patient_sd": float(proj.std()),
    }

    # (c) LOO probe on frozen features (train only)
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.metrics import roc_auc_score
    def loo(X):
        sc = np.zeros(len(X))
        for i in range(len(X)):
            m = np.ones(len(X), bool); m[i] = False
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000))
            clf.fit(X[m], labels[m]); sc[i] = clf.predict_proba(X[i:i + 1])[0, 1]
        return {"auroc": float(roc_auc_score(labels, sc)), "accuracy": float(((sc >= .5) == labels).mean())}
    report["loo_probe_mean_pool"] = loo(F)
    report["loo_probe_init_attention_pool"] = loo(np.stack(pooled_att))
    # Label-permutation null (fixed seed) for the mean-pool probe
    rng = np.random.default_rng(20261113); null = []
    true_labels = labels.copy()
    for _ in range(20):
        labels = rng.permutation(true_labels); null.append(loo(F)["auroc"])
    labels = true_labels
    report["loo_probe_permuted_label_auroc"] = {"mean": float(np.mean(null)), "max": float(np.max(null)), "n": 20}
    np.savez_compressed(OUT / "d02_frozen_features_pilot_train.npz", ids=np.array(train_ids), labels=labels, F=F,
                        init_p=np.array(init_p))
    (OUT / "d02_frozen_feature_probe.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    sys.exit(main())
