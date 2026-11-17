#!/usr/bin/env python3
r"""Diagnostic D01 (read-only): independent audit of the completed Stage 29 pilot.

Run from the project root:
  .\.venv\Scripts\python.exe results\stage29_learning_investigation_v1\code\d01_static_run_audit.py

Reads only Stage 28/29 files; never opens test PNGs; writes only into
results\stage29_learning_investigation_v1\d01_*.  CPU only.
"""
import csv, hashlib, json, math, sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
S28 = ROOT / "results" / "wsi_patch_budget_v1"
S29 = S28 / "stage29_pilot_v1"
OUT = ROOT / "results" / "stage29_learning_investigation_v1"


def rows(p):
    with open(p, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def auroc_pairs(labels, scores):
    """Independent O(n^2) Mann-Whitney AUROC (different algorithm than Stage 29)."""
    pos = [s for y, s in zip(labels, scores) if y == 1]
    neg = [s for y, s in zip(labels, scores) if y == 0]
    return sum((p > n) + .5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


def main():
    report = {}
    protocol = json.loads((S29 / "pilot_protocol.json").read_text(encoding="utf-8"))
    report["script_sha_matches_protocol"] = sha(ROOT / "scripts" / "29_train_patch_budget_baselines.py") == protocol["script_sha256"]
    splits = {r["case_submitter_id"]: r for r in rows(S28 / "patient_splits.csv")}
    pilot = {r["case_submitter_id"]: r for r in rows(S28 / "pilot_manifest.csv")}
    report["split_counts"] = Counter(f'{r["split"]}/{r["class_label"]}' for r in splits.values())
    report["pilot_counts"] = Counter(f'{r["split"]}/{r["class_label"]}' for r in pilot.values())
    report["pilot_label_split_consistent"] = all(
        splits[p]["class_label"] == r["class_label"] and splits[p]["split"] == r["split"] for p, r in pilot.items())
    tr = set(protocol["train_patients"]); va = set(protocol["validation_patients"])
    report["protocol_train_eq_manifest"] = tr == {p for p, r in pilot.items() if r["split"] == "train"}
    report["protocol_val_eq_manifest"] = va == {p for p, r in pilot.items() if r["split"] == "validation"}
    report["train_val_overlap"] = sorted(tr & va)
    report["any_pilot_test"] = any(r["split"] == "test" for r in pilot.values())
    # TSS (tissue source site) per class in pilot: site/class confounding is a data hypothesis.
    report["pilot_tss_by_class"] = {s: {c: dict(Counter(splits[p]["tss"] for p in pilot if pilot[p]["split"] == s and pilot[p]["class_label"] == c))
                                            for c in ("LUAD", "LUSC")} for s in ("train", "validation")}
    inv = rows(S28 / "patch_inventory.csv")
    per = Counter(r["case_submitter_id"] for r in inv)
    report["inventory_patches_per_pilot_patient"] = sorted(set(per[p] for p in pilot))
    report["inventory_pixel_dupes_across_patients"] = sum(
        1 for h, c in Counter((r["pixel_sha256"]) for r in inv).items() if c > 1)

    per_budget = {}
    for b in (64, 32, 16):
        run = S29 / "runs" / f"B{b}_seed20261113"
        preds = rows(run / "validation_predictions.csv")
        metrics = rows(run / "epoch_metrics.csv")
        epochs = {}
        for e in sorted({int(r["epoch"]) for r in preds}):
            ep = [r for r in preds if int(r["epoch"]) == e]
            ids = [r["case_submitter_id"] for r in ep]
            labels = [int(r["label"]) for r in ep]
            scores = [float(r["p_LUSC"]) for r in ep]
            label_ok = all((pilot[i]["class_label"] == "LUSC") == bool(y) for i, y in zip(ids, labels))
            m = next(x for x in metrics if int(x["epoch"]) == e)
            nll = sum(-math.log(s if y else 1 - s) for y, s in zip(labels, scores)) / len(ep)
            mean = sum(scores) / len(scores)
            sd = (sum((s - mean) ** 2 for s in scores) / len(scores)) ** .5
            mean_luad = sum(s for y, s in zip(labels, scores) if y == 0) / labels.count(0)
            mean_lusc = sum(s for y, s in zip(labels, scores) if y == 1) / labels.count(1)
            pred = [int(s >= .5) for s in scores]
            epochs[e] = {
                "n": len(ep), "ids_eq_validation": set(ids) == va, "labels_match_manifest": label_ok,
                "auroc_recomputed": auroc_pairs(labels, scores), "auroc_reported": float(m["validation_auroc"]),
                "nll_from_probs": nll, "val_loss_reported": float(m["validation_loss"]),
                "p_LUSC_mean": mean, "p_LUSC_sd": sd, "p_LUSC_min": min(scores), "p_LUSC_max": max(scores),
                "mean_p_LUSC_true_LUAD": mean_luad, "mean_p_LUSC_true_LUSC": mean_lusc,
                "n_pred_LUSC": sum(pred),
                "train_patches": int(m["train_patches"]), "optimizer_steps": int(m["optimizer_steps"]),
                "amp_skips": int(m["amp_skips"]),
            }
        per_budget[f"B{b}"] = epochs
    report["validation_audit"] = per_budget

    # Checkpoint parameter drift (CPU torch); compare final model to official weights + fresh head.
    try:
        import torch
        w0 = torch.load(S29 / "weights" / "resnet18-f37072fd.pth", map_location="cpu", weights_only=True)
        drift = {}
        for b in (64, 32, 16):
            ck = sorted((S29 / "runs" / f"B{b}_seed20261113" / "checkpoints").glob("checkpoint_*"))
            final = torch.load(ck[-1] / "state.pt", map_location="cpu", weights_only=True)
            sd = final["model"]
            enc = {k[len("encoder."):]: v for k, v in sd.items() if k.startswith("encoder.")}
            rel = []
            for k, v in w0.items():
                if k.startswith("fc."):
                    continue
                if v.dtype.is_floating_point and "running" not in k:
                    rel.append((k, float((enc[k] - v).norm() / (v.norm() + 1e-12))))
            bn_same = all(torch.equal(enc[k], w0[k]) for k in w0 if "running" in k)
            W = sd["classifier.weight"]; bias = sd["classifier.bias"]
            dW = W[1] - W[0]
            drift[f"B{b}"] = {
                "checkpoint": ck[-1].name, "epoch": final["progress"]["epoch"],
                "encoder_rel_change_median": sorted(r for _, r in rel)[len(rel) // 2],
                "encoder_rel_change_max": max(rel, key=lambda t: t[1]),
                "conv1_rel_change": dict(rel)["conv1.weight"],
                "bn_running_stats_equal_imagenet": bn_same,
                "classifier_bias": bias.tolist(),
                "classifier_dW_mean": float(dW.mean()), "classifier_dW_abs_mean": float(dW.abs().mean()),
                "classifier_dW_frac_positive": float((dW > 0).float().mean()),
                "lr_groups": [g["lr"] for g in final["optimizer"]["param_groups"]],
                "adam_step": [float(s["step"]) for s in list(final["optimizer"]["state"].values())[:1]],
            }
        report["checkpoint_drift"] = drift
    except Exception as e:  # pragma: no cover
        report["checkpoint_drift_error"] = repr(e)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "d01_static_run_audit.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, indent=1, default=str))


if __name__ == "__main__":
    sys.exit(main())
