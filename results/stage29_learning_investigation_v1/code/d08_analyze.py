#!/usr/bin/env python3
r"""D08 analysis (CPU): D07 reproduction check, curves, last-5-epoch comparison, shift/group association.
  .\.venv\Scripts\python.exe -B results\stage29_learning_investigation_v1\code\d08_analyze.py
Writes d08_analysis.json and d08_curves.png into results\stage29_learning_investigation_v1\.
"""
import csv, json
from pathlib import Path
import numpy as np

OUT = Path(__file__).resolve().parents[1]


def load(c):
    d = OUT / f"d08_head_lr_{c}_B64_20ep"
    rows = json.loads((d / "epochs.json").read_text())
    groups = list(csv.DictReader(open(d / "groups.csv")))
    probes = list(csv.DictReader(open(d / "probe_epochs4_5.csv")))
    return rows, groups, probes, json.loads((d / "summary.json").read_text())


def main():
    res = {}
    data = {c: load(c) for c in "AB"}
    # 1. reproduction of D07 epochs 1-5
    keys = ["val_ce", "val_auroc", "val_p_mean", "val_p_sd", "train_eval_ce", "train_eval_auroc",
            "train_eval_feat_abs_mean", "enc_update_mean", "head_update_mean"]
    for c in "AB":
        d07 = json.loads((OUT / f"d07_head_lr_{c}_B64" / "epochs.json").read_text())
        d08 = data[c][0]
        res[f"{c}_reproduces_D07_max_abs_diff"] = {k: max(abs(d07[e][k] - d08[e][k]) for e in range(0, 6) if k in d07[e]) for k in keys}
    # 2. last five epochs
    last = {}
    for c in "AB":
        rows = data[c][0][16:21]
        last[c] = {k: {"mean": float(np.mean([r[k] for r in rows])), "sd": float(np.std([r[k] for r in rows])),
                       "values": [round(r[k], 4) for r in rows]}
                   for k in ["train_eval_ce", "train_eval_auroc", "train_eval_balanced_accuracy", "val_ce", "val_auroc",
                             "val_balanced_accuracy", "val_macro_f1", "val_p_mean", "val_p_sd", "val_logit_diff_mean",
                             "train_eval_logit_diff_mean", "val_mean_p_diff_LUSC_minus_LUAD", "train_eval_mean_p_diff_LUSC_minus_LUAD",
                             "train_eval_feat_abs_mean", "train_eval_feat_between_patient_sd", "val_feat_abs_mean",
                             "val_feat_between_patient_sd", "val_n_pred_LUSC"]}
    res["last5_epochs_16_20"] = last
    # 3. epoch-to-epoch volatility of mean logit diff (train-eval / val), epochs 1-20
    for c in "AB":
        rows = data[c][0]
        for pfx in ("train_eval", "val"):
            ld = np.array([r[f"{pfx}_logit_diff_mean"] for r in rows])
            res[f"{c}_{pfx}_mean_logit_diff_abs_epoch_change"] = {"ep1_5": float(np.abs(np.diff(ld[0:6])).mean()),
                                                                  "ep6_20": float(np.abs(np.diff(ld[5:21])).mean()),
                                                                  "ep16_20": float(np.abs(np.diff(ld[15:21])).mean())}
    # 4. probe shift vs group composition (association only)
    for c in "AB":
        pr = data[c][2]
        shifts, nl, bias_shift = [], [], []
        for prev, cur in zip(pr, pr[1:]):
            if cur["epoch"] != prev["epoch"] or cur["after_group"] == "0":
                continue
            shifts.append(float(cur["probe_logit_diff_mean"]) - float(prev["probe_logit_diff_mean"]))
            nl.append(int(cur["n_LUSC_in_group"]))
            bias_shift.append(float(cur["bias_diff_LUSC_minus_LUAD"]) - float(prev["bias_diff_LUSC_minus_LUAD"]))
        shifts, nl, bias_shift = map(np.array, (shifts, nl, bias_shift))
        by = {int(k): {"n": int((nl == k).sum()), "mean_shift": float(shifts[nl == k].mean())} for k in sorted(set(nl))}
        res[f"{c}_probe_shift_vs_group_nLUSC"] = {
            "n_groups": int(len(shifts)), "spearman_like_corr": float(np.corrcoef(nl, shifts)[0, 1]) if len(set(nl)) > 1 else None,
            "by_n_LUSC": by, "mean_abs_shift": float(np.abs(shifts).mean()),
            "mean_abs_bias_diff_shift": float(np.abs(bias_shift).mean()),
            "probe_gap_ep4_start_end": [float(pr[0]["probe_logit_diff_class_gap"]), float(pr[-1]["probe_logit_diff_class_gap"])]}
        g = data[c][1]
        res[f"{c}_group_composition_counts"] = {k: sum(1 for r in g if r["n_LUSC"] == str(k)) for k in range(5)}
    res["cost"] = {c: {"wall_after_setup_s": data[c][3]["wall_seconds_after_setup"],
                       "recomputed_epochs_1_5_s": data[c][3]["recomputed_epochs_1_5_wall_seconds"]} for c in "AB"}
    (OUT / "d08_analysis.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=1))

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    panels = [("CE", "train_eval_ce", "val_ce"), ("AUROC", "train_eval_auroc", "val_auroc"),
              ("Balanced accuracy", "train_eval_balanced_accuracy", "val_balanced_accuracy"),
              ("mean p_LUSC", "train_eval_p_mean", "val_p_mean"), ("SD p_LUSC", "train_eval_p_sd", "val_p_sd"),
              ("n predicted LUSC (train /60, val /20)", "train_eval_n_pred_LUSC", "val_n_pred_LUSC"),
              ("mean p diff LUSC-LUAD", "train_eval_mean_p_diff_LUSC_minus_LUAD", "val_mean_p_diff_LUSC_minus_LUAD"),
              ("pooled |h|", "train_eval_feat_abs_mean", "val_feat_abs_mean"),
              ("between-patient SD h", "train_eval_feat_between_patient_sd", "val_feat_between_patient_sd")]
    fig, axes = plt.subplots(3, 3, figsize=(15, 11))
    col = {"A": "#1f77b4", "B": "#d62728"}
    for ax, (title, tk, vk) in zip(axes.flat, panels):
        for c in "AB":
            rows = data[c][0]
            e = [r["epoch"] for r in rows if tk in r]
            ax.plot(e, [r[tk] for r in rows if tk in r], "-", color=col[c], label=f"{c} train-eval")
            ev = [r["epoch"] for r in rows if vk in r]
            ax.plot(ev, [r[vk] for r in rows if vk in r], "--", color=col[c], label=f"{c} validation")
        ax.set_title(title); ax.set_xlabel("epoch"); ax.grid(alpha=.3)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("D08: A (head LR 1e-3) vs B (head LR 1e-4), encoder LR 1e-5, K=64, seed 20261113; validation n=20 (descriptive)")
    fig.tight_layout(); fig.savefig(OUT / "d08_curves.png", dpi=110)


if __name__ == "__main__":
    main()
