#!/usr/bin/env python3
r"""D08 (CPU): association of end-of-epoch train-eval mean logit diff (LUSC-LUAD) with label composition of the
epoch's final accumulation groups, and A-vs-B synchrony. Association only (one seed, shared data schedule).
  .\.venv\Scripts\python.exe -B results\stage29_learning_investigation_v1\code\d08_epoch_end_composition.py
(identical logic to the inline run whose output is d08_epoch_end_composition.txt; prints to stdout)
"""
import csv, json
from pathlib import Path
import numpy as np

OUT = Path(__file__).resolve().parents[1]
print("Association only (one seed, shared schedule): end-of-epoch train-eval mean logit diff (LUSC-LUAD) vs label composition of the epoch's final groups")
for c in "AB":
    g = list(csv.DictReader(open(OUT / f"d08_head_lr_{c}_B64_20ep/groups.csv")))
    e = json.load(open(OUT / f"d08_head_lr_{c}_B64_20ep/epochs.json"))
    ld = np.array([r["train_eval_logit_diff_mean"] for r in e[1:]])
    for last in (1, 2, 3):
        x = np.array([sum(int(r["n_LUSC"]) - int(r["n_LUAD"]) for r in [q for q in g if int(q["epoch"]) == ep][-last:]) for ep in range(1, 21)])
        print(f"{c}: last {last} group(s): Pearson r={np.corrcoef(x, ld)[0, 1]:+.2f}  (n=20 epochs)")
    print(f"  {c} train-eval mean LD:", [round(float(v), 2) for v in ld])
a = np.array([r["train_eval_logit_diff_mean"] for r in json.load(open(OUT / "d08_head_lr_A_B64_20ep/epochs.json"))[1:]])
b = np.array([r["train_eval_logit_diff_mean"] for r in json.load(open(OUT / "d08_head_lr_B_B64_20ep/epochs.json"))[1:]])
print(f"A vs B end-of-epoch train-eval mean LD: Pearson r={np.corrcoef(a, b)[0, 1]:+.2f}; sign agreement {int((np.sign(a) == np.sign(b)).sum())}/20")
