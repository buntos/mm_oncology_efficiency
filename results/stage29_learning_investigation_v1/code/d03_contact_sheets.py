#!/usr/bin/env python3
r"""Diagnostic D03 (CPU): contact sheets + simple pixel statistics for 4 LUAD + 4 LUSC PILOT-TRAIN patients.

Patients: first 4 per class by pilot_order_within_class (train split only) -- the same 8 used in D04.
Patches: the 16 Stage 29 B16 epoch-0 indices per patient (as used by D04) -> sheet of 16 per patient.
  .\.venv\Scripts\python.exe -B results\stage29_learning_investigation_v1\code\d03_contact_sheets.py
"""
import csv, json, importlib.util
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results" / "stage29_learning_investigation_v1" / "d03_contact_sheets"


def stage29():
    spec = importlib.util.spec_from_file_location("s29", ROOT / "scripts" / "29_train_patch_budget_baselines.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def overfit_ids(bundle):
    out = []
    for label in ("LUAD", "LUSC"):
        rows = sorted((r for r in bundle["pilot"].values() if r["split"] == "train" and r["class_label"] == label),
                      key=lambda r: int(r["pilot_order_within_class"]))
        out += [r["case_submitter_id"] for r in rows[:4]]
    return out


def main():
    s29 = stage29(); bundle = s29.load_bundle(ROOT)
    OUT.mkdir(parents=True, exist_ok=False)
    stats = []
    for pid in overfit_ids(bundle):
        label = bundle["patients"][pid]["class_label"]
        idx = s29.patch_indices(pid, 0, 16)
        sheet = Image.new("RGB", (4 * 256, 4 * 256 + 24), "white")
        d = ImageDraw.Draw(sheet)
        for j, i in enumerate(idx):
            row = bundle["patches"][pid][i]
            raw = s29.verified_png_bytes(row)
            im = Image.open(row["path"]).convert("RGB"); a = np.asarray(im).astype(float)
            sheet.paste(im, ((j % 4) * 256, 24 + (j // 4) * 256))
            d.text(((j % 4) * 256 + 4, 26 + (j // 4) * 256), f"#{row['index']} x{row['x_level0']} y{row['y_level0']}", fill="yellow")
            gray = a.mean(-1)
            stats.append({"patient": pid, "label": label, "tile_index": row["index"],
                          "x": int(row["x_level0"]), "y": int(row["y_level0"]),
                          "mean_rgb": [round(v, 1) for v in a.reshape(-1, 3).mean(0)],
                          "white_frac_gt220": round(float((gray > 220).mean()), 3),
                          "dark_frac_lt50": round(float((gray < 50).mean()), 3),
                          "std_gray": round(float(gray.std()), 1)})
        d.text((4, 4), f"{pid}  {label}  (pilot train; B16 epoch-0 indices)", fill="black")
        sheet.save(OUT / f"{label}_{pid}.png")
    with open(OUT / "patch_stats.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(stats[0])); w.writeheader(); w.writerows(stats)
    summ = {}
    for pid in {s["patient"] for s in stats}:
        ss = [s for s in stats if s["patient"] == pid]
        summ[pid] = {"label": ss[0]["label"], "mean_white_frac": round(np.mean([s["white_frac_gt220"] for s in ss]), 3),
                     "mean_rgb": [round(float(v), 1) for v in np.mean([s["mean_rgb"] for s in ss], 0)],
                     "unique_coords": len({(s["x"], s["y"]) for s in ss})}
    (OUT / "summary.json").write_text(json.dumps(summ, indent=2))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
