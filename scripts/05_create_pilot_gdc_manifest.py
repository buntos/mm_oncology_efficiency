from pathlib import Path
import requests
import pandas as pd


BASE_URL = "https://api.gdc.cancer.gov"

ROOT = Path(__file__).resolve().parents[1]

MANIFEST_DIR = ROOT / "data" / "manifests"

PILOT_FILES = MANIFEST_DIR / "pilot_files_stage4.csv"
OUT_FILE = MANIFEST_DIR / "gdc_manifest_pilot_stage4.txt"


pilot = pd.read_csv(PILOT_FILES)

file_ids = (
    pilot["file_id"]
    .dropna()
    .astype(str)
    .tolist()
)

if len(file_ids) != 20:
    raise RuntimeError(
        f"Expected 20 files, got {len(file_ids)}"
    )

if len(set(file_ids)) != len(file_ids):
    raise RuntimeError(
        "Duplicate UUIDs found in pilot files"
    )


payload = {
    "ids": file_ids,
}


print(
    f"Requesting GDC manifest for {len(file_ids)} files..."
)

response = requests.post(
    f"{BASE_URL}/manifest",
    json=payload,
    timeout=180,
)

response.raise_for_status()


OUT_FILE.write_bytes(
    response.content
)


print(
    f"Manifest saved to:"
)

print(
    f"  {OUT_FILE}"
)


text = OUT_FILE.read_text(
    encoding="utf-8"
)

lines = [
    line
    for line in text.splitlines()
    if line.strip()
]

print(
    f"Manifest lines: {len(lines)}"
)

print()
print("First lines:")
print()

for line in lines[:5]:
    print(line)

print()

if len(lines) != 21:
    print(
        "WARNING: expected 1 header + 20 file rows"
    )
else:
    print(
        "MANIFEST STATUS: PASS"
    )