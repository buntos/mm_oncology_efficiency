from pathlib import Path

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]

CASE_ID = "TCGA-85-8352"

TILE_DIR = (
    ROOT
    / "data"
    / "tiles"
    / "pilot_v2"
    / CASE_ID
)

OUT_FILE = (
    ROOT
    / "results"
    / "figures"
    / f"{CASE_ID}_numbered_tiles.jpg"
)

TILE_SIZE = 256
LABEL_HEIGHT = 28


tiles = sorted(
    TILE_DIR.glob("tile_*.png")
)

if len(tiles) != 64:
    raise RuntimeError(
        f"Expected 64 tiles, found {len(tiles)}"
    )


sheet = Image.new(
    "RGB",
    (
        8 * TILE_SIZE,
        8 * (TILE_SIZE + LABEL_HEIGHT),
    ),
    "white",
)

draw = ImageDraw.Draw(sheet)


for i, path in enumerate(tiles):

    image = Image.open(path).convert("RGB")

    col = i % 8
    row = i // 8

    x = col * TILE_SIZE
    y = row * (TILE_SIZE + LABEL_HEIGHT)

    sheet.paste(
        image,
        (x, y),
    )

    draw.rectangle(
        (
            x,
            y + TILE_SIZE,
            x + TILE_SIZE,
            y + TILE_SIZE + LABEL_HEIGHT,
        ),
        fill="white",
    )

    draw.text(
        (
            x + 5,
            y + TILE_SIZE + 5,
        ),
        f"#{i + 1:02d}",
        fill="black",
    )

    image.close()


sheet.save(
    OUT_FILE,
    quality=95,
)

print(f"Saved: {OUT_FILE}")