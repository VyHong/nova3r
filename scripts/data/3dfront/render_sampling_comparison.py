#!/usr/bin/env python3
"""Compare Cycles sampling limits with one fixed 3D-FRONT lighting setup."""
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "artifacts/3dfront_gpu_samples/04_SecondBedroom-6353/metadata.json"
OUTPUT = ROOT / "artifacts/3dfront_sampling_comparison"
SAMPLES = [16, 32, 64, 128, 256]
metadata = json.loads(SOURCE.read_text())
OUTPUT.mkdir(parents=True, exist_ok=True)

for samples in SAMPLES:
    folder = OUTPUT / f"samples_{samples:03d}"
    folder.mkdir(parents=True, exist_ok=True)
    env = dict(
        os.environ,
        PYTHONPATH=str(ROOT / "third_party/BlenderProc"),
        FRONT3D_SCENE=str(Path("/tmp/datasets/3dfront/3D-FRONT") / f"{metadata['scene']}.json"),
        FRONT3D_ROOM=metadata["room"],
        FRONT3D_OUTPUT=str(folder),
        FRONT3D_HDRI=metadata["hdri"],
        FRONT3D_HDRI_ROTATION=str(metadata["hdri_rotation_degrees"]),
        FRONT3D_HDRI_STRENGTH=str(metadata["hdri_strength"]),
        FRONT3D_CEILING_EMISSION_STRENGTH="1.1",
        FRONT3D_LAMP_EMISSION_STRENGTH="10.0",
        FRONT3D_WINDOW_STYLE="grid",
        FRONT3D_DOOR_STYLE="glazed",
        FRONT3D_TIME_OF_DAY="day",
        FRONT3D_ENVIRONMENT="city",
        FRONT3D_WIDTH="2048",
        FRONT3D_SAMPLES=str(samples),
        FRONT3D_SKIP_GLB="1",
    )
    started = time.perf_counter()
    with (folder / "render.log").open("w") as log:
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts/data/3dfront/render_front3d_scene.py")],
            cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
        )
    elapsed = time.perf_counter() - started
    if not (folder / "panorama.png").is_file():
        raise RuntimeError(f"{samples}-sample render failed with {result.returncode}")
    log_text = (folder / "render.log").read_text()
    match = re.search(r"Finished rendering after ([0-9.]+) seconds", log_text)
    (folder / "timing.json").write_text(json.dumps({
        "max_samples": samples,
        "cycles_render_seconds": float(match.group(1)) if match else None,
        "total_process_seconds": elapsed,
    }, indent=2) + "\n")

reference = np.asarray(Image.open(OUTPUT / "samples_256/panorama.png").convert("RGB"), dtype=np.float32) / 255.0
rows = []
font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 22)
sheet = Image.new("RGB", (1200, len(SAMPLES) * 350), "#15191f")
draw = ImageDraw.Draw(sheet)
for row, samples in enumerate(SAMPLES):
    folder = OUTPUT / f"samples_{samples:03d}"
    image = Image.open(folder / "panorama.png").convert("RGB")
    array = np.asarray(image, dtype=np.float32) / 255.0
    mse = float(np.mean((array - reference) ** 2))
    psnr = math.inf if mse == 0 else -10.0 * math.log10(mse)
    timing = json.loads((folder / "timing.json").read_text())
    result = {**timing, "psnr_vs_256_db": psnr}
    rows.append(result)
    y = row * 350
    sheet.paste(image.resize((600, 300)), (0, y))
    crop = image.crop((700, 250, 1340, 890)).resize((300, 300))
    sheet.paste(crop, (600, y))
    difference = np.clip(np.abs(array - reference) * 8.0, 0, 1)
    diff_image = Image.fromarray((difference * 255).astype(np.uint8))
    sheet.paste(diff_image.resize((300, 150)), (900, y))
    label = (f"{samples} max samples | render {timing['cycles_render_seconds']:.1f}s | "
             f"total {timing['total_process_seconds']:.1f}s | PSNR {psnr:.2f} dB")
    draw.text((8, y + 307), label, font=font, fill="white")
sheet.save(OUTPUT / "comparison.jpg", quality=94)
(OUTPUT / "comparison.json").write_text(json.dumps(rows, indent=2) + "\n")
print(json.dumps(rows, indent=2))
