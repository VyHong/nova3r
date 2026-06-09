import zipfile
import pickle
import sys
import types
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from training.nova3r.data.datasets.replica_utils import igibson_utils

sys.modules["utils"] = types.ModuleType("utils")
sys.modules["utils.igibson_utils"] = igibson_utils

os.makedirs("debug_points", exist_ok=True)

with zipfile.ZipFile("datasets/ReplicaPano/large_apartment_0_000.zip", "r") as z:
    for name in z.namelist():
        if name.endswith("data.pkl"):
            with z.open(name) as f:
                data = pickle.load(f)

            if "layout" in data and "horizon" in data["layout"]:
                bon = data["layout"]["horizon"]["bon"]

                # Check for an RGB image to use as background
                rgb_name = name.replace("data.pkl", "rgb.png")
                img_bg = None
                img_h, img_w = 512, 1024
                if rgb_name in z.namelist():
                    with z.open(rgb_name) as img_f:
                        img_bg = np.array(Image.open(img_f))
                        img_h, img_w = img_bg.shape[:2]

                plt.figure(figsize=(10, 5))
                if img_bg is not None:
                    plt.imshow(img_bg)
                else:
                    plt.gca().invert_yaxis()
                    plt.xlim(0, 1024)
                    plt.ylim(512, 0)

                x = np.arange(bon.shape[1])

                # Based on standard HorizonNet values in spherical space...
                # bon is actually giving 1D heights. In HorizonNet they usually map directly
                # via `y = (0.5 - bon_value)*height` or similar depending on their precise definition.
                # Actually, an angle-to-pixel mapping with Equirectangular projection:
                # Let's map normalized coordinates `[-pi/2, pi/2]` or similar to pixels.

                # The paper "HorizonNet" uses Y coordinate formulation.
                # Usually: y_pixel = img_h/2.0 - bon * img_h/np.pi
                # or just directly unnormalized.

                # We saw values around [-0.72, -0.16] for ceiling and [0.18, 0.79] for floor.
                # Note: `np.pi / 2` is ~1.57.
                # We can map equirectangular spherical coordinates linearly back to pixel heights:
                #   y_pixel = H/2 - (bon * H / pi)

                y_ceil = img_h / 2.0 - (bon[0] * img_h / np.pi)
                y_floor = img_h / 2.0 - (bon[1] * img_h / np.pi)

                plt.plot(x, y_ceil, "r-", linewidth=2, label="Ceiling (bon[0])")
                plt.plot(x, y_floor, "b-", linewidth=2, label="Floor (bon[1])")

                plt.legend()
                plt.title("2D Horizon Layout Boundaries")

                out_path = "debug_points/horizon_2d_vis.png"
                plt.savefig(out_path, bbox_inches="tight")
            break
