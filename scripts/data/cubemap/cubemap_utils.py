"""Small image utilities shared by cubemap generation commands."""

import cv2
import numpy as np


def pad_to_2to1(img, mode="constant", return_mask=False):
    """Pad an image at the bottom until its width:height ratio is 2:1."""
    height, width = img.shape[:2]
    target_height = width // 2

    if height >= target_height:
        if return_mask:
            return img, np.ones((height, width), dtype=np.uint8)
        return img

    pad_bottom = target_height - height
    pad_width = ((0, pad_bottom), (0, 0)) + ((0, 0),) * (img.ndim - 2)

    if mode == "edge":
        padded = np.pad(img, pad_width, mode="edge")
    elif mode == "constant":
        padded = np.pad(img, pad_width, mode="constant", constant_values=0)
    elif mode == "blur":
        blur = cv2.GaussianBlur(img, (51, 51), 0)
        bottom = blur[-1:].repeat(pad_bottom, axis=0)
        padded = np.vstack([img, bottom])
    else:
        raise ValueError("mode must be 'edge', 'constant', or 'blur'")

    if return_mask:
        mask = np.zeros((target_height, width), dtype=np.uint8)
        mask[:height] = 1
        return padded, mask
    return padded
