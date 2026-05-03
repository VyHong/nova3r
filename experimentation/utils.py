import numpy as np
import cv2


def pad_to_2to1(img, mode="constant", return_mask=False):
    """
    Pads an image to 2:1 aspect ratio (width = 2 * height)

    Args:
        img: H x W x C numpy array
        mode: 'edge', 'constant', or 'blur'
        return_mask: if True, also returns a mask of valid pixels

    Returns:
        padded_img (H_new x W x C)
        mask (optional, H_new x W) -> 1 = original, 0 = padded
    """
    H, W = img.shape[:2]

    target_H = W // 2  # enforce 2:1

    if H >= target_H:
        # already tall enough (rare), just return
        if return_mask:
            return img, np.ones((H, W), dtype=np.uint8)
        return img

    pad_total = target_H - H
    pad_top = 0
    pad_bottom = pad_total

    if mode == "edge":
        padded = np.pad(img, ((pad_top, pad_bottom), (0, 0), (0, 0)), mode="edge")

    elif mode == "constant":
        padded = np.pad(img, ((pad_top, pad_bottom), (0, 0), (0, 0)), mode="constant", constant_values=0)

    elif mode == "blur":
        # create blurred top/bottom bands
        blur = cv2.GaussianBlur(img, (51, 51), 0)

        top_pad = blur[:1].repeat(pad_top, axis=0)
        bottom_pad = blur[-1:].repeat(pad_bottom, axis=0)

        padded = np.vstack([top_pad, img, bottom_pad])

    else:
        raise ValueError("mode must be 'edge', 'constant', or 'blur'")

    if return_mask:
        mask = np.zeros((target_H, W), dtype=np.uint8)
        mask[pad_top : pad_top + H] = 1
        return padded, mask

    return padded
