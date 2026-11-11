from random import randint
import numpy as np
import cv2

def _to_rgb(image):
    return image.convert("RGB")


def _near_white(alpha):
    """Slightly randomised near-white colour; channels are clipped to 0-255."""
    return tuple(min(255, 255 + 50 - randint(0, 100)) for _ in range(3)) + (alpha,)

def _detect_strokes(
    rgb_u8,
    kernel_frac=0.05,
    max_cover=0.30,
    min_cc_frac=1e-4,
):
    """Detect bright, approximately white strokes using multi-space top-hat.

    Args:
        rgb_u8: RGB uint8 image array.
        kernel_frac: Top-hat kernel size relative to the smaller image dimension.
        max_cover: Increase the threshold if more than this fraction is selected.
        min_cc_frac: Minimum connected-component area as a fraction of image area.

    Returns:
        (mask_u8, stroke_width_est_px), where mask values are 0 or 255.
    """
    rgb = np.asarray(rgb_u8)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("rgb_u8 must have shape (H, W, 3) in RGB order.")
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)

    h, w = rgb.shape[:2]
    empty = np.zeros((h, w), dtype=np.uint8)
    min_dim = min(h, w)
    if min_dim < 3:
        return empty, 1.0

    # Select an odd kernel, bounded so it cannot exceed the image dimensions.
    k = max(5, int(round(min_dim * float(kernel_frac))))
    if k % 2 == 0:
        k += 1
    max_k = min_dim if min_dim % 2 else min_dim - 1
    k = min(k, max_k)
    k = max(3, k)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    min_channel = rgb.min(axis=2)

    # Both responses should be strong: grayscale suppresses color contrast,
    # while the minimum RGB channel favors nearly-white structures.
    th_gray = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    th_min = cv2.morphologyEx(min_channel, cv2.MORPH_TOPHAT, kernel)
    score = np.minimum(th_gray, th_min)

    if not np.any(score):
        return empty, 1.0

    otsu_t, _ = cv2.threshold(
        score, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    threshold = max(8.0, float(otsu_t))
    max_cover = float(np.clip(max_cover, 0.001, 0.95))

    # Raise the threshold until selected pixels fit the coverage budget.
    while threshold < 255.0:
        mask = (score > threshold).astype(np.uint8) * 255
        if float(np.count_nonzero(mask)) / float(h * w) <= max_cover:
            break
        threshold = min(255.0, threshold * 1.25 + 1.0)

    if threshold >= 255.0:
        mask = (score > threshold).astype(np.uint8) * 255

    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    min_area = max(4, int(round(h * w * max(0.0, float(min_cc_frac)))))
    clean = np.zeros_like(mask)

    # Reject tiny isolated noise components. Large structures are constrained
    # by max_cover above rather than by a brittle fixed maximum component size.
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area >= min_area:
            clean[labels == i] = 255

    if not np.any(clean):
        return clean, 1.0

    distance = cv2.distanceTransform(
        (clean > 0).astype(np.uint8), cv2.DIST_L2, 5
    )
    positive = distance[clean > 0]
    stroke_width = max(1.0, float(np.percentile(positive, 90) * 2.0))
    return clean, stroke_width