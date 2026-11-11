import numpy as np
import math
from VisibleWatermarking import ColorUtils
from cv2 import GaussianBlur

def mse(a, b):
    a, b = np.array(ColorUtils._to_rgb(a), np.float64), np.array(ColorUtils._to_rgb(b), np.float64)
    return float(np.mean((a - b) ** 2))


def psnr(a, b):
    m = mse(a, b)
    return float("inf") if m == 0 else 10 * math.log10(255.0 ** 2 / m)


def ssim(a, b):
    """SSIM (Wang et al. 2004): 11x11 Gaussian window, sigma 1.5, averaged over R,G,B."""
    a = np.array(ColorUtils._to_rgb(a), np.float64)
    b = np.array(ColorUtils._to_rgb(b), np.float64)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    blur = lambda x: GaussianBlur(x, (11, 11), 1.5)
    scores = []
    for ch in range(3):
        x, y = a[..., ch], b[..., ch]
        mx, my = blur(x), blur(y)
        sxx, syy, sxy = blur(x * x) - mx * mx, blur(y * y) - my * my, blur(x * y) - mx * my
        s = ((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx ** 2 + my ** 2 + c1) * (sxx + syy + c2))
        valid = s[5:-5, 5:-5] if min(s.shape) > 10 else s
        scores.append(valid.mean())
    return float(np.mean(scores))


def compare(ref, test):
    return {"PSNR(dB)": psnr(ref, test), "SSIM": ssim(ref, test), "MSE": mse(ref, test)}

