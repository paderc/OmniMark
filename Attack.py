from __future__ import annotations

import hashlib
import time

import cv2
import numpy as np
import torch
from PIL import Image
from OmniMark import ColorUtils

from simple_lama_inpainting import SimpleLama

# ==============================================================================
# DEVICE / GLOBAL SETTINGS
# ==============================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[torch] {torch.__version__} | cuda build: {torch.version.cuda} | using: {device}")
if device.type != "cuda":
    print("[torch] WARNING: running on CPU - LaMa will be slow.")

if device.type == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


# ==============================================================================
# PER-IMAGE MASK CACHE
# ==============================================================================

def _image_cache(rgb: np.ndarray, ctx):
    if not isinstance(ctx, dict):
        return {}
    key = hashlib.md5(rgb.tobytes()).hexdigest()
    if ctx.get("_wm_key") != key:
        ctx.pop("_wm_mask", None)
        ctx["_wm_key"] = key
    return ctx


def _get_mask(rgb: np.ndarray, ctx) -> np.ndarray:
    cache = _image_cache(rgb, ctx)
    if "_wm_mask" not in cache:
        mask, _ = _detect_watermark_mask(rgb)
        cache["_wm_mask"] = mask
    return cache["_wm_mask"]


# ==============================================================================
# BASIC HELPERS / CLASSIC ATTACKS
# ==============================================================================

def _rgb_u8(image) -> np.ndarray:
    if isinstance(image, Image.Image):
        image = ColorUtils._to_rgb(image)
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    elif arr.ndim == 3 and arr.shape[2] == 4:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError("Expected a PIL image or an RGB/RGBA/grayscale NumPy array.")
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def _odd_kernel(value: float, max_dim: int, minimum: int = 3) -> int:
    max_dim = int(max_dim)
    if max_dim < minimum:
        return 1
    upper = max_dim if max_dim % 2 else max_dim - 1
    value = max(minimum, int(round(value)))
    if value % 2 == 0:
        value += 1
    return min(value, upper)


def _feather(mask_u8: np.ndarray, sigma: float = 0.8) -> np.ndarray:
    alpha = (np.asarray(mask_u8) > 0).astype(np.float32)
    if sigma <= 0:
        return alpha
    return cv2.GaussianBlur(alpha, (0, 0), sigmaX=sigma, sigmaY=sigma,
                            borderType=cv2.BORDER_REPLICATE).clip(0.0, 1.0)


def _detect_watermark_mask(rgb_u8: np.ndarray, kernel_fracs=(0.035, 0.05, 0.075),
                           max_cover: float = 0.35, min_cc_frac: float = 2e-5,
                           padding_scale: float = 1.1):
    rgb = _rgb_u8(rgb_u8)
    h, w = rgb.shape[:2]
    empty = np.zeros((h, w), dtype=np.uint8)
    if min(h, w) < 5:
        return empty, 1.0

    def _collect(image: np.ndarray):
        masks, widths = [], []
        for frac in kernel_fracs:
            found, width_px = ColorUtils._detect_strokes(
                image, kernel_frac=float(frac), max_cover=float(max_cover),
                min_cc_frac=float(min_cc_frac))
            if found is None:
                found = empty.copy()
            found = np.asarray(found)
            if found.ndim == 3:
                found = cv2.cvtColor(found, cv2.COLOR_RGB2GRAY)
            if found.shape != (h, w):
                raise ValueError("ColorUtils._detect_strokes returned a mask with the wrong shape.")
            masks.append((found > 0).astype(np.uint8))
            widths.append(float(width_px) if width_px is not None else 1.0)
        stack = np.stack(masks, axis=0)
        votes = stack.sum(axis=0)
        middle = stack[len(stack) // 2] > 0
        consensus = votes >= 2
        return middle | consensus, middle, consensus, widths

    def _usable(m: np.ndarray) -> bool:
        cov = float(m.mean())
        return 1e-4 < cov <= max_cover

    combined, middle, consensus, widths = _collect(rgb)
    inv_combined, inv_middle, inv_consensus, inv_widths = _collect(255 - rgb)

    if not _usable(combined) and _usable(inv_combined):
        combined, middle, consensus, widths = inv_combined, inv_middle, inv_consensus, inv_widths

    if combined.mean() > max_cover:
        combined = consensus if consensus.any() and consensus.mean() <= max_cover else middle

    mask = combined.astype(np.uint8) * 255
    good_widths = [x for x in widths if np.isfinite(x) and x > 0]
    if not np.any(mask):
        return mask, max(1.0, float(np.median(good_widths or [1.0])))

    stroke_width = max(1.0, float(np.median(good_widths or [2.0])))
    close_size = min(_odd_kernel(stroke_width * 1.1, min(h, w)), 7)

    if close_size >= 3:
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)

    pad = int(np.clip(round(stroke_width * padding_scale), 1, 8)) 
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * pad + 1, 2 * pad + 1))
    expanded = cv2.dilate(mask, dilate_kernel, iterations=1)

    if expanded.mean() / 255.0 > min(0.50, max_cover * 2.0):
        expanded = mask

    return expanded, stroke_width


def _inpaint_feathered(rgb_u8: np.ndarray, mask_u8: np.ndarray,
                       radius: float = 5, feather: float = 0.8) -> np.ndarray:
    rgb = _rgb_u8(rgb_u8)
    mask = (np.asarray(mask_u8) > 0).astype(np.uint8) * 255
    if not np.any(mask):
        return rgb.copy()

    radius = max(1.0, float(radius))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    inpainted_bgr = cv2.inpaint(bgr, mask, radius, cv2.INPAINT_TELEA)
    inpainted = cv2.cvtColor(inpainted_bgr, cv2.COLOR_BGR2RGB)

    if feather <= 0:
        return inpainted

    alpha = _feather(mask, feather)[..., None]
    result = (1.0 - alpha) * rgb.astype(np.float32) + alpha * inpainted.astype(np.float32)
    return np.clip(np.rint(result), 0, 255).astype(np.uint8)


def attack_inpaint_blind(wm_image, passes: int = 2, radius: float | None = None,
                         feather: float = 0.8):
    img = _rgb_u8(wm_image)
    for _ in range(max(0, int(passes))):
        mask, stroke_width = _detect_watermark_mask(img)
        if not np.any(mask):
            break
        current_radius = (float(np.clip(round(stroke_width * 1.8), 3, 10))
                          if radius is None else max(1.0, float(radius)))
        img = _inpaint_feathered(img, mask, radius=current_radius, feather=feather)
    return Image.fromarray(img, mode="RGB")


def attack_unblend_hybrid(wm_image, inpaint_radius: float | None = None,
                          max_opacity: float = 0.5, feather: float = 0.8):
    rgb = _rgb_u8(wm_image)
    original = rgb.astype(np.float32)
    h, w = rgb.shape[:2]
    mask, stroke_width = _detect_watermark_mask(
        rgb, kernel_fracs=(0.025, 0.035, 0.05, 0.075), max_cover=0.40,
        min_cc_frac=5e-6, padding_scale=1.4)
    if not np.any(mask):
        return Image.fromarray(rgb, mode="RGB")

    radius = (float(np.clip(round(stroke_width * 1.8), 3, 10))
              if inpaint_radius is None else max(1.0, float(inpaint_radius)))
    max_opacity = float(np.clip(max_opacity, 0.10, 0.90))
    active = mask > 0

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    background_bgr = cv2.inpaint(bgr, mask, max(3.0, radius), cv2.INPAINT_TELEA)
    background = cv2.cvtColor(background_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)

    if active.any():
        wm_mean = float(original[active].mean())
        bg_mean = float(background[active].mean())
    else:
        wm_mean, bg_mean = 255.0, 0.0
    overlay_color = 255.0 if wm_mean >= bg_mean else 0.0

    direction = overlay_color - background
    delta = original - background
    numerator = np.sum(delta * direction, axis=2)
    denominator = np.sum(direction * direction, axis=2)
    alpha = np.divide(numerator, denominator, out=np.zeros((h, w), dtype=np.float32),
                      where=denominator > 16.0)
    alpha = np.clip(alpha, 0.0, 1.0)
    alpha[~active] = 0.0

    alpha = cv2.GaussianBlur(alpha, (3, 3), 0.6)
    alpha[~active] = 0.0
    alpha = np.clip(alpha, 0.0, 1.0)

    safe_alpha = np.minimum(alpha, max_opacity)
    divisor = np.maximum(1.0 - safe_alpha, 1.0 - max_opacity)
    recovered = (original - overlay_color * safe_alpha[..., None]) / divisor[..., None]
    recovered = np.clip(recovered, 0.0, 255.0)

    start = max(0.0, max_opacity - 0.15)
    end = min(1.0, max_opacity + 0.10)
    t = np.clip((alpha - start) / max(end - start, 1e-6), 0.0, 1.0)
    t = t * t * (3.0 - 2.0 * t)
    recovered = (1.0 - t[..., None]) * recovered + t[..., None] * background

    recovered[~active] = original[~active]

    if feather > 0:
        edge_alpha = _feather(mask, feather)[..., None]
        recovered = (1.0 - edge_alpha) * original + edge_alpha * recovered

    return Image.fromarray(np.clip(np.rint(recovered), 0, 255).astype(np.uint8), mode="RGB")


def _lama_crop_inpaint(lama, rgb: np.ndarray, mask: np.ndarray,
                       work_res: int = 2048, margin: int = 64,
                       feather: float = 2.0) -> np.ndarray:
    h, w = rgb.shape[:2]
    ys, xs = np.where(mask > 0)
    if len(ys) == 0: return rgb
    
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1

    if (x1 - x0) > 0.6 * w or (y1 - y0) > 0.6 * h:
        top, left, bot, right = 0, 0, h, w
    else:
        top, left = max(0, y0 - margin), max(0, x0 - margin)
        bot, right = min(h, y1 + margin), min(w, x1 + margin)

    crop  = rgb[top:bot, left:right]
    cmask = mask[top:bot, left:right]
    ch, cw = crop.shape[:2]

    scale = min(1.0, work_res / max(ch, cw))
    nw = max(8, int(round(cw * scale / 8)) * 8)
    nh = max(8, int(round(ch * scale / 8)) * 8)

    img_in = Image.fromarray(crop)
    msk_in = Image.fromarray(cmask).convert("L")
    if (nw, nh) != (cw, ch):
        img_in = img_in.resize((nw, nh), Image.Resampling.LANCZOS)
        msk_in = msk_in.resize((nw, nh), Image.Resampling.NEAREST)

    result = lama(img_in, msk_in).convert("RGB")
    if result.size != (cw, ch):
        result = result.resize((cw, ch), Image.Resampling.LANCZOS)

    blend = cv2.dilate(cmask, np.ones((7, 7), np.uint8))
    alpha = _feather(blend, feather)[..., None]
    blended = alpha * np.asarray(result, np.float32) + (1.0 - alpha) * crop.astype(np.float32)

    out = rgb.copy()
    out[top:bot, left:right] = np.clip(np.rint(blended), 0, 255).astype(np.uint8)
    return out


# ==============================================================================
# AI ATTACKS (SimpleLama)
# ==============================================================================

_lama_model = None

def _ensure_lama():
    global _lama_model
    if _lama_model is None:
        print("[SimpleLama] Loading model...")
        _lama_model = SimpleLama(device=device)
    return _lama_model


def attack_ai_simple_lama(wm_image, ctx=None, *, work_res: int = 2048):
    """Standard LaMa attack."""
    if not isinstance(wm_image, Image.Image):
        wm_image = Image.fromarray(wm_image)

    rgb = _rgb_u8(wm_image)
    mask = _get_mask(rgb, ctx)
    
    if not np.any(mask):
        return wm_image

    lama = _ensure_lama()
    try:
        with torch.inference_mode():
            out = _lama_crop_inpaint(lama, rgb, mask, work_res=work_res, margin=32)
        return Image.fromarray(out, mode="RGB")
    except Exception as e:
        print(f"[SimpleLama] Error: {e}")
        return wm_image


def attack_ai_lama_guided(wm_image, ctx=None, *, work_res: int = 2048):
    """
    LaMa Guided: Pre-inpaints the mask with Telea to blend the background,
    then passes it to LaMa to reconstruct realistic textures. Prevents
    LaMa from hallucinating solid black blobs on top of complex objects.
    """
    if not isinstance(wm_image, Image.Image):
        wm_image = Image.fromarray(wm_image)

    rgb = _rgb_u8(wm_image)
    mask = _get_mask(rgb, ctx)
    
    if not np.any(mask):
        return wm_image

    lama = _ensure_lama()
    try:
        # 1. OpenCV Telea removes the solid black/white pixels
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        pre_bgr = cv2.inpaint(bgr, mask, 5, cv2.INPAINT_TELEA)
        pre_rgb = cv2.cvtColor(pre_bgr, cv2.COLOR_BGR2RGB)
        
        # 2. LaMa sharpens the blurry Telea fill
        with torch.inference_mode():
            out = _lama_crop_inpaint(lama, pre_rgb, mask, work_res=work_res, margin=32)
        return Image.fromarray(out, mode="RGB")
    except Exception as e:
        print(f"[LaMa Guided] Error: {e}")
        return wm_image


def attack_ai_lama_residual(wm_image, ctx=None, *, work_res: int = 2048):
    """
    LaMa Residual: Runs LaMa, detects any leftover harsh anti-aliased pixels,
    and runs a second targeted LaMa pass to clean them up.
    """
    if not isinstance(wm_image, Image.Image):
        wm_image = Image.fromarray(wm_image)

    rgb = _rgb_u8(wm_image)
    mask = _get_mask(rgb, ctx)
    
    if not np.any(mask):
        return wm_image

    lama = _ensure_lama()
    try:
        with torch.inference_mode():
            # First pass
            out1 = _lama_crop_inpaint(lama, rgb, mask, work_res=work_res, margin=32)
            
            # Detect residual watermark (tighter limits)
            mask2, _ = _detect_watermark_mask(out1, max_cover=0.15, padding_scale=1.5)
            if np.any(mask2):
                # Second pass
                out2 = _lama_crop_inpaint(lama, out1, mask2, work_res=work_res, margin=16)
                return Image.fromarray(out2, mode="RGB")
            
            return Image.fromarray(out1, mode="RGB")
    except Exception as e:
        print(f"[LaMa Residual] Error: {e}")
        return wm_image


# ==============================================================================
# ATTACKS REGISTRY
# ==============================================================================
_ATTACKS_RAW = {
    "Inpaint (blind)":    lambda wm, ctx: attack_inpaint_blind(wm),
    "Unblend":            lambda wm, ctx: attack_unblend_hybrid(wm),
    "AI (LaMa)":          lambda wm, ctx: attack_ai_simple_lama(wm, ctx),
    "AI (LaMa Guided)":   lambda wm, ctx: attack_ai_lama_guided(wm, ctx),
    "AI (LaMa Residual)": lambda wm, ctx: attack_ai_lama_residual(wm, ctx),
}


def _timed(name, fn):
    def wrapper(wm, ctx):
        t0 = time.perf_counter()
        out = fn(wm, ctx)
        print(f"[time] {name}: {time.perf_counter() - t0:.1f}s")
        return out
    return wrapper


ATTACKS = {k: _timed(k, v) for k, v in _ATTACKS_RAW.items()}