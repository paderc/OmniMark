from __future__ import annotations

import functools
import math
import random
from itertools import cycle
from typing import Any, Callable

import cv2
import numpy as np
from PIL import Image, ImageChops, ImageDraw

from . import ColorUtils, TextUtils
from . import ProofWatermarks as pw

Renderer = Callable[..., "tuple[Image.Image, Image.Image]"]


# ---------- shared helpers ----------

def _rng(seed: int | None):
    """random.Random(seed), or the global `random` module when unseeded (as before)."""
    return random if seed is None else random.Random(seed)


def _centre(w: int, h: int) -> tuple[float, float, float, float]:
    return (w / 4, h / 4, 3 * w / 4, 3 * h / 4)


def _median_for_region(
    image: Image.Image,
    region: tuple[float, float, float, float],
    randomize: bool = False,
    jitter: int = 60,
    invert: bool | None = None,
    rng=random,
) -> tuple[int, int, int]:
    cropped = image.crop(region)
    pixels = np.array(cropped)
    median_color = np.median(pixels.reshape(-1, 3), axis=0).astype(int)

    if not randomize:
        return tuple(median_color)

    r, g, b = (int(c) for c in median_color)
    gray = int(round(0.299 * r + 0.587 * g + 0.114 * b))
    value = gray + rng.randint(-int(jitter), int(jitter))

    if invert is None:
        invert = rng.random() < 0.5
    if invert:
        value = 255 - value

    value = max(0, min(255, value))
    return (value, value, value)


def _compose(base: Image.Image, mask: Image.Image, rgb, opacity: float,
             holes: float, seed: int | None) -> tuple[Image.Image, Image.Image]:
    """Paint `mask` in `rgb` at `opacity` over `base`, punching holes first."""
    opacity = min(1.0, max(0.0, float(opacity)))
    overlay = Image.new("RGBA", base.size, (*rgb, 0))
    overlay.putalpha(mask.point(lambda v: round(v * opacity)))
    if holes > 0:
        overlay = pw.punch_holes(overlay, min(1.0, float(holes)), seed)
    marked = Image.alpha_composite(base.convert("RGBA"), overlay).convert("RGB")
    return marked, overlay.getchannel("A")


def _screen(mask: Image.Image, patch: Image.Image, x: int, y: int) -> None:
    """Merge `patch` into `mask` at (x, y) like alpha-over"""
    W, H = mask.size
    l, t, r, b = max(0, x), max(0, y), min(W, x + patch.width), min(H, y + patch.height)
    if r <= l or b <= t:
        return
    region = mask.crop((l, t, r, b))
    mask.paste(ImageChops.screen(region, patch.crop((l - x, t - y, r - x, b - y))), (l, t))


def _stamp_text(mask: Image.Image, text: str, font, cx: float, cy: float, angle: float) -> None:
    """Draw `text` centred on (cx, cy), rotated counter-clockwise around that point.
    """
    if not text:
        return
    l, t, r, b = font.getbbox(text, anchor="mm")
    R = math.ceil(math.hypot(max(-l, r), max(-t, b))) + 2
    fx, fy = cx - math.floor(cx), cy - math.floor(cy)
    patch = Image.new("L", (2 * R + 1, 2 * R + 1), 0)
    ImageDraw.Draw(patch).text((R + fx, R + fy), text, fill=255, anchor="mm", font=font)
    patch = patch.rotate(angle, resample=Image.BICUBIC, center=(R + fx, R + fy))
    _screen(mask, patch, math.floor(cx) - R, math.floor(cy) - R)


# holes

def hole_maker(wm_image: Image.Image, hole: float = 0.15, seed: int | None = 0,
               size: float = 1.0, **_old_options: Any) -> Image.Image:
    """Punch little holes into watermark text so pattern-matching attacks miss it.
    """
    if wm_image.mode in ("L", "1", "LA", "RGBA"):
        source = wm_image.convert("RGBA" if wm_image.mode in ("LA", "RGBA") else "L")
        return pw.punch_holes(source, hole, seed, size)

    from .Attack import _detect_strokes  # imported here: Attack may import this module
    rgb = np.array(ColorUtils._to_rgb(wm_image))
    mask, _ = _detect_strokes(rgb)
    strokes = (np.asarray(mask) > 0).astype(np.uint8) * 255
    holes = pw.hole_mask(strokes, hole, seed, size)[..., None]
    background = cv2.medianBlur(rgb, 21).astype(np.float32)
    out = rgb.astype(np.float32) * (1 - holes) + background * holes
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


# classic watermarkers

def render_pillow_centered(image, watermark, opacity=0.5, rotation=0, amount=1, *, holes=0.0, seed=None):
    base = ColorUtils._to_rgb(image)
    w, h = base.size
    amount = max(1, int(amount))
    rng = _rng(seed)
    font_it = cycle(TextUtils._fonts(0.7 * TextUtils._auto_font_size(w, h, watermark, amount)))
    color = _median_for_region(base, _centre(w, h), randomize=True, jitter=60, invert=True, rng=rng)

    mask = Image.new("L", (w, h), 0)
    for i in range(amount):
        for j in range(amount):
            angle = rotation + rng.random() * 10 - 5
            cx, cy = (i + 0.5) * w / amount, (j + 0.5) * h / amount
            _stamp_text(mask, watermark, next(font_it), cx, cy, angle)
    return _compose(base, mask, color, opacity, holes, seed)


def render_opencv_blend(image, watermark, opacity=0.5, rotation=0, amount=1, *, holes=0.0, seed=None):
    base = ColorUtils._to_rgb(image)
    w, h = base.size
    amount = max(1, int(amount))
    scale = TextUtils._auto_font_size(w, h, watermark, amount) / 50.0
    thickness = max(1, int(scale * 2))
    face = cv2.FONT_HERSHEY_SIMPLEX
    color = _median_for_region(base, _centre(w, h), randomize=True, jitter=60, invert=True, rng=_rng(seed))

    # One patch, rotated once, stamped into every cell (the old code warped the
    # full frame once per cell; the result is the same).
    (tw, th), baseline = cv2.getTextSize(watermark, face, scale, thickness)
    R = math.ceil(math.hypot(tw, th + 2 * baseline) / 2 + thickness) + 2
    patch = np.zeros((2 * R + 1, 2 * R + 1), np.uint8)
    cv2.putText(patch, watermark, (int(R - tw / 2), int(R + th / 2)), face, scale, 255, thickness, cv2.LINE_AA)
    patch = cv2.warpAffine(patch, cv2.getRotationMatrix2D((R, R), rotation, 1.0), patch.shape[::-1])

    mask = np.zeros((h, w), np.uint8)
    for i in range(amount):
        for j in range(amount):
            x = round((i + 0.5) * w / amount) - R
            y = round((j + 0.5) * h / amount) - R
            l, t, r, b = max(0, x), max(0, y), min(w, x + patch.shape[1]), min(h, y + patch.shape[0])
            if r > l and b > t:
                mask[t:b, l:r] = np.maximum(mask[t:b, l:r], patch[t - y:b - y, l - x:r - x])
    return _compose(base, Image.fromarray(mask), color, opacity, holes, seed)


def render_pillow_tiled_diagonal(image, watermark, opacity=0.5, rotation=30, amount=4, *, holes=0.0, seed=None):
    base = ColorUtils._to_rgb(image)
    w, h = base.size
    amount = max(1, int(amount))
    diag = int(math.hypot(w, h))
    row_h = diag / amount
    font = TextUtils._font(TextUtils._auto_font_size(w, h, watermark, amount))
    step_x = max(int(font.getlength(watermark + "    ")), 1)
    color = _median_for_region(base, _centre(w, h), randomize=True, jitter=60, invert=True, rng=_rng(seed))

    sheet = Image.new("L", (diag, diag), 0)
    d = ImageDraw.Draw(sheet)
    for r in range(amount):
        y = int((r + 0.5) * row_h)
        start = -((r % 2) * step_x // 2)
        for x in range(start, diag, step_x):
            d.text((x, y), watermark, fill=255, anchor="lm", font=font)
    sheet = sheet.rotate(rotation, resample=Image.BICUBIC)
    left, top = (diag - w) // 2, (diag - h) // 2
    return _compose(base, sheet.crop((left, top, left + w, top + h)), color, opacity, holes, seed)


# presets

PRESETS: dict[str, dict[str, Any]] = {
    "Event proof": {"pattern": "cross", "layers": "pattern,banner,block,caption"},
    "Stock lattice": {"pattern": "lattice", "layers": "pattern,block,caption"},
    "Stock band": {"pattern": "band", "layers": "pattern,block,caption"},
    "Diagonal rows": {"pattern": "rows", "layers": "pattern,caption"},
    "Wave lines": {"pattern": "waves", "layers": "pattern,block"},
    "Border frame": {"pattern": "frame", "layers": "pattern,block"},
}


def _contrast_color(image: Image.Image) -> str:
    """Black on bright images, white on dark ones (median of the whole frame)."""
    grey = np.asarray(ColorUtils._to_rgb(image).convert("L").resize((64, 64)))
    return "#000000" if float(np.median(grey)) > 140 else "#ffffff"


def _clockwise(rotation: float) -> float:
    """main.py rotation (counter-clockwise) to WatermarkParams.angle (clockwise), in [-180, 180]."""
    return ((-float(rotation) + 180) % 360) - 180


def params_for(name: str, image: Image.Image, text: str, opacity: float = 0.8, rotation: float = 0,
               amount: int = 3, *, holes: float = 0.0, seed: int | None = None):
    """(params, method) that ProofWatermarks uses for a preset; None for the classic three.

    Pass the params to ProofWatermarks.watermark_image(image, params, method)
    to get the same watermark through the TS-compatible front end.
    """
    amount = max(1, int(amount))
    common = {"text": text, "opacity": min(1.0, max(0.0, float(opacity))), "angle": _clockwise(rotation),
              "color": _contrast_color(image), "holes": min(1.0, max(0.0, float(holes))), "seed": seed}
    if name in PRESETS:
        return {**common, "count": min(50, amount), **PRESETS[name]}, "proof"
    if name == "TS mark":
        tile = {"mode": "tile", "scale": min(1.0, max(0.05, 0.9 / amount))} if amount > 1 else {}
        return {**common, **tile}, "mark"
    return None


def _preset(name: str) -> Renderer:
    def render_preset(image, watermark, opacity=0.8, rotation=0, amount=3, *, holes=0.0, seed=None):
        params, method = params_for(name, image, watermark, opacity, rotation, amount, holes=holes, seed=seed)
        marked, mask = pw.watermark_image(ColorUtils._to_rgb(image), params, method)
        return marked.convert("RGB"), mask
    render_preset.__name__ = "render_" + name.lower().replace(" ", "_")
    return render_preset


# registry

RENDERERS: dict[str, Renderer] = {
    "Pillow centered": render_pillow_centered,
    "OpenCV blend": render_opencv_blend,
    "Pillow tiled diagonal": render_pillow_tiled_diagonal,
    **{name: _preset(name) for name in PRESETS},
    "TS mark": _preset("TS mark"),
}


def _image_only(fn: Renderer) -> Callable[..., Image.Image]:
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)[0]
    return wrapper


wm_pillow_centered = _image_only(render_pillow_centered)
wm_opencv_blend = _image_only(render_opencv_blend)
wm_pillow_tiled_diagonal = _image_only(render_pillow_tiled_diagonal)

WATERMARKERS: dict[str, Callable[..., Image.Image]] = {name: _image_only(fn) for name, fn in RENDERERS.items()}


def mask_from_difference(original: Image.Image, marked: Image.Image, threshold: int = 6) -> Image.Image:
    """Approximate watermark mask for watermarkers that do not report one."""
    a = np.asarray(ColorUtils._to_rgb(original), np.int16)
    b = np.asarray(ColorUtils._to_rgb(marked), np.int16)
    return Image.fromarray(np.where(np.abs(a - b).max(axis=2) > threshold, 255, 0).astype(np.uint8))


def render(name: str, image: Image.Image, text: str, opacity: float = 0.8, rotation: float = 0,
           amount: int = 3, *, holes: float = 0.0, seed: int | None = None) -> tuple[Image.Image, Image.Image]:
    """(watermarked, mask) for any registry entry; entries added elsewhere get an estimated mask."""
    if name in RENDERERS:
        return RENDERERS[name](image, text, opacity, rotation, amount, holes=holes, seed=seed)
    marked = WATERMARKERS[name](image, text, opacity, rotation, amount)
    return marked, mask_from_difference(image, marked)