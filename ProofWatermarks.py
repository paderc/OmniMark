"""
NAME
       ProofWatermarks - layered visible watermarks (event-photo proofs and stock-photo styles)

SYNOPSIS
       python ProofWatermarks.py <input> <output> [--params <json>] [--method <name>] [--mask <path>] [--logo <path>]

DESCRIPTION
       The file provides two CLI methods and three Python API functions to apply 
       semi-transparent watermarks. It is driven by a JSON parameter schema
       (`WatermarkParams`), shared with the TypeScript `watermark.ts`.

METHODS (--method)
       proof   Layered watermark: pattern + banner + block + caption.
       mark    Single or tiled mark (text or logo). Closest to TS renderer.

OPTIONS (--params JSON keys)
       text              String. The watermark text (e.g., "ACME\n© 2025").
       opacity           Float 0..1. Global transparency. [Default: 0.5]
       scale             Float 0.05..1. Sizes the watermark. [Default: 0.25]
       color             Hex string (e.g., "#ffffff").
       angle             Float -180..180. Clockwise rotation. [Default: 0]
       mode              "single" | "tile". [Default: "single"], tile only works on `--method mark` (and the `block` layer of `proof`).
       position          "top-left", "center", "bottom-right", etc. [Default: "bottom-right"]
       margin            Int. Distance from edge in pixels. [Default: 20]
       shadow            Boolean. Contrast outline. [Default: true]
       spacing           Float 0..5. Gap between tiles. [Default: 0.6]
       stagger           Boolean. Offset alternating rows in tile mode. [Default: true]

EXTRAS (Supported in `proof` method only)
       layers            String or list. Any combo of: "pattern", "banner", "block", "caption".
                         [Default: all]
       pattern           String. "cross", "lattice", "rows", "band", "waves", "frame". 
                         [Default: "cross"]
       count             Int 1..50. Density of the pattern. [Default: varies by pattern]
       holes             Float 0..1. Punches holes in the strokes to defeat template attacks.
       seed              Int 0 to 4294967295. Makes holes and waves reproducible. 
                         [Default: random]
       logo              PIL.Image. Replaces the text in the `block` layer. 
                         (Must be passed via Python API or the --logo CLI flag)

QUOTING (Windows PowerShell)
       On PowerShell 5.1, inline JSON on the command line needs two things:
         1. Prefix the params argument with `--%` so PowerShell stops parsing.
         2. Escape every inner quote as \" (the CRT un-escapes it before JSON.parse).
       Without this, PowerShell splits the argument at the first space inside
       the JSON string (e.g. between "©" and "2025") and Python's argparse
       reports "unrecognized arguments: ...".
       Use `python` (direct), not `py -3.14`: the py launcher re-assembles
       the command line and drops the \" escapes.
       For anything non-trivial, write the params to a .json file instead.

CLI EXAMPLES (PowerShell 5.1)
       # Basic text watermark (layered proof style)
       PS> python ProofWatermarks.py in.jpg out.png --% --params "{\"text\":\"ACME\"}"

       # Text mark, bottom right (sharp/TS style)
       PS> python ProofWatermarks.py in.jpg out.png --method mark --% --params "{\"text\":\"© ACME\",\"position\":\"bottom-right\"}"

       # Event proof style (X pattern + text + caption bar)
       PS> python ProofWatermarks.py in.jpg out.png --% --params "{\"text\":\"ACME\",\"layers\":\"pattern,block,caption\",\"count\":1,\"opacity\":1.0}"

       # Anti-template holes
       PS> python ProofWatermarks.py in.jpg out.png --% --params "{\"text\":\"ACME\",\"holes\":0.15,\"seed\":42}"

       # Save the watermark mask
       PS> python ProofWatermarks.py in.jpg out.png --% --params "{\"text\":\"ACME\"}" --mask out_mask.png

       # Using a logo via CLI
       PS> python ProofWatermarks.py in.jpg out.png --logo brand.png --% --params "{\"text\":\"ignored\",\"layers\":\"block\"}"
"""
from __future__ import annotations

import io
import json
import math
import numbers
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable, Iterator, Mapping

import cv2
import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFont, ImageOps

# WatermarkParams

POSITIONS = ("top-left", "top-right", "bottom-left", "bottom-right", "center")
MODES = ("single", "tile")
FORMATS = ("jpeg", "png", "webp")
LAYERS = ("pattern", "banner", "block", "caption")
PATTERNS = ("cross", "lattice", "rows", "band", "waves", "frame")

DEFAULTS: dict[str, Any] = {
    "text": "WATERMARK",
    "logo": None,
    "mode": "single",
    "position": "bottom-right",
    "customPosition": None,
    "patternOffset": None,
    "opacity": 0.5,
    "scale": 0.25,
    "margin": 20,
    "color": "#ffffff",
    "angle": 0,
    "spacing": 0.6,
    "stagger": True,
    "shadow": True,
    "format": None,
    "quality": 90,
}
EXTRAS: dict[str, Any] = {"layers": None, "pattern": "cross", "count": None, "holes": 0,
                          "seed": None}
_ALIASES = {"custom_position": "customPosition", "pattern_offset": "patternOffset"}
_LAYER_ALIASES = {"cross": "pattern"}

PROOF = {
    "block": 2.4,          # centre block: 0.60 of the width at scale 0.25
    "banner": 1.75,        # top banner:   0.44
    "caption": 2.05,       # caption bar:  0.51
    "line": 0.06,          # X thickness = 0.06 * scale * shorter side
    "cross_aspect": 1.15,  # an X spans a centred box at most 1.15x as wide as tall
    "banner_top": 0.04,    # banner top = margin + 0.04 * height
    "gap_pad": 0.035,      # lines stop this far from text, x shorter side
    "next_line": 0.82,     # font size of lines after the first
    "lattice_angle": 35,   # lattice lines, degrees from horizontal
    "lattice_line": 0.35,  # lattice line width, x the X line width
    "wave_line": 0.45,     # wave line width, x the X line width
    "band_alpha": 0.45,    # band opacity, x opacity
    "frame": 0.32,         # frame width = 0.32 * scale * shorter side
    "frame_alpha": 0.55,   # frame strip opacity, x opacity
    "rows_angle": -30,     # rows direction when angle is 0 (clockwise, like sharp)
}
_DEFAULT_COUNT = {"cross": 1, "band": 1, "lattice": 3, "rows": 3, "waves": 3, "frame": 3}

BOLD_FONTS = ["arialbd.ttf", "Arial Bold.ttf", "DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf", "FreeSansBold.ttf"]
REGULAR_FONTS = ["arial.ttf", "Arial.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf", "FreeSans.ttf"]

_HEX = re.compile(r"#([0-9a-f]{3}|[0-9a-f]{6})", re.IGNORECASE)


@dataclass(frozen=True)
class Params:
    text: str
    logo: Image.Image | None
    mode: str
    position: str
    custom_position: tuple[float, float] | None
    pattern_offset: tuple[float, float] | None
    opacity: float
    scale: float
    margin: float
    color: str
    angle: float
    spacing: float
    stagger: bool
    shadow: bool
    format: str | None
    quality: int
    layers: tuple[str, ...]
    pattern: tuple[str, ...]
    count: int | None
    holes: float
    seed: int | None

    @property
    def rgb(self) -> tuple[int, int, int]:
        return _hex_rgb(self.color)

    def to_ts(self) -> dict[str, Any]:
        """camelCase dict for watermark.ts. Images are left out: send them as files."""
        rel = lambda v: None if v is None else {"x": v[0], "y": v[1]}
        return {
            "text": self.text, "mode": self.mode, "position": self.position,
            "customPosition": rel(self.custom_position), "patternOffset": rel(self.pattern_offset),
            "opacity": self.opacity, "scale": self.scale, "margin": self.margin,
            "color": self.color, "angle": self.angle, "spacing": self.spacing,
            "stagger": self.stagger, "shadow": self.shadow,
            "format": self.format, "quality": self.quality,
        }


def _js_round(v: float) -> int:
    return math.floor(v + 0.5)


def _num(value: Any, name: str, lo: float, hi: float) -> float:
    n = value
    if isinstance(value, str):
        try:
            n = math.nan if "_" in value else float(value)
        except ValueError:
            n = math.nan
    if isinstance(n, bool) or not isinstance(n, numbers.Real) or not math.isfinite(n) or not lo <= n <= hi:
        raise ValueError(f"{name} must be a number between {lo} and {hi}")
    return float(n)


def _bool(value: Any, name: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str) and value in ("true", "false"):
        return value == "true"
    raise ValueError(f"{name} must be true or false")


def _one_of(value: Any, options: tuple[str, ...], name: str) -> str:
    if not isinstance(value, str) or value not in options:
        raise ValueError(f"{name} must be one of: {', '.join(options)}")
    return value


def _names(value: Any, options: tuple[str, ...], name: str, aliases: Mapping[str, str] | None = None) -> tuple[str, ...]:
    items = value.split(",") if isinstance(value, str) else value
    try:
        names = tuple(dict.fromkeys((aliases or {}).get(s, s) for s in (str(v).strip() for v in items)))
    except TypeError:
        names = ()
    if not names or any(n not in options for n in names):
        raise ValueError(f"{name} must be one or more of: {', '.join(options)}")
    return names


def _relative(value: Any, name: str) -> tuple[float, float]:
    if isinstance(value, Mapping):
        x, y = value.get("x"), value.get("y")
    elif isinstance(value, (tuple, list)) and len(value) == 2:
        x, y = value
    else:
        raise ValueError(f"{name} must contain x and y between 0 and 1")
    return _num(x, f"{name}.x", 0, 1), _num(y, f"{name}.y", 0, 1)


def _image(value: Any, name: str) -> Image.Image | None:
    if value is None or isinstance(value, Image.Image):
        return value
    if isinstance(value, (bytes, bytearray, memoryview)):
        try:
            img = Image.open(io.BytesIO(bytes(value)))
            img.load()
            return img
        except Exception as exc:
            raise ValueError(f"{name} is not a readable image") from exc
    raise ValueError(f"{name} must be bytes or a PIL image")


def normalize_params(raw: Mapping[str, Any] | Params | None = None) -> Params:
    if isinstance(raw, Params):
        return raw
    merged: dict[str, Any] = {**DEFAULTS, **EXTRAS}
    for key, value in (raw or {}).items():
        key = _ALIASES.get(key, key)
        if key in merged and value is not None and not (isinstance(value, str) and value == ""):
            merged[key] = value

    color = merged["color"]
    if not isinstance(color, str) or not _HEX.fullmatch(color):
        raise ValueError("color must be a hex value like #fff or #ffffff")
    logo = _image(merged["logo"], "logo")
    fmt = None if merged["format"] is None else _one_of(merged["format"], FORMATS, "format")

    text = str(merged["text"]).strip()
    if logo is None:
        if not text:
            raise ValueError("text must not be empty when no logo is provided")
        if len(text) > 200:
            raise ValueError("text is too long (max 200 characters)")
    else:
        text = ""

    rel = lambda key: None if merged[key] is None else _relative(merged[key], key)
    return Params(
        text=text,
        logo=logo,
        mode=_one_of(merged["mode"], MODES, "mode"),
        position=_one_of(merged["position"], POSITIONS, "position"),
        custom_position=rel("customPosition"),
        pattern_offset=rel("patternOffset"),
        opacity=_num(merged["opacity"], "opacity", 0, 1),
        scale=_num(merged["scale"], "scale", 0.05, 1),
        margin=_num(merged["margin"], "margin", 0, 10_000),
        color=color,
        angle=_num(merged["angle"], "angle", -180, 180),
        spacing=_num(merged["spacing"], "spacing", 0, 5),
        stagger=_bool(merged["stagger"], "stagger"),
        shadow=_bool(merged["shadow"], "shadow"),
        format=fmt,
        quality=_js_round(_num(merged["quality"], "quality", 1, 100)),
        layers=LAYERS if merged["layers"] is None else _names(merged["layers"], LAYERS, "layers", _LAYER_ALIASES),
        pattern=_names(merged["pattern"], PATTERNS, "pattern"),
        count=None if merged["count"] is None else _js_round(_num(merged["count"], "count", 1, 50)),
        holes=_num(merged["holes"], "holes", 0, 1),
        seed=None if merged["seed"] is None else int(_num(merged["seed"], "seed", 0, 2**32 - 1)),
    )


# drawing helpers

def _hex_rgb(color: str) -> tuple[int, int, int]:
    h = color[1:]
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _contrast(rgb: tuple[int, int, int]) -> tuple[int, int, int]:
    r, g, b = rgb
    return (0, 0, 0) if (0.299 * r + 0.587 * g + 0.114 * b) / 255 > 0.5 else (255, 255, 255)


@lru_cache(maxsize=2)
def _font_path(bold: bool) -> str | None:
    for name in BOLD_FONTS if bold else REGULAR_FONTS:
        try:
            return ImageFont.truetype(name, 10).path
        except OSError:
            continue
    return None


@lru_cache(maxsize=512)
def _font(size: int, bold: bool) -> ImageFont.FreeTypeFont:
    path = _font_path(bold)
    return ImageFont.truetype(path, size) if path else ImageFont.load_default(size)


def _blank(size: tuple[int, int]) -> Image.Image:
    return Image.new("RGBA", size, (0, 0, 0, 0))


def _paint(size: tuple[int, int], coats: list) -> Image.Image:
    """Stack (mask, rgb, alpha) coats bottom to top into one RGBA image."""
    out = _blank(size)
    for mask, rgb, alpha in coats:
        if mask is None or alpha <= 0:
            continue
        layer = Image.new("RGBA", size, (*rgb, 0))
        layer.putalpha(mask.point(lambda v, a=alpha: _js_round(v * a)))
        out.alpha_composite(layer)
    return out


def _with_opacity(img: Image.Image, opacity: float) -> Image.Image:
    img = img.convert("RGBA")
    img.putalpha(img.getchannel("A").point(lambda v: _js_round(v * opacity)))
    return img


def _contain(img: Image.Image, max_w: int, max_h: int, enlarge: bool = True) -> Image.Image:
    k = min(max_w / img.width, max_h / img.height)
    if k >= 1 and not enlarge:
        return img
    size = (max(1, _js_round(img.width * k)), max(1, _js_round(img.height * k)))
    return img if size == img.size else img.resize(size, Image.LANCZOS)


def _fit(img: Image.Image, max_w: int, max_h: int) -> Image.Image:
    return _contain(img, max_w, max_h, enlarge=False)


def _rotate(img: Image.Image, angle: float) -> Image.Image:
    """Rotate like sharp: positive angles turn clockwise."""
    return img.rotate(-angle, Image.BICUBIC, expand=True) if angle else img


def _lines(p: Params) -> list[str]:
    return [ln.strip() for ln in p.text.split("\n")] if p.text else []


def _text_masks(lines: list[str], max_w: float, max_h: float | None, bold: bool, outline: bool):
    """Fill and outline masks for centred lines, sized so the widest line is max_w."""
    ratios = [1.0] + [PROOF["next_line"]] * (len(lines) - 1)

    def fonts_at(size: float):
        return [_font(max(1, round(size * r)), bold) for r in ratios]

    def extent(fonts):
        width = max(f.getlength(line) for f, line in zip(fonts, lines))
        return width, sum(sum(f.getmetrics()) for f in fonts)

    ref = 200
    w_ref, h_ref = extent(fonts_at(ref))
    size = ref * max_w / max(w_ref, 1.0)
    if max_h:
        size = min(size, ref * max_h / max(h_ref, 1.0))
    size = max(6, int(size))
    fonts = fonts_at(size)
    width, height = extent(fonts)

    stroke = max(1, _js_round(size * 0.045)) if outline else 0
    pad = stroke + size // 4 + 2
    canvas = (math.ceil(width) + 2 * pad, height + 2 * pad)
    fill = Image.new("L", canvas, 0)
    edge = Image.new("L", canvas, 0) if stroke else None
    y = pad
    for font, line in zip(fonts, lines):
        xy = (canvas[0] / 2, y)
        ImageDraw.Draw(fill).text(xy, line, font=font, fill=255, anchor="ma")
        if edge is not None:
            ImageDraw.Draw(edge).text(xy, line, font=font, fill=255, anchor="ma",
                                      stroke_width=stroke, stroke_fill=255)
        y += sum(font.getmetrics())
    box = (edge if edge is not None else fill).getbbox() or (0, 0, 1, 1)
    return fill.crop(box), (edge.crop(box) if edge is not None else None)


def _text_rgba(lines: list[str], max_w: float, max_h: float | None, p: Params) -> Image.Image:
    fill, edge = _text_masks(lines, max_w, max_h, bold=True, outline=p.shadow)
    ring = ImageChops.subtract(edge, fill) if edge is not None else None  # outline only around the glyphs
    return _paint(fill.size, [(ring, _contrast(p.rgb), p.opacity * 0.7), (fill, p.rgb, p.opacity)])


def _caption_rgba(text: str, max_w: float, max_h: float, p: Params) -> Image.Image:
    ink, _ = _text_masks([text], max_w * 0.92, max_h * 0.72, bold=False, outline=False)
    tw, th = ink.size
    px, py = max(3, _js_round(th * 0.45)), max(2, _js_round(th * 0.3))
    border = max(1, _js_round(th * 0.06)) if p.shadow else 0
    size = (tw + 2 * (px + border), th + 2 * (py + border))

    ring = Image.new("L", size, 255)
    ring.paste(0, (border, border, size[0] - border, size[1] - border))
    bar = Image.new("L", size, 0)
    bar.paste(255, (border, border, size[0] - border, size[1] - border))
    text = Image.new("L", size, 0)
    text.paste(ink, (px + border, py + border))

    dark = _contrast(p.rgb)
    return _paint(size, [(ring if border else None, dark, p.opacity * 0.7),
                         (bar, p.rgb, p.opacity), (text, dark, p.opacity)])


def _logo_rgba(source: Image.Image, max_w: int, max_h: int, p: Params) -> Image.Image:
    """Scale and apply opacity to the provided logo image."""
    img = _contain(source.convert("RGBA"), max_w, max_h)
    return _with_opacity(img, p.opacity)


# placement (same formulas as watermark.ts)

def _clamp(v: float, hi: float) -> int:
    return max(0, min(max(0, int(hi)), _js_round(v)))


def _single_xy(W: int, H: int, w: int, h: int, p: Params, preset: str | None = None) -> tuple[int, int]:
    m = p.margin
    if p.custom_position:
        x, y = (W - w) * p.custom_position[0], (H - h) * p.custom_position[1]
    else:
        x, y = {
            "top-left": (m, m),
            "top-right": (W - w - m, m),
            "bottom-left": (m, H - h - m),
            "bottom-right": (W - w - m, H - h - m),
            "center": ((W - w) / 2, (H - h) / 2),
        }[preset or p.position]
    return _clamp(x, W - w), _clamp(y, H - h)


def _tile_xy(W: int, H: int, w: int, h: int, p: Params) -> Iterator[tuple[int, int]]:
    cell_w, cell_h = w + _js_round(w * p.spacing), h + _js_round(h * p.spacing)
    k = 2 if p.stagger else 1
    period_w, period_h = cell_w * k, cell_h * k
    ox, oy = (p.pattern_offset or (0.0, 0.0))
    ox, oy = ox * period_w, oy * period_h
    anchors = ((0, 0), (cell_w, cell_h)) if p.stagger else ((0, 0),)
    for row in range(-2, math.ceil(H / period_h) + 2):
        for col in range(-2, math.ceil(W / period_w) + 2):
            for ax, ay in anchors:
                x, y = _js_round(col * period_w + ax + ox), _js_round(row * period_h + ay + oy)
                if x < W and y < H and x + w > 0 and y + h > 0:
                    yield x, y


def _stamp(canvas: Image.Image, mark: Image.Image, x: int, y: int) -> tuple[int, int, int, int]:
    W, H = canvas.size
    l, t, r, b = max(0, x), max(0, y), min(W, x + mark.width), min(H, y + mark.height)
    if r > l and b > t:
        canvas.alpha_composite(mark, dest=(l, t), source=(l - x, t - y, r - x, b - y))
    return x, y, x + mark.width, y + mark.height


def _place(canvas: Image.Image, mark: Image.Image, p: Params, preset: str | None = None):
    """Single mark (returns its box) or a tiled pattern (returns None)."""
    W, H = canvas.size
    if p.mode == "tile":
        for x, y in _tile_xy(W, H, mark.width, mark.height, p):
            _stamp(canvas, mark, x, y)
        return None
    mark = _fit(mark, W, H)
    return _stamp(canvas, mark, *_single_xy(W, H, mark.width, mark.height, p, preset))


def _hits(box: tuple[int, int, int, int], gaps: list[tuple[int, int, int, int]]) -> bool:
    return any(box[0] < g[2] and g[0] < box[2] and box[1] < g[3] and g[1] < box[3] for g in gaps)


def _clear(mask: np.ndarray, gaps: list[tuple[int, int, int, int]]) -> None:
    for x0, y0, x1, y1 in gaps:
        mask[max(0, y0):max(0, y1), max(0, x0):max(0, x1)] = 0


# ---------- patterns ----------

@dataclass
class _Ctx:
    gaps: list[tuple[int, int, int, int]]
    block: tuple[int, int, int, int] | None
    lines: list[str]


def _line_width(p: Params, size: tuple[int, int], factor: float = 1.0) -> int:
    return max(1, _js_round(PROOF["line"] * factor * p.scale * min(size)))


def _strokes_rgba(size: tuple[int, int], p: Params, polylines: list, width: int, gaps) -> Image.Image:
    """Antialiased polylines with an optional contrasting outline, cut away inside `gaps`."""
    W, H = size
    border = max(1, _js_round(width * 0.15)) if p.shadow else 0
    pts = [np.round(np.asarray(pl, np.float64) * 16).astype(np.int32) for pl in polylines]

    def draw(thickness: int) -> Image.Image:
        mask = np.zeros((H, W), np.uint8)
        if pts:
            cv2.polylines(mask, pts, False, 255, thickness, cv2.LINE_AA, 4)
        _clear(mask, gaps)
        return Image.fromarray(mask)

    core = draw(width)
    ring = ImageChops.subtract(draw(width + 2 * border), core) if border else None
    return _paint(size, [(ring, _contrast(p.rgb), p.opacity * 0.7), (core, p.rgb, p.opacity)])


def _pattern_cross(size, p: Params, ctx: _Ctx) -> Image.Image:
    W, H = size
    n = p.count or _DEFAULT_COUNT["cross"]
    cw, ch, m = W / n, H / n, p.margin / n
    lines = []
    for i in range(n):
        for j in range(n):
            box_h = ch - 2 * m
            box_w = min(cw - 2 * m, box_h * PROOF["cross_aspect"])
            if box_w < 2 or box_h < 2:
                continue
            x0, y0 = i * cw + (cw - box_w) / 2, j * ch + m
            x1, y1 = x0 + box_w, y0 + box_h
            lines += [[(x0, y0), (x1, y1)], [(x1, y0), (x0, y1)]]
    width = max(2, _js_round(PROOF["line"] * p.scale * min(W, H) / math.sqrt(n)))
    return _strokes_rgba(size, p, lines, width, ctx.gaps)


def _pattern_lattice(size, p: Params, ctx: _Ctx) -> Image.Image:
    W, H = size
    n = p.count or _DEFAULT_COUNT["lattice"]
    d = min(W, H) / n
    t = math.radians(PROOF["lattice_angle"])
    centre, reach = np.array([W / 2, H / 2]), math.hypot(W, H)
    k = math.ceil(reach / d) + 1
    lines, normals = [], []
    for s in (1, -1):
        u = np.array([math.cos(t), s * math.sin(t)])
        normal = np.array([-u[1], u[0]])
        normals.append(normal)
        for i in range(-k, k + 1):
            o = centre + i * d * normal
            lines.append([tuple(o - reach * u), tuple(o + reach * u)])
    out = _strokes_rgba(size, p, lines, _line_width(p, size, PROOF["lattice_line"]), ctx.gaps)

    if ctx.lines and ctx.lines[0]:
        mark = _text_rgba(ctx.lines[:1], 0.6 * d, 0.16 * d, p)
        solve = np.linalg.inv(np.array(normals))
        for a in range(-k, k):
            for b in range(-k, k):
                if (a + b) % 2:
                    continue
                x, y = centre + solve @ np.array([(a + 0.5) * d, (b + 0.5) * d])
                l, top = _js_round(x - mark.width / 2), _js_round(y - mark.height / 2)
                box = (l, top, l + mark.width, top + mark.height)
                if box[2] > 0 and box[3] > 0 and box[0] < W and box[1] < H and not _hits(box, ctx.gaps):
                    _stamp(out, mark, l, top)
    return out


def _pattern_rows(size, p: Params, ctx: _Ctx) -> Image.Image:
    W, H = size
    label = ctx.lines[0] if ctx.lines else ""
    if not label:
        return _blank(size)
    n = 2 * (p.count or _DEFAULT_COUNT["rows"])
    diag = math.ceil(math.hypot(W, H))
    row_h = diag / n
    fill, edge = _text_masks([label], diag, 0.42 * row_h, bold=True, outline=p.shadow)
    step = fill.width + max(4, _js_round(fill.height * 0.9))
    sheets = [Image.new("L", (diag, diag), 0) for _ in range(2 if edge is not None else 1)]
    for r in range(n):
        y = _js_round((r + 0.5) * row_h - fill.height / 2)
        for x in range(-((r % 2) * step // 2), diag, step):
            for sheet, mask in zip(sheets, (fill, edge)):
                sheet.paste(255, (x, y), mask)

    angle = p.angle or PROOF["rows_angle"]
    l, t = (diag - W) // 2, (diag - H) // 2

    def finish(sheet: Image.Image) -> Image.Image:
        arr = np.array(sheet.rotate(-angle, Image.BICUBIC).crop((l, t, l + W, t + H)))
        _clear(arr, ctx.gaps)
        return Image.fromarray(arr)

    core = finish(sheets[0])
    ring = ImageChops.subtract(finish(sheets[1]), core) if edge is not None else None
    return _paint(size, [(ring, _contrast(p.rgb), p.opacity * 0.7), (core, p.rgb, p.opacity)])


def _pattern_band(size, p: Params, ctx: _Ctx) -> Image.Image:
    W, H = size
    n = p.count or _DEFAULT_COUNT["band"]
    pad = _js_round(PROOF["gap_pad"] * min(W, H))
    y0, y1 = (ctx.block[1] - pad, ctx.block[3] + pad) if ctx.block else (0.42 * H, 0.58 * H)
    thick, mid = y1 - y0, (y0 + y1) / 2
    mask = np.zeros((H, W), np.uint8)
    mask[max(0, math.floor(y0)):max(0, math.ceil(y1))] = 255
    for i in range(1, n):
        yc = (mid + i * H / n) % H
        half = max(1.0, thick * 0.12)
        mask[max(0, math.floor(yc - half)):max(0, math.ceil(yc + half))] = 255
    return _paint(size, [(Image.fromarray(mask), _contrast(p.rgb), p.opacity * PROOF["band_alpha"])])


def _pattern_waves(size, p: Params, ctx: _Ctx) -> Image.Image:
    W, H = size
    n = 2 * (p.count or _DEFAULT_COUNT["waves"])
    rng = np.random.default_rng(None if p.seed is None else p.seed + 2)
    gap = H / n
    xs = np.linspace(-0.05 * W, 1.05 * W, max(64, W // 3))
    lines = []
    for i in range(n):
        wavelength = W / rng.uniform(1.2, 2.4)
        amp = gap * rng.uniform(0.25, 0.4)
        phase = rng.uniform(0, 2 * math.pi)
        tilt = rng.uniform(-0.06, 0.06)
        ys = (i + 0.5) * gap + amp * np.sin(2 * math.pi * xs / wavelength + phase) + tilt * (xs - W / 2)
        lines.append(np.column_stack([xs, ys]))
    return _strokes_rgba(size, p, lines, _line_width(p, size, PROOF["wave_line"]), ctx.gaps)


def _pattern_frame(size, p: Params, ctx: _Ctx) -> Image.Image:
    W, H = size
    m = _js_round(p.margin)
    t = max(8, _js_round(PROOF["frame"] * p.scale * min(W, H)))
    if W - 2 * m < 3 * t or H - 2 * m < 3 * t:
        return _blank(size)
    strip = Image.new("L", size, 0)
    draw = ImageDraw.Draw(strip)
    draw.rectangle((m, m, W - m - 1, H - m - 1), fill=255)
    draw.rectangle((m + t, m + t, W - m - t - 1, H - m - t - 1), fill=0)
    coats = [(strip, p.rgb, p.opacity * PROOF["frame_alpha"])]

    label = ctx.lines[0] if ctx.lines else ""
    if label:
        ink = Image.new("L", size, 0)
        across = W - 2 * (m + t)
        k = 2 * (p.count or _DEFAULT_COUNT["frame"])
        text, _ = _text_masks([label], 0.8 * across / k, 0.6 * t, bold=True, outline=False)
        for i in range(k):  # top and bottom
            x = _js_round(m + t + (i + 0.5) * across / k - text.width / 2)
            for y in (m + (t - text.height) // 2, H - m - t + (t - text.height) // 2):
                ink.paste(255, (x, y), text)
        down = H - 2 * (m + t)
        ks = max(1, int(down / (text.width / 0.8)))
        for side, rotated in ((m, text.rotate(90, expand=True)), (W - m - t, text.rotate(-90, expand=True))):
            for i in range(ks):  # left reads upwards, right reads downwards
                y = _js_round(m + t + (i + 0.5) * down / ks - rotated.height / 2)
                ink.paste(255, (side + (t - rotated.width) // 2, y), rotated)
        coats.append((ink, _contrast(p.rgb), p.opacity))
    return _paint(size, coats)


_PATTERNS: dict[str, Callable[[tuple[int, int], Params, _Ctx], Image.Image]] = {
    "cross": _pattern_cross, "lattice": _pattern_lattice, "rows": _pattern_rows,
    "band": _pattern_band, "waves": _pattern_waves, "frame": _pattern_frame,
}


# ---------- holes ----------

def hole_mask(alpha: np.ndarray, amount: float, seed: int | None = None, size: float = 1.0) -> np.ndarray:
    """Little holes inside the strokes of `alpha` (uint8), as a float mask in [0, 1]."""
    alpha = np.asarray(alpha, np.uint8)
    holes = np.zeros(alpha.shape, np.uint8)
    if amount <= 0 or not alpha.any():
        return holes.astype(np.float32)
    
    stroke = (alpha >= max(1, int(alpha.max()) // 2)).astype(np.uint8)
    dist = cv2.distanceTransform(stroke, cv2.DIST_L2, 3)
    ridge = (dist > 0.5) & (dist >= cv2.dilate(dist, np.ones((3, 3), np.uint8)))
    half = float(np.median(dist[ridge])) if ridge.any() else 1.0
    cap = max(0.8, half * size)

    core = (dist > 2.0 * max(half, 1.0)).astype(np.uint8)
    if core.any():
        grow = 2 * math.ceil(2.0 * max(half, 1.0)) + 3
        stroke &= 1 - cv2.dilate(core, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (grow, grow)))
        if not stroke.any():
            return holes.astype(np.float32)

    reach = 2 * math.ceil(cap) + 1
    local = np.minimum(cv2.dilate(dist, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (reach, reach))), cap)
    inside = np.flatnonzero(stroke.ravel())
    weights = 1.0 / np.maximum(local.flat[inside], 0.8) ** 2
    weights /= weights.sum()
    rng = np.random.default_rng(seed)
    target = amount * len(inside)

    def draw_candidates(n: int):
        picks = rng.choice(inside, size=n, p=weights)
        radii = rng.uniform(0.35, 0.75, n) * local.flat[picks]
        depth = dist.flat[picks]
        # Deep inside wide areas every hole would fit; keep about as many as in a stroke.
        ok = (depth >= radii) & (radii >= 0.5) & ((depth < cap) | (rng.random(n) < 0.45))
        return picks[ok], radii[ok], rng.uniform(0.55, 1.0, int(ok.sum()))

    pilot = draw_candidates(4096)
    mean_area = float(np.mean(math.pi * pilot[1] ** 2 * pilot[2])) * len(pilot[0]) / 4096 if len(pilot[0]) else 0
    if mean_area <= 0:
        return holes.astype(np.float32)
    picks, radii, squash = draw_candidates(int(min(1_000_000, 1.5 * target / mean_area + 64)))
    count = int(np.searchsorted(np.cumsum(math.pi * radii ** 2 * squash), target)) + 1
    angles = rng.uniform(0, 180, len(picks))
    jitter = rng.uniform(-0.5, 0.5, (len(picks), 2))
    W = alpha.shape[1]
    for idx, r, sq, ang, (jx, jy) in zip(picks[:count], radii[:count], squash, angles, jitter):
        y, x = divmod(int(idx), W)
        cv2.ellipse(holes, (_js_round((x + jx) * 4), _js_round((y + jy) * 4)),
                    (max(1, _js_round(r * 4)), max(1, _js_round(r * sq * 4))),
                    float(ang), 0, 360, 255, -1, cv2.LINE_AA, 2)  # shift=2: 1/4 px
    return holes.astype(np.float32) / 255


def punch_holes(overlay: Image.Image, amount: float, seed: int | None = None, size: float = 1.0) -> Image.Image:
    """Punch little holes into a watermark overlay (RGBA) or mask (L); same mode back."""
    if amount <= 0:
        return overlay
    rgba = overlay.mode == "RGBA"
    alpha = np.asarray(overlay.getchannel("A") if rgba else overlay.convert("L"))
    holes = hole_mask(alpha, amount, seed, size)
    new_alpha = Image.fromarray(np.round(alpha * (1 - holes)).astype(np.uint8))
    if not rgba:
        return new_alpha
    out = overlay.copy()
    out.putalpha(new_alpha)
    return out


# methods

def _mark(canvas: Image.Image, p: Params) -> None:
    """applyWatermark() from watermark.ts: one text or logo mark, single or tiled."""
    W, H = canvas.size
    width = max(16, _js_round(W * p.scale))
    max_h = width if p.mode == "tile" else _js_round(H * 0.9)
    mark = (_logo_rgba(p.logo, width, max_h, p) if p.logo is not None
            else _text_rgba(_lines(p), width, max_h, p))
    _place(canvas, _rotate(mark, p.angle), p)


def _proof(canvas: Image.Image, p: Params) -> None:
    """The layered watermark: pattern underneath, text layers on top."""
    W, H = canvas.size
    m, want, lines = p.margin, set(p.layers), _lines(p)
    pad = _js_round(PROOF["gap_pad"] * min(W, H))
    usable = max(16, int(W - 2 * m))
    width = lambda key: max(16, min(usable, _js_round(PROOF[key] * p.scale * W)))
    top = _blank((W, H))
    centred = lambda img, y: _stamp(top, img, _clamp((W - img.width) / 2, W - img.width), _clamp(y, H - img.height))
    gaps: list[tuple[int, int, int, int]] = []
    block = None

    if "block" in want:
        max_h = width("block") if p.mode == "tile" else _js_round(H * 0.45)
        main = (_logo_rgba(p.logo, width("block"), max_h, p) if p.logo is not None
                else _text_rgba(lines, width("block"), max_h, p))
        main = _rotate(main, p.angle)
        box = _place(top, main, p, preset="center")
        if box:
            gaps.append(box)
        fitted = _fit(main, W, H)
        block = box or ((W - fitted.width) // 2, (H - fitted.height) // 2,
                        (W + fitted.width) // 2, (H + fitted.height) // 2)

    if "banner" in want and lines:
        banner = _fit(_text_rgba(lines[:1], width("banner"), _js_round(H * 0.12), p), W, H)
        gaps.append(centred(banner, m + PROOF["banner_top"] * H))

    if "caption" in want and lines:
        caption = _fit(_caption_rgba(" - ".join(ln for ln in lines if ln), width("caption"),
                                     _js_round(H * 0.08), p), W, H)
        box = centred(caption, H - m - caption.height)
        gaps.append(box)

    if "pattern" in want:
        ctx = _Ctx(gaps=[(x0 - pad, y0 - pad, x1 + pad, y1 + pad) for x0, y0, x1, y1 in gaps],
                   block=block, lines=lines)
        for name in p.pattern:
            canvas.alpha_composite(_PATTERNS[name]((W, H), p, ctx))
    canvas.alpha_composite(top)


METHODS: dict[str, Callable[[Image.Image, Params], None]] = {"proof": _proof, "mark": _mark}


# ---------- public API ----------

def render_overlay(size: tuple[int, int], params: Mapping[str, Any] | Params | None = None,
                   method: str = "proof") -> Image.Image:
    """The watermark alone, as a transparent RGBA image of the given size."""
    if method not in METHODS:
        raise ValueError(f"method must be one of: {', '.join(METHODS)}")
    p = normalize_params(params)
    canvas = _blank(size)
    METHODS[method](canvas, p)
    if p.holes > 0:
        canvas = punch_holes(canvas, p.holes, None if p.seed is None else p.seed + 1)
    return canvas


def watermark_image(image: Image.Image, params: Mapping[str, Any] | Params | None = None,
                    method: str = "proof") -> tuple[Image.Image, Image.Image]:
    """Watermark a PIL image. Returns (result, mask); the mask is the watermark's alpha ("L")."""
    p = normalize_params(params)
    base = ImageOps.exif_transpose(image)
    overlay = render_overlay(base.size, p, method)
    keep_alpha = base.mode in ("RGBA", "LA", "PA") or "transparency" in base.info
    out = Image.alpha_composite(base.convert("RGBA"), overlay)
    return (out if keep_alpha else out.convert("RGB")), overlay.getchannel("A")


def apply_watermark(source: bytes | str | os.PathLike | Image.Image,
                    params: Mapping[str, Any] | Params | None = None,
                    method: str = "proof") -> dict[str, Any]:
    """Python twin of applyWatermark(): returns WatermarkResult fields plus the mask."""
    p = normalize_params(params)
    img = source if isinstance(source, Image.Image) else Image.open(
        io.BytesIO(source) if isinstance(source, (bytes, bytearray)) else source)
    src_format = (img.format or "").lower()
    out, mask = watermark_image(img, p, method)
    fmt = p.format or (src_format if src_format in FORMATS else "jpeg")
    return {"buffer": _encode(out, fmt, p.quality), "width": out.width, "height": out.height,
            "format": fmt, "mask": mask}


def _encode(img: Image.Image, fmt: str, quality: int) -> bytes:
    buf = io.BytesIO()
    if fmt == "jpeg":
        if img.mode != "RGB":
            img = Image.alpha_composite(Image.new("RGBA", img.size, (0, 0, 0, 255)), img.convert("RGBA"))
        img.convert("RGB").save(buf, "JPEG", quality=quality)
    elif fmt == "webp":
        img.save(buf, "WEBP", quality=quality)
    else:
        img.save(buf, "PNG")
    return buf.getvalue()


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Apply a proof- or stock-style watermark.")
    ap.add_argument("--params", default="{}", help='WatermarkParams as JSON, e.g. \'{"pattern": "lattice"}\'')
    ap.add_argument("--method", default="proof", choices=sorted(METHODS))
    ap.add_argument("--mask", help="also save the watermark mask as PNG")
    ap.add_argument("--logo", help="Path to a logo image")
    ap.add_argument("input")
    ap.add_argument("output")
    
    args = ap.parse_args()
    if args.logo:
        params["logo"] = open(args.logo, "rb").read()
    result = apply_watermark(args.input, json.loads(args.params), args.method)
    with open(args.output, "wb") as fh:
        fh.write(result["buffer"])
    if args.mask:
        result["mask"].save(args.mask)
