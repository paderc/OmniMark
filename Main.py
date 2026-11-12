"""
Visible-watermark attack bench.

Batch mode works as before: every watermarker x every attack in one figure,
plus a PSNR/SSIM/MSE table. Live mode opens an editor where sliders change
the watermark in real time; the attacks run when you press "Run attacks", or
after every change with "Auto-run attacks" ticked.

The live editor can load an optional logo/badge image. Watermarkers advertise
a logo slot only when they can use one:
    presets whose layer list contains "badge"  -> logo appears as the badge layer
    "TS mark"                                  -> logo replaces the text, TS-style
    the three classic renderers                -> no slot; logo is ignored
When no logo is loaded, nothing extra is drawn.

    python main.py                     asks for everything, batch by default
    python main.py photo.jpg --live    straight into the live editor
    python main.py photo.jpg --batch   straight into batch mode
    python main.py photo.jpg --live --logo brand.png
"""
import math
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, CheckButtons, RadioButtons, Slider, TextBox
from PIL import Image

from OmniMark import Watermarking as wm
from OmniMark import Measurers as ms
from OmniMark import Attack as atk
from OmniMark import ProofWatermarks as pwm
from OmniMark import ColorUtils

PREVIEW_SIDE = 900
MIN_FRAME_MS = 60


# ---------- helpers ----------

def ask(prompt, cast=str, default=None):
    try:
        raw = input(prompt).strip()
    except EOFError:
        raw = ""
    if not raw:
        return default
    try:
        return cast(raw)
    except ValueError:
        print(f"  could not read {raw!r}, using {default}")
        return default


def pick_file(title="Choose image"):
    try:
        from tkinter import Tk, filedialog
        root = Tk()
        root.withdraw()
        try:
            path = filedialog.askopenfilename(parent=root, title=title, filetypes=[
                ("Images", "*.png *.jpg *.jpeg *.webp *.bmp *.gif *.tif *.tiff"),
                ("All files", "*.*"),
            ])
        finally:
            root.destroy()
        return path or None
    except Exception:
        try:
            return input(f"{title} (path, empty to skip): ").strip() or None
        except EOFError:
            return None


def load_logo(path):
    if not path:
        return None, None
    try:
        img = Image.open(path)
        img.load()
        return img, Path(path).name
    except Exception as e:
        print(f"  could not load logo {path!r}: {type(e).__name__}: {e}")
        return None, None


def logo_slot(name):
    if name == "TS mark":
        return "logo"
    preset = wm.PRESETS.get(name)
    if preset and "badge" in str(preset.get("layers", "")).split(","):
        return "badge"
    return None


def render_with_logo(name, image, text, opacity, rotation, amount, *,
                     holes=0.0, seed=None, logo=None):
    slot = logo_slot(name)
    if slot is None or logo is None:
        return wm.render(name, image, text, opacity, rotation, amount,
                         holes=holes, seed=seed)
    params, method = wm.params_for(name, image, text, opacity, rotation, amount,
                                   holes=holes, seed=seed)
    params[slot] = logo
    marked, mask = pwm.watermark_image(ColorUtils._to_rgb(image), params, method)
    return marked.convert("RGB"), mask


def compare(a, b):
    try:
        return ms.compare(a, b)
    except Exception as e:
        print(f"[compare] {type(e).__name__}: {e}")
        return None


def format_metrics(m):
    if m is None:
        return "N/A"
    p = "inf" if math.isinf(m["PSNR(dB)"]) else f"{m['PSNR(dB)']:.2f}"
    return f"PSNR {p:>6} dB   SSIM {m['SSIM']:.4f}   MSE {m['MSE']:8.2f}"


# batch

def pick_watermarkers():
    names = list(wm.WATERMARKERS)
    print("\nWatermarkers:")
    for i, name in enumerate(names, 1):
        print(f"  {i:>2}. {name}")
    raw = ask("Run which, e.g. 1,4-6 [all]: ", str, "")
    if not raw:
        return names
    chosen = []
    try:
        for part in raw.split(","):
            a, _, b = part.strip().partition("-")
            chosen += [names[i - 1] for i in range(int(a), int(b or a) + 1) if 1 <= i <= len(names)]
    except ValueError:
        print("  could not read that, running all")
    return list(dict.fromkeys(chosen)) or names


def cell_size(image, nrows, max_height=13.0):
    aspect = image.height / image.width
    w, h = 3.2, 3.2 * aspect + 0.35
    if h * nrows > max_height:
        h = max_height / nrows
        w = max(1.2, (h - 0.35) / aspect)
    return w, h


def run_batch(image, text, opacity, rotation, amount, holes, names, logo=None):
    rows = []
    ncols = 2 + len(atk.ATTACKS)
    cell_w, cell_h = cell_size(image, len(names))
    fig, axes = plt.subplots(len(names), ncols, figsize=(cell_w * ncols, cell_h * len(names)), squeeze=False)
    extras = {"holes": holes} if holes else {}
    for r, wm_name in enumerate(names):
        marked, _ = render_with_logo(wm_name, image, text, opacity, rotation, amount,
                                     **extras, logo=logo)
        rows.append((wm_name, "watermark vs original", compare(image, marked)))
        panels = [("Original", image), (wm_name, marked)]

        ctx = dict(text=text, opacity=opacity, rotation=rotation, amount=amount, original=image)
        for atk_name, atk_fn in atk.ATTACKS.items():
            try:
                attacked = atk_fn(marked, ctx)
            except Exception as e:
                print(f"[{atk_name}] raised {type(e).__name__}: {e}")
                attacked = None
            if attacked is None:
                rows.append((wm_name, f"{atk_name} vs original", None))
                rows.append((wm_name, f"{atk_name} vs watermarked", None))
                panels.append((f"{atk_name} (failed)", None))
                continue
            rows.append((wm_name, f"{atk_name} vs original", compare(image, attacked)))
            rows.append((wm_name, f"{atk_name} vs watermarked", compare(marked, attacked)))
            panels.append((atk_name, attacked))

        for c, (title, im) in enumerate(panels):
            axes[r][c].imshow(im if im is not None else np.full((10, 10, 3), 200, dtype=np.uint8))
            axes[r][c].set_title(title, fontsize=8)
            axes[r][c].axis("off")

    print(f"\n{'Watermark':<22}{'Comparison':<36}{'PSNR(dB)':>10}{'SSIM':>9}{'MSE':>11}")
    print("-" * 88)
    for wm_name, label, m in rows:
        if m is None:
            print(f"{wm_name:<22}{label:<36}{'N/A':>10}{'N/A':>9}{'N/A':>11}")
            continue
        p = "inf" if math.isinf(m["PSNR(dB)"]) else f"{m['PSNR(dB)']:.2f}"
        print(f"{wm_name:<22}{label:<36}{p:>10}{m['SSIM']:>9.4f}{m['MSE']:>11.2f}")

    print("\nReading the table:")
    print("  'watermark vs original' : visual cost of the watermark")
    print("  'attack vs original'    : how close the attacked image is to the clean one")
    print("  'attack vs watermarked' : how much the attack changed the image")
    print("  'N/A'                   : the attack failed")
    plt.tight_layout()
    plt.show()


# live editor

class LiveEditor:
    def __init__(self, image, path, text, opacity, rotation, amount, holes,
                 logo=None, logo_name=None):
        self.full = image
        self.image = image.copy()
        self.image.thumbnail((PREVIEW_SIDE, PREVIEW_SIDE), Image.LANCZOS)
        self.path = Path(path)
        self.names = list(wm.WATERMARKERS)
        self.name = self.names[0]
        self.text = text
        self.logo = logo
        self.logo_name = logo_name
        self.marked = self.mask = None
        self.wm_line = ""
        self.measured = False
        self.stale = True

        n = 2 + len(atk.ATTACKS)
        self.fig = plt.figure(figsize=(max(11.0, 3.4 * n), 8.4))
        try:
            self.fig.canvas.manager.set_window_title("Watermark live editor")
        except AttributeError:
            pass

        width = 0.98 / n
        self.axes = [self.fig.add_axes([0.01 + c * width, 0.42, width - 0.01, 0.52]) for c in range(n)]
        self.status = self.fig.text(0.01, 0.385, "", fontsize=8, family="monospace", va="top")
        self._show(self.axes[0], self.image, "Original")

        # left: watermarkers
        ax = self.fig.add_axes([0.01, 0.02, 0.20, 0.28])
        ax.set_title("Watermarker", fontsize=9, loc="left")
        self.radio = RadioButtons(ax, self.names, active=0)
        for label in self.radio.labels:
            label.set_fontsize(8)
        self.radio.on_clicked(self._picked)

        # middle: sliders + text
        def slider(row, label, lo, hi, init, step=None):
            s = Slider(self.fig.add_axes([0.33, 0.27 - row * 0.045, 0.30, 0.03]),
                       label, lo, hi, valinit=min(hi, max(lo, init)), valstep=step)
            s.on_changed(self._changed)
            return s

        self.s_opacity = slider(0, "Opacity", 0, 1, opacity)
        self.s_rotation = slider(1, "Rotation", -180, 180, rotation, 1)
        self.s_amount = slider(2, "Amount", 1, 10, amount, 1)
        self.s_holes = slider(3, "Holes", 0, 0.6, holes, 0.01)
        self.s_seed = slider(4, "Seed", 0, 99, 1, 1)
        self.textbox = TextBox(self.fig.add_axes([0.33, 0.03, 0.30, 0.04]), "Text ", initial=text)
        self.textbox.on_submit(self._text)

        # right: checks, buttons, logo preview
        self.checks = CheckButtons(self.fig.add_axes([0.68, 0.16, 0.16, 0.11]),
                                   ["Show mask", "Auto-run attacks"], [False, False])
        self.checks.on_clicked(self._toggled)
        self.b_run = Button(self.fig.add_axes([0.86, 0.21, 0.12, 0.05]), "Run attacks")
        self.b_run.on_clicked(lambda _e: self.run_attacks())
        self.b_save = Button(self.fig.add_axes([0.86, 0.14, 0.12, 0.05]), "Save full size")
        self.b_save.on_clicked(lambda _e: self.save())

        # logo controls
        self.b_logo = Button(self.fig.add_axes([0.68, 0.09, 0.13, 0.05]), "Logo…")
        self.b_logo.on_clicked(lambda _e: self._load_logo())
        self.b_clear_logo = Button(self.fig.add_axes([0.82, 0.09, 0.03, 0.05]), "×")
        self.b_clear_logo.on_clicked(lambda _e: self._clear_logo())
        self.ax_logo = self.fig.add_axes([0.68, 0.02, 0.17, 0.06])
        self.ax_logo.set_xticks([]); self.ax_logo.set_yticks([])
        self._refresh_logo()

        self.fig.canvas.mpl_connect("button_release_event", self._released)

        self._last_frame = 0.0
        self._pending = False
        self._timer = self.fig.canvas.new_timer(interval=MIN_FRAME_MS)
        self._timer.single_shot = True
        self._timer.add_callback(self._flush)

        self.render()
        self.run_attacks()

    # properties used by the registry
    @property
    def args(self):
        return self.text, float(self.s_opacity.val), float(self.s_rotation.val), int(self.s_amount.val)

    @property
    def extras(self):
        return {"holes": float(self.s_holes.val), "seed": int(self.s_seed.val), "logo": self.logo}

    @property
    def auto(self):
        return self.checks.get_status()[1]

    # logo widgets
    def _refresh_logo(self):
        slot = logo_slot(self.name)
        self.b_logo.label.set_text("Logo…")
        self.b_logo.ax.set_facecolor("#f5f5f5" if slot else "#ececec")
        self.b_clear_logo.set_active(self.logo is not None)
        self.ax_logo.clear()
        self.ax_logo.set_xticks([]); self.ax_logo.set_yticks([])
        self.ax_logo.set_frame_on(False)
        if self.logo is None:
            self.ax_logo.text(0.5, 0.5, "(no logo loaded)",
                              ha="center", va="center", fontsize=7, color="0.55",
                              transform=self.ax_logo.transAxes)
        else:
            thumb = self.logo.copy()
            thumb.thumbnail((160, 80), Image.LANCZOS)
            if thumb.mode != "RGB" and thumb.mode != "RGBA":
                thumb = thumb.convert("RGBA")
            self.ax_logo.imshow(thumb)
        self.fig.canvas.draw_idle()

    def _logo_line(self):
        slot = logo_slot(self.name)
        if self.logo is None:
            return "logo: none" + ("" if slot else " (this watermarker has no logo slot)")
        if slot is None:
            return f"logo: {self.logo_name} (ignored here: no slot)"
        return f"logo: {self.logo_name} -> {slot}"

    def _load_logo(self):
        path = pick_file("Choose a logo / badge image")
        if not path:
            return
        logo, name = load_logo(path)
        if logo is None:
            self._set_status([self.wm_line or "no watermark yet",
                              f"could not load: {Path(path).name}"])
            return
        self.logo, self.logo_name = logo, name
        self._refresh_logo()
        self.render()
        if self.auto:
            self.run_attacks()

    def _clear_logo(self):
        if self.logo is None:
            return
        self.logo = self.logo_name = None
        self._refresh_logo()
        self.render()
        if self.auto:
            self.run_attacks()

    # rendering
    def _show(self, ax, img, title, gray=False, color="black"):
        ax.clear()
        ax.axis("off")
        if img is None:
            ax.imshow(np.full((10, 10, 3), 200, dtype=np.uint8))
        else:
            box = ax.get_window_extent()
            small = img.copy()
            small.thumbnail((max(16, int(box.width)), max(16, int(box.height))), Image.BILINEAR)
            if gray:
                ax.imshow(small, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
            else:
                ax.imshow(small, interpolation="nearest")
        ax.set_title(title, fontsize=8, color=color)

    def _set_status(self, lines):
        self.status.set_text("\n".join(lines + [self._logo_line()]))
        self.fig.canvas.draw_idle()

    def _show_watermark(self, elapsed_ms=None):
        show_mask = self.checks.get_status()[0] and self.mask is not None
        title = self.name + (" - mask" if show_mask else "")
        if elapsed_ms is not None:
            title += f" ({elapsed_ms:.0f} ms)"
        self._show(self.axes[1], self.mask if show_mask else self.marked, title, gray=show_mask)

    def render(self, measure=True):
        start = time.perf_counter()
        try:
            self.marked, self.mask = render_with_logo(
                self.name, self.image, *self.args, **self.extras,
            )
            self.wm_line = ""
        except Exception as e:
            self.marked = self.mask = None
            self.wm_line = f"watermark failed: {type(e).__name__}: {e}"
        self._show_watermark((time.perf_counter() - start) * 1000)
        self.measured = self.marked is None
        self.stale = True
        for ax, name in zip(self.axes[2:], atk.ATTACKS):
            ax.set_title(f"{name} - press Run attacks", fontsize=8, color="0.55")
        if measure:
            self.measure()
        else:
            self._set_status([self.wm_line or "let go of the slider to measure"])

    def measure(self):
        if not self.measured:
            self.wm_line = f"{'watermark vs original':<36}{format_metrics(compare(self.image, self.marked))}"
            self.measured = True
        self._set_status([self.wm_line])

    def run_attacks(self):
        if self.marked is None:
            return
        self.measure()
        self._set_status([self.wm_line, "running attacks..."])
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        text, opacity, rotation, amount = self.args
        ctx = dict(text=text, opacity=opacity, rotation=rotation, amount=amount, original=self.image)
        lines = [self.wm_line]
        for ax, (atk_name, atk_fn) in zip(self.axes[2:], atk.ATTACKS.items()):
            start = time.perf_counter()
            try:
                attacked = atk_fn(self.marked, ctx)
            except Exception as e:
                print(f"[{atk_name}] raised {type(e).__name__}: {e}")
                attacked = None
            elapsed = (time.perf_counter() - start) * 1000
            if attacked is None:
                self._show(ax, None, f"{atk_name} (failed)")
                lines.append(f"{atk_name[:24] + ' vs original':<36}failed")
                continue
            self._show(ax, attacked, f"{atk_name} ({elapsed:.0f} ms)")
            lines.append(f"{atk_name[:24] + ' vs original':<36}{format_metrics(compare(self.image, attacked))}")
        self.stale = False
        self._set_status(lines)

    def save(self):
        marked, mask = render_with_logo(self.name, self.full, *self.args, **self.extras)
        slug = "".join(c if c.isalnum() else "_" for c in self.name.lower())
        out = self.path.with_name(f"{self.path.stem}_{slug}.png")
        marked.save(out)
        mask.save(out.with_name(f"{out.stem}_mask.png"))
        print(f"Saved {out} and its mask")
        self._set_status([self.wm_line, f"saved {out.name} and {out.stem}_mask.png"])

    # widget callbacks
    def _changed(self, _value):
        if (time.perf_counter() - self._last_frame) * 1000 >= MIN_FRAME_MS:
            self._pending = False
            self.render(measure=False)
            self._last_frame = time.perf_counter()
        else:
            self._pending = True
            self._timer.stop()
            self._timer.start()

    def _flush(self):
        if self._pending:
            self._pending = False
            self.render(measure=False)
            self._last_frame = time.perf_counter()

    def _picked(self, label):
        self.name = label
        self._refresh_logo()
        self.render()

    def _text(self, text):
        if text != self.text:
            self.text = text
            self.render()
            if self.auto:
                self.run_attacks()

    def _toggled(self, label):
        if label == "Show mask":
            self._show_watermark()
            self.fig.canvas.draw_idle()
        elif self.auto and self.stale:
            self.run_attacks()

    def _released(self, event):
        if event.inaxes in self.axes:
            return
        self._flush()
        if not self.measured:
            self.measure()
        if self.auto and self.stale:
            self.run_attacks()

    def show(self):
        plt.show()


# ---------- CLI ----------

def parse_argv(argv):
    """Tiny parser: positional path plus --live/--batch/--logo PATH (or --logo=PATH)."""
    flags = {"live": False, "batch": False, "logo": None}
    positional = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--live":
            flags["live"] = True
        elif a == "--batch":
            flags["batch"] = True
        elif a == "--logo" and i + 1 < len(argv):
            flags["logo"] = argv[i + 1]; i += 1
        elif a.startswith("--logo="):
            flags["logo"] = a.split("=", 1)[1]
        elif not a.startswith("--"):
            positional.append(a)
        i += 1
    return positional, flags


def main():
    positional, flags = parse_argv(sys.argv[1:])
    path = positional[0] if positional else input("Enter image path: ").strip().strip('"')
    image = Image.open(path).convert("RGB")
    text = ask("Enter watermark: ", str, "")
    opacity = ask("Opacity 0-1 [0.8]: ", float, 0.8)
    rotation = ask("Rotation degrees [0]: ", float, 0.0)
    amount = ask("Amount [3]: ", int, 3)
    holes = ask("Holes 0-1, little holes in the watermark text [0]: ", float, 0.0)

    logo_path = flags["logo"] or ask("Optional logo/badge image (empty to skip): ", str, "")
    logo, logo_name = load_logo(logo_path)

    if flags["live"]:
        live = True
    elif flags["batch"]:
        live = False
    else:
        live = ask("Mode - [b]atch table or [l]ive editor [b]: ", str, "b").lower().startswith("l")

    if live:
        LiveEditor(image, path, text, opacity, rotation, amount, holes,
                   logo=logo, logo_name=logo_name).show()
    else:
        run_batch(image, text, opacity, rotation, amount, holes, pick_watermarkers(), logo=logo)


if __name__ == "__main__":
    main()