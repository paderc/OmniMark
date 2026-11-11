from PIL import ImageFont

def _font(size):
    """Best available TrueType font at `size` px, with a sized default as fallback."""
    size = max(8, int(size))
    for name in ("arial.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _fonts(size):
    """List of fonts to cycle through; never empty."""
    size = max(8, int(size))
    fonts = []
    for name in ("arial.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf",
                 "AGENCYR.TTF", "BAUHS93.TTF", "CASTELAR.TTF"):
        try:
            fonts.append(ImageFont.truetype(name, size))
        except OSError:
            continue
    return fonts or [_font(size)]


def _auto_font_size(w, h, text, amount):
    """Calculate max font size to fit text in a cell, using actual font metrics."""
    cell_w, cell_h = w / amount, h / amount
    if not text:
        return int(cell_h)
        
    # Use a reference size to measure actual text bounding box
    ref_size = 100
    font = _font(ref_size)
    bbox = font.getbbox(text)
    tw = max(1, bbox[2] - bbox[0])
    th = max(1, bbox[3] - bbox[1])
    
    # Target 80% of the cell to leave room for rotation/margins
    target_w, target_h = cell_w * 0.9, cell_h * 0.9
    
    # Scale reference size to fit both width and height
    scale = min(target_w / tw, target_h / th)
    return max(8, int(ref_size * scale))