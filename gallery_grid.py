# gallery_grid.py

from PIL import Image, ImageDraw, ImageFont
import numpy as np

# If you already have these, import from your modules instead:
from config import X_MIN, X_MAX, Y_MIN, Y_MAX, PALETTE
from image_generator import apply_palette  # your palette mapper

def render_fn_to_image(fn, w=600, h=200):
    """Evaluate a FunctionNode on a grid and return a PIL RGB image."""
    x = np.linspace(X_MIN, X_MAX, w)
    y = np.linspace(Y_MIN, Y_MAX, h)
    xx, yy = np.meshgrid(x, y)
    z = fn.evaluate(xx, yy)
    rgb = apply_palette(z, PALETTE)
    return Image.fromarray(rgb)

def compose_2x3_grid(images, cell_w=600, cell_h=200, margin=10, gap=10, bg=(18,18,18)):
    """Compose exactly 6 images into a 2×3 grid (2 across, 3 down)."""
    assert len(images) == 6, "Pass exactly 6 images."

    cols, rows = 2, 3
    canvas_w = 2*margin + cols*cell_w + (cols-1)*gap   # 1230 with 600/10
    canvas_h = 2*margin + rows*cell_h + (rows-1)*gap   # 640  with 200/10
    canvas = Image.new("RGB", (canvas_w, canvas_h), bg)

    positions = []
    k = 0
    for r in range(rows):
        for c in range(cols):
            x = margin + c*(cell_w + gap)
            y = margin + r*(cell_h + gap)
            canvas.paste(images[k], (x, y))
            positions.append((x, y))
            k += 1
    return canvas, positions, (canvas_w, canvas_h)

def annotate_numbers(canvas, positions, cell_w=600, cell_h=200):
    """Draw small number badges (1–6) on each cell."""
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("Arial.ttf", size=24)
    except:
        font = ImageFont.load_default()

    for idx, (x, y) in enumerate(positions, start=1):
        label = str(idx)
        bbox = draw.textbbox((0, 0), label, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        bx, by = x + 10, y + 10
        pad = 6
        draw.rounded_rectangle([bx-pad, by-pad, bx+tw+pad, by+th+pad],
                               radius=6, fill=(0, 0, 0, 140))
        draw.text(
            (x + 8, y + 8),   # small padding inside each cell
            label,
            font=font,
            fill=(255, 255, 255),
            stroke_fill=(0, 0, 0),
            stroke_width=3
        )
# ---- Example usage -------------------------------------------------
# fns = [fn1, fn2, fn3, fn4, fn5, fn6]  # build your six FunctionNodes
# imgs = [render_fn_to_image(fn) for fn in fns]
# canvas, positions, size = compose_2x3_grid(imgs, cell_w=600, cell_h=200, margin=10, gap=10)
# annotate_numbers(canvas, positions)
# canvas.show()  # or canvas.save("gallery.png")
