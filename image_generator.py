# image_generator.py
import numpy as np
from PIL import Image
from typing import List, Tuple
from config import X_MIN, X_MAX, Y_MIN, Y_MAX, IMAGE_WIDTH, IMAGE_HEIGHT


def generate_image(func, width=IMAGE_WIDTH, height=IMAGE_HEIGHT, xlim=(X_MIN, X_MAX), ylim=(Y_MIN, Y_MAX)):
    x = np.linspace(xlim[0], xlim[1], width)
    y = np.linspace(ylim[0], ylim[1], height)
    xx, yy = np.meshgrid(x, y)

    z = func.evaluate(xx, yy)

    # Normalise to 0–255
    z_norm = ((z - z.min()) / (z.max() - z.min()) * 255).astype(np.uint8)

    return Image.fromarray(z_norm, mode='L')

def apply_palette(z: np.ndarray, palette: list[tuple[int, int, int]]) -> np.ndarray:
    # Normalise z to 0..1
    z_min, z_max = z.min(), z.max()
    if z_max == z_min:
        z_norm = np.zeros_like(z)
    else:
        z_norm = (z - z_min) / (z_max - z_min)

    # Scale to palette index range
    scaled = z_norm * (len(palette) - 1)
    low_idx = np.floor(scaled).astype(int)
    high_idx = np.clip(low_idx + 1, 0, len(palette) - 1)
    frac = scaled - low_idx

    # Interpolate between palette colours
    rgb_array = np.zeros((*z.shape, 3), dtype=np.uint8)
    for c in range(3):  # R, G, B channels
        low_col = np.array([palette[i][c] for i in low_idx.flat]).reshape(z.shape)
        high_col = np.array([palette[i][c] for i in high_idx.flat]).reshape(z.shape)
        rgb_array[..., c] = (low_col * (1 - frac) + high_col * frac).astype(np.uint8)

    return rgb_array

def safe_normalize(z: np.ndarray) -> np.ndarray:
    z_min = z.min()
    z_max = z.max()
    if z_max == z_min:
        return np.zeros_like(z, dtype=np.uint8)  # or 128 for mid-gray
    z_norm = ((z - z_min) / (z_max - z_min) * 255).astype(np.uint8)
    return z_norm

def render_rgb_from_layers(layers: list, width=IMAGE_WIDTH, height=IMAGE_HEIGHT) -> Image.Image:
    assert len(layers) == 3, "You must pass exactly 3 layers for RGB output"

    rgb_array = np.zeros((height, width, 3), dtype=np.uint8)

    for i, layer in enumerate(layers):
        z = layer.evaluate(width, height)
        rgb_array[..., i] = safe_normalize(z)

    return Image.fromarray(rgb_array, mode='RGB')