# layers.py

from image_generator import generate_image
import numpy as np
from config import X_MIN, X_MAX, Y_MIN, Y_MAX

class Layer:
    def __init__(self, function_node, blend_mode='add', weight=1.0):
        self.function_node = function_node
        self.blend_mode = blend_mode
        self.weight = weight
    
    def __str__(self):
        return str(self.function_node)

    def render(self, width, height, xlim=(X_MIN, X_MAX), ylim=(Y_MIN, Y_MAX)):
        img = generate_image(self.function_node, width, height, xlim, ylim)
        return np.array(img, dtype=float) * self.weight
    
    def function(self):
        return self.function_node

    def evaluate(self, width: int, height: int) -> np.ndarray:
        x = np.linspace(-1, 1, width)
        y = np.linspace(-1, 1, height)
        xx, yy = np.meshgrid(x, y)

        return self.function_node.evaluate(xx, yy)

def blend_layers(layers, width=512, height=512, xlim=(X_MIN, X_MAX), ylim=(Y_MIN, Y_MAX)):
    base = np.zeros((height, width), dtype=float)

    for layer in layers:
        rendered = layer.render(width, height, xlim, ylim)
        if layer.blend_mode == 'add':
            base += rendered
        elif layer.blend_mode == 'subtract':
            base -= rendered
        elif layer.blend_mode == 'multiply':
            base *= rendered / 255.0  # normalize before multiply

    # Normalise the result to 0–255
    min_val = base.min()
    max_val = base.max()

    if max_val - min_val < 1e-5:
        # Prevent division by near-zero — treat as flat image
        norm = np.zeros_like(base, dtype=np.uint8)
    else:
        norm = ((base - min_val) / (max_val - min_val) * 255).astype(np.uint8)

    return norm

