from function_builder import random_layer_stack, random_polynomial_node
from function_nodes import (
    Const, X, Y, Radial, Angle, Offset, Noise, RelativeNoise,
    Sin, Cos, Abs, Add, Sub, Mul, Threshold, PolyNProximity
)
from layers import blend_layers, Layer
from PIL import Image
from image_generator import apply_palette, render_rgb_from_layers, generate_image
from config import PALETTE, IMAGE_HEIGHT, IMAGE_WIDTH, X_MIN, X_MAX, Y_MIN, Y_MAX
import numpy as np


#fn = PolyProximity(a=0.1, b=0, c=0, falloff='inverse')
#fn = random_polynomial_node(max_degree=3)
#fn = Mul(PolyNProximity([0.5, -0.3, 0.2], proximity_scale=1.5),Sin(Radial(0.5)))
#fn = Sub(Mul(PolyNProximity([0.2, 0.1, -0.3, 0.05], proximity_scale=0.5), Const(20)), Sin(Mul(Angle(), Const(8))))
#fn = Mul(PolyNProximity([0.4, -0.5, 0.25], proximity_scale=2.0),X())
#fn = Add(PolyNProximity([0.3, -0.6, 0.2], proximity_scale=0.1),PolyNProximity([-0.2, 0.4, -0.1, 0.05], proximity_scale=0.1))
#fn = Sin(Mul(PolyNProximity([0.1, -0.3, 0.2], proximity_scale=0.1),Const(1)))
#fn = Mul(PolyNProximity([0.3, 0.2, -0.1], proximity_scale=1.6),Mul(Sin(Mul(Radial(), Const(5))),Cos(Mul(Angle(), Const(7)))))
fn = Add(Sub(Mul(
    Offset(Cos(Mul(Radial(), Const(1))), ox=-0.5, oy=0.5),
    Offset(Cos(Mul(Radial(), Const(8))), ox=2.5, oy=0.5)),
    Offset(Cos(Mul(Radial(), Const(5))), ox=-1.5, oy=-0.5)),
    RelativeNoise(Mul(PolyNProximity([0.3, 0.2, -0.1], proximity_scale=1.2),Const(10)),0.1))

x = np.linspace(X_MIN, X_MAX, IMAGE_WIDTH)
y = np.linspace(Y_MIN, Y_MAX, IMAGE_HEIGHT)
xx, yy = np.meshgrid(x, y)

z = fn.evaluate(xx, yy)

rgb_array = apply_palette(z, PALETTE)

img = Image.fromarray(rgb_array)
img.show()

#img = generate_image(fn)

#layers = random_layer_stack()
#print("Generated Function Tree:")
#for i, layer in enumerate(layers):
#    print(f"Layer {i+1} function:")
#    print(layer)
#result_array = blend_layers(layers, width=512, height=512)
# Apply palette
#rgb_array = apply_palette(result_array, PALETTE)

#img = render_rgb_from_layers(layers, width=512, height=512)

#img = Image.fromarray(rgb_array, mode='RGB')
#img.save("random_layers.png")
#img.show()
#print("Image with random layers saved.")

