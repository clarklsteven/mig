from function_builder import (
    random_layer_stack, 
    random_polynomial_node, 
    random_function, 
    random_ripples, 
    random_combiner, 
    random_starburst, 
    random_starburst_field)
from function_nodes import (
    Const, X, Y, Radial, Angle, Offset, Noise, RelativeNoise,
    Sin, Cos, Abs, Add, Sub, Mul, Threshold, PolyNProximity, 
    Starburst, Twist, RadialAmplitude, PolynomialDistance, RippleFromDistance
)
from gallery_grid import (
    render_fn_to_image,
    compose_2x3_grid,
    annotate_numbers
)
from layers import blend_layers, Layer
from PIL import Image
from image_generator import apply_palette, render_rgb_from_layers, generate_image
from config import PALETTE, IMAGE_HEIGHT, IMAGE_WIDTH, X_MIN, X_MAX, Y_MIN, Y_MAX
import numpy as np
import random


#fn = PolyProximity(a=0.1, b=0, c=0, falloff='inverse')
#fn = random_polynomial_node(max_degree=3)
#fn = Mul(PolyNProximity([0.5, -0.3, 0.2], proximity_scale=1.5),Sin(Radial(0.5)))
#fn = Sub(Mul(PolyNProximity([0.2, 0.1, -0.3, 0.05], proximity_scale=0.5), Const(20)), Sin(Mul(Angle(), Const(8))))
#fn = Mul(PolyNProximity([0.4, -0.5, 0.25], proximity_scale=2.0),X())
#fn = Add(PolyNProximity([0.3, -0.6, 0.2], proximity_scale=0.1),PolyNProximity([-0.2, 0.4, -0.1, 0.05], proximity_scale=0.1))
#fn = Sin(Mul(PolyNProximity([0.1, -0.3, 0.2], proximity_scale=0.1),Const(1)))
#fn = Mul(PolyNProximity([0.3, 0.2, -0.1], proximity_scale=1.6),Mul(Sin(Mul(Radial(), Const(5))),Cos(Mul(Angle(), Const(7)))))
#fn = Add(Sub(Mul(
#    Offset(Cos(Mul(Radial(), Const(1))), ox=-0.5, oy=0.5),
#    Offset(Cos(Mul(Radial(), Const(8))), ox=2.5, oy=0.5)),
#    Offset(Cos(Mul(Radial(), Const(5))), ox=-1.5, oy=-0.5)),
#    RelativeNoise(Mul(PolyNProximity([0.3, 0.2, -0.1], proximity_scale=1.2),Const(10)),0.1))

#fn = Add(
#    Mul(
#        RelativeNoise(Offset(Sin(Mul(Radial(),Const(6))),ox=-0.5, oy=3.5),amplitude=0.1),
#        Add(Offset(Cos(Mul(Radial(),Const(7))),ox=-0.5, oy=0.5), Offset(Sin(Mul(Radial(),Const(2))), ox=0.5, oy=-0.5))
#        ),
#    Sub(Mul(PolyNProximity([0.1, -0.2, -0.1, 0.5], proximity_scale=0.7), Const(2.5)),
#        Mul(PolyNProximity([0.06, 0.3, -1.1], proximity_scale=0.7), Const(3.5))
#    )
#)

#images = []  # start with an empty list

#for _ in range(6):
#    fn1 = random_ripples()
#    fn2 = Mul(Offset(random_polynomial_node(max_degree=5), ox=random.uniform(-3, 3), oy=random.uniform(-1, 1)),Const(random.uniform(1, 10)))
#    fn3 = Mul(Offset(random_polynomial_node(max_degree=5), ox=random.uniform(-3, 3), oy=random.uniform(-1, 1)),Const(random.uniform(1, 10)))
#    fn4 = random_starburst_field()

#    fn = random_combiner(
#        random_combiner(
#            random_combiner(fn1, fn2),
#            fn3),
#        fn4)
#    img = render_fn_to_image(fn)      # from the grid code above
#    images.append(img)                # push into the list

#canvas, positions, size = compose_2x3_grid(images)
#annotate_numbers(canvas, positions)
#canvas.show()
#fn = RadialAmplitude(Starburst(n_arms=5, phase=1, falloff_mode="exp", power=1, scale=0.5), scale=0.5, bias=0.5)

#base = Add(random_ripples(), PolyNProximity([0.25, -0.15, 0.05], proximity_scale=1.3))
#fn = Twist(base, amount=0.5, power=1.1, scale=0.9)

#x = np.linspace(X_MIN, X_MAX, IMAGE_WIDTH)
#y = np.linspace(Y_MIN, Y_MAX, IMAGE_HEIGHT)
#xx, yy = np.meshgrid(x, y)
#z = fn.evaluate(xx, yy)
#rgb_array = apply_palette(z, PALETTE)

#img = Image.fromarray(rgb_array)

#img.show()

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


from recipes import Recipe, RenderSettings, save_recipe, render_recipe
from config import PALETTE, X_MIN, X_MAX, Y_MIN, Y_MAX
# build your function tree as usual:
fn = Add( Offset(Cos(Mul(Radial(), Const(7))), ox=0.4, oy=-0.2),
          PolyNProximity([0.3, 0.2, -0.1], proximity_scale=1.3) )

recipe = Recipe(
    schema=1,
    graph=fn.to_dict(),
    settings=RenderSettings(
        width=1200, height=400,
        x_min=X_MIN, x_max=X_MAX, y_min=Y_MIN, y_max=Y_MAX,
        palette=[list(c) for c in PALETTE],
        seed=None
    ),
    notes="Nice interference + curve"
)

fname = f"recipe_{recipe.fingerprint()}.json"
save_recipe(recipe, fname)

img = render_recipe(recipe)
img.save(f"render_{recipe.fingerprint()}.png")