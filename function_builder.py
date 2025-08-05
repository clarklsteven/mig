# function_builder.py
import random
from function_nodes import (
    Const, X, Y, Radial, Angle,
    Sin, Cos, Abs, Add, Sub, Mul, Threshold, PolyNProximity
)
from layers import Layer

TERMINALS = [X, Y, Radial, Angle, Const]
UNARY = [Sin, Cos, Abs, Threshold]
BINARY = [Add, Sub, Mul]

def random_terminal():
    choice = random.choice(TERMINALS)
    if choice is Const:
        return Const(random.uniform(0.5, 10))
    elif choice in [Radial, Angle]:
        return choice(cx=0, cy=0)  # could randomise these later
    else:
        return choice()

def random_function(depth=0, max_depth=4):
    if depth >= max_depth:
        return random_terminal()

    node_type = random.choices(
        population=['terminal', 'unary', 'binary'],
        weights=[1, 2, 3],  # favour binary ops slightly
        k=1
    )[0]

    if node_type == 'terminal':
        return random_terminal()

    elif node_type == 'unary':
        op = random.choice(UNARY)
        operand = random_function(depth + 1, max_depth)
        if op is Threshold:
            return Threshold(operand, threshold=random.uniform(-1, 1))
        else:
            return op(operand)

    elif node_type == 'binary':
        op = random.choice(BINARY)
        left = random_function(depth + 1, max_depth)
        right = random_function(depth + 1, max_depth)
        return op(left, right)

def random_layer_stack(
    min_layers=2, max_layers=5,
    max_function_depth=6,
    blend_modes=('add', 'subtract', 'multiply')
):
    num_layers = random.randint(min_layers, max_layers)
    #num_layers = 3
    layers = []

    for _ in range(num_layers):
        fn = random_function(max_depth=random.randint(2, max_function_depth))
        blend = random.choice(blend_modes)
        weight = random.uniform(0.3, 1.2)
        layers.append(Layer(fn, blend_mode=blend, weight=weight))

    return layers

def random_polynomial_node(max_degree=3, falloff='inverse') -> PolyNProximity:
    degree = random.randint(1, max_degree)
    coeffs = []

    for power in range(degree + 1):
        if power == 0:  # constant term
            coeff_range = (-5, 5)
        elif power == 1:  # linear term
            coeff_range = (-3, 3)
        elif power == 2:  # quadratic term
            coeff_range = (-1, 1)
        else:  # cubic and higher
            coeff_range = (-0.25, 0.25)

        coeffs.append(random.uniform(*coeff_range))
        print(f"Coefficient for x^{power}: {coeffs[-1]:.2f}")

    return PolyNProximity(coefficients=coeffs, falloff=falloff)