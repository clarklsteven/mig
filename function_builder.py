# function_builder.py
import random
from function_nodes import (
    Const, X, Y, Radial, Angle,
    Sin, Cos, Abs, Add, Sub, Mul, Threshold, 
    PolyNProximity, Noise, RelativeNoise, Offset, Starburst
)
from layers import Layer
import numpy as np

TERMINALS = [
    (X, 1),
    (Y, 1),
    (Radial, 1),
    (Angle, 0.2),
    (Const, 1.5),
    (PolyNProximity, 0.7),
    (Noise, 0.5)
]
UNARY = [
    (Sin, 2), 
    (Cos, 2),
    (Abs, 0.5),
    (Offset, 0.5),
    (RelativeNoise, 0.5),
    (Abs, 0.5)
]
BINARY = [Add, Sub, Mul]

def choose_terminal():
    funcs, weights = zip(*TERMINALS)
    return random.choices(funcs, weights=weights, k=1)[0]

def choose_unary():
    funcs, weights = zip(*UNARY)
    return random.choices(funcs, weights=weights, k=1)[0]

def random_terminal():
    choice = choose_terminal()
    if choice is Const:
        return Const(random.uniform(0.5, 10))
    elif choice in [Radial, Angle]:
        return choice(cx=random.uniform(-1, 1), cy=random.uniform(-1, 1))
    elif choice is Noise:
        return choice(amplitude=random.uniform(0.01, 0.2))
    elif choice is PolyNProximity:
        return random_polynomial_node(max_degree=4)
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
        op = choose_unary()
        operand = random_function(depth + 1, max_depth)
        if op is Threshold:
            return Threshold(operand, threshold=random.uniform(-1, 1))
        if op in (Sin, Cos):
            ox = random.uniform(-2.5, 2.5)
            oy = random.uniform(-1, 1)
            return Offset(operand, ox=ox, oy=oy)
        elif op is Offset:
            ox = random.uniform(-2.5, 2.5)
            oy = random.uniform(-1, 1)
            return Offset(operand, ox=ox, oy=oy)
        elif op is RelativeNoise:
            return RelativeNoise(operand, amplitude=random.uniform(0.01, 0.05))
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
            coeff_range = (-0.05, 0.05)
        elif power == 1:  # linear term
            coeff_range = (-0.03, 0.03)
        elif power == 2:  # quadratic term
            coeff_range = (-0.05, 0.05)
        else:  # cubic and higher
            coeff_range = (-0.01, 0.01)

        coeffs.append(random.uniform(*coeff_range))

    return PolyNProximity(coefficients=coeffs, falloff=falloff, proximity_scale=random.uniform(0.5, 2.0))

def random_sin():
    return RelativeNoise(
        Offset(
            Sin(Mul(Radial(),Const(random.uniform(0.5,15)))), 
            ox=random.uniform(-5, 5), oy=random.uniform(-5, 5)
        ), 
        amplitude=random.uniform(0.01, 0.1),
        seed=random.randint(0, 1000))

def random_combiner(fn1, fn2):
    op = random.choice(BINARY)
    if op is Add:
        return Add(fn1, fn2)
    elif op is Sub:
        return Sub(fn1, fn2)
    elif op is Mul:
        return Mul(fn1, fn2)
    else:
        raise ValueError(f"Unknown binary operation: {op}") 

def random_ripples():
    num_ripples = random.randint(1, 5)
    fn = random_sin()
    for _ in range(num_ripples-1):
        rr = random_sin()
        fn = random_combiner(fn, rr)

    return fn

def random_starburst():
    return Starburst(
        n_arms=random.randint(1, 7),
        phase=random.uniform(0, 2*np.pi),
        cx=random.uniform(-3, 3),
        cy=random.uniform(-1, 1),
        falloff_mode=random.choice(["poly", "exp"]),
        power=random.uniform(0.3, 3.0),
        scale=random.uniform(0.2, 1.2),
    )

def random_starburst_field():
    num_starbursts = random.randint(1, 5)
    fn = random_starburst()
    for _ in range(num_starbursts-1):
        rr = random_starburst()
        fn = random_combiner(fn, rr)    

    return fn