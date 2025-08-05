# function_nodes.py
import numpy as np
from config import X_MIN, X_MAX, Y_MIN, Y_MAX

class FunctionNode:
    def evaluate(self, x, y):
        raise NotImplementedError("Must implement in subclass")

class Const(FunctionNode):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return f"Const({self.value})"
    
    def evaluate(self, x, y):
        return np.full_like(x, self.value)

class X(FunctionNode):
    def evaluate(self, x, y):
        return x
    
    def __str__(self):
        return "X"

class Y(FunctionNode):
    def evaluate(self, x, y):
        return y
    
    def __str__(self):
        return "Y"

class Radial(FunctionNode):
    def __init__(self, cx=0, cy=0):
        self.cx = cx
        self.cy = cy
    
    def __str__(self):
        return f"Radial(cx={self.cx}, cy={self.cy})"

    def evaluate(self, x, y):
        return np.sqrt((x - self.cx)**2 + (y - self.cy)**2)

class Sin(FunctionNode):
    def __init__(self, operand):
        self.operand = operand
    
    def __str__(self):
        return f"Sin({self.operand})"

    def evaluate(self, x, y):
        return np.sin(self.operand.evaluate(x, y))

class Cos(FunctionNode):
    def __init__(self, operand):
        self.operand = operand
    
    def __str__(self):
        return f"Cos({self.operand})"

    def evaluate(self, x, y):
        return np.cos(self.operand.evaluate(x, y))

class Mul(FunctionNode):
    def __init__(self, left, right):
        self.left = left
        self.right = right
    
    def __str__(self):
        return f"Mul({self.left}, {self.right})"

    def evaluate(self, x, y):
        return self.left.evaluate(x, y) * self.right.evaluate(x, y)

class Add(FunctionNode):
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def __str__(self):
        return f"Add({self.left}, {self.right})"

    def evaluate(self, x, y):
        return self.left.evaluate(x, y) + self.right.evaluate(x, y)

class Sub(FunctionNode):
    def __init__(self, left, right):
        self.left = left
        self.right = right
    
    def __str__(self):
        return f"Sub({self.left}, {self.right})"

    def evaluate(self, x, y):
        return self.left.evaluate(x, y) - self.right.evaluate(x, y)

class Abs(FunctionNode):
    def __init__(self, operand):
        self.operand = operand
    
    def __str__(self):
        return f"Abs({self.operand})"

    def evaluate(self, x, y):
        return np.abs(self.operand.evaluate(x, y))

class Angle(FunctionNode):
    def __init__(self, cx=0, cy=0):
        self.cx = cx
        self.cy = cy
    
    def __str__(self):
        return f"Angle(cx={self.cx}, cy={self.cy})"

    def evaluate(self, x, y):
        return np.arctan2(y - self.cy, x - self.cx)

class Threshold(FunctionNode):
    def __init__(self, operand, threshold=0.5):
        self.operand = operand
        self.threshold = threshold

    def __str__(self):
        return f"Threshold({self.operand}, threshold={self.threshold})"

    def evaluate(self, x, y):
        return (self.operand.evaluate(x, y) > self.threshold).astype(float)
    
import numpy as np

class PolyNProximity(FunctionNode):
    def __init__(self, coefficients, falloff='inverse', proximity_scale=1.0, decay_rate=2.0, xlim=(X_MIN, X_MAX)):
        self.coefficients = coefficients
        self.falloff = falloff
        self.proximity_scale = proximity_scale
        self.decay_rate = decay_rate
        self.xlim = xlim  # domain for x

    def f(self, t):
        """Evaluate polynomial at t."""
        val = 0.0
        for power, coeff in enumerate(reversed(self.coefficients)):
            val += coeff * t**power
        return val

    def evaluate(self, x, y):
        height, width = x.shape

        # 1. Precompute curve points (one per image column)
        x_curve = np.linspace(self.xlim[0], self.xlim[1], width)
        y_curve = self.f(x_curve)

        # 2. Compute distances for every pixel to every curve point
        # Shape: (height, width, width)
        dx = x[..., np.newaxis] - x_curve[np.newaxis, np.newaxis, :]
        dy = y[..., np.newaxis] - y_curve[np.newaxis, np.newaxis, :]
        distances = np.sqrt(dx**2 + dy**2)

        # 3. Closest distance along the curve
        min_dist = np.min(distances, axis=2) / self.proximity_scale

        # 4. Apply falloff
        if self.falloff == 'inverse':
            return 1.0 / (1.0 + min_dist)
        elif self.falloff == 'exp':
            return np.exp(-min_dist * self.decay_rate)
        elif self.falloff == 'threshold':
            return (min_dist < 0.1).astype(float)
        else:
            return 1.0 / (1.0 + min_dist)

    def __str__(self):
        terms = [f"{c:.2f}x^{p}" for p, c in enumerate(reversed(self.coefficients))]
        return f"PolyNProximity({', '.join(terms)}, falloff={self.falloff})"

class Offset(FunctionNode):
    def __init__(self, fn, ox=0.0, oy=0.0):
        self.fn = fn
        self.ox = ox
        self.oy = oy

    def evaluate(self, x, y):
        return self.fn.evaluate(x - self.ox, y - self.oy)

    def __str__(self):
        return f"Offset({self.fn}, ox={self.ox:.2f}, oy={self.oy:.2f})"
    
class Noise(FunctionNode):
    def __init__(self, amplitude=0.1, relative=False, seed=None):
        self.amplitude = amplitude
        self.relative = relative
        self.seed = seed

    def evaluate(self, x, y):
        rng = np.random.default_rng(self.seed)
        base_noise = rng.uniform(-1, 1, size=x.shape)
        if self.relative:
            # Need to combine with an underlying function’s value
            raise NotImplementedError("Relative noise should wrap another function node")
        return self.amplitude * base_noise

    def __str__(self):
        return f"Noise(amplitude={self.amplitude}, relative={self.relative})"
    
class RelativeNoise(FunctionNode):
    def __init__(self, fn, amplitude=0.1, seed=None):
        self.fn = fn
        self.amplitude = amplitude
        self.seed = seed

    def evaluate(self, x, y):
        rng = np.random.default_rng(self.seed)
        base_val = self.fn.evaluate(x, y)
        noise = rng.uniform(-1, 1, size=x.shape)
        return base_val + (noise * self.amplitude * base_val)
    
    def __str__(self):
        return f"RelativeNoise({self.fn}, amplitude={self.amplitude})"