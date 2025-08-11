# function_nodes.py
import numpy as np
from config import X_MIN, X_MAX, Y_MIN, Y_MAX

# function_nodes.py (top-level)
NODE_REGISTRY = {}

def register_node(cls):
    NODE_REGISTRY[cls.__name__] = cls
    return cls

class FunctionNode:
    def evaluate(self, x, y):
        raise NotImplementedError

    # --- Serialization ---
    def to_dict(self):
        raise NotImplementedError

    @classmethod
    def from_dict(cls, data):
        t = data["type"]
        node_cls = NODE_REGISTRY[t]
        return node_cls._from_dict(data)   # each subclass implements _from_dict

    def walk_paths(self, path=()):
        out = [(path, self)]
        for child_name, child in self.get_children():
            out.extend(child.walk_paths(path + (child_name,)))
        return out

    def get_children(self):
        """Return a list of (name, child_node) pairs. Override in subclasses."""
        return []

@register_node
class Const(FunctionNode):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return f"Const({self.value})"
    
    def evaluate(self, x, y):
        return np.full_like(x, self.value)
    
    def to_dict(self):
        return {"type": "Const", "value": self.value}

    @classmethod
    def _from_dict(cls, d):
        return cls(d["value"])

@register_node
class X(FunctionNode):
    def evaluate(self, x, y):
        return x
    
    def __str__(self):
        return "X"
    
    def to_dict(self): return {"type":"X"}

    @classmethod
    def _from_dict(cls, d): 
        return cls()

@register_node
class Y(FunctionNode):
    def evaluate(self, x, y):
        return y
    
    def __str__(self):
        return "Y"
    
    def to_dict(self): return {"type":"Y"}

    @classmethod
    def _from_dict(cls, d): 
        return cls()

@register_node
class Radial(FunctionNode):
    def __init__(self, cx=0, cy=0):
        self.cx = cx
        self.cy = cy
    
    def __str__(self):
        return f"Radial(cx={self.cx}, cy={self.cy})"

    def evaluate(self, x, y):
        return np.sqrt((x - self.cx)**2 + (y - self.cy)**2)
    
    def to_dict(self):
        return {"type":"Radial", "cx": self.cx, "cy": self.cy}
    
    @classmethod
    def _from_dict(cls, d):
        return cls(d.get("cx", 0), d.get("cy", 0))

@register_node
class Sin(FunctionNode):
    def __init__(self, operand):
        self.operand = operand
    
    def __str__(self):
        return f"Sin({self.operand})"

    def evaluate(self, x, y):
        return np.sin(self.operand.evaluate(x, y))
    
    def to_dict(self):
        return {"type":"Sin", "operand": self.operand.to_dict()}
    
    @classmethod
    def _from_dict(cls, d):
        return cls(FunctionNode.from_dict(d["operand"]))
    
    def get_children(self):
        return [("operand", self.operand)]

@register_node
class Cos(FunctionNode):
    def __init__(self, operand):
        self.operand = operand
    
    def __str__(self):
        return f"Cos({self.operand})"

    def evaluate(self, x, y):
        return np.cos(self.operand.evaluate(x, y))
    
    def to_dict(self):
        return {"type":"Cos", "operand": self.operand.to_dict()}
    
    @classmethod
    def _from_dict(cls, d):
        return cls(FunctionNode.from_dict(d["operand"]))
    
    def get_children(self):
        return [("operand", self.operand)]

@register_node
class Mul(FunctionNode):
    def __init__(self, left, right):
        self.left = left
        self.right = right
    
    def __str__(self):
        return f"Mul({self.left}, {self.right})"

    def evaluate(self, x, y):
        return self.left.evaluate(x, y) * self.right.evaluate(x, y)
    def to_dict(self):
        return {"type":"Mul", "left": self.left.to_dict(), "right": self.right.to_dict()}
    @classmethod
    def _from_dict(cls, d):
        return cls(FunctionNode.from_dict(d["left"]), FunctionNode.from_dict(d["right"]))

    def get_children(self):
        return [("left", self.left), ("right", self.right)]

@register_node
class Add(FunctionNode):
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def __str__(self):
        return f"Add({self.left}, {self.right})"

    def evaluate(self, x, y):
        return self.left.evaluate(x, y) + self.right.evaluate(x, y)
    def to_dict(self):
        return {"type":"Add", "left": self.left.to_dict(), "right": self.right.to_dict()}
    @classmethod
    def _from_dict(cls, d):
        return cls(FunctionNode.from_dict(d["left"]), FunctionNode.from_dict(d["right"]))

    def get_children(self):
        return [("left", self.left), ("right", self.right)]

@register_node
class Sub(FunctionNode):
    def __init__(self, left, right):
        self.left = left
        self.right = right
    
    def __str__(self):
        return f"Sub({self.left}, {self.right})"

    def evaluate(self, x, y):
        return self.left.evaluate(x, y) - self.right.evaluate(x, y)
    def to_dict(self):
        return {"type":"Sub", "left": self.left.to_dict(), "right": self.right.to_dict()}
    @classmethod
    def _from_dict(cls, d):
        return cls(FunctionNode.from_dict(d["left"]), FunctionNode.from_dict(d["right"]))

    def get_children(self):
        return [("left", self.left), ("right", self.right)]

@register_node
class Abs(FunctionNode):
    def __init__(self, operand):
        self.operand = operand
    
    def __str__(self):
        return f"Abs({self.operand})"

    def evaluate(self, x, y):
        return np.abs(self.operand.evaluate(x, y))
    
    def to_dict(self):
        return {"type":"Abs", "operand": self.operand.to_dict()}
    
    @classmethod
    def _from_dict(cls, d):
        return cls(FunctionNode.from_dict(d["operand"]))
    
    def get_children(self):
        return [("operand", self.operand)]

@register_node
class Angle(FunctionNode):
    def __init__(self, cx=0, cy=0):
        self.cx = cx
        self.cy = cy
    
    def __str__(self):
        return f"Angle(cx={self.cx}, cy={self.cy})"

    def evaluate(self, x, y):
        return np.arctan2(y - self.cy, x - self.cx)
    
    def to_dict(self):
        return {"type":"Angle", "cx": self.cx, "cy": self.cy}
    
    @classmethod
    def _from_dict(cls, d):
        return cls(d.get("cx", 0), d.get("cy", 0))

@register_node
class Threshold(FunctionNode):
    def __init__(self, operand, threshold=0.5):
        self.operand = operand
        self.threshold = threshold

    def __str__(self):
        return f"Threshold({self.operand}, threshold={self.threshold})"

    def evaluate(self, x, y):
        return (self.operand.evaluate(x, y) > self.threshold).astype(float)
    
    def to_dict(self):
        return {"type":"Threshold", "operand": self.operand.to_dict(), "threshold": self.threshold}
    
    @classmethod
    def _from_dict(cls, d):
        return cls(FunctionNode.from_dict(d["operand"]), d.get("threshold", 0.5))
    
    def get_children(self):
        return [("operand", self.operand)]
    
@register_node
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
    
    def to_dict(self):
        return {
            "type":"PolyNProximity",
            "coefficients": self.coefficients,
            "proximity_scale": self.proximity_scale,
            "falloff": self.falloff,
            "decay_rate": self.decay_rate
        }
    @classmethod
    def _from_dict(cls, d):
        return cls(
            coefficients=[float(c) for c in d["coefficients"]],
            proximity_scale=float(d["proximity_scale"]),
            falloff=d["falloff"],
            decay_rate=float(d["decay_rate"])
    )

@register_node
class Offset(FunctionNode):
    def __init__(self, fn, ox=0.0, oy=0.0):
        self.fn = fn
        self.ox = ox
        self.oy = oy

    def evaluate(self, x, y):
        return self.fn.evaluate(x - self.ox, y - self.oy)

    def __str__(self):
        return f"Offset({self.fn}, ox={self.ox:.2f}, oy={self.oy:.2f})"
    
    def to_dict(self):
        return {"type":"Offset","ox":self.ox,"oy":self.oy,"child": self.fn.to_dict()}
    
    @classmethod
    def _from_dict(cls, d):
        return cls(FunctionNode.from_dict(d["child"]), d["ox"], d["oy"])
    
    def get_children(self):
        return [("child", self.fn)]
    
@register_node
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
    
    def to_dict(self):
        return {"type":"Noise","amplitude":self.amplitude,"seed":self.seed}
    
    @classmethod
    def _from_dict(cls, d): 
        return cls(d.get("amplitude",0.1), d.get("seed"))
    
@register_node
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
    
    def to_dict(self):
        return {"type":"RelativeNoise", "fn": self.fn.to_dict(), "amplitude": self.amplitude, "seed": self.seed}
    
    @classmethod
    def _from_dict(cls, d):
        return cls(FunctionNode.from_dict(d["fn"]), d.get("amplitude", 0.1), d.get("seed"))
    
    def get_children(self):
        return [("fn", self.fn)]
    
@register_node
class Starburst(FunctionNode):
    """
    sin(n * angle + phase) * falloff(r)
    where falloff(r) is polynomial or exponential.
    """
    def __init__(self, n_arms=8, phase=0.0, cx=0.0, cy=0.0,
                 falloff_mode="poly", power=2.0, scale=1.0):
        self.n_arms = n_arms
        self.phase = phase
        self.cx = cx
        self.cy = cy
        self.falloff_mode = falloff_mode  # "poly" or "exp"
        self.power = power                # 0.5..5 suggested
        self.scale = max(1e-6, scale)     # avoid div/0

    def evaluate(self, x, y):
        # shift to centre
        xs = x - self.cx
        ys = y - self.cy
        theta = np.arctan2(ys, xs)
        r = np.sqrt(xs*xs + ys*ys)

        # angular wave
        angular = np.sin(self.n_arms * theta + self.phase)

        # radial falloff
        rp = (r / self.scale) ** self.power
        if self.falloff_mode == "exp":
            env = np.exp(-rp)
        else:  # polynomial-like
            env = 1.0 / (1.0 + rp)

        return angular * env

    def __str__(self):
        return (f"Starburst(n={self.n_arms}, phase={self.phase:.2f}, "
                f"cx={self.cx:.2f}, cy={self.cy:.2f}, mode={self.falloff_mode}, "
                f"power={self.power:.2f}, scale={self.scale:.2f})")
    
    def to_dict(self):
        return {
            "type": "Starburst",
            "n_arms": self.n_arms,
            "phase": self.phase,
            "cx": self.cx,
            "cy": self.cy,
            "falloff_mode": self.falloff_mode,
            "power": self.power,
            "scale": self.scale
        }
    
    @classmethod
    def _from_dict(cls, d):
        return cls(
            n_arms=d.get("n_arms", 8),
            phase=d.get("phase", 0.0),
            cx=d.get("cx", 0.0),
            cy=d.get("cy", 0.0),
            falloff_mode=d.get("falloff_mode", "poly"),
            power=d.get("power", 2.0),
            scale=d.get("scale", 1.0)
        )

@register_node
class Twist(FunctionNode):
    """
    Warp coordinates in polar space: θ' = θ + amount * (r/scale)^power
    Evaluate child at (r, θ') and map back to (x,y).
    """
    def __init__(self, fn, amount=0.6, power=1.0, cx=0.0, cy=0.0, scale=1.0):
        self.fn = fn
        self.amount = amount
        self.power = power
        self.cx, self.cy = cx, cy
        self.scale = max(1e-6, scale)

    def evaluate(self, x, y):
        xs, ys = x - self.cx, y - self.cy
        r = np.sqrt(xs*xs + ys*ys)
        theta = np.arctan2(ys, xs)
        theta2 = theta + self.amount * (r / self.scale) ** self.power
        x2 = self.cx + r * np.cos(theta2)
        y2 = self.cy + r * np.sin(theta2)
        return self.fn.evaluate(x2, y2)

    def __str__(self):
        return f"Twist({self.fn}, amount={self.amount}, power={self.power}, cx={self.cx}, cy={self.cy}, scale={self.scale})"
    
    def to_dict(self):
        return {
            "type": "Twist",
            "fn": self.fn.to_dict(),
            "amount": self.amount,
            "power": self.power,
            "cx": self.cx,
            "cy": self.cy,
            "scale": self.scale
        }
    
    @classmethod
    def _from_dict(cls, d):
        return cls(
            FunctionNode.from_dict(d["fn"]),
            d.get("amount", 0.6),
            d.get("power", 1.0),
            d.get("cx", 0.0),
            d.get("cy", 0.0),
            d.get("scale", 1.0)
        )
    
    def get_children(self):
        return [("fn", self.fn)]

@register_node
class RadialAmplitude(FunctionNode):
    """
    Multiply child by a radial sinusoid envelope.
    amp = bias + (1-bias) * 0.5*(1 + sin(freq * r + phase))
    """
    def __init__(self, fn, freq=8.0, phase=0.0, cx=0.0, cy=0.0, scale=1.0, bias=0.0):
        self.fn = fn
        self.freq = freq
        self.phase = phase
        self.cx, self.cy = cx, cy
        self.scale = max(1e-6, scale)
        self.bias = bias  # 0..1 (how much base level passes through)

    def evaluate(self, x, y):
        r = np.sqrt((x - self.cx)**2 + (y - self.cy)**2) / self.scale
        amp = self.bias + (1.0 - self.bias) * 0.5 * (1.0 + np.sin(self.freq * r + self.phase))
        return self.fn.evaluate(x, y) * amp

    def __str__(self):
        return f"RadialAmplitude({self.fn}, freq={self.freq}, phase={self.phase}, cx={self.cx}, cy={self.cy}, scale={self.scale}, bias={self.bias})"
    
    def to_dict(self):
        return {
            "type": "RadialAmplitude",
            "fn": self.fn.to_dict(),
            "freq": self.freq,
            "phase": self.phase,
            "cx": self.cx,
            "cy": self.cy,
            "scale": self.scale,
            "bias": self.bias
        }
    
    @classmethod
    def _from_dict(cls, d):
        return cls(
            FunctionNode.from_dict(d["fn"]),
            d.get("freq", 8.0),
            d.get("phase", 0.0),
            d.get("cx", 0.0),
            d.get("cy", 0.0),
            d.get("scale", 1.0),
            d.get("bias", 0.0)
        )
    
    def get_children(self):
        return [("fn", self.fn)]

@register_node
class PolynomialDistance(FunctionNode):
    """
    Distance to polynomial y = f(x) defined by coefficients.
    Vectorised 'closest point along x-curve' via broadcasting.
    """
    def __init__(self, coefficients):
        self.coefficients = coefficients

    def f(self, t):
        val = 0.0
        for power, coeff in enumerate(reversed(self.coefficients)):
            val += coeff * t**power
        return val

    def evaluate(self, x, y):
        # Precompute curve samples aligned with pixel x
        x_curve = x[0, :]                     # (W,)
        y_curve = self.f(x_curve)             # (W,)
        dx = x[..., None] - x_curve[None, None, :]  # (H,W,W)
        dy = y[..., None] - y_curve[None, None, :]
        distances = np.sqrt(dx*dx + dy*dy)          # (H,W,W)
        return np.min(distances, axis=2)            # (H,W)

    def __str__(self):
        terms = [f"{c:.2f}x^{p}" for p, c in enumerate(reversed(self.coefficients))]
        return f"PolynomialDistance({', '.join(terms)})"
    
    def to_dict(self):
        return {
            "type": "PolynomialDistance",
            "coefficients": self.coefficients
        }
    
    @classmethod
    def _from_dict(cls, d):
        return cls(d["coefficients"]) 

@register_node
class RippleFromDistance(FunctionNode):
    """
    Turn a distance field into ripples with optional decay from the curve.
    value = sin(freq * D + phase) * envelope(D)
    """
    def __init__(self, dist_fn, freq=12.0, phase=0.0, mode="poly", power=1.5, scale=0.4):
        self.dist_fn = dist_fn
        self.freq = freq
        self.phase = phase
        self.mode = mode     # "poly" or "exp"
        self.power = power
        self.scale = max(1e-6, scale)

    def evaluate(self, x, y):
        D = self.dist_fn.evaluate(x, y)  # (H,W)
        rip = np.sin(self.freq * D + self.phase)
        u = (D / self.scale) ** self.power
        env = np.exp(-u) if self.mode == "exp" else 1.0 / (1.0 + u)
        return rip * env

    def __str__(self):
        return f"RippleFromDistance({self.dist_fn}, freq={self.freq}, phase={self.phase}, mode={self.mode}, power={self.power}, scale={self.scale})"
    
    def to_dict(self):
        return {
            "type": "RippleFromDistance",
            "dist_fn": self.dist_fn.to_dict(),
            "freq": self.freq,
            "phase": self.phase,
            "mode": self.mode,
            "power": self.power,
            "scale": self.scale
        }
    
    @classmethod
    def _from_dict(cls, d):
        return cls(
            FunctionNode.from_dict(d["dist_fn"]),
            d.get("freq", 12.0),
            d.get("phase", 0.0),
            d.get("mode", "poly"),
            d.get("power", 1.5),
            d.get("scale", 0.4)
        )
    
    def get_children(self):
        return [("dist_fn", self.dist_fn)]
