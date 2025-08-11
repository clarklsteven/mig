# ga_ops.py
import copy, json, random
from typing import Any, Dict, List, Tuple, Callable
from function_nodes import FunctionNode, NODE_REGISTRY, register_node

# ---- Tree utilities -------------------------------------------------

def deep_copy(obj):
    # robust copy for plain-json structures
    return json.loads(json.dumps(obj))

def walk_paths(node_dict: Dict[str, Any], path=()):
    fn_node = FunctionNode.from_dict(node_dict)
    print(fn_node)  # debug: show the function node structure
    return fn_node.walk_paths(path)

def get_at(root: Dict[str, Any], path: Tuple[str, ...]) -> Dict[str, Any]:
    cur = root
    for k in path:
        cur = cur[k]
    return cur

def set_at(root: Dict[str, Any], path: Tuple[str, ...], new_subtree: Dict[str, Any]) -> None:
    if not path:
        # replace root
        root.clear()
        root.update(new_subtree)
        return
    *parent_path, last = path
    parent = get_at(root, tuple(parent_path))
    parent[last] = new_subtree

# ---- Operator groups (safe swaps) -----------------------------------

UNARY_SIMPLE = ("Sin", "Cos", "Abs")              # same schema: {"type", "operand"}
BINARY = ("Add", "Sub", "Mul")                    # {"type","left","right"}

# Things with extra params (handle carefully; disable by default)
UNARY_PARAMETRIC = ("Offset", "RadialAmplitude", "Twist", "Threshold")

TERMINALS = (
    "Const", "X", "Y", "Radial", "Angle",
    "Noise", "PolyNProximity", "Starburst", "PolynomialDistance"
    # "RelativeNoise" and "RippleFromDistance" are unary wrappers in your codebase
)

# ---- Mutation helpers -----------------------------------------------

def jitter(value: float, rel_scale=0.15, abs_floor=1e-9, positive=False) -> float:
    """Gaussian jitter proportional to magnitude (15% by default)."""
    base = float(value)
    sigma = max(abs(base) * rel_scale, 1e-3)
    new = base + random.gauss(0, sigma)
    if positive:
        return max(abs_floor, new)
    return new

def mutate_numeric_fields(node: Dict[str, Any]) -> None:
    """Jitter numeric parameters in-place (gentle)."""
    t = node.get("type")
    if t == "Const":
        node["value"] = jitter(node["value"])
    elif t == "Offset":
        node["ox"] = jitter(node["ox"], rel_scale=0.25)
        node["oy"] = jitter(node["oy"], rel_scale=0.25)
    elif t == "Threshold":
        node["threshold"] = jitter(node["threshold"], rel_scale=0.3)
    elif t == "PolyNProximity":
        node["proximity_scale"] = jitter(node.get("proximity_scale", 1.2), rel_scale=0.25, positive=True)
        node["decay_rate"] = jitter(node.get("decay_rate", 2.0), rel_scale=0.25, positive=True)
        node["coefficients"] = [jitter(c, rel_scale=0.2) for c in node["coefficients"]]
    elif t == "Noise":
        node["amplitude"] = jitter(node.get("amplitude", 0.1), rel_scale=0.5, positive=True)
    elif t == "RelativeNoise":
        node["amplitude"] = jitter(node.get("amplitude", 0.1), rel_scale=0.5, positive=True)
    elif t == "Starburst":
        # n_arms is usually int; mutate softly and clamp
        n = max(3, int(round(jitter(node.get("n_arms", 8), rel_scale=0.2))))
        node["n_arms"] = n
        node["phase"] = jitter(node.get("phase", 0.0), rel_scale=0.1)
        node["cx"] = jitter(node.get("cx", 0.0), rel_scale=0.3)
        node["cy"] = jitter(node.get("cy", 0.0), rel_scale=0.3)
        node["power"] = max(0.1, jitter(node.get("power", 2.0), rel_scale=0.25))
        node["scale"] = max(1e-3, jitter(node.get("scale", 1.0), rel_scale=0.25))
    elif t == "RadialAmplitude":
        node["freq"] = max(0.1, jitter(node.get("freq", 8.0), rel_scale=0.25))
        node["phase"] = jitter(node.get("phase", 0.0), rel_scale=0.1)
        node["cx"] = jitter(node.get("cx", 0.0), rel_scale=0.3)
        node["cy"] = jitter(node.get("cy", 0.0), rel_scale=0.3)
        node["scale"] = max(1e-3, jitter(node.get("scale", 1.0), rel_scale=0.25))
        node["bias"]  = max(0.0, min(1.0, jitter(node.get("bias", 0.0), rel_scale=0.5)))
    elif t == "Twist":
        node["amount"] = jitter(node.get("amount", 0.5), rel_scale=0.25)
        node["power"]  = max(0.1, jitter(node.get("power", 1.0), rel_scale=0.25))
        node["cx"] = jitter(node.get("cx", 0.0), rel_scale=0.3)
        node["cy"] = jitter(node.get("cy", 0.0), rel_scale=0.3)
        node["scale"] = max(1e-3, jitter(node.get("scale", 1.0), rel_scale=0.25))
    elif t == "PolynomialDistance":
        node["coefficients"] = [jitter(c, rel_scale=0.2) for c in node["coefficients"]]
    elif t == "RippleFromDistance":
        node["freq"]  = max(0.1, jitter(node.get("freq", 12.0), rel_scale=0.25))
        node["phase"] = jitter(node.get("phase", 0.0), rel_scale=0.1)
        node["power"] = max(0.1, jitter(node.get("power", 1.5), rel_scale=0.25))
        node["scale"] = max(1e-3, jitter(node.get("scale", 0.4), rel_scale=0.25))

def maybe_swap_operator(node: Dict[str, Any], p_swap=0.25) -> None:
    """Swap operator within safe groups with probability p_swap."""
    if random.random() > p_swap: 
        return
    t = node.get("type")
    if t in UNARY_SIMPLE:
        node["type"] = random.choice([op for op in UNARY_SIMPLE if op != t])
    elif t in BINARY:
        node["type"] = random.choice([op for op in BINARY if op != t])
    # We avoid swapping parametric-unary & terminals by default (different schemas).
    # You can add rules later if you want, but this keeps things safe.

def mutate_graph(graph: Dict[str, Any],
                 p_point_mut=0.30,
                 p_op_swap=0.20,
                 rng: random.Random | None = None) -> Dict[str, Any]:
    """Return a mutated copy of the graph (dict)."""
    if rng is None:
        rng = random
    g = deep_copy(graph)

    paths = walk_paths(g)
    print(paths)  # debug: show all paths in the graph
    # choose a few points to mutate
    k = max(1, int(len(paths) * p_point_mut))
    print(f"Mutating {k} points in the graph with {len(paths)} total paths.")
    targets = [p for p, _ in rng.sample(paths, k)]
    print("Selected paths for mutation:", targets)

    for path in targets:
        node = get_at(g, path)
        # numeric tweaks
        mutate_numeric_fields(node)
        # operator swap (safe groups only)
        maybe_swap_operator(node, p_swap=p_op_swap)
    return g

# ---- Crossover ------------------------------------------------------

def crossover_graphs(a: Dict[str, Any], b: Dict[str, Any], rng: random.Random | None = None) -> Dict[str, Any]:
    """Single-point subtree crossover: replace a random subtree in A with one from B."""
    if rng is None:
        rng = random
    A = deep_copy(a)
    B = deep_copy(b)

    paths_a = [p for p, _ in walk_paths(A)]
    paths_b = [p for p, _ in walk_paths(B)]

    if not paths_a or not paths_b:
        return A  # fallback

    pa = rng.choice(paths_a)
    pb = rng.choice(paths_b)

    sub_b = deep_copy(get_at(B, pb))
    set_at(A, pa, sub_b)
    return A
