# recipes.py
import json, hashlib
from dataclasses import dataclass, asdict
from typing import Any, Dict, List
from function_nodes import FunctionNode
import numpy as np
from PIL import Image
from image_generator import apply_palette

@dataclass
class RenderSettings:
    width: int
    height: int
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    palette: List[List[int]]  # [[r,g,b],...]
    seed: int | None = None

@dataclass
class Recipe:
    schema: int
    graph: Dict[str, Any]     # function tree dict (node.to_dict())
    settings: RenderSettings
    notes: str = ""

    def to_json(self) -> str:
        # canonicalize for stable hashes
        return json.dumps({
            "schema": self.schema,
            "graph": self.graph,
            "settings": asdict(self.settings),
            "notes": self.notes
        }, separators=(",", ":"), sort_keys=True)

    def fingerprint(self) -> str:
        return hashlib.sha1(self.to_json().encode("utf-8")).hexdigest()[:10]

def save_recipe(recipe: Recipe, path: str):
    with open(path, "w") as f:
        f.write(recipe.to_json())

def load_recipe(path: str) -> Recipe:
    from function_nodes import FunctionNode  # ensure registry is loaded
    with open(path) as f:
        data = json.load(f)
    rs = data["settings"]
    settings = RenderSettings(
        width=rs["width"], height=rs["height"],
        x_min=rs["x_min"], x_max=rs["x_max"], y_min=rs["y_min"], y_max=rs["y_max"],
        palette=rs["palette"], seed=rs.get("seed")
    )
    return Recipe(schema=data["schema"], graph=data["graph"], settings=settings, notes=data.get("notes",""))

def render_recipe(recipe: Recipe) -> Image.Image:
    print(recipe.graph)
    from function_nodes import FunctionNode  # ensure registry is present
    fn = FunctionNode.from_dict(recipe.graph)

    w, h = recipe.settings.width, recipe.settings.height
    x = np.linspace(recipe.settings.x_min, recipe.settings.x_max, w)
    y = np.linspace(recipe.settings.y_min, recipe.settings.y_max, h)
    xx, yy = np.meshgrid(x, y)
    z = fn.evaluate(xx, yy)

    rgb = apply_palette(z, [tuple(c) for c in recipe.settings.palette])
    return Image.fromarray(rgb)
