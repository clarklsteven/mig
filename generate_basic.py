from function_builder import (
    random_polynomial_node, 
    random_ripples, 
    random_combiner, 
    random_starburst_field)
from function_nodes import (
    Const, Offset, Mul, FunctionNode
)
from gallery_grid import (
    render_fn_to_image,
    compose_2x3_grid,
    annotate_numbers
)
import random
from user_interaction import select_images
from ga_ops import mutate_graph, crossover_graphs

def random_high_level_function():
    """Generate a random high-level function tree."""
    fn1 = random_ripples()
    fn2 = Mul(Offset(random_polynomial_node(max_degree=5), ox=random.uniform(-3, 3), oy=random.uniform(-1, 1)), Const(random.uniform(1, 10)))
    fn3 = Mul(Offset(random_polynomial_node(max_degree=5), ox=random.uniform(-3, 3), oy=random.uniform(-1, 1)), Const(random.uniform(1, 10)))
    fn4 = random_starburst_field()

    return random_combiner(
        random_combiner(
            random_combiner(fn1, fn2),
            fn3),
        fn4)

images = []  # start with an empty list
recipes = []  # to store generated recipes

for _ in range(6):
    fn = random_high_level_function()  # generate a random function tree
    recipes.append(fn.to_dict())  # save the function tree as a recipe
    img = render_fn_to_image(fn)      # from the grid code above
    images.append(img)                # push into the list

canvas, positions, size = compose_2x3_grid(images)
annotate_numbers(canvas, positions)
canvas.show()

selected_indices = select_images(6)  # let the user select images

continue_loop = True

while continue_loop:
    next_generation = []
    for idx in selected_indices:
        next_generation.append(recipes[idx])  # collect selected recipes for the next generation

    g1 = mutate_graph(next_generation[0])
    next_generation.append(g1)  # add the mutated graph to the next generation
    g2 = mutate_graph(next_generation[1])
    next_generation.append(g2)  # add the second mutated graph
    crossover_result1 = crossover_graphs(next_generation[0], next_generation[1])
    next_generation.append(crossover_result1)  # add the crossover result
    random_graph = random_high_level_function().to_dict()  # generate a random graph
    next_generation.append(random_graph)  # add the random graph to the next generation

    images = []  # reset images list for the next generation
    for recipe in next_generation:
        img = render_fn_to_image(FunctionNode.from_dict(recipe))  # render each recipe
        images.append(img)  # append the rendered image to the list

    canvas, positions, size = compose_2x3_grid(images)
    annotate_numbers(canvas, positions)
    canvas.show()  # display the grid of images

    selected_indices = select_images(6)  # let the user select images
    if not selected_indices:
        continue_loop = False
