# mig_gui.py

import tkinter as tk
from tkinter import ttk
from PIL import ImageTk, Image

from function_builder import random_layer_stack
from layers import blend_layers

WIDTH, HEIGHT = 512, 512

class MIGApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Mathematical Image Generator")

        # Canvas for image display
        self.canvas = tk.Canvas(root, width=WIDTH, height=HEIGHT)
        self.canvas.pack(pady=10)

        # Generate button
        self.generate_button = ttk.Button(root, text="Generate New Image", command=self.generate_image)
        self.generate_button.pack()

        self.tk_image = None
        self.generate_image()

    def generate_image(self):
        # Generate new image
        layers = random_layer_stack()
        array = blend_layers(layers, width=WIDTH, height=HEIGHT)
        img = Image.fromarray(array, mode='L')

        # Convert for Tkinter display
        self.tk_image = ImageTk.PhotoImage(img)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)

# Run the app
if __name__ == "__main__":
    root = tk.Tk()
    app = MIGApp(root)
    root.mainloop()
