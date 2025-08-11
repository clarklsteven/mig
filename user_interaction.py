def select_images(num_images: int) -> list[int]:
    """
    Prompt the user to select images by number (1-based).
    Returns a list of selected indices (0-based).
    """
    while True:
        raw = input(f"Select images to keep (1-{num_images}, space or comma separated): ")
        # Allow spaces or commas
        parts = raw.replace(",", " ").split()
        try:
            indices = sorted(set(int(p) - 1 for p in parts))
        except ValueError:
            print("Invalid input. Please enter numbers only.")
            continue

        if all(0 <= i < num_images for i in indices):
            return indices
        else:
            print(f"Invalid range. Please choose between 1 and {num_images}.")
