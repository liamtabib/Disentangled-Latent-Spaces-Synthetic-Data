import pandas as pd
from PIL import Image
from torchvision.utils import save_image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import torch
import os
from collections import defaultdict
import random
from torchvision.utils import make_grid, save_image

def load_random_identities_images(identity_csv_path, images_dir, num_identities=7, images_per_identity=7, grid_save_path="identity_grid.jpg"):
    # Step 1: Read the CSV to create a mapping from identity to image file names
    df = pd.read_csv(identity_csv_path)
    # Adjust file names without zero padding and filter identities with at least 7 images
    identity_to_images = df.groupby('identity_ID')['idx'].apply(lambda x: x.astype(str).apply(lambda y: f"{int(y)}.jpg").tolist()).to_dict()

    # Filter out identities with fewer than images_per_identity images
    valid_identities = {identity: images for identity, images in identity_to_images.items() if len(images) >= images_per_identity}

    # Randomly select num_identities identities from those that have enough images
    selected_identities = random.sample(list(valid_identities.keys()), num_identities)
    
    # Transformations to be applied to each image
    transform = Compose([
        Resize((256, 256)),
        ToTensor(),
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    grid_images = []

    # Step 3: Load and prepare images
    for identity in selected_identities:
        images = valid_identities[identity][:images_per_identity]  # Take the first 7 images
        for image_name in images:
            image_path = os.path.join(images_dir, image_name)
            try:
                with Image.open(image_path).convert("RGB") as img:
                    img_tensor = transform(img)
                    grid_images.append(img_tensor.unsqueeze(0))  # Add batch dimension
            except FileNotFoundError as e:
                print(f"File not found: {e}")
                continue
    
    # Step 4: Generate the grid
    images_tensor = torch.cat(grid_images, dim=0)
    grid = make_grid(images_tensor, nrow=images_per_identity)
    
    # Step 5: Save the grid to a file
    save_image(grid, grid_save_path, normalize=True, value_range=(-1, 1))
    print(f"Saved grid of randomly selected {num_identities} identities to {grid_save_path}")

# Assuming you have these paths set correctly
identity_csv_path = "datasets/celebahq/identity_ID.csv"
images_dir = "datasets/celebahq/images"
grid_save_path = "identity_grid_random.jpg"

# Call the function to generate and save the grid
load_random_identities_images(identity_csv_path, images_dir, grid_save_path=grid_save_path)
