import torch
from torchvision import transforms
from PIL import Image
import pickle
import os
from tqdm import tqdm
import numpy as np
import time

# Assuming DisGAN is defined in src.models
from src.models import DisGAN

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    # Path to the trained DisGAN model
    trained_disgan_model_path = "output/2024-02-16_12-19-39/model_T_12"

    # Load the trained DisGAN model
    disgan_model = torch.load(trained_disgan_model_path, map_location=device)
    disgan_model.to(device)
    disgan_model.eval()

    # Define the image transform
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    # Directory containing the images
    image_dir = 'datasets/celebahq/images'
    encoded_images_w_star = []

    # Process each image in the directory
    for img_file in tqdm(os.listdir(image_dir), desc="Encoding Images"):
        if img_file.endswith('.jpg') or img_file.endswith('.png'):
            img_path = os.path.join(image_dir, img_file)
            img = Image.open(img_path).convert('RGB')
            img = transform(img).unsqueeze(0).to(device)

            # Use DisGAN to encode the image to w_plus and transform to w_star in one step
            with torch.no_grad():
                _, encoded_img_w_star = disgan_model(img)
                encoded_images_w_star.append(encoded_img_w_star.cpu().numpy())

    # Save the encoded w_star images
    directory = 'projects/disentanglement/encoded_images'
    if not os.path.exists(directory):
        os.makedirs(directory)
    w_star_path = os.path.join(directory, 'encoded_w_star.pkl')
    with open(w_star_path, 'wb') as f:
        # Convert list to a numpy array before saving
        encoded_images_w_star_array = np.array(encoded_images_w_star)
        pickle.dump(encoded_images_w_star_array, f)

    print(f"Encoded w_star images saved to {w_star_path}")

if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    print(f"Elapsed time: {elapsed_time / 60:.2f} minutes")
