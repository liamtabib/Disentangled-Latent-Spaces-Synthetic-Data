import os
import pickle
import torch
from src.models import StyleGANSynthesis  # Adjust the import path as necessary
from torchvision.utils import save_image  # Ensure this is imported for image saving

def generate_image_with_perturbation(pickle_path, generator_model_path, std_dev=0.2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the pre-trained StyleGAN generator
    generator = StyleGANSynthesis(generator_model_path).to(device)
    generator.eval()

    # Load the encoded latent vectors
    with open(pickle_path, 'rb') as f:
        w_plus_latents = pickle.load(f).to(device)
    
    # Assuming w_plus_latents is [30000, 1, 16, 512] and we're using the first one
    first_latent_vector = w_plus_latents[10]  # This should now be [1, 16, 512]

    # Add a perturbation to the latent vector
    noise = torch.randn_like(first_latent_vector) * std_dev
    perturbed_latent_vector = first_latent_vector + noise
    perturbed_latent_vector = perturbed_latent_vector.unsqueeze(0)  # Add batch dimension

    # Generate image
    with torch.no_grad():
        generated_image = generator(perturbed_latent_vector)

    # Save the generated image
    output_dir = os.path.dirname(pickle_path)
    output_path = os.path.join(output_dir, 'generated_image_with_perturbation.png')

    # Assuming generated_image is in the range [-1, 1]
    generated_image = (generated_image + 1) / 2  # Normalize to [0, 1] for saving
    save_image(generated_image, output_path)

    print(f"Generated image with perturbation saved to {output_path}")

if __name__ == "__main__":
    pickle_path = 'projects/disentanglement/encoded_images/encoded_w_plus.pkl'  # Update this path
    generator_model_path = 'projects/disentanglement/pretrained_models/stylegan3-r-ffhq-1024x1024.pkl'  # Update this path
    generate_image_with_perturbation(pickle_path, generator_model_path)
