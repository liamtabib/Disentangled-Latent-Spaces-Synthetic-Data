import os
import pickle
import torch
from torchvision.utils import save_image
import sys
sys.path.append(".")
from projects.disentanglement.src.models import StyleGANSynthesis  # Ensure this import matches your package structure

def generate_image_from_latent(pickle_path, generator_model_dir):
    # Load latent vectors from the pickle file
    with open(pickle_path, 'rb') as f:
        w_plus_latents = pickle.load(f).to(device)
    
    # Assuming the shape is [30000, 1, 16, 512], select the first latent vector
    first_latent_vector = w_plus_latents[7].squeeze(0)  # Adjust if necessary
    print(first_latent_vector.unsqueeze(0).shape)


    # Initialize StyleGAN Synthesis Model
    generator = StyleGANSynthesis(pretrained_model_dir=generator_model_dir).to(device)
    generator.eval()

    # Generate the image
    with torch.no_grad():
        generated_image = generator(first_latent_vector.unsqueeze(0))  # Add the batch dimension
    
    # Normalize and save the generated image
    output_path = os.path.join(os.path.dirname(pickle_path), 'generated_image.png')
    # Assuming the image tensor is in the range [-1, 1]
    save_image(generated_image * 0.5 + 0.5, output_path)
    
    print(f"Generated image saved to: {output_path}")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Update these paths
    pickle_path = 'projects/disentanglement/encoded_images/encoded_w_plus.pkl'
    generator_model_dir = 'projects/disentanglement/pretrained_models/stylegan3-r-ffhq-1024x1024.pkl'  # Update to where your StyleGAN3 model is stored

    generate_image_from_latent(pickle_path, generator_model_dir)
