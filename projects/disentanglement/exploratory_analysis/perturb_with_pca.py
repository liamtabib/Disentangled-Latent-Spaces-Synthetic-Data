import pickle
import torch
import os
from src.models import StyleGANSynthesis  # Adjust import path as necessary
from torchvision.utils import save_image

def add_first_principal_component_and_generate_image(latent_pickle_path, pca_components_path, generator_model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the latent vectors
    with open(latent_pickle_path, 'rb') as f:
        latent_vectors = pickle.load(f).to(device)
    
    # Load the PCA principal components
    with open(pca_components_path, 'rb') as f:
        pca_principal_components = pickle.load(f)
    
    # Assuming latent_vectors is [30000, 1, 16, 512] and pca_principal_components is [10, 8192]
    # Reshape the first latent vector to [8192]
    first_latent_vector = latent_vectors[10].squeeze(1).reshape(-1)

    # Get the first principal component
    first_principal_component = torch.tensor(pca_principal_components[3], dtype=torch.float32, device=device)
    print(first_principal_component)
    # Add the first principal component to the first latent vector
    modified_latent_vector = first_latent_vector + 500*first_principal_component

    # Reshape back to [1, 16, 512] for the generator
    modified_latent_vector_reshaped = modified_latent_vector.reshape(1, 16, 512)

    # Initialize the StyleGAN generator
    generator = StyleGANSynthesis(generator_model_path).to(device)
    generator.eval()

    # Generate the image using the modified latent vector
    with torch.no_grad():
        generated_image = generator(modified_latent_vector_reshaped)

    # Save the generated image
    output_dir = os.path.dirname(latent_pickle_path)
    output_path = os.path.join(output_dir, 'generated_image_with_first_pca_component.png')

    # Normalize and save the image assuming output in [-1, 1]
    generated_image = (generated_image + 1) / 2  # Normalize to [0, 1]
    save_image(generated_image[0], output_path)  # save_image expects no batch dimension

    print(f"Generated image with the first PCA component added saved to {output_path}")

if __name__ == "__main__":
    latent_pickle_path = 'projects/disentanglement/encoded_images/encoded_w_plus.pkl'  # Update this path
    pca_components_path = 'projects/disentanglement/encoded_images/latent_pca_components.pkl'  # Update this path
    generator_model_path = 'projects/disentanglement/pretrained_models/stylegan3-r-ffhq-1024x1024.pkl'  # Update this path
    add_first_principal_component_and_generate_image(latent_pickle_path, pca_components_path, generator_model_path)
