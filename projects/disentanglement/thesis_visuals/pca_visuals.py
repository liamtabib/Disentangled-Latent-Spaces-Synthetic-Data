import torch
import os
from torchvision import transforms
from PIL import Image
from torchvision.utils import save_image
from sklearn.decomposition import PCA
import sys
sys.path.append('.')
from projects.disentanglement.src.models import DisGAN, StyleGANSynthesis
import shutil
import numpy as np
from tqdm import tqdm
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_font(size=40):
    # Modify the path according to your operating system and the font availability
    font_paths = {
        'linux': '../../../../../usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf'
    }
    # Assuming the script is running on Linux
    font_path = font_paths['linux']
    try:
        font = ImageFont.truetype(font_path, size)
    except IOError:
        print("Font path is incorrect or font is not available. Using default font.")
        font = ImageFont.load_default()  # Fallback to default if specific font fails
    return font





def pca_with_perturbation(Generator, model, encoded_images, n_components=3, scales=[-2, -1.33, -0.67, 0, 0.67, 1.33, 2]):
    Generator.to(device)
    if isinstance(model, torch.nn.DataParallel):
        model = model.module    
    # Flatten the encoded images for PCA
    encoded_images = encoded_images.view(encoded_images.size(0), -1)
    half_feature_size = encoded_images.shape[-1] // 2  # Assuming the last dimension is the feature dimension

    first_half = encoded_images[:, :half_feature_size]
    # Perform PCA
    pca = PCA(n_components=n_components)
    pca.fit(first_half)
    principal_components = pca.components_  # Shape: [n_components, flattened_image_size]
    eigenvalues = pca.explained_variance_  # Shape: [n_components]

    save_path = 'projects/disentanglement/thesis_visuals/pca_images'
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path)

    encoded_images_tensor = torch.tensor(encoded_images, dtype=torch.float32, device=device)

    for i in range(n_components):
        # Get the i-th principal component
        principal_component = torch.tensor(principal_components[i], dtype=torch.float32, device=device)
        std_dev = torch.sqrt(torch.tensor(eigenvalues[i], device=device))

        for scale in scales:
            # Scale the principal component
            scaled_component = scale * std_dev * principal_component
            
            # Modify each latent vector by the scaled component
            modified_latent_vector = encoded_images_tensor[10].clone()  # Use .clone() to ensure we're not modifying the original
            modified_latent_vector[:half_feature_size] += scaled_component
            modified_latent_vector_reshaped = modified_latent_vector.reshape(1, 16, 512).to(device)
            modified_latent_vector_reshaped = model.inverse_T(modified_latent_vector_reshaped)

            # Generate the image
            with torch.no_grad():
                generated_image = Generator(modified_latent_vector_reshaped)

            generated_image = (generated_image + 1) / 2  # Normalize to [0, 1]

            # Convert tensor to PIL Image for drawing
            image_tensor = generated_image.cpu().squeeze(0).permute(1, 2, 0).numpy()
            image = Image.fromarray((image_tensor * 255).astype('uint8'))
            draw = ImageDraw.Draw(image)
            font = get_font(80)  # Use the custom function to get the font
            text = f'{scale} * σ'
            draw.text((10, 940), text, font=font, fill=(255, 255, 255))

            # Convert back to tensor
            image_tensor = np.array(image) / 255.0
            image_tensor = torch.tensor(image_tensor).permute(2, 0, 1)
            image_tensor = image_tensor.unsqueeze(0)
            filename = f'ID_pc_{i}_scale_{scale:.2f}.png'
            save_image(generated_image.squeeze(), os.path.join(save_path, filename))


    second_half = encoded_images[:,half_feature_size:]
    # Perform PCA
    pca = PCA(n_components=n_components)
    pca.fit(second_half)
    principal_components = pca.components_  # Shape: [n_components, flattened_image_size]
    eigenvalues = pca.explained_variance_  # Shape: [n_components]

    for i in range(n_components):
        # Get the i-th principal component
        principal_component = torch.tensor(principal_components[i], dtype=torch.float32, device=device)
        std_dev = torch.sqrt(torch.tensor(eigenvalues[i], device=device))

        for scale in scales:
            # Scale the principal component
            scaled_component = scale * std_dev * principal_component
            
            # Modify each latent vector by the scaled component
            modified_latent_vector = encoded_images_tensor[10].clone()  # Use .clone() to ensure we're not modifying the original
            modified_latent_vector[half_feature_size: ] += scaled_component
            modified_latent_vector_reshaped = modified_latent_vector.reshape(1, 16, 512).to(device)
            modified_latent_vector_reshaped = model.inverse_T(modified_latent_vector_reshaped)

            # Generate the image
            with torch.no_grad():
                generated_image = Generator(modified_latent_vector_reshaped)

            # Normalize the generated image to [0, 1]
            generated_image = (generated_image + 1) / 2  # Normalize to [0, 1]

            # Convert tensor to PIL Image for drawing
            image_tensor = generated_image.cpu().squeeze(0).permute(1, 2, 0).numpy()
            image = Image.fromarray((image_tensor * 255).astype('uint8'))
            draw = ImageDraw.Draw(image)
            font = get_font(80)  # Use the custom function to get the font
            text = f'{scale} * σ'
            draw.text((10, 940), text, font=font, fill=(255, 255, 255))

            # Convert back to tensor
            image_tensor = np.array(image) / 255.0
            image_tensor = torch.tensor(image_tensor).permute(2, 0, 1)
            image_tensor = image_tensor.unsqueeze(0)

            filename = f'NID_pc_{i}_scale_{scale:.2f}.png'
            save_image(generated_image.squeeze(), os.path.join(save_path, filename))
    
    print(f"images saved to {save_path}")



def encode_dataset(model, limit=30000):
    """
    Encodes all images from a specified directory into the latent space W^* by passing them through a DisGAN model.
    
    This function iterates over all images up to a specified limit in the dataset directory, applies necessary
    preprocessing transformations, and then encodes them using the provided DisGAN model. The latent representations
    (W^*) and their corresponding identity IDs are stored in tensors and returned. This function is primarily used
    for preparing data for subsequent deep learning tasks that require pre-encoded features.

    Parameters:
    - model (torch.nn.Module): The DisGAN model used for encoding the images into latent space.
    - limit (int): The maximum number of images to process. Default is 30000.

    Returns:
    - encoded_images_tensor (torch.Tensor): A tensor containing the encoded images in the latent space W^*.
    - identity_ids_tensor (torch.Tensor): A tensor containing identity IDs corresponding to each image.

    Note:
    - The function assumes the presence of 'identity_ID.csv' that maps image filenames to identity IDs.
    - Images are resized to 256x256 pixels and normalized as part of preprocessing before encoding.
    - The directory containing the images is hardcoded as 'datasets/celebahq/images'.
    """

    # Set the device for computation based on CUDA availability.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define the transformations for preprocessing the images.
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize images to 256x256 pixels.
        transforms.ToTensor(),  # Convert images to tensor format.
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # Normalize images.
    ])

    # Paths for the image directory and identity CSV.
    image_dir = 'datasets/celebahq/images'
    identity_csv_path = 'datasets/celebahq/identity_ID.csv'

    # Load the identity mappings from CSV.
    identity_df = pd.read_csv(identity_csv_path)
    # Adjust filenames to match the image file format.
    identity_df['orig_file'] = identity_df['idx'].apply(lambda x: str(x) + '.jpg')
    identity_dict = pd.Series(identity_df.identity_ID.values, index=identity_df.orig_file).to_dict()

    # Collect image files.
    img_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg') or f.endswith('.png')]
    img_files_sorted = sorted(img_files, key=lambda x: int(os.path.splitext(x)[0]))

    # Lists to hold encoded images and identity IDs.
    encoded_images = []
    identity_ids = []
    i = 0
    
    # Process each image up to the specified limit.
    for img_file in tqdm(img_files_sorted, desc="Encoding images"):
        if i == limit:
            break

        img_path = os.path.join(image_dir, img_file)
        img = Image.open(img_path).convert('RGB')
        img = transform(img)
        img = img.unsqueeze(0).to(device)

        # Encode the image using the DisGAN model.
        with torch.no_grad():
            _, w_hat = model(img)
            encoded_images.append(w_hat.cpu().numpy())
        
        # Retrieve and store the identity ID from the dictionary.
        identity_id = identity_dict.get(img_file, -1)
        identity_ids.append(identity_id)
        i += 1

    # Convert lists to tensors.
    encoded_images_tensor = torch.tensor(encoded_images).squeeze(1)
    identity_ids_tensor = torch.tensor(identity_ids)

    return encoded_images_tensor, identity_ids_tensor


def main():

    trained_disgan_model_path = "output/random/model_T_2"

    # Load the trained DisGAN model
    disgan_model = torch.load(trained_disgan_model_path, map_location=device)
    disgan_model.to(device)

    if isinstance(disgan_model, torch.nn.DataParallel):
       disgan_model = disgan_model.module
    disgan_model.eval() 

    encoded_images_tensor,_ = encode_dataset(disgan_model,30000)

    # Example usage
    generator_model_dir = 'projects/disentanglement/pretrained_models/stylegan3-r-ffhq-1024x1024.pkl'
    generator = StyleGANSynthesis(pretrained_model_dir=generator_model_dir).to(device)
    generator.eval()

    pca_with_perturbation(generator, disgan_model,encoded_images_tensor)

if __name__ == "__main__":
    main()



