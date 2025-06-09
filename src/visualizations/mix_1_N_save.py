
import os
from torchvision import transforms
from PIL import Image
import torch
from torchvision.utils import save_image
import sys
sys.path.append('.')
from disentanglement.models import DisGAN, StyleGANSynthesis
import shutil



# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
def save_mixed_id(Generator, model):
    """
    Generates and saves reconstructed and combined images.
    
    Args:
    - Generator (nn.Module): The generator model for image reconstruction.
    - model (nn.Module): DisGAN model used for encoding and decoding images.
    - save_path (str): Directory path where the generated images will be saved.
    """

    first_half_path = 'datasets/celebahq/images/11350.jpg'
    image_paths = [
        'datasets/celebahq/images/10668.jpg', 'datasets/celebahq/images/10651.jpg',
        'datasets/celebahq/images/11283.jpg', 'datasets/celebahq/images/11217.jpg',
        'datasets/celebahq/images/10964.jpg'
    ]

    # Define image transformations
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    Generator.to('cuda')
    model.to('cuda')

    save_path = 'src/visualizations/thesis_visuals/mix_1_N_save'
    if os.path.exists(save_path):
        # Remove the directory and its contents if it exists
        shutil.rmtree(save_path)

    # Create the directory
    os.makedirs(save_path)
    # Load and process the first half image
    img_first_half = Image.open(first_half_path).convert('RGB')
    img_tensor_first_half = transform(img_first_half).unsqueeze(0).to('cuda')
    with torch.no_grad():
        w_plus_first_half, _ = model(img_tensor_first_half)
        reconstructed_first_half = Generator(w_plus_first_half)
        reconstructed_first_half = (reconstructed_first_half * 0.5 + 0.5).clamp(0, 1)  # Normalize to [0, 1]
        save_image(reconstructed_first_half.squeeze(), os.path.join(save_path, 'reconstructed_first_half.png'))

    # Process each second half image
    for i, img_path in enumerate(image_paths):
        img = Image.open(img_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to('cuda')
        with torch.no_grad():
            w_plus, _ = model(img_tensor)
            reconstructed_img = Generator(w_plus)
            reconstructed_img = (reconstructed_img * 0.5 + 0.5).clamp(0, 1)  # Normalize to [0, 1]
            save_image(reconstructed_img.squeeze(), os.path.join(save_path, f'reconstructed_{i}.png'))

            # Combine the first half of the fixed image with the second half of the current image
            half_latent_space_size = w_plus.size(1) // 2
            combined_w_star = torch.cat([
                w_plus_first_half[:, :half_latent_space_size], 
                w_plus[:, half_latent_space_size:]
            ], dim=1)
            combined_w_plus = model.inverse_T(combined_w_star)
            combined_image = Generator(combined_w_plus)
            combined_image = (combined_image * 0.5 + 0.5).clamp(0, 1)  # Normalize to [0, 1]
            save_image(combined_image.squeeze(), os.path.join(save_path, f'combined_image_{i}.png'))


def main():
    # Path to the trained DisGAN model
    trained_disgan_model_path = "output/random/model_T_2"

    # Load the trained DisGAN model
    disgan_model = torch.load(trained_disgan_model_path, map_location=device)
    disgan_model.to(device)

    if isinstance(disgan_model, torch.nn.DataParallel):
       disgan_model = disgan_model.module
    disgan_model.eval() 



    generator_model_dir = 'src/disentanglement/pretrained_models/stylegan3-r-ffhq-1024x1024.pkl'
    generator = StyleGANSynthesis(pretrained_model_dir=generator_model_dir).to(device)
    generator.eval()

    save_mixed_id(generator,disgan_model)


if __name__ == "__main__":
    main()
