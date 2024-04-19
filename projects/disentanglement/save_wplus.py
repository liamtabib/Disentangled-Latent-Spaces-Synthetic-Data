import torch
from src.models import StyleGANEncoder
from torchvision import transforms
from PIL import Image
import pickle
import os
import pandas as pd
from tqdm import tqdm
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    """
    Main function to encode images using a pre-trained StyleGAN encoder and save both
    the encoded representations and their corresponding identity IDs, considering the
    filename mismatch due to leading zeros.
    """
    pretrained_encoder_dir = "projects/disentanglement/pretrained_models/restyle_pSp_ffhq.pt"
    encoder = StyleGANEncoder(pretrained_encoder_dir).to(device)
    encoder.eval()

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    image_dir = 'datasets/celebahq/images'
    identity_csv_path = 'datasets/celebahq/identity_ID.csv'

    # Load identity data, adjusting filename format
    identity_df = pd.read_csv(identity_csv_path)
    # Adjust filenames to match those in the image directory
    identity_df['orig_file'] = identity_df['idx'].apply(lambda x: str(x) +'.jpg')

    identity_dict = pd.Series(identity_df.identity_ID.values, index=identity_df.orig_file).to_dict()

    img_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg') or f.endswith('.png')]
    img_files_sorted = sorted(img_files, key=lambda x: int(os.path.splitext(x)[0]))

    encoded_images = []
    identity_ids = []

    for img_file in tqdm(img_files_sorted, desc="Encoding images"):
        img_path = os.path.join(image_dir, img_file)
        img = Image.open(img_path).convert('RGB')
        img = transform(img)
        img = img.unsqueeze(0).to(device)

        with torch.no_grad():
            encoded_img = encoder(img)
            encoded_images.append(encoded_img.cpu().numpy())
        
        # Here, we adjust the img_file format to match the CSV
        identity_id = identity_dict.get(img_file, -1)
        identity_ids.append(identity_id)

    encoded_images_tensor = torch.tensor(encoded_images).squeeze(1)
    identity_ids_tensor = torch.tensor(identity_ids)

    directory = 'projects/disentanglement/encoded_images'
    filename = 'encoded_w_plus_with_ids.pkl'
    output_pickle_path = os.path.join(directory, filename)

    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(output_pickle_path, 'wb') as f:
        pickle.dump({'encoded_images': encoded_images_tensor, 'identity_ids': identity_ids_tensor}, f)

    print(f"Encoded images and identity IDs saved to {output_pickle_path}")

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print("Elapsed time in minutes:", (end_time - start_time) / 60)
