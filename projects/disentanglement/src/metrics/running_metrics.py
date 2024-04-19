import torch
import numpy as np
from itertools import combinations
from tqdm import tqdm
import pandas as pd
import os
from torchvision import transforms
from PIL import Image
from torchvision.utils import save_image
from PIL import Image
from torchvision import transforms
import torch
from tqdm import tqdm
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
from itertools import combinations
import os
import torch
import torch.nn as nn
import numpy as np
import torch
import torch.nn as nn
import numpy as np
import random
from torchmetrics.image.fid import FrechetInceptionDistance
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class detection_FR_latent_space_distance(nn.Module):
    """
    Same function as FR_latent_space_distance but with a face cropping before"""

    def __init__(self, generator, face_detection, face_recognition, save_path):
        super(detection_FR_latent_space_distance, self).__init__()
        self.num_pairs = 2500
        self.set_seed()  
        self.generator = generator
        self.face_detection = face_detection
        self.face_recognition = face_recognition
        self.save_path = save_path

        with open(self.save_path, 'w') as file:
            file.write(f"Experiment Parameters:\nNumber of Pairs: {self.num_pairs}\n")
            file.write("epoch,FID,positives_distance, negatives_distance, Ratio\n")


    def set_seed(self):
            seed = 10
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)


    def select_pairs(self, identity_ids_tensor ):
        pairs = []
        indices = np.arange(len(identity_ids_tensor))
        np.random.shuffle(indices)

        for i in range(0, len(indices) - 1, 2):  # Step by 2 to pair adjacent elements and to ensure same image is not picked twice
            if len(pairs) >= self.num_pairs:
                break
            if identity_ids_tensor[indices[i]] != identity_ids_tensor[indices[i + 1]]:
                pairs.append((indices[i], indices[i + 1]))

        return pairs

    def forward(self, encoded_images_tensor,identity_ids_tensor ,model, epoch):

        if isinstance(model, torch.nn.DataParallel):
            model = model.module
        
        positives_distances = []
        negatives_distances = []

        pairs = self.select_pairs(identity_ids_tensor)

        fid = FrechetInceptionDistance(feature=64).to(device)


        for idx1, idx2 in tqdm(pairs, desc='Processing pairs'):

            w_star_i, w_star_j = encoded_images_tensor[idx1].unsqueeze(0).to(device), encoded_images_tensor[idx2].unsqueeze(0).to(device)

            w_star_i = w_star_i.view(w_star_i.size(0), -1)  # Reshapes to [batchsize, 8192]
            w_star_j = w_star_j.view(w_star_j.size(0), -1)  # Reshapes to [batchsize, 8192]


            half_latent_space_size = w_star_j.size(1) // 2

            w_star_i_identity = torch.cat([w_star_i[:, :half_latent_space_size],w_star_j[:, half_latent_space_size:]], dim=1).view(-1, 16, 512)
            w_star_j_identity = torch.cat([w_star_j[:, :half_latent_space_size],w_star_i[:, half_latent_space_size:]], dim=1).view(-1, 16, 512)
            w_star_i = w_star_i.view(-1, 16, 512)
            w_star_j = w_star_j.view(-1, 16, 512)

            # map to W^+
            w_plus_i = model.inverse_T(w_star_i)
            w_plus_j = model.inverse_T(w_star_j)
            w_plus_i_identity = model.inverse_T(w_star_i_identity)
            W_plus_j_identity = model.inverse_T(w_star_j_identity)

                       # map to I
            i_image = self.generator(w_plus_i)
            j_image = self.generator(w_plus_j)
            i_identity_image = self.generator(w_plus_i_identity)
            j_identity_image = self.generator(W_plus_j_identity)

            # Normalize I
            i_image = (i_image * 0.5 + 0.5).clamp(0, 1)
            j_image = (j_image * 0.5 + 0.5).clamp(0, 1)
            i_identity_image = (i_identity_image * 0.5 + 0.5).clamp(0, 1)
            j_identity_image = (j_identity_image * 0.5 + 0.5).clamp(0, 1)


            # Create a transform to convert tensors to PIL Images
            to_pil = transforms.ToPILImage()

            # Apply the transform to your images
            i_image_pil = to_pil(i_image.squeeze().cpu())
            j_image_pil = to_pil(j_image.squeeze().cpu())
            i_identity_image_pil = to_pil(i_identity_image.squeeze().cpu())
            j_identity_image_pil = to_pil(j_identity_image.squeeze().cpu())


            distance=self.get_distance(i_image_pil, i_identity_image_pil)
            if distance is not None:
                positives_distances.append(distance)

            distance = self.get_distance(j_image_pil, j_identity_image_pil)
            if distance is not None:
                positives_distances.append(distance)

            distance = self.get_distance(i_image_pil, j_identity_image_pil)
            if distance is not None:
                negatives_distances.append(distance)

            distance = self.get_distance(j_image_pil, i_identity_image_pil)
            if distance is not None:
                negatives_distances.append(distance)

            real_images = torch.cat([i_image,j_image], dim=0)
            fake_images = torch.cat([i_identity_image,j_identity_image], dim=0)

            fid.update(real_images.to(torch.uint8).to(device), real=True)
            fid.update(fake_images.to(torch.uint8).to(device), real=False)

            del real_images, fake_images
            torch.cuda.empty_cache()

        average_positive_distances = np.mean(positives_distances)
        average_negative_distances = np.mean(negatives_distances)

        
        fid_score = fid.compute().cpu()
          
        self.save_statistics(epoch, fid_score, average_positive_distances, average_negative_distances)
    

    def get_distance(self, img1, img2):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Ensure device is defined

        cropped_image1, prob1 = self.face_detection(img1, return_prob=True)
        cropped_image2, prob2 = self.face_detection(img2, return_prob=True)

        if cropped_image1 is None or cropped_image2 is None:
            print("Face detection failed for one or both images. Prediction cannot be made.")
            return None

        # Move cropped images to the selected device
        cropped_image1 = cropped_image1.to(device)
        cropped_image2 = cropped_image2.to(device)

        # Generate embeddings
        with torch.no_grad():
            embedding1 = self.face_recognition(cropped_image1.unsqueeze(0)).squeeze()  # Add and remove batch dimension
            embedding2 = self.face_recognition(cropped_image2.unsqueeze(0)).squeeze()

        # Compute cosine similarity between the two embeddings
        embedding1_norm = embedding1 / (embedding1.norm(p=2) + 1e-6)  # Normalize embeddings
        embedding2_norm = embedding2 / (embedding2.norm(p=2) + 1e-6)
        
        similarity = torch.dot(embedding1_norm, embedding2_norm).item()  # Compute dot product
        return similarity


    def save_statistics(self, epoch, fid_score, positives_distance, negatives_distance):
            ratio_distance = positives_distance / negatives_distance
            with open(self.save_path, 'a') as file:
                file.write(
                    f"{epoch},{fid_score:.3f},{positives_distance:.3f},{negatives_distance:.3f},{ratio_distance:.3f}\n"
                )






def combine_images_to_grid(Generator, model, save_path):
    """
    Generates a grid of images where the first row and column display original images,
    and each cell in the grid combines halves of latent vectors from two different images.
    The grid's dimension is dynamically determined based on the number of input images.
    
    Args:
    - Generator (nn.Module): The generator model for image reconstruction.
    - model (nn.Module): DisGAN model used for encoding and decoding images.
    - image_paths (list): List of paths to the input images.
    - save_path (str): Path where the generated grid image will be saved.
    """


    image_paths = [
        'datasets/celebahq/images/11350.jpg', 'datasets/celebahq/images/10668.jpg',
        'datasets/celebahq/images/10651.jpg', 'datasets/celebahq/images/11283.jpg',
        'datasets/celebahq/images/11217.jpg', 'datasets/celebahq/images/10964.jpg'
    ]

    # Handle case where the model is wrapped in DataParallel
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    Generator.to('cuda')
    model.to('cuda')

    # Define image transformations
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # Process each image to get its original reconstructed version
    original_images = []
    for image_path in image_paths:
        img = Image.open(image_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to('cuda')
        with torch.no_grad():
            # Encode and decode using DisGAN to get the reconstructed image
            w_plus, _ = model(img_tensor)
            reconstructed_img = Generator(w_plus)
            reconstructed_img = (reconstructed_img * 0.5 + 0.5).clamp(0, 1)
            original_images.append(reconstructed_img)

    # Determine the grid size: number of images + 1 for the header row/column
    grid_size = len(image_paths) + 1
    # Initialize grid with a placeholder for the top-left corner
    white_image = torch.ones_like(original_images[0])
    grid_images = []
    # Construct the grid row by row
    for i in range(len(image_paths)+1): #rows
        for j in range(len(image_paths)+1): #columns
            if i == j == 0:
                grid_images.append(white_image)
            elif i == 0 and j != 0:
                grid_images.append(original_images[j-1])
            elif i != 0 and j == 0:
                grid_images.append(original_images[i-1])
                # If the row and column indices are the same, use the original image
            elif i == j:
                grid_images.append(original_images[i-1])
            else:
                # Combine the first half of the i-th image's latent vector with the second half of the j-th image's
                with torch.no_grad():

                    _, w_star_i = model(transform(Image.open(image_paths[i-1]).convert('RGB')).unsqueeze(0).to('cuda'))
                    _, w_star_j = model(transform(Image.open(image_paths[j-1]).convert('RGB')).unsqueeze(0).to('cuda'))

                    w_star_i = w_star_i.view(w_star_i.size(0), -1)  # Reshapes to [batchsize, 8192]
                    w_star_j = w_star_j.view(w_star_j.size(0), -1)  # Reshapes to [batchsize, 8192]
                    half_latent_space_size = w_star_j.size(1) // 2

                    combined_w_star = torch.cat([w_star_i[:, :half_latent_space_size],w_star_j[:, half_latent_space_size:]], dim=1) #The rows are identity part, columns are non-id

                    combined_w_star = combined_w_star.view(-1, 16, 512)

                    # map to W^+
                    combined_w_plus = model.inverse_T(combined_w_star)

                    combined_image = Generator(combined_w_plus)
                    
                    # Normalize the images to [0, 1] for visualization
                    combined_img = (combined_image * 0.5 + 0.5).clamp(0, 1)

                    grid_images.append(combined_img)
        # Concatenate all images in the row horizontally
       
    grid_image = torch.cat(grid_images, dim=2)
    grid_image = grid_image.squeeze(0)

    # Assuming grid_images is your list of (n+1)^2 image tensors
    # First, reshape grid_images into a matrix of image tensors for easier manipulation
    image_matrix = [grid_images[i:i + grid_size] for i in range(0, len(grid_images), grid_size)]

    # Concatenate images within each row
    rows = [torch.cat(row_images, dim=3) for row_images in image_matrix]

    # Concatenate rows to form the grid
    final_grid = torch.cat(rows, dim=2).squeeze(0)

    # Save the final grid to a file
    save_image(final_grid, save_path, nrow=grid_size)


def save_perturbed_images(Generator, model, encoded_images_tensor, save_path):
    """
    For each selected image, this function encodes the image, perturbs each half of the latent vector in different ways,
    generates new images for each perturbed vector, and saves the original along with the perturbed images.

    The perturbation involves using the mean of the latent vectors from a set of encoded images to perturb
    the halves of another set of latent vectors, creating variations in the identity and non-identity aspects of the images.

    Parameters:
    - Generator: The generator model of a GAN used to generate images from latent vectors.
    - model: The DisGAN model used for encoding images and their inverse transformation.
    - encoded_images_tensor: A tensor of encoded images used to calculate the mean latent vector for perturbation.
    - save_path: Path where the generated image grid will be saved.

    This function demonstrates manipulating the latent space to observe changes in generated images, specifically focusing
    on identity and non-identity aspects by perturbing different halves of the latent space.
    """

    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    Generator.to('cuda')

    image_paths = [
        'datasets/celebahq/images/771.jpg', 'datasets/celebahq/images/1170.jpg',
        'datasets/celebahq/images/1732.jpg', 'datasets/celebahq/images/1826.jpg',
        'datasets/celebahq/images/19978.jpg', 'datasets/celebahq/images/19336.jpg'
    ]

    encoded_images_tensor = encoded_images_tensor.reshape(30000, -1)
    means = torch.mean(encoded_images_tensor, dim=0).unsqueeze(0).to('cuda')

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    all_images = []

    for image_path in tqdm(image_paths, desc="Processing images"):
        img = Image.open(image_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to('cuda')

        with torch.no_grad():
            w_plus, w_hat = model(img_tensor)

            # Generate perturbed examples
            w_hat_reshaped = w_hat.view(w_hat.size(0), -1)  # Reshapes to [batchsize, 8192]
            half_latent_space_size = w_hat_reshaped.size(1) // 2
            
            #Perturb using w' = - w + 2u on each half, keeping the other half intact
            # Tensor with the first half zero and the second half unchanged
            zeros_first_half_w_hat_second_half = torch.cat([
            torch.zeros(w_hat_reshaped[:, :half_latent_space_size].shape).to(w_hat_reshaped.device), 
            w_hat_reshaped[:, half_latent_space_size:]], dim=1)
            # Tensor with the second half zero and the first half unchanged
            zeros_second_half_w_hat_first_half = torch.cat([
            w_hat_reshaped[:, :half_latent_space_size], 
            torch.zeros(w_hat_reshaped[:, half_latent_space_size:].shape).to(w_hat_reshaped.device)], dim=1)

            zeros_first_half_means_second_half = torch.cat([
            torch.zeros(means[:, :half_latent_space_size].shape).to(means.device), 
            means[:, half_latent_space_size:]], dim=1)
            # Tensor with the second half zero and the first half unchanged
            zeros_second_half_means_first_half = torch.cat([
            means[:, :half_latent_space_size], 
            torch.zeros(means[:, half_latent_space_size:].shape).to(means.device)], dim=1)

            resampled_identity = zeros_first_half_w_hat_second_half - zeros_second_half_w_hat_first_half + 2 * zeros_second_half_means_first_half
            resampled_non_identity = zeros_second_half_w_hat_first_half - zeros_first_half_w_hat_second_half + 2 * zeros_first_half_means_second_half
            
            resampled_identity = resampled_identity.view(-1, 16, 512)
            resampled_non_identity = resampled_non_identity.view(-1, 16, 512)

            # map to W^+
            reconstructed_w_plus_resampled_identity = model.inverse_T(resampled_identity)
            reconstructed_w_plus_resampled_non_identity = model.inverse_T(resampled_non_identity)

            generated_img = Generator(w_plus)
            reconstructed_img_resampled_identity = Generator(reconstructed_w_plus_resampled_identity)
            reconstructed_img_resampled_non_identity = Generator(reconstructed_w_plus_resampled_non_identity)
            

            # Normalize the images to [0, 1] for visualization
            generated_img_normalized = (generated_img * 0.5 + 0.5).clamp(0, 1)
            reconstructed_img_resampled_identity_normalized = (reconstructed_img_resampled_identity * 0.5 + 0.5).clamp(0, 1)
            reconstructed_img_resampled_non_identity_normalized = (reconstructed_img_resampled_non_identity * 0.5 + 0.5).clamp(0, 1)

            combined_images = torch.cat((
                generated_img_normalized,
                reconstructed_img_resampled_identity_normalized,
                reconstructed_img_resampled_non_identity_normalized
            ), dim=3)

            all_images.append(combined_images)

    # Concatenate all rows vertically
    all_images_combined = torch.cat(all_images, dim=2).squeeze(0)
    save_image(all_images_combined, save_path, nrow=3)  # Adjust nrow to 4 for 4 images per row
    print('----')
    print(encoded_images_tensor.shape)


def ratio_metrics(encoded_images, identity_ids):
    """
    Calculate intra-identity and inter-identity distances.
    
    Args:
    - encoded_images: Tensor of shape [N, D_1, D_2] where N is the number of images
    - identity_ids: Tensor of shape [N], where each element is the identity ID
                    corresponding to the encoded images.
    
    Returns:
    - ratio_identity: the ratio of intra to inter distance for the identity part of the tensor
    - ratio_non_identity: the ratio of intra to inter distance for the non-identity part of the tensor
    """
    intra_distances_identity = []
    inter_distances_identity = []


    intra_distances_non_identity = []
    inter_distances_non_identity = []

    encoded_images = encoded_images.view(encoded_images.size(0), -1)
    half_size = encoded_images.size(1) // 2
    first_half = encoded_images[:, :half_size]
    second_half = encoded_images[:,half_size: ]

    # Convert tensors to numpy for easier manipulation
    encoded_first_half_np = first_half.numpy()
    encoded_second_half_np = second_half.numpy()
    identity_ids_np = identity_ids.numpy()

    # Iterate over each unique identity
    for identity in tqdm(np.unique(identity_ids_np), desc="Processing identities"):
        same_id_indices = np.where(identity_ids_np == identity)[0]
        diff_id_indices = np.where(identity_ids_np != identity)[0]

        # Intra-identity pairs
        for pair in combinations(same_id_indices, 2):
            distance_identity = np.linalg.norm(encoded_first_half_np[pair[0]] - encoded_first_half_np[pair[1]])
            distance_non_identity = np.linalg.norm(encoded_second_half_np[pair[0]] - encoded_second_half_np[pair[1]])
            intra_distances_identity.append(distance_identity)
            intra_distances_non_identity.append(distance_non_identity)

        # Inter-identity pairs - randomly sample to reduce computation
        if len(diff_id_indices) > 1:
            sampled_diff_ids = np.random.choice(diff_id_indices, 2, replace=False)
            distance_identity = np.linalg.norm(encoded_first_half_np[sampled_diff_ids[0]] - encoded_first_half_np[sampled_diff_ids[1]])
            distance_non_identity = np.linalg.norm(encoded_second_half_np[sampled_diff_ids[0]] - encoded_second_half_np[sampled_diff_ids[1]])
            inter_distances_identity.append(distance_identity)
            inter_distances_non_identity.append(distance_non_identity)

    # Calculate average distances
    intra_distances_identity = np.mean(intra_distances_identity)
    inter_distances_identity = np.mean(inter_distances_identity)

    intra_distances_non_identity = np.mean(intra_distances_non_identity)
    inter_distances_non_identity = np.mean(inter_distances_non_identity)

    ratio_identity = intra_distances_identity/inter_distances_identity
    ratio_non_identity = intra_distances_non_identity / inter_distances_non_identity

    return ratio_identity , ratio_non_identity 



def ratio_identity_part(encoded_images, identity_ids):
    """
    Calculate the average intra-identity distance, the average inter-identity distance, and 
    their ratio for the identity part of encoded images. The identity part refers to the first half
    of the encoded representations. This function facilitates understanding the compactness and 
    separability of identity representations.

    Args:
        encoded_images (Tensor): A tensor of shape [N, D_1, D_2], where N is the number of images,
                                 representing encoded images.
        identity_ids (Tensor): A tensor of shape [N], where each element is the identity ID
                               corresponding to the encoded images. Assumes integer values for IDs.

    Returns:
        tuple: Contains three elements:
            - intra_identity_avg_dist (float): The average euclidean distance between pairs of images
                                               with the same identity ID for the first half of their encoded representations.
            - inter_identity_avg_dist (float): The average euclidean distance between pairs of images
                                               with different identity IDs, sampled from the first half of their encoded representations.
            - ratio_identity (float): The ratio of intra_identity_avg_dist to inter_identity_avg_dist, providing
                                      a measure of how well identities are represented in the encoded space.

    Note:
        - The function computes distances using the first half of the encoded representations to focus
          on the identity-related features.
        - Due to potentially large number of inter-identity comparisons, the function randomly samples
          one pair of different identity images to represent the inter-identity distance. This approach
          is a computational trade-off to make the calculation more tractable.
    """

    intra_distances_identity = []
    inter_distances_identity = []

    encoded_images = encoded_images.view(encoded_images.size(0), -1)
    half_size = encoded_images.size(1) // 2
    first_half = encoded_images[:, :half_size]

    # Convert tensors to numpy for easier manipulation
    encoded_first_half_np = first_half.numpy()
    identity_ids_np = identity_ids.numpy()

    # Iterate over each unique identity
    for identity in tqdm(np.unique(identity_ids_np), desc="Processing identities"):
        same_id_indices = np.where(identity_ids_np == identity)[0]
        diff_id_indices = np.where(identity_ids_np != identity)[0]

        # Intra-identity pairs
        for pair in combinations(same_id_indices, 2):
            norm_v1 = np.linalg.norm(encoded_first_half_np[pair[0]])
            norm_v2 = np.linalg.norm(encoded_first_half_np[pair[1]])
            distance_identity = np.dot(encoded_first_half_np[pair[0]], encoded_first_half_np[pair[1]]) / (norm_v1 * norm_v2)
            #distance_identity = np.linalg.norm(encoded_first_half_np[pair[0]] - encoded_first_half_np[pair[1]])
            intra_distances_identity.append(distance_identity)

        # Inter-identity pairs - randomly sample to reduce computation
        if len(diff_id_indices) > 1:
            sampled_diff_ids = np.random.choice(diff_id_indices, 2, replace=False)
            norm_v1 = np.linalg.norm(encoded_first_half_np[sampled_diff_ids[0]])
            norm_v2 = np.linalg.norm(encoded_first_half_np[sampled_diff_ids[1]])
            distance_identity = np.dot(encoded_first_half_np[sampled_diff_ids[0]], encoded_first_half_np[sampled_diff_ids[1]]) / (norm_v1 * norm_v2)
            #distance_identity = np.linalg.norm(encoded_first_half_np[sampled_diff_ids[0]] - encoded_first_half_np[sampled_diff_ids[1]])
            inter_distances_identity.append(distance_identity)

    # Calculate average distances
    average_positive_distances_first_half = np.mean(intra_distances_identity)
    average_negative_distances_first_half = np.mean(inter_distances_identity)

    ratio_distances = average_positive_distances_first_half/average_negative_distances_first_half

    return average_positive_distances_first_half , average_negative_distances_first_half, ratio_distances 




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
        i += 1
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

    # Convert lists to tensors.
    encoded_images_tensor = torch.tensor(encoded_images).squeeze(1)
    identity_ids_tensor = torch.tensor(identity_ids)

    return encoded_images_tensor, identity_ids_tensor




class FR_latent_space_distance(nn.Module):
    """
    This module computes the face recognition (FR) latent space distances for pairs of images of same ID or not, where one
    of the images is generated through mixing the original with another, where either the ID part of the latent space
    is mixed or the non-ID part. It also calculates FID on the mixed images.

    Parameters:
    - generator (torch.nn.Module): The generator model from a GAN, used to generate images from latent vectors.
    - face_detection (nn.Module): A face detection model used to preprocess images before passing them to the FR model.
    - face_recognition (nn.Module): A pre-trained face recognition model used to encode images into face embeddings.
    - save_path (str): Path to save the output statistics of the experiment.

    Attributes:
    - num_pairs (int): The number of image pairs to process for calculating distances.
    - generator, face_detection, face_recognition: Model components initialized via the constructor.
    - save_path (str): File path for saving experimental results.
    """

    def __init__(self, generator, face_detection, face_recognition, save_path):
        super(FR_latent_space_distance, self).__init__()
        self.num_pairs = 2500
        self.set_seed()  # Set a fixed seed for reproducibility
        self.generator = generator
        self.face_detection = face_detection
        self.face_recognition = face_recognition
        self.save_path = save_path

        # Initialize the results file and write the header
        with open(self.save_path, 'w') as file:
            file.write(f"Experiment Parameters:\nNumber of Pairs: {self.num_pairs}\n")
            file.write("epoch,FID,positives_distance, negatives_distance, Ratio\n")

    def set_seed(self):
        """Set the seed for random number generation to ensure reproducibility."""
        seed = 10
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def select_pairs(self, identity_ids_tensor):
        """
        Randomly selects pairs of indices for generating pairs of images. Ensures that pairs either
        share the same identity or have different identities.

        Parameters:
        - identity_ids_tensor (torch.Tensor): A tensor of identity IDs for all images.

        Returns:
        - pairs (list of tuples): A list where each tuple contains the indices of two images.
        """
        pairs = []
        indices = np.arange(len(identity_ids_tensor))
        np.random.shuffle(indices)  # Randomly shuffle indices to randomize pair selection

        for i in range(0, len(indices) - 1, 2):  # Step by 2 to form pairs
            if len(pairs) >= self.num_pairs:
                break
            if identity_ids_tensor[indices[i]] != identity_ids_tensor[indices[i + 1]]:
                pairs.append((indices[i], indices[i + 1]))

        return pairs

    def forward(self, encoded_images_tensor, identity_ids_tensor, model, epoch):
        """
        Processes pairs of images to compute the FR latent space distance and FID score.
        Images are generated, converted to embeddings, and distances are calculated.

        Parameters:
        - encoded_images_tensor (torch.Tensor): Tensor of encoded images' latent vectors.
        - identity_ids_tensor (torch.Tensor): Tensor of identity IDs corresponding to each image.
        - model (torch.nn.Module): The DisGAN model used for inverse mapping from latent space.
        - epoch (int): Current epoch number, used for tracking in output files.

        Outputs results to the specified save file and updates FID calculations.
        """
        if isinstance(model, torch.nn.DataParallel):
            model = model.module  # Handling DataParallel wrapped models

        positives_distances = []
        negatives_distances = []

        pairs = self.select_pairs(identity_ids_tensor)

        fid = FrechetInceptionDistance(feature=64).to(device)  # Initialize FID calculation

        for idx1, idx2 in tqdm(pairs, desc='Processing pairs'):
            # Process each pair to generate images and compute distances
            w_star_i, w_star_j = encoded_images_tensor[idx1].unsqueeze(0).to(device), encoded_images_tensor[idx2].unsqueeze(0).to(device)

            w_star_i = w_star_i.view(w_star_i.size(0), -1)  # Reshape to flatten the embeddings
            w_star_j = w_star_j.view(w_star_j.size(0), -1)

            half_latent_space_size = w_star_j.size(1) // 2

            # Create combined latent vectors by swapping halves (to simulate same identity but different attributes)
            w_star_i_identity = torch.cat([w_star_i[:, :half_latent_space_size], w_star_j[:, half_latent_space_size:]], dim=1).view(-1, 16, 512)
            w_star_j_identity = torch.cat([w_star_j[:, :half_latent_space_size], w_star_i[:, half_latent_space_size:]], dim=1).view(-1, 16, 512)
            w_star_i = w_star_i.view(-1, 16, 512)
            w_star_j = w_star_j.view(-1, 16, 512)

            # Inverse transform to W+ space and generate images
            w_plus_i = model.inverse_T(w_star_i)
            w_plus_j = model.inverse_T(w_star_j)
            w_plus_i_identity = model.inverse_T(w_star_i_identity)
            W_plus_j_identity = model.inverse_T(w_star_j_identity)

            # Generate images using the generator model
            i_image = self.generator(w_plus_i)
            j_image = self.generator(w_plus_j)
            i_identity_image = self.generator(w_plus_i_identity)
            j_identity_image = self.generator(W_plus_j_identity)

            # Normalize images to [0, 1] and resize for face recognition
            i_image = (i_image * 0.5 + 0.5).clamp(0, 1)
            j_image = (j_image * 0.5 + 0.5).clamp(0, 1)
            i_identity_image = (i_identity_image * 0.5 + 0.5).clamp(0, 1)
            j_identity_image = (j_identity_image * 0.5 + 0.5).clamp(0, 1)

            i_image = F.interpolate(i_image, size=(160, 160), mode='bilinear', align_corners=False)
            j_image = F.interpolate(j_image, size=(160, 160), mode='bilinear', align_corners=False)
            i_identity_image = F.interpolate(i_identity_image, size=(160, 160), mode='bilinear', align_corners=False)
            j_identity_image = F.interpolate(j_identity_image, size=(160, 160), mode='bilinear', align_corners=False)

            # Compute distances using cosine similarity between image embeddings
            distance = self.get_distance(i_image, i_identity_image)
            positives_distances.append(distance)

            distance = self.get_distance(j_image, j_identity_image)
            positives_distances.append(distance)

            distance = self.get_distance(i_image, j_identity_image)
            negatives_distances.append(distance)

            distance = self.get_distance(j_image, i_identity_image)
            negatives_distances.append(distance)

            # Update real and fake images for FID calculation
            real_images = torch.cat([i_image, j_image], dim=0)
            fake_images = torch.cat([i_identity_image, j_identity_image], dim=0)

            fid.update(real_images.to(torch.uint8).to(device), real=True)
            fid.update(fake_images.to(torch.uint8).to(device), real=False)

            # Clean up to save memory
            del real_images, fake_images
            torch.cuda.empty_cache()

        # Calculate average distances and FID score
        average_positive_distances = np.mean(positives_distances)
        average_negative_distances = np.mean(negatives_distances)

        fid_score = fid.compute().cpu()

        # Save computed statistics to file
        self.save_statistics(epoch, fid_score, average_positive_distances, average_negative_distances)

    def get_distance(self, img1, img2):
        """
        Computes the cosine similarity between the embeddings of two images.

        Parameters:
        - img1, img2 (torch.Tensor): Tensors representing images to compare.

        Returns:
        - similarity (float): The cosine similarity between the two image embeddings.
        """
        # Compute embeddings without updating gradients
        with torch.no_grad():
            embedding1 = self.face_recognition(img1).squeeze_()
            embedding2 = self.face_recognition(img2).squeeze_()

        # Normalize embeddings to unit vectors
        embedding1_norm = embedding1 / (embedding1.norm(p=2) + 1e-6)
        embedding2_norm = embedding2 / (embedding2.norm(p=2) + 1e-6)

        # Calculate cosine similarity
        similarity = torch.dot(embedding1_norm, embedding2_norm).item()
        return similarity

    def save_statistics(self, epoch, fid_score, positives_distance, negatives_distance):
        """
        Writes the calculated distances and FID score for the current epoch to the results file.

        Parameters:
        - epoch (int): The current epoch number.
        - fid_score (float): The Frechet Inception Distance score.
        - positives_distance (float): Average cosine similarity for images supposed to be of the same identity.
        - negatives_distance (float): Average cosine similarity for images supposed to be of different identities.
        """
        ratio_distance = positives_distance / negatives_distance  # Calculate ratio of distances for analysis
        with open(self.save_path, 'a') as file:
            file.write(
                f"{epoch},{fid_score:.3f},{positives_distance:.3f},{negatives_distance:.3f},{ratio_distance:.3f}\n"
            )
