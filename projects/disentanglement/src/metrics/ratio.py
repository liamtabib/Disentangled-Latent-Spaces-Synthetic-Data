import pickle
import torch
import numpy as np
from itertools import combinations
from tqdm import tqdm

def load_data(pickle_file_path):
    """Load the encoded images and identity IDs from a pickle file."""
    with open(pickle_file_path, 'rb') as f:
        data = pickle.load(f)
    return data['encoded_images'], data['identity_ids']

def calculate_distances(encoded_images, identity_ids):
    """
    Calculate intra-identity and inter-identity distances.
    
    Args:
    - encoded_images: Tensor of shape [N, D_1, D_2] where N is the number of images
    - identity_ids: Tensor of shape [N], where each element is the identity ID
                    corresponding to the encoded images.
    
    Returns:
    - intra_identity_distance: Average distance between pairs of encodings of the same identity.
    - inter_identity_distance: Average distance between pairs of encodings of different identities.
    """
    intra_distances = []
    inter_distances = []

    # Convert tensors to numpy for easier manipulation
    encoded_images_np = encoded_images.numpy()
    print(encoded_images_np.shape)
    identity_ids_np = identity_ids.numpy()

    # Iterate over each unique identity
    for identity in tqdm(np.unique(identity_ids_np), desc="Processing identities"):
        same_id_indices = np.where(identity_ids_np == identity)[0]
        diff_id_indices = np.where(identity_ids_np != identity)[0]

        # Intra-identity pairs
        for pair in combinations(same_id_indices, 2):
            distance = np.linalg.norm(encoded_images_np[pair[0]] - encoded_images_np[pair[1]])
            intra_distances.append(distance)

        # Inter-identity pairs - randomly sample to reduce computation
        if len(diff_id_indices) > 1:
            sampled_diff_ids = np.random.choice(diff_id_indices, 2, replace=False)
            distance = np.linalg.norm(encoded_images_np[sampled_diff_ids[0]] - encoded_images_np[sampled_diff_ids[1]])
            inter_distances.append(distance)

    # Calculate average distances
    intra_identity_distance = np.mean(intra_distances)
    inter_identity_distance = np.mean(inter_distances)

    return intra_identity_distance, inter_identity_distance


def main():
    pickle_file_path = 'projects/disentanglement/encoded_images/encoded_w_plus_with_ids.pkl'
    encoded_images, identity_ids = load_data(pickle_file_path)
    load_data(pickle_file_path)

    print(f"Total encoded images: {encoded_images.shape[0]}")
    print(f"Unique identity IDs: {len(torch.unique(identity_ids))}")

    intra_identity_distance, inter_identity_distance = calculate_distances(encoded_images, identity_ids)

    print(f"Intra-identity Distance: {intra_identity_distance}")
    print(f"Inter-identity Distance: {inter_identity_distance}")

if __name__ == "__main__":
    main()
