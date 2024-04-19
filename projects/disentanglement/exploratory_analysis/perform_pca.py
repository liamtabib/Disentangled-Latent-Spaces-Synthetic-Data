import pickle
import numpy as np
from sklearn.decomposition import PCA
import os

def perform_pca_and_save(pickle_path, n_components=10):
    # Load the latent vectors
    with open(pickle_path, 'rb') as f:
        latent_vectors = pickle.load(f)

    # Assuming the latent vectors are in the shape [30000, 1, 16, 512]
    # We first remove the unnecessary dimension and then reshape to [30000, 8192]
    latent_vectors = latent_vectors.squeeze(1).reshape(len(latent_vectors), -1).numpy()

    # Perform PCA
    pca = PCA(n_components=n_components)
    pca.fit(latent_vectors)

    # Extract and store the principal components
    principal_components = pca.components_  # Shape: [n_components, 8192]
    print('----')
    print(principal_components.shape)

    # Store the first 10 principal components
    output_dir = os.path.dirname(pickle_path)
    pca_filename = 'latent_pca_components.pkl'
    pca_output_path = os.path.join(output_dir, pca_filename)

    with open(pca_output_path, 'wb') as f:
        pickle.dump(principal_components, f)

    print(f"First {n_components} principal components saved to {pca_output_path}")

if __name__ == "__main__":
    # Path to the pickle file containing the latent vectors
    pickle_path = 'projects/disentanglement/encoded_images/encoded_w_plus.pkl'  # Update this path
    perform_pca_and_save(pickle_path)
