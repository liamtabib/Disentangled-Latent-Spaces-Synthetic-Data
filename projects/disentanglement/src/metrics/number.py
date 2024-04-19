import pickle
import pandas as pd

def load_pickle_data(pickle_file_path):
    """
    Load the encoded images and identity IDs from a pickle file.
    
    Parameters:
    - pickle_file_path (str): Path to the pickle file.
    
    Returns:
    - identity_ids (list): List of identity IDs.
    """
    with open(pickle_file_path, 'rb') as f:
        data = pickle.load(f)
    identity_ids = data['identity_ids'].numpy()  # Assuming identity_ids is a tensor
    return identity_ids

def main():
    # Path to your pickle file containing encoded images and identity IDs
    pickle_file_path = 'projects/disentanglement/encoded_images/encoded_w_plus_with_ids.pkl'
    
    # Load identity IDs from the pickle file
    identity_ids = load_pickle_data(pickle_file_path)
    
    # Convert identity IDs to a pandas Series for easy counting and manipulation
    identity_series = pd.Series(identity_ids)
    
    # Count the number of images per identity
    image_counts_per_identity = identity_series.value_counts()
    
    # Sort the counts in descending order
    sorted_image_counts = image_counts_per_identity.sort_values(ascending=False)
    
    # Print summary information
    print(f"Total number of images: {identity_series.size}")
    print(f"Number of unique identities: {sorted_image_counts.size}")
    
    # Print the distribution of images per identity
    print("\nDistribution of images per identity:")
    print(sorted_image_counts.describe())
    
    # Print the counts for each identity, sorted by the number of images
    print("\nNumber of images per identity (sorted):")
    print(sorted_image_counts.head(10))  # Adjust as necessary to display more or fewer identities

if __name__ == "__main__":
    main()
