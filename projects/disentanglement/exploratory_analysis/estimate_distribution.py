import pickle
import torch

def compute_stats_and_print(pickle_path):
    # Load the pickle file
    with open(pickle_path, 'rb') as f:
        encoded_images_tensor = pickle.load(f)
    
    encoded_images_tensor = encoded_images_tensor.reshape(30000, -1)
    # Now shape is [30000, 8192]

    # Compute the mean and standard deviation for each of the 8192 dimensions
    means = torch.mean(encoded_images_tensor, dim=0)
    std_devs = torch.std(encoded_images_tensor, dim=0)

    print(means.shape)

    # Print out the statistics for the first 10 dimensions
    print("Statistics for the first 10 dimensions:")
    for i in range(10):
        print(f"Dimension {i+1} - Mean: {means[i].item():.6f}, Std Dev: {std_devs[i].item():.6f}")

if __name__ == "__main__":
    # Replace with the path to your saved pickle file
    pickle_path = 'projects/disentanglement/encoded_images/encoded_w_plus.pkl'  # Update this path
    compute_stats_and_print(pickle_path)
