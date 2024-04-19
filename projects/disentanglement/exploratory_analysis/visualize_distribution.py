import os
import pickle
import torch
import matplotlib.pyplot as plt

def compute_stats_and_plot_distributions(pickle_path, num_dimensions=2):
    # Load the pickle file
    with open(pickle_path, 'rb') as f:
        encoded_images_tensor = pickle.load(f)
    
    # Remove the unnecessary dimension and reshape
    encoded_images_tensor = encoded_images_tensor.squeeze(1).reshape(30000, -1)

    # Get the directory of the pickle file to save histograms in the same location
    directory = os.path.dirname(pickle_path)

    # Plot and save histograms for the first num_dimensions dimensions
    for i in range(num_dimensions):
        tensor_slice = encoded_images_tensor[:, i].numpy()
        plt.figure(figsize=(10, 6))
        plt.hist(tensor_slice, bins=50, alpha=0.75, color='blue')
        plt.title(f'Histogram of Dimension {i+1}')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.grid(True)

        # Construct histogram file name and save path
        histogram_filename = f'histogram_dimension_{i+1}.png'
        histogram_save_path = os.path.join(directory, histogram_filename)
        
        # Save the histogram to the same directory as the pickle file
        plt.savefig(histogram_save_path)
        plt.close()  # Close the plot to avoid displaying it inline if you're using a notebook

        print(f"Histogram saved to {histogram_save_path}")

if __name__ == "__main__":
    # Replace with the path to your saved pickle file

    pickle_path = 'projects/disentanglement/encoded_images/encoded_w_plus.pkl'  # Update this path
    compute_stats_and_plot_distributions(pickle_path)
