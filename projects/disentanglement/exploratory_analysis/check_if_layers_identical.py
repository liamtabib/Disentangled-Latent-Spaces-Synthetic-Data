import pickle

def check_layers(pickle_path):
    # Load the pickle file
    with open(pickle_path, 'rb') as f:
        encoded_images_tensor = pickle.load(f)
    
    # Access the first item in the tensor
    first_item = encoded_images_tensor[0]  # This has a shape [1, 16, 512]
    
    # Iterate through each of the 16 layers
    for i in range(first_item.shape[1]):  # Assuming the second dimension is the layer dimension
        # Extract and print the first three elements of the i-th layer
        layer_i_first_3_elements = first_item[0, i, :3].numpy()
        print(f"Layer {i+1} first 3 elements:", layer_i_first_3_elements)

if __name__ == "__main__":
    # Replace with the path to your saved pickle file
    pickle_path = 'projects/disentanglement/encoded_images/encoded_w_plus.pkl'
    check_layers(pickle_path)
