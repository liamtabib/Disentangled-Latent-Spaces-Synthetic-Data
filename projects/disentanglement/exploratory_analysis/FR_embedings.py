import torch
from PIL import Image
from torchvision import transforms
from facenet_pytorch import MTCNN, InceptionResnetV1

def compute_cosine_similarity(embedding1, embedding2):
    """Computes the cosine similarity between two embeddings."""
    # Normalize the embeddings to unit vectors
    embedding1_norm = embedding1 / (embedding1.norm(p=2) + 1e-6)  # Adding a small epsilon to avoid division by zero
    embedding2_norm = embedding2 / (embedding2.norm(p=2) + 1e-6)
    
    # Compute cosine similarity as dot product of the normalized vectors
    similarity = torch.dot(embedding1_norm, embedding2_norm).item()
    return similarity

def get_embeddings(model, cropped_image1, cropped_image2):
    """Generates embeddings for two tensors and returns them."""
    # Ensure tensors are in the correct shape (add batch dimension if missing)
    if len(cropped_image1.shape) == 3:
        cropped_image1 = cropped_image1.unsqueeze(0)
    if len(cropped_image2.shape) == 3:
        cropped_image2 = cropped_image2.unsqueeze(0)
    
    with torch.no_grad():
        embedding1 = model(cropped_image1).squeeze()  # Remove batch dimension
        embedding2 = model(cropped_image2).squeeze()
        
    return embedding1, embedding2

# Initialize MTCNN and InceptionResnetV1
mtcnn = MTCNN(keep_all=False)
resnet = InceptionResnetV1(pretrained='vggface2').eval()

# Load images
# Replace these with your image paths
image_path1 = 'datasets/celebahq/images/7765.jpg'
image_path2 = 'datasets/celebahq/images/9283.jpg'

image1 = Image.open(image_path1)
image2 = Image.open(image_path2)

# Detect and crop faces
cropped_image1, _ = mtcnn(image1, return_prob=True)
cropped_image2, _ = mtcnn(image2, return_prob=True)

# Calculate embeddings
if cropped_image1 is not None and cropped_image2 is not None:
    embedding1, embedding2 = get_embeddings(resnet, cropped_image1, cropped_image2)
    print("Embeddings calculated successfully.")
    
    # Compute the cosine similarity between the two embeddings
    similarity = compute_cosine_similarity(embedding1, embedding2)
    print(f"Cosine similarity between the embeddings: {similarity}")
else:
    print("Face detection failed for one or both images.")
