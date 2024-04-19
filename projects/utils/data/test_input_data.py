import os

import albumentations as A
import numpy as np
import torchvision.transforms as transforms


def reverse_normalize(image, mean, std):
    """
    Reverse normalization of an image.

    Args:
        image (numpy.ndarray): The normalized image (values in the range [0, 1]).
        mean (list): List of mean values used during normalization (R, G, B).
        std (list): List of standard deviation values used during normalization (R, G, B).

    Returns:
        numpy.ndarray: The reverse normalized image with values in the range [0, 255].

    """
    reverse_transform = A.Compose(
        [
            A.Normalize(
                mean=[-m / s for m, s in zip(mean, std)],
                std=[1 / s for s in std],
                max_pixel_value=1.0,
            ),
        ]
    )
    image_np = reverse_transform(image=image)["image"]
    rescaled_image = (image_np * 255.0).astype(np.uint8)
    return rescaled_image


def save_random_images_from_dataset(
    dataset, output_dir, MEAN=[0.5, 0.5, 0.5], STD=[0.5, 0.5, 0.5], num_images_to_save=20
):
    """
    Save random images from a dataset to the specified output directory.

    Args:
        dataset: The dataset to extract images from. (Assuming the dataset contains a "img" key for images)
        output_dir (str): The directory where the images will be saved.
        num_images_to_save (int): The number of images to save. Default is 50.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    i = 0
    tensor_to_pil = transforms.ToPILImage()
    for batch in dataset:
        for image in batch["img"]:
            if i >= num_images_to_save:
                break
            i += 1

            image_np = image.numpy().transpose(1, 2, 0)

            # Assuming MEAN and STD are defined globally
            rescaled_image = reverse_normalize(image_np, MEAN, STD)

            filename = f"batch_{i}.png"

            output_path = os.path.join(output_dir, filename)
            pil_image = tensor_to_pil(rescaled_image)
            pil_image.save(output_path)
    print("Saved images for debugging")
