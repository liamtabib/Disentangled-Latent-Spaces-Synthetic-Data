import os

import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2

from projects.utils.data.augmentations.custom_augmentations import hflip_image


def fr_aug_v_0():
    augmentation = A.Compose(
        [
            A.Resize(112, 112, p=1),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=20, p=0.5, border_mode=cv2.BORDER_CONSTANT),
            A.Blur(blur_limit=(15, 15), p=0.2),
            A.RandomBrightnessContrast(p=0.2),
            A.MultiplicativeNoise(multiplier=[0.5, 1.5], elementwise=True, per_channel=True, p=0.2),
            # Randomly erase a square portion of the image
            A.Cutout(num_holes=1, max_h_size=30, max_w_size=30, fill_value=0, p=0.2),
            A.Normalize([0.5 for _ in range(3)], [0.5 for _ in range(3)]),
            ToTensorV2(),
        ]
    )
    augmentation_name = "./projects/config/augmentation/fr_aug_v_0.yaml"
    return augmentation, augmentation_name


def basic_aug():
    augmentation = A.Compose(
        [A.Normalize([0.5 for _ in range(3)], [0.5 for _ in range(3)]), ToTensorV2()]
    )
    augmentation_name = "./projects/config/augmentation/basic_augmentation.yaml"
    return augmentation, augmentation_name


def example_aug():
    """Example of how to create a new set of augmentations.

    Returns:
        augmentation: Augmentatition created with Almbumentations.
        str: Path where to save the augmentation.
    """
    augmentation = A.Compose(
        [A.Perspective(), hflip_image(p=0.5), A.OneOf([A.RGBShift(), A.HueSaturationValue()])]
    )
    augmentation_name = "./projects/config/augmentation/example_augmentation.yaml"
    return augmentation, augmentation_name


if __name__ == "__main__":
    # To create a new set of augmentations create a new function and then
    # save the new augmentation sequence as a new yaml file in the config folder.

    """WARNING! Be carefull to not overwrite any existing yaml file"""
    augmentation, augmentation_name = basic_aug()

    if os.path.exists(augmentation_name):
        print("WARNING! A file with this name already exists!")
        if input("Do you want to overwrite it? [Y/N]") in ["Y", "yes", "Yes", "y"]:
            A.save(augmentation, augmentation_name, data_format="yaml")
            print(f"The file {os.path.split(augmentation_name)[1]} has been updated")
    else:
        A.save(augmentation, augmentation_name, data_format="yaml")
