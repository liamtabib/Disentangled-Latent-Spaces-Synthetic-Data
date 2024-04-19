import albumentations as A
import cv2
import numpy as np

# ------------------Custom augmentations------------------------------------------
# Lambda transforms use custom transformation functions provided by the user.
# For those types of transforms, Albumentations saves only the name and the
# position in the augmentation pipeline.
# To deserialize an augmentation pipeline with Lambda transforms, you need
# to manually provide all Lambda transform instances
# using the lambda_transforms argument.


def hflip_image(image, **kwargs):
    return cv2.flip(image, 1)


class NormalizeToRange(A.Lambda):
    """Normalizes image from [0, 255] to specified range."""

    def __init__(self, min_val=-1, max_val=1, always_apply=True, p=1):
        super(NormalizeToRange, self).__init__(
            image=self.apply, name="normalize_to_range", always_apply=always_apply, p=p
        )

        self.min_val = min_val
        self.max_val = max_val

    def apply(self, image, **params):
        # Assumes image values in range [0, 255].
        return image / 255 * (self.max_val - self.min_val) + self.min_val


class ToFloat32(A.Lambda):
    """Converts image to float32 precision."""

    def __init__(self, always_apply=True, p=1):
        super(ToFloat32, self).__init__(
            image=self.apply, name="to_float32", always_apply=always_apply, p=p
        )

    def apply(self, image, **params):
        return image.astype(dtype=np.float32)
