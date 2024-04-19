import cv2
import numpy as np

INTER = {
    "nearest": cv2.INTER_NEAREST,
    "linear": cv2.INTER_LINEAR,
    "area": cv2.INTER_AREA,
    "cubic": cv2.INTER_CUBIC,
    "lanczos4": cv2.INTER_LANCZOS4,
}
BORDER = {
    "constant": cv2.BORDER_CONSTANT,
    "replicate": cv2.BORDER_REPLICATE,
    "reflect": cv2.BORDER_REFLECT,
    "reflect101": cv2.BORDER_REFLECT101,
    "wrap": cv2.BORDER_WRAP,
}


class FaceCropper:
    def __init__(
        self,
        standard_landmark,
        crop_size,
        face_factor=0.45,
        interpolation_type="cubic",
        border_type="constant",
    ):
        super().__init__()
        self.standard_landmark = standard_landmark

        if isinstance(crop_size, (list, tuple)) and len(crop_size):
            self.crop_size_h, self.crop_size_w = crop_size[0], crop_size[1]
        elif isinstance(crop_size, int):
            self.crop_size_h, self.crop_size_w = crop_size
        else:
            raise Exception(
                "Invalid `crop_size`. `crop_size` must be "
                "either int for `crop_size_h = crop_size_w` or"
                "tuple for `crop_size_h != crop_size_w`"
            )

        self.face_factor = face_factor
        self.interpolation_type = interpolation_type
        self.border_type = border_type

    def align_and_crop(self, img, src_landmark):
        """
        The class method for alignment and cropping the image.
        :param img: face image to be aligned and cropped.
        :param src_landmark: landmark of the current image.
        :return: cropped image and new landmark.
        """

        # Compute target landmark
        target_landmark = self.compute_target_landmark()

        # Estimate transformation matrix
        tform_matrix = cv2.estimateAffinePartial2D(target_landmark, src_landmark)[0]

        # Warp image by given transformation
        img_crop = cv2.warpAffine(
            img,
            tform_matrix,
            (self.crop_size_h, self.crop_size_w),
            flags=cv2.WARP_INVERSE_MAP + INTER[self.interpolation_type],
            borderMode=BORDER[self.border_type],
        )

        # Obtain transformed landmarks
        tformed_landmarks = cv2.transform(
            np.expand_dims(src_landmark, axis=0),
            cv2.invertAffineTransform(tform_matrix),
        )[0]

        return img_crop, tformed_landmarks

    def compute_target_landmark(self):
        return self.standard_landmark * max(
            self.crop_size_h, self.crop_size_w
        ) * self.face_factor + np.array([self.crop_size_w // 2, self.crop_size_h // 2])
