import argparse
import multiprocessing
import re
from functools import partial
from pathlib import Path

import cv2
import numpy as np
import tqdm

from projects.utils.data.cropping.face_cropper import FaceCropper

ROOT = Path(__file__).absolute().parents[2]


def crop_image(pair, face_cropper, src_landmarks, n_landmark, save_dir):
    i, img_path = pair

    # Cropping the image
    img = cv2.imread(str(img_path))
    img_cropped, tformed_landmarks = face_cropper.align_and_crop(img, src_landmarks[i])

    # Saving the cropped image
    cv2.imwrite(str(save_dir / img_path.name), img_cropped)

    tformed_landmarks.shape = -1
    target_landmark = ("%s" + "%.1f" * n_landmark * 2) % (
        (img_path.stem,) + tuple(tformed_landmarks)
    )
    return target_landmark


def main(args):
    img_paths = list(Path(args.img_dir).glob("*.jpg"))
    img_paths = sorted(img_paths, key=lambda path: int(path.stem))

    print("Loading landmarks...")
    with open(args.landmark_file) as f:
        line = f.readline()
    n_landmark = len(re.split("[ ]+", line)[1:]) // 2

    src_landmarks = np.genfromtxt(
        args.landmark_file, dtype=float, usecols=range(1, n_landmark * 2 + 1)
    ).reshape(-1, n_landmark, 2)
    standard_landmark = np.genfromtxt(args.standard_landmark_file, dtype=float).reshape(
        n_landmark, 2
    )

    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    print("Cropping images...")
    face_cropper = FaceCropper(
        standard_landmark,
        (args.crop_size_h, args.crop_size_w),
        args.face_factor,
        args.interpolation_type,
        args.border_type,
    )

    pool = multiprocessing.Pool(args.n_workers)

    crop_func = partial(
        crop_image,
        face_cropper=face_cropper,
        src_landmarks=src_landmarks,
        n_landmark=n_landmark,
        save_dir=args.save_dir,
    )
    dataset_length = len(img_paths)
    target_landmarks = list(
        tqdm.tqdm(
            pool.imap(
                crop_func,
                zip(range(dataset_length), img_paths),
                chunksize=int(dataset_length / (args.n_workers)),
            ),
            total=dataset_length,
        )
    )
    pool.close()
    pool.join()

    # Saving transformed landmarks
    landmarks_path = save_dir / "target_landmarks.txt"
    with open(landmarks_path, "w") as f:
        for landmark_str in target_landmarks:
            if landmark_str:
                f.write(landmark_str + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", default=ROOT / "datasets/celeba/img_celeba", type=Path)
    parser.add_argument(
        "--landmark_file",
        default=ROOT / "datasets/celeba/annotations/landmark.txt",
        type=Path,
    )
    parser.add_argument(
        "--standard_landmark_file",
        default=ROOT / "datasets/celeba/annotations/standard_landmark_68pts.txt",
        type=Path,
    )
    parser.add_argument("--save_dir", default=ROOT / "datasets/celeba/cropped", type=Path)
    parser.add_argument("--crop_size_h", default=112, type=int)
    parser.add_argument("--crop_size_w", default=112, type=int)
    parser.add_argument(
        "--interpolation_type",
        default="cubic",
        choices=["nearest", "linear", "area", "cubic", "lanczos4"],
        type=str,
    )
    parser.add_argument(
        "--border_type",
        default="constant",
        choices=["constant", "replicate", "reflect", "reflect101", "wrap"],
        type=str,
    )
    parser.add_argument(
        "--face_factor",
        default=0.45,
        type=float,
        help="The factor of face area relative to the output image.",
    )
    parser.add_argument("--n_workers", default=16, type=int)
    args = parser.parse_args()
    main(args)
