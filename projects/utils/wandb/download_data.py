import argparse
import glob
import os
import zipfile
from pathlib import Path

import gdown
import pandas as pd
import py7zr
import torchvision
import tqdm
import wget

import wandb

ROOT = Path(__file__).absolute().parents[3]


def synchronize_data(root, dataset_type, dataset_version):
    """
    Synchronizes the data by downloading the dataset from WandB repository or the original source.

    Args:
        root (str): The root directory where the dataset will be downloaded.
        dataset_type (str): The type of dataset to be downloaded.
        dataset_version (str): The version of the dataset to be downloaded.

    Returns:
        pandas.DataFrame: The dataframe containing the downloaded dataset.

    Raises:
        RuntimeError: If the dataset is not downloaded.
        ValueError: If an invalid dataset type is provided.
    """
    print("Check dataset..")
    dataset_functions = {
        "aligned_celeba": download_aligned_celeba,
        "original_celeba": download_original_celeba,
        "lfw": download_lfw,
    }
    if not download_dataset_from_wandb(root, dataset_type, dataset_version):
        if input(
            "There was an error downloading data from WandB. Do you want to download it from the original source? [Y/N]"
        ) in ["Y", "yes", "Yes", "y"]:
            if dataset_type in dataset_functions:
                dataset_functions[dataset_type](root)
            elif dataset_type == "cropped_celeba":
                raise ValueError(
                    "The CelebA dataset needs custom cropping before being used. Run the `crop_celeba` script first."
                )
            else:
                raise ValueError("Invalid `dataset_type`.")
        else:
            raise RuntimeError("The dataset was not downloaded")


def download_lfw(data_root=ROOT / "datasets"):
    """
    Downloads the LFW dataset and creates a dataframe with annotations.

    Args:
        data_root (str, optional): The root directory where the dataset will be downloaded.
        Defaults to ROOT / "datasets".

    Returns:
        pandas.DataFrame: The dataframe containing the dataset annotations.
    """
    torchvision.datasets.LFWPeople(root=data_root, download=True)

    df = pd.DataFrame(columns=["filename", "id", "split"])
    id = 0
    root_folder = str(data_root) + "/lfw-py/lfw_funneled"
    subfolders = glob.glob(root_folder + "/*")
    for folder in subfolders:
        if os.path.isdir(folder):
            image_files = glob.glob(os.path.join(folder, "*.jpg"))
            for file in image_files:
                file = os.path.join(os.path.basename(os.path.dirname(file)), os.path.basename(file))
                row = {"filename": file, "id": id, "split": 3}
                df = df.append(row, ignore_index=True)
            id += 1

    # Save the dataframe to CSV
    annotations_csv_path = os.path.join(data_root, "lfw-py", "annotations.csv")
    df.to_csv(annotations_csv_path, index=False)

    # Rename the folder
    new_folder_path = os.path.join(data_root, "lfw-py", "images")
    os.rename(root_folder, new_folder_path)
    return df


def download_aligned_celeba(data_root=ROOT / "datasets"):
    """
    Downloads the aligned CelebA dataset.

    Args:
        data_root (str, optional): The root directory where the dataset will be downloaded.
        Defaults to ROOT / "datasets".
    """

    dataset_folder = data_root / "aligned_celeba"
    print("Downloading data from the official repository")
    base_url = "https://graal.ift.ulaval.ca/public/celeba/"
    file_list = [
        "img_align_celeba.zip",
        "list_attr_celeba.txt",
        "identity_CelebA.txt",
        "list_bbox_celeba.txt",
        "list_landmarks_align_celeba.txt",
        "list_eval_partition.txt",
    ]
    os.makedirs(dataset_folder, exist_ok=True)
    os.makedirs(dataset_folder / "annotations", exist_ok=True)

    for file in file_list:
        url = f"{base_url}/{file}"
        print(f" Downloading {file}", end="\n")
        if file != "img_align_celeba.zip":
            gdown.download(url, f"{dataset_folder}/annotations/{file}")
        else:
            gdown.download(url, f"{dataset_folder}/{file}")

    with zipfile.ZipFile(dataset_folder / "img_align_celeba.zip", "r") as ziphandler:
        print(" Extracting images...", end="\n")
        ziphandler.extractall(dataset_folder)
    print(" Removing .zip file...", end="\n")
    os.remove(dataset_folder / "img_align_celeba.zip")


def download_original_celeba(data_root=ROOT / "datasets"):
    """
    Downloads the original CelebA dataset.

    Args:
        data_root (str, optional): The root directory where the dataset will be downloaded.
        Defaults to ROOT / "datasets".
    """
    print("Downloading data from the official repository")
    # Path to folder with the dataset
    dataset_folder = data_root / "celeba"
    os.makedirs(dataset_folder, exist_ok=True)

    download_original_celeba_images(dataset_folder)
    download_original_celeba_annotations(dataset_folder)

    img_archives = glob.glob(str(dataset_folder / "img_celeba.7z.*"))
    anno_archives = [dataset_folder / "result.7z", dataset_folder / "annotations.zip"]

    for archive in [*img_archives, *anno_archives]:
        if os.path.exists(archive):
            os.remove(archive)


def download_original_celeba_images(dataset_folder):
    """
    Downloads the images of the original CelebA dataset.

    Args:
        dataset_folder (str): The folder where the dataset images will be saved.
    """
    files = [
        "img_celeba.7z.001",
        "img_celeba.7z.002",
        "img_celeba.7z.003",
        "img_celeba.7z.004",
        "img_celeba.7z.005",
        "img_celeba.7z.006",
        "img_celeba.7z.007",
        "img_celeba.7z.008",
        "img_celeba.7z.009",
        "img_celeba.7z.010",
        "img_celeba.7z.011",
        "img_celeba.7z.012",
        "img_celeba.7z.013",
        "img_celeba.7z.014",
    ]
    urls = [
        "https://drive.google.com/uc?id=0B7EVK8r0v71pQy1YUGtHeUM2dUE",
        "https://drive.google.com/uc?id=0B7EVK8r0v71peFphOHpxODd5SjQ",
        "https://drive.google.com/uc?id=0B7EVK8r0v71pMk5FeXRlOXcxVVU",
        "https://drive.google.com/uc?id=0B7EVK8r0v71peXc4WldxZGFUbk0",
        "https://drive.google.com/uc?id=0B7EVK8r0v71pMktaV1hjZUJhLWM",
        "https://drive.google.com/uc?id=0B7EVK8r0v71pbWFfbGRDOVZxOUU",
        "https://drive.google.com/uc?id=0B7EVK8r0v71pQlZrOENSOUhkQ3c",
        "https://drive.google.com/uc?id=0B7EVK8r0v71pLVltX2F6dzVwT0E",
        "https://drive.google.com/uc?id=0B7EVK8r0v71pVlg5SmtLa1ZiU0k",
        "https://drive.google.com/uc?id=0B7EVK8r0v71pa09rcFF4THRmSFU",
        "https://drive.google.com/uc?id=0B7EVK8r0v71pNU9BZVBEMF9KN28",
        "https://drive.google.com/uc?id=0B7EVK8r0v71pTVd3R2NpQ0FHaGM",
        "https://drive.google.com/uc?id=0B7EVK8r0v71paXBad2lfSzlzSlk",
        "https://drive.google.com/uc?id=0B7EVK8r0v71pcTFwT1VFZzkzZk0",
    ]

    if os.path.exists(dataset_folder / "img_celeba"):
        print("Dataset images already downloaded.")
        return

    if len(glob.glob(str(dataset_folder / "img_celeba.7z.*"))) != len(files):
        # Download original image zip files from Google Drive
        for i, file in enumerate(files):
            print(f" Downloading {file}", end="\n")
            gdown.download(urls[i], os.path.join(dataset_folder, file), quiet=False)

    # Assemble into one zip file and extract images
    unzip_7z(dataset_folder, files)


def download_original_celeba_annotations(dataset_folder):
    """
    Downloads the annotations of the original CelebA dataset.

    Args:
        dataset_folder (str): The folder where the dataset annotations will be saved.
    """
    if os.path.exists(dataset_folder / "annotations"):
        print("Dataset annotations already downloaded.")
        return

    # Download annotations
    url = "https://drive.google.com/uc?id=1xd-d1WRnbt3yJnwh5ORGZI3g-YS-fKM9"
    file_annot = "annotations.zip"

    print("Downloading annotations...", end="\n")
    gdown.download(url, os.path.join(dataset_folder, file_annot), quiet=False)

    with zipfile.ZipFile(dataset_folder / file_annot, "r") as ziphandler:
        print(" Extracting annotations...", end="\n")
        ziphandler.extractall(dataset_folder / "annotations")

    # Download additional annotation files
    url = "https://graal.ift.ulaval.ca/public/celeba/"
    files = ["list_attr_celeba.txt", "identity_CelebA.txt", "list_eval_partition.txt"]

    for file in files:
        if not os.path.exists(dataset_folder / "annotations" / file):
            print(f" Downloading {file}", end="\n")
            wget.download(f"{url}/{file}", dataset_folder / "annotations" / "file")


def unzip_7z(dataset_folder, files_to_assemble):
    """
    Assembles and extracts a 7z archive.

    Args:
        dataset_folder (str): The folder where the dataset will be extracted.
        files_to_assemble (list): The list of 7z files to be assembled.

    Raises:
        RuntimeError: If the number of 7z files to assemble is incorrect.
    """
    assembled = dataset_folder / "result.7z"

    if os.path.exists(assembled):
        os.remove(assembled)

    print("Assembling archive splits...")
    with open(assembled, "ab") as outfile:
        for fname in tqdm.tqdm(files_to_assemble):
            with open(dataset_folder / fname, "rb") as infile:
                outfile.write(infile.read())

    print("Extracting assembled archive... (NOTE: This may take some time)")
    with py7zr.SevenZipFile(assembled, "r") as archive:
        archive.extractall(path=dataset_folder)
    print("Extraction done!")


def download_dataset_from_wandb(data_root, dataset_name, dataset_version):
    """
    Downloads a dataset from the WandB repository.

    Args:
        data_root (str): The root directory where the dataset will be downloaded.
        dataset_name (str): The name of the dataset.
        dataset_version (str): The version of the dataset.

    Returns:
        bool: True if the dataset was successfully downloaded, False otherwise.
    """
    print("Downloading dataset from WandB repository..")
    try:
        artifact = wandb.use_artifact(
            f"diffuse_datasets/datasets/{dataset_name}:{dataset_version}",
            type="dataset",
        )
        print("Downloading data in ", data_root / dataset_name)
        artifact.download(root=data_root / dataset_name)
        return True
    except wandb.Error as e:
        print(f"An error occurred while downloading the dataset: {e}")
        return False


if __name__ == "__main__":
    """Run this script with appropriate arguments to download one of the available datasets."""

    wandb.init()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_root",
        default=ROOT / "datasets",
        help="Folder to download dataset to.",
    )
    parser.add_argument(
        "--dataset_type",
        help="""
                Dataset type is synonymous with a dataset name. If the version you are looking for
                is missing in `choices`, feel free to add it.
            """,
        choices=["celeba", "aligned_celeba", "cropped_celeba", "synthetic_celeba"],
    )
    parser.add_argument(
        "--dataset_version",
        help="""
                Datasets can have versioned histories of changes.
                Available versions for a datasets can be found on WandB.
            """,
    )

    args = parser.parse_args()
    synchronize_data(args.dataset_root, args.dataset_type, args.dataset_version)
