import math
import os
import shutil

import hydra
import pandas as pd
from tqdm import tqdm

PERC_IDS_DEBUG = 0.005


# Function to extract the last part of a path
def extract_last_part(path):
    return os.path.basename(path)


def extract_samples(subset_single_ID, number_of_samples, df_list, split):
    """
    Extracts a specified number of samples from a DataFrame subset_single_ID and adds them to df_list.

    Args:
        subset_single_ID (pd.DataFrame): Subset of DataFrame containing samples.
        number_of_samples (int): Number of samples to extract.
        df_list (list): List of DataFrames to which the extracted samples will be appended.
        split (int): Split value for the extracted samples.

    Returns:
        tuple: Updated df_list and subset_single_ID after the extraction.
        subset_single_ID (pd.DataFrame): Subset of DataFrame containing samples.
    """
    samples = subset_single_ID.head(number_of_samples)
    samples["split"] = split
    df_list.append(samples)
    return df_list, subset_single_ID.iloc[number_of_samples:]


def create_dataset_folder(work_dir, dataset):
    debug_dataset_path = os.path.join(work_dir, "datasets/debug_" + dataset["dataset_type"])
    os.makedirs(debug_dataset_path, exist_ok=True)
    os.makedirs(os.path.join(debug_dataset_path, "images"), exist_ok=True)
    return debug_dataset_path


def extract_unique_ids(annotation_path, min_samples=4):
    """
    Extracts unique IDs and their counts from an annotation CSV file, filter out the
    IDs without the minimum number of images.

    Args:
        annotation_path (str): Path to the annotation CSV file.
        min_samples (int): Minumum number of images the IDs must have.

    Returns:
        tuple: Unique IDs and corresponding counts, DataFrame containing train/validation set.
    """
    df = pd.read_csv(annotation_path)
    # Since we are creating a debug dataset we do not need a lot of data and we can
    # extract it only from the train set.
    if any(df["split"].str.isdigit()):
        train_set = df[df["split"] == 0]
    else:
        train_set = df[df["split"] == "train"]
    # Find unique IDs and their corresponding counts
    unique_ids = train_set["id"].value_counts()
    # Filter out IDs with less than the min_samples, on for each split
    unique_ids = unique_ids[unique_ids > min_samples]
    return unique_ids, train_set


def copy_imgs_to_debug_folder(reshuffled_data, images_path, debug_dataset_path):
    for img_name in reshuffled_data["filename"]:
        src_path = os.path.join(images_path, img_name)
        dst_path = os.path.join(debug_dataset_path, "images", os.path.basename(img_name))
        shutil.copy(src_path, dst_path)


@hydra.main(config_path="../../../config", config_name="config_FR", version_base="1.1")
def main(cfg):
    # Load the data
    for dataset in cfg.datasets.values():
        # create a new folder
        debug_dataset_path = create_dataset_folder(cfg.work_dir, dataset)
        unique_ids, data = extract_unique_ids(dataset.annotation_file)
        df_list = []
        # Define the maximum number of unique IDs to use for debugging purposes
        max_ids = max(5, len(unique_ids) * PERC_IDS_DEBUG)
        for n, (id, count) in tqdm(enumerate(unique_ids.iteritems(), 1), total=max_ids):
            if n > max_ids:
                # Stop the loop if the maximum number of unique IDs has been reached
                break
            # Filter samples with the current ID and assign a new ID to them
            subset_single_ID = data[data["id"] == id]
            subset_single_ID["id"] = n

            # To reduce the number of samples per identity we divide the total number of
            # images for that ID by 4 (number of splits we want to make)
            number_of_samples = math.ceil((count / 4))

            df_list, subset_single_ID = extract_samples(
                subset_single_ID, number_of_samples, df_list, "train"
            )

            df_list, subset_single_ID = extract_samples(
                subset_single_ID, number_of_samples, df_list, "val"
            )

            df_list, _ = extract_samples(
                subset_single_ID, number_of_samples, df_list, "test_same_ID"
            )
            df_list, _ = extract_samples(subset_single_ID, number_of_samples, df_list, "test")

        reshuffled_data = pd.concat(df_list, ignore_index=True)
        # Apply the function to update the column with the extracted values
        reshuffled_data["filename"] = reshuffled_data["filename"].apply(extract_last_part)
        reshuffled_data.to_csv(os.path.join(debug_dataset_path, "annotations.csv"), index=False)
        copy_imgs_to_debug_folder(reshuffled_data, dataset.images_dir, debug_dataset_path)


if __name__ == "__main__":
    print(
        "This data is not meant to be used for gathering results since the test set has the same ids of the training set. This data should be used only to debug the code and check the pipeline"
    )
    main()
