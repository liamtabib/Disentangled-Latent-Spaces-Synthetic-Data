import math
import os
from pathlib import Path

import hydra
import pandas as pd
import tqdm
from omegaconf import DictConfig

import wandb
from projects.utils.data.dataset import parse_original_celeba_annotation


def check_attr_values(df, attr, attr_value, attr2, attr2_value):
    """
    Filter the dataframe based on the values of two attributes.

    Args:
        df (DataFrame): The input dataframe.
        attr (str): The name of the first attribute column.
        attr_value: The value to check for in the first attribute column.
        attr2 (str): The name of the second attribute column.
        attr2_value: The value to check for in the second attribute column.

    Returns:
        DataFrame: The filtered dataframe where both attribute columns have the specified values.
    """
    mask = (df[attr] == attr_value) & (df[attr2] == attr2_value)
    return df[~mask]


def clean_celeba_annotation(df):
    """
    Clean CelebA annotations by removing inconsistent or redundant attribute combinations.

    Args:
        df (DataFrame): The input dataframe containing CelebA annotations.

    Returns:
        DataFrame: The cleaned dataframe with consistent attribute combinations.
    """
    print("Cleaning inconsistent annotations...")
    attr_combinations = [
        ("Gray_Hair", 1, "Blond_Hair", 1),
        ("Gray_Hair", 1, "Black_Hair", 1),
        ("Gray_Hair", 1, "Brown_Hair", 1),
        ("Blond_Hair", 1, "Black_Hair", 1),
        ("Blond_Hair", 1, "Brown_Hair", 1),
        ("Black_Hair", 1, "Brown_Hair", 1),
        ("Gray_Hair", 1, "Young", 1),
        ("Bald", 1, "Receding_Hairline", 1),
        ("Bald", 1, "Bangs", 1),
        ("Receding_Hairline", 1, "Bangs", 1),
        ("Goatee", 1, "No_Beard", 1),
        ("Mustache", 1, "No_Beard", 1),
        ("Sideburns", 1, "No_Beard", 1),
        ("Goatee", 1, "Male", -1),
        ("Goatee", 1, "Mustache", -1),
    ]

    for attrs in attr_combinations:
        df = check_attr_values(df, attrs[0], attrs[1], attrs[2], attrs[3])

    df["Beard"] = df["No_Beard"].astype(int) * -1
    df = df.drop(columns=["Wearing_Lipstick", "Attractive", "No_Beard", "Sideburns"])
    return df


def extract_test_split(df, unique_ids, min_data_count=5):
    split_test = pd.DataFrame()
    IDs_to_keep = []
    for id in tqdm.tqdm(unique_ids, total=len(unique_ids)):
        # Extract all the rows for a single ID
        data = df[df["id"] == id]
        if len(data) < min_data_count:
            split_test = pd.concat([split_test, data])
        else:
            IDs_to_keep.append(id)
    split_test["split"] = "test"
    return IDs_to_keep, split_test


def extract_split(df, unique_ids, split_name, min_data_count=5):
    split_df = pd.DataFrame()
    IDs_to_keep = []
    for id in tqdm.tqdm(unique_ids, total=len(unique_ids)):
        data = df[df["id"] == id]
        if len(data) < min_data_count:
            split_df = pd.concat([split_df, data])
        else:
            IDs_to_keep.append(id)
    split_df["split"] = split_name
    return IDs_to_keep, split_df


def extract_val_split(df, unique_ids):
    split_val = pd.DataFrame()
    for id in tqdm.tqdm(unique_ids, total=len(unique_ids)):
        # Extract all the rows for a single ID
        data = df[df["id"] == id]

        split_val = pd.concat([split_val, data.iloc[[0]]])
        # Remove the extracted row from 'df'
        df = df.drop(data.index[0])
    split_val["split"] = "val"
    return df, split_val


def extract_test_same_ID_split(df, unique_ids):
    split_test_same_ID = pd.DataFrame()
    for id in tqdm.tqdm(unique_ids, total=len(unique_ids)):
        # Extract all the rows for a single ID
        data = df[df["id"] == id]

        split_test_same_ID = pd.concat([split_test_same_ID, data.iloc[[0]]])
        # Remove the extracted row from 'df'
        df = df.drop(data.index[0])
    split_test_same_ID["split"] = "test_same_ID"
    return df, split_test_same_ID


def extract_splits_train_with_same_IDs(df, unique_ids):
    split_train_a = pd.DataFrame()
    split_train_b = pd.DataFrame()
    for id in tqdm.tqdm(unique_ids, total=len(unique_ids)):
        # Extract all the rows for a single ID
        data = df[df["id"] == id]
        split_train_a = pd.concat([split_train_a, data.head(math.ceil(len(data) * 0.5))])
        split_train_b = pd.concat([split_train_b, data.tail(math.ceil(len(data) * 0.5))])
    split_train_a["split"] = "train_A_same_IDs"
    split_train_b["split"] = "train_B_same_IDs"
    return split_train_a, split_train_b


def extract_splits_training_with_different_IDs(df, unique_ids):
    split_train_a = pd.DataFrame()
    split_train_b = pd.DataFrame()
    split_val_a = pd.DataFrame()
    split_val_b = pd.DataFrame()
    split_test_same_ID_a = pd.DataFrame()
    split_test_same_ID_b = pd.DataFrame()
    for index, id in tqdm.tqdm(enumerate(unique_ids), total=len(unique_ids)):
        # Extract all the rows for a single ID
        data = df[df["id"] == id]
        if index % 2 == 0:
            split_val_a = pd.concat([split_val_a, data.iloc[[0]]])
            split_test_same_ID_a = pd.concat([split_test_same_ID_a, data.iloc[[1]]])
            data = data[2:]
            split_train_a = pd.concat([split_train_a, data])
        else:
            split_val_b = pd.concat([split_val_b, data.iloc[[0]]])
            split_test_same_ID_b = pd.concat([split_test_same_ID_b, data.iloc[[1]]])
            data = data[2:]
            split_train_b = pd.concat([split_train_b, data])
    split_train_a["split"] = "train_A_different_IDs"
    split_train_b["split"] = "train_B_different_IDs"
    split_val_a["split"] = "val_A_different_IDs"
    split_val_b["split"] = "val_B_different_IDs"
    split_test_same_ID_a["split"] = "test_same_ID_a_different_IDs"
    split_test_same_ID_b["split"] = "test_same_ID_b_different_IDs"

    return (
        split_val_a,
        split_test_same_ID_a,
        split_train_a,
        split_val_b,
        split_test_same_ID_b,
        split_train_b,
    )


def split_dataset_by_IDs(df):
    """
    Split the dataset based on IDs.

    Args:
        df (DataFrame): The input dataframe containing CelebA annotations.

    Returns:
        DataFrame: The split dataframe based on IDs.
    """
    print("Split dataset by IDs...")
    unique_ids = df["id"].unique()
    unique_ids, split_test = extract_test_split(df, unique_ids)
    (
        split_val_a_different_IDs,
        split_test_same_ID_a_different_IDs,
        split_train_a_different_IDs,
        split_val_b_different_IDs,
        split_test_same_ID_b_different_IDs,
        split_train_b_different_IDs,
    ) = extract_splits_training_with_different_IDs(df, unique_ids)

    df, split_val = extract_val_split(df, unique_ids)

    df, split_test_same_ID = extract_test_same_ID_split(df, unique_ids)

    split_train_a_same_IDs, split_train_b_same_IDs = extract_splits_train_with_same_IDs(
        df, unique_ids
    )
    df["split"] = "train"

    # # Combine the datasets and save
    return pd.concat(
        [
            df,
            split_val,
            split_train_a_same_IDs,
            split_train_b_same_IDs,
            split_train_a_different_IDs,
            split_train_b_different_IDs,
            split_val_a_different_IDs,
            split_val_b_different_IDs,
            split_test_same_ID_a_different_IDs,
            split_test_same_ID_b_different_IDs,
            split_test,
            split_test_same_ID,
        ],
        ignore_index=True,
    )


def split_dataset_by_attrs(df):
    """
    Split the dataset based on attributes.

    Args:
        df (DataFrame): The input dataframe containing CelebA annotations.

    Returns:
        DataFrame: The split dataframe based on attributes.
    """
    print("Split dataset by attributes...")
    split_train_no_attr, split_train_with_attr, split_val, split_test_same_ID = [], [], [], []
    unique_ids = df["id"].drop_duplicates().tolist()
    for id in tqdm.tqdm(unique_ids, total=len(unique_ids)):
        # Extract all rows for a single ID
        df_subset = df[df["id"] == id]

        # Common IDs for combination ('Brown_Hair', 'Eyeglasses', 'Beard'): 596

        # Extract all images for this ID that do not have these attributes
        data_no_attr = df_subset[
            (df_subset["Eyeglasses"] == -1)
            & (df_subset["Brown_Hair"] == -1)
            & (df_subset["Beard"] == -1)
        ]

        # Extract all images for this ID that have at least one of these attributes
        data_with_attr = df_subset[
            (df_subset["Eyeglasses"] == 1)
            | (df_subset["Brown_Hair"] == 1)
            | (df_subset["Beard"] == 1)
        ]
        if len(data_no_attr) > 0:
            split_train_no_attr.append(data_no_attr)
        if len(data_with_attr) > 0:
            if len(data_with_attr) >= 3:
                split_val.append(data_with_attr.iloc[[0]])
                split_test_same_ID.append(data_with_attr.iloc[[1]])
                split_train_with_attr.append(data_with_attr[2:])
            else:
                split_train_with_attr.append(data_with_attr)

    split_train_no_attr = pd.concat(split_train_no_attr)
    split_train_with_attr = pd.concat(split_train_with_attr)
    split_val = pd.concat(split_val)
    split_test_same_ID = pd.concat(split_test_same_ID)
    split_train_no_attr["split"] = "train_no_attr"
    split_train_with_attr["split"] = "train_with_attr"
    split_val["split"] = "val_with_attr"
    split_test_same_ID["split"] = "test_same_ID_with_attr"
    # # Combine the datasets and save
    return pd.concat(
        [
            split_train_with_attr,
            split_train_no_attr,
            split_val,
            split_test_same_ID,
        ],
        ignore_index=True,
    )


def load_original_celeba_annotation_file(ROOT):
    """
    Downloads the original CelebA dataset if it doesn't exist locally and
    returns the annotations as a Pandas DataFrame.

    Args:
        ROOT (str): The root directory of the dataset.

    Returns:
        pd.DataFrame: The CelebA dataset annotations as a DataFrame.
    """
    annotation_file = ROOT / Path("celeba/annotations.csv")
    if not os.path.exists(annotation_file):
        run = wandb.init()
        artifact = run.use_artifact("diffuse_datasets/datasets/celeba:v1", type="dataset")
        artifact.download()
        df = parse_original_celeba_annotation(ROOT / Path("celeba"))
        return df
    return pd.read_csv(annotation_file)


@hydra.main(config_path="../../../config", config_name="config_FR", version_base="1.1")
def main(cfg: DictConfig) -> None:
    data = load_original_celeba_annotation_file(cfg.data_root)
    data = clean_celeba_annotation(data)

    df_by_attrs = split_dataset_by_attrs(data)
    df_dy_IDs = split_dataset_by_IDs(data)
    df = pd.concat([df_by_attrs, df_dy_IDs], ignore_index=True)
    print("Images per split ", df.groupby("split").size())

    df.to_csv(cfg.data_root / Path("cropped_celeba/annotations_shuffled.csv"), index=False)


if __name__ == "__main__":
    main()
