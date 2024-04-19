import os

import pandas as pd
import tqdm


def update_label_ids(data, start_id, id_mapping):
    """
    Update the label IDs in the given DataFrame with new IDs from the provided mapping.

    Args:
        data (DataFrame): Input DataFrame containing the 'label' column.
        start_id (int): Starting ID value for the new labels.
        id_mapping (dict): Dictionary mapping original label IDs to new IDs.

    Returns:
        int: The next available ID value.
        dict: Updated ID mapping.
    """
    unique_labels = data["label"].unique()
    for label in unique_labels:
        if int(label) not in id_mapping:
            id_mapping[int(label)] = start_id
            start_id += 1
    return start_id, id_mapping


def create_ID_mapping(file_mapping, folder_path):
    """
    Create a mapping of label IDs for the provided CSV files.

    Args:
        file_mapping (dict): Dictionary mapping file types to file names.
        folder_path (str): Path to the folder containing the CSV files.

    Returns:
        dict: Mapping of original label IDs to new IDs.
    """
    id_mapping = {}
    current_id = 1
    for _, file_name in file_mapping.items():
        # Read the CSV file into a DataFrame
        file_path = os.path.join(folder_path, file_name)
        data = pd.read_csv(file_path).dropna(how="all")
        current_id, id_mapping = update_label_ids(data, current_id, id_mapping)
    return id_mapping


def split_dataset_by_IDs(df, data_type="", IDs_for_testing=100):
    """
    Split the dataset based on IDs.

    Args:
        df (DataFrame): The input dataframe containing CelebA annotations.
        IDs_for_testing (int): Number of IDs to use for testing.

    Returns:
        DataFrame: The split dataframe based on IDs.
    """
    print("Split dataset by IDs...")

    split_train = pd.DataFrame(columns=df.columns)
    split_val = pd.DataFrame(columns=df.columns)
    split_test = pd.DataFrame(columns=df.columns)
    split_test_same_ID = pd.DataFrame(columns=df.columns)

    unique_ids = df["id"].value_counts()
    for index, id in tqdm.tqdm(enumerate(unique_ids.keys()), total=len(unique_ids)):
        # Extract all the rows for a single ID
        data = df[df["id"] == id]
        # IDs_for_testing is an Arbitrary value used as a treshold to extract the test split.
        if index < IDs_for_testing:
            split_test = pd.concat([split_test, data])
            continue
        split_val = pd.concat([split_val, data.iloc[:2]])
        split_test_same_ID = pd.concat([split_test_same_ID, data.iloc[2:4]])
        data = data.iloc[4:]
        split_train = pd.concat([split_train, data])
    split_train["split"] = "train" + data_type
    split_val["split"] = "val" + data_type
    split_test["split"] = "test" + data_type
    split_test_same_ID["split"] = "test_same_ID" + data_type
    # # Combine the datasets and save
    return pd.concat(
        [split_train, split_val, split_test, split_test_same_ID],
        ignore_index=True,
    )


def merge_csv_files(folder_path):
    """
    Merge the original CSV annotation files in the faceid repository.

    Args:
        folder_path (str): Path to the folder containing the CSV files.

    Returns:
        DataFrame: Merged dataset with updated label IDs and splits.
    """
    merged_data = []

    file_mapping = {"test": "testing.csv", "val": "val_triplets.csv", "train": "train_triplets.csv"}

    id_mapping = create_ID_mapping(file_mapping, folder_path)

    for file_type, file_name in file_mapping.items():
        # Read the CSV file into a DataFrame
        file_path = os.path.join(folder_path, file_name)
        data = pd.read_csv(file_path).dropna(how="all")

        # Reset the index of the DataFrame
        data.reset_index(drop=True, inplace=True)

        data["id"] = data["label"].replace(id_mapping)

        # Convert the values in the "id" column to integers
        data["id"] = data["id"].astype(int)

        data["split"] = file_type
        merged_data.append(data)

    return pd.concat(merged_data)


def dataset_cleanup(df):
    """
    Clean up the dataset by filtering rows based on conditions.

    Args:
        df (DataFrame): Input dataframe containing CelebA annotations.

    Returns:
        DataFrame: Cleaned dataset.
    """
    print(f"Initial dataset size {len(df)}")
    # Create a boolean mask to filter rows
    mask = (df["glasses_present"] != 1) & (df["face_mask_present"] != 1)
    # Apply the mask to exclude rows
    df = df[mask]
    # Calculate means and standard deviations
    mean_ex = df["rotation_pcs_ex"].median()
    std_dev_ex = df["rotation_pcs_ex"].std()

    mean_ey = df["rotation_pcs_ey"].mean()
    std_dev_ey = df["rotation_pcs_ey"].std()

    mean_ez = df["rotation_pcs_ez"].mean()
    std_dev_ez = df["rotation_pcs_ez"].std()
    std_dev_threshold = 0.9

    # Filter the DataFrame
    df = df[
        (df["rotation_pcs_ex"] >= mean_ex - std_dev_threshold * std_dev_ex)
        & (df["rotation_pcs_ex"] <= mean_ex + std_dev_threshold * std_dev_ex)
        & (df["rotation_pcs_ey"] >= mean_ey - std_dev_threshold * std_dev_ey)
        & (df["rotation_pcs_ey"] <= mean_ey + std_dev_threshold * std_dev_ey)
        & (df["rotation_pcs_ez"] >= mean_ez - std_dev_threshold * std_dev_ez)
        & (df["rotation_pcs_ez"] <= mean_ez + std_dev_threshold * std_dev_ez)
    ]
    print(f"after removing glasses and face masks {len(df)}")
    return df


def remove_unused_columns(df):
    """
    Remove unused columns from the dataframe.

    Args:
        df (DataFrame): Input dataframe.

    Returns:
        DataFrame: Dataframe with only the specified columns.
    """
    # List of columns to keep
    columns_to_keep = ["filename", "label", "id", "split"]
    # Select and keep only the specified columns
    return df[columns_to_keep]


if __name__ == "__main__":
    input_folder_path = "/mnt/nas/RD/Projects/datasets/faceid"
    output_folder = "/mnt/nas/RD/Projects/diffuse/datasets/faceid/seye_nir"
    df = merge_csv_files(input_folder_path)
    simplified_df = dataset_cleanup(df)

    for dataset, data_type in zip([df, simplified_df], ["", "_simplified"]):
        dataset = split_dataset_by_IDs(dataset, data_type=data_type)
        dataset = remove_unused_columns(dataset)
        print(f"Number of images per split {data_type}", dataset.groupby("split").size())
        dataset.to_csv(output_folder + f"/annotations{data_type}.csv", index=False)
