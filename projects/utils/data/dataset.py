import random
from functools import partial
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

ATTRIBUTES = [
    "5_o_Clock_Shadow",
    "Arched_Eyebrows",
    "Attractive",
    "Bags_Under_Eyes",
    "Bald",
    "Bangs",
    "Big_Lips",
    "Big_Nose",
    "Black_Hair",
    "Blond_Hair",
    "Blurry",
    "Brown_Hair",
    "Bushy_Eyebrows",
    "Chubby",
    "Double_Chin",
    "Eyeglasses",
    "Goatee",
    "Gray_Hair",
    "Heavy_Makeup",
    "High_Cheekbones",
    "Male",
    "Mouth_Slightly_Open",
    "Mustache",
    "Narrow_Eyes",
    "No_Beard",
    "Oval_Face",
    "Pale_Skin",
    "Pointy_Nose",
    "Receding_Hairline",
    "Rosy_Cheeks",
    "Sideburns",
    "Smiling",
    "Straight_Hair",
    "Wavy_Hair",
    "Wearing_Earrings",
    "Wearing_Hat",
    "Wearing_Lipstick",
    "Wearing_Necklace",
    "Wearing_Necktie",
    "Young",
]

DATASET_CFG = {
    "image_dir": "images",
    "identity": "annotations/identity_CelebA.txt",
    "attributes": "annotations/list_attr_celeba.txt",
    "bbox": "annotations/list_bbox_celeba.txt",
    "landmarks": "annotations/list_landmarks_align_celeba.txt",
    "split": "annotations/list_eval_partition.txt",
    "test": "annotations/test_label.txt",
    "train": "annotations/train_label.txt",
    "val": "annotations/val_label.txt",
}


def parse_original_celeba_annotation(dataset_path):
    read_csv_func = partial(pd.read_csv, low_memory=False, sep=" ", header=None)
    df_i = read_csv_func(dataset_path / DATASET_CFG["identity"], names=["filename", "id"])
    df_s = read_csv_func(dataset_path / DATASET_CFG["split"], names=["filename", "split"])
    # Skip first row containing column names.
    df_a = read_csv_func(
        dataset_path / DATASET_CFG["attributes"],
        names=["filename", *ATTRIBUTES],
        skiprows=1,
        skipinitialspace=True,
    )

    # Merge annotations into single DataFrame.
    df = pd.merge(pd.merge(df_i, df_s, on="filename"), df_a, on="filename")
    df["split"] = df["split"].replace({0: "train", 1: "val", 2: "test"})
    return df


class BaseDataset(Dataset):
    """Base class for loading data.

    Annotations are loaded and made available as a DataFrame under `self.df_anno`.

    Note that if this class is instantiated without implementations of `_create_samples`
    and `__get_item`, the samples will not be accessible.
    """

    def __init__(self, dataset_cfg, split, start_id, FAR_settings, augmentation=None):
        super().__init__()
        self.FAR_settings = FAR_settings
        self._setup_dataset_paths(dataset_cfg)
        self._validate_dataset_split(split)
        self.df_anno = self._load_annotations()
        if start_id > 0:
            self._id_mapping(start_id)
        self._get_num_ids()
        self._check_attributes()
        self.augmentation = augmentation

        try:
            self.samples = self._create_samples()
        except NotImplementedError:
            print("Warning: `create_samples` not implemented.Dataset will not be iterable.")
            self.samples = None

    def _id_mapping(self, start_id):
        self.df_anno["id"] += start_id

    def _check_attributes(self):
        self.missing_attributes = any(
            [attr for attr in ATTRIBUTES if attr not in self.df_anno.columns]
        )
        if self.missing_attributes:
            print(f"Missing celeba attributes from the annotation file {self.annotation_file}")
            self.df_anno = self.df_anno[["filename", "id", "split"]]

    def _get_num_ids(self):
        self.ids_unique = self.df_anno["id"].unique()

    def _setup_dataset_paths(self, dataset_cfg):
        self.img_dir = Path(dataset_cfg.images_dir)
        self.annotation_file = Path(dataset_cfg.annotation_file)
        if not self._check_exists():
            raise RuntimeError("Dataset not found or missing folders")

    def _validate_dataset_split(self, split):
        # Map split type to ID found in annotations.
        if split.isdigit():
            valid_splits = [0, 1, 2, 3]  # train  # val  # test  # test_same_ID
        else:
            valid_splits = [
                "test_same_ID_simplified",
                "train_simplified",
                "val_simplified",
                "test_simplified",
                "test_same_ID",
                "train",
                "val",
                "test",
                "train_A_same_IDs",
                "train_B_same_IDs",
                "train_A_different_IDs",
                "val_A_different_IDs",
                "test_same_ID_A_different_IDs",
                "test_A_different_IDs",
                "train_B_different_IDs",
            ]
        assert split in valid_splits, f"Split type must be one of {valid_splits}."
        self.split = split

    def _load_annotations(self):
        print("Loading annotations...")
        # Load annotations from file if they have already been parsed
        if self.annotation_file.is_file():
            df = pd.read_csv(self.annotation_file, low_memory=False)
        else:
            df = parse_original_celeba_annotation(self.dataset_path)
            # Save annotations
            df.to_csv(self.annotation_file)

        # Remove samples not in current split.
        df = df[df["split"] == self.split]
        return df

    def _create_samples(self):
        """Override this method to create dataset samples to iterate over."""
        raise NotImplementedError

    def __getitem__(self, index):
        """Override this method to return a loaded sample."""
        raise NotImplementedError

    def _check_exists(self):
        """Check the image folder and anotations.csv exist or if"""
        if self.img_dir.exists() and self.annotation_file.exists():
            return True
        return all((self.annotation_file / fname).exists() for fname in DATASET_CFG.values())

    def __len__(self):
        return len(self.samples)

    def get_random_labels(self, n: int) -> torch.Tensor:
        assert n > 0
        return torch.as_tensor(self.df_anno[ATTRIBUTES].sample(n, replace=True).values).to(
            torch.float32
        )

    def get_label(self, id):
        # TODO: create new function to avoid constant reading of df_anno
        # e.g. preload ids
        ids = [row["id"] for _, row in self.df_anno.iterrows()]
        onehot = np.zeros(self.label_dim, dtype=np.float32)

        onehot[ids[id]] = 1
        label = onehot

        return label.copy()

    def load_img(self, path):
        img = cv2.imread(str(path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img / 255.0
        if self.augmentation is not None:
            img = self.augmentation(image=img)["image"]
        return img

    @property
    def label_dim(self):
        # TODO: create new function to avoid constant reading of df_anno
        #  e.g. preload ids
        # TODO: New system for calculating the number of IDs in the data
        # There are in total 8192 unique IDs but the actual numbers range from
        #  1 to 10177
        ids = [row["id"] for _, row in self.df_anno.iterrows()]
        return max(set(ids))


class SingleSampleDataset(BaseDataset):
    def __init__(self, dataset_cfg, split, start_id, FAR_settings, augmentation):
        super().__init__(dataset_cfg, split, start_id, FAR_settings, augmentation)

    def _create_sample(self, row):
        if self.missing_attributes == False:
            attributes = row[ATTRIBUTES]
            attr_vec = attributes.values.astype(np.float32)
            attr2idx = {attr_name: idx for idx, attr_name in enumerate(ATTRIBUTES)}
        else:
            attr_vec = 0
            attr2idx = 0

        return {
            "id": row["id"],
            "path": self.img_dir / row["filename"],
            "img_name": row["filename"],
            "attributes": attr_vec,
            "attr2idx": attr2idx,
        }

    def _create_samples(self):
        samples = [self._create_sample(row) for _, row in self.df_anno.iterrows()]
        return random.shuffle(samples)

    def __getitem__(self, index):
        sample_info = self.samples[index]
        img = self.load_img(sample_info["path"])
        return {
            "id": sample_info["id"] - 1,
            "img": img,
            "img_name": sample_info["img_name"],
            "attributes": sample_info["attributes"],
            "path": str(sample_info["path"]),
        }

    def get_attributes(self, index):
        return self.samples[index]["attributes"]


class PairedSampleDataset(BaseDataset):
    def __init__(self, dataset_cfg, split, start_id, FAR_settings, augmentation):
        super().__init__(dataset_cfg, split, start_id, FAR_settings, augmentation)

    def _create_sample(self, row):
        return {
            "id": row["id"],
            "path": self.img_dir / row["filename"],
            "img_name": row["filename"],
        }

    def _create_samples(self):
        """
        Create pairs of samples with their corresponding labels.

        This method generates pairs of samples from the dataset, where each pair consists of two samples
        and a label indicating whether they have the same ID or not. The pairs are created in a way that
        ensures a higher number of positive (same ID) pairs compared to negative (different ID) pairs.

        Returns:
            all_pairs (list): A list of tuples, where each tuple represents a pair of samples and its label.
                Each tuple has the form (sample1, sample2, label), where sample1 and sample2 are dictionaries
                representing the samples and label is an integer (1 for positive pairs, 0 for negative pairs).
        """
        all_samples = [self._create_sample(row) for _, row in self.df_anno.iterrows()]
        id_to_samples = {}
        # Create a dict with the samples using the person ID as key
        for sample in all_samples:
            sample_id = sample["id"]
            if sample_id in id_to_samples:
                id_to_samples[sample_id].append(sample)
            else:
                id_to_samples[sample_id] = [sample]
        all_pairs = []

        for samples_list in id_to_samples.values():
            num_samples = len(samples_list)
            if num_samples > 1:
                positive_pairs = []
                for i in range(num_samples):
                    for j in range(i + 1, num_samples):
                        sample1 = samples_list[i]
                        sample2 = samples_list[j]
                        positive_pairs.append((sample1, sample2, 1))  # Positive pair
                    if len(positive_pairs) > self.FAR_settings.max_samples_per_ID:
                        break
                all_pairs.extend(positive_pairs)
                if len(all_pairs) > self.FAR_settings.max_samples:
                    break

        num_positive_pairs = len(all_pairs)
        negative_pairs = []
        for _ in range(num_positive_pairs):
            pair_ids = random.sample(list(id_to_samples.keys()), 2)
            sample1 = random.choice(id_to_samples[pair_ids[0]])
            sample2 = random.choice(id_to_samples[pair_ids[1]])
            negative_pairs.append((sample1, sample2, 0))  # Negative pair
        all_pairs.extend(negative_pairs)
        return all_pairs

    def __getitem__(self, index):
        sample_info = self.samples[index]
        return (
            self.load_img(sample_info[0]["path"]),
            self.load_img(sample_info[1]["path"]),
            sample_info[2],
        )
