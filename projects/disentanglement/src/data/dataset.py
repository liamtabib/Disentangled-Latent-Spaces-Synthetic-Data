import os
import re
import glob
import random
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import Dataset
from ext.stylegan3_editing.notebooks.notebook_utils import run_alignment


def load_dataset_config():
    """Load dataset configuration from config.yaml"""
    try:
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)
        return config["dataset"]
    except (FileNotFoundError, KeyError):
        # Fallback to default config if config.yaml is not found
        return {
            "images_dir": "datasets/celebahq/images",
            "attributes_file": "datasets/celebahq/CelebAMask-HQ-attribute-anno.txt",
            "identities_file": "datasets/celebahq/identity_ID.csv",
        }


DATASET_CFG = load_dataset_config()


class CelebAHQDataset(Dataset):
    def __init__(self, work_dir, split_idx, sample_pairs=True, dataset_cfg=None, transform=None):
        super(CelebAHQDataset).__init__()
        self.work_dir = work_dir
        self.split_idx = split_idx
        self.transform = transform
        self.sample_pairs = sample_pairs

        if dataset_cfg is None:
            self.dataset_cfg = DATASET_CFG
        else:
            self.dataset_cfg = dataset_cfg

        self._setup_dataset_paths()
        self.sample = self._create_samples()

    def _setup_dataset_paths(self):
        self.img_dir = Path(self.work_dir, self.dataset_cfg["images_dir"])
        self.attributes_file = Path(self.work_dir, self.dataset_cfg["attributes_file"])
        if not self._check_exists():
            raise RuntimeError("Dataset not found or missing folders")

    def _check_exists(self):
        """Check the image folder and anotations.csv exist or if"""
        if self.img_dir.exists() and self.attributes_file.exists():
            return True
        return RuntimeError(
            f"Image dir {self.img_dir} and attribute file {self.attributes_file} "
            f"not found or missing folders"
        )

    def _create_samples(self):
        img_files = np.array(
            sorted(
                glob.glob(os.path.join(self.img_dir, "*")),
                key=lambda x: float(re.findall("(\d+)", x)[0]),
            )
        )
        sampled_img_path = img_files

        attributes = pd.read_csv(self.attributes_file, sep="\s+")
        assert attributes.shape[1] == 40, "The size of the attribute file is not correct"

        sampled_attributes = attributes[
            attributes.index.isin(
                [img_path.split(os.path.sep)[-1] for img_path in sampled_img_path]
            )
        ]
        sampled_attributes.replace({-1: 0, 1: 1}, inplace=True)
        sampled_attributes = sampled_attributes.to_numpy().astype(np.float32)
        self.attributes_names = np.array(attributes.columns)

        # Create random pairs
        if self.sample_pairs:
            samples = list(zip(img_files, sampled_attributes))
            random.shuffle(samples)
            random_pairs = np.array(list(zip(samples[:-1], samples[1:])), dtype=object)

            return {"images": random_pairs[:, :, 0], "attributes": random_pairs[:, :, 1]}

        else:
            return {"images": sampled_img_path, "attributes": sampled_attributes}

    def _img_preprocess(self, img):
        
        img = cv2.imread(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = self.transform(img)

        return img

    def __getitem__(self, index):
        img_path = self.sample["images"][index]
        attributes = self.sample["attributes"][index]

        if self.sample_pairs:
            img1 = self._img_preprocess(img_path[0])
            img2 = self._img_preprocess(img_path[1])
            return (
                (img1, img2),
                (
                    torch.tensor(attributes[0], dtype=torch.long),
                    torch.tensor(attributes[1], dtype=torch.long),
                ),
            )
        else:
            img = run_alignment(img_path)
            if img is not None:
                img = self._img_preprocess(img)
            else:
                img = self._img_preprocess(img_path)
            return img, attributes

    def __len__(self):
        return len(self.split_idx)


class NPairsCelebAHQDataset(CelebAHQDataset):
    """
    Outputs the positive pair for the contrastive learning based on identity IDs.
    Args:
        CelebAHQDataset:
    Returns:

    """

    def __init__(self, work_dir, split_idx, sample_pairs=True, dataset_cfg=None, transform=None):
        super().__init__(
            work_dir, split_idx, sample_pairs=sample_pairs, dataset_cfg=dataset_cfg, transform=transform
        )
        self.identity_ids = pd.read_csv(DATASET_CFG["identities"])
        self.samples = self._create_sample()

    def _create_sample(self):
        img_paths = glob.glob(os.path.join(self.img_dir, "*"))
        img_file = [int(img_dir.split(os.path.sep)[-1][:-4]) for img_dir in img_paths]
        samples = pd.DataFrame({"img_paths": img_paths, "img_file": img_file})
        samples["ids"] = samples.img_file
        dict_to_map = dict(zip(self.identity_ids.idx, self.identity_ids.identity_ID))
        samples["ids"] = samples["ids"].map(dict_to_map)
        return samples

    def __getitem__(self, idx):
        anchor = self.samples.loc[idx]
        positive = self.samples[self.samples.ids == anchor.ids]
        positive = positive.loc[np.random.choice(positive.index)]

        img_anchor = self._img_preprocess(anchor.img_paths)
        img_pos = self._img_preprocess(positive.img_paths)

        id_tensor= torch.tensor(positive.ids, dtype=torch.long)


        return (img_anchor, img_pos), id_tensor