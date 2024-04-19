from torch.utils.data import ConcatDataset, DataLoader, DistributedSampler

from projects.utils.data.dataset import PairedSampleDataset, SingleSampleDataset


def get_data_loader(
    dl_cfg,
    datasets,
    FAR_settings=None,
    augmentation=None,
    rank=0,
    num_gpus=1,
    exclude_datasets=False,
    duplicated_IDs=False,
):
    """Create a data loader for training or evaluation.

    Args:
        dl_cfg (dict): Configuration for the data loader.
        datasets (dict): Dictionary containing the dataset names and their configurations.
        FAR_settings (dict): Dictionary containing the configuration for the FAR samples.
        augmentation (callable, optional): Data augmentation function. Defaults to None.
        rank (int, optional): Rank of the current process in distributed training. Defaults to 0.
        num_gpus (int, optional): Number of GPUs used for training. Defaults to 1.
        exclude_datasets(list, optional): The names of the datasets you want to esclude.
        duplicated_IDs(bool, optional): Is set to true if you have two datasets with the same IDs.

    Returns:
        tuple: A tuple containing the data loader and the number of unique IDs in the dataset.
    """
    cat_datasets, samplers, ids_unique = [], [], []
    Dataset = PairedSampleDataset if FAR_settings else SingleSampleDataset
    for dataset_name, dataset_cfg in datasets.items():
        if exclude_datasets and dataset_name in exclude_datasets:
            print("skipping ", dataset_cfg)
            continue

        print(f"Loading the {dataset_name} dataset")
        start_id = 0 if len(ids_unique) == 0 or duplicated_IDs else max(ids_unique)
        dataset = Dataset(dataset_cfg, dl_cfg.split, start_id, FAR_settings, augmentation)

        list_ids = dataset.ids_unique.tolist()
        if len(list_ids) == 0:  # it checks if the dataset has IDs for the split we want to load
            print(f"Data {dataset_name} has 0 unique ids so it is skipped")
            continue

        sampler = None
        if num_gpus > 1:
            sampler = DistributedSampler(
                dataset, shuffle=dl_cfg.shuffle, num_replicas=num_gpus, rank=rank
            )
            dl_cfg.shuffle = False

        samplers.append(sampler)
        cat_datasets.append(dataset)
        ids_unique += dataset.ids_unique.tolist()

    new_dl_cfg = {key: value for key, value in dl_cfg.items() if key != "split"}
    return (
        DataLoader(ConcatDataset(cat_datasets), sampler=sampler, pin_memory=True, **new_dl_cfg),
        ids_unique,
    )
