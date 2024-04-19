import os

import hydra
from omegaconf import DictConfig

import wandb


@hydra.main(config_path="../../../projects/config", config_name="config_FR", version_base="1.1")
def main(cfg: DictConfig) -> None:
    # Use debug datasets if in debug mode, otherwise use regular datasets
    datasets = cfg.debug_datasets.values() if cfg.debug_mode else cfg.datasets.values()

    for dataset in datasets:
        # Initialize W&B run for dataset upload
        run = wandb.init(
            project="datasets",
            entity="diffuse_datasets",
            name=f"Upload dataset {dataset['dataset_type']}: {dataset['dataset_version']}",
        )
        try:
            # Create artifact for dataset
            artifact = wandb.Artifact(dataset["dataset_type"], type="dataset")

            # Add dataset directory to artifact
            dataset_path = os.path.join(cfg["data_root"], dataset["dataset_type"])
            artifact.add_dir(dataset_path)

            # Log artifact and alias with dataset version
            run.log_artifact(artifact, aliases=[dataset["dataset_version"]])
            print(f"Dataset {dataset['dataset_type']} {dataset['dataset_version']} uploaded.")
        except Exception as e:
            # Handle any exceptions and log error message
            print(
                f"Error uploading dataset {dataset['dataset_type']} {dataset['dataset_version']}: {e}"
            )
        finally:
            # Finish W&B run
            run.finish()


if __name__ == "__main__":
    wandb.login()
    main()
    wandb.finish()
