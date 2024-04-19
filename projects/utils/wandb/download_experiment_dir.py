import argparse
from pathlib import Path

import wandb


def download_experiment_dir_from_wandb(args) -> None:
    """
    Download an experiment directory from WandB.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    run = wandb.init()
    log_dir = Path(f"../../../logs/{args.project_name}")  # Change if desired.
    experiment_url = (
        f"diffuse_datasets/{args.project_name}/{args.experiment_name}:{args.experiment_version}"
    )
    experiment_save_dir = log_dir / args.experiment_name

    artifact = run.use_artifact(experiment_url, type="experiment")
    _ = artifact.download(experiment_save_dir)
    run.finish()
    print("Done downloading!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project_name",
        help="Name of project to download artifact from, i.e. `face_generation` or `face_recognition`.",
        choices=["face_generation", "face_recognition"],
        default="face_generation",
    )
    parser.add_argument(
        "--experiment_name", help="Experiment name to download.", default="20230303_170400"
    )
    parser.add_argument(
        "--experiment_version", help="Experiment version to download.", default="v0"
    )
    args = parser.parse_args()
    download_experiment_dir_from_wandb(args)
