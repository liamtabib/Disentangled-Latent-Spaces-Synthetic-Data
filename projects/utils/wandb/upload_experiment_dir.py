import argparse
from pathlib import Path

import wandb


def upload_experiment_dir_to_wandb(experiment_dir) -> None:
    """
    Uploads an experiment directory to WandB.

    Args:
        experiment_dir (str): Path to the experiment directory to upload.
    """
    experiment_dir = Path(experiment_dir).absolute()

    run = wandb.init(
        project=args.project_name,
        entity="diffuse_datasets",
        name=f"Upload experiment dir: {experiment_dir.stem}",
    )
    artifact = wandb.Artifact(experiment_dir.stem, type="experiment")
    artifact.add_dir(str(experiment_dir))
    run.log_artifact(artifact)

    print("Uploading experiment, don't exit the program...")
    run.finish()
    print("Done uploading!")


if __name__ == "__main__":
    """Run this script with a saved experiment dir to upload it to WandB."""

    wandb.login()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project_name",
        help="Name of project to upload artifact to, i.e. `face_generation` or `face_recognition`.",
        choices=["face_generation", "face_recognition"],
    )
    parser.add_argument("--experiment_dir", help="Path to experiment dir to upload.")
    args = parser.parse_args()
    upload_experiment_dir_to_wandb(args.experiment_dir)

    wandb.finish()
