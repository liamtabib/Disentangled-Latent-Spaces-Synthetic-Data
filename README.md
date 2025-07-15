# Disentangled Latent Spaces for Synthetic Data

**[Official Published Paper](https://www.diva-portal.org/smash/record.jsf?pid=diva2:1887438)**

This repository contains the implementation of a Master's thesis project that develops bijective transformations for disentangled latent representations in synthetic facial data generation. The core idea is to learn a transformation that can separate identity information from other facial attributes in the latent space, enabling more controlled synthetic data generation.

## Getting Started

To get this project running on your system, you'll need to install the package and set up the necessary dependencies. The process is designed to be straightforward, though you'll need to download some external repositories and datasets.

First, install the package in development mode by running `pip install -e .` in the project root. This will set up all the Python dependencies and make the project modules available for import.

Next, you'll need to set up the external dependencies. We've provided a script to automate this process: `chmod +x scripts/setup_deps.sh && ./scripts/setup_deps.sh`. This script will clone the necessary repositories including StyleGAN3, StyleGAN3 editing tools, and InsightFace into the `ext/` directory.

Once the dependencies are set up, you'll need to download the CelebAMask-HQ dataset from the [official repository](https://github.com/switchablenorms/CelebAMask-HQ) and place it in the `datasets/celebahq/` directory. You'll also need to download the pre-trained models and place them in `projects/disentanglement/pretrained_models/`.

With everything in place, you can start training by running `python projects/disentanglement/train.py`. The training process will use the configuration settings from `config.yaml` and begin learning the disentangled representations.

## Configuration

The project uses a centralized configuration system through the `config.yaml` file. This makes it easy to adjust training parameters such as batch sizes, learning rates, and loss function weights without modifying the source code. You can also configure dataset paths and model locations to match your system setup. The configuration file includes toggles for different loss functions, allowing you to experiment with various combinations during training.

## Architecture

This implementation builds on several key components working together to achieve disentangled latent representations. The foundation is a pre-trained StyleGAN3 synthesis network that generates high-quality facial images from latent codes. On top of this, we implement a NICE (Non-linear Independent Components Estimation) network that provides the bijective transformation crucial for disentanglement. The complete system, called DisGAN, combines a StyleGAN3 encoder with our trainable NICE transformation.

The approach supports multiple loss functions that can be combined during training. The switch loss encourages the model to maintain identity consistency, while contrastive loss helps separate different identities in the latent space. Landmark loss preserves facial structure, and discriminator loss creates an adversarial training dynamic that drives the separation of identity information from other facial attributes. The system includes comprehensive evaluation metrics including face recognition distance, DCI (Disentanglement, Completeness, Informativeness), and PCA analysis to measure the quality of the learned representations.

## Project Structure

The codebase is organized within the `projects/disentanglement/` directory. The main training script `train.py` coordinates the entire training process, while the `src/` directory contains the core implementation modules. Dataset handling logic resides in `src/data/`, evaluation metrics are implemented in `src/metrics/`, and visualization tools for analyzing the learned representations are in `src/visualisations/`. The model definitions including StyleGAN3, NICE, and DisGAN are in `models.py`, while the various loss functions are implemented in `losses.py`.

## Troubleshooting

If you encounter import errors, make sure you've installed the package in development mode using `pip install -e .`. Missing external dependencies can usually be resolved by running the setup script `./scripts/setup_deps.sh`. For CUDA-related issues, check the device settings in your `config.yaml` file to ensure they match your system configuration. If the system can't find your dataset, verify that the paths in `config.yaml` point to the correct locations where you've placed the CelebAMask-HQ data and pre-trained models.

## Additional Scripts

After installation, the package provides convenient console commands for training. You can use `train-disentanglement` to start the main training process or `train-discriminators` to specifically train the identity discriminator networks. These commands are equivalent to running the Python scripts directly but provide a cleaner interface for regular use.

