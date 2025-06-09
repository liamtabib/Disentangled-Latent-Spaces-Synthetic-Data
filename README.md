# Repository for Masters Thesis in Machine Learning: Disentangled Latent Space for Synthetic Data

## Environment Setup

### Dataset
- **CelebAMask-HQ Dataset**: Download the CelebAHQ dataset from [here](https://github.com/switchablenorms/CelebAMask-HQ) and place it into the `datasets/` directory.

### Pre-trained Models
- **Model Resources**: Download all necessary pre-trained models and place them into `src/disentanglement/pretrained_models`. These include:
  - StyleGAN3 Generator
  - StyleGAN3 Encoder
  - Landmark detectors

## Code Structure
- **The main directory**: The `src/disentanglement/` directory contains the implementation of this project.
- **External modules**: The `ext/` directory contains all external modules such as StyleGAN3, StyleGAN3 Encoder, and insightface (ArcFace recognition model). Clone those repositories in github and place them under a newly created `ext/` directory under root.

## Training the Disentangled Latent Space:
To start the training of the bijective transformation:

0. Setup the environment, including the dataset, pre-trained models, and external repositories.
1. Install required Python packages:
   pip install -r requirements.txt
2. Start the training process:
    python src/train.py

