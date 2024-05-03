# Disentangled Latent Spaces Project

## Environment Setup

### Dataset
- **CelebAMask-HQ Dataset**: Download the CelebAHQ dataset from [here](https://github.com/switchablenorms/CelebAMask-HQ) and place it into the `datasets/` directory.

### Pre-trained Models
- **Model Resources**: Download all necessary pre-trained models and place them into `projects/disentanglement/pretrained_models`. These include:
  - StyleGAN3
  - StyleGAN3 Encoder
  - Landmark detectors

## Code Structure
- **The work**: The `projects/disentanglement/` directory contains the implementation of this project.
- **External modules**: The `ext/` directory contains all external modules such as StyleGAN3, StyleGAN3 Encoder, and insightface (ArcFace recognition model). Clone those repositories and place them under `ext/`.

## Training the T transformation:
To run the project, follow these steps:
1. Install required Python packages:
   pip install -r requirements.txt
2. Start the training process:
    python projects/disentanglement/train.py

