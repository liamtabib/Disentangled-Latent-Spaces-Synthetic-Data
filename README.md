# Disentangled Latent Spaces for Synthetic Data

**[Official Published Paper](https://www.diva-portal.org/smash/record.jsf?pid=diva2:1887438)**

This project implements bijective transformations for disentangled latent representations in synthetic facial data generation. The goal is to learn a transformation that separates identity information from other facial attributes in StyleGAN3's latent space, enabling more controlled synthetic data generation.

## Quick Setup

Getting this running is straightforward, though you'll need to grab some external dependencies and datasets:

```bash
# Install the package in development mode
pip install -e .

# Automatically setup external dependencies (StyleGAN3, InsightFace, etc.)
chmod +x scripts/setup_deps.sh && ./scripts/setup_deps.sh
```

You'll also need to download the [CelebAMask-HQ dataset](https://github.com/switchablenorms/CelebAMask-HQ) and place it in `datasets/celebahq/`, plus some pre-trained models that go in `projects/disentanglement/pretrained_models/`.

Once everything's set up, start training with:
```bash
python projects/disentanglement/train.py
```

The training will use your `config.yaml` settings and begin learning the disentangled representations.

## How It Works

The core idea is pretty elegant: we use a NICE (Non-linear Independent Components Estimation) network to create a bijective transformation that separates identity information from other facial attributes in StyleGAN3's latent space. 

The system (called DisGAN) combines a pre-trained StyleGAN3 encoder with our trainable NICE transformation. During training, we use several loss functions working together:

- **Switch loss** maintains identity consistency across transformations
- **Contrastive loss** pushes different identities apart in latent space  
- **Landmark loss** preserves facial structure
- **Discriminator loss** creates adversarial dynamics to separate identity from attributes

We evaluate the results with face recognition distance, DCI metrics, and PCA analysis to measure how well the representations are actually disentangled.

## Configuration

Everything's controlled through `config.yaml` - you can adjust batch sizes, learning rates, loss weights, dataset paths, and toggle different loss functions without touching the code. This makes it easy to experiment with different training setups.

## Project Structure

```
projects/disentanglement/
├── train.py                    # Main training script
├── src/
│   ├── data/                  # Dataset handling  
│   ├── metrics/               # Evaluation metrics
│   ├── visualisations/        # Analysis tools
│   ├── models.py              # StyleGAN3, NICE, DisGAN definitions
│   └── losses.py              # Loss function implementations
```

## Console Commands

After installation, you get some handy commands:
```bash
train-disentanglement          # Main training
train-discriminators           # Train identity discriminators
```

## Common Issues

- **Import errors**: Run `pip install -e .` first
- **Missing dependencies**: Run `./scripts/setup_deps.sh`  
- **CUDA problems**: Check device settings in `config.yaml`
- **Dataset not found**: Verify paths in `config.yaml`

