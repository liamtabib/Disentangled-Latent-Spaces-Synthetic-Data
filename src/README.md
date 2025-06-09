## Code Structure

### Main Scripts
- **train.py**: Contains the main training loop for the models.

### Source Files
Located in the `src/disentanglement/` directory, these files include:

- **/data/dataset.py**: Manages the CelebAHQ dataset.
- **models.py**: Defines all model classes.
- **losses.py**: Contains loss computation modules.
- **utils.py**: Provides utility functions.
- **metrics/running_metrics.py**: Tracks and evaluates metrics during training.
- **discriminators_training.py**: Manages the training of identity discriminators.

### Visualizations
- **thesis_visuals/**: Stores all visualizations for the research paper.

### Detailed Model Descriptions in `models.py`
- **StyleGANSynthesis**: Processes a w-vector from latent space through a synthesis network to produce an image.
- **StyleGANEncoder**: Converts a real image into the w^+ latent space.
- **NICE**: Implements a bijective transformation T.
- **DIsGAN**: A composite model integrating a fixed encoder with a trainable NICE.

## Usage
Execute `train.py` to start training the generative models or `discriminators_training.py` for training the identity discriminators.
