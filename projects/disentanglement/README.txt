References:
1) StyleGAN3: https://github.com/NVlabs/stylegan3
2) StyleGAN3 Encoder: https://github.com/yuval-alaluf/stylegan3-editing

All pre-trained weights are located in disentanglement/pretrained_models.


Dataset:
Download the CelebAMask-HQ (CelebAHQ) dataset here: https://github.com/switchablenorms/CelebAMask-HQ

Code structure:
train.py: contains the main training loop.
ext/ contains all external models such as StyleGAN3, StyleGAN3Encoder and insightface (ArcFace recognition model)
src/ contains all source files:
	- /data/dataset.py contains the dataset cass for CelebAHQ dataset.
	- models.py contains all model classes
	- metrics/DCI.py computes the DCI metric. The code is takes from https://github.com/betterze/StyleSpace
	- loss/ contains modules for all losses in the paper 35 and for the idea with contrastive learning.

To run the code:
1) pip install -r requirements.txt
2) python train.py to start the training of the model.

The models.py script:

The class StyleGANSynthesis takes w-vector (latent space), appliys the synthesis network and outputs a generated image.
The class StyleGANEncoder takes a real image and outputs w-vector. 
The class NICE is the bi-jective transformation T from the paper 35. 
The class DIsGAN is out main class that combines the fixed encoder and trainable NICE.