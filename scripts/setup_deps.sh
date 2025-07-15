#!/bin/bash

# Setup script for external dependencies
# This script downloads and sets up all required external repositories

set -e  # Exit on any error

echo "Setting up external dependencies for Disentangled Latent Spaces project..."

# Create ext directory if it doesn't exist
mkdir -p ext
cd ext

# Clone StyleGAN3
if [ ! -d "stylegan3" ]; then
    echo "Cloning StyleGAN3..."
    git clone https://github.com/NVlabs/stylegan3.git
    cd stylegan3
    # Install StyleGAN3 dependencies
    echo "Installing StyleGAN3 dependencies..."
    pip install -r requirements.txt
    cd ..
else
    echo "StyleGAN3 already exists, skipping..."
fi

# Clone StyleGAN3 Editing
if [ ! -d "stylegan3_editing" ]; then
    echo "Cloning StyleGAN3 Editing..."
    git clone https://github.com/yuval-alaluf/stylegan3-editing.git stylegan3_editing
    cd stylegan3_editing
    # Install editing dependencies
    echo "Installing StyleGAN3 editing dependencies..."
    pip install -r requirements.txt
    cd ..
else
    echo "StyleGAN3 editing already exists, skipping..."
fi

# Clone InsightFace (ArcFace)
if [ ! -d "insightface" ]; then
    echo "Cloning InsightFace..."
    git clone https://github.com/deepinsight/insightface.git
    cd insightface
    # Install insightface dependencies
    echo "Installing InsightFace dependencies..."
    pip install -r requirements.txt
    cd ..
else
    echo "InsightFace already exists, skipping..."
fi

cd ..

# Create necessary directories
echo "Creating necessary directories..."
mkdir -p datasets/celebahq
mkdir -p projects/disentanglement/pretrained_models

echo "External dependencies setup complete!"
echo ""
echo "Next steps:"
echo "1. Download CelebAHQ dataset to datasets/celebahq/"
echo "2. Download pre-trained models to projects/disentanglement/pretrained_models/"
echo "3. Run: pip install -e . to install the package in development mode"
echo ""
echo "For CelebAHQ dataset, visit: https://github.com/switchablenorms/CelebAMask-HQ"