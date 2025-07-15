from setuptools import setup, find_packages

setup(
    name="disentangled-latent-spaces",
    version="0.1.0",
    description="Masters Thesis: Disentangled Latent Space for Synthetic Data",
    author="Liam Tabibzadeh",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.1.0",
        "torchvision>=0.16.0",
        "numpy>=1.26.0",
        "opencv-python>=4.9.0",
        "pillow>=10.2.0",
        "matplotlib>=3.8.0",
        "scikit-learn>=1.4.0",
        "pandas>=2.1.0",
        "tqdm>=4.66.0",
        "pyyaml>=6.0.0",
        "hydra-core>=1.3.0",
        "wandb>=0.16.0",
        "facenet-pytorch>=2.5.0",
        "dlib>=19.24.0",
        "albumentations>=1.3.0",
        "plotly>=5.18.0",
        "seaborn>=0.13.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        "gpu": [
            "torch[cuda]>=2.1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "train-disentanglement=projects.disentanglement.train:main",
            "train-discriminators=projects.disentanglement.src.discriminators_training:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    package_data={
        "projects.disentanglement": ["*.yaml", "*.yml"],
    },
    include_package_data=True,
)