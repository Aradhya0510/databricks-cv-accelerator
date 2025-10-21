"""
Setup configuration for Databricks Computer Vision Accelerator
"""

from setuptools import setup, find_packages
import os

# Read version from databricks_cv_accelerator/__init__.py
version = {}
with open(os.path.join("databricks_cv_accelerator", "__init__.py")) as f:
    for line in f:
        if line.startswith("__version__"):
            exec(line, version)

# Read long description from README
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Core dependencies that are always needed
install_requires = [
    "lightning>=2.5.2",
    "optuna>=3.2.0",
    "pycocotools>=2.0.6",
    "timm>=0.9.0",
    "albumentations==2.0.8",
    "opencv-python>=4.8.0",
    "numpy==1.26.4",
]

# Development dependencies
dev_requires = [
    "pytest>=7.3.1",
    "pytest-cov>=4.1.0",
]

# Documentation dependencies
docs_requires = [
    "sphinx>=6.1.3",
    "sphinx-rtd-theme>=1.2.0",
]

# All optional dependencies combined
all_requires = dev_requires + docs_requires

setup(
    name="databricks-cv-accelerator",
    version=version.get("__version__", "0.1.0"),
    author="Aradhya Chouhan",
    author_email="aradhya.chouhan@databricks.com",
    description="An advanced, modular computer vision accelerator for Databricks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Aradhya0510/databricks-cv-architecture",
    packages=find_packages(include=["databricks_cv_accelerator", "databricks_cv_accelerator.*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=install_requires,
    extras_require={
        "dev": dev_requires,
        "docs": docs_requires,
        "all": all_requires,
    },
    include_package_data=True,
    zip_safe=False,
    keywords=[
        "computer vision",
        "deep learning",
        "pytorch",
        "lightning",
        "databricks",
        "mlflow",
        "object detection",
        "image classification",
        "semantic segmentation",
        "instance segmentation",
    ],
)

