"""
Setup configuration for HAMNet package
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# Read requirements
def read_requirements(filename):
    """Read requirements from file"""
    with open(filename, 'r') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="hamnet",
    version="1.0.0",
    author="HAMNet Development Team",
    author_email="hamnet@example.com",
    description="Hierarchical Attention-based Multimodal Network for Biological Age Prediction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/hamnet",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements("requirements.txt"),
    extras_require={
        "dev": read_requirements("requirements-dev.txt"),
        "gpu": [
            "torch>=2.0.0+cu118",
            "torchvision>=0.15.0+cu118",
            "torchaudio>=2.0.0+cu118",
        ],
        "cpu": [
            "torch>=2.0.0+cpu",
            "torchvision>=0.15.0+cpu",
            "torchaudio>=2.0.0+cpu",
        ],
        "all": read_requirements("requirements.txt") + read_requirements("requirements-dev.txt"),
    },
    entry_points={
        "console_scripts": [
            "hamnet-train=hamnet.cli.train:main",
            "hamnet-eval=hamnet.cli.evaluate:main",
            "hamnet-config=hamnet.cli.config:main",
            "hamnet-visualize=hamnet.cli.visualize:main",
        ],
    },
    include_package_data=True,
    package_data={
        "hamnet": [
            "configs/*.yaml",
            "configs/*.json",
            "examples/*.py",
            "tests/*.py",
        ],
    },
    zip_safe=False,
    project_urls={
        "Bug Reports": "https://github.com/your-org/hamnet/issues",
        "Source": "https://github.com/your-org/hamnet",
        "Documentation": "https://hamnet.readthedocs.io/",
    },
    keywords="biological-age prediction multimodal deep-learning pytorch healthcare ai",
)