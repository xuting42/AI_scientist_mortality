from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="hambae",
    version="1.0.0",
    author="HAMBAE Development Team",
    author_email="hambae@example.com",
    description="Hierarchical Adaptive Multi-modal Biological Age Estimation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/hambae",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "pytest-cov>=2.12.0",
            "pytest-mock>=3.6.0",
            "black>=21.0.0",
            "flake8>=3.9.0",
            "isort>=5.9.0",
            "mypy>=0.910",
            "pre-commit>=2.15.0",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=0.5.0",
            "myst-parser>=0.15.0",
        ],
        "retinal": [
            "opencv-python>=4.5.0",
            "Pillow>=8.3.0",
            "imageio>=2.9.0",
        ],
        "genetic": [
            "pysam>=0.16.0",
            "biopython>=1.79",
        ],
        "metabolomics": [
            "matchms>=0.14.0",
            "pyteomics>=4.4.0",
        ],
        "gnn": [
            "torch-geometric>=2.0.0",
            "dgl>=0.7.0",
        ],
        "distributed": [
            "horovod>=0.24.0",
            "deepspeed>=0.5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "hambae-train=hambae.scripts.train_tier1:main",
            "hambae-evaluate=hambae.scripts.evaluate:main",
            "hambae-deploy=hambae.scripts.deploy:main",
        ],
    },
    include_package_data=True,
    package_data={
        "hambae": ["config.yaml", "*.json"],
    },
    project_urls={
        "Bug Reports": "https://github.com/your-org/hambae/issues",
        "Source": "https://github.com/your-org/hambae",
        "Documentation": "https://hambae.readthedocs.io/",
    },
)