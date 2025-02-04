import sys

from setuptools import find_packages, setup

try:
    import torch

    if not torch.cuda.is_available():
        print("Error: PyTorch is installed but CUDA is not available.")
        sys.exit(1)
except ImportError:
    print("Error: PyTorch is not installed. Install it manually using:")
    print("pip install torch torchvision torchaudio")
    sys.exit(1)

setup(
    name="mbp-pytorch",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "torchvision",
        "torchaudio",
    ],
    author="XinYu Piao",
    author_email="xypiao97@korea.ac.kr",
    description="A deep learning package that requires GPU-enabled PyTorch.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/HPIC/MBP.git",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Operating System :: POSIX :: Linux",
        "Operating System :: Microsoft :: Windows",
    ],
    python_requires=">=3.10",
    include_package_data=True,  # LICENSE 포함 가능
)
