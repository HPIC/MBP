import sys

from setuptools import find_packages, setup

REQUIRED_PYTHON_VERSION = (3, 10)


def check_python_version():
    if sys.version_info < REQUIRED_PYTHON_VERSION:
        print(
            "Error: This package requires Python {} or higher.".format(
                ".".join(map(str, REQUIRED_PYTHON_VERSION))
            )
        )
        sys.exit(1)


def check_pytorch_cuda():
    try:
        import torch

        if not torch.cuda.is_available():
            print("Error: PyTorch is installed but CUDA is not available.")
            sys.exit(1)
    except ImportError:
        print("Error: PyTorch is not installed. Install it manually using:")
        print("pip install torch torchvision torchaudio")
        sys.exit(1)


check_python_version()
check_pytorch_cuda()


setup(
    name="mbp-pytorch",
    version="0.2.6",
    packages=find_packages(),
    author="XinYu Piao",
    author_email="xypiao97@korea.ac.kr",
    description="Enabling large batch size training for DNN models beyond the memory limit",
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
    include_package_data=True,
)
