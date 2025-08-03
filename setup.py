from setuptools import setup, find_packages

setup(
    name="audio_recognition",
    version="0.0.1",
    author="Jiheng Li",
    author_email="jiheng.li.1@vanderbilt.edu",
    description="Deep Learning Algorithm for Music Recognition",
    packages=find_packages(exclude=["data*"]),
    include_package_data=True,
    install_requires=[
        "torch",
        "torchvision",
        "torchaudio",
        "transformers",
        "faiss-cpu",
        "librosa",
        "pydub",
    ],
    python_requires=">=3.8",
)
