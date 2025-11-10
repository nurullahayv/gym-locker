"""
Setup script for Gym-Locker package
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="gym-locker",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Egocentric Target Tracking Environment for Reinforcement Learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/YOUR_USERNAME/gym-locker",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "gymnasium>=0.29.0",
        "pygame>=2.5.0",
        "numpy>=1.24.0",
        "opencv-python>=4.8.0",
        "torch>=2.0.0",
        "stable-baselines3>=2.0.0",
        "sb3-contrib>=2.0.0",
        "simple-pid>=1.0.1",
        "matplotlib>=3.7.0",
        "tensorboard>=2.13.0",
        "tqdm>=4.65.0",
        "pillow>=10.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        "marl": [
            "pettingzoo>=1.24.0",
            "ray[rllib]>=2.6.0",
        ],
    },
)
