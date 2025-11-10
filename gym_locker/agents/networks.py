"""
Custom PyTorch Network Architectures for RL Agents
"""

import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym


class SimpleMLP(BaseFeaturesExtractor):
    """
    Simple Multi-Layer Perceptron for vector state space.

    Architecture:
        Input (4) -> Hidden (64) -> Hidden (64) -> Output (features_dim)
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 64):
        """
        Args:
            observation_space: Observation space
            features_dim: Number of features to extract
        """
        super().__init__(observation_space, features_dim)

        input_dim = observation_space.shape[0]

        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.net(observations)


class DeepMLP(BaseFeaturesExtractor):
    """
    Deeper MLP for more complex learning.

    Architecture:
        Input (4) -> Hidden (128) -> Hidden (128) -> Hidden (64) -> Output (features_dim)
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 64):
        super().__init__(observation_space, features_dim)

        input_dim = observation_space.shape[0]

        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.net(observations)


class SimpleCNN(BaseFeaturesExtractor):
    """
    Convolutional Neural Network for image state space (84x84x1).

    Architecture:
        Conv(32, 8x8, stride=4) -> ReLU ->
        Conv(64, 4x4, stride=2) -> ReLU ->
        Conv(64, 3x3, stride=1) -> ReLU ->
        Flatten -> Linear(512) -> ReLU
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512):
        """
        Args:
            observation_space: Image observation space (H, W, C)
            features_dim: Number of features to extract
        """
        super().__init__(observation_space, features_dim)

        # Assume observation_space.shape = (84, 84, 1) or (84, 84, 3)
        n_input_channels = observation_space.shape[2]

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            sample_input = torch.zeros(1, n_input_channels, 84, 84)
            n_flatten = self.cnn(sample_input).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Convert from (B, H, W, C) to (B, C, H, W)
        observations = observations.permute(0, 3, 1, 2).float() / 255.0
        return self.linear(self.cnn(observations))


class NatureCNN(BaseFeaturesExtractor):
    """
    Nature CNN architecture from DQN paper (Mnih et al., 2015).

    This is a deeper variant suitable for more complex visual tasks.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512):
        super().__init__(observation_space, features_dim)

        n_input_channels = observation_space.shape[2]

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape
        with torch.no_grad():
            sample_input = torch.zeros(1, n_input_channels, 84, 84)
            n_flatten = self.cnn(sample_input).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        observations = observations.permute(0, 3, 1, 2).float() / 255.0
        return self.linear(self.cnn(observations))


def get_feature_extractor(state_mode: str, architecture: str = "simple"):
    """
    Get appropriate feature extractor class based on state mode.

    Args:
        state_mode: "vector" or "image"
        architecture: "simple" or "deep"

    Returns:
        Feature extractor class
    """
    if state_mode == "vector":
        if architecture == "simple":
            return SimpleMLP
        elif architecture == "deep":
            return DeepMLP
        else:
            raise ValueError(f"Unknown architecture: {architecture}")

    elif state_mode == "image":
        if architecture == "simple":
            return SimpleCNN
        elif architecture == "deep" or architecture == "nature":
            return NatureCNN
        else:
            raise ValueError(f"Unknown architecture: {architecture}")

    else:
        raise ValueError(f"Unknown state_mode: {state_mode}")
