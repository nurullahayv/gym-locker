"""
Evader Agent Implementations
"""

import numpy as np
from typing import Optional


class SimpleEvader:
    """
    Simple rule-based evader that tries to escape from center.

    This evader moves away from the center with some noise and momentum
    to create a realistic escape pattern.
    """

    def __init__(self, difficulty: float = 0.5, momentum: float = 0.7):
        """
        Args:
            difficulty: Controls noise level (0.0 = easy, 1.0 = hard)
            momentum: Momentum coefficient for smooth movement (0.0-1.0)
        """
        self.difficulty = difficulty
        self.momentum_coef = momentum
        self.momentum = np.zeros(2)

    def get_action(self, observation: np.ndarray) -> np.ndarray:
        """
        Generate escape action based on target position.

        Args:
            observation: State vector [x_diff, y_diff, width, height]

        Returns:
            np.ndarray: Evader action [escape_x, escape_y]
        """
        # Extract position from observation
        x_diff = observation[0]
        y_diff = observation[1]

        # Escape direction: away from center (0, 0)
        distance = np.sqrt(x_diff**2 + y_diff**2) + 1e-6
        escape_direction = np.array([-x_diff, -y_diff]) / distance

        # Add random noise based on difficulty
        noise = np.random.randn(2) * 0.3 * self.difficulty
        action = escape_direction + noise

        # Apply momentum for smoother movement
        self.momentum = self.momentum_coef * self.momentum + (1 - self.momentum_coef) * action

        # Clip to valid action range
        action = np.clip(self.momentum, -1.0, 1.0)

        return action.astype(np.float32)

    def reset(self):
        """
        Reset evader state for new episode.
        """
        self.momentum = np.zeros(2)


class RandomEvader:
    """
    Random walk evader with momentum.

    This evader performs a random walk, making it unpredictable
    but not strategically escaping.
    """

    def __init__(self, difficulty: float = 0.5, momentum: float = 0.8):
        """
        Args:
            difficulty: Controls movement magnitude
            momentum: Momentum coefficient for smooth movement
        """
        self.difficulty = difficulty
        self.momentum_coef = momentum
        self.momentum = np.zeros(2)

    def get_action(self, observation: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Generate random action.

        Args:
            observation: Not used, kept for API consistency

        Returns:
            np.ndarray: Random action [x, y]
        """
        # Random action
        random_action = np.random.randn(2) * self.difficulty

        # Apply momentum
        self.momentum = self.momentum_coef * self.momentum + (1 - self.momentum_coef) * random_action

        # Clip to valid range
        action = np.clip(self.momentum, -1.0, 1.0)

        return action.astype(np.float32)

    def reset(self):
        """
        Reset evader state.
        """
        self.momentum = np.zeros(2)


class LearnedEvader:
    """
    Placeholder for learned evader (MARL Phase 2).

    This will be a neural network-based evader trained with RL.
    """

    def __init__(self, model_path: Optional[str] = None):
        """
        Args:
            model_path: Path to trained model (for future use)
        """
        self.model_path = model_path
        self.model = None

        # For now, use simple evader as fallback
        self.fallback = SimpleEvader(difficulty=0.8)

    def get_action(self, observation: np.ndarray) -> np.ndarray:
        """
        Generate action using learned policy.

        Args:
            observation: State vector

        Returns:
            np.ndarray: Evader action
        """
        if self.model is None:
            # Use fallback evader
            return self.fallback.get_action(observation)

        # TODO: Implement learned evader
        # action, _ = self.model.predict(observation, deterministic=True)
        # return action

        raise NotImplementedError("Learned evader not yet implemented (Phase 2)")

    def reset(self):
        """
        Reset evader state.
        """
        if self.fallback is not None:
            self.fallback.reset()
