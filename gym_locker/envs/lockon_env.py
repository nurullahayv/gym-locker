"""
LockOn Environment - Egocentric Target Tracking Simulation
"""

import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
import cv2
from typing import Optional, Tuple, Dict, Any


class LockOnEnv(gym.Env):
    """
    Egocentric target tracking environment simulating a lock-on system.

    The pursuer agent tries to keep the evader (target) centered on screen
    at a specific size (representing distance) for 5 seconds to achieve lock-on.

    State Space:
        - Option A (Vector): [x_diff, y_diff, width, height] - Shape (4,)
        - Option B (Image): 84x84 grayscale image - Shape (84, 84, 1)

    Action Space:
        - Continuous: [pan_x, tilt_y] in range [-1.0, 1.0]
        - pan_x: horizontal correction vector
        - tilt_y: vertical correction vector

    Reward Function:
        - Centering error penalty: -(x_diff^2 + y_diff^2)
        - Size error penalty: -(size_error^2) * 0.1
        - Lock-on reward: +1 per step when locked
        - Big bonus: +500 for maintaining lock for 5 seconds
        - Termination penalty: -1000 if target escapes
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        render_mode: Optional[str] = None,
        state_mode: str = "vector",  # "vector" or "image"
        screen_width: int = 800,
        screen_height: int = 600,
        target_area_ratio: float = 0.30,  # Target should occupy 30% of screen
        speed_coefficient: float = 2.0,
        evader_type: str = "simple",  # "simple", "random", or "learned"
        evader_difficulty: float = 0.5,  # 0.0 (easy) to 1.0 (hard)
    ):
        super().__init__()

        # Configuration
        self.render_mode = render_mode
        self.state_mode = state_mode
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.target_area_ratio = target_area_ratio
        self.speed_coefficient = speed_coefficient
        self.evader_type = evader_type
        self.evader_difficulty = evader_difficulty

        # Constants
        self.FPS = self.metadata["render_fps"]
        self.LOCK_ON_DURATION = 5  # seconds
        self.LOCK_ON_STEPS = int(self.LOCK_ON_DURATION * self.FPS)
        self.CENTER_X = screen_width // 2
        self.CENTER_Y = screen_height // 2
        self.SCREEN_AREA = screen_width * screen_height
        self.TARGET_AREA = self.SCREEN_AREA * target_area_ratio

        # Lock-on tolerance (how close to perfect to count as "locked")
        self.LOCK_POSITION_TOLERANCE = 30  # pixels
        self.LOCK_SIZE_TOLERANCE = 0.05  # 5% area difference

        # Action space: [pan_x, tilt_y]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

        # Observation space
        if state_mode == "vector":
            # [x_diff, y_diff, width, height]
            self.observation_space = spaces.Box(
                low=np.array([-screen_width, -screen_height, 0, 0]),
                high=np.array([screen_width, screen_height, screen_width, screen_height]),
                dtype=np.float32
            )
        elif state_mode == "image":
            # 84x84 grayscale image
            self.observation_space = spaces.Box(
                low=0, high=255, shape=(84, 84, 1), dtype=np.uint8
            )
        else:
            raise ValueError(f"Invalid state_mode: {state_mode}")

        # Pygame initialization
        self.screen = None
        self.clock = None
        if render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode((screen_width, screen_height))
            pygame.display.set_caption("Gym-Locker: Lock-On Simulation")
            self.clock = pygame.time.Clock()

        # State variables
        self.target_x = 0.0
        self.target_y = 0.0
        self.target_width = 0.0
        self.target_height = 0.0
        self.target_distance = 1.0  # Normalized distance
        self.lock_on_timer = 0
        self.total_lock_time = 0.0
        self.steps = 0
        self.total_bonus_earned = 0

        # Evader state
        self.evader_momentum = np.zeros(2)

        # Font for rendering
        if render_mode == "human":
            self.font = pygame.font.Font(None, 36)
            self.small_font = pygame.font.Font(None, 24)

    def _get_evader_action(self) -> np.ndarray:
        """
        Generate evader's action based on evader_type.

        Returns:
            np.ndarray: Evader's action [escape_x, escape_y]
        """
        if self.evader_type == "simple":
            # Escape away from center
            # Current position relative to center
            dx = self.target_x
            dy = self.target_y

            # Normalize and add noise
            distance = np.sqrt(dx**2 + dy**2) + 1e-6
            escape_direction = np.array([-dx, -dy]) / distance

            # Add some random jitter
            noise = np.random.randn(2) * 0.3 * self.evader_difficulty
            action = escape_direction + noise

            # Add momentum for smoother movement
            self.evader_momentum = 0.7 * self.evader_momentum + 0.3 * action
            action = self.evader_momentum

            # Clip to valid range
            action = np.clip(action, -1.0, 1.0)
            return action.astype(np.float32)

        elif self.evader_type == "random":
            # Random walk with momentum
            random_action = np.random.randn(2) * self.evader_difficulty
            self.evader_momentum = 0.8 * self.evader_momentum + 0.2 * random_action
            action = np.clip(self.evader_momentum, -1.0, 1.0)
            return action.astype(np.float32)

        elif self.evader_type == "learned":
            # Placeholder for learned evader (MARL Phase 2)
            # For now, use simple evader
            return self._get_evader_action_simple()

        else:
            raise ValueError(f"Invalid evader_type: {self.evader_type}")

    def _update_target_position(
        self,
        pursuer_action: np.ndarray,
        evader_action: np.ndarray
    ) -> None:
        """
        Update target position and size based on net motion vector.

        Physics:
            net_vector_xy = evader_action - pursuer_action
            position += net_vector_xy * speed_coefficient
        """
        # X-Y position update
        net_vector_x = evader_action[0] - pursuer_action[0]
        net_vector_y = evader_action[1] - pursuer_action[1]

        self.target_x += net_vector_x * self.speed_coefficient
        self.target_y += net_vector_y * self.speed_coefficient

        # Evader also has a "distance" action (simplified: random walk in Z)
        # This affects target size
        evader_z = np.random.randn() * 0.1 * self.evader_difficulty
        pursuer_z = 0.0  # Pursuer doesn't control zoom in this version
        net_vector_z = evader_z - pursuer_z

        self.target_distance += net_vector_z * 0.5
        self.target_distance = np.clip(self.target_distance, 0.3, 3.0)

        # Convert distance to size (inverse relationship)
        # At distance=1.0, target should be at TARGET_AREA
        base_size = np.sqrt(self.TARGET_AREA)
        size = base_size / self.target_distance

        self.target_width = size
        self.target_height = size

    def _is_locked_on(self) -> bool:
        """
        Check if target is currently locked on.

        Returns:
            bool: True if position and size are within tolerance
        """
        position_error = np.sqrt(self.target_x**2 + self.target_y**2)

        current_area = self.target_width * self.target_height
        area_error = abs(self.TARGET_AREA - current_area) / self.TARGET_AREA

        position_locked = position_error < self.LOCK_POSITION_TOLERANCE
        size_locked = area_error < self.LOCK_SIZE_TOLERANCE

        return position_locked and size_locked

    def _is_target_escaped(self) -> bool:
        """
        Check if target has escaped (outside screen bounds).

        Returns:
            bool: True if target is outside screen
        """
        half_width = self.target_width / 2
        half_height = self.target_height / 2

        abs_x = self.CENTER_X + self.target_x
        abs_y = self.CENTER_Y + self.target_y

        escaped_x = (abs_x + half_width < 0) or (abs_x - half_width > self.screen_width)
        escaped_y = (abs_y + half_height < 0) or (abs_y - half_height > self.screen_height)

        return escaped_x or escaped_y

    def _calculate_reward(self, terminated: bool) -> float:
        """
        Calculate reward for current step.

        Returns:
            float: Reward value
        """
        reward = 0.0

        # 1. Centering error penalty
        position_error = self.target_x**2 + self.target_y**2
        reward -= position_error * 0.001  # Scale down for stability

        # 2. Size error penalty
        current_area = self.target_width * self.target_height
        area_error = (self.TARGET_AREA - current_area) ** 2
        reward -= area_error * 0.00001  # Scale down

        # 3. Lock-on reward
        if self._is_locked_on():
            reward += 1.0
            self.lock_on_timer += 1

            # 4. Big bonus for 5-second lock
            if self.lock_on_timer >= self.LOCK_ON_STEPS:
                reward += 500.0
                self.total_bonus_earned += 1
                self.lock_on_timer = 0  # Reset for next bonus
        else:
            self.lock_on_timer = 0

        # 5. Termination penalty
        if terminated:
            reward -= 1000.0

        return reward

    def _get_observation(self) -> np.ndarray:
        """
        Get current observation based on state_mode.

        Returns:
            np.ndarray: Observation
        """
        if self.state_mode == "vector":
            return np.array([
                self.target_x,
                self.target_y,
                self.target_width,
                self.target_height
            ], dtype=np.float32)

        elif self.state_mode == "image":
            # Render current frame
            frame = self._render_frame()

            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

            # Resize to 84x84
            resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)

            # Add channel dimension
            observation = np.expand_dims(resized, axis=-1)

            return observation.astype(np.uint8)

    def _render_frame(self) -> np.ndarray:
        """
        Render current state to RGB array.

        Returns:
            np.ndarray: RGB image of shape (height, width, 3)
        """
        if self.screen is None:
            # Create virtual screen for rgb_array mode
            surface = pygame.Surface((self.screen_width, self.screen_height))
        else:
            surface = self.screen

        # Black background
        surface.fill((0, 0, 0))

        # Draw center lock box (30% of screen)
        lock_box_size = int(np.sqrt(self.TARGET_AREA))
        lock_box_rect = pygame.Rect(
            self.CENTER_X - lock_box_size // 2,
            self.CENTER_Y - lock_box_size // 2,
            lock_box_size,
            lock_box_size
        )
        pygame.draw.rect(surface, (50, 50, 50), lock_box_rect, 2)

        # Draw center crosshair
        pygame.draw.line(
            surface, (100, 100, 100),
            (self.CENTER_X - 20, self.CENTER_Y),
            (self.CENTER_X + 20, self.CENTER_Y),
            1
        )
        pygame.draw.line(
            surface, (100, 100, 100),
            (self.CENTER_X, self.CENTER_Y - 20),
            (self.CENTER_X, self.CENTER_Y + 20),
            1
        )

        # Draw target (evader)
        target_abs_x = self.CENTER_X + self.target_x
        target_abs_y = self.CENTER_Y + self.target_y

        target_rect = pygame.Rect(
            int(target_abs_x - self.target_width / 2),
            int(target_abs_y - self.target_height / 2),
            int(self.target_width),
            int(self.target_height)
        )

        # Color: Red if not locked, Green if locked
        target_color = (0, 255, 0) if self._is_locked_on() else (255, 0, 0)
        pygame.draw.rect(surface, target_color, target_rect)

        # Draw lock-on indicator
        if self.render_mode == "human":
            lock_progress = self.lock_on_timer / self.LOCK_ON_STEPS
            lock_text = f"Lock: {lock_progress*100:.1f}%"
            text_surface = self.small_font.render(lock_text, True, (255, 255, 255))
            surface.blit(text_surface, (10, 10))

            # Draw total bonuses earned
            bonus_text = f"Bonuses: {self.total_bonus_earned}"
            bonus_surface = self.small_font.render(bonus_text, True, (255, 255, 0))
            surface.blit(bonus_surface, (10, 40))

            # Draw steps
            steps_text = f"Steps: {self.steps}"
            steps_surface = self.small_font.render(steps_text, True, (200, 200, 200))
            surface.blit(steps_surface, (10, 70))

        # Convert to RGB array
        frame = pygame.surfarray.array3d(surface)
        frame = np.transpose(frame, (1, 0, 2))  # pygame uses (width, height, channels)

        return frame

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to initial state.

        Returns:
            observation: Initial observation
            info: Additional information
        """
        super().reset(seed=seed)

        # Reset target to random position near center
        self.target_x = self.np_random.uniform(-50, 50)
        self.target_y = self.np_random.uniform(-50, 50)
        self.target_distance = self.np_random.uniform(0.8, 1.2)

        # Calculate initial size
        base_size = np.sqrt(self.TARGET_AREA)
        size = base_size / self.target_distance
        self.target_width = size
        self.target_height = size

        # Reset counters
        self.lock_on_timer = 0
        self.total_lock_time = 0.0
        self.steps = 0
        self.total_bonus_earned = 0
        self.evader_momentum = np.zeros(2)

        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def step(
        self,
        action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.

        Args:
            action: Pursuer's action [pan_x, tilt_y]

        Returns:
            observation: New observation
            reward: Reward for this step
            terminated: Whether episode ended due to failure
            truncated: Whether episode ended due to time limit
            info: Additional information
        """
        self.steps += 1

        # Get evader's action
        evader_action = self._get_evader_action()

        # Update target position and size
        self._update_target_position(action, evader_action)

        # Check termination conditions
        terminated = self._is_target_escaped()
        truncated = False  # Gymnasium handles max_episode_steps

        # Calculate reward
        reward = self._calculate_reward(terminated)

        # Get new observation
        observation = self._get_observation()
        info = self._get_info()

        # Render if needed
        if self.render_mode == "human":
            self._render_frame()
            pygame.display.flip()
            self.clock.tick(self.FPS)

        return observation, reward, terminated, truncated, info

    def _get_info(self) -> Dict[str, Any]:
        """
        Get additional information about current state.

        Returns:
            dict: Information dictionary
        """
        return {
            "target_x": self.target_x,
            "target_y": self.target_y,
            "target_width": self.target_width,
            "target_height": self.target_height,
            "lock_on_progress": self.lock_on_timer / self.LOCK_ON_STEPS,
            "is_locked": self._is_locked_on(),
            "total_bonuses": self.total_bonus_earned,
            "steps": self.steps
        }

    def render(self) -> Optional[np.ndarray]:
        """
        Render the environment.

        Returns:
            np.ndarray or None: RGB array if render_mode is "rgb_array"
        """
        if self.render_mode == "rgb_array":
            return self._render_frame()
        elif self.render_mode == "human":
            # Already rendered in step()
            return None

    def close(self):
        """
        Clean up resources.
        """
        if self.screen is not None:
            pygame.quit()
            self.screen = None
