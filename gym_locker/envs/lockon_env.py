"""
LockOn Environment - Egocentric Target Tracking Simulation
"""

import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
import cv2
from typing import Optional, Tuple, Dict, Any
from collections import deque


class LockOnEnv(gym.Env):
    """
    Egocentric target tracking environment simulating a lock-on system.

    The pursuer agent must track a fast, agile evader and maintain lock-on
    for 5 consecutive seconds to succeed.

    Lock-on Conditions:
        1. Target must be inside the lock box (center square, 30% of screen)
        2. Target must fill at least 5% of the lock box area
        Both conditions must be maintained for 5 seconds for successful lock-on.

    State Space:
        - Option A (Vector): [x_diff, y_diff, width, height] - Shape (4,)
        - Option B (Image): 84x84 grayscale image - Shape (84, 84, 1)

    Action Space:
        - Continuous: [pan_x, tilt_y] in range [-1.0, 1.0]
        - pan_x: horizontal correction vector
        - tilt_y: vertical correction vector

    Reward Function:
        - Centering error penalty: -(x_diff^2 + y_diff^2) * 0.001
        - Size error penalty: -(size_error^2) * 0.00001
        - Lock-on reward: +1 per step when locked
        - Big bonus: +500 for maintaining lock for 5 seconds
        - Termination penalty: -1000 if target escapes

    Evader Dynamics:
        - Evader is faster than pursuer (1.5x speed multiplier by default)
        - Makes tracking challenging and requires skilled control
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        render_mode: Optional[str] = None,
        state_mode: str = "vector",  # "vector" or "image"
        screen_width: int = 800,
        screen_height: int = 600,
        target_area_ratio: float = 0.30,  # Lock box size (30% of screen)
        speed_coefficient: float = 4.0,  # Higher speed for more dynamic tracking
        evader_speed_multiplier: float = 1.5,  # Evader is faster than pursuer
        evader_type: str = "simple",  # "simple", "random", or "learned"
        evader_difficulty: float = 0.5,  # 0.0 (easy) to 1.0 (hard)
        observation_delay: int = 30,  # Observation delay in frames (1 sec at 30 FPS)
    ):
        super().__init__()

        # Configuration
        self.render_mode = render_mode
        self.state_mode = state_mode
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.target_area_ratio = target_area_ratio
        self.speed_coefficient = speed_coefficient
        self.evader_speed_multiplier = evader_speed_multiplier
        self.evader_type = evader_type
        self.evader_difficulty = evader_difficulty
        self.observation_delay = observation_delay

        # Constants
        self.FPS = self.metadata["render_fps"]
        self.LOCK_ON_DURATION = 5  # seconds
        self.LOCK_ON_STEPS = int(self.LOCK_ON_DURATION * self.FPS)
        self.CENTER_X = screen_width // 2
        self.CENTER_Y = screen_height // 2
        self.SCREEN_AREA = screen_width * screen_height

        # Lock box area (30% of screen)
        self.LOCK_BOX_AREA = self.SCREEN_AREA * target_area_ratio
        self.LOCK_BOX_SIZE = int(np.sqrt(self.LOCK_BOX_AREA))

        # Lock-on conditions:
        # 1. Target must be inside the lock box
        # 2. Target must fill at least 5% of the lock box area
        self.TARGET_FILL_RATIO = 0.05  # Target should fill at least 5% of lock box
        self.TARGET_FILL_TOLERANCE = 1.0  # Accept any size >= 5%

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

        # Action visualization (store last actions for rendering)
        self.last_pursuer_action = np.zeros(2)
        self.last_evader_action = np.zeros(2)

        # Evader state
        self.evader_momentum = np.zeros(2)

        # Observation delay buffer (makes tracking harder)
        # Agent sees observations from 'observation_delay' frames ago
        self.observation_buffer = deque(maxlen=observation_delay + 1)

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
            # Aggressive escape strategy
            # Current position relative to center
            dx = self.target_x
            dy = self.target_y

            # Strategy depends on whether we're in lock box or not
            distance_from_center = np.sqrt(dx**2 + dy**2)
            lock_box_radius = self.LOCK_BOX_SIZE / 2

            if distance_from_center < lock_box_radius * 0.7:
                # Inside or near lock box: ESCAPE aggressively
                escape_direction = np.array([-dx, -dy]) / (distance_from_center + 1e-6)

                # Add strong random jitter for unpredictability
                noise = np.random.randn(2) * 0.6 * self.evader_difficulty
                action = escape_direction * 1.2 + noise  # Amplified escape
            else:
                # Far from lock box: Random evasive maneuvers
                noise = np.random.randn(2) * 0.8 * self.evader_difficulty
                action = self.evader_momentum * 0.5 + noise

            # Less momentum for more agility
            self.evader_momentum = 0.5 * self.evader_momentum + 0.5 * action
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
            net_vector_xy = (evader_action * evader_speed_multiplier) - pursuer_action
            position += net_vector_xy * speed_coefficient

        Evader is more agile/fast than pursuer, making tracking challenging.
        """
        # X-Y position update
        # Evader is faster/more agile (multiplied by evader_speed_multiplier)
        evader_x = evader_action[0] * self.evader_speed_multiplier
        evader_y = evader_action[1] * self.evader_speed_multiplier

        net_vector_x = evader_x - pursuer_action[0]
        net_vector_y = evader_y - pursuer_action[1]

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
        # At distance=1.0, target should fill 30% of lock box
        desired_target_area = self.LOCK_BOX_AREA * self.TARGET_FILL_RATIO
        base_size = np.sqrt(desired_target_area)
        size = base_size / self.target_distance

        self.target_width = size
        self.target_height = size

    def _is_locked_on(self) -> bool:
        """
        Check if target is currently locked on.

        Lock-on conditions:
        1. Target must be inside the lock box (centered square)
        2. Target must fill at least 5% of the lock box area

        Returns:
            bool: True if both conditions are met
        """
        # Calculate target's absolute position and bounds
        target_abs_x = self.CENTER_X + self.target_x
        target_abs_y = self.CENTER_Y + self.target_y

        target_left = target_abs_x - self.target_width / 2
        target_right = target_abs_x + self.target_width / 2
        target_top = target_abs_y - self.target_height / 2
        target_bottom = target_abs_y + self.target_height / 2

        # Calculate lock box bounds (centered square)
        lock_box_left = self.CENTER_X - self.LOCK_BOX_SIZE / 2
        lock_box_right = self.CENTER_X + self.LOCK_BOX_SIZE / 2
        lock_box_top = self.CENTER_Y - self.LOCK_BOX_SIZE / 2
        lock_box_bottom = self.CENTER_Y + self.LOCK_BOX_SIZE / 2

        # Condition 1: Target must be inside the lock box (at least partially)
        inside_box = (
            target_right > lock_box_left and
            target_left < lock_box_right and
            target_bottom > lock_box_top and
            target_top < lock_box_bottom
        )

        # Condition 2: Target must fill at least 5% of lock box area
        target_area = self.target_width * self.target_height
        fill_ratio = target_area / self.LOCK_BOX_AREA

        # Check if fill ratio is at least 5%
        fill_ok = fill_ratio >= self.TARGET_FILL_RATIO

        # Both conditions must be true for successful lock-on
        return inside_box and fill_ok

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

        Reward components:
        1. Penalty for being off-center
        2. Penalty for wrong size (should fill 30% of lock box)
        3. Reward for successful lock-on (+1 per step)
        4. Big bonus for maintaining lock for 5 seconds (+500)
        5. Penalty for escape (-1000)

        Returns:
            float: Reward value
        """
        reward = 0.0

        # 1. Centering error penalty
        position_error = self.target_x**2 + self.target_y**2
        reward -= position_error * 0.001  # Scale down for stability

        # 2. Size error penalty
        # Target should fill 30% of lock box
        current_area = self.target_width * self.target_height
        desired_area = self.LOCK_BOX_AREA * self.TARGET_FILL_RATIO
        area_error = (desired_area - current_area) ** 2
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
        lock_box_rect = pygame.Rect(
            self.CENTER_X - self.LOCK_BOX_SIZE // 2,
            self.CENTER_Y - self.LOCK_BOX_SIZE // 2,
            self.LOCK_BOX_SIZE,
            self.LOCK_BOX_SIZE
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

        # Draw action vectors (arrows)
        target_abs_x = self.CENTER_X + self.target_x
        target_abs_y = self.CENTER_Y + self.target_y

        # Helper function to draw arrow
        def draw_arrow(surf, color, start, end, width=2):
            """Draw an arrow from start to end"""
            # Draw line
            pygame.draw.line(surf, color, start, end, width)

            # Calculate arrow head
            dx = end[0] - start[0]
            dy = end[1] - start[1]
            length = np.sqrt(dx*dx + dy*dy)

            if length > 5:  # Only draw arrowhead if line is long enough
                # Normalize
                dx /= length
                dy /= length

                # Arrow head size
                head_length = min(15, length * 0.3)
                head_width = head_length * 0.6

                # Perpendicular vector
                perp_x = -dy
                perp_y = dx

                # Arrow head points
                point1 = (
                    int(end[0] - head_length * dx + head_width * perp_x),
                    int(end[1] - head_length * dy + head_width * perp_y)
                )
                point2 = (
                    int(end[0] - head_length * dx - head_width * perp_x),
                    int(end[1] - head_length * dy - head_width * perp_y)
                )

                pygame.draw.polygon(surf, color, [end, point1, point2])

        # Vector scale for visualization (larger for visibility)
        vector_scale = 50

        # Evader action vector (from target, RED)
        if np.linalg.norm(self.last_evader_action) > 0.01:
            evader_start = (int(target_abs_x), int(target_abs_y))
            evader_end = (
                int(target_abs_x + self.last_evader_action[0] * vector_scale),
                int(target_abs_y + self.last_evader_action[1] * vector_scale)
            )
            draw_arrow(surface, (255, 100, 100), evader_start, evader_end, 3)

        # Pursuer action vector (from center, BLUE)
        if np.linalg.norm(self.last_pursuer_action) > 0.01:
            pursuer_start = (self.CENTER_X, self.CENTER_Y)
            pursuer_end = (
                int(self.CENTER_X + self.last_pursuer_action[0] * vector_scale),
                int(self.CENTER_Y + self.last_pursuer_action[1] * vector_scale)
            )
            draw_arrow(surface, (100, 100, 255), pursuer_start, pursuer_end, 3)

        # Draw warning zone if target is close to screen edges
        warning_margin = 100  # pixels from edge
        target_center_x = target_abs_x
        target_center_y = target_abs_y

        near_left = target_center_x < warning_margin
        near_right = target_center_x > self.screen_width - warning_margin
        near_top = target_center_y < warning_margin
        near_bottom = target_center_y > self.screen_height - warning_margin

        in_warning_zone = near_left or near_right or near_top or near_bottom

        if in_warning_zone:
            # Draw red warning border
            pygame.draw.rect(surface, (255, 0, 0),
                           (0, 0, self.screen_width, self.screen_height), 5)

            # Draw warning text
            if self.render_mode == "human":
                warning_text = "WARNING: TARGET ESCAPING!"
                warning_surface = self.font.render(warning_text, True, (255, 0, 0))
                text_rect = warning_surface.get_rect(center=(self.screen_width // 2, 30))
                surface.blit(warning_surface, text_rect)

        # Draw target (evader)
        target_rect = pygame.Rect(
            int(target_abs_x - self.target_width / 2),
            int(target_abs_y - self.target_height / 2),
            int(self.target_width),
            int(self.target_height)
        )

        # Color: Green if locked, Yellow if in warning zone, Red otherwise
        if self._is_locked_on():
            target_color = (0, 255, 0)  # Green - locked
        elif in_warning_zone:
            target_color = (255, 255, 0)  # Yellow - warning
        else:
            target_color = (255, 0, 0)  # Red - normal

        pygame.draw.rect(surface, target_color, target_rect)

        # Draw lock-on indicator
        if self.render_mode == "human":
            y_offset = 10

            # Lock progress
            lock_progress = self.lock_on_timer / self.LOCK_ON_STEPS
            lock_text = f"Lock: {lock_progress*100:.1f}%"
            text_surface = self.small_font.render(lock_text, True, (255, 255, 255))
            surface.blit(text_surface, (10, y_offset))
            y_offset += 30

            # Total bonuses
            bonus_text = f"Bonuses: {self.total_bonus_earned}"
            bonus_surface = self.small_font.render(bonus_text, True, (255, 255, 0))
            surface.blit(bonus_surface, (10, y_offset))
            y_offset += 30

            # Steps
            steps_text = f"Steps: {self.steps}"
            steps_surface = self.small_font.render(steps_text, True, (200, 200, 200))
            surface.blit(steps_surface, (10, y_offset))
            y_offset += 30

            # Vector magnitudes
            pursuer_mag = np.linalg.norm(self.last_pursuer_action)
            evader_mag = np.linalg.norm(self.last_evader_action)

            pursuer_text = f"Pursuer: {pursuer_mag:.2f}"
            pursuer_surface = self.small_font.render(pursuer_text, True, (100, 100, 255))
            surface.blit(pursuer_surface, (10, y_offset))
            y_offset += 25

            evader_text = f"Evader: {evader_mag:.2f}"
            evader_surface = self.small_font.render(evader_text, True, (255, 100, 100))
            surface.blit(evader_surface, (10, y_offset))
            y_offset += 25

            # Distance from center
            distance = np.sqrt(self.target_x**2 + self.target_y**2)
            distance_text = f"Distance: {distance:.0f}px"
            distance_surface = self.small_font.render(distance_text, True, (200, 200, 200))
            surface.blit(distance_surface, (10, y_offset))
            y_offset += 25

            # Observation delay warning
            delay_sec = self.observation_delay / self.FPS
            delay_text = f"Delay: {delay_sec:.1f}s"
            delay_surface = self.small_font.render(delay_text, True, (255, 165, 0))  # Orange
            surface.blit(delay_surface, (10, y_offset))

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

        # Reset target to random position (wider area for more movement)
        # Can start anywhere in the lock box
        max_offset = self.LOCK_BOX_SIZE // 3
        self.target_x = self.np_random.uniform(-max_offset, max_offset)
        self.target_y = self.np_random.uniform(-max_offset, max_offset)
        self.target_distance = self.np_random.uniform(0.5, 1.5)

        # Calculate initial size
        # Target should fill 30% of lock box at distance=1.0
        desired_target_area = self.LOCK_BOX_AREA * self.TARGET_FILL_RATIO
        base_size = np.sqrt(desired_target_area)
        size = base_size / self.target_distance
        self.target_width = size
        self.target_height = size

        # Reset counters
        self.lock_on_timer = 0
        self.total_lock_time = 0.0
        self.steps = 0
        self.total_bonus_earned = 0
        self.evader_momentum = np.zeros(2)

        # Reset action visualization
        self.last_pursuer_action = np.zeros(2)
        self.last_evader_action = np.zeros(2)

        # Reset observation buffer
        self.observation_buffer.clear()

        observation = self._get_observation()

        # Fill buffer with initial observation
        for _ in range(self.observation_delay + 1):
            self.observation_buffer.append(observation.copy())

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

        # Store actions for visualization
        self.last_pursuer_action = action.copy()
        self.last_evader_action = evader_action.copy()

        # Update target position and size
        self._update_target_position(action, evader_action)

        # Check termination conditions
        terminated = self._is_target_escaped()
        truncated = False  # Gymnasium handles max_episode_steps

        # Calculate reward
        reward = self._calculate_reward(terminated)

        # Get current (true) observation
        current_observation = self._get_observation()

        # Add to observation buffer
        self.observation_buffer.append(current_observation.copy())

        # Return delayed observation to agent (makes tracking harder)
        if len(self.observation_buffer) < self.observation_delay + 1:
            # During warm-up, return current observation
            delayed_observation = current_observation
        else:
            # Return observation from 'observation_delay' frames ago
            delayed_observation = self.observation_buffer[0]

        info = self._get_info()

        # Render if needed
        if self.render_mode == "human":
            self._render_frame()
            pygame.display.flip()
            self.clock.tick(self.FPS)

        return delayed_observation, reward, terminated, truncated, info

    def _get_info(self) -> Dict[str, Any]:
        """
        Get additional information about current state.

        Returns:
            dict: Information dictionary
        """
        # Check if in warning zone
        target_abs_x = self.CENTER_X + self.target_x
        target_abs_y = self.CENTER_Y + self.target_y
        warning_margin = 100

        in_warning_zone = (
            target_abs_x < warning_margin or
            target_abs_x > self.screen_width - warning_margin or
            target_abs_y < warning_margin or
            target_abs_y > self.screen_height - warning_margin
        )

        return {
            "target_x": self.target_x,
            "target_y": self.target_y,
            "target_width": self.target_width,
            "target_height": self.target_height,
            "lock_on_progress": self.lock_on_timer / self.LOCK_ON_STEPS,
            "is_locked": self._is_locked_on(),
            "total_bonuses": self.total_bonus_earned,
            "steps": self.steps,
            "in_warning_zone": in_warning_zone,
            "pursuer_action_magnitude": float(np.linalg.norm(self.last_pursuer_action)),
            "evader_action_magnitude": float(np.linalg.norm(self.last_evader_action)),
            "distance_from_center": float(np.sqrt(self.target_x**2 + self.target_y**2)),
            "observation_delay_frames": self.observation_delay,
            "observation_delay_seconds": self.observation_delay / self.FPS
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
