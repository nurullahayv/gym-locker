"""
PID Controller for Target Tracking Baseline
"""

import numpy as np
from simple_pid import PID


class PIDController:
    """
    Dual-PID controller for X-Y target tracking.

    Uses two independent PID controllers for X and Y axes
    to maintain target at center of screen.
    """

    def __init__(
        self,
        kp: float = 0.1,
        ki: float = 0.01,
        kd: float = 0.05,
        output_limits: tuple = (-1.0, 1.0)
    ):
        """
        Initialize PID controllers for X and Y axes.

        Args:
            kp: Proportional gain
            ki: Integral gain
            kd: Derivative gain
            output_limits: Output saturation limits
        """
        self.pid_x = PID(
            Kp=kp,
            Ki=ki,
            Kd=kd,
            setpoint=0.0,  # Target: center at x=0
            output_limits=output_limits
        )

        self.pid_y = PID(
            Kp=kp,
            Ki=ki,
            Kd=kd,
            setpoint=0.0,  # Target: center at y=0
            output_limits=output_limits
        )

        # Sample time (will be set based on environment FPS)
        self.sample_time = 1.0 / 30.0  # Default 30 FPS

    def set_sample_time(self, fps: int):
        """
        Set sample time based on environment FPS.

        Args:
            fps: Frames per second
        """
        self.sample_time = 1.0 / fps
        self.pid_x.sample_time = self.sample_time
        self.pid_y.sample_time = self.sample_time

    def set_gains(self, kp: float, ki: float, kd: float):
        """
        Update PID gains.

        Args:
            kp: Proportional gain
            ki: Integral gain
            kd: Derivative gain
        """
        self.pid_x.tunings = (kp, ki, kd)
        self.pid_y.tunings = (kp, ki, kd)

    def get_action(self, observation: np.ndarray) -> np.ndarray:
        """
        Compute control action based on current observation.

        Args:
            observation: State vector [x_diff, y_diff, width, height]

        Returns:
            np.ndarray: Control action [pan_x, tilt_y]
        """
        # Extract position errors
        x_error = observation[0]
        y_error = observation[1]

        # PID computes correction to bring error to setpoint (0)
        # Since error = current - setpoint, and setpoint=0,
        # error = current position
        # PID will output correction in opposite direction
        correction_x = -self.pid_x(x_error)
        correction_y = -self.pid_y(y_error)

        action = np.array([correction_x, correction_y], dtype=np.float32)

        return action

    def reset(self):
        """
        Reset PID controllers (clear integral term).
        """
        self.pid_x.reset()
        self.pid_y.reset()


class HybridPIDRLController:
    """
    Hybrid controller combining PID baseline with RL correction.

    PID provides stable baseline tracking, RL learns fine-tuning corrections.
    """

    def __init__(
        self,
        pid_controller: PIDController,
        rl_model,
        pid_weight: float = 0.7,
        rl_weight: float = 0.3
    ):
        """
        Initialize hybrid controller.

        Args:
            pid_controller: PID controller instance
            rl_model: Trained RL model (Stable-Baselines3)
            pid_weight: Weight for PID output (0-1)
            rl_weight: Weight for RL output (0-1)
        """
        self.pid_controller = pid_controller
        self.rl_model = rl_model
        self.pid_weight = pid_weight
        self.rl_weight = rl_weight

        # Normalize weights
        total_weight = pid_weight + rl_weight
        self.pid_weight /= total_weight
        self.rl_weight /= total_weight

    def get_action(
        self,
        observation: np.ndarray,
        deterministic: bool = True
    ) -> np.ndarray:
        """
        Compute hybrid action.

        Args:
            observation: State vector
            deterministic: Use deterministic RL policy

        Returns:
            np.ndarray: Combined action
        """
        # Get PID action
        pid_action = self.pid_controller.get_action(observation)

        # Get RL action
        rl_action, _ = self.rl_model.predict(observation, deterministic=deterministic)

        # Weighted combination
        action = self.pid_weight * pid_action + self.rl_weight * rl_action

        # Clip to valid range
        action = np.clip(action, -1.0, 1.0)

        return action.astype(np.float32)

    def reset(self):
        """
        Reset both controllers.
        """
        self.pid_controller.reset()

    def set_weights(self, pid_weight: float, rl_weight: float):
        """
        Update blending weights.

        Args:
            pid_weight: Weight for PID output
            rl_weight: Weight for RL output
        """
        total_weight = pid_weight + rl_weight
        self.pid_weight = pid_weight / total_weight
        self.rl_weight = rl_weight / total_weight


def tune_pid_gains(
    env,
    kp_range: tuple = (0.05, 0.3),
    ki_range: tuple = (0.0, 0.05),
    kd_range: tuple = (0.0, 0.1),
    n_trials: int = 20,
    n_episodes: int = 10
) -> dict:
    """
    Simple grid search for PID gain tuning.

    Args:
        env: Gymnasium environment
        kp_range: Range for proportional gain (min, max)
        ki_range: Range for integral gain (min, max)
        kd_range: Range for derivative gain (min, max)
        n_trials: Number of random trials
        n_episodes: Episodes per trial

    Returns:
        dict: Best gains and score
    """
    best_gains = None
    best_score = float('-inf')

    print(f"Tuning PID gains with {n_trials} trials, {n_episodes} episodes each...")

    for trial in range(n_trials):
        # Random sample
        kp = np.random.uniform(*kp_range)
        ki = np.random.uniform(*ki_range)
        kd = np.random.uniform(*kd_range)

        # Create controller
        controller = PIDController(kp=kp, ki=ki, kd=kd)
        controller.set_sample_time(env.metadata["render_fps"])

        # Evaluate
        total_reward = 0.0
        for _ in range(n_episodes):
            obs, _ = env.reset()
            controller.reset()
            done = False

            while not done:
                action = controller.get_action(obs)
                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                done = terminated or truncated

        avg_reward = total_reward / n_episodes

        if avg_reward > best_score:
            best_score = avg_reward
            best_gains = {"kp": kp, "ki": ki, "kd": kd}

        print(f"Trial {trial+1}/{n_trials}: Kp={kp:.4f}, Ki={ki:.4f}, Kd={kd:.4f}, Score={avg_reward:.2f}")

    print(f"\nBest gains: Kp={best_gains['kp']:.4f}, Ki={best_gains['ki']:.4f}, Kd={best_gains['kd']:.4f}")
    print(f"Best score: {best_score:.2f}")

    return best_gains
