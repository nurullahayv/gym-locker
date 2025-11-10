"""
PID Controller for Target Tracking Baseline with Acceleration Control
"""

import numpy as np
from simple_pid import PID


class PIDController:
    """
    Triple-PID controller for 3D acceleration control (X, Y, Z axes).

    Now controls 3D acceleration instead of direct position:
    - X/Y: Position tracking with velocity feedback
    - Z: Size control (distance management via forward/backward acceleration)

    Observation: [target_x, target_y, target_w, target_vx, target_vy, pursuer_vx, pursuer_vy, pursuer_vz]
    Action: [acc_x, acc_y, acc_z]
    """

    def __init__(
        self,
        kp_xy: float = 0.05,      # Position gain (XY)
        kd_xy: float = 0.08,      # Velocity gain (XY) - acts as derivative
        kp_z: float = 0.3,        # Size gain (Z)
        ki_z: float = 0.01,       # Integral for size control
        kd_z: float = 0.1,        # Size change rate
        output_limits: tuple = (-1.0, 1.0),
        desired_size: float = 50.0  # Desired target size (pixels)
    ):
        """
        Initialize PID controllers for 3D acceleration control.

        Args:
            kp_xy: Proportional gain for XY position
            kd_xy: Derivative gain for XY (velocity feedback)
            kp_z: Proportional gain for Z (size)
            ki_z: Integral gain for Z
            kd_z: Derivative gain for Z
            output_limits: Output saturation limits [-1, 1]
            desired_size: Target size in pixels
        """
        # XY position PIDs (no integral to avoid overshoot)
        self.pid_x = PID(
            Kp=kp_xy,
            Ki=0.0,  # No integral for position (causes overshoot)
            Kd=0.0,  # Derivative handled separately via velocity
            setpoint=0.0,
            output_limits=output_limits
        )

        self.pid_y = PID(
            Kp=kp_xy,
            Ki=0.0,
            Kd=0.0,
            setpoint=0.0,
            output_limits=output_limits
        )

        # Z axis (size control) - full PID
        self.pid_z = PID(
            Kp=kp_z,
            Ki=ki_z,
            Kd=kd_z,
            setpoint=desired_size,
            output_limits=output_limits
        )

        # Gains for velocity feedback (manual implementation)
        self.kd_xy = kd_xy

        # Sample time
        self.sample_time = 1.0 / 30.0

        # Desired size
        self.desired_size = desired_size

    def set_sample_time(self, fps: int):
        """Set sample time based on environment FPS."""
        self.sample_time = 1.0 / fps
        self.pid_x.sample_time = self.sample_time
        self.pid_y.sample_time = self.sample_time
        self.pid_z.sample_time = self.sample_time

    def set_gains(self, kp_xy: float = None, kd_xy: float = None,
                  kp_z: float = None, ki_z: float = None, kd_z: float = None):
        """
        Update PID gains.

        Args:
            kp_xy: XY position gain
            kd_xy: XY velocity gain
            kp_z: Z position gain
            ki_z: Z integral gain
            kd_z: Z derivative gain
        """
        if kp_xy is not None:
            self.pid_x.Kp = kp_xy
            self.pid_y.Kp = kp_xy
        if kd_xy is not None:
            self.kd_xy = kd_xy
        if kp_z is not None:
            self.pid_z.Kp = kp_z
        if ki_z is not None:
            self.pid_z.Ki = ki_z
        if kd_z is not None:
            self.pid_z.Kd = kd_z

    def get_action(self, observation: np.ndarray) -> np.ndarray:
        """
        Compute 3D acceleration control action.

        Args:
            observation: [target_x, target_y, target_w, target_vx, target_vy,
                         pursuer_vx, pursuer_vy, pursuer_vz]

        Returns:
            np.ndarray: Control action [acc_x, acc_y, acc_z]
        """
        # Extract state
        target_x = observation[0]
        target_y = observation[1]
        target_w = observation[2]
        target_vx = observation[3]
        target_vy = observation[4]
        # pursuer velocities available at [5, 6, 7] but not used directly

        # ===== XY Acceleration Control =====
        # Position error → desired acceleration
        acc_x_position = -self.pid_x(target_x)
        acc_y_position = -self.pid_y(target_y)

        # Velocity feedback (damping) → reduce acceleration if already moving toward target
        # If target is moving away (positive velocity when target is positive), add damping
        acc_x_velocity = -self.kd_xy * target_vx
        acc_y_velocity = -self.kd_xy * target_vy

        # Combined XY acceleration
        acc_x = acc_x_position + acc_x_velocity
        acc_y = acc_y_position + acc_y_velocity

        # ===== Z Acceleration Control (Size) =====
        # Positive acc_z = forward acceleration = approaching = target gets bigger
        # We want target_w ≈ desired_size
        # If target too small → accelerate forward (positive acc_z)
        # If target too big → accelerate backward (negative acc_z)
        size_error = target_w - self.desired_size
        acc_z = -self.pid_z(target_w)  # PID will output correction

        # Combine action
        action = np.array([acc_x, acc_y, acc_z], dtype=np.float32)

        # Clip to valid range
        action = np.clip(action, -1.0, 1.0)

        return action

    def reset(self):
        """Reset PID controllers (clear integral terms)."""
        self.pid_x.reset()
        self.pid_y.reset()
        self.pid_z.reset()


class HybridPIDRLController:
    """
    Hybrid controller combining PID baseline with RL correction.

    PID provides stable baseline tracking, RL learns fine-tuning corrections.
    Now supports 3D acceleration control.
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
        Compute hybrid action (3D acceleration).

        Args:
            observation: State vector (8D)
            deterministic: Use deterministic RL policy

        Returns:
            np.ndarray: Combined action [acc_x, acc_y, acc_z]
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
        """Reset both controllers."""
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
    kp_xy_range: tuple = (0.02, 0.1),
    kd_xy_range: tuple = (0.05, 0.15),
    kp_z_range: tuple = (0.1, 0.5),
    n_trials: int = 20,
    n_episodes: int = 10
) -> dict:
    """
    Simple grid search for PID gain tuning (3D acceleration control).

    Args:
        env: Gymnasium environment
        kp_xy_range: Range for XY position gain (min, max)
        kd_xy_range: Range for XY velocity gain (min, max)
        kp_z_range: Range for Z size gain (min, max)
        n_trials: Number of random trials
        n_episodes: Episodes per trial

    Returns:
        dict: Best gains and score
    """
    best_gains = None
    best_score = float('-inf')

    print(f"Tuning PID gains (3D acceleration) with {n_trials} trials, {n_episodes} episodes each...")

    for trial in range(n_trials):
        # Random sample
        kp_xy = np.random.uniform(*kp_xy_range)
        kd_xy = np.random.uniform(*kd_xy_range)
        kp_z = np.random.uniform(*kp_z_range)

        # Create controller
        controller = PIDController(kp_xy=kp_xy, kd_xy=kd_xy, kp_z=kp_z)
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
            best_gains = {"kp_xy": kp_xy, "kd_xy": kd_xy, "kp_z": kp_z}

        print(f"Trial {trial+1}/{n_trials}: Kp_xy={kp_xy:.4f}, Kd_xy={kd_xy:.4f}, Kp_z={kp_z:.4f}, Score={avg_reward:.2f}")

    print(f"\nBest gains: Kp_xy={best_gains['kp_xy']:.4f}, Kd_xy={best_gains['kd_xy']:.4f}, Kp_z={best_gains['kp_z']:.4f}")
    print(f"Best score: {best_score:.2f}")

    return best_gains
