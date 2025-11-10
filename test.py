"""
Test and demo script for Gym-Locker environment
"""

import argparse
import time
import numpy as np

import gymnasium as gym
from stable_baselines3 import PPO, SAC, TD3

import gym_locker
from gym_locker.agents.pid_controller import PIDController, HybridPIDRLController
from gym_locker.utils.visualization import create_video_from_frames


def test_environment():
    """
    Test basic environment functionality.
    """
    print("=" * 60)
    print("Testing Environment")
    print("=" * 60)

    env = gym.make('LockOn-v0', render_mode="human", state_mode="vector")

    print("\nObservation Space:", env.observation_space)
    print("Action Space:", env.action_space)

    # Test reset
    obs, info = env.reset()
    print("\nInitial observation:", obs)
    print("Initial info:", info)

    # Test random actions
    print("\nRunning random policy for 200 steps...")
    for step in range(200):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            print(f"\nEpisode ended at step {step}")
            print(f"  Final reward: {reward}")
            print(f"  Terminated: {terminated}")
            print(f"  Truncated: {truncated}")
            obs, info = env.reset()

        time.sleep(0.03)  # Slow down for visualization

    env.close()
    print("\nEnvironment test completed!")


def test_pid_controller():
    """
    Test PID controller in real-time.
    """
    print("=" * 60)
    print("Testing PID Controller")
    print("=" * 60)

    env = gym.make('LockOn-v0', render_mode="human", state_mode="vector")
    pid = PIDController(kp_xy=0.1, kd_xy=0.05, kp_z=0.3, ki_z=0.01, kd_z=0.1)
    pid.set_sample_time(env.metadata["render_fps"])

    print("\nRunning PID controller for 1 episode...")

    obs, _ = env.reset()
    pid.reset()

    total_reward = 0
    step = 0
    done = False

    while not done:
        action = pid.get_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step += 1

        done = terminated or truncated

        time.sleep(0.03)

    print(f"\nEpisode completed!")
    print(f"  Total reward: {total_reward:.2f}")
    print(f"  Episode length: {step}")
    print(f"  Total bonuses: {info['total_bonuses']}")

    env.close()


def test_rl_agent(model_path, state_mode="vector"):
    """
    Test trained RL agent in real-time.

    Args:
        model_path: Path to trained model
        state_mode: State representation mode
    """
    print("=" * 60)
    print("Testing RL Agent")
    print("=" * 60)

    # Load model
    if "PPO" in model_path:
        model = PPO.load(model_path)
    elif "SAC" in model_path:
        model = SAC.load(model_path)
    elif "TD3" in model_path:
        model = TD3.load(model_path)
    else:
        raise ValueError("Cannot determine algorithm from model path")

    env = gym.make('LockOn-v0', render_mode="human", state_mode=state_mode)

    print(f"\nRunning RL agent for 1 episode...")

    obs, _ = env.reset()
    total_reward = 0
    step = 0
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step += 1

        done = terminated or truncated

        time.sleep(0.03)

    print(f"\nEpisode completed!")
    print(f"  Total reward: {total_reward:.2f}")
    print(f"  Episode length: {step}")
    print(f"  Total bonuses: {info['total_bonuses']}")

    env.close()


def test_hybrid_controller(model_path, state_mode="vector", pid_weight=0.7, rl_weight=0.3):
    """
    Test hybrid PID+RL controller.

    Args:
        model_path: Path to trained RL model
        state_mode: State representation mode
        pid_weight: Weight for PID output
        rl_weight: Weight for RL output
    """
    print("=" * 60)
    print("Testing Hybrid PID+RL Controller")
    print(f"  PID Weight: {pid_weight}")
    print(f"  RL Weight: {rl_weight}")
    print("=" * 60)

    # Load RL model
    if "PPO" in model_path:
        rl_model = PPO.load(model_path)
    elif "SAC" in model_path:
        rl_model = SAC.load(model_path)
    elif "TD3" in model_path:
        rl_model = TD3.load(model_path)
    else:
        raise ValueError("Cannot determine algorithm from model path")

    # Create environment
    env = gym.make('LockOn-v0', render_mode="human", state_mode=state_mode)

    # Create hybrid controller
    pid = PIDController(kp_xy=0.1, kd_xy=0.05, kp_z=0.3, ki_z=0.01, kd_z=0.1)
    pid.set_sample_time(env.metadata["render_fps"])
    hybrid = HybridPIDRLController(pid, rl_model, pid_weight, rl_weight)

    print(f"\nRunning hybrid controller for 1 episode...")

    obs, _ = env.reset()
    hybrid.reset()

    total_reward = 0
    step = 0
    done = False

    while not done:
        action = hybrid.get_action(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step += 1

        done = terminated or truncated

        time.sleep(0.03)

    print(f"\nEpisode completed!")
    print(f"  Total reward: {total_reward:.2f}")
    print(f"  Episode length: {step}")
    print(f"  Total bonuses: {info['total_bonuses']}")

    env.close()


def record_video(
    controller_type="pid",
    model_path=None,
    state_mode="vector",
    output_path="output.mp4",
    n_episodes=1
):
    """
    Record video of agent performance.

    Args:
        controller_type: "pid", "rl", or "hybrid"
        model_path: Path to RL model (if applicable)
        state_mode: State representation mode
        output_path: Output video path
        n_episodes: Number of episodes to record
    """
    print("=" * 60)
    print(f"Recording Video: {controller_type}")
    print("=" * 60)

    # Create environment with rgb_array rendering
    env = gym.make('LockOn-v0', render_mode="rgb_array", state_mode=state_mode)

    # Create controller
    if controller_type == "pid":
        controller = PIDController(kp_xy=0.1, kd_xy=0.05, kp_z=0.3, ki_z=0.01, kd_z=0.1)
        controller.set_sample_time(env.metadata["render_fps"])
    elif controller_type == "rl":
        if model_path is None:
            raise ValueError("model_path required for RL controller")
        if "PPO" in model_path:
            controller = PPO.load(model_path)
        elif "SAC" in model_path:
            controller = SAC.load(model_path)
        elif "TD3" in model_path:
            controller = TD3.load(model_path)
    elif controller_type == "hybrid":
        if model_path is None:
            raise ValueError("model_path required for hybrid controller")
        if "PPO" in model_path:
            rl_model = PPO.load(model_path)
        elif "SAC" in model_path:
            rl_model = SAC.load(model_path)
        elif "TD3" in model_path:
            rl_model = TD3.load(model_path)
        pid = PIDController(kp_xy=0.1, kd_xy=0.05, kp_z=0.3, ki_z=0.01, kd_z=0.1)
        pid.set_sample_time(env.metadata["render_fps"])
        controller = HybridPIDRLController(pid, rl_model, 0.7, 0.3)
    else:
        raise ValueError(f"Unknown controller_type: {controller_type}")

    # Record frames
    all_frames = []

    for episode in range(n_episodes):
        obs, _ = env.reset()
        if hasattr(controller, 'reset'):
            controller.reset()

        done = False
        episode_frames = []

        while not done:
            # Get action
            if controller_type == "pid":
                action = controller.get_action(obs)
            elif controller_type == "rl":
                action, _ = controller.predict(obs, deterministic=True)
            elif controller_type == "hybrid":
                action = controller.get_action(obs, deterministic=True)

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)

            # Render frame
            frame = env.render()
            episode_frames.append(frame)

            done = terminated or truncated

        all_frames.extend(episode_frames)
        print(f"Episode {episode+1}/{n_episodes} recorded: {len(episode_frames)} frames")

    # Create video
    create_video_from_frames(all_frames, output_path, fps=env.metadata["render_fps"])

    env.close()


def main():
    parser = argparse.ArgumentParser(description="Test Gym-Locker Environment")

    parser.add_argument("--mode", type=str, default="env",
                        choices=["env", "pid", "rl", "hybrid", "video"],
                        help="Test mode")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Path to trained model (for RL/hybrid modes)")
    parser.add_argument("--state-mode", type=str, default="vector",
                        choices=["vector", "image"],
                        help="State representation mode")
    parser.add_argument("--pid-weight", type=float, default=0.7,
                        help="PID weight for hybrid controller")
    parser.add_argument("--rl-weight", type=float, default=0.3,
                        help="RL weight for hybrid controller")
    parser.add_argument("--output", type=str, default="output.mp4",
                        help="Output video path (for video mode)")
    parser.add_argument("--n-episodes", type=int, default=1,
                        help="Number of episodes (for video mode)")

    args = parser.parse_args()

    if args.mode == "env":
        test_environment()
    elif args.mode == "pid":
        test_pid_controller()
    elif args.mode == "rl":
        if args.model_path is None:
            raise ValueError("--model-path required for RL mode")
        test_rl_agent(args.model_path, args.state_mode)
    elif args.mode == "hybrid":
        if args.model_path is None:
            raise ValueError("--model-path required for hybrid mode")
        test_hybrid_controller(
            args.model_path,
            args.state_mode,
            args.pid_weight,
            args.rl_weight
        )
    elif args.mode == "video":
        controller_type = "pid"
        if args.model_path is not None:
            # Determine if hybrid based on weights
            if args.pid_weight > 0 and args.rl_weight > 0:
                controller_type = "hybrid"
            else:
                controller_type = "rl"

        record_video(
            controller_type=controller_type,
            model_path=args.model_path,
            state_mode=args.state_mode,
            output_path=args.output,
            n_episodes=args.n_episodes
        )

    print("\nTest completed!")


if __name__ == "__main__":
    main()
