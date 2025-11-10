"""
Training script for Gym-Locker environment
"""

import argparse
import os
from datetime import datetime
import numpy as np

import gymnasium as gym
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

import gym_locker
from gym_locker.agents.pid_controller import PIDController, tune_pid_gains
from gym_locker.agents.networks import get_feature_extractor
from gym_locker.utils.visualization import plot_training_metrics


def make_env(state_mode="vector", render_mode=None, evader_difficulty=0.5, evader_speed_multiplier=1.5):
    """Create and wrap environment."""
    env = gym.make(
        'LockOn-v0',
        render_mode=render_mode,
        state_mode=state_mode,
        evader_difficulty=evader_difficulty,
        evader_speed_multiplier=evader_speed_multiplier
    )
    return env


def evaluate_pid_baseline(n_episodes=100, render=False):
    """
    Evaluate PID controller as baseline.

    Args:
        n_episodes: Number of episodes to evaluate
        render: Whether to render (slower)

    Returns:
        dict: Evaluation results
    """
    print("=" * 60)
    print("PHASE 1: PID Baseline Evaluation")
    print("=" * 60)

    # Create environment
    render_mode = "human" if render else None
    env = make_env(state_mode="vector", render_mode=render_mode)

    # Create PID controller
    pid = PIDController(kp=0.1, ki=0.01, kd=0.05)
    pid.set_sample_time(env.metadata["render_fps"])

    # Evaluate
    episode_rewards = []
    episode_lengths = []
    lock_times = []

    for episode in range(n_episodes):
        obs, _ = env.reset()
        pid.reset()

        episode_reward = 0
        episode_length = 0
        lock_time = 0
        done = False

        while not done:
            action = pid.get_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)

            episode_reward += reward
            episode_length += 1
            if info['is_locked']:
                lock_time += 1

            done = terminated or truncated

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        lock_times.append(lock_time)

        if (episode + 1) % 10 == 0:
            print(f"Episode {episode+1}/{n_episodes}: "
                  f"Reward={episode_reward:.2f}, "
                  f"Length={episode_length}, "
                  f"Lock Time={lock_time}")

    env.close()

    # Statistics
    results = {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'mean_lock_time': np.mean(lock_times),
        'rewards': episode_rewards,
        'lengths': episode_lengths,
        'lock_times': lock_times
    }

    print("\n" + "=" * 60)
    print("PID Baseline Results:")
    print(f"  Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"  Mean Episode Length: {results['mean_length']:.2f}")
    print(f"  Mean Lock Time: {results['mean_lock_time']:.2f}")
    print("=" * 60 + "\n")

    return results


def train_rl_agent(
    algorithm="PPO",
    state_mode="vector",
    architecture="simple",
    total_timesteps=100000,
    evader_difficulty=0.5,
    save_dir="models",
    log_dir="logs"
):
    """
    Train RL agent.

    Args:
        algorithm: RL algorithm ("PPO", "SAC", "TD3")
        state_mode: State representation ("vector" or "image")
        architecture: Network architecture ("simple" or "deep")
        total_timesteps: Total training timesteps
        evader_difficulty: Evader difficulty (0.0-1.0)
        save_dir: Directory to save models
        log_dir: Directory for tensorboard logs

    Returns:
        Trained model
    """
    print("=" * 60)
    print(f"PHASE 2: Training {algorithm} Agent")
    print(f"  State Mode: {state_mode}")
    print(f"  Architecture: {architecture}")
    print(f"  Total Timesteps: {total_timesteps}")
    print(f"  Evader Difficulty: {evader_difficulty}")
    print("=" * 60)

    # Create directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = os.path.join(save_dir, f"{algorithm}_{state_mode}_{timestamp}")
    tensorboard_dir = os.path.join(log_dir, f"{algorithm}_{state_mode}_{timestamp}")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)

    # Create training environment
    train_env = make_env(state_mode=state_mode, evader_difficulty=evader_difficulty)
    train_env = Monitor(train_env)
    train_env = DummyVecEnv([lambda: train_env])

    # Create evaluation environment
    eval_env = make_env(state_mode=state_mode, evader_difficulty=evader_difficulty)
    eval_env = Monitor(eval_env)

    # Get feature extractor
    feature_extractor_class = get_feature_extractor(state_mode, architecture)

    # Policy kwargs
    policy_kwargs = dict(
        features_extractor_class=feature_extractor_class,
        features_extractor_kwargs=dict(features_dim=64),
    )

    # Create model
    if algorithm == "PPO":
        model = PPO(
            "MlpPolicy" if state_mode == "vector" else "CnnPolicy",
            train_env,
            policy_kwargs=policy_kwargs,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            verbose=1,
            tensorboard_log=tensorboard_dir
        )
    elif algorithm == "SAC":
        model = SAC(
            "MlpPolicy" if state_mode == "vector" else "CnnPolicy",
            train_env,
            policy_kwargs=policy_kwargs,
            learning_rate=3e-4,
            buffer_size=100000,
            learning_starts=1000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            verbose=1,
            tensorboard_log=tensorboard_dir
        )
    elif algorithm == "TD3":
        model = TD3(
            "MlpPolicy" if state_mode == "vector" else "CnnPolicy",
            train_env,
            policy_kwargs=policy_kwargs,
            learning_rate=3e-4,
            buffer_size=100000,
            learning_starts=1000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            verbose=1,
            tensorboard_log=tensorboard_dir
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=model_dir,
        name_prefix=f"{algorithm}_model"
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=model_dir,
        log_path=model_dir,
        eval_freq=5000,
        deterministic=True,
        render=False
    )

    # Train
    print("\nStarting training...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, eval_callback],
        progress_bar=True
    )

    # Save final model
    final_model_path = os.path.join(model_dir, f"{algorithm}_final.zip")
    model.save(final_model_path)
    print(f"\nFinal model saved to: {final_model_path}")

    # Cleanup
    train_env.close()
    eval_env.close()

    return model, model_dir


def evaluate_model(model_path, n_episodes=100, render=False, state_mode="vector"):
    """
    Evaluate trained model.

    Args:
        model_path: Path to saved model
        n_episodes: Number of episodes to evaluate
        render: Whether to render
        state_mode: State representation mode

    Returns:
        dict: Evaluation results
    """
    print("=" * 60)
    print("Evaluating Trained Model")
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

    # Create environment
    render_mode = "human" if render else None
    env = make_env(state_mode=state_mode, render_mode=render_mode)

    # Evaluate
    episode_rewards = []
    episode_lengths = []
    lock_times = []

    for episode in range(n_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        lock_time = 0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            episode_reward += reward
            episode_length += 1
            if info['is_locked']:
                lock_time += 1

            done = terminated or truncated

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        lock_times.append(lock_time)

        if (episode + 1) % 10 == 0:
            print(f"Episode {episode+1}/{n_episodes}: "
                  f"Reward={episode_reward:.2f}, "
                  f"Length={episode_length}, "
                  f"Lock Time={lock_time}")

    env.close()

    # Statistics
    results = {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'mean_lock_time': np.mean(lock_times),
        'rewards': episode_rewards,
        'lengths': episode_lengths,
        'lock_times': lock_times
    }

    print("\n" + "=" * 60)
    print("Model Evaluation Results:")
    print(f"  Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"  Mean Episode Length: {results['mean_length']:.2f}")
    print(f"  Mean Lock Time: {results['mean_lock_time']:.2f}")
    print("=" * 60 + "\n")

    return results


def main():
    parser = argparse.ArgumentParser(description="Train Gym-Locker Agent")

    parser.add_argument("--mode", type=str, default="train",
                        choices=["baseline", "train", "eval", "all"],
                        help="Execution mode")
    parser.add_argument("--algorithm", type=str, default="PPO",
                        choices=["PPO", "SAC", "TD3"],
                        help="RL algorithm")
    parser.add_argument("--state-mode", type=str, default="vector",
                        choices=["vector", "image"],
                        help="State representation mode")
    parser.add_argument("--architecture", type=str, default="simple",
                        choices=["simple", "deep"],
                        help="Network architecture")
    parser.add_argument("--timesteps", type=int, default=100000,
                        help="Total training timesteps")
    parser.add_argument("--evader-difficulty", type=float, default=0.5,
                        help="Evader difficulty (0.0-1.0)")
    parser.add_argument("--n-episodes", type=int, default=100,
                        help="Number of evaluation episodes")
    parser.add_argument("--render", action="store_true",
                        help="Render environment during evaluation")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Path to model for evaluation")
    parser.add_argument("--save-dir", type=str, default="models",
                        help="Directory to save models")
    parser.add_argument("--log-dir", type=str, default="logs",
                        help="Directory for logs")

    args = parser.parse_args()

    if args.mode == "baseline" or args.mode == "all":
        # Evaluate PID baseline
        pid_results = evaluate_pid_baseline(
            n_episodes=args.n_episodes,
            render=args.render
        )

    if args.mode == "train" or args.mode == "all":
        # Train RL agent
        model, model_dir = train_rl_agent(
            algorithm=args.algorithm,
            state_mode=args.state_mode,
            architecture=args.architecture,
            total_timesteps=args.timesteps,
            evader_difficulty=args.evader_difficulty,
            save_dir=args.save_dir,
            log_dir=args.log_dir
        )

        # Evaluate trained model
        final_model_path = os.path.join(model_dir, f"{args.algorithm}_final.zip")
        rl_results = evaluate_model(
            model_path=final_model_path,
            n_episodes=args.n_episodes,
            render=args.render,
            state_mode=args.state_mode
        )

    if args.mode == "eval":
        if args.model_path is None:
            raise ValueError("--model-path required for eval mode")

        # Evaluate specific model
        results = evaluate_model(
            model_path=args.model_path,
            n_episodes=args.n_episodes,
            render=args.render,
            state_mode=args.state_mode
        )

    print("\nTraining/Evaluation completed!")


if __name__ == "__main__":
    main()
