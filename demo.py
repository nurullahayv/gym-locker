"""
Simple demo script for Gym-Locker environment
"""

import gymnasium as gym
import gym_locker
import time


def demo_random_policy():
    """
    Demo with random actions.
    """
    print("=" * 60)
    print("Demo: Random Policy")
    print("=" * 60)

    env = gym.make('LockOn-v0', render_mode="human", state_mode="vector")

    for episode in range(3):
        print(f"\nEpisode {episode + 1}/3")
        obs, info = env.reset()

        total_reward = 0
        steps = 0
        done = False

        while not done and steps < 500:
            # Random action
            action = env.action_space.sample()

            # Step
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1

            done = terminated or truncated

            # Slow down for visualization
            time.sleep(0.03)

        print(f"  Episode finished: Reward={total_reward:.2f}, Steps={steps}")

    env.close()
    print("\nDemo completed!")


def demo_manual_control():
    """
    Demo with keyboard control (arrow keys).
    """
    print("=" * 60)
    print("Demo: Manual Control")
    print("=" * 60)
    print("\nControls:")
    print("  Arrow Keys: Move target correction")
    print("  ESC: Quit")
    print("  R: Reset episode")
    print("\nStarting in 3 seconds...")
    time.sleep(3)

    import pygame

    env = gym.make('LockOn-v0', render_mode="human", state_mode="vector")
    obs, info = env.reset()

    running = True
    total_reward = 0
    steps = 0

    while running:
        # Default action (no movement)
        action = [0.0, 0.0]

        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    obs, info = env.reset()
                    total_reward = 0
                    steps = 0
                    print("\nEpisode reset")

        # Get keyboard state
        keys = pygame.key.get_pressed()

        # Map arrow keys to actions
        if keys[pygame.K_LEFT]:
            action[0] -= 0.5
        if keys[pygame.K_RIGHT]:
            action[0] += 0.5
        if keys[pygame.K_UP]:
            action[1] -= 0.5
        if keys[pygame.K_DOWN]:
            action[1] += 0.5

        # Clip action to valid range
        action = [max(-1.0, min(1.0, a)) for a in action]

        # Step
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1

        # Check if episode ended
        if terminated or truncated:
            print(f"\nEpisode ended: Reward={total_reward:.2f}, Steps={steps}")
            obs, info = env.reset()
            total_reward = 0
            steps = 0

        time.sleep(0.03)

    env.close()
    print("\nManual control demo completed!")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Gym-Locker Demo")
    parser.add_argument("--mode", type=str, default="random",
                        choices=["random", "manual"],
                        help="Demo mode")

    args = parser.parse_args()

    if args.mode == "random":
        demo_random_policy()
    elif args.mode == "manual":
        demo_manual_control()


if __name__ == "__main__":
    main()
