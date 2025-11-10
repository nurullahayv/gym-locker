"""
Visualization utilities for Gym-Locker
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional
import cv2


def render_frame(
    screen_width: int,
    screen_height: int,
    target_x: float,
    target_y: float,
    target_width: float,
    target_height: float,
    is_locked: bool = False
) -> np.ndarray:
    """
    Render a frame without Pygame (for headless environments).

    Args:
        screen_width: Screen width in pixels
        screen_height: Screen height in pixels
        target_x: Target X offset from center
        target_y: Target Y offset from center
        target_width: Target width
        target_height: Target height
        is_locked: Whether target is locked on

    Returns:
        np.ndarray: RGB image
    """
    # Create black canvas
    frame = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)

    center_x = screen_width // 2
    center_y = screen_height // 2

    # Draw center lock box
    screen_area = screen_width * screen_height
    lock_box_size = int(np.sqrt(screen_area * 0.30))
    lock_box_x1 = center_x - lock_box_size // 2
    lock_box_y1 = center_y - lock_box_size // 2
    lock_box_x2 = lock_box_x1 + lock_box_size
    lock_box_y2 = lock_box_y1 + lock_box_size
    cv2.rectangle(frame, (lock_box_x1, lock_box_y1), (lock_box_x2, lock_box_y2), (50, 50, 50), 2)

    # Draw center crosshair
    cv2.line(frame, (center_x - 20, center_y), (center_x + 20, center_y), (100, 100, 100), 1)
    cv2.line(frame, (center_x, center_y - 20), (center_x, center_y + 20), (100, 100, 100), 1)

    # Draw target
    target_abs_x = int(center_x + target_x)
    target_abs_y = int(center_y + target_y)
    target_x1 = int(target_abs_x - target_width / 2)
    target_y1 = int(target_abs_y - target_height / 2)
    target_x2 = int(target_x1 + target_width)
    target_y2 = int(target_y1 + target_height)

    target_color = (0, 255, 0) if is_locked else (255, 0, 0)
    cv2.rectangle(frame, (target_x1, target_y1), (target_x2, target_y2), target_color, -1)

    return frame


def plot_training_metrics(
    rewards: List[float],
    episode_lengths: List[int],
    lock_times: List[float],
    save_path: Optional[str] = None
):
    """
    Plot training metrics.

    Args:
        rewards: List of episode rewards
        episode_lengths: List of episode lengths
        lock_times: List of total lock-on times per episode
        save_path: Path to save figure (optional)
    """
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))

    # Moving average helper
    def moving_average(data, window=10):
        if len(data) < window:
            return data
        return np.convolve(data, np.ones(window)/window, mode='valid')

    # Plot rewards
    axes[0].plot(rewards, alpha=0.3, label='Raw')
    if len(rewards) > 10:
        axes[0].plot(moving_average(rewards, 10), label='MA(10)', linewidth=2)
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Total Reward')
    axes[0].set_title('Episode Rewards')
    axes[0].legend()
    axes[0].grid(True)

    # Plot episode lengths
    axes[1].plot(episode_lengths, alpha=0.3, label='Raw')
    if len(episode_lengths) > 10:
        axes[1].plot(moving_average(episode_lengths, 10), label='MA(10)', linewidth=2)
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Steps')
    axes[1].set_title('Episode Length')
    axes[1].legend()
    axes[1].grid(True)

    # Plot lock-on times
    axes[2].plot(lock_times, alpha=0.3, label='Raw')
    if len(lock_times) > 10:
        axes[2].plot(moving_average(lock_times, 10), label='MA(10)', linewidth=2)
    axes[2].set_xlabel('Episode')
    axes[2].set_ylabel('Lock-On Time (steps)')
    axes[2].set_title('Total Lock-On Time per Episode')
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Training metrics saved to {save_path}")
    else:
        plt.show()


def plot_comparison(
    pid_rewards: List[float],
    rl_rewards: List[float],
    hybrid_rewards: Optional[List[float]] = None,
    save_path: Optional[str] = None
):
    """
    Compare performance of different controllers.

    Args:
        pid_rewards: PID controller rewards
        rl_rewards: RL controller rewards
        hybrid_rewards: Hybrid controller rewards (optional)
        save_path: Path to save figure
    """
    plt.figure(figsize=(12, 6))

    def moving_average(data, window=10):
        if len(data) < window:
            return data
        return np.convolve(data, np.ones(window)/window, mode='valid')

    # Plot PID
    plt.plot(moving_average(pid_rewards, 10), label='PID', linewidth=2, alpha=0.7)

    # Plot RL
    plt.plot(moving_average(rl_rewards, 10), label='RL', linewidth=2, alpha=0.7)

    # Plot Hybrid if available
    if hybrid_rewards is not None:
        plt.plot(moving_average(hybrid_rewards, 10), label='Hybrid (PID+RL)', linewidth=2, alpha=0.7)

    plt.xlabel('Episode')
    plt.ylabel('Total Reward (MA 10)')
    plt.title('Controller Performance Comparison')
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Comparison plot saved to {save_path}")
    else:
        plt.show()


def create_video_from_frames(
    frames: List[np.ndarray],
    output_path: str,
    fps: int = 30
):
    """
    Create video from list of frames.

    Args:
        frames: List of RGB frames (numpy arrays)
        output_path: Output video path (.mp4)
        fps: Frames per second
    """
    if len(frames) == 0:
        print("No frames to create video")
        return

    height, width, _ = frames[0].shape

    # Define codec and create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame in frames:
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)

    out.release()
    print(f"Video saved to {output_path}")
