"""
Gym-Locker: Gymnasium-based Lock-On simulation environment
"""

from gymnasium.envs.registration import register

register(
    id='LockOn-v0',
    entry_point='gym_locker.envs:LockOnEnv',
    max_episode_steps=1000,
)

__version__ = '0.1.0'
