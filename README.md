# Gym-Locker (Gymnasium-LockOn)

**Egocentric Target Tracking Environment for Reinforcement Learning**

Gym-Locker is a Gymnasium-based simulation environment that trains RL agents to maintain a lock-on target tracking system. The agent must keep a moving target centered on screen at a specific size (representing distance) for 5 seconds to achieve a successful lock-on.


> **⚠️ Latest Update (v2.0)**: Environment now uses **two-sided acceleration control** with **full state observation** (8D with velocities). Both pursuer and evader have realistic flight physics with momentum, drag, and angular velocity limits. This is a breaking change - retrain all models!
![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![Gymnasium](https://img.shields.io/badge/gymnasium-0.29%2B-green)

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Environment Details](#environment-details)
- [Training](#training)
- [Testing](#testing)
- [Kaggle Training](#kaggle-training)
- [Roadmap](#roadmap)
- [Contributing](#contributing)

## Features

- **Egocentric Simulation**: Simplified "ben-merkezci" (ego-centric) tracking without full 3D physics
- **Gymnasium Compatible**: Standard RL environment interface
- **Multiple State Representations**:
  - Vector mode: `[target_x, target_y, target_w, target_vx, target_vy, pursuer_vx, pursuer_vy, pursuer_vz]` (8D with velocities, Markovian)
  - Image mode: 84x84 grayscale (for CNN)
- **PID Baseline**: Traditional control theory baseline for comparison
- **Hybrid PID+RL**: Combine stability of PID with learning capability of RL
- **Multiple Algorithms**: Support for PPO, SAC, TD3
- **Evader Types**: Simple rule-based, random, or learned (MARL Phase 2)
- **Real-time Visualization**: Pygame rendering for debugging and demos
- **Kaggle Support**: Headless training on GPU

## Installation

### Local Installation

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/gym-locker.git
cd gym-locker

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### Kaggle Setup

1. Upload `notebooks/kaggle_training.ipynb` to Kaggle
2. Enable GPU in Settings
3. Clone this repository or add as dataset
4. Run the notebook

## Quick Start

### Test the Environment

```bash
# Test basic environment functionality
python test.py --mode env

# Test PID controller
python test.py --mode pid
```

### Train an Agent

```bash
# Train PPO agent with vector state (recommended for beginners)
python train.py --mode train --algorithm PPO --state-mode vector --timesteps 100000

# Train SAC agent with image state
python train.py --mode train --algorithm SAC --state-mode image --timesteps 500000
```

### Evaluate Trained Agent

```bash
# Evaluate with visualization
python test.py --mode rl --model-path models/PPO_final.zip --render

# Test hybrid PID+RL controller
python test.py --mode hybrid --model-path models/PPO_final.zip --pid-weight 0.7 --rl-weight 0.3
```

### Record Video

```bash
# Record PID controller
python test.py --mode video --output pid_demo.mp4

# Record RL agent
python test.py --mode video --model-path models/PPO_final.zip --output rl_demo.mp4
```

## Project Structure

```
gym-locker/
├── gym_locker/                 # Main package
│   ├── __init__.py            # Package initialization
│   ├── envs/
│   │   ├── __init__.py
│   │   └── lockon_env.py      # LockOnEnv environment
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── evader.py          # Evader implementations
│   │   ├── pid_controller.py  # PID baseline
│   │   └── networks.py        # PyTorch networks
│   └── utils/
│       ├── __init__.py
│       └── visualization.py   # Plotting and video tools
├── notebooks/
│   └── kaggle_training.ipynb  # Kaggle training notebook
├── models/                     # Saved models (created during training)
├── logs/                       # TensorBoard logs
├── train.py                    # Training script
├── test.py                     # Testing script
├── requirements.txt            # Dependencies
└── README.md                   # This file
```

## Environment Details

### State Space

**Option A - Vector (Recommended for beginners)**
```python
state = [x_diff, y_diff, width, height]
```
- `x_diff`: Horizontal offset from center (pixels)
- `y_diff`: Vertical offset from center (pixels)
- `width`: Target width (inversely proportional to distance)
- `height`: Target height (inversely proportional to distance)

**Option B - Image (Advanced)**
- 84x84 grayscale image
- Requires CNN architecture

### Action Space

**3D Acceleration Control** - realistic physics-based control:
```python
action = [acc_x, acc_y, acc_z]  # Each in range [-1.0, 1.0]
```
- `acc_x`: Horizontal acceleration
- `acc_y`: Vertical acceleration  
- `acc_z`: Forward/backward acceleration (controls distance to target)

**Key Change**: Agent now controls acceleration, not direct position. Pursuer has momentum, inertia, drag, and angular velocity limits.

### Reward Function

The reward function encourages:
1. **Centering**: Penalty for position error `-(x²+ y²)`
2. **Size matching**: Penalty for area error `-(area_error²) * 0.1`
3. **Lock-on maintenance**: `+1` per step when locked
4. **5-second lock bonus**: `+500` for maintaining lock for 5 seconds
5. **Escape penalty**: `-1000` if target escapes screen

### Physics Model

The environment uses a simplified "net vector" physics:

```
Net motion = Evader action - Pursuer action

Position update:
  x_new = x_old + net_x * speed_coefficient
  y_new = y_old + net_y * speed_coefficient

Distance update (affects size):
  distance_new = distance_old + net_z * speed_coefficient
  width = base_size / distance
```

## Training

### Phase 1: PID Baseline

Establish baseline performance:

```bash
python train.py --mode baseline --n-episodes 100
```

This evaluates a PID controller to establish a baseline score.

### Phase 2: RL Training

#### Vector State (Fast, Recommended)

```bash
# PPO (default)
python train.py --mode train \
    --algorithm PPO \
    --state-mode vector \
    --architecture simple \
    --timesteps 100000 \
    --evader-difficulty 0.5

# SAC (continuous control)
python train.py --mode train \
    --algorithm SAC \
    --state-mode vector \
    --timesteps 200000
```

#### Image State (Advanced)

```bash
python train.py --mode train \
    --algorithm PPO \
    --state-mode image \
    --architecture deep \
    --timesteps 500000
```

### Phase 3: Hybrid PID+RL

Test hybrid controller:

```bash
python test.py --mode hybrid \
    --model-path models/PPO_final.zip \
    --pid-weight 0.7 \
    --rl-weight 0.3
```

### Monitoring Training

Use TensorBoard to monitor training:

```bash
tensorboard --logdir logs/
```

## Testing

### Interactive Testing

```bash
# Test environment with random actions
python test.py --mode env

# Test PID controller
python test.py --mode pid

# Test trained RL agent
python test.py --mode rl --model-path models/PPO_final.zip

# Test hybrid controller
python test.py --mode hybrid --model-path models/PPO_final.zip
```

### Batch Evaluation

```bash
python train.py --mode eval \
    --model-path models/PPO_final.zip \
    --n-episodes 100
```

## Kaggle Training

For GPU-accelerated training:

1. Open `notebooks/kaggle_training.ipynb` in Kaggle
2. Enable GPU in Settings
3. Update configuration cells:
   ```python
   CONFIG = {
       'algorithm': 'PPO',
       'state_mode': 'vector',
       'total_timesteps': 500000,
       ...
   }
   ```
4. Run all cells
5. Download trained models from output

## Roadmap

### V0.1 - Setup ✅
- [x] Gymnasium environment
- [x] Pygame rendering
- [x] Vector and image state modes

### V0.2 - PID Baseline ✅
- [x] PID controller implementation
- [x] PID tuning utilities
- [x] Baseline evaluation

### V0.3 - RL Training ✅
- [x] PPO/SAC/TD3 support
- [x] Custom network architectures
- [x] Training scripts
- [x] Evaluation tools

### V0.4 - Advanced Features ✅
- [x] Hybrid PID+RL controller
- [x] Video recording
- [x] Kaggle notebook

### V1.0 - MARL (Future)
- [ ] Learned evader agent
- [ ] Multi-agent training (PettingZoo/RLlib)
- [ ] Self-play training
- [ ] Competitive evaluation

## Command Reference

### Training Commands

```bash
# Baseline evaluation
python train.py --mode baseline

# Train RL agent
python train.py --mode train --algorithm PPO --timesteps 100000

# Evaluate model
python train.py --mode eval --model-path models/PPO_final.zip

# Full pipeline (baseline + train + eval)
python train.py --mode all --algorithm PPO --timesteps 100000
```

### Testing Commands

```bash
# Test environment
python test.py --mode env

# Test PID
python test.py --mode pid

# Test RL agent
python test.py --mode rl --model-path models/PPO_final.zip

# Test hybrid
python test.py --mode hybrid --model-path models/PPO_final.zip

# Record video
python test.py --mode video --model-path models/PPO_final.zip --output demo.mp4
```

## Performance Tips

1. **Start with vector state**: Faster training, easier to debug
2. **Use PPO first**: Good default algorithm, stable training
3. **Tune evader difficulty**: Start with 0.3-0.5, increase gradually
4. **Monitor lock-on time**: Key metric for success
5. **Use Kaggle for long training**: Free GPU acceleration

## Troubleshooting

### Pygame/SDL Errors

If you get SDL errors on headless systems:
```python
import os
os.environ['SDL_VIDEODRIVER'] = 'dummy'
```

### Low Rewards

- Decrease evader difficulty
- Increase training timesteps
- Try SAC/TD3 for continuous control
- Check reward scaling in environment

### Slow Training

- Use vector state instead of image
- Reduce batch size
- Use Kaggle GPU
- Vectorize environments (future enhancement)

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Citation

If you use this environment in your research, please cite:

```bibtex
@software{gym_locker,
  title = {Gym-Locker: Egocentric Target Tracking Environment},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/YOUR_USERNAME/gym-locker}
}
```

## License

MIT License - see LICENSE file for details

## Acknowledgments

- Built with [Gymnasium](https://gymnasium.farama.org/)
- RL algorithms from [Stable-Baselines3](https://stable-baselines3.readthedocs.io/)
- Inspired by real-world target tracking systems

---

**Happy Training! 🎯**
