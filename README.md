# Phase 1: Grid World + Predictive Model

Implementation of [Ha & Schmidhuber's "World Models"](https://worldmodels.github.io/) — a neural network that learns to "dream" by predicting future states of a 2D grid environment.

## Architecture

Three neural modules working together:

1. **Vision (V)** — Convolutional VAE compresses 64x64 RGB observations to a 32-dim latent vector `z`
2. **Memory (M)** — GRU takes `(z_t, action_t)` and predicts `z_{t+1}`, reward, done
3. **Controller (C)** — Linear policy maps `(z, hidden_state)` to action, trained with CMA-ES

Once trained, the agent "dreams" — generates imagined trajectories entirely inside the model.

## Grid World

- 16x16 grid rendered as 64x64 RGB (4px per cell)
- Entities: Agent (blue), Walls (gray), Food (green), Hazards (red), Goal (gold)
- Actions: UP, DOWN, LEFT, RIGHT
- Rewards: Food +1.0, Hazard -1.0, Goal +5.0, Step -0.01

## Setup

```bash
pip install -e ".[dev]"
```

## Quick Start: Interactive Play

```bash
python scripts/play_interactive.py
```

Arrow keys to move, ESC to quit.

## Training Pipeline

Run each step sequentially:

```bash
# 1. Collect random rollouts (~3 min)
python scripts/01_collect_rollouts.py

# 2. Train VAE (~10 min)
python scripts/02_train_vae.py

# 3. Encode observations to latent vectors (~2 min)
python scripts/03_encode_dataset.py

# 4. Train RNN world model (~10 min)
python scripts/04_train_rnn.py

# 5. Train controller via CMA-ES (~20 min)
python scripts/05_train_controller.py

# 6. Generate dream trajectories
python scripts/06_dream.py

# 7. Visualize results
python scripts/07_visualize.py
```

## Tests

```bash
# All tests
pytest tests/

# End-to-end mini pipeline
pytest tests/test_pipeline.py
```

## Configuration

Edit `world_model/configs/phase1_default.yaml` to adjust hyperparameters.

## Tech Stack

- Python 3.11+, PyTorch (MPS on Apple Silicon)
- Pygame for interactive visualization
- CMA-ES for controller optimization
- No external RL libraries — everything from scratch
