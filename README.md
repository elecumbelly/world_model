# Phase 1: Grid World + Predictive Model

Implementation of [Ha & Schmidhuber's "World Models"](https://worldmodels.github.io/) — a neural network that learns to "dream" by predicting future states of a 2D grid environment.

## Architecture

Three neural modules working together:

1. **Vision (V)** — Convolutional VAE compresses 64x64 RGB observations to a 32-dim latent vector `z`
2. **Memory (M)** — GRU takes `(z_t, action_t)` and predicts `z_{t+1}`, reward, done
3. **Controller (C)** — Linear policy maps `(z, hidden_state)` to action, trained with CMA-ES

Once trained, the agent "dreams" — generates imagined trajectories entirely inside the model.

```
                         ┌─────────────────────────────────────────────┐
                         │           WORLD MODEL ARCHITECTURE          │
                         └─────────────────────────────────────────────┘

  ┌──────────────────┐        ┌──────────────────┐        ┌──────────────────┐
  │   VISION (V)     │        │   MEMORY (M)     │        │  CONTROLLER (C)  │
  │   ConvVAE        │        │   GRU            │        │  Linear Policy   │
  │                  │        │                  │        │                  │
  │  64x64x3 RGB     │        │  z_t (32)        │        │  z_t (32)        │
  │       │          │        │  action_t (4)    │        │  h_t (256)       │
  │  ┌────▼────┐     │        │       │          │        │       │          │
  │  │ Encoder │     │        │  ┌────▼────┐     │        │  ┌────▼────┐     │
  │  │ 4xConv  │     │        │  │  GRU    │     │        │  │ Linear  │     │
  │  │ stride-2│     │        │  │ h=256   │     │        │  │ 288→4   │     │
  │  └────┬────┘     │        │  └────┬────┘     │        │  └────┬────┘     │
  │       │          │        │       │          │        │       │          │
  │   ┌───▼───┐      │        │  ┌────▼─────┐   │        │   action (argmax)│
  │   │ μ, σ² │      │        │  │ z_{t+1}  │   │        │                  │
  │   └───┬───┘      │        │  │ reward   │   │        │  1,156 params    │
  │       │          │        │  │ done     │   │        │  Trained: CMA-ES │
  │    z (32-dim)    │        │  └──────────┘   │        │                  │
  │       │          │        │                  │        │                  │
  │  ┌────▼────┐     │        │                  │        │                  │
  │  │ Decoder │     │        │                  │        │                  │
  │  │ 4xDeconv│     │        │                  │        │                  │
  │  └────┬────┘     │        │                  │        │                  │
  │       │          │        │                  │        │                  │
  │  64x64x3 recon   │        │                  │        │                  │
  └──────────────────┘        └──────────────────┘        └──────────────────┘
     Trained: SGD                Trained: SGD              Trained: CMA-ES
     β-VAE (β=0.5)              MSE + BCE loss             Population: 64


  ┌──────────────────────────────────────────────────────────────────────────┐
  │                        DATA FLOW (INFERENCE)                            │
  │                                                                         │
  │   Environment ──obs──▶ V ──z──┬──▶ C ──action──▶ Environment           │
  │                               │       ▲                                 │
  │                               ▼       │                                 │
  │                               M ──h_t─┘                                │
  │                               │                                         │
  │                          z_{t+1}, reward, done                          │
  └──────────────────────────────────────────────────────────────────────────┘


  ┌──────────────────────────────────────────────────────────────────────────┐
  │                        DATA FLOW (DREAMING)                             │
  │                                                                         │
  │   Initial obs ──▶ V ──z₀──┐                                            │
  │                            │    ┌──────────────────────┐                │
  │                            ▼    ▼                      │                │
  │                     C ──action──▶ M ──z_{t+1}──┐       │                │
  │                     ▲              │            │       │                │
  │                     │              ▼            ▼       │                │
  │                     └──────── h_{t+1}    V.decode ──▶ dream frame       │
  │                                                                         │
  │   No environment needed — agent imagines entirely from learned model    │
  └──────────────────────────────────────────────────────────────────────────┘
```

## Grid World

```
  ┌──────────────────────────────────────────────┐
  │              GRID WORLD LEGEND                │
  │                                               │
  │   ██ Black    = Empty         reward:  0      │
  │   ██ Gray     = Wall          (impassable)    │
  │   ██ Blue     = Agent         reward: -0.01/step │
  │   ██ Green    = Food          reward: +1.0    │
  │   ██ Red      = Hazard        reward: -1.0    │
  │   ██ Gold     = Goal          reward: +5.0    │
  │                                               │
  │   16x16 grid → 64x64 RGB (4px per cell)      │
  │   Max 200 steps/episode                       │
  │   Food respawns after 10 steps                │
  │                                               │
  │   Actions: ↑ UP  ↓ DOWN  ← LEFT  → RIGHT     │
  │   Wall collision = no movement                │
  └──────────────────────────────────────────────┘

  Example 8x8 grid (actual is 16x16):

  ████████████████████████████████
  ██                      ██  ██
  ██  ████  ██████  ████      ██
  ██      ██      ████    ██  ██
  ██  ██      ██      ██████  ██
  ██  ██  ██████  ██          ██
  ██          ██      ████    ██
  ████████████████████████████████

  ██ = wall   ██ = agent   ██ = food   ██ = hazard   ██ = goal
```

## Training Pipeline

```
  ┌─────────────────────────────────────────────────────────────────────────┐
  │                     TRAINING PIPELINE (Sequential)                      │
  └─────────────────────────────────────────────────────────────────────────┘

  Step 1                Step 2                Step 3
  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐
  │  Collect     │      │  Train VAE   │      │  Encode      │
  │  Rollouts    │─────▶│              │─────▶│  Dataset     │
  │              │      │  20 epochs   │      │              │
  │  1000 random │      │  β=0.5       │      │  VAE.encode  │
  │  episodes    │      │  lr=1e-3     │      │  all obs     │
  └──────┬───────┘      └──────┬───────┘      └──────┬───────┘
         │                     │                     │
         ▼                     ▼                     ▼
  data/rollouts/        checkpoints/           data/encoded/
  rollout_*.npz         vae/best.pt            encoded_*.npz
  (obs, act, rew, done) (model weights)        (latents, act, rew, done)


  Step 4                Step 5                Step 6 & 7
  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐
  │  Train RNN   │      │  Train       │      │  Dream &     │
  │              │─────▶│  Controller  │─────▶│  Visualize   │
  │  30 epochs   │      │              │      │              │
  │  GRU h=256   │      │  CMA-ES      │      │  Compare     │
  │  lr=1e-3     │      │  pop=64      │      │  real vs     │
  └──────┬───────┘      │  100 gens    │      │  dream       │
         │              └──────┬───────┘      └──────────────┘
         ▼                     ▼
  checkpoints/           checkpoints/          Outputs:
  rnn/best.pt            controller/best.pt    - real_vs_dream.gif
  (model weights)        (model weights)       - vae_reconstructions.png
                                               - latent_space.png
                                               - loss curves

  Total time: ~25-45 min on Apple Silicon (M1 Pro)
```

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
# All tests (41 tests)
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
