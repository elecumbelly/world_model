from dataclasses import dataclass, field, asdict
from pathlib import Path

import yaml


@dataclass
class EnvConfig:
    grid_size: int = 16
    render_size: int = 64
    max_steps: int = 200
    wall_density: float = 0.1
    num_food: int = 5
    num_hazards: int = 3
    food_respawn_steps: int = 10
    reward_food: float = 1.0
    reward_hazard: float = -1.0
    reward_goal: float = 5.0
    reward_step: float = -0.01


@dataclass
class VAEConfig:
    latent_dim: int = 32
    input_channels: int = 3
    beta: float = 0.5
    lr: float = 1e-3
    batch_size: int = 64
    epochs: int = 20


@dataclass
class RNNConfig:
    latent_dim: int = 32
    action_dim: int = 4
    hidden_dim: int = 256
    lr: float = 1e-3
    batch_size: int = 32
    epochs: int = 30
    seq_len: int = 50


@dataclass
class ControllerConfig:
    latent_dim: int = 32
    hidden_dim: int = 256
    action_dim: int = 4
    population_size: int = 64
    generations: int = 100
    num_rollouts: int = 16
    sigma_init: float = 0.1


@dataclass
class CollectConfig:
    num_rollouts: int = 1000
    policy: str = "random"


@dataclass
class Config:
    env: EnvConfig = field(default_factory=EnvConfig)
    vae: VAEConfig = field(default_factory=VAEConfig)
    rnn: RNNConfig = field(default_factory=RNNConfig)
    controller: ControllerConfig = field(default_factory=ControllerConfig)
    collect: CollectConfig = field(default_factory=CollectConfig)
    seed: int = 42
    data_dir: str = "data"
    checkpoint_dir: str = "checkpoints"


def load_config(path: str | Path) -> Config:
    path = Path(path)
    if not path.exists():
        return Config()
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    cfg = Config()
    for section_name, section_cls in [
        ("env", EnvConfig), ("vae", VAEConfig), ("rnn", RNNConfig),
        ("controller", ControllerConfig), ("collect", CollectConfig),
    ]:
        if section_name in raw:
            setattr(cfg, section_name, section_cls(**raw[section_name]))
    for key in ("seed", "data_dir", "checkpoint_dir"):
        if key in raw:
            setattr(cfg, key, raw[key])
    return cfg


def save_config(cfg: Config, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(asdict(cfg), f, default_flow_style=False, sort_keys=False)
