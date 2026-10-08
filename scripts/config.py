"""Configuration; relative paths are resolved from the repository root."""

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Config:
    data_dir: str = "data"
    run_dir: str = "artifacts/main"
    seed: int = 42
    val_fraction: float = 0.2
    image_model: str = "resnet18.a1_in1k"
    text_model: str = "distilbert/distilbert-base-uncased"
    pretrained: bool = True
    image_size: int = 224
    max_length: int = 128
    projection_dim: int = 256
    dropout: float = 0.2
    use_mass: bool = False
    image_trainable: str = "all"
    text_trainable_layers: int = 2
    freeze_batchnorm: bool = True
    batch_size: int = 4
    accumulation_steps: int = 4
    num_workers: int = 0
    epochs: int = 25
    warmup_epochs: int = 2
    patience: int = 6
    image_lr: float = 0.0001
    text_lr: float = 0.00002
    head_lr: float = 0.001
    weight_decay: float = 0.0001
    grad_clip: float = 1.0
    amp: bool = True
    device: str = "cuda"
    deterministic: bool = True
    log_every: int = 100

    def validate(self) -> None:
        for name in ("batch_size", "accumulation_steps", "epochs", "patience", "max_length"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if not 0 < self.val_fraction < 0.5:
            raise ValueError("val_fraction must be between 0 and 0.5")
        if not 0 <= self.warmup_epochs < self.epochs:
            raise ValueError("warmup_epochs must be smaller than epochs")
        if self.image_trainable not in {"all", "last_stage", "none"}:
            raise ValueError("image_trainable: all, last_stage or none")
        if self.text_trainable_layers < 0 or self.num_workers < 0:
            raise ValueError("Layer and worker counts cannot be negative")
        if self.image_size < 32:
            raise ValueError("image_size must be at least 32")

    def path(self, value: str) -> Path:
        p = Path(value).expanduser()
        return p if p.is_absolute() else ROOT / p

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(source: str | Path | dict | Config) -> Config:
    if isinstance(source, Config):
        cfg = Config(**source.to_dict())
    else:
        if isinstance(source, dict):
            values = source.copy()
        else:
            path = Path(source)
            if not path.is_absolute():
                path = ROOT / path
            with path.open(encoding="utf-8") as f:
                values = yaml.safe_load(f)
        unknown = set(values) - {f.name for f in fields(Config)}
        if unknown:
            raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
        cfg = Config(**values)
    cfg.validate()
    return cfg
