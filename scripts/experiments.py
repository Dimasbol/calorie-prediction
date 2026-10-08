"""Notebook orchestration: run/reuse, validation-only selection and learning curves."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch

from scripts.config import ROOT, load_config
from scripts.utils import train


def run_or_load(config, resume: bool = False) -> dict:
    cfg = load_config(config)
    checkpoint = cfg.path(cfg.run_dir) / "best.pt"
    if resume or not checkpoint.exists():
        return train(cfg, resume=resume)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    runtime_keys = {"epochs", "device", "num_workers", "log_every"}
    changed = [key for key, value in cfg.to_dict().items()
               if key not in runtime_keys and state["config"].get(key) != value]
    if changed:
        raise ValueError(f"Existing checkpoint differs from the config: {changed}. Use another run_dir.")
    result = {"best_checkpoint": str(checkpoint), "best_val_mae": state["best_mae"],
              "best_epoch": state["best_epoch"]}
    print("Reusing the saved checkpoint. Set RESUME_TRAINING=True to continue training.")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def compare_validation_runs(config_paths: list[str]) -> pd.DataFrame:
    rows, reference = [], None
    for path in config_paths:
        cfg = load_config(path)
        if cfg.use_mass:
            raise ValueError("Do not mix a mass-input model with the two-input model selection")
        folder = cfg.path(cfg.run_dir)
        split_table = pd.read_csv(folder / "split_manifest.csv").sort_values("dish_id").reset_index(drop=True)
        if reference is not None and not reference.equals(split_table):
            raise ValueError("Experiments use different train/validation splits")
        reference = split_table
        scores = json.loads((folder / "val_metrics.json").read_text(encoding="utf-8"))
        rows.append({"config": path, "run_dir": cfg.run_dir, "image_model": cfg.image_model,
                     "val_mae": scores["mae"], "val_rmse": scores["rmse"]})
    return pd.DataFrame(rows).sort_values("val_mae").reset_index(drop=True)


def plot_history(run_dir: str | Path, output_path: str | Path):
    folder = Path(run_dir)
    if not folder.is_absolute():
        folder = ROOT / folder
    history = pd.read_csv(folder / "history.csv")
    fig, axis = plt.subplots(figsize=(9, 4))
    axis.plot(history.epoch, history.train_mae, label="Train (augmentation/dropout active)")
    axis.plot(history.epoch, history.val_mae, label="Validation (evaluation mode)")
    axis.axhline(50, color="grey", linestyle="--", label="Target: 50 kcal")
    axis.set(xlabel="Epoch", ylabel="MAE, kcal", title="Learning curves")
    axis.legend()
    fig.tight_layout()
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=150)
    plt.close(fig)
    return history
