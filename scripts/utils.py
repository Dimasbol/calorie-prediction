"""Reproducible training entry point: train(config_path, resume=False)."""

import argparse
import gc
import json
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import AutoTokenizer
import yaml

from scripts.config import Config, load_config
from scripts.data import dataset_fingerprint, manifest, prepare_data
from scripts.dataset import DishDataset, build_transform, make_loader
from scripts.model import CalorieModel


def seed_everything(seed: int, deterministic: bool = True) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(deterministic)


def get_device(value: str) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Select the .venv kernel, install the CUDA PyTorch "
                           "wheel and check the NVIDIA driver. See START_HERE.md. "
                           "Training will not silently fall back to CPU.")
    return device


def regression_metrics(target, prediction) -> dict:
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    if target.shape != prediction.shape or not target.size:
        raise ValueError("Targets and predictions must have the same nonempty shape")
    if not np.isfinite(target).all() or not np.isfinite(prediction).all():
        raise ValueError("Non-finite target or prediction")
    error = prediction - target
    denom = np.square(target - target.mean()).sum()
    return {"n": int(target.size), "mae": float(np.abs(error).mean()),
            "rmse": float(np.sqrt(np.square(error).mean())),
            "r2": float(1 - np.square(error).sum() / denom) if denom > 0 else None,
            "median_ae": float(np.median(np.abs(error))),
            "p90_ae": float(np.quantile(np.abs(error), 0.9)),
            "bias": float(error.mean()), "negative_predictions": int((prediction < 0).sum())}


def fit_scalers(train_frame: pd.DataFrame, use_mass: bool) -> dict:
    y = train_frame["total_calories"].to_numpy(dtype=np.float64)
    mass = np.log1p(train_frame["total_mass"].to_numpy(dtype=float)) if use_mass else np.zeros(len(y))
    return {"target_mean": float(y.mean()), "target_std": max(float(y.std()), 1.0),
            "target_median": float(np.median(y)),
            "mass_mean": float(mass.mean()), "mass_std": max(float(mass.std()), 1e-6)}


def build_optimizer(model: CalorieModel, cfg: Config) -> AdamW:
    # Include even temporarily frozen parameters: warm-up later unfreezes them.
    groups = [
        {"params": list(model.image_encoder.parameters()), "lr": cfg.image_lr, "name": "image_encoder"},
        {"params": list(model.text_encoder.parameters()), "lr": cfg.text_lr, "name": "text_encoder"},
        {"params": list(model.image_projection.parameters()), "lr": cfg.head_lr, "name": "image_projection"},
        {"params": list(model.text_projection.parameters()), "lr": cfg.head_lr, "name": "text_projection"},
        {"params": list(model.regressor.parameters()), "lr": cfg.head_lr, "name": "regressor"},
    ]
    if model.mass_projection is not None:
        groups.append({"params": list(model.mass_projection.parameters()), "lr": cfg.head_lr,
                       "name": "mass_projection"})
    optimizer = AdamW(groups, weight_decay=cfg.weight_decay)
    assert_optimizer_covers_model(model, optimizer)
    return optimizer


def assert_optimizer_covers_model(model: nn.Module, optimizer) -> None:
    ids = [id(p) for group in optimizer.param_groups for p in group["params"]]
    if len(ids) != len(set(ids)):
        raise AssertionError("A parameter occurs in more than one optimizer group")
    if set(ids) != {id(p) for p in model.parameters()}:
        raise AssertionError("The optimizer does not cover every model parameter")


def move_batch(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()}


def model_forward(model: nn.Module, batch: dict):
    return model(batch["image"], batch["input_ids"], batch["attention_mask"], batch["mass"])


def finish_accumulation(model, optimizer, scaler, sample_count: int, grad_clip: float) -> None:
    """Normalize by actual sample count, including the final incomplete batch."""
    scaler.unscale_(optimizer)
    for parameter in model.parameters():
        if parameter.grad is not None:
            parameter.grad.div_(sample_count)
    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)


def train_epoch(model, loader, optimizer, scaler, stats, cfg, device) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    criterion = nn.L1Loss(reduction="sum")
    total_error, total_count, group_count = 0.0, 0, 0
    use_amp = cfg.amp and device.type == "cuda"
    for step, raw_batch in enumerate(loader, 1):
        batch = move_batch(raw_batch, device)
        normalized_target = (batch["target"] - stats["target_mean"]) / stats["target_std"]
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            prediction = model_forward(model, batch)
        loss = criterion(prediction.float(), normalized_target.float())
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite loss; retry with amp: false and a smaller learning rate")
        scaler.scale(loss).backward()
        count = batch["target"].numel()
        group_count += count
        total_count += count
        total_error += loss.detach().item() * stats["target_std"]
        if step % cfg.accumulation_steps == 0 or step == len(loader):
            finish_accumulation(model, optimizer, scaler, group_count, cfg.grad_clip)
            group_count = 0
        if cfg.log_every and (step % cfg.log_every == 0 or step == len(loader)):
            print(f"  batch {step}/{len(loader)} | train MAE so far {total_error / total_count:.3f} kcal", flush=True)
    return total_error / total_count


@torch.inference_mode()
def predict_loader(model, loader, stats: dict, device: torch.device) -> pd.DataFrame:
    model.eval()
    rows = []
    # FP32 validation/inference: one consistent policy for checkpoint selection and test.
    for raw_batch in loader:
        batch = move_batch(raw_batch, device)
        prediction = model_forward(model, batch).float() * stats["target_std"] + stats["target_mean"]
        for dish_id, target, value in zip(batch["dish_id"], batch["target"].cpu().tolist(), prediction.cpu().tolist()):
            rows.append({"dish_id": dish_id, "total_calories": target, "prediction": value})
    result = pd.DataFrame(rows)
    result["abs_error"] = (result["prediction"] - result["total_calories"]).abs()
    return result


def atomic_save(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def save_json(value: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def record_environment(run_dir: Path) -> None:
    metadata = {"python": platform.python_version(), "platform": platform.platform(),
                "torch": str(torch.__version__), "cuda_runtime": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    save_json(metadata, run_dir / "environment.json")
    result = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True)
    (run_dir / "requirements.lock.txt").write_text(result.stdout, encoding="utf-8")


def train(config: str | Path | dict | Config, resume: bool = False) -> dict:
    cfg = load_config(config)
    seed_everything(cfg.seed, cfg.deterministic)
    device = get_device(cfg.device)
    run_dir = cfg.path(cfg.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    last_path, best_path = run_dir / "last.pt", run_dir / "best.pt"
    if not resume and (last_path.exists() or best_path.exists()):
        raise FileExistsError(f"{run_dir} already contains a run. Use resume=True or a new run_dir.")
    if resume and not last_path.exists():
        raise FileNotFoundError(f"Cannot resume: {last_path} does not exist")
    frame, _, splits = prepare_data(cfg)
    fingerprint = dataset_fingerprint(frame)
    split_table = manifest(splits)
    state = torch.load(last_path, map_location="cpu", weights_only=True) if resume else None
    if state is not None:
        allowed_changes = {"epochs", "device", "num_workers", "log_every"}
        differences = [key for key, value in cfg.to_dict().items()
                       if key not in allowed_changes and state["config"].get(key) != value]
        if differences or fingerprint != state["dataset_fingerprint"]:
            raise ValueError(f"Resume dataset/configuration mismatch: {differences}")
        stored_manifest = pd.read_csv(run_dir / "split_manifest.csv")
        if not split_table.equals(stored_manifest):
            raise ValueError("Resume split manifest differs")
    else:
        split_table.to_csv(run_dir / "split_manifest.csv", index=False)
    stats = state["scalers"] if state else fit_scalers(splits["train"], cfg.use_mass)
    tokenizer_source = run_dir / "tokenizer" if state else cfg.text_model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, local_files_only=bool(state))
    tokenizer.save_pretrained(run_dir / "tokenizer")
    model = CalorieModel(cfg, text_config=state["text_config"] if state else None,
                         initialize_pretrained=False if state else None).to(device)
    if state:
        model.load_state_dict(state["model_state"])
    else:
        nn.init.normal_(model.regressor[-1].weight, mean=0, std=0.001)
        nn.init.constant_(model.regressor[-1].bias, (stats["target_median"] - stats["target_mean"]) / stats["target_std"])
    image_norm = state["image_norm"] if state else {
        "mean": list(model.image_encoder.pretrained_cfg["mean"]),
        "std": list(model.image_encoder.pretrained_cfg["std"])}
    loaders = {}
    for name in ("train", "val"):
        dataset = DishDataset(splits[name], tokenizer,
                              build_transform(cfg.image_size, name == "train", **image_norm),
                              cfg.max_length, stats["mass_mean"], stats["mass_std"], cfg.use_mass)
        loaders[name] = make_loader(dataset, cfg, name == "train")
    optimizer = build_optimizer(model, cfg)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and device.type == "cuda")
    best_mae, best_epoch, bad_epochs, start_epoch, history = float("inf"), 0, 0, 1, []
    if state:
        optimizer.load_state_dict(state["optimizer_state"])
        scheduler.load_state_dict(state["scheduler_state"])
        scaler.load_state_dict(state["amp_state"])
        best_mae, best_epoch, bad_epochs = state["best_mae"], state["best_epoch"], state["bad_epochs"]
        start_epoch, history = state["epoch"] + 1, state["history"]
    del state
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False), encoding="utf-8")
    record_environment(run_dir)
    print(f"Device: {device} | train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}", flush=True)
    print(f"Effective batch: {cfg.batch_size * cfg.accumulation_steps} | use_mass={cfg.use_mass}", flush=True)
    if device.type == "cuda":
        print(torch.cuda.get_device_name(device), flush=True)
    for epoch in range(start_epoch, cfg.epochs + 1):
        # Epoch-specific seeds allow an interrupted run to resume at an epoch boundary.
        seed_everything(cfg.seed + epoch, cfg.deterministic)
        loaders["train"].generator.manual_seed(cfg.seed + epoch)
        warmup = epoch <= cfg.warmup_epochs
        model.configure_trainable(warmup)
        assert_optimizer_covers_model(model, optimizer)
        count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Epoch {epoch}/{cfg.epochs} | {'head warm-up' if warmup else 'fine-tuning'} | trainable={count:,}", flush=True)
        started = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        train_mae = train_epoch(model, loaders["train"], optimizer, scaler, stats, cfg, device)
        validation = predict_loader(model, loaders["val"], stats, device)
        scores = regression_metrics(validation["total_calories"], validation["prediction"])
        if not warmup:
            scheduler.step(scores["mae"])
        history.append({"epoch": epoch, "train_mae": train_mae, "val_mae": scores["mae"],
                        "val_rmse": scores["rmse"], "seconds": time.perf_counter() - started,
                        "peak_gpu_gb": float(torch.cuda.max_memory_allocated(device) / 2**30) if device.type == "cuda" else 0.0,
                        **{f"lr_{group['name']}": group["lr"] for group in optimizer.param_groups}})
        improved = scores["mae"] < best_mae
        if improved:
            best_mae, best_epoch, bad_epochs = scores["mae"], epoch, 0
        elif not warmup:
            bad_epochs += 1
        payload = {"format_version": 1, "config": cfg.to_dict(), "model_state": model.state_dict(),
                   "text_config": model.text_encoder.config.to_dict(), "image_norm": image_norm,
                   "scalers": stats, "epoch": epoch, "best_epoch": best_epoch, "best_mae": best_mae,
                   "dataset_fingerprint": fingerprint}
        if improved:
            atomic_save(payload, best_path)
            validation.to_csv(run_dir / "val_predictions.csv", index=False)
            save_json(scores, run_dir / "val_metrics.json")
        atomic_save({**payload, "optimizer_state": optimizer.state_dict(),
                     "scheduler_state": scheduler.state_dict(), "amp_state": scaler.state_dict(),
                     "bad_epochs": bad_epochs, "history": history}, last_path)
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
        print(f"  train MAE={train_mae:.3f} | val MAE={scores['mae']:.3f} | "
              f"best={best_mae:.3f} | {history[-1]['seconds']:.1f}s", flush=True)
        if bad_epochs >= cfg.patience:
            print("Early stopping: no validation improvement", flush=True)
            break
    result = {"best_checkpoint": str(best_path), "best_val_mae": best_mae, "best_epoch": best_epoch}
    save_json(result, run_dir / "train_summary.json")
    del model, optimizer, scheduler, scaler, loaders
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/main.yaml")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    print(train(args.config, resume=args.resume))


if __name__ == "__main__":
    main()
