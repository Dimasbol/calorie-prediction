"""Images and pre-tokenized ingredient names, with portion-preserving transforms."""

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF

from scripts.config import Config


class Letterbox:
    """Fit the complete photo into a square without cropping or distortion."""

    def __init__(self, size: int):
        self.size = size

    def __call__(self, image: Image.Image) -> Image.Image:
        image = ImageOps.contain(image, (self.size, self.size), Image.Resampling.BILINEAR)
        canvas = Image.new("RGB", (self.size, self.size), (124, 116, 104))
        canvas.paste(image, ((self.size - image.width) // 2, (self.size - image.height) // 2))
        return canvas


class RandomQuarterTurn:
    def __call__(self, image: Image.Image) -> Image.Image:
        return TF.rotate(image, 90 * random.randrange(4))


def build_transform(size: int, training: bool, mean: list | tuple, std: list | tuple):
    steps = [Letterbox(size)]
    if training:
        steps.extend([transforms.RandomHorizontalFlip(), RandomQuarterTurn(),
                      transforms.ColorJitter(brightness=0.10, contrast=0.10, saturation=0.08)])
    steps.extend([transforms.ToTensor(), transforms.Normalize(mean, std)])
    return transforms.Compose(steps)


class DishDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, tokenizer, transform, max_length: int,
                 mass_mean: float = 0.0, mass_std: float = 1.0, use_mass: bool = False):
        self.frame = frame.reset_index(drop=True).copy()
        self.transform = transform
        self.tokens = tokenizer(self.frame["ingredients_text"].tolist(), padding="max_length",
                                truncation=True, max_length=max_length, return_tensors="pt")
        self.labels = self.frame["total_calories"].to_numpy(dtype=np.float32)
        self.mass = np.zeros((len(self.frame), 1), dtype=np.float32)
        if use_mass:
            values = self.frame["total_mass"].to_numpy(dtype=np.float32)
            if not np.isfinite(values).all() or (values <= 0).any():
                raise ValueError("Missing or invalid portion mass")
            self.mass[:, 0] = (np.log1p(values) - mass_mean) / mass_std

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        # data/images/<dish_id>/rgb.png is resolved in scripts/data.py.
        with Image.open(Path(row["image_path"])) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            image = self.transform(image)
        return {"image": image, "input_ids": self.tokens["input_ids"][index],
                "attention_mask": self.tokens["attention_mask"][index],
                "mass": torch.from_numpy(self.mass[index]),
                "target": torch.tensor(self.labels[index]), "dish_id": row["dish_id"]}


def seed_worker(worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def make_loader(dataset: Dataset, cfg: Config, training: bool) -> DataLoader:
    generator = torch.Generator().manual_seed(cfg.seed)
    return DataLoader(dataset, batch_size=cfg.batch_size, shuffle=training,
                      num_workers=cfg.num_workers, pin_memory=cfg.device.startswith("cuda"),
                      worker_init_fn=seed_worker, generator=generator,
                      persistent_workers=False, drop_last=False)
