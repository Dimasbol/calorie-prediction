"""Read tables, resolve ingredient names, audit images and split official train."""

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from sklearn.model_selection import GroupShuffleSplit

from scripts.config import Config, load_config


def normalize_ingredient_id(value: object) -> int:
    text = str(value).strip()
    match = re.fullmatch(r"(?:ingr_)?(\d+)(?:\.0)?", text)
    if match is None:
        raise ValueError(f"Invalid ingredient id: {value!r}")
    return int(match.group(1))


def canonical_ingredients(names: list[str]) -> str:
    """Order has no nutritional meaning; use the same canonicalization at inference."""
    cleaned = {re.sub(r"\s+", " ", str(x)).strip().lower() for x in names}
    return ", ".join(sorted(cleaned - {""})) or "unknown ingredients"


def read_tables(data_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    data_dir = Path(data_dir)
    required = [data_dir / "dish.csv", data_dir / "ingredients.csv"]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path}. See START_HERE.md, data extraction.")
    dishes = pd.read_csv(required[0], dtype={"dish_id": str})
    ingredients = pd.read_csv(required[1])
    expected = {"dish_id", "total_calories", "total_mass", "ingredients", "split"}
    if not expected.issubset(dishes.columns):
        raise ValueError(f"dish.csv lacks columns: {expected - set(dishes.columns)}")
    if not {"id", "ingr"}.issubset(ingredients.columns):
        raise ValueError("ingredients.csv must have id and ingr columns")
    if dishes["dish_id"].isna().any() or dishes["dish_id"].duplicated().any():
        raise ValueError("dish_id must be present and unique; inspect the source table")
    dishes["dish_id"] = dishes["dish_id"].str.strip()
    if dishes["dish_id"].duplicated().any():
        raise ValueError("Duplicate dish_id after stripping whitespace")
    if not dishes["dish_id"].map(lambda x: bool(re.fullmatch(r"[\w.-]+", x)) and x not in {".", ".."}).all():
        raise ValueError("dish_id must be a single safe folder name")
    dishes["split"] = dishes["split"].astype(str).str.strip().str.lower()
    if set(dishes["split"]) != {"train", "test"}:
        raise ValueError(f"Expected train and test; received {sorted(dishes['split'].unique())}")
    dishes["total_calories"] = pd.to_numeric(dishes["total_calories"], errors="raise")
    y = dishes["total_calories"].to_numpy(dtype=float)
    if not np.isfinite(y).all() or (y < 0).any():
        raise ValueError("Calories contain missing, non-finite or negative values; do not silently drop them")
    dishes["total_mass"] = pd.to_numeric(dishes["total_mass"], errors="coerce")
    ingredients["normalized_id"] = ingredients["id"].map(normalize_ingredient_id)
    if ingredients["normalized_id"].duplicated().any() or ingredients["ingr"].isna().any():
        raise ValueError("Ingredient dictionary has duplicate ids or missing names")
    lookup = ingredients.set_index("normalized_id")["ingr"].astype(str).to_dict()
    ingredient_ids, names = [], []
    for row in dishes.itertuples():
        parts = [] if pd.isna(row.ingredients) else str(row.ingredients).split(";")
        ids = list(dict.fromkeys(normalize_ingredient_id(x) for x in parts if x.strip()))
        missing = [x for x in ids if x not in lookup]
        if missing:
            raise ValueError(f"Unknown ingredient ids in {row.dish_id}: {missing}")
        ingredient_ids.append(ids)
        names.append([lookup[x].strip().lower() for x in ids])
    dishes["ingredient_ids"] = ingredient_ids
    dishes["ingredient_names"] = names
    dishes["ingredients_text"] = [canonical_ingredients(x) for x in names]
    dishes["n_ingredients"] = [len(set(x)) for x in names]
    dishes["image_path"] = [str(data_dir / "images" / x / "rgb.png") for x in dishes["dish_id"]]
    return dishes, ingredients


def audit_images(frame: pd.DataFrame, cache_path: Path) -> pd.DataFrame:
    """Hash decoded RGB pixels, not PNG metadata; cache by path, size and mtime."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    records, fresh = [], {}
    for i, row in enumerate(frame.itertuples(), 1):
        path = Path(row.image_path)
        if not path.is_file():
            raise FileNotFoundError(f"Missing image: {path}")
        stat = path.stat()
        key = str(path.resolve())
        signature = [stat.st_size, stat.st_mtime_ns]
        item = cache.get(key)
        if item is None or item.get("signature") != signature:
            try:
                with Image.open(path) as source:
                    img = ImageOps.exif_transpose(source).convert("RGB")
                    img.load()
                    payload = f"{img.width}x{img.height}:".encode() + img.tobytes()
                    item = {"signature": signature, "width": img.width, "height": img.height,
                            "image_hash": hashlib.sha256(payload).hexdigest()}
            except Exception as exc:
                raise ValueError(f"Unreadable image {path}: {exc}") from exc
        fresh[key] = item
        records.append({k: item[k] for k in ("width", "height", "image_hash")})
        if i % 1000 == 0:
            print(f"Image audit: {i}/{len(frame)}", flush=True)
    cache_path.write_text(json.dumps(fresh), encoding="utf-8")
    return pd.concat([frame.reset_index(drop=True), pd.DataFrame(records)], axis=1)


def make_splits(frame: pd.DataFrame, cfg: Config) -> dict[str, pd.DataFrame]:
    """Never use official test targets to build train/validation splits."""
    train_pool = frame.loc[frame["split"].eq("train")].reset_index(drop=True)
    test = frame.loc[frame["split"].eq("test")].reset_index(drop=True)
    overlap = set(train_pool["image_hash"]) & set(test["image_hash"])
    if overlap:
        raise ValueError(f"Found {len(overlap)} identical photos across official train/test. "
                         "Review the audit before claiming independent test quality.")
    splitter = GroupShuffleSplit(n_splits=1, test_size=cfg.val_fraction, random_state=cfg.seed)
    tr_idx, va_idx = next(splitter.split(train_pool, groups=train_pool["image_hash"]))
    train_frame = train_pool.iloc[tr_idx].reset_index(drop=True)
    val = train_pool.iloc[va_idx].reset_index(drop=True)
    if not len(train_frame) or not len(val) or not len(test):
        raise ValueError("One of the required splits is empty")
    return {"train": train_frame, "val": val, "test": test}


def manifest(splits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return pd.concat([part[["dish_id", "image_hash"]].assign(subset=name)
                      for name, part in splits.items()], ignore_index=True)


def prepare_data(config: str | Path | dict | Config) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    cfg = load_config(config)
    frame, ingredients = read_tables(cfg.path(cfg.data_dir))
    frame = audit_images(frame, cfg.path("data/processed/image_audit_cache.json"))
    splits = make_splits(frame, cfg)
    if cfg.use_mass:
        mass = frame["total_mass"].to_numpy(dtype=float)
        if not np.isfinite(mass).all() or (mass <= 0).any():
            raise ValueError("The mass experiment requires finite positive mass for every dish")
    return frame, ingredients, splits


def dataset_fingerprint(frame: pd.DataFrame) -> str:
    columns = ["dish_id", "total_calories", "total_mass", "ingredients_text", "split", "image_hash"]
    content = frame[columns].sort_values("dish_id").to_csv(index=False, float_format="%.12g")
    return hashlib.sha256(content.encode("utf-8")).hexdigest()
