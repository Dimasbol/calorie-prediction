"""Load a portable model, evaluate official test and inspect the five largest errors."""

import argparse
import json
from collections import Counter
from pathlib import Path
from textwrap import fill

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from transformers import AutoTokenizer

from scripts.config import ROOT, load_config
from scripts.data import canonical_ingredients, dataset_fingerprint, prepare_data
from scripts.dataset import DishDataset, build_transform, make_loader
from scripts.download_data import sha256_file
from scripts.model import CalorieModel
from scripts.utils import get_device, predict_loader, regression_metrics, save_json


def load_predictor(checkpoint: str | Path, device: str = "cuda") -> dict:
    path = Path(checkpoint)
    if not path.is_absolute():
        path = ROOT / path
    state = torch.load(path, map_location="cpu", weights_only=True)
    cfg = load_config(state["config"])
    cfg.device = device
    model = CalorieModel(cfg, text_config=state["text_config"], initialize_pretrained=False)
    model.load_state_dict(state["model_state"])
    model.to(get_device(device)).eval()
    tokenizer = AutoTokenizer.from_pretrained(path.parent / "tokenizer", local_files_only=True)
    return {"model": model, "tokenizer": tokenizer, "config": cfg, "state": state,
            "checkpoint": path, "device": get_device(device)}


@torch.inference_mode()
def predict_one(predictor: dict, image_path: str | Path, ingredient_names: list[str],
                mass_g: float | None = None) -> float:
    cfg, state = predictor["config"], predictor["state"]
    if not ingredient_names or any(not str(name).strip() for name in ingredient_names):
        raise ValueError("Pass a nonempty list of ingredient names")
    transform = build_transform(cfg.image_size, False, **state["image_norm"])
    with Image.open(image_path) as source:
        image = transform(ImageOps.exif_transpose(source).convert("RGB")).unsqueeze(0).to(predictor["device"])
    tokens = predictor["tokenizer"]([canonical_ingredients(ingredient_names)], padding="max_length",
                                    truncation=True, max_length=cfg.max_length, return_tensors="pt")
    stats = state["scalers"]
    mass = torch.zeros((1, 1), device=predictor["device"])
    if cfg.use_mass:
        if mass_g is None or not np.isfinite(mass_g) or mass_g <= 0:
            raise ValueError("This checkpoint requires a positive portion mass in grams")
        mass[0, 0] = (np.log1p(mass_g) - stats["mass_mean"]) / stats["mass_std"]
    prediction = predictor["model"](image, tokens["input_ids"].to(predictor["device"]),
                                     tokens["attention_mask"].to(predictor["device"]), mass)
    return float(prediction.item() * stats["target_std"] + stats["target_mean"])


def plot_top_errors(predictions: pd.DataFrame, output: Path, n: int = 5) -> None:
    top = predictions.nlargest(n, "abs_error")
    fig, axes = plt.subplots(len(top), 2, figsize=(13, 3.8 * len(top)),
                             gridspec_kw={"width_ratios": [1, 1.7]}, squeeze=False)
    for (photo_axis, text_axis), row in zip(axes, top.itertuples()):
        with Image.open(row.image_path) as source:
            photo_axis.imshow(source.convert("RGB"))
        photo_axis.axis("off")
        text_axis.axis("off")
        text = (f"{row.dish_id}\nActual: {row.total_calories:.1f} kcal\nPredicted: {row.prediction:.1f} kcal\n"
                f"Absolute error: {row.abs_error:.1f} kcal\nMass (for analysis): {row.total_mass:.0f} g\n\n"
                + fill(row.ingredients_text, 72))
        text_axis.text(0, 0.95, text, va="top", fontsize=11)
    fig.tight_layout()
    fig.savefig(output, dpi=130, bbox_inches="tight")
    plt.close(fig)


def write_error_analysis(predictions: pd.DataFrame, train_frame: pd.DataFrame, cfg, output: Path):
    counts = Counter(name for names in train_frame["ingredient_names"] for name in set(names))
    mass95 = float(train_frame["total_mass"].quantile(0.95))
    target95 = float(train_frame["total_calories"].quantile(0.95))
    rows = []
    for row in predictions.nlargest(5, "abs_error").itertuples():
        rare = [name for name in row.ingredient_names if counts[name] <= 5]
        facts = [f"Ошибка {row.abs_error:.1f} ккал; масса {row.total_mass:.0f} г; ингредиентов {row.n_ingredients}."]
        hypotheses = []
        if row.total_mass > mass95:
            facts.append(f"Масса выше 95-го перцентиля train ({mass95:.0f} г).")
            hypotheses.append("Нетипичный размер порции может осложнять оценку по одному ракурсу.")
        if row.total_calories > target95:
            facts.append(f"Калорийность выше 95-го перцентиля train ({target95:.1f} ккал).")
            hypotheses.append("Модель могла недообучиться на редких высококалорийных порциях.")
        if rare:
            facts.append("Редкие или отсутствующие в train ингредиенты: " + ", ".join(rare) + ".")
            hypotheses.append("Мало наблюдений с таким составом могло затруднить обобщение.")
        if not cfg.use_mass:
            hypotheses.append("Модель не знает массу порции и граммовки отдельных ингредиентов.")
        rows.append(f"## {row.dish_id}\n\n" + " ".join(facts) + "\n\n"
                    + f"Истина: {row.total_calories:.1f}; прогноз: {row.prediction:.1f} ккал.\n\n"
                    + "Ингредиенты: " + row.ingredients_text + ".\n\n"
                    + "Гипотезы: " + " ".join(hypotheses) + "\n\n"
                    + "Визуальные наблюдения: пока не добавлены.\n")
    output.write_text("# Пять крупнейших ошибок\n\nФакты вычислены из данных; причины ниже — гипотезы.\n\n"
                      + "\n".join(rows), encoding="utf-8")


def evaluate_checkpoint(checkpoint: str | Path, device: str = "cuda", data_dir: str | None = None) -> dict:
    predictor = load_predictor(checkpoint, device)
    cfg, state = predictor["config"], predictor["state"]
    if data_dir is not None:
        cfg.data_dir = data_dir
    frame, _, splits = prepare_data(cfg)
    if dataset_fingerprint(frame) != state["dataset_fingerprint"]:
        raise ValueError("Dataset content differs from the dataset recorded during training")
    stats = state["scalers"]
    dataset = DishDataset(splits["test"], predictor["tokenizer"],
                          build_transform(cfg.image_size, False, **state["image_norm"]), cfg.max_length,
                          stats["mass_mean"], stats["mass_std"], cfg.use_mass)
    prediction = predict_loader(predictor["model"], make_loader(dataset, cfg, False), stats, predictor["device"])
    details = splits["test"].drop(columns="total_calories")
    prediction = prediction.merge(details, on="dish_id", how="left", validate="one_to_one")
    scores = regression_metrics(prediction["total_calories"], prediction["prediction"])
    checkpoint_hash = sha256_file(predictor["checkpoint"])
    scores.update({"target_met": bool(scores["mae"] < 50), "use_mass": cfg.use_mass,
                   "best_epoch": state["best_epoch"], "checkpoint_sha256": checkpoint_hash,
                   "checkpoint_selection": "minimum validation MAE", "subset": "official test"})
    run_dir = predictor["checkpoint"].parent
    output_dir = ROOT / "reports" / run_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)
    for destination in (run_dir, output_dir):
        save_json(scores, destination / "test_metrics.json")
        portable = prediction.drop(columns=["image_path", "ingredient_ids", "ingredient_names"])
        portable.to_csv(destination / "test_predictions.csv", index=False)
    plot_top_errors(prediction, output_dir / "top5_errors.png")
    write_error_analysis(prediction, splits["train"], cfg, output_dir / "ERROR_ANALYSIS.md")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].scatter(prediction["total_calories"], prediction["prediction"], s=12, alpha=0.5)
    bound = max(prediction["total_calories"].max(), prediction["prediction"].max(), 1)
    axes[0].plot([0, bound], [0, bound], "k--")
    axes[0].set(xlabel="Actual calories", ylabel="Predicted calories", title="Official test")
    axes[1].hist(prediction["prediction"] - prediction["total_calories"], bins=35)
    axes[1].set(xlabel="Prediction - target, kcal", title="Signed errors")
    fig.tight_layout()
    fig.savefig(output_dir / "test_diagnostics.png", dpi=150)
    plt.close(fig)
    status = "достигнут" if scores["target_met"] else "не достигнут"
    report = (f"# Финальная оценка\n\nВходы: фото + ингредиенты{' + масса' if cfg.use_mass else ''}.\n\n"
              f"Модель выбрана по validation, эпоха {state['best_epoch']}. Официальный test: {scores['n']} блюд.\n\n"
              f"MAE = **{scores['mae']:.3f} ккал**, RMSE = {scores['rmse']:.3f} ккал.\n\n"
              f"Порог MAE < 50 ккал **{status}**.\n\n"
              "Крупнейшие ошибки приведены в ERROR_ANALYSIS.md и на рисунке top5_errors.png.\n\n"
              "Test использован после выбора модели по validation.\n")
    (output_dir / "FINAL_REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps(scores, ensure_ascii=False, indent=2), flush=True)
    del predictor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"metrics": scores, "predictions": prediction, "report_dir": str(output_dir)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="artifacts/main/best.pt")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    evaluate_checkpoint(args.checkpoint, device=args.device)


if __name__ == "__main__":
    main()
