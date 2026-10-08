"""Actual data exploration; all numbers and plots are computed from the dataset."""

import json
from collections import Counter
from pathlib import Path
from textwrap import fill

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


def plot_examples(frame: pd.DataFrame, path: Path, seed: int = 42, n: int = 6):
    sample = frame.sample(min(n, len(frame)), random_state=seed)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), squeeze=False)
    for axis in axes.flat:
        axis.axis("off")
    for axis, row in zip(axes.flat, sample.itertuples()):
        with Image.open(row.image_path) as image:
            axis.imshow(image.convert("RGB"))
        axis.set_title(f"{row.dish_id}\n{row.total_calories:.1f} kcal | {row.total_mass:.0f} g\n"
                       + fill(row.ingredients_text, width=48), fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def run_eda(frame: pd.DataFrame, ingredients: pd.DataFrame, splits: dict,
            output_dir: str | Path = "reports/eda") -> dict:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    train = splits["train"]
    split_stats = frame.groupby("split").agg(n=("dish_id", "size"),
                                             calories_mean=("total_calories", "mean"),
                                             calories_median=("total_calories", "median"),
                                             mass_median=("total_mass", "median"))
    split_stats.to_csv(out / "split_summary.csv")
    summary = {"n_dishes": len(frame), "n_ingredient_dictionary": len(ingredients),
               "official_splits": frame["split"].value_counts().to_dict(),
               "training_splits": {name: len(part) for name, part in splits.items()},
               "missing_source_values": frame[["total_calories", "total_mass", "ingredients"]].isna().sum().to_dict(),
               "duplicate_images": int(frame["image_hash"].duplicated().sum()),
               "zero_calorie_dishes": int(frame["total_calories"].eq(0).sum()),
               "calories": frame["total_calories"].describe().to_dict(),
               "ingredient_count": frame["n_ingredients"].describe().to_dict(),
               "train_mass_calorie_pearson": float(train[["total_calories", "total_mass"]].corr().iloc[0, 1]),
               "image_sizes": frame.groupby(["width", "height"]).size().rename("n").reset_index().to_dict("records")}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].hist(train["total_calories"], bins=45, color="#2878ad")
    axes[0].set(xlabel="Calories, kcal", ylabel="Dishes", title="Training target: full range")
    axes[1].hist(np.log1p(train["total_calories"]), bins=35, color="#268779")
    axes[1].set(xlabel="log(1 + calories)", title="Training target: log scale for EDA")
    axes[2].scatter(train["total_mass"], train["total_calories"], s=10, alpha=0.35, color="#b35b30")
    axes[2].set(xlabel="Mass, g", ylabel="Calories, kcal", title="Training mass vs calories")
    fig.tight_layout()
    fig.savefig(out / "distributions.png", dpi=150)
    plt.close(fig)
    counts = Counter(name for names in train["ingredient_names"] for name in set(names))
    top = pd.Series(dict(counts.most_common(20))).sort_values()
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    top.plot.barh(ax=axes[0], color="#2878ad")
    axes[0].set(xlabel="Training dishes containing the ingredient", title="Most frequent ingredients")
    axes[1].hist(train["n_ingredients"], bins=np.arange(train["n_ingredients"].max() + 2) - 0.5,
                 color="#268779")
    axes[1].set(xlabel="Ingredient count", ylabel="Dishes", title="Ingredients per training dish")
    fig.tight_layout()
    fig.savefig(out / "ingredients.png", dpi=140)
    plt.close(fig)
    plot_examples(train, out / "sample_dishes.png")
    frame.loc[frame["total_calories"].eq(0), ["dish_id", "ingredients_text", "total_mass", "split"]].to_csv(
        out / "zero_calorie_dishes.csv", index=False)
    frame.nlargest(10, "total_calories")[["dish_id", "total_calories", "total_mass", "split", "ingredients_text"]].to_csv(
        out / "largest_targets.csv", index=False)
    pd.DataFrame(counts.most_common(), columns=["ingredient", "train_dish_count"]).to_csv(
        out / "ingredient_frequency.csv", index=False)
    rare = sum(value <= 5 for value in counts.values())
    report = f"""# Исследовательский анализ данных

В данных {len(frame)} блюд и {len(ingredients)} записей справочника ингредиентов.
Официальный train: {summary['official_splits']['train']}, test: {summary['official_splits']['test']}.
Внутри официального train: обучение — {len(train)}, валидация — {len(splits['val'])}.
Разбиение фиксировано seed=42; одинаковые фотографии объединяются в одну группу.

Медиана калорийности по исходному набору: {frame.total_calories.median():.2f} ккал;
минимум — {frame.total_calories.min():.2f}, максимум — {frame.total_calories.max():.2f} ккал.
Блюд с нулевой целевой калорийностью: {summary['zero_calorie_dishes']}.
Точных повторов фотографий: {summary['duplicate_images']}.
Все указанные в таблице изображения прочитаны; все ID ингредиентов разрешены в названия.

В обучающей части коэффициент корреляции массы с калорийностью равен
{summary['train_mass_calorie_pearson']:.3f}. Корреляция сама по себе не доказывает причинность
и не позволяет восстановить калорийность без знания состава. Масса не поступает в основную модель.
Ингредиентов, встречающихся не более чем в пяти обучающих блюдах: {rare}.

## Принятые решения

1. Решается регрессия общей калорийности порции, а не ккал на 100 г.
2. ID ингредиентов заменены именами. Текст нормализован и отсортирован; справочник не содержит target.
3. Используется перенос обучения: ResNet18 для изображения и DistilBERT для английских названий.
4. Умеренные изменения цвета, отражение и повороты на 90° сохраняют состав блюда.
   Агрессивные обрезки и удаление участков изображения исключены: калорийность оставшейся части неизвестна.
5. Главный критерий — MAE в ккал. Дополнительно измеряются RMSE, R², медиана и 90-й перцентиль ошибки.
6. Нулевые значения и большие порции не удаляются автоматически. Их списки сохранены для проверки.
7. Все решения по модели и остановке обучения принимаются по validation; test не участвует в подборе.
8. Точные RGB-дубликаты проверены. Похожие ракурсы, повторные рецепты и зависимые сессии съемки
   этим тестом не исключаются: в задании нет отдельного идентификатора сессии/рецепта для группировки.

## Следующий этап

На основе анализа выбран подход с объединением признаков фотографии и текста.
Его качество будет оцениваться после обучения по MAE на официальном test.
"""
    (out / "EDA_REPORT.md").write_text(report, encoding="utf-8")
    return summary
