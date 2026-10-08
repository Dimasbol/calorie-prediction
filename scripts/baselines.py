"""Validation references: median and TF-IDF + Ridge (never fitted on test)."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from scripts.utils import regression_metrics


def evaluate_baselines(splits: dict, output_dir: str | Path) -> pd.DataFrame:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train, val = splits["train"], splits["val"]
    results = []
    median = float(train["total_calories"].median())
    results.append({"model": "train_median", **regression_metrics(val["total_calories"], np.full(len(val), median))})
    for alpha in (1.0, 10.0, 100.0):
        pipeline = Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
                             ("ridge", Ridge(alpha=alpha, solver="lsqr"))])
        pipeline.fit(train["ingredients_text"], train["total_calories"])
        prediction = pipeline.predict(val["ingredients_text"])
        results.append({"model": f"tfidf_ridge_alpha_{alpha:g}", **regression_metrics(val["total_calories"], prediction)})
    table = pd.DataFrame(results).sort_values("mae").reset_index(drop=True)
    table.to_csv(output_dir / "baseline_validation.csv", index=False)
    return table
