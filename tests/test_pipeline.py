"""Offline engineering checks; synthetic data are not project quality evidence."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image
from transformers import BertTokenizerFast, DistilBertConfig

from scripts.config import Config
from scripts.data import make_splits, normalize_ingredient_id, prepare_data
from scripts.dataset import DishDataset, build_transform, make_loader
from scripts.evaluate import load_predictor, predict_one
import scripts.evaluate as evaluation_module
from scripts.export_model import export_model
from scripts.model import CalorieModel
from scripts.utils import (assert_optimizer_covers_model, build_optimizer, finish_accumulation,
                           fit_scalers, predict_loader, regression_metrics, train)


@pytest.fixture()
def tiny_project(tmp_path):
    torch.set_num_threads(2)
    data = tmp_path / "data"
    data.mkdir()
    pd.DataFrame({"id": [1, 2, 3], "ingr": ["rice", "salmon", "broccoli"]}).to_csv(data / "ingredients.csv", index=False)
    rng = np.random.default_rng(42)
    rows = []
    for index in range(23):
        dish_id = f"dish_{index:03d}"
        folder = data / "images" / dish_id
        folder.mkdir(parents=True)
        pixels = rng.integers(0, 256, size=(48, 64, 3), dtype=np.uint8)
        Image.fromarray(pixels).save(folder / "rgb.png")
        rows.append({"dish_id": dish_id, "total_calories": 100 + index * 7,
                     "total_mass": 50 + index * 2, "ingredients": "ingr_0000000001;ingr_0000000002",
                     "split": "train" if index < 18 else "test"})
    pd.DataFrame(rows).to_csv(data / "dish.csv", index=False)
    tokenizer_path = tmp_path / "tiny_text"
    tokenizer_path.mkdir()
    vocabulary = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "rice", "salmon", "broccoli", ","]
    (tokenizer_path / "vocab.txt").write_text("\n".join(vocabulary), encoding="utf-8")
    tokenizer = BertTokenizerFast(vocab_file=str(tokenizer_path / "vocab.txt"), do_lower_case=True)
    tokenizer.save_pretrained(tokenizer_path)
    DistilBertConfig(vocab_size=len(tokenizer), dim=32, hidden_dim=64, n_layers=2,
                     n_heads=4, max_position_embeddings=64, dropout=0.0, attention_dropout=0.0).save_pretrained(tokenizer_path)
    cfg = Config(data_dir=str(data), run_dir=str(tmp_path / "run"), device="cpu", amp=False,
                 pretrained=False, text_model=str(tokenizer_path), image_size=64,
                 max_length=16, projection_dim=16, dropout=0.0, image_trainable="last_stage",
                 text_trainable_layers=1, epochs=2, warmup_epochs=1, batch_size=4,
                 accumulation_steps=3, log_every=0)
    return cfg


def test_id_normalization():
    assert normalize_ingredient_id("ingr_0000000122") == 122
    assert normalize_ingredient_id("122") == 122
    assert normalize_ingredient_id(122.0) == 122
    with pytest.raises(ValueError):
        normalize_ingredient_id("bad_id")


def test_group_split_keeps_duplicate_photos_together():
    frame = pd.DataFrame({"dish_id": [str(i) for i in range(12)],
                          "split": ["train"] * 10 + ["test"] * 2,
                          "image_hash": ["duplicate", "duplicate"] + [str(i) for i in range(2, 12)]})
    splits = make_splits(frame, Config())
    assert not set(splits["train"].image_hash) & set(splits["val"].image_hash)
    frame.loc[11, "image_hash"] = "duplicate"
    with pytest.raises(ValueError, match="official train/test"):
        make_splits(frame, Config())


def test_metrics_weight_samples_instead_of_batches():
    scores = regression_metrics([0, 0, 0, 0, 0], [0, 0, 0, 0, 10])
    assert scores["mae"] == 2.0
    assert scores["r2"] is None


def test_accumulation_normalizes_an_incomplete_group():
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    # Microbatches of lengths 2 and 1 must equal a single sample-weighted batch.
    for values in ([1.0, 2.0], [3.0]):
        output = model(torch.tensor(values).view(-1, 1))
        scaler.scale(output.sum()).backward()
    finish_accumulation(model, optimizer, scaler, sample_count=3, grad_clip=100)
    assert model.weight.item() == pytest.approx(0.8)


def test_all_parameters_in_optimizer(tiny_project):
    model = CalorieModel(tiny_project)
    model.configure_trainable(warmup=True)
    optimizer = build_optimizer(model, tiny_project)
    model.configure_trainable(warmup=False)
    assert_optimizer_covers_model(model, optimizer)
    optimizer.param_groups[-1]["params"] = []
    with pytest.raises(AssertionError):
        assert_optimizer_covers_model(model, optimizer)


def test_train_checkpoint_reload_and_resume(tiny_project, monkeypatch, tmp_path):
    cfg = tiny_project
    result = train(cfg)
    assert Path(result["best_checkpoint"]).is_file()
    history = pd.read_csv(cfg.path(cfg.run_dir) / "history.csv")
    assert len(history) == 2
    predictor = load_predictor(result["best_checkpoint"], device="cpu")
    _, _, splits = prepare_data(cfg)
    assert predictor["state"]["scalers"] == fit_scalers(splits["train"], False)
    test = splits["test"]
    dataset = DishDataset(test, predictor["tokenizer"],
                          build_transform(cfg.image_size, False, **predictor["state"]["image_norm"]), cfg.max_length)
    predicted = predict_loader(predictor["model"], make_loader(dataset, cfg, False),
                               predictor["state"]["scalers"], torch.device("cpu"))
    assert predicted.dish_id.tolist() == test.dish_id.tolist()
    assert np.isfinite(predicted.prediction).all()
    row = test.iloc[0]
    single = predict_one(predictor, row.image_path, row.ingredient_names)
    assert single == pytest.approx(predicted.prediction.iloc[0], abs=0.002)
    monkeypatch.setattr(evaluation_module, "ROOT", tmp_path)
    evaluation = evaluation_module.evaluate_checkpoint(result["best_checkpoint"], device="cpu")
    assert evaluation["metrics"]["n"] == len(test)
    assert (Path(evaluation["report_dir"]) / "top5_errors.png").is_file()
    archive = export_model(cfg.run_dir)
    assert archive.is_file()
    with pytest.raises(FileExistsError):
        train(cfg)
    cfg.epochs = 3
    resumed = train(cfg, resume=True)
    assert Path(resumed["best_checkpoint"]).is_file()
    assert len(pd.read_csv(cfg.path(cfg.run_dir) / "history.csv")) == 3


def test_mass_experiment_requires_mass(tiny_project):
    cfg = tiny_project
    cfg.use_mass = True
    model = CalorieModel(cfg).eval()
    assert_optimizer_covers_model(model, build_optimizer(model, cfg))
    image = torch.zeros(1, 3, 64, 64)
    ids = torch.ones(1, 8, dtype=torch.long)
    mask = torch.ones_like(ids)
    with torch.inference_mode():
        with pytest.raises(ValueError, match="requires portion mass"):
            model(image, ids, mask)
        assert model(image, ids, mask, torch.zeros(1, 1)).shape == (1,)
