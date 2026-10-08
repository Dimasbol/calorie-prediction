После локального обучения появится `main/`:

- `best.pt` — веса лучшей модели по validation MAE и параметры предобработки.
- `last.pt` — модель, оптимизатор, scheduler и AMP для продолжения обучения.
- `tokenizer/` — локальные файлы токенизатора, обязательны для инференса.
- `history.csv`, `val_metrics.json`, `config.yaml`, `split_manifest.csv`.
- `environment.json`, `requirements.lock.txt` — фактическое окружение запуска.

После тестирования создаются `test_metrics.json`, `test_predictions.csv`.
Для передачи модели выполните `python -m scripts.export_model --run-dir artifacts/main`.
Передайте полученный `main_model.zip` отдельной ссылкой или вложением GitHub Release.
Большие веса не нужно добавлять обычным Git-коммитом.

Обученный checkpoint пока отсутствует. Он сохраняется после запуска обучения.
