"""Package inference weights and tokenizer, leaving training optimizer state out."""

import argparse
import json
import zipfile
from pathlib import Path

from scripts.config import ROOT
from scripts.download_data import sha256_file


def export_model(run_dir: str | Path) -> Path:
    path = Path(run_dir)
    if not path.is_absolute():
        path = ROOT / path
    required = [path / "best.pt", path / "tokenizer", path / "config.yaml"]
    for item in required:
        if not item.exists():
            raise FileNotFoundError(f"Missing inference artifact: {item}")
    files = [path / name for name in ("best.pt", "config.yaml", "environment.json", "requirements.lock.txt",
                                      "val_metrics.json", "test_metrics.json", "split_manifest.csv",
                                      "history.csv", "train_summary.json")
             if (path / name).is_file()]
    files.extend(sorted((path / "tokenizer").rglob("*")))
    files = [item for item in files if item.is_file()]
    checksums = {item.relative_to(path).as_posix(): sha256_file(item) for item in files}
    checksum_path = path / "SHA256SUMS.json"
    checksum_path.write_text(json.dumps(checksums, indent=2), encoding="utf-8")
    archive = path.parent / f"{path.name}_model.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=3) as bundle:
        for item in files + [checksum_path]:
            bundle.write(item, Path(path.name) / item.relative_to(path))
    print(f"Model bundle: {archive}\nSHA256: {sha256_file(archive)}")
    return archive


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default="artifacts/main")
    args = parser.parse_args()
    export_model(args.run_dir)


if __name__ == "__main__":
    main()
