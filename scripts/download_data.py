"""Download the course archive or extract an already downloaded nutrition.zip."""

import argparse
import hashlib
import json
import shutil
import time
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

from scripts.config import ROOT


PUBLIC_LINK = "https://disk.yandex.ru/d/kz9g5msVqtahiw"
ARCHIVE_SHA256 = "73e2cf7bc9c9c454f0b47426e8f83597d26cab3bc050b17e97aa9ba57ddabe4d"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_dataset(archive: Path, destination: Path) -> None:
    if (destination / "dish.csv").exists():
        raise FileExistsError(f"{destination}/dish.csv already exists. Do not unpack twice.")
    checksum = sha256_file(archive)
    if checksum != ARCHIVE_SHA256:
        raise ValueError(f"Archive SHA256 differs from the inspected course dataset: {checksum}. "
                         "Check the downloaded file and the course dataset version.")
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zip_file:
        selected = []
        for item in zip_file.infolist():
            parts = PurePosixPath(item.filename).parts
            if item.is_dir() or not parts or parts[0] != "data":
                continue
            relative = Path(*parts[1:])
            if relative.as_posix() not in {"dish.csv", "ingredients.csv"} and relative.parts[0] != "images":
                continue
            target = (destination / relative).resolve()
            if not target.is_relative_to(destination.resolve()):
                raise ValueError("Unsafe path in archive")
            selected.append((item, target))
        if not selected:
            raise ValueError("Expected data/dish.csv, data/ingredients.csv and data/images in archive")
        for index, (item, target) in enumerate(selected, 1):
            target.parent.mkdir(parents=True, exist_ok=True)
            with zip_file.open(item) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)
            if index % 1000 == 0:
                print(f"Extracted {index}/{len(selected)} files", flush=True)
    print(f"Dataset extracted to {destination}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, help="Existing nutrition.zip; skips downloading")
    args = parser.parse_args()
    archive = args.archive or ROOT / "downloads" / "nutrition.zip"
    if args.archive is None and not archive.exists():
        query = urllib.parse.urlencode({"public_key": PUBLIC_LINK})
        url = "https://cloud-api.yandex.net/v1/disk/public/resources/download?" + query
        with urllib.request.urlopen(url, timeout=60) as response:
            download_url = json.load(response)["href"]
        archive.parent.mkdir(parents=True, exist_ok=True)
        partial = archive.with_suffix(".zip.part")
        started, last, total = time.monotonic(), time.monotonic(), 0
        with urllib.request.urlopen(download_url, timeout=120) as source, partial.open("wb") as output:
            while chunk := source.read(8 * 1024 * 1024):
                output.write(chunk)
                total += len(chunk)
                if time.monotonic() - last > 10:
                    print(f"Downloaded {total / 1e6:.0f} MB; {time.monotonic() - started:.0f}s", flush=True)
                    last = time.monotonic()
        partial.replace(archive)
    extract_dataset(archive, ROOT / "data")


if __name__ == "__main__":
    main()
