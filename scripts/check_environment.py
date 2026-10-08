"""Run before opening the notebook: python -m scripts.check_environment."""

import importlib.metadata
import platform
import sys

import torch


def main():
    print("Python:", sys.version)
    print("Executable:", sys.executable)
    print("OS:", platform.platform())
    for package in ["torch", "torchvision", "timm", "transformers", "numpy", "pandas"]:
        print(f"{package}: {importlib.metadata.version(package)}")
    print("PyTorch CUDA runtime:", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable. Read START_HERE.md before starting training.")
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        print(f"GPU {index}: {properties.name}; VRAM {properties.total_memory / 2**30:.1f} GiB")
    matrix = torch.randn(256, 256, device="cuda")
    result = matrix @ matrix.T
    torch.cuda.synchronize()
    assert torch.isfinite(result).all()
    print("CUDA tensor calculation: OK")


if __name__ == "__main__":
    main()
