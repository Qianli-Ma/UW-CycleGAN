from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import nn

from models import UWCycleGANGenerator


def expand_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def resolve_split_file(split_dir: str | Path, split: str) -> Path:
    split_dir_path = expand_path(split_dir)
    split_file = split_dir_path / f"{split}.txt"
    if split_file.exists():
        return split_file
    fallback = split_dir_path.parent / f"{split}.txt"
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"Could not find split file '{split_file}' or '{fallback}'.")


class EvalWrapper(nn.Module):
    def __init__(self, generator: nn.Module) -> None:
        super().__init__()
        self.generator = generator

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * 2.0 - 1.0
        y = self.generator(x)
        return ((y + 1.0) * 0.5).clamp(0.0, 1.0)


def load_config(checkpoint: dict[str, Any], fallback_path: Path) -> dict[str, Any]:
    if "config" in checkpoint:
        return checkpoint["config"]
    with fallback_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_loader(config: dict[str, Any], split: str, device: torch.device) -> Any:
    src = expand_path("~/UIE/experiments/models/transfuse-gan/src")
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    local_datasets = (Path(__file__).resolve().parent / "datasets").resolve()
    loaded_datasets = sys.modules.get("datasets")
    loaded_file = getattr(loaded_datasets, "__file__", None)
    if loaded_file and Path(loaded_file).resolve().is_relative_to(local_datasets):
        for module_name in [name for name in sys.modules if name == "datasets" or name.startswith("datasets.")]:
            del sys.modules[module_name]
    from datasets.loaders import build_uieb_loader

    data = config["data"]
    split_dir = expand_path(data["split_dir"])
    if not (split_dir / f"{split}.txt").exists():
        split_dir = resolve_split_file(split_dir, split).parent
    return build_uieb_loader(
        input_dir=expand_path(data["raw_dir"]),
        target_dir=expand_path(data["ref_dir"]),
        split_dir=split_dir,
        image_size=int(data["image_size"]),
        batch_size=1,
        num_workers=int(data["num_workers"]),
        split=split,
        shuffle=False,
        drop_last=False,
        pin_memory=device.type == "cuda",
    )


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


@torch.no_grad()
def inference_stats(model: nn.Module, dataloader: Any, device: torch.device) -> tuple[float, int]:
    model.eval()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    total_ms = 0.0
    images = 0
    for batch in dataloader:
        x = batch["input"].to(device, non_blocking=True)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        total_ms += (time.perf_counter() - start) * 1000.0
        images += x.shape[0]
    peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    return total_ms / max(1, images), peak


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate UW-CycleGAN on UIEB.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    checkpoint = torch.load(expand_path(args.checkpoint), map_location=device)
    config = load_config(checkpoint, expand_path(args.config))
    generator = UWCycleGANGenerator(int(config["model"]["base_channels"])).to(device)
    generator.load_state_dict(checkpoint["G"])
    model = EvalWrapper(generator).to(device)

    dataloader = build_loader(config, args.split, device)
    ms_per_image, peak_vram = inference_stats(model, dataloader, device)

    src = expand_path("~/UIE/experiments/models/transfuse-gan/src")
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from evaluation.benchmark import evaluate_paired_loader

    metrics = evaluate_paired_loader(model, dataloader, device)
    print(f"Parameters: {parameter_count(generator):,}")
    print(f"Inference: {ms_per_image:.3f} ms/image")
    print(f"Peak VRAM: {peak_vram / (1024 ** 2):.2f} MiB")
    for key in ["PSNR", "SSIM", "CIEDE2000", "chroma_ab", "UCIQE", "UIQM"]:
        print(f"{key}: {metrics[key]:.6f}")


if __name__ == "__main__":
    main()
