from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
import yaml
import time

from trainer import UWCycleGANTrainer, expand_path


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    output_dir = Path(config["paths"]["output_dir"]).expanduser()
    if not output_dir.is_absolute():
        output_dir = path.parent / output_dir
    config["paths"]["output_dir"] = str(output_dir)
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train UW-CycleGAN on UIEB.")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = expand_path(args.config)
    config = load_config(config_path)
    device = torch.device(args.device)
    trainer = UWCycleGANTrainer(config, device)
    trainer.train()


if __name__ == "__main__":
    start_time = time.perf_counter()
    main()
    end_time = time.perf_counter()
    total_seconds = end_time - start_time
    
    minutes = total_seconds // 60
    seconds = total_seconds % 60
    print(f"Training took {int(minutes)}m {seconds:.1f}s")

