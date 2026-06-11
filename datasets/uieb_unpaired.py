from __future__ import annotations

import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from utils.blur import gaussian_blur


def _load_split(split_file: Path) -> list[str]:
    with split_file.open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


class UIEBUnpaired(Dataset):
    def __init__(
        self,
        raw_dir: str | Path,
        ref_dir: str | Path,
        split_file: str | Path,
        image_size: int = 256,
        blur_sigma: float = 3.0,
        blur_kernel: int = 21,
    ) -> None:
        self.raw_dir = Path(raw_dir).expanduser()
        self.ref_dir = Path(ref_dir).expanduser()
        self.split_file = Path(split_file).expanduser()
        self.image_size = image_size
        self.blur_sigma = blur_sigma
        self.blur_kernel = blur_kernel
        self.files = _load_split(self.split_file)
        self.epoch = 0
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        if not self.files:
            raise RuntimeError(f"No filenames found in split file '{self.split_file}'.")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.files)

    def _open_image(self, path: Path) -> Image.Image:
        if not path.exists():
            raise FileNotFoundError(f"Image file '{path}' does not exist.")
        with Image.open(path) as image:
            return image.convert("RGB")

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        x_name = self.files[idx]
        rng = random.Random(self.epoch * len(self.files) + idx)
        y_name = self.files[rng.randrange(len(self.files))]

        x = self.transform(self._open_image(self.raw_dir / x_name))
        y = self.transform(self._open_image(self.ref_dir / y_name))
        z = gaussian_blur(y.unsqueeze(0), self.blur_kernel, self.blur_sigma).squeeze(0)
        return {"x": x, "y": y, "z": z}


class UIEBPairedEval(Dataset):
    def __init__(
        self,
        raw_dir: str | Path,
        ref_dir: str | Path,
        split_file: str | Path,
        image_size: int = 256,
    ) -> None:
        self.raw_dir = Path(raw_dir).expanduser()
        self.ref_dir = Path(ref_dir).expanduser()
        self.split_file = Path(split_file).expanduser()
        self.files = _load_split(self.split_file)
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ]
        )

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        filename = self.files[idx]
        with Image.open(self.raw_dir / filename) as input_image:
            x = self.transform(input_image.convert("RGB"))
        with Image.open(self.ref_dir / filename) as target_image:
            y = self.transform(target_image.convert("RGB"))
        return {"input": x, "target": y}
