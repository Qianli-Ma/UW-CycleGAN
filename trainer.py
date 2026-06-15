from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image

from datasets.uieb_unpaired import UIEBPairedEval, UIEBUnpaired
from losses import (
    content_loss,
    cycle_loss,
    discriminator_adv_loss,
    discriminator_badv_loss,
    generator_adv_loss,
)
from models import UWCycleGANDiscriminator, UWCycleGANGenerator, VGG19Conv44


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


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def linear_lr(epoch: int, total_epochs: int, decay_start: int) -> float:
    if epoch <= decay_start:
        return 1.0
    decay_epochs = max(1, total_epochs - decay_start)
    return max(0.0, 1.0 - (epoch - decay_start) / decay_epochs)


class EvalWrapper(nn.Module):
    def __init__(self, generator: nn.Module) -> None:
        super().__init__()
        self.generator = generator

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * 2.0 - 1.0
        y = self.generator(x)
        return ((y + 1.0) * 0.5).clamp(0.0, 1.0)


class UWCycleGANTrainer:
    def __init__(self, config: dict[str, Any], device: torch.device) -> None:
        self.config = config
        self.device = device
        seed_everything(int(config["train"]["seed"]))

        model_cfg = config["model"]
        base_channels = int(model_cfg["base_channels"])
        self.g = UWCycleGANGenerator(base_channels).to(device)
        self.f = UWCycleGANGenerator(base_channels).to(device)
        self.dx = UWCycleGANDiscriminator(base_channels).to(device)
        self.dy = UWCycleGANDiscriminator(base_channels).to(device)
        self.vgg = VGG19Conv44().to(device).eval()

        train_cfg = config["train"]
        self.opt_g = torch.optim.Adam(
            list(self.g.parameters()) + list(self.f.parameters()),
            lr=float(train_cfg["lr_g"]),
            betas=(float(train_cfg["beta1"]), float(train_cfg["beta2"])),
        )
        self.opt_d = torch.optim.Adam(
            list(self.dx.parameters()) + list(self.dy.parameters()),
            lr=float(train_cfg["lr_d"]),
            betas=(float(train_cfg["beta1"]), float(train_cfg["beta2"])),
        )
        self.scaler = GradScaler("cuda", enabled=bool(train_cfg["amp"]) and device.type == "cuda")

        self.output_dir = expand_path(config["paths"]["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(self.output_dir / "tensorboard")
        self.best_psnr = float("-inf")

        self.train_loader = self._build_train_loader()
        self.val_loader = self._build_eval_loader("val")
        self.evaluate_paired_loader = self._load_thesis_evaluator()

    def _build_train_loader(self) -> DataLoader:
        data_cfg = self.config["data"]
        model_cfg = self.config["model"]
        dataset = UIEBUnpaired(
            raw_dir=data_cfg["raw_dir"],
            ref_dir=data_cfg["ref_dir"],
            split_file=resolve_split_file(data_cfg["split_dir"], "train"),
            image_size=int(data_cfg["image_size"]),
            blur_sigma=float(model_cfg["blur_sigma"]),
            blur_kernel=int(model_cfg["blur_kernel"]),
        )
        return DataLoader(
            dataset,
            batch_size=int(self.config["train"]["batch_size"]),
            shuffle=True,
            num_workers=int(data_cfg["num_workers"]),
            pin_memory=self.device.type == "cuda",
            drop_last=True,
        )

    def _build_eval_loader(self, split: str) -> DataLoader:
        data_cfg = self.config["data"]
        dataset = UIEBPairedEval(
            raw_dir=data_cfg["raw_dir"],
            ref_dir=data_cfg["ref_dir"],
            split_file=resolve_split_file(data_cfg["split_dir"], split),
            image_size=int(data_cfg["image_size"]),
        )
        return DataLoader(
            dataset,
            batch_size=int(self.config["train"]["batch_size"]),
            shuffle=False,
            num_workers=int(data_cfg["num_workers"]),
            pin_memory=self.device.type == "cuda",
            drop_last=False,
        )

    def _load_thesis_evaluator(self) -> Any:
        src = expand_path("~/UIE/experiments/models/transfuse-gan/src")
        if str(src) not in sys.path:
            sys.path.insert(0, str(src))
        from evaluation.benchmark import evaluate_paired_loader

        return evaluate_paired_loader

    def train(self) -> None:
        epochs = int(self.config["train"]["epochs"])
        decay_start = int(self.config["train"]["decay_start"])
        save_interval = int(self.config["train"]["save_interval"])
        val_interval = int(self.config["train"]["val_interval"])

        for epoch in range(1, epochs + 1):
            if hasattr(self.train_loader.dataset, "set_epoch"):
                self.train_loader.dataset.set_epoch(epoch)
            lr_factor = linear_lr(epoch, epochs, decay_start)
            self._set_lr(self.opt_g, float(self.config["train"]["lr_g"]) * lr_factor)
            self._set_lr(self.opt_d, float(self.config["train"]["lr_d"]) * lr_factor)
            stats = self._train_epoch(epoch)
            for key, value in stats.items():
                self.writer.add_scalar(f"train/{key}", value, epoch)
            self.writer.add_scalar("train/lr_g", self.opt_g.param_groups[0]["lr"], epoch)
            self.writer.add_scalar("train/lr_d", self.opt_d.param_groups[0]["lr"], epoch)

            if epoch % val_interval == 0:
                metrics = self.validate()
                for key, value in metrics.items():
                    self.writer.add_scalar(f"val/{key}", value, epoch)
                if metrics["PSNR"] > self.best_psnr:
                    self.best_psnr = metrics["PSNR"]
                    self.save_checkpoint(self.output_dir / "best.pth", epoch, metrics)

            if epoch % save_interval == 0:
                self.save_checkpoint(self.output_dir / f"epoch_{epoch:04d}.pth", epoch)

        self.writer.close()

    def _train_epoch(self, epoch: int) -> dict[str, float]:
        self.g.train()
        self.f.train()
        self.dx.train()
        self.dy.train()
        totals: dict[str, float] = {}
        start = time.perf_counter()

        for step, batch in enumerate(self.train_loader, start=1):
            x = batch["x"].to(self.device, non_blocking=True)
            y = batch["y"].to(self.device, non_blocking=True)
            z = batch["z"].to(self.device, non_blocking=True)
            amp_enabled = self.scaler.is_enabled()

            self.opt_g.zero_grad(set_to_none=True)
            with autocast(self.device.type, enabled=amp_enabled):
                fake_y = self.g(x)
                fake_x = self.f(y)
                rec_x = self.f(fake_y)
                rec_y = self.g(fake_x)
                loss_g_adv = generator_adv_loss(self.dy(fake_y))
                loss_f_adv = generator_adv_loss(self.dx(fake_x))
                loss_cyc = cycle_loss(rec_x, x, rec_y, y)
                loss_con = content_loss(self.vgg, fake_y, x, fake_x, y)
                loss_g = loss_g_adv + loss_f_adv + loss_cyc + loss_con
            self.scaler.scale(loss_g).backward()
            self.scaler.step(self.opt_g)

            self.opt_d.zero_grad(set_to_none=True)
            with autocast(self.device.type, enabled=amp_enabled):
                pred_x = self.dx(x)
                pred_fake_x = self.dx(fake_x.detach())
                loss_dx = discriminator_adv_loss(pred_x, pred_fake_x)
                pred_y = self.dy(y)
                pred_z = self.dy(z)
                pred_fake_y = self.dy(fake_y.detach())
                loss_dy = discriminator_badv_loss(pred_y, pred_z, pred_fake_y)
                loss_d = loss_dx + loss_dy
            self.scaler.scale(loss_d).backward()
            self.scaler.step(self.opt_d)
            self.scaler.update()
            
            peak = torch.cuda.max_memory_allocated() / 1024**3
            print(f"Peak training VRAM: {peak:.3f} GB")
            torch.cuda.reset_peak_memory_stats()


            batch_stats = {
                "loss_g": loss_g.item(),
                "loss_g_adv": loss_g_adv.item(),
                "loss_f_adv": loss_f_adv.item(),
                "loss_cyc": loss_cyc.item(),
                "loss_con": loss_con.item(),
                "loss_d": loss_d.item(),
                "loss_dx": loss_dx.item(),
                "loss_dy": loss_dy.item(),
            }
            for key, value in batch_stats.items():
                totals[key] = totals.get(key, 0.0) + value

            if step == 1:
                save_image(((fake_y[:2] + 1.0) * 0.5).clamp(0, 1), self.output_dir / f"epoch_{epoch:04d}_sample.png")

        elapsed = time.perf_counter() - start
        count = max(1, len(self.train_loader))
        averages = {key: value / count for key, value in totals.items()}
        averages["seconds"] = elapsed
        return averages

    @torch.no_grad()
    def validate(self) -> dict[str, float]:
        wrapper = EvalWrapper(self.g).to(self.device)
        metrics = self.evaluate_paired_loader(wrapper, self.val_loader, self.device)
        return {key: float(value) for key, value in metrics.items()}

    def save_checkpoint(self, path: Path, epoch: int, metrics: dict[str, float] | None = None) -> None:
        checkpoint = {
            "epoch": epoch,
            "G": self.g.state_dict(),
            "F": self.f.state_dict(),
            "D_X": self.dx.state_dict(),
            "D_Y": self.dy.state_dict(),
            "opt_g": self.opt_g.state_dict(),
            "opt_d": self.opt_d.state_dict(),
            "best_psnr": self.best_psnr,
            "config": self.config,
            "metrics": metrics or {},
        }
        torch.save(checkpoint, path)
        if metrics:
            (path.with_suffix(".json")).write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    @staticmethod
    def _set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
        for group in optimizer.param_groups:
            group["lr"] = lr
