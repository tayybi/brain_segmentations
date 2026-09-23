#!/usr/bin/env python3
"""Reproducible 2.5D CNN-Transformer SSL study for mouse brain MRI masks.

The runner deliberately uses only images from the training animal split during
masked-image pretraining. Validation and test animals are never seen before
supervised evaluation.
"""

import argparse
import csv
import json
import random
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "data" / "dataset_split_70_15_15"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
SLICE_PATTERN = re.compile(r"(?P<animal>.+)_slice_(?P<slice>\d+)$")


@dataclass
class RunConfig:
    data_root: str
    output_dir: str
    image_size: int = 256
    batch_size: int = 4
    # Zero is portable in restricted environments and avoids multiprocessing
    # failures; increase this only on a local training workstation.
    workers: int = 0
    seed: int = 120225
    epochs: int = 100
    patience: int = 15
    lr: float = 3e-4
    weight_decay: float = 1e-4
    label_fraction: float = 1.0
    mask_ratio: float = 0.5
    patch_size: int = 16


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def image_paths(directory):
    return sorted(p for p in Path(directory).iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES and not p.name.startswith("."))


def parse_slice(path):
    match = SLICE_PATTERN.match(path.stem)
    if not match:
        raise ValueError(f"Expected '<animal>_slice_<number>' filename, got {path.name}")
    return match.group("animal"), int(match.group("slice"))


def animal_slice_index(images_dir):
    index = defaultdict(dict)
    for path in image_paths(images_dir):
        animal, slice_number = parse_slice(path)
        index[animal][slice_number] = path
    return index


class Brain25DDataset(Dataset):
    """Three neighboring MRI slices predict the mask of the center slice."""

    def __init__(self, images_dir, masks_dir=None, image_size=256, animal_subset=None, augment=False):
        self.index = animal_slice_index(images_dir)
        if animal_subset is not None:
            allowed = set(animal_subset)
            self.index = {animal: slices for animal, slices in self.index.items() if animal in allowed}
        self.items = [(animal, number) for animal, slices in sorted(self.index.items()) for number in sorted(slices)]
        self.masks_dir = Path(masks_dir) if masks_dir else None
        self.image_size = image_size
        self.augment = augment
        if not self.items:
            raise ValueError(f"No valid images found in {images_dir}")

    @property
    def animals(self):
        return sorted(self.index)

    def _read(self, path):
        image = Image.open(path).convert("L").resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        return torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0)

    def _mask(self, path):
        mask_path = self.masks_dir / path.name
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask not found: {mask_path}")
        mask = Image.open(mask_path).convert("L").resize((self.image_size, self.image_size), Image.Resampling.NEAREST)
        return torch.from_numpy((np.asarray(mask) > 127).astype(np.float32))

    def __getitem__(self, item):
        animal, number = self.items[item]
        slices = self.index[animal]
        center = slices[number]
        neighbors = [slices.get(number - 1, center), center, slices.get(number + 1, center)]
        image = torch.stack([self._read(path) for path in neighbors])
        flip_horizontal = self.augment and random.random() < 0.5
        flip_vertical = self.augment and random.random() < 0.5
        if flip_horizontal:
            image = image.flip(-1)
        if flip_vertical:
            image = image.flip(-2)
        if self.masks_dir is None:
            return image
        mask = self._mask(center)
        if flip_horizontal:
            mask = mask.flip(-1)
        if flip_vertical:
            mask = mask.flip(-2)
        return image, mask, animal

    def __len__(self):
        return len(self.items)


class ConvTransformerEncoder(nn.Module):
    def __init__(self, width=64, depth=4, heads=8):
        super().__init__()
        # A 16x16 token grid at 256x256 input keeps attention tractable on this dataset.
        self.stem = nn.Sequential(nn.Conv2d(3, width, 3, stride=2, padding=1), nn.BatchNorm2d(width), nn.GELU(), nn.Conv2d(width, width, 3, stride=2, padding=1), nn.BatchNorm2d(width), nn.GELU())
        self.down = nn.Sequential(nn.Conv2d(width, width * 2, 3, stride=2, padding=1), nn.BatchNorm2d(width * 2), nn.GELU(), nn.Conv2d(width * 2, width * 4, 3, stride=2, padding=1), nn.BatchNorm2d(width * 4), nn.GELU())
        layer = nn.TransformerEncoderLayer(d_model=width * 4, nhead=heads, dim_feedforward=width * 8, dropout=0.1, activation="gelu", batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(width * 4)

    def forward(self, x):
        shallow = self.stem(x)
        encoded = self.down(shallow)
        batch, channels, height, width = encoded.shape
        tokens = self.transformer(encoded.flatten(2).transpose(1, 2))
        encoded = self.norm(tokens).transpose(1, 2).reshape(batch, channels, height, width)
        return shallow, encoded


class HybridSegmentationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = ConvTransformerEncoder()
        self.decoder = nn.Sequential(nn.Conv2d(256, 128, 3, padding=1), nn.GELU(), nn.ConvTranspose2d(128, 64, 2, stride=2), nn.GELU(), nn.ConvTranspose2d(64, 32, 2, stride=2), nn.GELU())
        self.skip = nn.Conv2d(64, 32, 1)
        self.head = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        shallow, encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        logits = self.head(decoded + self.skip(shallow))
        return F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)


class MaskedReconstructionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = ConvTransformerEncoder()
        self.decoder = nn.Sequential(nn.ConvTranspose2d(256, 128, 2, stride=2), nn.GELU(), nn.ConvTranspose2d(128, 64, 2, stride=2), nn.GELU(), nn.Conv2d(64, 1, 1), nn.Sigmoid())

    def forward(self, x):
        _, encoded = self.encoder(x)
        reconstruction = self.decoder(encoded)
        return F.interpolate(reconstruction, size=x.shape[-2:], mode="bilinear", align_corners=False)


def random_patch_mask(images, patch_size, ratio):
    masked = images.clone()
    batch, _, height, width = images.shape
    grid_h, grid_w = height // patch_size, width // patch_size
    count = int(grid_h * grid_w * ratio)
    for i in range(batch):
        for location in torch.randperm(grid_h * grid_w, device=images.device)[:count]:
            row, col = divmod(location.item(), grid_w)
            masked[i, :, row * patch_size:(row + 1) * patch_size, col * patch_size:(col + 1) * patch_size] = 0
    return masked


def dice_loss(logits, target):
    probs = torch.sigmoid(logits)
    intersection = (probs * target).sum((1, 2, 3))
    return 1 - ((2 * intersection + 1e-6) / (probs.sum((1, 2, 3)) + target.sum((1, 2, 3)) + 1e-6)).mean()


def segmentation_loss(logits, target):
    return 0.5 * F.binary_cross_entropy_with_logits(logits, target) + 0.5 * dice_loss(logits, target)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    per_animal = defaultdict(lambda: [0, 0, 0])
    for images, masks, animals in loader:
        prediction = torch.sigmoid(model(images.to(device))).cpu() > 0.5
        truth = masks.bool().unsqueeze(1)
        for pred, target, animal in zip(prediction, truth, animals):
            per_animal[animal][0] += (pred & target).sum().item()
            per_animal[animal][1] += pred.sum().item()
            per_animal[animal][2] += target.sum().item()
    rows = []
    for animal, (intersection, predicted, target) in sorted(per_animal.items()):
        dice = (2 * intersection + 1e-6) / (predicted + target + 1e-6)
        iou = (intersection + 1e-6) / (predicted + target - intersection + 1e-6)
        rows.append({"animal": animal, "dice": dice, "iou": iou})
    return rows


def choose_animals(dataset, fraction, seed):
    animals = dataset.animals
    if not 0 < fraction <= 1:
        raise ValueError("label_fraction must be in (0, 1]")
    keep = max(1, round(len(animals) * fraction))
    return sorted(random.Random(seed).sample(animals, keep))


def loaders(config, split, supervised, augment=False, animal_subset=None):
    root = Path(config.data_root) / split
    dataset = Brain25DDataset(root / "images", root / "masks" if supervised else None, config.image_size, animal_subset, augment)
    return dataset, DataLoader(dataset, batch_size=config.batch_size, shuffle=augment, num_workers=config.workers, pin_memory=torch.cuda.is_available())


def save_checkpoint(path, model, config, **metadata):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": asdict(config), **metadata}, path)


def run_pretrain(config, device):
    _, loader = loaders(config, "train", supervised=False, augment=True)
    model = MaskedReconstructionModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    for epoch in range(1, config.epochs + 1):
        model.train(); losses = []
        for images in loader:
            images = images.to(device)
            reconstruction = model(random_patch_mask(images, config.patch_size, config.mask_ratio))
            loss = F.l1_loss(reconstruction, images[:, 1:2])
            optimizer.zero_grad(); loss.backward(); optimizer.step(); losses.append(loss.item())
        print(f"pretrain epoch {epoch:03d} | reconstruction_l1={np.mean(losses):.5f}")
    checkpoint = Path(config.output_dir) / "pretrain_encoder.pt"
    save_checkpoint(checkpoint, model, config, stage="pretrain")
    return checkpoint


def run_finetune(config, device, pretrained_checkpoint=None):
    train_ds, train_loader = loaders(config, "train", supervised=True, augment=False)
    selected_animals = choose_animals(train_ds, config.label_fraction, config.seed)
    _, train_loader = loaders(config, "train", supervised=True, augment=True, animal_subset=selected_animals)
    _, val_loader = loaders(config, "val", supervised=True)
    model = HybridSegmentationModel().to(device)
    if pretrained_checkpoint:
        state = torch.load(pretrained_checkpoint, map_location=device, weights_only=False)["state_dict"]
        model.encoder.load_state_dict({key.removeprefix("encoder."): value for key, value in state.items() if key.startswith("encoder.")})
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    best, stale = -1.0, 0
    checkpoint = Path(config.output_dir) / "best_segmentation.pt"
    for epoch in range(1, config.epochs + 1):
        model.train(); losses = []
        for images, masks, _ in train_loader:
            logits = model(images.to(device)); loss = segmentation_loss(logits, masks.to(device).unsqueeze(1))
            optimizer.zero_grad(); loss.backward(); optimizer.step(); losses.append(loss.item())
        val_rows = evaluate(model, val_loader, device)
        val_dice = float(np.mean([row["dice"] for row in val_rows]))
        print(f"finetune epoch {epoch:03d} | train_loss={np.mean(losses):.5f} | val_animal_dice={val_dice:.4f}")
        if val_dice > best:
            best, stale = val_dice, 0
            save_checkpoint(checkpoint, model, config, stage="finetune", val_animal_dice=best, training_animals=selected_animals)
        else:
            stale += 1
            if stale >= config.patience:
                break
    return checkpoint


def write_results(config, checkpoint, device):
    _, loader = loaders(config, "test", supervised=True)
    model = HybridSegmentationModel().to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=False)["state_dict"]
    model.load_state_dict(state)
    rows = evaluate(model, loader, device)
    output = Path(config.output_dir); output.mkdir(parents=True, exist_ok=True)
    with (output / "test_per_animal_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["animal", "dice", "iou"]); writer.writeheader(); writer.writerows(rows)
    summary = {"n_test_animals": len(rows), "mean_dice": float(np.mean([r["dice"] for r in rows])), "std_dice": float(np.std([r["dice"] for r in rows], ddof=1)) if len(rows) > 1 else 0.0, "mean_iou": float(np.mean([r["iou"] for r in rows]))}
    (output / "test_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["pretrain", "finetune", "evaluate", "full"])
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "experiments" / "hybrid_ssl")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--epochs", type=int, default=100); parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=4); parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=256); parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4); parser.add_argument("--label-fraction", type=float, default=1.0)
    parser.add_argument("--mask-ratio", type=float, default=0.5); parser.add_argument("--patch-size", type=int, default=16); parser.add_argument("--seed", type=int, default=120225)
    args = parser.parse_args(); set_seed(args.seed)
    config = RunConfig(**{key: str(value) if key in {"data_root", "output_dir"} else value for key, value in vars(args).items() if key not in {"command", "checkpoint"}})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); print(f"device={device}")
    if args.command == "pretrain": run_pretrain(config, device)
    elif args.command == "finetune": run_finetune(config, device, args.checkpoint)
    elif args.command == "evaluate":
        if not args.checkpoint: parser.error("evaluate requires --checkpoint")
        write_results(config, args.checkpoint, device)
    else:
        pretrain = run_pretrain(config, device)
        checkpoint = run_finetune(config, device, pretrain)
        write_results(config, checkpoint, device)


if __name__ == "__main__":
    main()
