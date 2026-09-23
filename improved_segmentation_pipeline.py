import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

import timm


SEED = 120225


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class Config:
    data_root: Path = Path("/home/tali1/brain_segmentations/data/dataset_split_70_15_15")
    train_images_dir: Path = data_root / "train" / "images"
    train_masks_dir: Path = data_root / "train" / "masks"
    val_images_dir: Path = data_root / "val" / "images"
    val_masks_dir: Path = data_root / "val" / "masks"
    checkpoint_path: Path = Path("/home/tali1/brain_segmentations/checkpoints/best_brain_segmentation.pt")
    img_size: int = 256
    patch_size: int = 8
    batch_size: int = 4
    epochs: int = 80
    lr_backbone: float = 3e-5
    lr_decoder: float = 3e-4
    weight_decay: float = 1e-4
    patience: int = 12
    mask_ratio: float = 0.5
    num_classes: int = 1


CFG = Config()


class PairDataset(Dataset):
    def __init__(self, images_dir, masks_dir, img_size=CFG.img_size, augment=False):
        self.images = sorted(
            [
                p
                for p in Path(images_dir).iterdir()
                if p.is_file() and not p.name.startswith(".") and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
            ]
        )
        self.masks_dir = Path(masks_dir)
        self.img_size = img_size
        self.augment = augment
        self.resize_img = transforms.Resize((img_size, img_size), interpolation=InterpolationMode.BILINEAR)
        self.resize_mask = transforms.Resize((img_size, img_size), interpolation=InterpolationMode.NEAREST)

    def _load_mask(self, image_path: Path):
        stem = image_path.stem
        for ext in [".png", ".tif", ".tiff", ".jpg", ".jpeg"]:
            mask_path = self.masks_dir / (stem + ext)
            if mask_path.exists():
                return Image.open(mask_path).convert("L")
        raise FileNotFoundError(f"Mask not found for {image_path}")

    def _random_flip_rotate(self, img, mask):
        if random.random() < 0.5:
            img = transforms.functional.hflip(img)
            mask = transforms.functional.hflip(mask)
        if random.random() < 0.5:
            img = transforms.functional.vflip(img)
            mask = transforms.functional.vflip(mask)
        if random.random() < 0.5:
            angle = random.uniform(-10, 10)
            img = transforms.functional.rotate(img, angle, interpolation=InterpolationMode.BILINEAR)
            mask = transforms.functional.rotate(mask, angle, interpolation=InterpolationMode.NEAREST)
        return img, mask

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = Image.open(image_path).convert("L")
        mask = self._load_mask(image_path)

        image = self.resize_img(image)
        mask = self.resize_mask(mask)

        if self.augment:
            image, mask = self._random_flip_rotate(image, mask)

        image = transforms.ToTensor()(image)
        image = (image - image.mean()) / (image.std() + 1e-6)

        mask_np = np.array(mask)
        mask_bin = (mask_np > 127).astype(np.int64)
        mask_t = torch.from_numpy(mask_bin)
        return image, mask_t


class ViTBackbone(nn.Module):
    def __init__(self, img_size=CFG.img_size, patch=CFG.patch_size, model_name="vit_base_patch16_224"):
        super().__init__()
        self.patch = patch
        self.grid = img_size // patch
        self.embed_dim = 768
        self.proj = nn.Conv2d(1, self.embed_dim, kernel_size=patch, stride=patch)
        self.pos = nn.Parameter(torch.zeros(1, self.grid * self.grid, self.embed_dim))
        self.enc = timm.create_model(model_name, pretrained=True, num_classes=0)
        if hasattr(self.enc, "patch_embed"):
            for p in self.enc.patch_embed.parameters():
                p.requires_grad = False

    def forward(self, x):
        tokens = self.proj(x)
        B, C, H, W = tokens.shape
        tokens = tokens.flatten(2).transpose(1, 2)
        tokens = tokens + self.pos
        for block in self.enc.blocks:
            tokens = block(tokens)
        return self.enc.norm(tokens)


class ResidualDecoder(nn.Module):
    def __init__(self, embed_dim=768, img_size=CFG.img_size, patch=CFG.patch_size, num_classes=CFG.num_classes):
        super().__init__()
        h = img_size // patch
        self.h = h
        self.patch = patch
        self.img_size = img_size

        self.pre = nn.Sequential(
            nn.Conv2d(embed_dim, 512, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(512, 256, 3, padding=1),
            nn.GELU(),
        )
        self.up1 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.block1 = nn.Sequential(nn.Conv2d(128, 128, 3, padding=1), nn.GELU())
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.block2 = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), nn.GELU())
        self.up3 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.block3 = nn.Sequential(nn.Conv2d(32, 32, 3, padding=1), nn.GELU())
        self.out = nn.Conv2d(32, num_classes, 1)

    def forward(self, tokens):
        B, N, C = tokens.shape
        x = tokens.transpose(1, 2).reshape(B, C, self.h, self.h)
        x = self.pre(x)
        x = self.block1(self.up1(x))
        x = self.block2(self.up2(x))
        x = self.block3(self.up3(x))
        x = F.interpolate(x, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False)
        return self.out(x)


class BCEPlusDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        logits = logits.squeeze(1)
        probs = torch.sigmoid(logits)
        targets = targets.float()

        bce_loss = self.bce(logits, targets)
        inter = (probs * targets).sum()
        den = probs.sum() + targets.sum() + 1e-6
        dice = 1.0 - (2.0 * inter / den)
        return 0.5 * bce_loss + 0.5 * dice


def dice_score(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()
    targets = targets.float().unsqueeze(1)
    inter = (preds * targets).sum(dim=(0, 1, 2, 3))
    den = preds.sum(dim=(0, 1, 2, 3)) + targets.sum(dim=(0, 1, 2, 3)) + eps
    return (2.0 * inter / den).mean().item()


class BrainSegmentationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = ViTBackbone()
        self.decoder = ResidualDecoder(embed_dim=self.backbone.embed_dim)

    def forward(self, x):
        features = self.backbone(x)
        return self.decoder(features)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    num_samples = 0

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        logits = model(images)
        loss = criterion(logits, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        num_samples += images.size(0)

    return total_loss / max(num_samples, 1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    scores = []
    num_samples = 0

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)
        logits = model(images)
        loss = BCEPlusDiceLoss()(logits, masks)

        total_loss += loss.item() * images.size(0)
        num_samples += images.size(0)
        scores.append(dice_score(logits, masks))

    return total_loss / max(num_samples, 1), float(np.mean(scores)) if scores else 0.0


def main():
    print("Device:", DEVICE)
    CFG.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    train_ds = PairDataset(CFG.train_images_dir, CFG.train_masks_dir, augment=True)
    val_ds = PairDataset(CFG.val_images_dir, CFG.val_masks_dir, augment=False)

    train_loader = DataLoader(train_ds, batch_size=CFG.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=CFG.batch_size, shuffle=False, num_workers=2)

    model = BrainSegmentationModel().to(DEVICE)
    criterion = BCEPlusDiceLoss()
    optimizer = torch.optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": CFG.lr_backbone, "weight_decay": CFG.weight_decay},
            {"params": model.decoder.parameters(), "lr": CFG.lr_decoder, "weight_decay": CFG.weight_decay},
        ]
    )

    best_val_dice = float("-inf")
    no_improve = 0

    for epoch in range(1, CFG.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_loss, val_dice = evaluate(model, val_loader, DEVICE)
        print(f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_dice={val_dice:.4f}")

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            no_improve = 0
            torch.save({"state_dict": model.state_dict(), "val_dice": val_dice, "val_loss": val_loss}, CFG.checkpoint_path)
        else:
            no_improve += 1

        if no_improve >= CFG.patience:
            print("Early stopping triggered.")
            break


if __name__ == "__main__":
    main()
