"""
pretrain_image.py — supervised multi-label classification pretraining on OpenImages latents

Purpose:
  Supervised multi-label classification pretraining on large-scale OpenImages VAE-encoded latents,
  teaching the ConvNeXt/ResNet/ViT backbone stem general latent -> semantics representations.
  The pretrain run produces a backbone checkpoint for train_image.py latent mode to load and finetune.

Features:
  - Fully self-contained; modifies no existing training/data code
  - Supports ConvNeXt / ResNet / ViT backbones (uniform torchvision loading, aligned with train_image.py)
  - ViT uses torchvision IN1K_V1 weights, conv_proj channel-expanded for latents, fixed 224x224 input
  - Uniform resize to target_hw (default 224x224), aligned with Image/IN1K
  - Masked BCE loss: computed only on labeled classes (Open Images partial-annotation property)
  - Metrics: mAP + overall accuracy + loss
  - Train/val curves plotted automatically

Usage:
    python pretrain_image.py
    (edit the CONFIG at the bottom)
"""
import os
import sys
import csv
import json
import glob
import math
import random
import datetime
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm
from torch.utils.data import Dataset, DataLoader, Sampler
from tqdm import tqdm

warnings.filterwarnings("ignore")

# Import only the minimal shared helpers from utils_common (no existing files modified)
# --- repo layout: shared training utilities live in <repo>/common/ ---
import os as _os, sys as _sys
_common = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "common")
if _common not in _sys.path:
    _sys.path.insert(0, _common)
from utils_common import seed_everything, save_json
from models_zoo import get_backbone_family


# ===========================================================================
# 1. Data loading
# ===========================================================================
def load_classes(classes_csv):
    """Load classes.csv -> {LabelName: ClassName}, an ordered list"""
    label_to_name = {}
    with open(classes_csv, "r", encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) >= 2:
                label_to_name[row[0].strip()] = row[1].strip()
    # fixed order
    all_labels = sorted(label_to_name.keys())
    label_to_idx = {lbl: i for i, lbl in enumerate(all_labels)}
    idx_to_name = {i: label_to_name[lbl] for i, lbl in enumerate(all_labels)}
    return label_to_idx, idx_to_name


def load_all_classifications(labels_dir, label_to_idx):
    """Scan all batch_*_classifications.csv files and aggregate into:
    {ImageID: {class_idx: confidence(0 or 1)}}
    """
    image_annotations = {}
    csv_files = sorted(glob.glob(os.path.join(labels_dir, "batch_*_classifications.csv")))
    print(f"[Labels] scanning {len(csv_files)} batch label files ...")

    for csv_path in tqdm(csv_files, desc="Loading labels"):
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                image_id = row["ImageID"].strip()
                label_name = row["LabelName"].strip()
                confidence = int(row["Confidence"])
                if label_name not in label_to_idx:
                    continue
                cls_idx = label_to_idx[label_name]
                if image_id not in image_annotations:
                    image_annotations[image_id] = {}
                image_annotations[image_id][cls_idx] = confidence

    print(f"[Labels] {len(image_annotations)} images have labels")
    return image_annotations


# ---------------------------------------------------------------------------
# Safety CSV label loading (predictions.csv format, for distill_classify safety-domain data)
# ---------------------------------------------------------------------------
SAFETY_CLASS_NAMES = ["porn", "gore", "ip_0", "ip_1", "ip_2", "ip_3", "ip_4", "ip_5"]
SAFETY_NUM_CLASSES = len(SAFETY_CLASS_NAMES)  # 8


def _map_ip_copyright(val):
    """Map an ip_copyright raw value -> ip_label (0-5), same as train_image.py.
    1->0, 2->1, 3->2, 4->3, 5->4, 0/6->5 (other)"""
    val = int(val)
    if 1 <= val <= 5:
        return val - 1
    return 5  # 0 or 6 -> "other"


def load_safety_labels(safety_csv):
    """Read predictions.csv (safety label format) into a structure compatible with load_all_classifications.

    Returns:
        image_annotations: {image_id: {cls_idx: confidence(0/1)}}
        where cls_idx indexes SAFETY_CLASS_NAMES:
            0=porn, 1=gore, 2-7=ip_0~ip_5 (one-hot)

    Label mapping rules (exactly the same as train_image.py):
        pornographic==2 → porn=1; else → 0
        violence_gore==2 → gore=1; else → 0
        ip_copyright 1-5 → ip_label 0-4 (one-hot); 0/6 → ip_5
    """
    import pandas as pd
    print(f"[Safety Labels] reading {safety_csv} ...")
    df = pd.read_csv(safety_csv, usecols=["file_path", "pornographic", "violence_gore", "ip_copyright"])
    # filter invalid rows
    df = df.dropna(subset=["file_path", "pornographic", "violence_gore", "ip_copyright"])
    df["pornographic"] = df["pornographic"].astype(int)
    df["violence_gore"] = df["violence_gore"].astype(int)
    df["ip_copyright"] = df["ip_copyright"].astype(int)

    image_annotations = {}
    for _, row in df.iterrows():
        # image_id = the file-name stem (no extension), aligned with the latent .pth file names
        fname = os.path.basename(row["file_path"])
        image_id = os.path.splitext(fname)[0]

        annotations = {}
        # Class 0: porn
        annotations[0] = 1.0 if row["pornographic"] == 2 else 0.0
        # Class 1: gore
        annotations[1] = 1.0 if row["violence_gore"] == 2 else 0.0
        # Classes 2-7: ip one-hot
        ip_label = _map_ip_copyright(row["ip_copyright"])
        for i in range(6):
            annotations[2 + i] = 1.0 if i == ip_label else 0.0

        image_annotations[image_id] = annotations

    print(f"[Safety Labels] {len(image_annotations)} samples in total")
    return image_annotations


def build_manifest(latent_dir, image_annotations):
    """Intersect with the latent directory and return the list of valid samples.
    Two directory layouts are supported:
      - flat: latent_dir/*.pth
      - subdirs: latent_dir/0/*.pth, latent_dir/1/*.pth, ...
    The file list is preferably loaded from the cache file {latent_dir}/_latent_files.cache.txt,
    avoiding an os.listdir() scan of the network storage at every startup (OpenImages has millions of files).
    Use build_latent_cache.py to generate the cache in advance.
    """
    cache_path = os.path.join(latent_dir, "_latent_files.cache.txt")

    if os.path.exists(cache_path):
        # load the file list from the cache (one relative path per line)
        print(f"[Manifest] loading from cache: {cache_path}")
        with open(cache_path, "r") as f:
            lines = [line.strip() for line in f if line.strip()]
        latent_files = []
        for line in lines:
            full_path = os.path.join(latent_dir, line)
            fname = os.path.basename(line)
            latent_files.append((fname, full_path))
        print(f"[Manifest] cached file count: {len(latent_files)}")
    else:
        # scan the directory (first run, or the cache does not exist)
        print(f"[Manifest] scanning directory (first run; the network storage may take minutes): {latent_dir}")
        sys.stdout.flush()
        # first check whether .pth files sit directly at the top level
        top_files = [f for f in os.listdir(latent_dir) if f.endswith(".pth")]
        if top_files:
            # flat mode
            latent_files = [(f, os.path.join(latent_dir, f)) for f in top_files]
        else:
            # subdir mode: scan all first-level subdirectories
            latent_files = []
            for sub in sorted(os.listdir(latent_dir)):
                sub_path = os.path.join(latent_dir, sub)
                if os.path.isdir(sub_path):
                    for f in os.listdir(sub_path):
                        if f.endswith(".pth"):
                            latent_files.append((f, os.path.join(sub_path, f)))
        print(f"[Manifest] scanned {len(latent_files)} .pth files")

        # save the cache for later runs
        try:
            cache_lines = [os.path.relpath(p, latent_dir) for _, p in latent_files]
            with open(cache_path, "w") as f:
                f.write("\n".join(cache_lines))
            print(f"[Manifest] cache saved: {cache_path}")
        except OSError:
            print("[Manifest] [WARN] cannot write the cache file (directory may be read-only)")

    manifest = []
    n_no_label = 0
    for fname, full_path in latent_files:
        image_id = os.path.splitext(fname)[0]
        if image_id not in image_annotations:
            n_no_label += 1
            continue
        manifest.append({
            "image_id": image_id,
            "latent_path": full_path,
            "annotations": image_annotations[image_id],
        })
    print(f"[Manifest] valid samples: {len(manifest)}, skipped without labels: {n_no_label}")
    return manifest


def _visualize_latent_grid(latent, title, save_path):
    """Visualize a latent [C,H,W] as a grid and save it.
    Column count adapts: 4 columns when C<=16 (16ch -> 4x4); 8 columns when C>16 (32ch -> 4x8).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    C = int(latent.shape[0])
    ncols = 4 if C <= 16 else 8
    nrows = (C + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.2, nrows * 2.2))
    axes_flat = np.array(axes).reshape(-1)

    for ch_idx in range(C):
        ax = axes_flat[ch_idx]
        ax.imshow(latent[ch_idx].numpy(), cmap="viridis", aspect="auto")
        ax.set_title(f"Ch {ch_idx}", fontsize=8)
        ax.axis("off")
    for i in range(C, len(axes_flat)):
        axes_flat[i].axis("off")

    fig.suptitle(title, fontsize=9, y=0.99)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def _resize_and_crop_for_preview(latent, target_hw):
    """Same logic as Dataset._resize_and_crop (center crop), for previews"""
    C, h, w = latent.shape
    th, tw = target_hw
    scale = max(th / h, tw / w)
    if abs(scale - 1.0) > 1e-4:
        new_h = max(th, int(round(h * scale)))
        new_w = max(tw, int(round(w * scale)))
        latent = F.interpolate(
            latent.unsqueeze(0), size=(new_h, new_w),
            mode="bilinear", align_corners=False,
        ).squeeze(0)
    _, h2, w2 = latent.shape
    if h2 > th:
        top = (h2 - th) // 2
        latent = latent[:, top:top + th, :]
    if w2 > tw:
        left = (w2 - tw) // 2
        latent = latent[:, :, left:left + tw]
    return latent


def preview_data(manifest, idx_to_name, images_dir, output_dir,
                 target_hw=(128, 128), num_preview=100, model_name="z-image-turbo"):
    """Preview the dataset: copy raw images + multi-channel latent visualizations (raw + processed).
    Output directories:
      - preview/images/         raw images
      - preview/latents_raw/    raw latent visualizations
      - preview/latents_crop/   latent visualizations after resize+crop
    """
    import shutil

    img_out_dir = os.path.join(output_dir, "images")
    lat_raw_dir = os.path.join(output_dir, "latents_raw")
    lat_crop_dir = os.path.join(output_dir, "latents_crop")
    os.makedirs(img_out_dir, exist_ok=True)
    os.makedirs(lat_raw_dir, exist_ok=True)
    os.makedirs(lat_crop_dir, exist_ok=True)

    rng = random.Random(0)
    samples = rng.sample(manifest, min(num_preview, len(manifest)))

    n_saved = 0
    n_no_img = 0
    for item in samples:
        image_id = item["image_id"]

        # extract positive label names (confidence=1)
        pos_labels = []
        for cls_idx, conf in item["annotations"].items():
            if conf >= 1.0:
                name = idx_to_name.get(cls_idx, f"cls{cls_idx}")
                name = name.replace("/", "-").replace(" ", "_").replace("(", "").replace(")", "")
                pos_labels.append(name)
        if not pos_labels:
            label_str = "NO_POS_LABEL"
        else:
            label_str = "_".join(pos_labels[:5])
            if len(pos_labels) > 5:
                label_str += f"_+{len(pos_labels)-5}more"

        # --- save the raw image ---
        img_path = None
        for ext in [".jpg", ".png", ".jpeg"]:
            candidate = os.path.join(images_dir, f"{image_id}{ext}")
            if os.path.exists(candidate):
                img_path = candidate
                break
        if img_path is not None:
            ext = os.path.splitext(img_path)[1]
            dst_name = f"{image_id}__{label_str}{ext}"
            shutil.copy2(img_path, os.path.join(img_out_dir, dst_name))
        else:
            n_no_img += 1

        # --- latent visualization ---
        latent_path = item["latent_path"]
        if os.path.exists(latent_path):
            try:
                latent = torch.load(latent_path, map_location="cpu", weights_only=True)
            except Exception:
                latent = torch.load(latent_path, map_location="cpu", weights_only=False)
            latent = _squeeze_latent_to_3d(latent, model_name)
            latent = latent.float()  # [C, H, W]

            h, w = latent.shape[1], latent.shape[2]

            # raw latent
            title_raw = f"{image_id} (raw {h}x{w})\n{label_str}"
            _visualize_latent_grid(
                latent, title_raw,
                os.path.join(lat_raw_dir, f"{image_id}__{label_str}_raw.png"),
            )

            # processed latent (resize + crop)
            latent_crop = _resize_and_crop_for_preview(latent, target_hw)
            title_crop = f"{image_id} (crop {target_hw[0]}x{target_hw[1]})\n{label_str}"
            _visualize_latent_grid(
                latent_crop, title_crop,
                os.path.join(lat_crop_dir, f"{image_id}__{label_str}_crop.png"),
            )

        n_saved += 1

    print(f"[Preview] saved {n_saved} previews -> {output_dir}")
    print(f"  raw images: {img_out_dir}")
    print(f"  raw latents: {lat_raw_dir}")
    print(f"  processed latents: {lat_crop_dir}")
    if n_no_img > 0:
        print(f"  (skipped {n_no_img} raw images: not found)")


def split_train_val(manifest, val_size=10000, seed=42):
    """Randomly draw a fixed number of samples from the manifest as the val set; the rest become train.

    Uses an independent RNG (fixed seed) so the val set is identical across runs,
    without touching the global random state.

    Args:
        manifest: the full sample list
        val_size: val-set size, default 10000
        seed: random seed for reproducible val-set sampling
    """
    rng = random.Random(seed)
    shuffled = sorted(manifest, key=lambda x: x["image_id"])  # sort by ID first for determinism
    rng.shuffle(shuffled)
    val_size = min(val_size, len(shuffled) // 2)  # val never exceeds 50% of the total
    val_manifest = shuffled[:val_size]
    train_manifest = shuffled[val_size:]
    print(f"[Val Split] val_size={val_size} (seed={seed}), "
          f"train={len(train_manifest)}, val={len(val_manifest)}")
    return train_manifest, val_manifest


def _squeeze_latent_to_3d(latent, model_name):
    """Squeeze a latent from its on-disk layout to 3D [C, H, W] by model name.

    Each model has a fixed on-disk layout; the hardcoded dispatch avoids heuristic squeeze misjudging the channel dim:
      - z-image-turbo:   [1, C, H, W]    → squeeze(0)            → [C, H, W]
      - qwen-image-2512: [1, C, 1, H, W] → squeeze(0).squeeze(1) → [C, H, W]
      - internvl-u:      [1, C, H, W] -> squeeze(0) -> [C, H, W]  (reuses the Qwen VAE; data generation added no T dim)
                         or [1, C, 1, H, W] -> squeeze(0).squeeze(1) -> [C, H, W]  (5D compatible)
    To add a model, append an elif branch below.
    """
    if model_name in ("qwen-image-2512", "internvl-u"):
        # saved [1, C, 1, H, W] (5D), or internvl-u's [1, C, H, W] (4D, no T dim)
        if latent.ndim == 5:
            latent = latent.squeeze(0).squeeze(1)  # → [C, H, W]
        elif latent.ndim == 4 and latent.shape[1] == 1:
            # rare case: batch already squeezed but T=1 kept -> [C, 1, H, W]
            latent = latent.squeeze(1)
        elif latent.ndim == 4 and latent.shape[0] == 1:
            # internvl-u: data generation added no T=1 dim; saved as [1, C, H, W]
            latent = latent.squeeze(0)
        elif latent.ndim == 3:
            pass  # already [C, H, W]
        else:
            raise ValueError(
                f"{model_name} expects [1,C,1,H,W] or [1,C,H,W] or [C,H,W], got ndim={latent.ndim}, shape={tuple(latent.shape)}"
            )
    elif model_name in ("z-image-turbo", "hunyuan-image-2_1", "flux2-klein-base-9b"):
        # saved [1, C, H, W] (4D)
        # z-image-turbo: C=16, hunyuan-image-2_1: C=64, flux2-klein-base-9b: C=32
        if latent.ndim == 4:
            latent = latent.squeeze(0)
        elif latent.ndim == 3:
            pass
        else:
            raise ValueError(
                f"{model_name} expects [1,C,H,W] or [C,H,W], got ndim={latent.ndim}, shape={tuple(latent.shape)}"
            )
    else:
        # default compat: try a generic squeeze
        if latent.ndim == 5 and latent.shape[0] == 1:
            latent = latent.squeeze(0)
        if latent.ndim == 4 and latent.shape[0] == 1:
            latent = latent.squeeze(0)
        if latent.ndim == 4 and latent.shape[1] == 1:
            latent = latent.squeeze(1)
    return latent


def compute_latent_channel_stats(manifest, num_samples=500, seed=42, model_name="z-image-turbo"):
    """Sample N latents from the training set and compute per-channel mean/std.
    Returns: (mean: Tensor[C], std: Tensor[C], info: dict)
    """
    rng = random.Random(seed)
    n_req = min(num_samples, len(manifest))
    sub = rng.sample(manifest, n_req)

    running_sum = None
    running_sumsq = None
    count = 0
    n_loaded = 0
    n_skipped_nan = 0
    n_failed = 0

    for item in tqdm(sub, desc="Computing latent stats", ncols=80):
        try:
            try:
                latent = torch.load(item["latent_path"], map_location="cpu", weights_only=True)
            except Exception:
                latent = torch.load(item["latent_path"], map_location="cpu", weights_only=False)
        except Exception:
            n_failed += 1
            continue
        latent = _squeeze_latent_to_3d(latent, model_name)
        latent = latent.float()  # [C, H, W]

        # skip corrupted latents containing NaN/Inf so they do not pollute the stats
        if torch.isnan(latent).any() or torch.isinf(latent).any():
            n_skipped_nan += 1
            continue

        C = latent.shape[0]
        if running_sum is None:
            running_sum = torch.zeros(C, dtype=torch.float64)
            running_sumsq = torch.zeros(C, dtype=torch.float64)
        flat = latent.reshape(C, -1).double()
        running_sum += flat.sum(dim=1)
        running_sumsq += (flat ** 2).sum(dim=1)
        count += flat.shape[1]
        n_loaded += 1

    # safety net: raise on zero valid samples instead of producing NaN
    if n_loaded == 0 or count == 0:
        raise RuntimeError(
            f"compute_latent_channel_stats: no valid latents for stats "
            f"(requested={n_req}, failed={n_failed}, skipped_nan={n_skipped_nan}). "
            f"Check whether the latent files are corrupted or the path is correct."
        )

    # compute in float64 throughout, convert to float32 at the end; avoids precision loss making variance negative
    mean64 = running_sum / count
    var64 = (running_sumsq / count) - mean64 * mean64
    var64 = var64.clamp(min=0)
    std64 = var64.sqrt()

    mean = mean64.float()
    std = std64.float().clamp(min=1e-6)

    if n_skipped_nan > 0:
        print(f"  [Latent Stats WARNING] skipped {n_skipped_nan} latent files containing NaN/Inf")
    if n_failed > 0:
        print(f"  [Latent Stats WARNING] {n_failed} files failed to load")

    info = {
        "num_channels": int(mean.shape[0]),
        "num_samples_used": n_loaded,
        "num_skipped_nan": n_skipped_nan,
        "num_failed": n_failed,
        "mean_per_channel": mean.tolist(),
        "std_per_channel": std.tolist(),
        "mean_overall": float(mean.mean()),
        "std_overall": float(std.mean()),
    }
    print(f"  [Latent Stats] channels={info['num_channels']}, samples={n_loaded}, "
          f"mean_overall={info['mean_overall']:.4f}, std_overall={info['std_overall']:.4f}")
    return mean, std, info


def _skip_none_on_error(getitem_fn):
    """Defensive decorator for __getitem__: on a data read/preprocess error, print a warning and return None (skip the sample).
    A single corrupted record or transient storage read failure (e.g. FileNotFoundError on a file that does exist
    but is briefly unreadable) then cannot abort training — collate_fn filters the None; empty batches are skipped.
    """
    def wrapper(self, idx):
        try:
            return getitem_fn(self, idx)
        except Exception as e:
            try:
                item = self.manifest[idx]
                _id = item.get("image_id") or item.get("latent_path") or idx
            except Exception:
                _id = idx
            print(f"  [WARN] Dataset.__getitem__ skipping sample idx={idx} ({_id}): "
                  f"{type(e).__name__}: {e}")
            return None
    return wrapper


class OpenImagesLatentDataset(Dataset):
    """OpenImages latent multi-label dataset.
    Fixed output size (target_h, target_w); strategy: crop first (keeps the real distribution), pad when too small.
    Returns: (latent_tensor[C, target_h, target_w], target[num_classes], mask[num_classes])
    """
    def __init__(self, manifest, num_classes, target_hw=(128, 128),
                 data_aug=True, is_train=True,
                 latent_mean=None, latent_std=None,
                 model_name="z-image-turbo"):
        self.manifest = manifest
        self.num_classes = num_classes
        self.target_h, self.target_w = target_hw
        self.data_aug = data_aug and is_train
        self.is_train = is_train
        self.model_name = model_name
        # per-channel normalize stats
        if latent_mean is not None and latent_std is not None:
            self.latent_mean = latent_mean.detach().float().view(-1, 1, 1)
            self.latent_std = latent_std.detach().float().view(-1, 1, 1).clamp(min=1e-6)
        else:
            self.latent_mean = None
            self.latent_std = None

    def __len__(self):
        return len(self.manifest)

    def _resize_and_crop(self, latent):
        """Resize a latent [C,H,W] to a fixed (target_h, target_w).
        Strategy: resize the short side to target -> proportional scaling -> crop the long side.
          1. scale = max(target_h/H, target_w/W)
          2. if scale != 1: bilinear resize so both sides >= target
          3. crop to (target_h, target_w); random crop in train, center crop in val
        """
        C, h, w = latent.shape
        th, tw = self.target_h, self.target_w

        # compute the scale so both sides >= target
        scale = max(th / h, tw / w)
        if abs(scale - 1.0) > 1e-4:
            new_h = max(th, int(round(h * scale)))
            new_w = max(tw, int(round(w * scale)))
            latent = F.interpolate(
                latent.unsqueeze(0), size=(new_h, new_w),
                mode="bilinear", align_corners=False,
            ).squeeze(0)

        # crop to the target size
        _, h2, w2 = latent.shape
        if h2 > th:
            top = random.randint(0, h2 - th) if self.is_train else (h2 - th) // 2
            latent = latent[:, top:top + th, :]
        if w2 > tw:
            left = random.randint(0, w2 - tw) if self.is_train else (w2 - tw) // 2
            latent = latent[:, :, left:left + tw]

        return latent

    @_skip_none_on_error
    def __getitem__(self, idx):
        item = self.manifest[idx]
        try:
            latent = torch.load(item["latent_path"], map_location="cpu", weights_only=True)
        except Exception:
            latent = torch.load(item["latent_path"], map_location="cpu", weights_only=False)

        # squeeze to [C, H, W] per the model layout
        latent = _squeeze_latent_to_3d(latent, self.model_name)

        # to float32 for training
        latent = latent.float()

        # NaN/Inf guard: skip corrupted latent files
        if torch.isnan(latent).any() or torch.isinf(latent).any():
            return None

        # per-channel normalize (same as ImageLatentDataset in train_image.py)
        if self.latent_mean is not None:
            latent = (latent - self.latent_mean) / self.latent_std

        # fixed size: short-side resize + crop
        latent = self._resize_and_crop(latent)

        # data augmentation (train + data_aug=True only)
        if self.data_aug:
            latent = self._augment(latent)

        # build target and mask
        target = torch.zeros(self.num_classes, dtype=torch.float32)
        mask = torch.zeros(self.num_classes, dtype=torch.float32)
        for cls_idx, conf in item["annotations"].items():
            target[cls_idx] = float(conf)
            mask[cls_idx] = 1.0

        return latent, target, mask


    def _augment(self, x):
        """Latent-space data augmentation (aligned with ImageLatentDataset._augment):
        1. Random HFlip
        2. Random crop + resize back (crop 90% area, resize to the original size)
        3. Gaussian noise (50% chance, sigma=0.01)
        4. Scale Jitter (×0.95~1.05)
        """
        # 1. hflip
        if random.random() > 0.5:
            x = x.flip(-1)
        # 2. random crop + resize back
        if x.shape[-2] >= 32 and x.shape[-1] >= 32:
            c, h, w = x.shape
            ch = max(16, int(h * 0.9))
            cw = max(16, int(w * 0.9))
            top = random.randint(0, h - ch)
            left = random.randint(0, w - cw)
            x = x[:, top:top + ch, left:left + cw]
            x = F.interpolate(x.unsqueeze(0), size=(h, w),
                              mode="bilinear", align_corners=False).squeeze(0)
        # 3. gaussian noise
        if random.random() > 0.5:
            x = x + 0.01 * torch.randn_like(x)
        # 4. scale jitter
        return x * random.uniform(0.95, 1.05)


def fixed_collate_fn(batch):
    """Fixed-size collate: filter None then stack"""
    batch = [b for b in batch if b is not None]
    if not batch:
        return None, None, None
    latents, targets, masks = zip(*batch)
    return torch.stack(latents), torch.stack(targets), torch.stack(masks)


class OpenImagesImageClassifyDataset(Dataset):
    """OpenImages RGB image multi-label classification dataset.
    Loads raw RGB images (not latents), for the image-level pretraining control experiment.
    Returns: (image_tensor[3, image_size, image_size], target[num_classes], mask[num_classes])
    """
    def __init__(self, manifest, images_dir, num_classes,
                 image_size=224, data_aug=True, is_train=True):
        self.manifest = manifest
        self.images_dir = images_dir
        self.num_classes = num_classes
        self.image_size = image_size
        self.data_aug = data_aug and is_train
        self.is_train = is_train
        # ImageNet normalization params
        self.image_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.image_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __len__(self):
        return len(self.manifest)

    @_skip_none_on_error
    def __getitem__(self, idx):
        item = self.manifest[idx]
        image_id = item["image_id"]

        # load the image
        img_path = None
        for ext in [".jpg", ".png", ".jpeg"]:
            candidate = os.path.join(self.images_dir, f"{image_id}{ext}")
            if os.path.exists(candidate):
                img_path = candidate
                break
        if img_path is None:
            return None

        try:
            from PIL import Image
            img = Image.open(img_path).convert("RGB")
        except Exception:
            return None

        # resize + data augmentation
        if self.data_aug:
            # RandomResizedCrop + HFlip
            import torchvision.transforms as T
            transform = T.Compose([
                T.RandomResizedCrop(self.image_size, scale=(0.7, 1.0)),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
            ])
        else:
            import torchvision.transforms as T
            transform = T.Compose([
                T.Resize(int(self.image_size * 256 / 224)),
                T.CenterCrop(self.image_size),
                T.ToTensor(),
            ])
        img_tensor = transform(img)  # [3, image_size, image_size]

        # ImageNet normalization
        img_tensor = (img_tensor - self.image_mean) / self.image_std

        # build target and mask
        target = torch.zeros(self.num_classes, dtype=torch.float32)
        mask = torch.zeros(self.num_classes, dtype=torch.float32)
        for cls_idx, conf in item["annotations"].items():
            target[cls_idx] = float(conf)
            mask[cls_idx] = 1.0

        return img_tensor, target, mask


def distill_collate_fn(batch):
    """Distillation dataset collate: filter None then stack"""
    batch = [b for b in batch if b is not None]
    if not batch:
        return None, None
    latents, images = zip(*batch)
    return torch.stack(latents), torch.stack(images)


class OpenImagesDistillDataset(Dataset):
    """Dataset for feature distillation: each sample returns (latent_tensor, image_tensor).
    - latent: .pth file -> [C, target_h, target_w] (same preprocessing as OpenImagesLatentDataset)
    - image: the matching .jpg/.png -> resize + ImageNet normalize -> [3, image_size, image_size]
    Paired by file name (latent name minus .pth = image name minus .jpg/.png).

    Two latent directory layouts are supported:
    - flat mode (num_steps=0): manifest carries "latent_path"; load directly
    - step mode (num_steps>0): manifest carries "latent_root" + "image_id",
      random step in training, fixed_step in eval (aligned with finetune)

    Args:
        manifest: [{"image_id": ..., "latent_path"/"latent_root": ...}, ...]
        images_dir: image directory path
        target_hw: latent output size [H, W]
        image_size: teacher image input size (square)
        image_mean/image_std: ImageNet normalization params
        data_aug: whether to augment the latent
        model_name: model name matching the latent layout
        num_steps: number of step subdirs (0=flat, >0=step mode)
        step_mode: "random" | "fixed"
        fixed_step: the step used when step_mode="fixed" (for eval)
    """
    def __init__(self, manifest, images_dir, target_hw=(128, 128),
                 image_size=224, image_mean=None, image_std=None,
                 data_aug=True, is_train=True,
                 latent_mean=None, latent_std=None,
                 model_name="z-image-turbo",
                 num_steps=0, step_mode="random", fixed_step=3):
        self.manifest = manifest
        self.images_dir = images_dir
        self.target_h, self.target_w = target_hw
        self.image_size = image_size
        self.image_mean = torch.tensor(image_mean or [0.485, 0.456, 0.406]).view(3, 1, 1)
        self.image_std = torch.tensor(image_std or [0.229, 0.224, 0.225]).view(3, 1, 1)
        self.data_aug = data_aug and is_train
        self.is_train = is_train
        self.model_name = model_name
        # step-based loading (aligned with the finetune ImageLatentDataset)
        self.num_steps = int(num_steps)
        self.step_mode = str(step_mode)
        self.fixed_step = int(fixed_step)
        # per-channel latent normalize stats
        if latent_mean is not None and latent_std is not None:
            self.latent_mean = latent_mean.detach().float().view(-1, 1, 1)
            self.latent_std = latent_std.detach().float().view(-1, 1, 1).clamp(min=1e-6)
        else:
            self.latent_mean = None
            self.latent_std = None

    def __len__(self):
        return len(self.manifest)

    def _resize_and_crop(self, latent):
        """Resize a latent [C,H,W] to a fixed (target_h, target_w)."""
        C, h, w = latent.shape
        th, tw = self.target_h, self.target_w
        scale = max(th / h, tw / w)
        if abs(scale - 1.0) > 1e-4:
            new_h = max(th, int(round(h * scale)))
            new_w = max(tw, int(round(w * scale)))
            latent = F.interpolate(
                latent.unsqueeze(0), size=(new_h, new_w),
                mode="bilinear", align_corners=False,
            ).squeeze(0)
        _, h2, w2 = latent.shape
        if h2 > th:
            top = random.randint(0, h2 - th) if self.is_train else (h2 - th) // 2
            latent = latent[:, top:top + th, :]
        if w2 > tw:
            left = random.randint(0, w2 - tw) if self.is_train else (w2 - tw) // 2
            latent = latent[:, :, left:left + tw]
        return latent

    def _load_and_preprocess_image(self, image_id):
        """Load an RGB image with the standard ImageNet preprocessing."""
        from PIL import Image
        import torchvision.transforms.functional as TF

        img_path = None
        for ext in [".jpg", ".png", ".jpeg"]:
            candidate = os.path.join(self.images_dir, f"{image_id}{ext}")
            if os.path.exists(candidate):
                img_path = candidate
                break
        if img_path is None:
            return None

        img = Image.open(img_path).convert("RGB")
        # Resize to square (stretch)
        img = TF.resize(img, [self.image_size, self.image_size])
        # To tensor [0, 1]
        img_tensor = TF.to_tensor(img)  # [3, H, W]
        # ImageNet normalize
        img_tensor = (img_tensor - self.image_mean) / self.image_std
        return img_tensor

    def _augment_latent(self, x):
        """Latent data augmentation, identical to OpenImagesLatentDataset._augment."""
        if random.random() > 0.5:
            x = x.flip(-1)
        if x.shape[-2] >= 32 and x.shape[-1] >= 32:
            c, h, w = x.shape
            ch = max(16, int(h * 0.9))
            cw = max(16, int(w * 0.9))
            top = random.randint(0, h - ch)
            left = random.randint(0, w - cw)
            x = x[:, top:top + ch, left:left + cw]
            x = F.interpolate(x.unsqueeze(0), size=(h, w),
                              mode="bilinear", align_corners=False).squeeze(0)
        if random.random() > 0.5:
            x = x + 0.01 * torch.randn_like(x)
        return x * random.uniform(0.95, 1.05)

    def _pick_step(self):
        """Aligned with the finetune ImageLatentDataset._pick_step."""
        if self.step_mode == "random":
            return random.randrange(self.num_steps)
        return self.fixed_step

    @_skip_none_on_error
    def __getitem__(self, idx):
        item = self.manifest[idx]
        image_id = item["image_id"]

        # --- load the latent ---
        if self.num_steps > 0:
            # step mode: latent_root/{step}/{image_id}.pth
            step = self._pick_step()
            latent_path = os.path.join(item["latent_root"], str(step), f"{image_id}.pth")
        else:
            # flat mode: use latent_path directly
            latent_path = item["latent_path"]
        if not os.path.exists(latent_path):
            return None
        try:
            latent = torch.load(latent_path, map_location="cpu", weights_only=True)
        except Exception:
            latent = torch.load(latent_path, map_location="cpu", weights_only=False)

        latent = _squeeze_latent_to_3d(latent, self.model_name)
        latent = latent.float()

        if torch.isnan(latent).any() or torch.isinf(latent).any():
            return None

        # per-channel normalize
        if self.latent_mean is not None:
            latent = (latent - self.latent_mean) / self.latent_std

        # fixed size
        latent = self._resize_and_crop(latent)

        # data augmentation
        if self.data_aug:
            latent = self._augment_latent(latent)

        # --- load the matching image ---
        image_tensor = self._load_and_preprocess_image(image_id)
        if image_tensor is None:
            return None

        return latent, image_tensor


class OpenImagesDistillClassifyDataset(Dataset):
    """End-to-end distill+classify joint dataset: each sample returns (latent, image, target, mask).
    Combines OpenImagesDistillDataset (providing latent+image pairs) and
    OpenImagesLatentDataset (providing classification labels).

    For pretrain_mode="distill_classify":
      - latent -> student feature extraction + the classification head
      - image -> teacher feature extraction (distillation supervision)
      - target/mask -> classification loss
    """
    def __init__(self, manifest, images_dir, num_classes, target_hw=(224, 224),
                 image_size=224, image_mean=None, image_std=None,
                 data_aug=True, is_train=True,
                 latent_mean=None, latent_std=None,
                 model_name="z-image-turbo"):
        self.manifest = manifest
        self.images_dir = images_dir
        self.num_classes = num_classes
        self.target_h, self.target_w = target_hw
        self.image_size = image_size
        self.image_mean = torch.tensor(image_mean or [0.485, 0.456, 0.406]).view(3, 1, 1)
        self.image_std = torch.tensor(image_std or [0.229, 0.224, 0.225]).view(3, 1, 1)
        self.data_aug = data_aug and is_train
        self.is_train = is_train
        self.model_name = model_name
        if latent_mean is not None and latent_std is not None:
            self.latent_mean = latent_mean.detach().float().view(-1, 1, 1)
            self.latent_std = latent_std.detach().float().view(-1, 1, 1).clamp(min=1e-6)
        else:
            self.latent_mean = None
            self.latent_std = None

    def __len__(self):
        return len(self.manifest)

    def _resize_and_crop(self, latent):
        C, h, w = latent.shape
        th, tw = self.target_h, self.target_w
        scale = max(th / h, tw / w)
        if abs(scale - 1.0) > 1e-4:
            new_h = max(th, int(round(h * scale)))
            new_w = max(tw, int(round(w * scale)))
            latent = F.interpolate(
                latent.unsqueeze(0), size=(new_h, new_w),
                mode="bilinear", align_corners=False,
            ).squeeze(0)
        _, h2, w2 = latent.shape
        if h2 > th:
            top = random.randint(0, h2 - th) if self.is_train else (h2 - th) // 2
            latent = latent[:, top:top + th, :]
        if w2 > tw:
            left = random.randint(0, w2 - tw) if self.is_train else (w2 - tw) // 2
            latent = latent[:, :, left:left + tw]
        return latent

    def _load_and_preprocess_image(self, image_id):
        from PIL import Image
        import torchvision.transforms.functional as TF
        img_path = None
        for ext in [".jpg", ".png", ".jpeg"]:
            candidate = os.path.join(self.images_dir, f"{image_id}{ext}")
            if os.path.exists(candidate):
                img_path = candidate
                break
        if img_path is None:
            return None
        img = Image.open(img_path).convert("RGB")
        img = TF.resize(img, [self.image_size, self.image_size])
        img_tensor = TF.to_tensor(img)
        img_tensor = (img_tensor - self.image_mean) / self.image_std
        return img_tensor

    def _augment_latent(self, x):
        if random.random() > 0.5:
            x = x.flip(-1)
        if x.shape[-2] >= 32 and x.shape[-1] >= 32:
            c, h, w = x.shape
            ch = max(16, int(h * 0.9))
            cw = max(16, int(w * 0.9))
            top = random.randint(0, h - ch)
            left = random.randint(0, w - cw)
            x = x[:, top:top + ch, left:left + cw]
            x = F.interpolate(x.unsqueeze(0), size=(h, w),
                              mode="bilinear", align_corners=False).squeeze(0)
        if random.random() > 0.5:
            x = x + 0.01 * torch.randn_like(x)
        return x * random.uniform(0.95, 1.05)

    @_skip_none_on_error
    def __getitem__(self, idx):
        item = self.manifest[idx]
        # load the latent
        try:
            latent = torch.load(item["latent_path"], map_location="cpu", weights_only=True)
        except Exception:
            latent = torch.load(item["latent_path"], map_location="cpu", weights_only=False)
        latent = _squeeze_latent_to_3d(latent, self.model_name)
        latent = latent.float()
        if torch.isnan(latent).any() or torch.isinf(latent).any():
            return None
        if self.latent_mean is not None:
            latent = (latent - self.latent_mean) / self.latent_std
        latent = self._resize_and_crop(latent)
        if self.data_aug:
            latent = self._augment_latent(latent)

        # load the image (for the teacher)
        image_id = item["image_id"]
        image_tensor = self._load_and_preprocess_image(image_id)
        if image_tensor is None:
            return None

        # build the classification labels
        target = torch.zeros(self.num_classes, dtype=torch.float32)
        mask = torch.zeros(self.num_classes, dtype=torch.float32)
        for cls_idx, conf in item.get("annotations", {}).items():
            target[cls_idx] = float(conf)
            mask[cls_idx] = 1.0

        return latent, image_tensor, target, mask


def distill_classify_collate_fn(batch):
    """End-to-end distill+classify collate: filter None then stack the 4-tuples"""
    batch = [b for b in batch if b is not None]
    if not batch:
        return None, None, None, None
    latents, images, targets, masks = zip(*batch)
    return torch.stack(latents), torch.stack(images), torch.stack(targets), torch.stack(masks)

# ===========================================================================
# 2. Model building (self-contained; no models_zoo dependency)
# ===========================================================================
_TVM_WEIGHTS = {
    "convnext_tiny":   ("convnext_tiny",   "ConvNeXt_Tiny_Weights"),
    "convnext_small":  ("convnext_small",  "ConvNeXt_Small_Weights"),
    "convnext_base":   ("convnext_base",   "ConvNeXt_Base_Weights"),
    "convnext_large":  ("convnext_large",  "ConvNeXt_Large_Weights"),
    "vit_b_16":        ("vit_b_16",        "ViT_B_16_Weights"),
    "vit_l_16":        ("vit_l_16",        "ViT_L_16_Weights"),
    "swin_v2_t":       ("swin_v2_t",       "Swin_V2_T_Weights"),
    "swin_v2_s":       ("swin_v2_s",       "Swin_V2_S_Weights"),
    "swin_v2_b":       ("swin_v2_b",       "Swin_V2_B_Weights"),
    "resnet18":        ("resnet18",        "ResNet18_Weights"),
    "resnet50":        ("resnet50",        "ResNet50_Weights"),
    "resnet101":       ("resnet101",       "ResNet101_Weights"),
    "resnet152":       ("resnet152",       "ResNet152_Weights"),
}


def expand_conv_weight(old_weight, target_in_chans):
    """Expand conv weights [out, 3, kH, kW] to [out, target_in_chans, kH, kW].
    The first 3 channels reuse the original weights; extra channels are filled with the mean of the first 3.
    """
    out_c, in_c, kh, kw = old_weight.shape
    if in_c == target_in_chans:
        return old_weight.clone()
    new_weight = torch.zeros(out_c, target_in_chans, kh, kw, dtype=old_weight.dtype)
    new_weight[:, :in_c] = old_weight
    mean_weight = old_weight.mean(dim=1, keepdim=True)  # [out, 1, kH, kW]
    for i in range(in_c, target_in_chans):
        new_weight[:, i:i+1] = mean_weight
    return new_weight


def build_pretrain_backbone(backbone_name, in_chans, pretrained=True,
                            stem_type="single", stem_channels=None, stem_init="expand_in1k"):
    """Build the backbone for pretraining.
    - CNN (convnext/resnet): stem re-channelled -> global avg pool -> feat_dim
    - ViT (vit_b_16/vit_l_16): torchvision IN1K_V1 weights, conv_proj channel-expanded for latents

    Args:
        stem_type: "single" | "progressive" | "original"
            - single: one Conv2d(in_chans -> backbone_stem_out) layer, for latent inputs
            - progressive: progressive multi-layer stem [in_chans -> ... -> backbone_stem_out], for latent inputs
            - original: keep the backbone's original stem untouched (for standard RGB input with in_chans=3,
                        used by the classify_image control experiment)
        stem_channels: per-layer channel list in progressive mode, e.g. [16, 32, 64, 128]. Progressive only.
        stem_init: "expand_in1k" | "trunc_normal" | "kaiming" (single/progressive modes only)
    """
    # ---- backbone loading (ConvNeXt/ViT/ResNet all through torchvision) ----
    if backbone_name not in _TVM_WEIGHTS:
        raise ValueError(f"unsupported {backbone_name}; options: {list(_TVM_WEIGHTS.keys())}")

    ctor_name, weights_attr = _TVM_WEIGHTS[backbone_name]
    weights = getattr(tvm, weights_attr).IMAGENET1K_V1 if pretrained else None
    m = getattr(tvm, ctor_name)(weights=weights)

    if stem_type == "progressive":
        # --- progressive stem ---
        from models_zoo import ProgressiveStem
        if stem_channels is None:
            stem_channels = [in_chans, 32, 64, 128]
        assert stem_channels[0] == in_chans, (
            f"stem_channels[0]={stem_channels[0]} must equal in_chans={in_chans}")
        prog_stem = ProgressiveStem(stem_channels, stem_init=stem_init)

        family = get_backbone_family(backbone_name)
        if family == "convnext":
            orig_out_ch = m.features[0][0].out_channels
            assert stem_channels[-1] == orig_out_ch, (
                f"stem_channels[-1]={stem_channels[-1]} must equal the backbone stem output channels {orig_out_ch}")
            m.features[0] = prog_stem
            feat_dim = m.classifier[2].in_features
            m.classifier[2] = nn.Identity()
        elif family == "swin_v2":
            orig_out_ch = m.features[0][0].out_channels
            assert stem_channels[-1] == orig_out_ch, (
                f"stem_channels[-1]={stem_channels[-1]} must equal the backbone stem output channels {orig_out_ch}")
            m.features[0] = prog_stem
            feat_dim = m.head.in_features
            m.head = nn.Identity()
        elif family == "vit":
            raise ValueError("ViT does not support progressive stem; use stem_type='single'")
        else:  # resnet
            orig_out_ch = m.conv1.out_channels
            assert stem_channels[-1] == orig_out_ch, (
                f"stem_channels[-1]={stem_channels[-1]} must equal the backbone conv1 output channels {orig_out_ch}")
            m.conv1 = prog_stem
            m.bn1 = nn.Identity()
            m.relu = nn.Identity()
            feat_dim = m.fc.in_features
            m.fc = nn.Identity()

    elif stem_type == "single":
        # --- single stem (original logic) ---
        family = get_backbone_family(backbone_name)
        if family == "convnext":
            old_stem = m.features[0][0]
            new_stem = nn.Conv2d(
                in_chans, old_stem.out_channels,
                kernel_size=old_stem.kernel_size, stride=old_stem.stride,
                padding=old_stem.padding, bias=(old_stem.bias is not None),
            )
            with torch.no_grad():
                if stem_init == "expand_in1k":
                    new_stem.weight.copy_(expand_conv_weight(old_stem.weight.data, in_chans))
                    if old_stem.bias is not None:
                        new_stem.bias.copy_(old_stem.bias.data)
                elif stem_init == "trunc_normal":
                    nn.init.trunc_normal_(new_stem.weight, std=0.02)
                    if new_stem.bias is not None:
                        nn.init.zeros_(new_stem.bias)
                elif stem_init == "kaiming":
                    nn.init.kaiming_normal_(new_stem.weight, mode="fan_out", nonlinearity="relu")
                    if new_stem.bias is not None:
                        nn.init.zeros_(new_stem.bias)
                else:
                    raise ValueError(f"unsupported stem_init={stem_init!r}")
            m.features[0][0] = new_stem
            feat_dim = m.classifier[2].in_features
            m.classifier[2] = nn.Identity()
        elif family == "swin_v2":
            old_stem = m.features[0][0]
            new_stem = nn.Conv2d(
                in_chans, old_stem.out_channels,
                kernel_size=old_stem.kernel_size, stride=old_stem.stride,
                padding=old_stem.padding, bias=(old_stem.bias is not None),
            )
            with torch.no_grad():
                if stem_init == "expand_in1k":
                    new_stem.weight.copy_(expand_conv_weight(old_stem.weight.data, in_chans))
                    if old_stem.bias is not None:
                        new_stem.bias.copy_(old_stem.bias.data)
                elif stem_init == "trunc_normal":
                    nn.init.trunc_normal_(new_stem.weight, std=0.02)
                    if new_stem.bias is not None:
                        nn.init.zeros_(new_stem.bias)
                elif stem_init == "kaiming":
                    nn.init.kaiming_normal_(new_stem.weight, mode="fan_out", nonlinearity="relu")
                    if new_stem.bias is not None:
                        nn.init.zeros_(new_stem.bias)
                else:
                    raise ValueError(f"unsupported stem_init={stem_init!r}")
            m.features[0][0] = new_stem
            feat_dim = m.head.in_features
            m.head = nn.Identity()
        elif family == "vit":
            old_conv = m.conv_proj
            new_conv = nn.Conv2d(
                in_chans, old_conv.out_channels,
                kernel_size=old_conv.kernel_size, stride=old_conv.stride,
                padding=old_conv.padding, bias=(old_conv.bias is not None),
            )
            with torch.no_grad():
                if stem_init == "expand_in1k":
                    new_conv.weight.copy_(expand_conv_weight(old_conv.weight.data, in_chans))
                    if old_conv.bias is not None:
                        new_conv.bias.copy_(old_conv.bias.data)
                elif stem_init == "trunc_normal":
                    nn.init.trunc_normal_(new_conv.weight, std=0.02)
                    if new_conv.bias is not None:
                        nn.init.zeros_(new_conv.bias)
                elif stem_init == "kaiming":
                    nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")
                    if new_conv.bias is not None:
                        nn.init.zeros_(new_conv.bias)
                else:
                    raise ValueError(f"unsupported stem_init={stem_init!r}")
            m.conv_proj = new_conv
            feat_dim = m.heads[0].in_features
            m.heads = nn.Identity()
        else:  # resnet
            old_stem = m.conv1
            new_stem = nn.Conv2d(
                in_chans, old_stem.out_channels,
                kernel_size=old_stem.kernel_size, stride=old_stem.stride,
                padding=old_stem.padding, bias=(old_stem.bias is not None),
            )
            with torch.no_grad():
                if stem_init == "expand_in1k":
                    new_stem.weight.copy_(expand_conv_weight(old_stem.weight.data, in_chans))
                    if old_stem.bias is not None:
                        new_stem.bias.copy_(old_stem.bias.data)
                elif stem_init == "trunc_normal":
                    nn.init.trunc_normal_(new_stem.weight, std=0.02)
                    if new_stem.bias is not None:
                        nn.init.zeros_(new_stem.bias)
                elif stem_init == "kaiming":
                    nn.init.kaiming_normal_(new_stem.weight, mode="fan_out", nonlinearity="relu")
                    if new_stem.bias is not None:
                        nn.init.zeros_(new_stem.bias)
                else:
                    raise ValueError(f"unsupported stem_init={stem_init!r}")
            m.conv1 = new_stem
            feat_dim = m.fc.in_features
            m.fc = nn.Identity()
    elif stem_type == "original":
        # --- keep the backbone's original stem (for standard RGB input with in_chans=3) ---
        family = get_backbone_family(backbone_name)
        if family == "convnext":
            feat_dim = m.classifier[2].in_features
            m.classifier[2] = nn.Identity()
        elif family == "swin_v2":
            feat_dim = m.head.in_features
            m.head = nn.Identity()
        elif family == "vit":
            feat_dim = m.heads[0].in_features
            m.heads = nn.Identity()
        else:  # resnet
            feat_dim = m.fc.in_features
            m.fc = nn.Identity()
    else:
        raise ValueError(f"unsupported stem_type={stem_type!r}; options: 'single' / 'progressive' / 'original'")

    print(f"  [Pretrain Backbone] {backbone_name}, stem_type={stem_type}, stem_init={stem_init}, "
          f"in_chans={in_chans}, feat_dim={feat_dim}")
    return m, feat_dim


class PretrainModel(nn.Module):
    """Pretrain model: backbone + dropout + multi-label classification head.
    Supports parameterized stem_type="single" / "progressive".
    """
    def __init__(self, backbone_name, in_chans, num_classes, pretrained=True, dropout=0.1,
                 stem_type="single", stem_channels=None, stem_init="expand_in1k"):
        super().__init__()
        self.backbone, self.feat_dim = build_pretrain_backbone(
            backbone_name, in_chans, pretrained,
            stem_type=stem_type, stem_channels=stem_channels, stem_init=stem_init,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(self.feat_dim, num_classes)

    def forward(self, x):
        feat = self.backbone(x)          # [B, feat_dim]
        feat = self.dropout(feat)
        logits = self.head(feat)          # [B, num_classes]
        return logits

    def extract_feat(self, x):
        """Extract features only, no head (for export in distill mode)."""
        return self.backbone(x)

    def extract_stage_features(self, x):
        """Extract intermediate per-stage features (for stage-wise distillation).
        Distinguishes ConvNeXt / ViT automatically; returns dict: {"stage1": ..., ..., "gap": ...}
        """
        from models_zoo import extract_stage_features
        return extract_stage_features(self.backbone, x)

    def get_backbone_state_dict(self):
        """Export the backbone weights (key format: backbone.xxx) for downstream LatentCNNMultiTask loading."""
        return {k: v for k, v in self.state_dict().items() if k.startswith("backbone.")}


# ===========================================================================
# 3. Losses and evaluation
# ===========================================================================
def masked_bce_loss(logits, targets, mask):
    """Masked BCEWithLogitsLoss: computed only where mask=1"""
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    loss = loss * mask
    return loss.sum() / mask.sum().clamp(min=1)


def compute_mAP(all_targets, all_probs, all_masks):
    """Compute masked mAP: AP only over labeled classes"""
    # all_targets, all_probs, all_masks: [N, C]
    num_classes = all_targets.shape[1]
    aps = []
    for c in range(num_classes):
        mask_c = all_masks[:, c] > 0.5
        if mask_c.sum() < 2:
            continue
        t_c = all_targets[mask_c, c]
        p_c = all_probs[mask_c, c]
        if t_c.sum() == 0 or t_c.sum() == len(t_c):
            continue  # all-positive or all-negative; skip
        # simple AP computation
        sorted_idx = np.argsort(-p_c)
        t_sorted = t_c[sorted_idx]
        tp_cumsum = np.cumsum(t_sorted)
        precision_at_k = tp_cumsum / np.arange(1, len(t_sorted) + 1)
        ap = (precision_at_k * t_sorted).sum() / t_sorted.sum()
        aps.append(ap)
    return np.mean(aps) if aps else 0.0


def compute_overall_accuracy(all_targets, all_preds, all_masks):
    """Overall accuracy: over all labeled (sample, class) pairs"""
    mask_flat = all_masks.flatten() > 0.5
    t_flat = all_targets.flatten()[mask_flat]
    p_flat = all_preds.flatten()[mask_flat]
    correct = (t_flat == p_flat).sum()
    total = len(t_flat)
    return correct / max(1, total)


# ===========================================================================
# 4. Training and evaluation loops (iteration-based)
# ===========================================================================
def infinite_loader(loader):
    """Infinitely cycling DataLoader (reshuffled each round)"""
    while True:
        for batch in loader:
            yield batch


@torch.no_grad()
def evaluate(model, loader, device, global_iter, split_name="Val"):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_targets = []
    all_probs = []
    all_masks = []

    pbar = tqdm(loader, desc=f"Iter {global_iter} [{split_name}]", leave=False)
    for batch in pbar:
        if batch[0] is None:
            continue
        latents, targets, masks = batch
        latents = latents.to(device)
        targets = targets.to(device)
        masks = masks.to(device)

        logits = model(latents)
        loss = masked_bce_loss(logits, targets, masks)

        bs = latents.size(0)
        total_loss += loss.item() * bs
        total_samples += bs

        probs = torch.sigmoid(logits).cpu().numpy()
        all_targets.append(targets.cpu().numpy())
        all_probs.append(probs)
        all_masks.append(masks.cpu().numpy())

    avg_loss = total_loss / max(1, total_samples)
    all_targets = np.concatenate(all_targets, axis=0)
    all_probs = np.concatenate(all_probs, axis=0)
    all_masks = np.concatenate(all_masks, axis=0)

    all_preds = (all_probs > 0.5).astype(np.float32)
    mAP = compute_mAP(all_targets, all_probs, all_masks)
    acc = compute_overall_accuracy(all_targets, all_preds, all_masks)

    print(f"  [{split_name}] loss={avg_loss:.4f}, mAP={mAP:.4f}, acc={acc:.4f}")
    return avg_loss, mAP, acc


# ===========================================================================
# 5. Plotting
# ===========================================================================
def plot_training_curves(history, save_path):
    """Plot train/val curves (x-axis is iteration)"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    iters = history["iter"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Loss
    ax = axes[0, 0]
    ax.plot(iters, history["train_loss"], "b-o", label="Train Loss", markersize=3)
    ax.plot(iters, history["val_loss"], "r-o", label="Val Loss", markersize=3)
    ax.set_title("Loss")
    ax.set_xlabel("Iteration")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # mAP
    ax = axes[0, 1]
    ax.plot(iters, history["val_mAP"], "g-o", label="Val mAP", markersize=3)
    ax.set_title("Validation mAP")
    ax.set_xlabel("Iteration")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Accuracy
    ax = axes[1, 0]
    ax.plot(iters, history["train_acc"], "b-o", label="Train Acc", markersize=3)
    ax.plot(iters, history["val_acc"], "r-o", label="Val Acc", markersize=3)
    ax.set_title("Overall Accuracy")
    ax.set_xlabel("Iteration")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # LR
    ax = axes[1, 1]
    if "lr" in history and history["lr"]:
        ax.plot(iters, history["lr"], "m-", label="Learning Rate")
        ax.set_title("Learning Rate")
        ax.set_xlabel("Iteration")
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# ===========================================================================
# 6. Checkpoint saving
# ===========================================================================
def save_checkpoint(model, optimizer, global_iter, history, ckpt_dir, tag="latest"):
    """Save the full checkpoint + backbone-only checkpoint into the ckpts/ subdirectory"""
    ckpts_dir = os.path.join(ckpt_dir, "ckpts")
    os.makedirs(ckpts_dir, exist_ok=True)

    # full model
    full_path = os.path.join(ckpts_dir, f"model_{tag}.pth")
    torch.save({
        "global_iter": global_iter,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "history": history,
    }, full_path)

    # backbone-only (loaded downstream by train_image.py)
    backbone_path = os.path.join(ckpts_dir, f"backbone_{tag}.pth")
    torch.save({
        "global_iter": global_iter,
        "state_dict": model.get_backbone_state_dict(),
    }, backbone_path)

    return full_path, backbone_path


# ===========================================================================
# 7. Main training flows
# ===========================================================================
def build_ckpt_dir(config):
    now = datetime.datetime.now()
    now_str = now.strftime("%Y%m%d_%H%M%S")
    today_str = now.strftime("%Y%m%d")
    backbone = config["backbone"]
    in_chans = config["in_chans"]
    target_hw = config.get("target_hw", [128, 128])
    model_name = config.get("model", "z-image-turbo")  # which model's latents

    # naming convention: with exp_tag given, use the concise "{timestamp}_{exp_tag}" format,
    # otherwise the full parameter-concatenation format
    exp_tag = config.get("exp_tag", "")
    if exp_tag:
        dir_name = f"{now_str}_{exp_tag}"
    else:
        dir_name = (
            f"{now_str}_pretrain-openimages-{backbone}-{in_chans}ch"
            f"_hw{target_hw[0]}x{target_hw[1]}"
            f"_bs{config['bs']}_lr{config['lr']}"
            f"_seed{config['seed']}"
        )
    return os.path.join(
        config["ckpt_root"], today_str, model_name, "pretrain_latent",
        dir_name,
    )


def run_pretrain(config):
    """Main entry: dispatch to the matching pretraining flow by pretrain_mode.

    pretrain_mode values:
        - "distill":          latent feature distillation pretraining (teacher RGB -> student latent)
        - "classify":         latent multi-label classification pretraining (OpenImages latents)
        - "classify_image":   image multi-label classification pretraining (OpenImages RGB images, control experiment)
        - "distill_classify": end-to-end distill+classify joint training (L_cls + lambda*L_distill)
    """
    pretrain_mode = config.get("pretrain_mode", "classify")
    if pretrain_mode == "distill":
        run_distill_pretrain(config)
    elif pretrain_mode == "classify":
        run_classify_pretrain(config)
    elif pretrain_mode == "classify_image":
        run_classify_image_pretrain(config)
    elif pretrain_mode == "distill_classify":
        run_distill_classify_pretrain(config)
    else:
        raise ValueError(f"unsupported pretrain_mode={pretrain_mode!r}; "
                         f"options: 'classify' / 'distill' / 'classify_image' / 'distill_classify'")


# ===========================================================================
# Distillation pretraining monitoring plots
# ===========================================================================
def plot_distill_history(history, save_dir):
    """Plot the distillation pretraining monitoring curves and save them to save_dir/plots/.

    Produces 3 plots (file names carry an iter suffix to avoid overwrites):
      1. loss_curves_iter{N}.png       — train_loss + val_loss (total loss) + LR
      2. loss_components_iter{N}.png   — per-layer loss components (gap, stage1, stage2, ...)
      3. cosine_similarity_iter{N}.png — per-layer cosine similarity curves
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    iters = history.get("iter", [])
    if len(iters) < 2:
        return  # too few data points; skip plotting

    current_iter = iters[-1]  # latest iter, used as the file-name suffix
    plots_dir = os.path.join(save_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # --- plot 1: loss curves (train + val) + LR ---
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(iters, history["train_loss"], "b-", linewidth=1.5, label="train_loss")
    if history.get("val_loss"):
        ax1.plot(iters, history["val_loss"], "r-", linewidth=1.5, label="val_loss")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Loss")
    ax1.set_title("Distillation Loss")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper left")
    # LR on the right axis
    if history.get("lr"):
        ax2 = ax1.twinx()
        ax2.plot(iters, history["lr"], "g--", linewidth=1, alpha=0.6, label="lr")
        ax2.set_ylabel("Learning Rate")
        ax2.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"loss_curves_iter{current_iter}.png"), dpi=150)
    plt.close()

    # --- plot 2: loss components ---
    loss_keys = [k for k in history if k.startswith("train_loss_") and history[k]]
    if loss_keys:
        fig, ax = plt.subplots(figsize=(10, 5))
        for k in sorted(loss_keys):
            label = k.replace("train_loss_", "")
            ax.plot(iters, history[k], linewidth=1.5, label=label)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss")
        ax.set_title("Loss Components (per stage)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"loss_components_iter{current_iter}.png"), dpi=150)
        plt.close()

    # --- plot 3: cosine similarity ---
    cos_keys = [k for k in history if k.startswith("val_cos_sim_") and history[k]]
    if cos_keys:
        fig, ax = plt.subplots(figsize=(10, 5))
        for k in sorted(cos_keys):
            label = k.replace("val_cos_sim_", "")
            ax.plot(iters, history[k], linewidth=1.5, marker=".", markersize=3, label=label)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Cosine Similarity")
        ax.set_title("Feature Cosine Similarity (per stage)")
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"cosine_similarity_iter{current_iter}.png"), dpi=150)
        plt.close()


def run_distill_pretrain(config):
    """Feature-distillation pretraining: align teacher (RGB image) -> student (latent) features.

    Two distillation modes are supported:
      - GAP-only: align only the final GAP features [B, 1024]  (distill_stages=[])
      - Stage-wise: align intermediate feature maps + GAP features  (distill_stages=[1,2,3,4])
        Requires teacher_image_size == target_hw so teacher/student per-stage resolutions align (not needed in GAP-pool mode).
    """
    from models_zoo import build_teacher_backbone, ProgressiveStem, extract_stage_features

    seed_everything(config["seed"])
    if config.get("torch_home"):
        os.environ["TORCH_HOME"] = str(config["torch_home"])
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")

    # ---- output directory ----
    ckpt_dir = build_ckpt_dir(config)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(os.path.join(ckpt_dir, "meta"), exist_ok=True)
    os.makedirs(os.path.join(ckpt_dir, "plots"), exist_ok=True)
    save_json(config, os.path.join(ckpt_dir, "meta", "config.json"))

    # logging
    log_path = os.path.join(ckpt_dir, "train.log")
    sys.stdout = Logger(log_path)
    print(f"[Config] pretrain_mode = distill")
    print(f"[Config] ckpt_dir = {ckpt_dir}")
    print(f"[Config] device = {device}")
    for k, v in config.items():
        print(f"  {k}: {v}")

    # ---- build the manifest (distillation needs no labels, just the latent file list) ----
    print("\n" + "=" * 60)
    print("Stage 1: build the data")
    print("=" * 60)
    latent_dir = config["latent_dir"]
    images_dir = config.get("images_dir")
    if images_dir is None:
        images_dir = os.path.join(os.path.dirname(latent_dir.rstrip("/")), "images")
    
    # step-based directory detection: if latent_dir holds only digit-named subdirs (0,1,2,...), treat it as step mode
    latent_num_steps = int(config.get("latent_num_steps", 0))
    latent_val_step = int(config.get("latent_val_step", 3))
    is_step_mode = latent_num_steps > 0
    
    if is_step_mode:
        # step mode: scan just one step dir for the image_id list (avoids Nx duplication)
        # prefer the val_step dir, since eval always needs it
        scan_step_dir = os.path.join(latent_dir, str(latent_val_step))
        if not os.path.isdir(scan_step_dir):
            # fallback: scan step 0
            scan_step_dir = os.path.join(latent_dir, "0")
        print(f"  step mode: num_steps={latent_num_steps}, val_step={latent_val_step}")
        print(f"  scanning directory: {scan_step_dir}")
    
        cache_path = os.path.join(latent_dir, "_latent_ids.cache.txt")
        if os.path.exists(cache_path):
            print(f"  reading cache: {cache_path}")
            with open(cache_path, "r") as f:
                image_ids = [line.strip() for line in f if line.strip()]
            print(f"  loaded {len(image_ids)} image_ids from cache")
        else:
            print("  scanning the latent directory (first run; the network storage may take minutes)...")
            sys.stdout.flush()
            pth_files = [f for f in os.listdir(scan_step_dir) if f.endswith(".pth")]
            image_ids = [os.path.splitext(f)[0] for f in pth_files]
            print(f"  found {len(image_ids)} .pth files")
            try:
                with open(cache_path, "w") as f:
                    f.write("\n".join(image_ids))
                print(f"  cache saved: {cache_path}")
            except OSError:
                print("  [WARN] cannot write the cache file")
    
        manifest = [{"image_id": iid, "latent_root": latent_dir} for iid in image_ids]
    else:
        # flat mode (OpenImages): scan .pth files directly
        cache_path = os.path.join(latent_dir, "_latent_files.cache.txt")
        if os.path.exists(cache_path):
            print(f"  reading cache: {cache_path}")
            with open(cache_path, "r") as f:
                lines = [line.strip() for line in f if line.strip()]
            latent_files = []
            for line in lines:
                full_path = os.path.join(latent_dir, line)
                fname = os.path.basename(line)
                latent_files.append((fname, full_path))
            print(f"  loaded {len(latent_files)} .pth files from cache")
        else:
            print("  scanning the latent directory (first run; the network storage may take minutes)...")
            sys.stdout.flush()
            top_pth = [f for f in os.listdir(latent_dir) if f.endswith(".pth")]
            latent_files = [(f, os.path.join(latent_dir, f)) for f in top_pth]
            cache_lines = top_pth
            print(f"  found {len(latent_files)} .pth files")
            try:
                with open(cache_path, "w") as f:
                    f.write("\n".join(cache_lines))
                print(f"  cache saved: {cache_path}")
            except OSError:
                print("  [WARN] cannot write the cache file")
    
        manifest = [{"image_id": os.path.splitext(f)[0], "latent_path": p} for f, p in latent_files]
    print(f"  valid samples: {len(manifest)}")
    if not manifest:
        raise RuntimeError("no valid samples; check whether latent_dir is correct")

    # split train/val (a fixed val_size count)
    val_size = int(config.get("val_size", 10000))
    train_manifest, val_manifest = split_train_val(
        manifest, val_size=val_size, seed=config["seed"],
    )

    save_json({
        "train_count": len(train_manifest),
        "val_count": len(val_manifest),
        "total_pairs": len(manifest),
    }, os.path.join(ckpt_dir, "meta", "data_info.json"))

    # ---- Dataset & DataLoader ----
    target_hw = tuple(config.get("target_hw", [128, 128]))
    model_name = config.get("model", "z-image-turbo")

    # latent normalize
    latent_mean, latent_std = None, None
    if bool(config.get("latent_enable_normalize", False)):
        print("\n  Computing per-channel latent normalization stats ...")
        latent_mean, latent_std, latent_stats_info = compute_latent_channel_stats(
            train_manifest,
            num_samples=config.get("latent_stats_num_samples", 500),
            seed=config["seed"],
            model_name=model_name,
        )
        save_json(latent_stats_info, os.path.join(ckpt_dir, "meta", "latent_stats.json"))
    else:
        print("\n  [Latent Stats] latent_enable_normalize=False")

    teacher_image_size = config.get("teacher_image_size", 224)
    teacher_image_mean = config.get("teacher_image_mean", [0.485, 0.456, 0.406])
    teacher_image_std = config.get("teacher_image_std", [0.229, 0.224, 0.225])

    train_ds = OpenImagesDistillDataset(
        train_manifest, images_dir, target_hw=target_hw,
        image_size=teacher_image_size, image_mean=teacher_image_mean, image_std=teacher_image_std,
        data_aug=bool(config.get("data_aug", True)), is_train=True,
        latent_mean=latent_mean, latent_std=latent_std,
        model_name=model_name,
        num_steps=latent_num_steps, step_mode="random", fixed_step=latent_val_step,
    )
    val_ds = OpenImagesDistillDataset(
        val_manifest, images_dir, target_hw=target_hw,
        image_size=teacher_image_size, image_mean=teacher_image_mean, image_std=teacher_image_std,
        data_aug=False, is_train=False,
        latent_mean=latent_mean, latent_std=latent_std,
        model_name=model_name,
        num_steps=latent_num_steps, step_mode="fixed", fixed_step=latent_val_step,
    )
    train_loader = DataLoader(
        train_ds, batch_size=config["bs"], shuffle=True,
        num_workers=config.get("nw", 4), collate_fn=distill_collate_fn,
        pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config["bs"], shuffle=False,
        num_workers=config.get("nw", 4), collate_fn=distill_collate_fn,
        pin_memory=True, drop_last=False,
    )
    print(f"  target_hw={target_hw}, train={len(train_ds)}, val={len(val_ds)}")

    # ---- teacher model (frozen) ----
    print("\n" + "=" * 60)
    print("Stage 2: build teacher & student")
    print("=" * 60)
    teacher_backbone_name = config.get("teacher_backbone", "convnext_base")
    teacher_ckpt_path = config.get("teacher_ckpt", "") or None
    teacher, teacher_feat_dim = build_teacher_backbone(
        backbone_name=teacher_backbone_name,
        pretrained=config.get("teacher_pretrained", True),
        ckpt_path=teacher_ckpt_path,
    )
    teacher = teacher.to(device)
    teacher.eval()
    print(f"  Teacher: {teacher_backbone_name}, feat_dim={teacher_feat_dim}, frozen=True")

    # ---- student model ----
    stem_type = config.get("stem_type", "single")
    stem_channels = config.get("stem_channels", None)
    stem_init = config.get("stem_init", "expand_in1k")

    student = PretrainModel(
        backbone_name=config["backbone"],
        in_chans=config["in_chans"],
        num_classes=1,  # distill mode does not use the head, but it still needs initializing
        pretrained=config.get("backbone_pretrained", True),
        dropout=config.get("dropout", 0.1),
        stem_type=stem_type, stem_channels=stem_channels, stem_init=stem_init,
    )
    student = student.to(device)

    # train_scope handling
    train_scope = config.get("train_scope", "full")
    if train_scope == "stem_only":
        # freeze the backbone blocks; train the stem only
        for n, p in student.named_parameters():
            p.requires_grad = False
        # unfreeze the stem
        if stem_type == "progressive":
            if get_backbone_family(config["backbone"]) in ("convnext", "swin_v2"):
                for p in student.backbone.features[0].parameters():
                    p.requires_grad = True
            elif get_backbone_family(config["backbone"]) == "vit":
                raise ValueError("ViT does not support progressive stem")
            else:
                for p in student.backbone.conv1.parameters():
                    p.requires_grad = True
        else:
            if get_backbone_family(config["backbone"]) in ("convnext", "swin_v2"):
                for p in student.backbone.features[0][0].parameters():
                    p.requires_grad = True
            elif get_backbone_family(config["backbone"]) == "vit":
                for p in student.backbone.conv_proj.parameters():
                    p.requires_grad = True
            else:
                for p in student.backbone.conv1.parameters():
                    p.requires_grad = True

    student_feat_dim = student.feat_dim
    total_params = sum(p.numel() for p in student.parameters())
    trainable_params = sum(p.numel() for p in student.parameters() if p.requires_grad)
    print(f"  Student: {config['backbone']}, stem_type={stem_type}, stem_init={stem_init}")
    print(f"  Student feat_dim={student_feat_dim}, train_scope={train_scope}")
    print(f"  Total params: {total_params:,}, Trainable: {trainable_params:,}")

    # feature-dim alignment layer (when teacher/student feat_dims differ)
    proj_layer = None
    if student_feat_dim != teacher_feat_dim:
        proj_layer = nn.Linear(student_feat_dim, teacher_feat_dim).to(device)
        print(f"  adding projection layer: {student_feat_dim} -> {teacher_feat_dim}")

    # ---- stage-wise distillation config ----
    # distill_stages: the intermediate layers to distill, e.g. [1,2,3,4]; empty = GAP final features only
    # with stage-wise, teacher_image_size should match target_hw for spatial alignment
    distill_stages = config.get("distill_stages", [])  # empty list = GAP only
    distill_stage_weight = float(config.get("distill_stage_weight", 1.0))  # total stage-loss weight
    distill_stage_pool = config.get("distill_stage_pool", "gap")  # "gap" | "pixel"
    enable_stage_distill = len(distill_stages) > 0

    if enable_stage_distill:
        print(f"  Stage-wise distill: stages={distill_stages}, weight={distill_stage_weight}, "
              f"pool={distill_stage_pool}")
        # cross-size stage-wise distillation guard: different backbone variants have different per-stage channels, causing dim mismatches
        if student_feat_dim != teacher_feat_dim:
            raise ValueError(
                f"stage-wise distillation does not support cross-size backbones! student feat_dim={student_feat_dim} "
                f"!= teacher feat_dim={teacher_feat_dim}. Different per-stage channels make the cosine_similarity dims mismatch."
                f"Use distill_stages=[] (GAP-only) for cross-size distillation, or use same-size backbones."
            )
        target_h = config.get("target_hw", [224, 224])[0]
        if teacher_image_size != target_h:
            print(f"  [WARN] stage-wise distillation recommends teacher_image_size={target_h} for spatial alignment; "
                  f"current teacher_image_size={teacher_image_size}")

    # ---- Loss ----
    distill_loss_type = config.get("distill_loss", "cosine")
    print(f"  Distill loss: {distill_loss_type}")

    def _compute_feat_loss(feat_s, feat_t):
        """Compute the distillation loss between two feature vectors / feature maps.
        feat_s, feat_t: [B, D] or [B, C, H, W]
        """
        if distill_loss_type == "cosine":
            if feat_s.ndim == 4:
                # feature map: cosine along the channel dim (each spatial position independently)
                b, c, h, w = feat_s.shape
                fs_flat = feat_s.permute(0, 2, 3, 1).reshape(-1, c)  # [B*H*W, C]
                ft_flat = feat_t.permute(0, 2, 3, 1).reshape(-1, c)
                cos_sim = F.cosine_similarity(fs_flat, ft_flat, dim=-1)
                return (1.0 - cos_sim).mean()
            else:
                cos_sim = F.cosine_similarity(feat_s, feat_t, dim=-1)
                return (1.0 - cos_sim).mean()
        elif distill_loss_type == "mse":
            return F.mse_loss(feat_s, feat_t)
        elif distill_loss_type == "smooth_l1":
            return F.smooth_l1_loss(feat_s, feat_t)
        else:
            raise ValueError(f"unsupported distill_loss={distill_loss_type!r}")

    def compute_total_distill_loss(student_feats, teacher_feats):
        """Compute the total distillation loss = GAP loss + stage_weight * sum(stage_losses).
        Args:
            student_feats: dict {"stage1":..,"stage2":..,"gap":..} or a single [B,D] tensor
            teacher_feats: dict or a single [B,D] tensor
        Returns:
            (total_loss, loss_dict) for logging
        """
        loss_dict = {}

        # GAP loss (always computed)
        if isinstance(student_feats, dict):
            gap_s = student_feats["gap"]
            gap_t = teacher_feats["gap"]
        else:
            gap_s = student_feats
            gap_t = teacher_feats

        if proj_layer is not None:
            gap_s_proj = proj_layer(gap_s)
        else:
            gap_s_proj = gap_s
        gap_loss = _compute_feat_loss(gap_s_proj, gap_t)
        loss_dict["gap"] = gap_loss.item()
        total_loss = gap_loss

        # stage losses (when enabled)
        if enable_stage_distill and isinstance(student_feats, dict):
            stage_loss_sum = 0.0
            for stage_idx in distill_stages:
                stage_key = f"stage{stage_idx}"
                if stage_key not in student_feats or stage_key not in teacher_feats:
                    continue
                fs = student_feats[stage_key]  # [B, C, H, W]
                ft = teacher_feats[stage_key]  # [B, C, H, W]
                if distill_stage_pool == "gap":
                    # GAP first, then the loss
                    fs_pooled = fs.mean(dim=[2, 3])  # [B, C]
                    ft_pooled = ft.mean(dim=[2, 3])  # [B, C]
                    s_loss = _compute_feat_loss(fs_pooled, ft_pooled)
                else:  # "pixel"
                    # per-pixel loss directly on the feature map
                    s_loss = _compute_feat_loss(fs, ft)
                loss_dict[stage_key] = s_loss.item()
                stage_loss_sum = stage_loss_sum + s_loss

            n_stages = len(distill_stages)
            if n_stages > 0:
                total_loss = total_loss + distill_stage_weight * (stage_loss_sum / n_stages)

        loss_dict["total"] = total_loss.item()
        return total_loss, loss_dict

    # ---- Optimizer & Scheduler ----
    trainable = [p for p in student.parameters() if p.requires_grad]
    if proj_layer is not None:
        trainable += list(proj_layer.parameters())
    optimizer = torch.optim.AdamW(
        trainable, lr=config["lr"], weight_decay=config.get("weight_decay", 0.01),
    )
    max_iters = int(config["max_iters"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_iters, eta_min=config.get("lr_min", 1e-6),
    )
    print(f"  Optimizer: AdamW, lr={config['lr']}")
    print(f"  Scheduler: CosineAnnealing, T_max={max_iters} iters")

    # ---- iteration-based training loop ----
    eval_every = int(config.get("eval_every_iters", 1000))
    print(f"\n" + "=" * 60)
    print(f"Stage 3: distillation training starts (evaluate every {eval_every} iters, {max_iters} iters in total)")
    print("=" * 60)

    history = {
        "iter": [],
        "train_loss": [],       # total loss
        "train_loss_gap": [],   # GAP component
        "val_loss": [],
        "val_cos_sim_gap": [],  # GAP cosine sim
        "lr": [],
    }
    # add stage keys dynamically per the number of distilled layers
    for _s in distill_stages:
        history[f"train_loss_stage{_s}"] = []
        history[f"val_cos_sim_stage{_s}"] = []

    best_cos_sim = -1.0
    grad_clip = config.get("grad_clip_max_norm", 1.0)

    running_loss = 0.0
    running_loss_dict = defaultdict(float)  # per-component accumulators
    running_samples = 0

    student.train()
    train_iter_gen = infinite_loader(train_loader)
    pbar = tqdm(range(1, max_iters + 1), desc="Distill", ncols=100)

    for global_iter in pbar:
        batch = next(train_iter_gen)
        if batch[0] is None:
            continue
        latents, images = batch
        latents = latents.to(device)
        images = images.to(device)

        # Teacher forward (frozen)
        with torch.no_grad():
            if enable_stage_distill:
                teacher_feats = extract_stage_features(teacher, images)
            else:
                teacher_feats = teacher(images)  # [B, teacher_feat_dim]

        # Student forward
        if enable_stage_distill:
            student_feats = student.extract_stage_features(latents)
        else:
            student_feats = student.extract_feat(latents)  # [B, student_feat_dim]

        loss, loss_dict = compute_total_distill_loss(student_feats, teacher_feats)

        # NaN guard
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"\n  [WARN] iter {global_iter}: loss is NaN/Inf, skipping")
            optimizer.zero_grad()
            continue

        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(trainable, grad_clip)
        optimizer.step()
        scheduler.step()

        bs = latents.size(0)
        running_loss += loss.item() * bs
        # accumulate per-component losses
        running_loss_dict["gap"] += loss_dict.get("gap", 0.0) * bs
        for _s in distill_stages:
            running_loss_dict[f"stage{_s}"] += loss_dict.get(f"stage{_s}", 0.0) * bs
        running_samples += bs

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # ---- periodic evaluation ----
        if global_iter % eval_every == 0 or global_iter == max_iters:
            train_avg_loss = running_loss / max(1, running_samples)

            # val evaluation
            student.eval()
            val_loss_sum = 0.0
            val_cos_sum_gap = 0.0
            val_stage_cos = {f"stage{s}": 0.0 for s in distill_stages}
            val_samples = 0
            with torch.no_grad():
                for vbatch in val_loader:
                    if vbatch[0] is None:
                        continue
                    v_latents, v_images = vbatch
                    v_latents = v_latents.to(device)
                    v_images = v_images.to(device)
                    if enable_stage_distill:
                        vt_feats = extract_stage_features(teacher, v_images)
                        vs_feats = student.extract_stage_features(v_latents)
                    else:
                        vt_feats = teacher(v_images)
                        vs_feats = student.extract_feat(v_latents)
                    v_loss, _ = compute_total_distill_loss(vs_feats, vt_feats)
                    # GAP cosine sim
                    if isinstance(vs_feats, dict):
                        gap_s = vs_feats["gap"]
                        gap_t = vt_feats["gap"]
                    else:
                        gap_s = vs_feats
                        gap_t = vt_feats
                    if proj_layer is not None:
                        gap_s = proj_layer(gap_s)
                    cos_gap = F.cosine_similarity(gap_s, gap_t, dim=-1).mean()
                    vbs = v_latents.size(0)
                    val_loss_sum += v_loss.item() * vbs
                    val_cos_sum_gap += cos_gap.item() * vbs
                    # per-layer cosine similarity
                    if enable_stage_distill and isinstance(vs_feats, dict):
                        for _s in distill_stages:
                            key = f"stage{_s}"
                            fs_stage = vs_feats[key]   # [B, C, H, W]
                            ft_stage = vt_feats[key]
                            fs_pooled = fs_stage.mean(dim=[2, 3])  # [B, C]
                            ft_pooled = ft_stage.mean(dim=[2, 3])
                            sc = F.cosine_similarity(fs_pooled, ft_pooled, dim=-1).mean()
                            val_stage_cos[key] += sc.item() * vbs
                    val_samples += vbs
            student.train()

            val_avg_loss = val_loss_sum / max(1, val_samples)
            val_avg_cos_gap = val_cos_sum_gap / max(1, val_samples)

            # record history
            current_lr = scheduler.get_last_lr()[0]
            history["iter"].append(global_iter)
            history["train_loss"].append(float(train_avg_loss))
            history["train_loss_gap"].append(
                float(running_loss_dict["gap"] / max(1, running_samples)))
            for _s in distill_stages:
                history[f"train_loss_stage{_s}"].append(
                    float(running_loss_dict[f"stage{_s}"] / max(1, running_samples)))
            history["val_loss"].append(float(val_avg_loss))
            history["val_cos_sim_gap"].append(float(val_avg_cos_gap))
            for _s in distill_stages:
                key = f"stage{_s}"
                history[f"val_cos_sim_{key}"].append(
                    float(val_stage_cos[key] / max(1, val_samples)))
            history["lr"].append(float(current_lr))

            # print the log
            cos_parts = [f"cos_gap={val_avg_cos_gap:.4f}"]
            for _s in distill_stages:
                cos_parts.append(
                    f"cos_s{_s}={val_stage_cos[f'stage{_s}'] / max(1, val_samples):.4f}")
            cos_str = ", ".join(cos_parts)
            print(f"\n  [Iter {global_iter}] loss={train_avg_loss:.4f}, "
                  f"val_loss={val_avg_loss:.4f}, {cos_str}, lr={current_lr:.2e}")

            # save the checkpoint
            is_best = val_avg_cos_gap > best_cos_sim
            if is_best:
                best_cos_sim = val_avg_cos_gap
            save_checkpoint(student, optimizer, global_iter, history, ckpt_dir,
                            tag=f"iter{global_iter}")
            if is_best:
                save_checkpoint(student, optimizer, global_iter, history, ckpt_dir,
                                tag="best")
                print(f"  \u2605 New best cos_gap={val_avg_cos_gap:.4f}")

            # save the history JSON + plots
            save_json(history, os.path.join(ckpt_dir, "meta", "history.json"))
            plot_distill_history(history, ckpt_dir)

            # reset the running stats
            running_loss = 0.0
            running_loss_dict = defaultdict(float)
            running_samples = 0

    # ---- write experiment_info.json (used by train to auto-locate the ckpt) ----
    best_ckpt_path = os.path.join(ckpt_dir, "ckpts", "backbone_best.pth")
    experiment_info = {
        "exp_tag": config.get("exp_tag", ""),
        "pretrain_mode": "distill",
        "model": config.get("model", "z-image-turbo"),
        "backbone": config["backbone"],
        "stem_type": config.get("stem_type", "progressive"),
        "stem_init": config.get("stem_init", "trunc_normal"),
        "train_scope": config.get("train_scope", "full"),
        "distill_stages": config.get("distill_stages", []),
        "best_metric": {"val_cos_sim": best_cos_sim},
        "best_ckpt": best_ckpt_path,
        "ckpt_dir": ckpt_dir,
    }
    save_json(experiment_info, os.path.join(ckpt_dir, "meta", "experiment_info.json"))

    print(f"\n{'='*60}")
    print(f"Distillation pretraining done! Best val_cos_sim={best_cos_sim:.4f}")
    print(f"Checkpoint: {ckpt_dir}")
    print(f"Best ckpt: {best_ckpt_path}")
    print(f"{'='*60}")


def run_distill_classify_pretrain(config):
    """End-to-end distill+classify joint pretraining: L_total = L_cls + lambda * L_distill.

    Joint end-to-end training combining teacher RGB feature-distillation and safety-classification label supervision.
    The student produces both features (for distillation alignment) and logits (for classification).

    Config parameters:
      - distill_lambda: distillation loss weight (default 1.0)
      - teacher_ckpt: custom teacher ckpt path (empty string = use IN1K)
      - labels_dir: classification label directory
      - other parameters are the same as distill / classify mode
    """
    from models_zoo import build_teacher_backbone, extract_stage_features

    seed_everything(config["seed"])
    if config.get("torch_home"):
        os.environ["TORCH_HOME"] = str(config["torch_home"])
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")

    # ---- output directory ----
    ckpt_dir = build_ckpt_dir(config)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(os.path.join(ckpt_dir, "meta"), exist_ok=True)
    os.makedirs(os.path.join(ckpt_dir, "plots"), exist_ok=True)
    save_json(config, os.path.join(ckpt_dir, "meta", "config.json"))

    log_path = os.path.join(ckpt_dir, "train.log")
    sys.stdout = Logger(log_path)
    print(f"[Config] pretrain_mode = distill_classify")
    print(f"[Config] ckpt_dir = {ckpt_dir}")
    print(f"[Config] device = {device}")
    for k, v in config.items():
        print(f"  {k}: {v}")

    # ---- load the class labels ----
    print("\n" + "=" * 60)
    print("Stage 1: load data and labels")
    print("=" * 60)

    label_format = config.get("label_format", "openimages")
    print(f"  label_format: {label_format}")

    if label_format == "safety_csv":
        # safety dataset format: read predictions.csv directly
        safety_csv = config.get("safety_label_csv") or config.get("classes_csv", "")
        if not safety_csv or not os.path.isfile(safety_csv):
            raise FileNotFoundError(
                f"label_format='safety_csv' but the label file was not found: {safety_csv}\n"
                f"Set safety_label_csv to point to predictions.csv"
            )
        image_annotations = load_safety_labels(safety_csv)
        num_classes = SAFETY_NUM_CLASSES
        idx_to_name = {i: name for i, name in enumerate(SAFETY_CLASS_NAMES)}
    else:
        # OpenImages format (default)
        label_to_idx, idx_to_name = load_classes(config["classes_csv"])
        num_classes = len(label_to_idx)
        image_annotations = load_all_classifications(config["labels_dir"], label_to_idx)

    print(f"  num classes: {num_classes}")
    print(f"  class names: {idx_to_name}")

    # build the manifest (needs both latent and annotation)
    manifest = build_manifest(config["latent_dir"], image_annotations)
    if not manifest:
        raise RuntimeError("no valid samples; check that latent_dir and labels_dir/safety_label_csv align")

    # image directory
    images_dir = config.get("images_dir")
    if images_dir is None:
        images_dir = os.path.join(os.path.dirname(config["latent_dir"].rstrip("/")), "images")

    # split train/val
    target_hw = tuple(config.get("target_hw", [224, 224]))
    val_size = int(config.get("val_size", 10000))
    train_manifest, val_manifest = split_train_val(
        manifest, val_size=val_size, seed=config["seed"],
    )
    print(f"  train={len(train_manifest)}, val={len(val_manifest)}")

    save_json({
        "num_classes": num_classes,
        "train_count": len(train_manifest),
        "val_count": len(val_manifest),
    }, os.path.join(ckpt_dir, "meta", "data_info.json"))
    save_json(idx_to_name, os.path.join(ckpt_dir, "meta", "class_map.json"))

    # Latent normalize
    model_name = config.get("model", "z-image-turbo")
    latent_mean, latent_std = None, None
    if bool(config.get("latent_enable_normalize", False)):
        print("\n  Computing per-channel latent normalization stats ...")
        latent_mean, latent_std, latent_stats_info = compute_latent_channel_stats(
            train_manifest, num_samples=config.get("latent_stats_num_samples", 500),
            seed=config["seed"], model_name=model_name,
        )
        save_json(latent_stats_info, os.path.join(ckpt_dir, "meta", "latent_stats.json"))

    # ---- Dataset & DataLoader ----
    teacher_image_size = config.get("teacher_image_size", 224)
    teacher_image_mean = config.get("teacher_image_mean", [0.485, 0.456, 0.406])
    teacher_image_std = config.get("teacher_image_std", [0.229, 0.224, 0.225])

    train_ds = OpenImagesDistillClassifyDataset(
        train_manifest, images_dir, num_classes, target_hw=target_hw,
        image_size=teacher_image_size, image_mean=teacher_image_mean, image_std=teacher_image_std,
        data_aug=bool(config.get("data_aug", False)), is_train=True,
        latent_mean=latent_mean, latent_std=latent_std, model_name=model_name,
    )
    val_ds = OpenImagesDistillClassifyDataset(
        val_manifest, images_dir, num_classes, target_hw=target_hw,
        image_size=teacher_image_size, image_mean=teacher_image_mean, image_std=teacher_image_std,
        data_aug=False, is_train=False,
        latent_mean=latent_mean, latent_std=latent_std, model_name=model_name,
    )
    train_loader = DataLoader(
        train_ds, batch_size=config["bs"], shuffle=True,
        num_workers=config.get("nw", 4), collate_fn=distill_classify_collate_fn,
        pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config["bs"], shuffle=False,
        num_workers=config.get("nw", 4), collate_fn=distill_classify_collate_fn,
        pin_memory=True, drop_last=False,
    )
    print(f"  target_hw={target_hw}, train={len(train_ds)}, val={len(val_ds)}")

    # ---- teacher model (frozen) ----
    print("\n" + "=" * 60)
    print("Stage 2: build teacher & student")
    print("=" * 60)
    teacher_backbone_name = config.get("teacher_backbone", "convnext_base")
    teacher_ckpt_path = config.get("teacher_ckpt", "") or None
    teacher, teacher_feat_dim = build_teacher_backbone(
        backbone_name=teacher_backbone_name,
        pretrained=config.get("teacher_pretrained", True),
        ckpt_path=teacher_ckpt_path,
    )
    teacher = teacher.to(device)
    teacher.eval()
    print(f"  Teacher: {teacher_backbone_name}, feat_dim={teacher_feat_dim}, frozen=True")

    # ---- student model (with classification head) ----
    stem_type = config.get("stem_type", "progressive")
    student = PretrainModel(
        backbone_name=config["backbone"],
        in_chans=config["in_chans"],
        num_classes=num_classes,
        pretrained=config.get("backbone_pretrained", True),
        dropout=config.get("dropout", 0.1),
        stem_type=stem_type,
        stem_channels=config.get("stem_channels"),
        stem_init=config.get("stem_init", "trunc_normal"),
    )
    student = student.to(device)

    # train_scope
    train_scope = config.get("train_scope", "full")
    if train_scope == "stem_only":
        for n, p in student.named_parameters():
            p.requires_grad = False
        if stem_type == "progressive":
            if get_backbone_family(config["backbone"]) in ("convnext", "swin_v2"):
                for p in student.backbone.features[0].parameters():
                    p.requires_grad = True
            elif get_backbone_family(config["backbone"]) == "vit":
                raise ValueError("ViT does not support progressive stem")
            else:
                for p in student.backbone.conv1.parameters():
                    p.requires_grad = True
        else:
            if get_backbone_family(config["backbone"]) in ("convnext", "swin_v2"):
                for p in student.backbone.features[0][0].parameters():
                    p.requires_grad = True
            elif get_backbone_family(config["backbone"]) == "vit":
                for p in student.backbone.conv_proj.parameters():
                    p.requires_grad = True
            else:
                for p in student.backbone.conv1.parameters():
                    p.requires_grad = True
        # unfreeze the head
        for n, p in student.named_parameters():
            if not n.startswith("backbone."):
                p.requires_grad = True

    student_feat_dim = student.feat_dim
    total_params = sum(p.numel() for p in student.parameters())
    trainable_params = sum(p.numel() for p in student.parameters() if p.requires_grad)
    print(f"  Student: {config['backbone']}, stem_type={stem_type}")
    print(f"  Student feat_dim={student_feat_dim}, num_classes={num_classes}")
    print(f"  train_scope={train_scope}")
    print(f"  Total params: {total_params:,}, Trainable: {trainable_params:,}")

    # feature-dim alignment layer
    proj_layer = None
    if student_feat_dim != teacher_feat_dim:
        proj_layer = nn.Linear(student_feat_dim, teacher_feat_dim).to(device)
        print(f"  adding projection layer: {student_feat_dim} -> {teacher_feat_dim}")

    # ---- loss config ----
    distill_lambda = float(config.get("distill_lambda", 1.0))
    distill_loss_type = config.get("distill_loss", "cosine")
    print(f"  Loss: L_cls + {distill_lambda} * L_distill ({distill_loss_type})")

    def _compute_distill_loss(feat_s, feat_t):
        if distill_loss_type == "cosine":
            cos_sim = F.cosine_similarity(feat_s, feat_t, dim=-1)
            return (1.0 - cos_sim).mean()
        elif distill_loss_type == "mse":
            return F.mse_loss(feat_s, feat_t)
        elif distill_loss_type == "smooth_l1":
            return F.smooth_l1_loss(feat_s, feat_t)
        else:
            raise ValueError(f"unsupported distill_loss={distill_loss_type!r}")

    # ---- Optimizer & Scheduler ----
    trainable = [p for p in student.parameters() if p.requires_grad]
    if proj_layer is not None:
        trainable += list(proj_layer.parameters())
    optimizer = torch.optim.AdamW(
        trainable, lr=config["lr"], weight_decay=config.get("weight_decay", 0.01),
    )
    max_iters = int(config["max_iters"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_iters, eta_min=config.get("lr_min", 1e-6),
    )
    print(f"  Optimizer: AdamW, lr={config['lr']}")
    print(f"  Scheduler: CosineAnnealing, T_max={max_iters} iters")

    # ---- training loop ----
    eval_every = int(config.get("eval_every_iters", 1000))
    grad_clip = config.get("grad_clip_max_norm", 1.0)
    print(f"\n" + "=" * 60)
    print(f"Stage 3: end-to-end training starts (evaluate every {eval_every} iters, {max_iters} iters in total)")
    print(f"  distill_lambda={distill_lambda}")
    print("=" * 60)

    history = {
        "iter": [], "train_loss": [], "train_loss_cls": [], "train_loss_distill": [],
        "val_loss": [], "val_mAP": [], "val_acc": [], "val_cos_sim": [], "lr": [],
    }
    best_mAP = -1.0

    running_loss = 0.0
    running_loss_cls = 0.0
    running_loss_distill = 0.0
    running_samples = 0

    student.train()
    train_iter_gen = infinite_loader(train_loader)
    pbar = tqdm(range(1, max_iters + 1), desc="Distill+Cls", ncols=120)

    for global_iter in pbar:
        batch = next(train_iter_gen)
        if batch[0] is None:
            continue
        latents, images, targets, masks = batch
        latents = latents.to(device)
        images = images.to(device)
        targets = targets.to(device)
        masks = masks.to(device)

        # Teacher forward (frozen)
        with torch.no_grad():
            teacher_feats = teacher(images)  # [B, teacher_feat_dim]

        # student forward: used for both distillation and classification
        student_feats = student.extract_feat(latents)  # [B, student_feat_dim]
        logits = student.head(student.dropout(student_feats))  # [B, num_classes]

        # classification loss
        loss_cls = masked_bce_loss(logits, targets, masks)

        # distillation loss
        if proj_layer is not None:
            student_feats_proj = proj_layer(student_feats)
        else:
            student_feats_proj = student_feats
        loss_distill = _compute_distill_loss(student_feats_proj, teacher_feats)

        # total loss
        loss = loss_cls + distill_lambda * loss_distill

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"\n  [WARN] iter {global_iter}: loss is NaN/Inf, skipping")
            optimizer.zero_grad()
            continue

        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(trainable, grad_clip)
        optimizer.step()
        scheduler.step()

        bs = latents.size(0)
        running_loss += loss.item() * bs
        running_loss_cls += loss_cls.item() * bs
        running_loss_distill += loss_distill.item() * bs
        running_samples += bs

        pbar.set_postfix({"L": f"{loss.item():.4f}",
                          "cls": f"{loss_cls.item():.4f}",
                          "dist": f"{loss_distill.item():.4f}"})

        # ---- periodic evaluation ----
        if global_iter % eval_every == 0 or global_iter == max_iters:
            train_avg_loss = running_loss / max(1, running_samples)
            train_avg_cls = running_loss_cls / max(1, running_samples)
            train_avg_distill = running_loss_distill / max(1, running_samples)

            # Validation
            student.eval()
            val_loss_sum = 0.0
            val_cos_sum = 0.0
            val_samples = 0
            all_targets_list = []
            all_probs_list = []
            all_masks_list = []

            with torch.no_grad():
                for vbatch in val_loader:
                    if vbatch[0] is None:
                        continue
                    v_lat, v_img, v_tgt, v_msk = vbatch
                    v_lat = v_lat.to(device)
                    v_img = v_img.to(device)
                    v_tgt = v_tgt.to(device)
                    v_msk = v_msk.to(device)

                    vt_feats = teacher(v_img)
                    vs_feats = student.extract_feat(v_lat)
                    v_logits = student.head(student.dropout(vs_feats))

                    v_loss_cls = masked_bce_loss(v_logits, v_tgt, v_msk)
                    if proj_layer is not None:
                        vs_proj = proj_layer(vs_feats)
                    else:
                        vs_proj = vs_feats
                    v_loss_dist = _compute_distill_loss(vs_proj, vt_feats)
                    v_loss = v_loss_cls + distill_lambda * v_loss_dist

                    cos_sim = F.cosine_similarity(vs_proj, vt_feats, dim=-1).mean()

                    vbs = v_lat.size(0)
                    val_loss_sum += v_loss.item() * vbs
                    val_cos_sum += cos_sim.item() * vbs
                    val_samples += vbs

                    all_targets_list.append(v_tgt.cpu().numpy())
                    all_probs_list.append(torch.sigmoid(v_logits).cpu().numpy())
                    all_masks_list.append(v_msk.cpu().numpy())

            student.train()

            val_avg_loss = val_loss_sum / max(1, val_samples)
            val_avg_cos = val_cos_sum / max(1, val_samples)

            all_targets_np = np.concatenate(all_targets_list, axis=0)
            all_probs_np = np.concatenate(all_probs_list, axis=0)
            all_masks_np = np.concatenate(all_masks_list, axis=0)
            val_mAP = compute_mAP(all_targets_np, all_probs_np, all_masks_np)
            all_preds_np = (all_probs_np > 0.5).astype(np.float32)
            val_acc = compute_overall_accuracy(all_targets_np, all_preds_np, all_masks_np)

            current_lr = scheduler.get_last_lr()[0]
            history["iter"].append(global_iter)
            history["train_loss"].append(float(train_avg_loss))
            history["train_loss_cls"].append(float(train_avg_cls))
            history["train_loss_distill"].append(float(train_avg_distill))
            history["val_loss"].append(float(val_avg_loss))
            history["val_mAP"].append(float(val_mAP))
            history["val_acc"].append(float(val_acc))
            history["val_cos_sim"].append(float(val_avg_cos))
            history["lr"].append(float(current_lr))

            print(f"\n  [Iter {global_iter}] loss={train_avg_loss:.4f} "
                  f"(cls={train_avg_cls:.4f}, dist={train_avg_distill:.4f}), "
                  f"val_loss={val_avg_loss:.4f}, mAP={val_mAP:.4f}, "
                  f"acc={val_acc:.4f}, cos={val_avg_cos:.4f}, lr={current_lr:.2e}")

            # Checkpoint
            is_best = val_mAP > best_mAP
            if is_best:
                best_mAP = val_mAP
            save_checkpoint(student, optimizer, global_iter, history, ckpt_dir,
                            tag=f"iter{global_iter}")
            if is_best:
                save_checkpoint(student, optimizer, global_iter, history, ckpt_dir,
                                tag="best")
                print(f"  ★ New best mAP={val_mAP:.4f}")

            save_json(history, os.path.join(ckpt_dir, "meta", "history.json"))
            # plots (reuse the classify plotting functions, plus extra distill info)
            plot_path = os.path.join(ckpt_dir, "plots", f"curves_iter{global_iter}.png")
            plot_training_curves(history, plot_path)

            running_loss = 0.0
            running_loss_cls = 0.0
            running_loss_distill = 0.0
            running_samples = 0

    # ---- training finished ----
    best_ckpt_path = os.path.join(ckpt_dir, "ckpts", "backbone_best.pth")
    experiment_info = {
        "exp_tag": config.get("exp_tag", ""),
        "pretrain_mode": "distill_classify",
        "model": config.get("model", "z-image-turbo"),
        "backbone": config["backbone"],
        "stem_type": config.get("stem_type", "progressive"),
        "stem_init": config.get("stem_init", "trunc_normal"),
        "train_scope": config.get("train_scope", "full"),
        "distill_lambda": distill_lambda,
        "distill_loss": distill_loss_type,
        "best_metric": {"val_mAP": best_mAP},
        "best_ckpt": best_ckpt_path,
        "ckpt_dir": ckpt_dir,
    }
    save_json(experiment_info, os.path.join(ckpt_dir, "meta", "experiment_info.json"))

    print(f"\n{'='*60}")
    print(f"End-to-end distill+classify done! Best val_mAP={best_mAP:.4f}")
    print(f"Checkpoint: {ckpt_dir}")
    print(f"Best ckpt: {best_ckpt_path}")
    print(f"{'='*60}")


def run_classify_pretrain(config):
    """Latent multi-label classification pretraining (OpenImages latents + multi-label BCE).

    Standard classification pretraining: train a classification backbone on OpenImages VAE-encoded latents,
    producing backbone_best.pth for downstream finetuning.
    """
    seed_everything(config["seed"])
    if config.get("torch_home"):
        os.environ["TORCH_HOME"] = str(config["torch_home"])
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")

    # ---- output directory ----
    ckpt_dir = build_ckpt_dir(config)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(os.path.join(ckpt_dir, "meta"), exist_ok=True)
    os.makedirs(os.path.join(ckpt_dir, "plots"), exist_ok=True)
    save_json(config, os.path.join(ckpt_dir, "meta", "config.json"))

    # logging
    log_path = os.path.join(ckpt_dir, "train.log")
    sys.stdout = Logger(log_path)
    print(f"[Config] ckpt_dir = {ckpt_dir}")
    print(f"[Config] device = {device}")
    for k, v in config.items():
        print(f"  {k}: {v}")

    # ---- load the labels ----
    print("\n" + "=" * 60)
    print("Stage 1: load classes and labels")
    print("=" * 60)
    label_to_idx, idx_to_name = load_classes(config["classes_csv"])
    num_classes = len(label_to_idx)
    print(f"  num classes: {num_classes}")

    image_annotations = load_all_classifications(config["labels_dir"], label_to_idx)

    # ---- build the manifest ----
    print("\n" + "=" * 60)
    print("Stage 2: build the training data")
    print("=" * 60)
    manifest = build_manifest(config["latent_dir"], image_annotations)
    if not manifest:
        raise RuntimeError("no valid samples; check that latent_dir and labels align")

    # ---- preview: sanity-check the labels before training ----
    images_dir = config.get("images_dir")
    if images_dir is None:
        # infer from latent_dir's sibling by default: .../latents -> .../images
        images_dir = os.path.join(os.path.dirname(config["latent_dir"].rstrip("/")), "images")
    preview_dir = os.path.join(ckpt_dir, "preview")
    target_hw = tuple(config.get("target_hw", [128, 128]))
    preview_data(manifest, idx_to_name, images_dir, preview_dir,
                 target_hw=target_hw,
                 num_preview=config.get("num_preview", 30),
                 model_name=config.get("model", "z-image-turbo"))

    val_size = int(config.get("val_size", 10000))
    train_manifest, val_manifest = split_train_val(
        manifest, val_size=val_size, seed=config["seed"],
    )

    # save the manifest info
    save_json({
        "num_classes": num_classes,
        "train_count": len(train_manifest),
        "val_count": len(val_manifest),
        "total_latent_files": len(manifest) + 0,  # manifest already filtered
    }, os.path.join(ckpt_dir, "meta", "data_info.json"))
    save_json(idx_to_name, os.path.join(ckpt_dir, "meta", "class_map.json"))

    # ---- Dataset & DataLoader ----
    target_hw = tuple(config.get("target_hw", [128, 128]))

    # compute per-channel latent normalization stats (aligned with train_image.py)
    # compute per-channel latent normalization stats (only when normalize is on)
    latent_mean, latent_std = None, None
    if bool(config.get("latent_enable_normalize", False)):
        print("\n  Computing per-channel latent normalization stats ...")
        latent_mean, latent_std, latent_stats_info = compute_latent_channel_stats(
            train_manifest,
            num_samples=config.get("latent_stats_num_samples", 500),
            seed=config["seed"],
            model_name=config.get("model", "z-image-turbo"),
        )
        save_json(latent_stats_info, os.path.join(ckpt_dir, "meta", "latent_stats.json"))
    else:
        print("\n  [Latent Stats] latent_enable_normalize=False; skipping per-channel normalization, using raw latent values")

    model_name = config.get("model", "z-image-turbo")
    train_ds = OpenImagesLatentDataset(
        train_manifest, num_classes, target_hw=target_hw,
        data_aug=bool(config.get("data_aug", True)), is_train=True,
        latent_mean=latent_mean, latent_std=latent_std,
        model_name=model_name,
    )
    val_ds = OpenImagesLatentDataset(
        val_manifest, num_classes, target_hw=target_hw,
        data_aug=False, is_train=False,
        latent_mean=latent_mean, latent_std=latent_std,
        model_name=model_name,
    )
    train_loader = DataLoader(
        train_ds, batch_size=config["bs"], shuffle=True,
        num_workers=config.get("nw", 4), collate_fn=fixed_collate_fn,
        pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config["bs"], shuffle=False,
        num_workers=config.get("nw", 4), collate_fn=fixed_collate_fn,
        pin_memory=True, drop_last=False,
    )
    print(f"  target_hw={target_hw}, train={len(train_ds)}, val={len(val_ds)}")

    # ---- model ----
    print("\n" + "=" * 60)
    print("Stage 3: build the model")
    print("=" * 60)
    model = PretrainModel(
        backbone_name=config["backbone"],
        in_chans=config["in_chans"],
        num_classes=num_classes,
        pretrained=config.get("backbone_pretrained", True),
        dropout=config.get("dropout", 0.1),
        stem_type=config.get("stem_type", "single"),
        stem_channels=config.get("stem_channels"),
        stem_init=config.get("stem_init", "expand_in1k"),
    )
    model = model.to(device)

    # train_scope handling (aligned with distill mode)
    stem_type = config.get("stem_type", "single")
    train_scope = config.get("train_scope", "full")
    if train_scope == "stem_only":
        # freeze the backbone blocks; train the stem + head only
        for n, p in model.named_parameters():
            p.requires_grad = False
        # unfreeze the stem
        if stem_type == "progressive":
            if get_backbone_family(config["backbone"]) in ("convnext", "swin_v2"):
                for p in model.backbone.features[0].parameters():
                    p.requires_grad = True
            elif get_backbone_family(config["backbone"]) == "vit":
                raise ValueError("ViT does not support progressive stem")
            else:
                for p in model.backbone.conv1.parameters():
                    p.requires_grad = True
        else:
            if get_backbone_family(config["backbone"]) in ("convnext", "swin_v2"):
                for p in model.backbone.features[0][0].parameters():
                    p.requires_grad = True
            elif get_backbone_family(config["backbone"]) == "vit":
                for p in model.backbone.conv_proj.parameters():
                    p.requires_grad = True
            else:
                for p in model.backbone.conv1.parameters():
                    p.requires_grad = True
        # unfreeze the head
        for n, p in model.named_parameters():
            if not n.startswith("backbone."):
                p.requires_grad = True

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Backbone: {config['backbone']}, in_chans={config['in_chans']}")
    print(f"  Head: Linear({model.feat_dim}, {num_classes})")
    print(f"  train_scope={train_scope}")
    print(f"  Total params: {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}")

    # ---- Optimizer & Scheduler ----
    backbone_lr = config.get("backbone_lr", config["lr"])
    head_lr = config.get("head_lr", config["lr"])
    param_groups = [
        {"params": [p for n, p in model.named_parameters()
                    if p.requires_grad and n.startswith("backbone.")],
         "lr": backbone_lr},
        {"params": [p for n, p in model.named_parameters()
                    if p.requires_grad and not n.startswith("backbone.")],
         "lr": head_lr},
    ]
    optimizer = torch.optim.AdamW(
        param_groups, lr=config["lr"], weight_decay=config.get("weight_decay", 0.01),
    )
    max_iters = int(config["max_iters"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_iters, eta_min=config.get("lr_min", 1e-6),
    )
    print(f"  Optimizer: AdamW, backbone_lr={backbone_lr}, head_lr={head_lr}")
    print(f"  Scheduler: CosineAnnealing, T_max={max_iters} iters")
    
    # ---- iteration-based training loop ----
    eval_every = int(config.get("eval_every_iters", 1000))
    print(f"\n" + "=" * 60)
    print(f"Stage 4: training starts (evaluate every {eval_every} iters, {max_iters} iters in total)")
    print("=" * 60)
    
    history = {"iter": [], "train_loss": [], "val_loss": [],
               "train_acc": [], "val_acc": [], "val_mAP": [], "lr": []}
    best_mAP = -1.0
    grad_clip = config.get("grad_clip_max_norm", 1.0)
    
    # training state
    running_loss = 0.0
    running_correct = 0
    running_labeled = 0
    running_samples = 0
    
    model.train()
    train_iter = infinite_loader(train_loader)
    pbar = tqdm(range(1, max_iters + 1), desc="Training", ncols=100)
    
    for global_iter in pbar:
        batch = next(train_iter)
        if batch[0] is None:
            continue
        latents, targets, masks = batch
        latents = latents.to(device)
        targets = targets.to(device)
        masks = masks.to(device)
    
        logits = model(latents)
        loss = masked_bce_loss(logits, targets, masks)

        # NaN guard: skip abnormal batches
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"\n  [WARN] iter {global_iter}: loss is NaN/Inf, skipping batch")
            optimizer.zero_grad()
            continue
    
        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()
    
        bs = latents.size(0)
        running_loss += loss.item() * bs
        running_samples += bs
    
        with torch.no_grad():
            preds = (torch.sigmoid(logits) > 0.5).float()
            mask_bool = masks > 0.5
            running_correct += ((preds == targets) * mask_bool).sum().item()
            running_labeled += mask_bool.sum().item()
    
        # tqdm display
        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            acc=f"{running_correct / max(1, running_labeled):.4f}",
            lr=f"{scheduler.get_last_lr()[0]:.2e}",
        )
    
        # ---- evaluate + save every eval_every iters ----
        if global_iter % eval_every == 0 or global_iter == max_iters:
            train_loss = running_loss / max(1, running_samples)
            train_acc = running_correct / max(1, running_labeled)
            current_lr = scheduler.get_last_lr()[0]
    
            print(f"\n--- Iter {global_iter}/{max_iters} (lr={current_lr:.2e}) ---")
            print(f"  [Train] loss={train_loss:.4f}, acc={train_acc:.4f} "
                  f"(samples={running_samples})")
    
            # validation
            val_loss, val_mAP, val_acc = evaluate(
                model, val_loader, device, global_iter, "Val",
            )
            model.train()  # back to training mode

            # record history
            history["iter"].append(global_iter)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)
            history["val_mAP"].append(val_mAP)
            history["lr"].append(current_lr)
    
            print(f"  Summary: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
                  f"val_loss={val_loss:.4f}, val_mAP={val_mAP:.4f}, val_acc={val_acc:.4f}")
    
            # save the checkpoint (a unique name each time)
            save_checkpoint(model, optimizer, global_iter, history,
                           ckpt_dir, tag=f"iter{global_iter}")
    
            # save the best
            if val_mAP > best_mAP:
                best_mAP = val_mAP
                save_checkpoint(model, optimizer, global_iter, history,
                               ckpt_dir, tag="best")
                print(f"  \u2605 New best mAP={val_mAP:.4f} @ iter {global_iter}")
    
            # save the plot (a unique name each time)
            plot_path = os.path.join(
                ckpt_dir, "plots", f"curves_iter{global_iter}.png"
            )
            plot_training_curves(history, plot_path)
            save_json(history, os.path.join(ckpt_dir, "meta", "history.json"))
            print(f"  [Saved] ckpt + plot @ iter {global_iter}")
    
            # reset the running stats
            running_loss = 0.0
            running_correct = 0
            running_labeled = 0
            running_samples = 0
    
    # ---- training finished ----
    # write experiment_info.json (used by train to auto-locate the ckpt)
    best_ckpt_path = os.path.join(ckpt_dir, "ckpts", "backbone_best.pth")
    experiment_info = {
        "exp_tag": config.get("exp_tag", ""),
        "pretrain_mode": "classify",
        "model": config.get("model", "z-image-turbo"),
        "backbone": config["backbone"],
        "stem_type": config.get("stem_type", "progressive"),
        "stem_init": config.get("stem_init", "trunc_normal"),
        "train_scope": config.get("train_scope", "full"),
        "best_metric": {"val_mAP": best_mAP},
        "best_ckpt": best_ckpt_path,
        "ckpt_dir": ckpt_dir,
    }
    save_json(experiment_info, os.path.join(ckpt_dir, "meta", "experiment_info.json"))

    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"  Best val mAP: {best_mAP:.4f}")
    print(f"  Output dir: {ckpt_dir}")
    print(f"  Best ckpt: {best_ckpt_path}")
    print("=" * 60)


# ===========================================================================
# 7b. Image-level classification pretraining (control experiment: raw RGB images, no latents)
# ===========================================================================

def build_image_manifest(images_dir, image_annotations):
    """Build the image-level manifest.
    Scan the image files in images_dir and intersect with the labels.
    Save a cache after the first scan; later runs read it directly.
    """
    cache_path = os.path.join(images_dir, "_image_files.cache.txt")
    if os.path.exists(cache_path):
        print(f"  reading the image cache: {cache_path}")
        with open(cache_path, "r") as f:
            image_files = [line.strip() for line in f if line.strip()]
        print(f"  loaded {len(image_files)} image files from cache")
    else:
        print("  scanning the images directory (first run; the network storage may take minutes)...")
        sys.stdout.flush()
        image_files = [f for f in os.listdir(images_dir)
                       if f.lower().endswith((".jpg", ".png", ".jpeg"))]
        print(f"  found {len(image_files)} image files")
        try:
            with open(cache_path, "w") as f:
                f.write("\n".join(image_files))
            print(f"  cache saved: {cache_path}")
        except OSError:
            print("  [WARN] cannot write the cache file")

    manifest = []
    n_no_label = 0
    for fname in image_files:
        image_id = os.path.splitext(fname)[0]
        if image_id not in image_annotations:
            n_no_label += 1
            continue
        manifest.append({
            "image_id": image_id,
            "annotations": image_annotations[image_id],
        })
    print(f"[Image Manifest] valid samples: {len(manifest)}, skipped without labels: {n_no_label}")
    return manifest


def run_classify_image_pretrain(config):
    """Image-level classification pretraining (control experiment).
    Similar to run_classify_pretrain, but the input is RGB images instead of latents.
    Verifies whether the OpenImages pretraining gain comes from data volume rather than latent-domain adaptation.
    """
    seed_everything(config["seed"])
    if config.get("torch_home"):
        os.environ["TORCH_HOME"] = str(config["torch_home"])
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")

    # ---- output directory ----
    ckpt_dir = build_ckpt_dir(config)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(os.path.join(ckpt_dir, "meta"), exist_ok=True)
    os.makedirs(os.path.join(ckpt_dir, "plots"), exist_ok=True)
    save_json(config, os.path.join(ckpt_dir, "meta", "config.json"))

    # logging
    log_path = os.path.join(ckpt_dir, "train.log")
    sys.stdout = Logger(log_path)
    print(f"[Config] pretrain_mode = classify_image")
    print(f"[Config] ckpt_dir = {ckpt_dir}")
    print(f"[Config] device = {device}")
    for k, v in config.items():
        print(f"  {k}: {v}")

    # ---- load the labels ----
    print("\n" + "=" * 60)
    print("Stage 1: load classes and labels")
    print("=" * 60)
    label_to_idx, idx_to_name = load_classes(config["classes_csv"])
    num_classes = len(label_to_idx)
    print(f"  num classes: {num_classes}")

    image_annotations = load_all_classifications(config["labels_dir"], label_to_idx)

    # ---- build the manifest (scan the images directory) ----
    print("\n" + "=" * 60)
    print("Stage 2: build the training data")
    print("=" * 60)
    images_dir = config["images_dir"]
    manifest = build_image_manifest(images_dir, image_annotations)
    if not manifest:
        raise RuntimeError("no valid samples; check that images_dir and labels align")

    val_size = int(config.get("val_size", 10000))
    train_manifest, val_manifest = split_train_val(
        manifest, val_size=val_size, seed=config["seed"],
    )

    # save the manifest info
    save_json({
        "num_classes": num_classes,
        "train_count": len(train_manifest),
        "val_count": len(val_manifest),
        "total_images": len(manifest),
    }, os.path.join(ckpt_dir, "meta", "data_info.json"))
    save_json(idx_to_name, os.path.join(ckpt_dir, "meta", "class_map.json"))

    # ---- Dataset & DataLoader ----
    image_size = int(config.get("image_size", 224))
    train_ds = OpenImagesImageClassifyDataset(
        train_manifest, images_dir, num_classes,
        image_size=image_size,
        data_aug=bool(config.get("data_aug", True)), is_train=True,
    )
    val_ds = OpenImagesImageClassifyDataset(
        val_manifest, images_dir, num_classes,
        image_size=image_size,
        data_aug=False, is_train=False,
    )
    train_loader = DataLoader(
        train_ds, batch_size=config["bs"], shuffle=True,
        num_workers=config.get("nw", 4), collate_fn=fixed_collate_fn,
        pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config["bs"], shuffle=False,
        num_workers=config.get("nw", 4), collate_fn=fixed_collate_fn,
        pin_memory=True, drop_last=False,
    )
    print(f"  image_size={image_size}, train={len(train_ds)}, val={len(val_ds)}")

    # ---- model (original backbone, in_chans=3, stem untouched) ----
    print("\n" + "=" * 60)
    print("Stage 3: build the model (image-level, in_chans=3, original stem)")
    print("=" * 60)
    model = PretrainModel(
        backbone_name=config["backbone"],
        in_chans=3,
        num_classes=num_classes,
        pretrained=config.get("backbone_pretrained", True),
        dropout=config.get("dropout", 0.1),
        stem_type="original",
        stem_channels=None,
        stem_init="expand_in1k",
    )
    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Backbone: {config['backbone']}, in_chans=3 (RGB)")
    print(f"  Head: Linear({model.feat_dim}, {num_classes})")
    print(f"  Total params: {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}")

    # ---- Optimizer & Scheduler ----
    backbone_lr = config.get("backbone_lr", config["lr"])
    head_lr = config.get("head_lr", config["lr"])
    param_groups = [
        {"params": [p for n, p in model.named_parameters()
                    if p.requires_grad and n.startswith("backbone.")],
         "lr": backbone_lr},
        {"params": [p for n, p in model.named_parameters()
                    if p.requires_grad and not n.startswith("backbone.")],
         "lr": head_lr},
    ]
    optimizer = torch.optim.AdamW(
        param_groups, lr=config["lr"], weight_decay=config.get("weight_decay", 0.01),
    )
    max_iters = int(config["max_iters"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_iters, eta_min=config.get("lr_min", 1e-6),
    )
    print(f"  Optimizer: AdamW, backbone_lr={backbone_lr}, head_lr={head_lr}")
    print(f"  Scheduler: CosineAnnealing, T_max={max_iters} iters")

    # ---- iteration-based training loop ----
    eval_every = int(config.get("eval_every_iters", 5000))
    print(f"\n" + "=" * 60)
    print(f"Stage 4: training starts (evaluate every {eval_every} iters, {max_iters} iters in total)")
    print("=" * 60)

    history = {"iter": [], "train_loss": [], "val_loss": [],
               "train_acc": [], "val_acc": [], "val_mAP": [], "lr": []}
    best_mAP = -1.0
    grad_clip = config.get("grad_clip_max_norm", 1.0)

    running_loss = 0.0
    running_correct = 0
    running_labeled = 0
    running_samples = 0

    model.train()
    train_iter = infinite_loader(train_loader)
    pbar = tqdm(range(1, max_iters + 1), desc="ImageClassify", ncols=100)

    for global_iter in pbar:
        batch = next(train_iter)
        if batch[0] is None:
            continue
        images, targets, masks = batch
        images = images.to(device)
        targets = targets.to(device)
        masks = masks.to(device)

        logits = model(images)
        loss = masked_bce_loss(logits, targets, masks)

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"\n  [WARN] iter {global_iter}: loss is NaN/Inf, skipping batch")
            optimizer.zero_grad()
            continue

        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()

        bs = images.size(0)
        running_loss += loss.item() * bs
        running_samples += bs

        with torch.no_grad():
            preds = (torch.sigmoid(logits) > 0.5).float()
            mask_bool = masks > 0.5
            running_correct += ((preds == targets) * mask_bool).sum().item()
            running_labeled += mask_bool.sum().item()

        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            acc=f"{running_correct / max(1, running_labeled):.4f}",
            lr=f"{scheduler.get_last_lr()[0]:.2e}",
        )

        # ---- evaluate + save every eval_every iters ----
        if global_iter % eval_every == 0 or global_iter == max_iters:
            train_loss = running_loss / max(1, running_samples)
            train_acc = running_correct / max(1, running_labeled)
            current_lr = scheduler.get_last_lr()[0]

            print(f"\n--- Iter {global_iter}/{max_iters} (lr={current_lr:.2e}) ---")
            print(f"  [Train] loss={train_loss:.4f}, acc={train_acc:.4f} "
                  f"(samples={running_samples})")

            val_loss, val_mAP, val_acc = evaluate(
                model, val_loader, device, global_iter, "Val",
            )
            model.train()

            history["iter"].append(global_iter)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)
            history["val_mAP"].append(val_mAP)
            history["lr"].append(current_lr)

            print(f"  Summary: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
                  f"val_loss={val_loss:.4f}, val_mAP={val_mAP:.4f}, val_acc={val_acc:.4f}")

            save_checkpoint(model, optimizer, global_iter, history,
                           ckpt_dir, tag=f"iter{global_iter}")

            if val_mAP > best_mAP:
                best_mAP = val_mAP
                save_checkpoint(model, optimizer, global_iter, history,
                               ckpt_dir, tag="best")
                print(f"  \u2605 New best mAP={val_mAP:.4f} @ iter {global_iter}")

            plot_path = os.path.join(
                ckpt_dir, "plots", f"curves_iter{global_iter}.png"
            )
            plot_training_curves(history, plot_path)
            save_json(history, os.path.join(ckpt_dir, "meta", "history.json"))
            print(f"  [Saved] ckpt + plot @ iter {global_iter}")

            running_loss = 0.0
            running_correct = 0
            running_labeled = 0
            running_samples = 0

    # ---- training finished ----
    best_ckpt_path = os.path.join(ckpt_dir, "ckpts", "backbone_best.pth")
    experiment_info = {
        "exp_tag": config.get("exp_tag", ""),
        "pretrain_mode": "classify_image",
        "model": config.get("model", "image-pretrain"),
        "backbone": config["backbone"],
        "stem_type": "original",
        "stem_init": "N/A",
        "train_scope": config.get("train_scope", "full"),
        "best_metric": {"val_mAP": best_mAP},
        "best_ckpt": best_ckpt_path,
        "ckpt_dir": ckpt_dir,
    }
    save_json(experiment_info, os.path.join(ckpt_dir, "meta", "experiment_info.json"))

    print("\n" + "=" * 60)
    print("Image Classify Pretrain Complete!")
    print(f"  Best val mAP: {best_mAP:.4f}")
    print(f"  Output dir: {ckpt_dir}")
    print(f"  Best ckpt: {best_ckpt_path}")
    print("=" * 60)


# ===========================================================================
# 8. Logger (minimal implementation, tee to file)
# ===========================================================================
class Logger:
    """Write to stdout and a file simultaneously"""
    def __init__(self, log_path):
        self.terminal = sys.__stdout__
        self.log_file = open(log_path, "a", encoding="utf-8")

    def write(self, msg):
        self.terminal.write(msg)
        self.log_file.write(msg)
        self.log_file.flush()

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()


# ===========================================================================
# CONFIG & Entry
# ===========================================================================
if __name__ == "__main__":
    # # ---------- hunyuan-image-2_1 CONFIG ----------
    # CONFIG = {
    #     # ============================================================
    #     # 1. runtime
    #     # ============================================================
    #     "seed": 42,
    #     "device": "cuda:0",
    #     "ckpt_root": "./outputs/checkpoints",
    #     "torch_home": "/path/to/models/",
    #     "model": "hunyuan-image-2_1",
    #     "exp_tag": "",
    #
    #     # ============================================================
    #     # 2. pretrain mode
    #     # ============================================================
    #     "pretrain_mode": "distill",
    #
    #     # ============================================================
    #     # 3. data paths
    #     # ============================================================
    #     "latent_dir":   "/path/to/data/open-images-v7/train/latents_hunyuan_image_2_1",
    #     "images_dir":   "/path/to/data/open-images-v7/train/images",
    #     "labels_dir":   "/path/to/data/open-images-v7/train/labels",
    #     "classes_csv":  "/path/to/data/open-images-v7/train/labels/classes.csv",
    #     "label_format":      "openimages",
    #     "safety_label_csv": "",
    #     "val_size":      10000,
    #     "latent_num_steps": 0,
    #     "latent_val_step":  3,
    #
    #     # ============================================================
    #     # 4. model + stem config
    #     # ============================================================
    #     "backbone":            "convnext_base",
    #     "backbone_pretrained": True,
    #     "in_chans":            64,             # hunyuan-image-2_1 VAE channels
    #     "dropout":             0.1,
    #     "stem_type":           "single",       # main-run config
    #     "stem_channels":       [64, 80, 100, 128],  # progressive backup; not run
    #     "stem_init":           "trunc_normal",
    #     "train_scope":         "full",
    #
    #     # ============================================================
    #     # 5. distillation params
    #     # ============================================================
    #     "teacher_backbone":    "convnext_base",
    #     "teacher_pretrained":  True,
    #     "teacher_ckpt":        "",
    #     "teacher_image_size":  128,
    #     "teacher_image_mean":  [0.485, 0.456, 0.406],
    #     "teacher_image_std":   [0.229, 0.224, 0.225],
    #     "distill_loss":        "cosine",
    #     "distill_lambda":      1.0,
    #     "distill_stages":       [1, 2, 3, 4],
    #     "distill_stage_weight": 1.0,
    #     "distill_stage_pool":   "gap",
    #
    #     # ============================================================
    #     # 6. training hyperparams
    #     # ============================================================
    #     "max_iters":      100000,
    #     "eval_every_iters": 2000,
    #     "bs":             4,               # 64ch * 224^2 is heavy on GPU memory; lower bs
    #     "nw":             8,
    #     "lr":             1e-4,
    #     "backbone_lr":    5e-5,
    #     "head_lr":        5e-4,
    #     "weight_decay":   0.01,
    #     "lr_min":         1e-6,
    #     "grad_clip_max_norm": 1.0,
    #
    #     # ============================================================
    #     # 7. data preprocessing
    #     # ============================================================
    #     "target_hw": [224, 224],   # keep at 224: the stem weights only fit this input size
    #     "data_aug": False,
    #     "latent_enable_normalize": False,
    #     "latent_stats_num_samples": 1000,
    #     "num_preview": 30,
    # }

    # # ---------- flux2-klein-base-9b CONFIG ----------
    # CONFIG = {
    #     # ============================================================
    #     # 1. runtime
    #     # ============================================================
    #     "seed": 42,
    #     "device": "cuda:0",
    #     "ckpt_root": "./outputs/checkpoints",
    #     "torch_home": "/path/to/models/",
    #     "model": "flux2-klein-base-9b",
    #     "exp_tag": "",
    #
    #     # ============================================================
    #     # 2. pretrain mode
    #     # ============================================================
    #     "pretrain_mode": "distill",
    #
    #     # ============================================================
    #     # 3. data paths
    #     # ============================================================
    #     "latent_dir":   "/path/to/data/open-images-v7/train/latents_flux2_klein_base_9b",
    #     "images_dir":   "/path/to/data/open-images-v7/train/images",
    #     "labels_dir":   "/path/to/data/open-images-v7/train/labels",
    #     "classes_csv":  "/path/to/data/open-images-v7/train/labels/classes.csv",
    #     "label_format":      "openimages",
    #     "safety_label_csv": "",
    #     "val_size":      10000,
    #     "latent_num_steps": 0,
    #     "latent_val_step":  3,
    #
    #     # ============================================================
    #     # 4. model + stem config
    #     # ============================================================
    #     "backbone":            "convnext_base",
    #     "backbone_pretrained": True,
    #     "in_chans":            32,             # flux2-klein-base-9b VAE channels
    #     "dropout":             0.1,
    #     "stem_type":           "single",       # main-run config
    #     "stem_channels":       [32, 48, 80, 128],  # progressive backup; not run
    #     "stem_init":           "trunc_normal",
    #     "train_scope":         "full",
    #
    #     # ============================================================
    #     # 5. distillation params
    #     # ============================================================
    #     "teacher_backbone":    "convnext_base",
    #     "teacher_pretrained":  True,
    #     "teacher_ckpt":        "",
    #     "teacher_image_size":  128,
    #     "teacher_image_mean":  [0.485, 0.456, 0.406],
    #     "teacher_image_std":   [0.229, 0.224, 0.225],
    #     "distill_loss":        "cosine",
    #     "distill_lambda":      1.0,
    #     "distill_stages":       [1, 2, 3, 4],
    #     "distill_stage_weight": 1.0,
    #     "distill_stage_pool":   "gap",
    #
    #     # ============================================================
    #     # 6. training hyperparams
    #     # ============================================================
    #     "max_iters":      100000,
    #     "eval_every_iters": 2000,
    #     "bs":             8,               # 32ch * 224^2, roughly 2x memory
    #     "nw":             8,
    #     "lr":             1e-4,
    #     "backbone_lr":    5e-5,
    #     "head_lr":        5e-4,
    #     "weight_decay":   0.01,
    #     "lr_min":         1e-6,
    #     "grad_clip_max_norm": 1.0,
    #
    #     # ============================================================
    #     # 7. data preprocessing
    #     # ============================================================
    #     "target_hw": [224, 224],   # keep at 224: the stem weights only fit this input size
    #     "data_aug": False,
    #     "latent_enable_normalize": False,
    #     "latent_stats_num_samples": 1000,
    #     "num_preview": 30,
    # }

    # # ---------- qwen-image-2512 CONFIG ----------
    # CONFIG = {
    #     # ============================================================
    #     # 1. runtime
    #     # ============================================================
    #     "seed": 42,
    #     "device": "cuda:0",
    #     "ckpt_root": "./outputs/checkpoints",
    #     "torch_home": "/path/to/models/",
    #     "model": "qwen-image-2512",  # which model's latents; appears in the ckpt_dir path
    #
    #     # ============================================================
    #     # 2. pretrain mode
    #     # ============================================================
    #     "pretrain_mode": "distill",    # "classify" = multi-label classification | "distill" = feature distillation
    #
    #     # ============================================================
    #     # 3. data paths
    #     # ============================================================
    #     "latent_dir":   "/path/to/data/open-images-v7/train/latents_qwen_image_2512",
    #     "images_dir":   "/path/to/data/open-images-v7/train/images",
    #     # needed only when pretrain_mode="classify":
    #     "labels_dir":   "/path/to/data/open-images-v7/train/labels",
    #     "classes_csv":  "/path/to/data/open-images-v7/train/labels/classes.csv",
    #     "val_size":      10000,
    #
    #     # ============================================================
    #     # 4. model + stem config
    #     # ============================================================
    #     "backbone":            "convnext_base",
    #     "backbone_pretrained": True,           # start from IN1K weights
    #     "in_chans":            16,             # qwen-image-2512 VAE channels
    #     "dropout":             0.1,
    #     # stem structure parameters (must align with train_image.py)
    #     "stem_type":           "progressive",  # "single" = one-layer stem | "progressive" = progressive multi-layer stem
    #     "stem_channels":       [16, 32, 64, 128],  # per-layer channels [input ch, middle..., output ch]; progressive only
    #     "stem_init":           "trunc_normal",     # "expand_in1k" | "trunc_normal" | "kaiming"
    #     # freeze mode
    #     "train_scope":         "full",         # "full" = all params trained | "stem_only" = freeze backbone blocks, train stem + head only
    #     # ViT pretraining config (torchvision IN1K_V1; shares the loading path with CNN):
    #     # "backbone": "vit_b_16",   # or "vit_l_16"
    #
    #     # ============================================================
    #     # 5. distillation params (effective only when pretrain_mode="distill")
    #     # ============================================================
    #     "teacher_backbone":    "convnext_base",
    #     "teacher_pretrained":  True,
    #     "teacher_image_size":  128,            # 128 aligns with the student's 128x128 latents spatially; enables mid-layer distillation
    #     "teacher_image_mean":  [0.485, 0.456, 0.406],
    #     "teacher_image_std":   [0.229, 0.224, 0.225],
    #     "distill_loss":        "cosine",       # "cosine" | "mse" | "smooth_l1"
    #
    #     # stage-wise mid-layer distillation (needs teacher_image_size=128 for spatial alignment)
    #     "distill_stages":       [1, 2, 3, 4],  # empty [] = GAP only; [1,2,3,4] = all mid layers
    #     "distill_stage_weight": 1.0,           # total stage-loss weight (GAP loss weight is always 1.0)
    #     "distill_stage_pool":   "gap",         # "gap" = loss after GAP | "pixel" = per-pixel loss on feature maps
    #
    #     # ============================================================
    #     # 6. training hyperparams
    #     # ============================================================
    #     "max_iters":      500000,           # total training iterations
    #     "eval_every_iters": 2000,          # evaluate + save every N iters
    #     "bs":             16,
    #     "nw":             8,
    #     "lr":             1e-4,
    #     "backbone_lr":    5e-5,              # small backbone lr (classify mode only)
    #     "head_lr":        5e-4,              # larger head lr (classify mode only)
    #     "weight_decay":   0.01,
    #     "lr_min":         1e-6,
    #     "grad_clip_max_norm": 1.0,
    #
    #     # ============================================================
    #     # 7. data preprocessing
    #     # ============================================================
    #     "target_hw": [128, 128],   # fixed latent output size [H, W]; crop first, pad when too small
    #     "data_aug": False,         # spatial aug off in distill mode to keep teacher-student feature spaces aligned
    #     # latent per-channel normalization switch: False = no normalization; use raw latent values
    #     "latent_enable_normalize": False,
    #     "latent_stats_num_samples": 1000,
    #     "num_preview": 30,
    # }

    # Paths can be overridden via environment variables (see the README "Configuration" section):
    #   INGUARD_MODELS_ROOT / INGUARD_DATA_ROOT / INGUARD_CKPT_ROOT
    _MODELS_ROOT = os.environ.get("INGUARD_MODELS_ROOT", "/path/to/models")
    _DATA_ROOT   = os.environ.get("INGUARD_DATA_ROOT", "/path/to/data")
    _CKPT_ROOT   = os.environ.get("INGUARD_CKPT_ROOT", "./outputs/checkpoints")

    # ---------- z-image-turbo CONFIG ----------
    CONFIG = {
        # ============================================================
        # 1. runtime
        # ============================================================
        "seed": 42,
        "device": "cuda:0",
        "ckpt_root": _CKPT_ROOT,
        "torch_home": f"{_MODELS_ROOT}/",
        "model": "z-image-turbo",  # which model's latents; appears in the ckpt_dir path
        "exp_tag": "",               # experiment tag; when non-empty the dir is named {timestamp}_{exp_tag}

        # ============================================================
        # 2. pretrain mode
        # ============================================================
        # "classify"       = latent multi-label classification (latent tensor input; OpenImagesLatentDataset)
        # "distill"        = latent feature distillation (teacher RGB -> student latent feature alignment)
        # "classify_image" = image multi-label classification (RGB image input; OpenImagesImageClassifyDataset, control experiment)
        "pretrain_mode": "distill",

        # ============================================================
        # 3. data paths
        # ============================================================
        "latent_dir":   f"{_DATA_ROOT}/open-images-v7/train/latents",  # used by distill/classify modes
        "images_dir":   f"{_DATA_ROOT}/open-images-v7/train/images",  # distill (teacher), classify (preview), classify_image (main data source)
        # needed when pretrain_mode="classify" or "classify_image":
        "labels_dir":   f"{_DATA_ROOT}/open-images-v7/train/labels",
        "classes_csv":  f"{_DATA_ROOT}/open-images-v7/train/labels/classes.csv",
        # needed only for distill_classify + the safety dataset (script patch):
        "label_format":      "openimages",   # "openimages" | "safety_csv"
        "safety_label_csv": "",              # points to predictions.csv when label_format="safety_csv"
        "val_size":      10000,             # val-set sample count (random from the manifest; fixed seed, reproducible)
        # step-based latent directory layout (for safety-domain data; keep 0 = flat mode for OpenImages)
        "latent_num_steps": 0,                 # 0=flat mode (OpenImages), >0=step-subdir mode (safety dataset)
        "latent_val_step":  3,                 # the fixed step used by eval in step mode

        # ============================================================
        # 4. model + stem config
        # ============================================================
        "backbone":            "convnext_base",
        "backbone_pretrained": True,           # start from IN1K weights
        "in_chans":            16,             # latent channels (patched to 3 by the script in classify_image mode)
        "dropout":             0.1,
        # stem structure parameters (distill/classify modes must align with train_image.py)
        # classify_image mode uses stem_type="original" (original stem kept; patched by the script)
        "stem_type":           "progressive",  # "single" | "progressive" | "original" (original is for classify_image only)
        "stem_channels":       [16, 32, 64, 128],  # per-layer channels [input ch, middle..., output ch]; progressive only
        "stem_init":           "trunc_normal",     # "expand_in1k" | "trunc_normal" | "kaiming" (ignored in original mode)
        # freeze mode
        "train_scope":         "full",          # "full" = all params trained | "stem_only" = freeze backbone blocks, train stem + head only
        # ViT pretraining config (torchvision IN1K_V1; shares the loading path with CNN):
        # "backbone": "vit_b_16",   # or "vit_l_16"

        # ============================================================
        # 5. distillation params (effective when pretrain_mode="distill" or "distill_classify")
        # ============================================================
        "teacher_backbone":    "convnext_base",
        "teacher_pretrained":  True,
        "teacher_ckpt":        "",             # custom teacher ckpt path (empty string = use IN1K)
        "teacher_image_size":  224,            # matches target_hw 224x224; enables mid-layer distillation spatial alignment
        "teacher_image_mean":  [0.485, 0.456, 0.406],
        "teacher_image_std":   [0.229, 0.224, 0.225],
        "distill_loss":        "cosine",       # "cosine" | "mse" | "smooth_l1"
        "distill_lambda":      1.0,            # distillation loss weight (distill_classify mode only): L = L_cls + lambda*L_distill

        # stage-wise mid-layer distillation (distill mode only; needs teacher_image_size == target_hw for spatial alignment)
        # distill_stages: the mid stages to distill; empty = GAP final features only
        #   stage1: 128ch 32x32, stage2: 256ch 16x16, stage3: 512ch 8x8, stage4: 1024ch 4x4
        "distill_stages":       [1, 2, 3, 4],  # empty [] = GAP only; [1,2,3,4] = all mid layers
        "distill_stage_weight": 1.0,           # total stage-loss weight (GAP loss weight is always 1.0)
        "distill_stage_pool":   "gap",         # "gap" = loss after GAP | "pixel" = per-pixel loss on feature maps

        # ============================================================
        # 6. training hyperparams
        # ============================================================
        "max_iters":      100000,           # total training iterations
        "eval_every_iters": 2000,          # evaluate + save every N iters
        "bs":             8,  # 224x224 feature maps are large; lower bs to avoid OOM
        "nw":             8,
        "lr":             1e-4,
        "backbone_lr":    5e-5,              # small backbone lr (classify mode only)
        "head_lr":        5e-4,              # larger head lr (classify mode only)
        "weight_decay":   0.01,
        "lr_min":         1e-6,
        "grad_clip_max_norm": 1.0,

        # ============================================================
        # 7. data preprocessing
        # ============================================================
        "target_hw": [224, 224],   # fixed latent output size [H, W] — unified 224x224, aligned with Image/IN1K
        "data_aug": False,         # distill/classify: False (latents need spatial alignment) | classify_image: True (RGB images augment safely)
        # latent per-channel normalization switch: False = no normalization; use raw latent values
        "latent_enable_normalize": False,
        "latent_stats_num_samples": 1000,
        "num_preview": 30,
    }

    run_pretrain(CONFIG)
