"""
train_prompt.py — multi-task safety classification training script based on prompt_embeds

Each generative model trains its own MLP; Masked Mean Pooling pools the
variable-length prompt_embeds into a fixed-dim vector feeding three classification heads (porn/gore/ip).

Purpose: front-end gating — quickly judge the prompt risk level and route potentially risky traffic to the downstream latent/image models.

Supported image generation models (--model_name values):
  - z-image-turbo       : text_dim=2560,  prompt_embeds format list[tensor[seq_len, D]],  no mask
  - qwen-image-2512     : text_dim=3584,  prompt_embeds format tensor[1, seq_len, D],    no mask
  - internvl-u          : text_dim=4096,  prompt_embeds format tensor[3, seq_len, D],    has mask (prompt_attention_mask_forward)
                                   note: VLM (Qwen2.5, hidden=2048) takes the last 2 layers concatenated → D=2x2048=4096; decoder_projector=Identity (no projection)
  - hunyuan-image-2_1   : text_dim=3584,  prompt_embeds format tensor[1, seq_len, D],    has mask (prompt_embeds_mask_forward)
  - flux2-klein-base-9b : text_dim=12288, prompt_embeds format tensor[1, seq_len, D],    has mask (prompt_attention_mask_forward)

  input_dim is inferred automatically by probe_data() from the actual .pth files; no manual specification needed.
  The mask is auto-detected by resolve_mask_dir(): Masked Mean Pooling when a mask exists, plain Mean Pooling otherwise.

Model structure:
  Mean Pooling / Masked Mean Pooling([seq_len, D] → [D])
    → Linear(D, hidden_dim) + LayerNorm + GELU
    → N × (Linear + LayerNorm + GELU + Dropout + Residual)
    → head_porn (6 classes, 0-5)
    → head_gore (6 classes, 0-5)
    → head_ip   (8 classes, 0-7)

Prompt label system:
  - porn: 0=Safe, 1=Mild, 2=Suggestive Attire, 3=Strong Suggestive, 4=Near-Explicit, 5=Explicit
  - gore: 0=Safe, 1=Violent Action, 2=Violent with Harm, 3=Mild Tension, 4=Bloody Suggestion, 5=Extreme Gore
  - ip:   0=Safe, 1=Snow White, 2=Doraemon, 3=Minion, 4=Elsa, 5=SpongeBob, 6=Associated IP, 7=Other

GT gating-evaluation definitions (against the actual image labels):
  - porn positive: gt.pornographic == 2   (0=safe, 1=borderline counted as non-porn)
  - gore positive: gt.violence_gore == 2  (0=safe, 1=violence counted as non-gore)
  - IP positive:   gt.ip_copyright ∈ {1,2,3,4,5}  (only the 5 concrete characters, excluding 6/7)

Data directory structure:
  {data_root}/{model_name}/
  ├── trainset-xxx/                ← train_data_dir (auto-matches the subdir containing "trainset")
  │   ├── prompt_embeds_forward/          ← per-sample prompt_embeds (.pth)
  │   ├── prompt_embeds_mask_forward/     ← attention_mask (.pth) (hunyuan-image-2_1)
  │   └── prompt_attention_mask_forward/  ← attention_mask (.pth) (flux2-klein-base-9b, internvl-u)
  └── testset-xxx/                 ← test_data_dir (auto-matches the subdir containing "testset" or "eval")
      └── (same structure as above)
  Note: z-image-turbo and qwen-image-2512 have no mask dir; plain mean pooling is used directly.

Output directory (auto-generated):
  {ckpt_root}/{today}/{model_name}/prompt-mlp-multitask-porn-gore-ip/{datetime_BS_LR_H_L_drop_seed}/

Usage:
    # simplest — auto-infer train_data_dir / test_data_dir / output_dir
    python train_prompt.py \
        --model_name z-image-turbo \
        --train_csv /path/to/trainset_labeled_class.csv \
        --test_csv /path/to/testset_labeled_class.csv \
        --gt_csv /path/to/predictions_checked.csv

    # simplest — CSV and data dirs all use the built-in defaults
    python train_prompt.py --model_name z-image-turbo

    # manually specify the data dirs
    python train_prompt.py \
        --model_name qwen-image-2512 \
        --train_data_dir /data/.../trainset-seed42-1328-10steps \
        --test_data_dir /data/.../testset-seed42-1328-10steps \
        --train_csv ... --test_csv ... --gt_csv ...
"""

import os
import sys
import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler
import pandas as pd
import numpy as np
from tqdm import tqdm

# --- repo layout: shared training utilities live in <repo>/common/ ---
import os as _os, sys as _sys
_common = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "common")
if _common not in _sys.path:
    _sys.path.insert(0, _common)
from utils_common import (
    Logger,
    seed_everything,
    print_section,
    print_table,
    print_kv_table,
    save_json,
    save_history_and_plots,
    snapshot_code_dir,
    collate_fn,
)


# ========================== formatting utilities ==========================

def fmt_pct_ratio(num, den):
    """Return a percentage string in the "xx.xx% (num/den)" format."""
    num = int(num)
    den = int(den)
    if den <= 0:
        return f"0.00% ({num}/{den})"
    return f"{num / den * 100:.2f}% ({num}/{den})"


# ========================== data loading utilities ==========================

def load_prompt_embeds(pth_path):
    """Uniformly load prompt_embeds in various formats → [seq_len, D]

    Supported formats:
      - list[tensor[seq_len, D]]       (z-image-turbo)
      - tensor[1, seq_len, D]          (qwen/hunyuan/flux2, batch=1)
      - tensor[3, seq_len, D]          (internvl-u, triple CFG, take cond=index 0)
      - tensor[seq_len, D]             (already pooled, or no batch dim)
    """
    data = torch.load(pth_path, map_location='cpu', weights_only=True)
    if isinstance(data, list):
        # Z-Image-Turbo format: list[tensor[seq_len, D]]
        t = data[0]
        if t.dim() == 2:
            return t
        elif t.dim() == 3:
            return t.squeeze(0)
    elif isinstance(data, torch.Tensor):
        if data.dim() == 3:
            if data.shape[0] == 1:
                # regular: [1, seq_len, D] → [seq_len, D]
                return data.squeeze(0)
            else:
                # InternVL-U triple CFG: [3, seq_len, D]
                # order: [0]=cond, [1]=part_cond, [2]=uncond
                # (in the pipeline noise_pred.chunk(3) → cond, part_cond, uncond)
                # take the fully-conditional embedding (the first) as the classification input
                return data[0]
        elif data.dim() == 2:  # [seq_len, D]
            return data
    raise ValueError(f"Unexpected prompt_embeds format in {pth_path}: type={type(data)}")


def load_prompt_mask(pth_path):
    """Load a mask → [seq_len] or None

    Supported formats:
      - tensor[1, seq_len]   (hunyuan/flux2/internvl-u, batch=1)
      - tensor[3, seq_len]   (compatible with the triple-CFG full mask, take cond=index 0)
      - tensor[seq_len]     (no batch dim)
    """
    if pth_path is None or not os.path.exists(pth_path):
        return None
    data = torch.load(pth_path, map_location='cpu', weights_only=True)
    if data is None:
        return None
    if isinstance(data, torch.Tensor):
        if data.dim() == 2:
            if data.shape[0] == 1:
                # regular: [1, seq_len] → [seq_len]
                return data.squeeze(0)
            else:
                # InternVL-U triple CFG: [3, seq_len] → take the cond (index 0) mask
                return data[0]
        elif data.dim() == 1:  # [seq_len]
            return data
    return None


def masked_mean_pool(embeds, mask=None):
    """
    embeds: [seq_len, D]
    mask: [seq_len] (0/1) or None
    returns: [D]
    """
    embeds = embeds.float()
    if mask is None:
        return embeds.mean(dim=0)
    mask_f = mask.float().unsqueeze(-1)  # [seq_len, 1]
    valid_sum = mask_f.sum(dim=0).clamp(min=1.0)
    return (embeds * mask_f).sum(dim=0) / valid_sum


def resolve_mask_dir(data_dir):
    """
    Resolve the mask directory path.
    - hunyuan-image-2_1: prompt_embeds_mask_forward/
    - flux2-klein-base-9b / internvl-u: prompt_attention_mask_forward/ (fallback)
    - z-image-turbo / qwen-image-2512: no mask dir; the standard path is returned but does not exist,
      so load_prompt_mask later returns None → plain mean pooling
    """
    standard = os.path.join(data_dir, "prompt_embeds_mask_forward")
    if os.path.isdir(standard):
        return standard
    fallback = os.path.join(data_dir, "prompt_attention_mask_forward")
    if os.path.isdir(fallback):
        return fallback
    return standard  # dir does not exist → load_prompt_mask returns None → mean pooling


def probe_data(data_dir, csv_path, num_samples=5):
    """
    Probe the data format before training; auto-infer input_dim and mask availability.
    Returns: (input_dim, has_mask)
    """
    df = pd.read_csv(csv_path, usecols=['id'], dtype=str, nrows=500)
    embeds_dir = os.path.join(data_dir, "prompt_embeds_forward")
    mask_dir = resolve_mask_dir(data_dir)

    print_section("[Data Probe] probing the data format")
    print(f"  embeds_dir: {embeds_dir}")
    print(f"  mask_dir:   {mask_dir}")

    found = 0
    input_dim = None
    has_mask = False

    for _, row in df.iterrows():
        event_id = str(row['id']).strip()
        embed_path = os.path.join(embeds_dir, f"{event_id}.pth")
        if not os.path.exists(embed_path):
            continue

        embeds = load_prompt_embeds(embed_path)
        mask_path = os.path.join(mask_dir, f"{event_id}.pth")
        mask = load_prompt_mask(mask_path)

        if input_dim is None:
            input_dim = embeds.shape[-1]

        if mask is not None:
            has_mask = True
            valid_tokens = int(mask.sum().item())
            total_tokens = mask.shape[0]
            print(f"  Sample {found}: embeds shape={list(embeds.shape)}, "
                  f"mask shape={list(mask.shape)}, valid_tokens={valid_tokens}/{total_tokens}")
        else:
            print(f"  Sample {found}: embeds shape={list(embeds.shape)}, mask=None")

        found += 1
        if found >= num_samples:
            break

    if input_dim is None:
        raise RuntimeError(f"no valid prompt_embeds files found under {embeds_dir}!")

    print(f"\n  → inferred input_dim = {input_dim}")
    print(f"  → mask: {'enabled (Masked Mean Pooling)' if has_mask else 'unused (Mean Pooling)'}")
    print()

    return input_dim, has_mask


# ========================== Dataset ==========================

class PromptEmbedsDataset(Dataset):
    """
    Loads prompt_embeds and does the pooling inside __getitem__,
    returning (pooled_embed[D], label_porn, label_gore, label_ip).
    """

    def __init__(self, csv_path, data_dir, has_mask=True):
        self.data_dir = data_dir
        self.has_mask = has_mask
        self.embeds_dir = os.path.join(data_dir, "prompt_embeds_forward")
        self.mask_dir = resolve_mask_dir(data_dir)

        df = pd.read_csv(csv_path, dtype=str)
        df = df.dropna(subset=['id', 'label_porn_risk_level', 'label_gore_risk_level', 'label_ip_risk_level'])
        self.samples = df.reset_index(drop=True)
        print(f"  Dataset: {len(self.samples)} rows from CSV (embeds filtered dynamically during training)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        row = self.samples.iloc[idx]
        event_id = str(row['id']).strip()

        # load the embeds
        embed_path = os.path.join(self.embeds_dir, f"{event_id}.pth")
        if not os.path.exists(embed_path):
            return None  # filtered out by collate_fn

        try:
            embeds = load_prompt_embeds(embed_path)
        except Exception:
            return None

        # load the mask
        mask = None
        if self.has_mask:
            mask_path = os.path.join(self.mask_dir, f"{event_id}.pth")
            mask = load_prompt_mask(mask_path)

        # Pooling → [D]
        pooled = masked_mean_pool(embeds, mask)

        # Labels
        label_porn = int(row['label_porn_risk_level'])
        label_gore = int(row['label_gore_risk_level'])
        label_ip = int(row['label_ip_risk_level'])

        return pooled, label_porn, label_gore, label_ip


# ========================== Model ==========================

class PromptMultiTaskMLP(nn.Module):
    """
    Multi-task classification MLP over prompt embeddings.
    Structure: Projection → N × (Linear + LN + GELU + Dropout + Residual) → three classification heads
    """

    def __init__(self, input_dim, hidden_dim=1024, num_layers=3, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.proj_ln = nn.LayerNorm(hidden_dim)

        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            for _ in range(num_layers)
        ])

        self.head_porn = nn.Linear(hidden_dim, 6)   # 0-5
        self.head_gore = nn.Linear(hidden_dim, 6)   # 0-5
        self.head_ip = nn.Linear(hidden_dim, 8)     # 0-7

    def forward(self, x):
        """x: [B, input_dim] → (logits_porn, logits_gore, logits_ip)"""
        x = self.proj_ln(F.gelu(self.proj(x)))
        for block in self.blocks:
            x = x + block(x)
        return self.head_porn(x), self.head_gore(x), self.head_ip(x)


# ========================== evaluation ==========================

@torch.no_grad()
def evaluate_prompt_metrics(model, loader, device, max_batches=None, desc="Evaluating"):
    """Evaluate multi-class accuracy on the prompt labels; also computes test_loss.
    max_batches: cap on how many batches are evaluated (None = all)
    """
    model.eval()
    all_pred_porn, all_pred_gore, all_pred_ip = [], [], []
    all_gt_porn, all_gt_gore, all_gt_ip = [], [], []
    total_loss = 0.0
    n_samples = 0

    total = min(len(loader), max_batches) if max_batches else len(loader)
    pbar = tqdm(loader, total=total, desc=desc, leave=False)
    batch_count = 0
    for batch in pbar:
        if batch is None:
            continue
        pooled, lp, lg, li = batch
        pooled = pooled.to(device)
        lp = lp.to(device)
        lg = lg.to(device)
        li = li.to(device)

        logits_porn, logits_gore, logits_ip = model(pooled)

        loss_porn = F.cross_entropy(logits_porn, lp)
        loss_gore = F.cross_entropy(logits_gore, lg)
        loss_ip = F.cross_entropy(logits_ip, li)
        loss = loss_porn + loss_gore + loss_ip

        bs = pooled.size(0)
        total_loss += loss.item() * bs
        n_samples += bs

        all_pred_porn.append(logits_porn.argmax(dim=1).cpu())
        all_pred_gore.append(logits_gore.argmax(dim=1).cpu())
        all_pred_ip.append(logits_ip.argmax(dim=1).cpu())
        all_gt_porn.append(lp.cpu())
        all_gt_gore.append(lg.cpu())
        all_gt_ip.append(li.cpu())

        batch_count += 1
        if max_batches and batch_count >= max_batches:
            break
    pbar.close()

    pred_porn = torch.cat(all_pred_porn).numpy()
    pred_gore = torch.cat(all_pred_gore).numpy()
    pred_ip = torch.cat(all_pred_ip).numpy()
    gt_porn = torch.cat(all_gt_porn).numpy()
    gt_gore = torch.cat(all_gt_gore).numpy()
    gt_ip = torch.cat(all_gt_ip).numpy()

    acc_porn = (pred_porn == gt_porn).mean()
    acc_gore = (pred_gore == gt_gore).mean()
    acc_ip = (pred_ip == gt_ip).mean()
    test_loss = total_loss / max(n_samples, 1)

    n_total = len(pred_porn)
    n_correct_porn = int((pred_porn == gt_porn).sum())
    n_correct_gore = int((pred_gore == gt_gore).sum())
    n_correct_ip = int((pred_ip == gt_ip).sum())

    return {
        'acc_porn': acc_porn,
        'acc_gore': acc_gore,
        'acc_ip': acc_ip,
        'test_loss': test_loss,
        'pred_porn': pred_porn,
        'pred_gore': pred_gore,
        'pred_ip': pred_ip,
        'gt_porn': gt_porn,
        'gt_gore': gt_gore,
        'gt_ip': gt_ip,
        'n_total': n_total,
        'n_correct_porn': n_correct_porn,
        'n_correct_gore': n_correct_gore,
        'n_correct_ip': n_correct_ip,
    }


def save_badcases(metrics, test_csv, save_dir, step_label="final"):
    """
    Save per-head badcases (samples where pred != GT) to CSV.
    Reads the id and prompt fields from test_csv.
    Each head outputs one file: badcase_porn_{step}.csv, badcase_gore_{step}.csv, badcase_ip_{step}.csv
    """
    os.makedirs(save_dir, exist_ok=True)

    # read id and prompt from test_csv (same filtering logic as the Dataset)
    df = pd.read_csv(test_csv, dtype=str)
    df = df.dropna(subset=['label_porn_risk_level', 'label_gore_risk_level', 'label_ip_risk_level'])
    df = df.reset_index(drop=True)
    # make sure a prompt column exists
    if 'prompt' not in df.columns:
        print(f"  [Badcase] ✖ no 'prompt' column in test_csv, skipping badcase saving")
        return

    ids = df['id'].values
    prompts = df['prompt'].values

    pred_porn = metrics['pred_porn']
    pred_gore = metrics['pred_gore']
    pred_ip = metrics['pred_ip']
    gt_porn = metrics['gt_porn']
    gt_gore = metrics['gt_gore']
    gt_ip = metrics['gt_ip']

    n = len(pred_porn)
    # ensure alignment
    if len(ids) < n:
        print(f"  [Badcase] ✖ test_csv rows({len(ids)}) < predictions({n}), skipping")
        return

    for task_name, pred, gt in [
        ('porn', pred_porn, gt_porn),
        ('gore', pred_gore, gt_gore),
        ('ip', pred_ip, gt_ip),
    ]:
        mask = pred != gt
        if mask.sum() == 0:
            print(f"  [Badcase] {task_name}: 0 badcases")
            continue

        bad_ids = ids[:n][mask]
        bad_prompts = prompts[:n][mask]
        bad_pred = pred[mask]
        bad_gt = gt[mask]

        df_bad = pd.DataFrame({
            'id': bad_ids,
            'prompt': bad_prompts,
            f'pred_{task_name}': bad_pred,
            f'gt_{task_name}': bad_gt,
        })
        out_path = os.path.join(save_dir, f"badcase_{task_name}_{step_label}.csv")
        df_bad.to_csv(out_path, index=False)
        print(f"  [Badcase] {task_name}: {mask.sum()} cases → {out_path}")


# ========================== gating evaluation ==========================

def save_gating_badcases(pred_porn, pred_gore, pred_ip, test_csv, gt_csv, save_dir, step_label,
                         porn_T=3, gore_T=2):
    """
    Save the gating missed-recall and false-alarm badcases.
    Gating rule: pred_porn >= porn_T OR pred_gore >= gore_T OR pred_ip ∈ {1-5}

    missed: GT is unsafe but not gated in
    false_alarm: GT is safe but gated in

    Two files are saved each time:
      - gating_missed_{step_label}.csv
      - gating_false_alarm_{step_label}.csv
    """
    os.makedirs(save_dir, exist_ok=True)

    # read test_csv (same filtering logic as the Dataset)
    df_test = pd.read_csv(test_csv, dtype=str)
    df_test = df_test.dropna(subset=['label_porn_risk_level', 'label_gore_risk_level', 'label_ip_risk_level'])
    df_test = df_test.reset_index(drop=True)
    if 'prompt' not in df_test.columns:
        print(f"  [GatingBadcase] ✖ no 'prompt' column in test_csv, skipping")
        return

    # read the GT (defensive: drop rows with an empty filename to avoid TypeError from trailing blank lines)
    df_gt = pd.read_csv(gt_csv, dtype=str)
    n_before = len(df_gt)
    df_gt = df_gt.dropna(subset=['filename'])
    n_dropped = n_before - len(df_gt)
    if n_dropped > 0:
        print(f"  [GatingBadcase][GT] dropped {n_dropped} rows with empty filename ({n_before} rows → {len(df_gt)} valid)")
    df_gt['event_id'] = df_gt['filename'].apply(lambda x: os.path.splitext(x)[0])
    df = df_test.merge(df_gt[['event_id', 'pornographic', 'violence_gore', 'ip_copyright']],
                       left_on='id', right_on='event_id', how='inner')
    # drop failed-labeling rows so the count aligns with what the DataLoader actually evaluated
    df = df.dropna(subset=['pornographic', 'violence_gore', 'ip_copyright'])
    df = df.reset_index(drop=True)

    n = len(pred_porn)
    if len(df) != n:
        print(f"  [GatingBadcase] ⚠️ merged rows({len(df)}) != predictions({n}); truncating to align")
        df = df.iloc[:n].copy()

    gt_porn_val = pd.to_numeric(df['pornographic'], errors='coerce').fillna(0).astype(int).values
    gt_gore_val = pd.to_numeric(df['violence_gore'], errors='coerce').fillna(0).astype(int).values
    gt_ip_val = pd.to_numeric(df['ip_copyright'], errors='coerce').fillna(0).astype(int).values

    gt_porn_pos = (gt_porn_val == 2)
    gt_gore_pos = (gt_gore_val == 2)
    gt_ip_pos = np.isin(gt_ip_val, [1, 2, 3, 4, 5])
    gt_any_unsafe = gt_porn_pos | gt_gore_pos | gt_ip_pos

    # gating flags
    ip_flagged = np.isin(pred_ip, [1, 2, 3, 4, 5])
    flagged = (pred_porn >= porn_T) | (pred_gore >= gore_T) | ip_flagged

    # missed: GT unsafe but not gated in
    missed_mask = gt_any_unsafe & (~flagged)
    # false alarm: GT safe but gated in
    false_alarm_mask = (~gt_any_unsafe) & flagged

    ids = df['id'].values
    prompts = df['prompt'].values

    # missed
    if missed_mask.sum() > 0:
        df_missed = pd.DataFrame({
            'id': ids[missed_mask],
            'prompt': prompts[missed_mask],
            'pred_porn': pred_porn[missed_mask],
            'pred_gore': pred_gore[missed_mask],
            'pred_ip': pred_ip[missed_mask],
            'gt_porn': gt_porn_val[missed_mask],
            'gt_gore': gt_gore_val[missed_mask],
            'gt_ip': gt_ip_val[missed_mask],
        })
        path = os.path.join(save_dir, f"gating_missed_{step_label}.csv")
        df_missed.to_csv(path, index=False)
        print(f"  [GatingBadcase] missed: {missed_mask.sum()} cases → {path}")
    else:
        print(f"  [GatingBadcase] missed: 0 (porn_T={porn_T}, gore_T={gore_T})")

    # false alarm
    if false_alarm_mask.sum() > 0:
        df_fa = pd.DataFrame({
            'id': ids[false_alarm_mask],
            'prompt': prompts[false_alarm_mask],
            'pred_porn': pred_porn[false_alarm_mask],
            'pred_gore': pred_gore[false_alarm_mask],
            'pred_ip': pred_ip[false_alarm_mask],
            'gt_porn': gt_porn_val[false_alarm_mask],
            'gt_gore': gt_gore_val[false_alarm_mask],
            'gt_ip': gt_ip_val[false_alarm_mask],
        })
        path = os.path.join(save_dir, f"gating_false_alarm_{step_label}.csv")
        df_fa.to_csv(path, index=False)
        print(f"  [GatingBadcase] false alarm: {false_alarm_mask.sum()} cases → {path}")
    else:
        print(f"  [GatingBadcase] false alarm: 0 (porn_T={porn_T}, gore_T={gore_T})")


def eval_gating_metrics(pred_porn, pred_gore, pred_ip, test_csv, gt_csv, save_path=None):
    """
    Gating-effectiveness evaluation based on model predictions + image/video GT.
    When save_path is not None, saves the combined-threshold table as CSV.

    GT definitions:
      image side:
        - porn positive: gt.pornographic == 2
        - gore positive: gt.violence_gore == 2
        - IP positive: gt.ip_copyright ∈ {1,2,3,4,5}
      Prompt side (GT aligned with the gating thresholds):
        - positive: label_porn >= porn_T OR label_gore >= gore_T OR label_ip ∈ {1-5}
        - negative: none of the above

    Returns: a dict with metrics at each threshold
    """
    # load the GT (defensive: drop rows with an empty filename to avoid TypeError from trailing blank lines)
    df_test = pd.read_csv(test_csv, dtype=str)
    # same filtering logic as the Dataset: drop rows whose labels are NaN
    df_test = df_test.dropna(subset=['label_porn_risk_level', 'label_gore_risk_level', 'label_ip_risk_level'])
    df_test = df_test[['id', 'label_porn_risk_level', 'label_gore_risk_level', 'label_ip_risk_level']].reset_index(drop=True)
    df_gt = pd.read_csv(gt_csv, dtype=str)
    n_before = len(df_gt)
    df_gt = df_gt.dropna(subset=['filename'])
    n_dropped = n_before - len(df_gt)
    if n_dropped > 0:
        print(f"  [GT] dropped {n_dropped} rows with empty filename ({n_before} rows → {len(df_gt)} valid)")
    df_gt['event_id'] = df_gt['filename'].apply(lambda x: os.path.splitext(x)[0])

    # merge, keeping the order consistent
    df = df_test.merge(df_gt[['event_id', 'pornographic', 'violence_gore', 'ip_copyright']],
                       left_on='id', right_on='event_id', how='inner')
    # drop failed-labeling rows (status=error → NaN labels) so the count aligns with what the DataLoader actually evaluated
    n_before_label_filter = len(df)
    df = df.dropna(subset=['pornographic', 'violence_gore', 'ip_copyright'])
    n_label_dropped = n_before_label_filter - len(df)
    if n_label_dropped > 0:
        print(f"  [GT] dropped {n_label_dropped} rows with missing labels (after merge {n_before_label_filter} → {len(df)} valid)")

    # safety check: ensure the GT row count matches the prediction count
    n_pred = len(pred_porn)
    if len(df) != n_pred:
        print(f"  [GT] ⚠️ valid GT rows({len(df)}) != predictions({n_pred}); truncating to align")
        df = df.iloc[:n_pred].reset_index(drop=True)
    gt_porn_val = pd.to_numeric(df['pornographic'], errors='coerce').fillna(0).astype(int).values
    gt_gore_val = pd.to_numeric(df['violence_gore'], errors='coerce').fillna(0).astype(int).values
    gt_ip_val = pd.to_numeric(df['ip_copyright'], errors='coerce').fillna(0).astype(int).values

    # prompt-side labels (for the prompt-side Recall / FalseAlarm)
    prompt_porn_val = pd.to_numeric(df['label_porn_risk_level'], errors='coerce').fillna(0).astype(int).values
    prompt_gore_val = pd.to_numeric(df['label_gore_risk_level'], errors='coerce').fillna(0).astype(int).values
    prompt_ip_val = pd.to_numeric(df['label_ip_risk_level'], errors='coerce').fillna(0).astype(int).values

    # GT definition: only ==2 counts as positive (image side)
    gt_porn_pos = (gt_porn_val == 2)
    gt_porn_neg = (gt_porn_val <= 1)
    gt_gore_pos = (gt_gore_val == 2)
    gt_gore_neg = (gt_gore_val <= 1)
    gt_ip_pos = np.isin(gt_ip_val, [1, 2, 3, 4, 5])  # only the 5 concrete characters
    gt_any_unsafe = gt_porn_pos | gt_gore_pos | gt_ip_pos

    n_total = len(df)
    n_porn_pos = gt_porn_pos.sum()
    n_gore_pos = gt_gore_pos.sum()
    n_ip_pos = gt_ip_pos.sum()
    n_safe = (~gt_any_unsafe).sum()

    print_section("[Gating Evaluation] based on image/video GT")
    print(f"  total samples: {n_total}")
    print(f"  Porn positive (GT=2): {n_porn_pos}")
    print(f"  Gore positive (GT=2): {n_gore_pos}")
    print(f"  IP positive (GT∈1-5): {n_ip_pos}")
    print(f"  fully safe (none of the above): {n_safe}")

    # ---- single dimension: Porn ----
    print_section("Porn single-dimension threshold sweep", char="-")
    rows = []
    n_porn_neg = int(gt_porn_neg.sum())
    for T in [1, 2, 3, 4, 5]:
        flagged = (pred_porn >= T)
        recall_num = int(flagged[gt_porn_pos].sum())
        fa_num = int(flagged[gt_porn_neg].sum())
        rows.append([f"≥{T}",
                     fmt_pct_ratio(recall_num, n_porn_pos),
                     fmt_pct_ratio(fa_num, n_porn_neg)])
    print_table(["threshold", "Recall(porn=2)", "FalseAlarm(non-porn)"], rows)
    
    # ---- single dimension: Gore ----
    print_section("Gore single-dimension threshold sweep", char="-")
    rows = []
    n_gore_neg = int(gt_gore_neg.sum())
    for T in [1, 2, 3, 4, 5]:
        flagged = (pred_gore >= T)
        recall_num = int(flagged[gt_gore_pos].sum())
        fa_num = int(flagged[gt_gore_neg].sum())
        rows.append([f"≥{T}",
                     fmt_pct_ratio(recall_num, n_gore_pos),
                     fmt_pct_ratio(fa_num, n_gore_neg)])
    print_table(["threshold", "Recall(gore=2)", "FalseAlarm(non-gore)"], rows)
    
    # ---- per-character IP recall ----
    ip_names = {1: 'Snow White', 2: 'Doraemon', 3: 'Minion', 4: 'Elsa', 5: 'SpongeBob'}
    print_section("Per-character IP recall", char="-")
    rows = []
    for cls, name in ip_names.items():
        gt_cls = (gt_ip_val == cls)
        n_cls = int(gt_cls.sum())
        hit = int(((pred_ip == cls) & gt_cls).sum())
        rows.append([name, str(n_cls), str(hit), fmt_pct_ratio(hit, n_cls)])
    
    # IP overall (1-5)
    ip_flagged = np.isin(pred_ip, [1, 2, 3, 4, 5])
    ip_recall_num = int(ip_flagged[gt_ip_pos].sum())
    ip_fa_num = int(ip_flagged[~gt_ip_pos].sum())
    n_ip_neg = int((~gt_ip_pos).sum())
    rows.append(["IP overall (1-5)", str(int(n_ip_pos)), str(ip_recall_num),
                 fmt_pct_ratio(ip_recall_num, n_ip_pos)])
    rows.append(["IP false alarm", str(n_ip_neg), str(ip_fa_num),
                 fmt_pct_ratio(ip_fa_num, n_ip_neg)])
    print_table(["character", "GT samples", "hits", "Recall"], rows)
    
    # ---- combined thresholds (Cartesian product) ----
    print_section("Combined thresholds: porn_T × gore_T + fixed IP (pred∈1-5)", char="-")
    print("  gating rule: pred_porn ≥ porn_T  OR  pred_gore ≥ gore_T  OR  pred_ip ∈ {1-5}")
    print("  image-side GT: pornographic==2 / violence_gore==2 / ip_copyright∈{1-5}")
    print("  prompt-side GT: label_porn≥porn_T / label_gore≥gore_T / label_ip∈{1-5} (aligned with the gating thresholds)")
    headers = ["porn_T", "gore_T", "ImgRecall", "PornRecall", "GoreRecall",
               "IPRecall", "ImgFA", "HitRate", "PromptRecall", "PromptFA"]
    combo_rows = []
    n_any_unsafe = int(gt_any_unsafe.sum())
    for porn_T in [1, 2, 3, 4, 5]:
        for gore_T in [1, 2, 3, 4, 5]:
            flagged = (pred_porn >= porn_T) | (pred_gore >= gore_T) | ip_flagged

            # image-side metrics
            overall_num = int(flagged[gt_any_unsafe].sum())
            porn_num = int(flagged[gt_porn_pos].sum())
            gore_num = int(flagged[gt_gore_pos].sum())
            ip_num = int(flagged[gt_ip_pos].sum())
            fa_num = int(flagged[~gt_any_unsafe].sum())
            n_flagged = int(flagged.sum())
            hit_num = int(gt_any_unsafe[flagged].sum())

            # prompt-side metrics (GT aligned with the gating thresholds)
            prompt_unsafe = (prompt_porn_val >= porn_T) | (prompt_gore_val >= gore_T) | np.isin(prompt_ip_val, [1, 2, 3, 4, 5])
            n_prompt_unsafe = int(prompt_unsafe.sum())
            n_prompt_safe = int((~prompt_unsafe).sum())
            prompt_recall_num = int(flagged[prompt_unsafe].sum())
            prompt_fa_num = int(flagged[~prompt_unsafe].sum())

            combo_rows.append({
                'porn_T': porn_T, 'gore_T': gore_T,
                'OverallRecall': fmt_pct_ratio(overall_num, n_any_unsafe),
                'PornRecall': fmt_pct_ratio(porn_num, n_porn_pos),
                'GoreRecall': fmt_pct_ratio(gore_num, n_gore_pos),
                'IPRecall': fmt_pct_ratio(ip_num, n_ip_pos),
                'FalseAlarm': fmt_pct_ratio(fa_num, n_safe),
                'HitRate': fmt_pct_ratio(hit_num, n_flagged),
                'PromptRecall': fmt_pct_ratio(prompt_recall_num, n_prompt_unsafe),
                'PromptFalseAlarm': fmt_pct_ratio(prompt_fa_num, n_prompt_safe),
            })

    # print
    print_rows = [[f"≥{r['porn_T']}", f"≥{r['gore_T']}",
                   r['OverallRecall'], r['PornRecall'],
                   r['GoreRecall'], r['IPRecall'],
                   r['FalseAlarm'], r['HitRate'],
                   r['PromptRecall'], r['PromptFalseAlarm']] for r in combo_rows]
    print_table(headers, print_rows)
    
    # save CSV
    if save_path:
        df_combo = pd.DataFrame(combo_rows)
        df_combo.to_csv(save_path, index=False)
        print(f"  → gating-evaluation results saved: {save_path}")
    
    print()
    return combo_rows


# ========================== training-curve visualization ==========================

def plot_prompt_training_curves(history, save_path):
    """Plot the training curves: train_loss, val_loss, per-class acc"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    steps = history.get('step', [])
    if not steps:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (0,0) Train Loss
    ax = axes[0, 0]
    ax.plot(steps, history.get('train_loss', []), marker='.', label='train_loss')
    ax.set_title('Train Loss')
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # (0,1) Val Loss
    ax = axes[0, 1]
    ax.plot(steps, history.get('val_loss', []), marker='.', color='orange', label='val_loss')
    ax.set_title('Val Loss')
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # (1,0) Accuracy by category
    ax = axes[1, 0]
    ax.plot(steps, history.get('acc_porn', []), marker='.', label='acc_porn')
    ax.plot(steps, history.get('acc_gore', []), marker='.', label='acc_gore')
    ax.plot(steps, history.get('acc_ip', []), marker='.', label='acc_ip')
    ax.set_title('Val Accuracy (per category)')
    ax.set_xlabel('Step')
    ax.set_ylabel('Accuracy')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # (1,1) Average Accuracy
    ax = axes[1, 1]
    ax.plot(steps, history.get('acc_avg', []), marker='.', color='red', label='acc_avg')
    ax.set_title('Val Average Accuracy')
    ax.set_xlabel('Step')
    ax.set_ylabel('Accuracy')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[Plot] training curves saved: {save_path}")


# ========================== Smoke Test ==========================

def run_smoke_test(model, train_loader, eval_loader, optimizer, device, output_dir,
                   eval_steps):
    """
    Quick pre-training validation that the whole pipeline works:
    run train + eval on a few batches and make sure nothing errors.
    Model/optimizer states are restored afterwards. Evaluation uses the
    validation loader.
    """
    print("=" * 80)
    print("[Smoke Test] quick pre-start pipeline validation (5 batches train + eval + gating)")
    print("=" * 80)

    # save states
    model_state = {k: v.clone() for k, v in model.state_dict().items()}
    optim_state = optimizer.state_dict()

    smoke_dir = os.path.join(output_dir, "_smoke")
    os.makedirs(smoke_dir, exist_ok=True)

    try:
        # train 5 batches
        model.train()
        batch_count = 0
        pbar = tqdm(train_loader, total=5, desc="[Smoke] Train", leave=False)
        for batch in pbar:
            if batch is None:
                continue
            pooled, lp, lg, li = batch
            pooled = pooled.to(device)
            lp, lg, li = lp.to(device), lg.to(device), li.to(device)

            logits_porn, logits_gore, logits_ip = model(pooled)
            loss = (F.cross_entropy(logits_porn, lp) +
                    F.cross_entropy(logits_gore, lg) +
                    F.cross_entropy(logits_ip, li))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_count += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            if batch_count >= 5:
                break
        pbar.close()
        print(f"  [Smoke] ✔ Train {batch_count} batches, loss={loss.item():.4f}")

        # evaluate (only 5 batches)
        metrics = evaluate_prompt_metrics(model, eval_loader, device,
                                          max_batches=5, desc="[Smoke] Eval")
        print(f"  [Smoke] ✔ Eval: "
              f"acc_porn={fmt_pct_ratio(metrics['n_correct_porn'], metrics['n_total'])} "
              f"acc_gore={fmt_pct_ratio(metrics['n_correct_gore'], metrics['n_total'])} "
              f"acc_ip={fmt_pct_ratio(metrics['n_correct_ip'], metrics['n_total'])} "
              f"val_loss={metrics['test_loss']:.4f}")

        # gating (needs full predictions to align with the CSV; skipped here)
        print("  [Smoke] ⊘ Gating eval skipped (partial eval; cannot align with the full CSV)")

    except Exception as e:
        print(f"  [Smoke] ✘ Pipeline failed: {e}")
        raise RuntimeError(f"Smoke test failed; check the data and model config.") from e

    # restore states
    model.load_state_dict(model_state)
    optimizer.load_state_dict(optim_state)
    print("  [Smoke] ✔ model/optimizer states restored")
    print("=" * 80)
    print()


# ========================== train/val split ==========================

def split_prompt_train_val(train_dataset, val_ratio, seed):
    """Hold out a random validation subset from the training set.

    Model selection uses this validation split. A seeded global random holdout
    keeps val proportional to the training label distribution. Mutates
    train_dataset in place to keep the train rows and returns a shallow-copied
    dataset carrying the held-out val rows.
    """
    import copy
    df = train_dataset.samples
    n = len(df)
    n_val = int(round(n * val_ratio))
    n_val = max(0, min(n_val, n - 1))  # always keep at least one training sample
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    val_mask = np.zeros(n, dtype=bool)
    val_mask[perm[:n_val]] = True

    val_df = df[val_mask].reset_index(drop=True)
    train_df = df[~val_mask].reset_index(drop=True)

    val_dataset = copy.copy(train_dataset)  # shallow: shares embeds_dir/mask_dir/has_mask
    val_dataset.samples = val_df
    train_dataset.samples = train_df
    print(f"  [Val Split] random holdout: train={len(train_df)} val={len(val_df)} "
          f"(val_ratio={val_ratio}, seed={seed})")
    return val_dataset


# ========================== ckpt directory naming ==========================

def build_prompt_ckpt_dir(args):
    """Auto-generate the output dir: {ckpt_root}/{today}/{model_name}/prompt-mlp/{datetime_params}"""
    import datetime
    now = datetime.datetime.now()
    today_str = now.strftime("%Y%m%d")
    now_str = now.strftime("%Y%m%d_%H%M%S")

    exp_name = "prompt-mlp-multitask-porn-gore-ip"
    param_str = (f"{now_str}_BS{args.batch_size}_LR{args.lr}"
                 f"_H{args.hidden_dim}_L{args.num_layers}"
                 f"_drop{args.dropout}_seed{args.seed}")

    return os.path.join(args.ckpt_root, today_str, args.model_name, exp_name, param_str)


# ========================== Main ==========================

# ============================================================
# Default path config — the train_csv / test_csv shared by all models, plus the per-model
# gt_csv / train_data_dir / test_data_dir
# To update the data version, change it only here; no need to pass args manually each time
# ============================================================
_VALID_MODELS = ["z-image-turbo", "qwen-image-2512", "internvl-u", "hunyuan-image-2_1", "flux2-klein-base-9b"]

# Paths can be overridden via environment variables (see the README "Configuration" section):
#   INGUARD_OUTPUT_ROOT — output root for RevGen generated data (outputs/revgen)
#   INGUARD_DATA_ROOT   — dataset CSV / OpenImages root dir
#   INGUARD_CKPT_ROOT   — checkpoint output root dir
_DATASET_ROOT = os.environ.get("INGUARD_DATA_ROOT", "/path/to/data")
_REVGEN_ROOT = os.environ.get("INGUARD_OUTPUT_ROOT", "./outputs/revgen")

_DEFAULT_TRAIN_CSV = f"{_DATASET_ROOT}/RevGen/trainset_labeled.csv"
_DEFAULT_TEST_CSV = f"{_DATASET_ROOT}/RevGen/testset_labeled.csv"
_DEFAULT_TRAIN_DATA_DIR = {
    "z-image-turbo":      f"{_REVGEN_ROOT}/z-image-turbo/trainset-seed42-1024-9steps",
    "qwen-image-2512":    f"{_REVGEN_ROOT}/qwen-image-2512/trainset-seed42-1328-10steps",
    "internvl-u":         f"{_REVGEN_ROOT}/internvl-u/trainset-seed42-1024-20steps",
    "hunyuan-image-2_1":  f"{_REVGEN_ROOT}/hunyuan-image-2_1/trainset-seed42-2048-10steps",
    "flux2-klein-base-9b": f"{_REVGEN_ROOT}/flux2-klein-base-9b/trainset-seed42-1024-10steps",
}
_DEFAULT_TEST_DATA_DIR = {
    "z-image-turbo":      f"{_REVGEN_ROOT}/z-image-turbo/testset-seed42-1024-9steps",
    "qwen-image-2512":    f"{_REVGEN_ROOT}/qwen-image-2512/testset-seed42-1328-10steps",
    "internvl-u":         f"{_REVGEN_ROOT}/internvl-u/testset-seed42-1024-20steps",
    "hunyuan-image-2_1":  f"{_REVGEN_ROOT}/hunyuan-image-2_1/testset-seed42-2048-10steps",
    "flux2-klein-base-9b": f"{_REVGEN_ROOT}/flux2-klein-base-9b/testset-seed42-1024-10steps",
}
_DEFAULT_GT_CSV = {
    "z-image-turbo": (
        f"{_REVGEN_ROOT}/z-image-turbo/testset-seed42-1024-9steps/labels_llm/"
        "predictions.csv"
    ),
    "qwen-image-2512": (
        f"{_REVGEN_ROOT}/qwen-image-2512/testset-seed42-1328-10steps/labels_llm/"
        "predictions.csv"
    ),
    "internvl-u": (
        f"{_REVGEN_ROOT}/internvl-u/testset-seed42-1024-20steps/labels_llm/"
        "predictions.csv"
    ),
    "hunyuan-image-2_1": (
        f"{_REVGEN_ROOT}/hunyuan-image-2_1/testset-seed42-2048-10steps/labels_llm/"
        "predictions.csv"
    ),
    "flux2-klein-base-9b": (
        f"{_REVGEN_ROOT}/flux2-klein-base-9b/testset-seed42-1024-10steps/labels_llm/"
        "predictions.csv"
    ),
}


def main():
    parser = argparse.ArgumentParser(description="Prompt Embeds multi-task safety classification training")
    parser.add_argument("--model_name", type=str, required=True,
                        choices=_VALID_MODELS,
                        help=f"image generation model id; valid values: {', '.join(_VALID_MODELS)}")
    parser.add_argument("--train_data_dir", type=str, default=None,
                        help="Dir holding the training prompt_embeds. If unspecified, auto-searched under {data_root}/{model_name}/"
                             "for a subdir containing the 'trainset' keyword")
    parser.add_argument("--test_data_dir", type=str, default=None,
                        help="Dir holding the test prompt_embeds. If unspecified, auto-searched under {data_root}/{model_name}/"
                             "for a subdir containing the 'testset'/'eval' keyword")
    parser.add_argument("--data_root", type=str,
                        default=_REVGEN_ROOT,
                        help="Data root dir (used by auto-inference)")
    parser.add_argument("--train_csv", type=str, default=None,
                        help=f"Training label CSV (default: {_DEFAULT_TRAIN_CSV})")
    parser.add_argument("--test_csv", type=str, default=None,
                        help=f"Test label CSV (default: {_DEFAULT_TEST_CSV})")
    parser.add_argument("--gt_csv", type=str, default=None,
                        help="Image GT label CSV (for gating evaluation). If unspecified, chosen by model_name automatically")
    parser.add_argument("--ckpt_root", type=str,
                        default=os.environ.get("INGUARD_CKPT_ROOT", "./outputs/checkpoints"),
                        help="Checkpoint output root dir")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Manually specify the output dir (overrides the auto-generated one when set)")
    parser.add_argument("--hidden_dim", type=int, default=1024, help="MLP hidden dim")
    parser.add_argument("--num_layers", type=int, default=3, help="MLP residual block count")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size (prompt_embeds are large; default 8)")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--eval_steps", type=int, default=None,
                        help="Evaluate every N steps. Default is 1/8 epoch")
    parser.add_argument("--max_train_samples", type=int, default=None,
                        help="Cap the training-set size (randomly sampled each epoch). Default None = use all")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--device", type=str, default=None, help="Device (auto-selected by default)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--val_ratio", type=float, default=0.1,
                        help="Fraction of the training set held out as a validation split "
                             "used for model selection.")
    parser.add_argument("--eval_test_every_step", action="store_true",
                        help="Also evaluate the test set at every eval step for monitoring. "
                             "OFF by default.")
    args = parser.parse_args()

    # ---------- auto-fill default paths ----------
    if args.train_csv is None:
        args.train_csv = _DEFAULT_TRAIN_CSV
        print(f"[Default] train_csv: {args.train_csv}")
    if args.test_csv is None:
        args.test_csv = _DEFAULT_TEST_CSV
        print(f"[Default] test_csv:  {args.test_csv}")
    if args.gt_csv is None:
        if args.model_name in _DEFAULT_GT_CSV:
            args.gt_csv = _DEFAULT_GT_CSV[args.model_name]
            print(f"[Default] gt_csv:   {args.gt_csv}")
        else:
            raise ValueError(
                f"--gt_csv not specified, and model '{args.model_name}' has no default gt_csv config. "
                f"Specify it via --gt_csv, or add a config entry in _DEFAULT_GT_CSV."
            )

    # random seed
    seed_everything(args.seed)

    # device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------- data_dir: prefer built-in defaults; otherwise auto-infer ----------
    if args.train_data_dir is None:
        if args.model_name in _DEFAULT_TRAIN_DATA_DIR:
            args.train_data_dir = _DEFAULT_TRAIN_DATA_DIR[args.model_name]
            print(f"[Default] train_data_dir: {args.train_data_dir}")
        else:
            raise ValueError(
                f"--train_data_dir not specified, and model '{args.model_name}' has no default config. "
                f"Specify it via --train_data_dir, or add a config entry in _DEFAULT_TRAIN_DATA_DIR."
            )

    if args.test_data_dir is None:
        if args.model_name in _DEFAULT_TEST_DATA_DIR:
            args.test_data_dir = _DEFAULT_TEST_DATA_DIR[args.model_name]
            print(f"[Default] test_data_dir:  {args.test_data_dir}")
        else:
            raise ValueError(
                f"--test_data_dir not specified, and model '{args.model_name}' has no default config. "
                f"Specify it via --test_data_dir, or add a config entry in _DEFAULT_TEST_DATA_DIR."
            )
    print()

    # output dir: auto-generated or manually specified
    output_dir = args.output_dir or build_prompt_ckpt_dir(args)
    os.makedirs(output_dir, exist_ok=True)
    # subdirectories
    ckpts_dir = os.path.join(output_dir, "ckpts")
    plots_dir = os.path.join(output_dir, "plots")
    meta_dir = os.path.join(output_dir, "meta")
    badcase_dir = os.path.join(output_dir, "badcase")
    for d in [ckpts_dir, plots_dir, meta_dir, badcase_dir]:
        os.makedirs(d, exist_ok=True)

    # Logger
    sys.stdout = Logger(os.path.join(output_dir, "train_prompt.log"))

    # print the config
    print_kv_table([
        ("model_name", args.model_name),
        ("train_data_dir", args.train_data_dir),
        ("test_data_dir", args.test_data_dir),
        ("data_root", args.data_root),
        ("train_csv", args.train_csv),
        ("test_csv", args.test_csv),
        ("gt_csv", args.gt_csv),
        ("ckpt_root", args.ckpt_root),
        ("output_dir", output_dir),
        ("hidden_dim", str(args.hidden_dim)),
        ("num_layers", str(args.num_layers)),
        ("dropout", str(args.dropout)),
        ("batch_size", str(args.batch_size)),
        ("lr", str(args.lr)),
        ("weight_decay", str(args.weight_decay)),
        ("epochs", str(args.epochs)),
        ("device", str(device)),
        ("seed", str(args.seed)),
    ], title="[Config] training config")

    # ---- data probe (infer the format from the training dir) ----
    input_dim, has_mask = probe_data(args.train_data_dir, args.train_csv)

    # ---- build the Datasets ----
    print_section("[Dataset] loading datasets", char="-")
    print("  training set:")
    train_dataset = PromptEmbedsDataset(args.train_csv, args.train_data_dir, has_mask=has_mask)
    print("  test set:")
    test_dataset = PromptEmbedsDataset(args.test_csv, args.test_data_dir, has_mask=has_mask)
    print()

    # ---- hold out a validation split from the training set ----
    val_dataset = split_prompt_train_val(train_dataset, args.val_ratio, args.seed)
    print()

    # ---- label distribution stats ----
    def print_label_distribution(df, title, label_cols=None):
        """Print the value distribution of the given columns (NaN skipped automatically)"""
        if label_cols is None:
            label_cols = ['label_porn_risk_level', 'label_gore_risk_level', 'label_ip_risk_level']
        print(f"  {title}:")
        for col in label_cols:
            if col not in df.columns:
                continue
            series = pd.to_numeric(df[col], errors='coerce')
            n_nan = series.isna().sum()
            counts = series.dropna().astype(int).value_counts().sort_index()
            total = counts.sum()
            dist_str = "  ".join([f"{k}:{v}({v/total*100:.1f}%)" for k, v in counts.items()])
            col_short = col.replace('label_', '').replace('_risk_level', '')
            nan_info = f"  [NaN={n_nan}]" if n_nan > 0 else ""
            print(f"    {col_short}: {dist_str}{nan_info}")
        print()

    print_section("[Label Distribution] label distribution stats", char="-")

    # training-set prompt label distribution
    train_df = train_dataset.samples
    print_label_distribution(train_df, "training-set prompt labels")

    # test-set prompt label distribution
    test_df = test_dataset.samples
    print_label_distribution(test_df, "test-set prompt labels")

    # test-set GT label distribution
    gt_df = pd.read_csv(args.gt_csv, dtype=str)
    gt_label_cols = [c for c in ['pornographic', 'violence_gore', 'ip_copyright'] if c in gt_df.columns]
    print_label_distribution(gt_df, "test-set GT labels (image/video)", label_cols=gt_label_cols)

    # GT binarized stats (positive counts; NaN skipped)
    gt_df_int = gt_df.copy()
    for c in gt_label_cols:
        gt_df_int[c] = pd.to_numeric(gt_df_int[c], errors='coerce')
    n_gt = len(gt_df_int)
    porn_pos = (gt_df_int['pornographic'] == 2).sum() if 'pornographic' in gt_df_int.columns else 0
    gore_pos = (gt_df_int['violence_gore'] == 2).sum() if 'violence_gore' in gt_df_int.columns else 0
    ip_pos = gt_df_int['ip_copyright'].isin([1,2,3,4,5]).sum() if 'ip_copyright' in gt_df_int.columns else 0
    print(f"  GT positive-sample stats (binarized):")
    print(f"    porn (==2): {porn_pos}/{n_gt} ({porn_pos/n_gt*100:.2f}%)")
    print(f"    gore (==2): {gore_pos}/{n_gt} ({gore_pos/n_gt*100:.2f}%)")
    print(f"    ip (∈{{1-5}}): {ip_pos}/{n_gt} ({ip_pos/n_gt*100:.2f}%)")
    print()

    # ---- DataLoader ----
    if args.max_train_samples and args.max_train_samples < len(train_dataset):
        train_sampler = RandomSampler(train_dataset, replacement=False,
                                      num_samples=args.max_train_samples)
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, sampler=train_sampler,
            num_workers=args.num_workers, pin_memory=True, drop_last=False,
            collate_fn=collate_fn
        )
        print(f"[Train] randomly sampling {args.max_train_samples} items per epoch "
              f"(from {len(train_dataset)} total)")
    else:
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=True, drop_last=False,
            collate_fn=collate_fn
        )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, drop_last=False,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, drop_last=False,
        collate_fn=collate_fn
    )

    # eval_steps: default is 1/8 epoch
    eval_steps = args.eval_steps or max(1, len(train_loader) // 8)
    print(f"[Train] eval_steps={eval_steps} (total steps/epoch={len(train_loader)})")
    print()

    # ---- build the model ----
    model = PromptMultiTaskMLP(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[Model] PromptMultiTaskMLP: input_dim={input_dim}, hidden_dim={args.hidden_dim}, "
          f"num_layers={args.num_layers}, params={n_params:,}")
    print()

    # ---- optimizer + scheduler ----
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ---- code snapshot ----
    snapshot_code_dir(output_dir)

    # ---- Smoke Test ----
    run_smoke_test(
        model=model, train_loader=train_loader, eval_loader=val_loader,
        optimizer=optimizer, device=device, output_dir=output_dir,
        eval_steps=eval_steps,
    )

    # ---- training loop ----
    print_section("[Training] start training")
    best_acc_avg = 0.0
    best_step = -1
    best_ckpt_path = None
    global_step = 0
    eval_count = 0
    history = {
        'step': [], 'train_loss': [], 'val_loss': [],
        'acc_porn': [], 'acc_gore': [], 'acc_ip': [], 'acc_avg': [],
    }

    def run_eval_and_save(step_label):
        """Evaluate on validation + save ckpt + plot.

        Pass --eval_test_every_step to additionally run test metrics for
        monitoring.
        """
        nonlocal best_acc_avg, best_step, eval_count, best_ckpt_path
        eval_count += 1

        metrics = evaluate_prompt_metrics(model, val_loader, device, desc="Evaluating (val)")
        acc_avg = (metrics['acc_porn'] + metrics['acc_gore'] + metrics['acc_ip']) / 3.0

        history['step'].append(global_step)
        history['train_loss'].append(running_loss / max(running_n, 1))
        history['val_loss'].append(float(metrics['test_loss']))
        history['acc_porn'].append(float(metrics['acc_porn']))
        history['acc_gore'].append(float(metrics['acc_gore']))
        history['acc_ip'].append(float(metrics['acc_ip']))
        history['acc_avg'].append(float(acc_avg))

        n_total_eval = metrics['n_total']
        print(f"  [{step_label}] step={global_step} | "
              f"train_loss={running_loss/max(running_n,1):.4f} val_loss={metrics['test_loss']:.4f} | "
              f"val_acc_porn={fmt_pct_ratio(metrics['n_correct_porn'], n_total_eval)} "
              f"val_acc_gore={fmt_pct_ratio(metrics['n_correct_gore'], n_total_eval)} "
              f"val_acc_ip={fmt_pct_ratio(metrics['n_correct_ip'], n_total_eval)} | "
              f"val_avg={acc_avg*100:.2f}%")

        # optional test monitoring; OFF by default
        if args.eval_test_every_step:
            t_metrics = evaluate_prompt_metrics(model, test_loader, device, desc="Monitoring (test)")
            t_acc_avg = (t_metrics['acc_porn'] + t_metrics['acc_gore'] + t_metrics['acc_ip']) / 3.0
            print(f"    [monitor] test_loss={t_metrics['test_loss']:.4f} test_avg={t_acc_avg*100:.2f}%")

        # save the ckpt (records the val metrics that drive selection)
        ckpt_payload = {
            'global_step': global_step,
            'eval_count': eval_count,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'acc_avg': acc_avg,
            'acc_porn': float(metrics['acc_porn']),
            'acc_gore': float(metrics['acc_gore']),
            'acc_ip': float(metrics['acc_ip']),
            'val_loss': float(metrics['test_loss']),
            'val_acc_avg': acc_avg,
            'input_dim': input_dim,
            'hidden_dim': args.hidden_dim,
            'num_layers': args.num_layers,
            'dropout': args.dropout,
            'model_name': args.model_name,
        }
        # ckpt_step{N}.pth: saved at every eval step; downstream selection picks the step
        # by highest val acc_avg (train_image.py PE2 and scripts/export_weights.py),
        # trying best_prompt_model_step{N}.pth first and falling back to ckpt_step{N}.pth
        ckpt_path = os.path.join(ckpts_dir, f"ckpt_step{global_step:06d}.pth")
        torch.save(ckpt_payload, ckpt_path)

        if acc_avg > best_acc_avg:
            best_acc_avg = acc_avg
            best_step = global_step
            # best_prompt_model_step{N}.pth: saved only when val acc_avg hits a new high
            # OSS does not support overwriting: always saved as a separate step-suffixed file
            best_ckpt_path = os.path.join(ckpts_dir, f"best_prompt_model_step{global_step:06d}.pth")
            torch.save(ckpt_payload, best_ckpt_path)
            print(f"    ★ New best (val)! val_avg_acc={acc_avg:.4f} → {best_ckpt_path}")

        # plot the training curves (a separate step-suffixed file per eval, since OSS overwriting is unsupported)
        plot_prompt_training_curves(
            history, os.path.join(plots_dir, f"training_curves_step{global_step:06d}.png")
        )

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_n = 0
        correct_porn, correct_gore, correct_ip = 0, 0, 0
        steps_in_epoch = 0

        pbar = tqdm(train_loader, desc=f"Epoch [{epoch}/{args.epochs}]", leave=False)
        for batch in pbar:
            if batch is None:
                continue
            pooled, lp, lg, li = batch
            pooled = pooled.to(device)
            lp = lp.to(device)
            lg = lg.to(device)
            li = li.to(device)

            logits_porn, logits_gore, logits_ip = model(pooled)

            loss_porn = F.cross_entropy(logits_porn, lp)
            loss_gore = F.cross_entropy(logits_gore, lg)
            loss_ip = F.cross_entropy(logits_ip, li)
            loss = loss_porn + loss_gore + loss_ip

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            bs = pooled.size(0)
            running_loss += loss.item() * bs
            correct_porn += (logits_porn.argmax(1) == lp).sum().item()
            correct_gore += (logits_gore.argmax(1) == lg).sum().item()
            correct_ip += (logits_ip.argmax(1) == li).sum().item()
            running_n += bs
            global_step += 1
            steps_in_epoch += 1

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                porn_acc=f"{correct_porn/running_n:.4f}",
                gore_acc=f"{correct_gore/running_n:.4f}",
                ip_acc=f"{correct_ip/running_n:.4f}"
            )

            # intra-epoch evaluation
            if global_step % eval_steps == 0:
                run_eval_and_save(f"Epoch {epoch} step {steps_in_epoch}")
                model.train()  # back to training mode

        # Epoch end: if the last eval was not exactly at the epoch boundary, eval once more
        if global_step % eval_steps != 0:
            run_eval_and_save(f"Epoch {epoch} end")
            model.train()

        scheduler.step()
        avg_loss = running_loss / max(running_n, 1)
        print(f"Epoch {epoch:02d}/{args.epochs} done | loss={avg_loss:.4f} | "
              f"lr={scheduler.get_last_lr()[0]:.6f}")

    print(f"\n[Training Done] Best step={best_step} (selected by val acc_avg={best_acc_avg:.4f})")
    print(f"  Checkpoint: {best_ckpt_path}")

    # save history
    save_json(history, os.path.join(meta_dir, "history.json"))

    # ---- load the best model for the final evaluation ----
    if best_ckpt_path is None or not os.path.exists(best_ckpt_path):
        print("[Final Eval] ⚠ no best ckpt found, skipping the final evaluation")
        return
    ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    print_section(f"[Final Eval] loading the best (val-selected) model "
                  f"(step {ckpt.get('global_step', 'N/A')}) for a one-shot TEST evaluation")

    # prompt-label evaluation
    metrics = evaluate_prompt_metrics(model, test_loader, device)
    n_total_eval = metrics['n_total']
    print(f"  Prompt Metrics: "
          f"acc_porn={fmt_pct_ratio(metrics['n_correct_porn'], n_total_eval)} "
          f"acc_gore={fmt_pct_ratio(metrics['n_correct_gore'], n_total_eval)} "
          f"acc_ip={fmt_pct_ratio(metrics['n_correct_ip'], n_total_eval)}")

    # gating evaluation
    eval_gating_metrics(
        pred_porn=metrics['pred_porn'],
        pred_gore=metrics['pred_gore'],
        pred_ip=metrics['pred_ip'],
        test_csv=args.test_csv,
        gt_csv=args.gt_csv,
        save_path=os.path.join(meta_dir, "gating_final_best.csv"),
    )

    # save the final predictions (only the actually-evaluated samples; rows with missing labels dropped)
    # includes a prompt column, read directly by downstream SAGE scripts
    df_test = pd.read_csv(args.test_csv, dtype=str)
    df_test = df_test.dropna(subset=['label_porn_risk_level', 'label_gore_risk_level', 'label_ip_risk_level'])
    keep_cols = ['id', 'prompt'] if 'prompt' in df_test.columns else ['id']
    df_test = df_test[keep_cols].reset_index(drop=True)
    if len(df_test) != len(metrics['pred_porn']):
        print(f"  [WARN] valid test_csv rows({len(df_test)}) != predictions({len(metrics['pred_porn'])}); truncating to align")
        n_pred = len(metrics['pred_porn'])
        df_test = df_test.iloc[:n_pred]
    df_test['pred_porn'] = metrics['pred_porn']
    df_test['pred_gore'] = metrics['pred_gore']
    df_test['pred_ip'] = metrics['pred_ip']
    pred_csv_path = os.path.join(meta_dir, "test_predictions.csv")
    df_test.to_csv(pred_csv_path, index=False)
    print(f"\n[Saved] predictions: {pred_csv_path}")

    # save badcases (per classification head)
    save_badcases(metrics, args.test_csv, badcase_dir, step_label="best")

    # save gating badcases (missed + false alarm)
    save_gating_badcases(
        pred_porn=metrics['pred_porn'],
        pred_gore=metrics['pred_gore'],
        pred_ip=metrics['pred_ip'],
        test_csv=args.test_csv, gt_csv=args.gt_csv,
        save_dir=badcase_dir,
        step_label="best",
    )


if __name__ == "__main__":
    main()
