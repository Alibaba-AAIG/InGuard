"""
train_utils.py — shared training utilities

Shared utility functions for the multi-task safety-classification training
flow, including:
  - Losses (masked_cross_entropy_loss, build_task_masks_from_train_groups)
  - Training loop (train_one_epoch)
  - Evaluation (evaluate)
  - History bookkeeping (init_history, append_metrics, append_epoch_history)
  - LR scheduling (build_lr_scheduler)
  - Smoke Test (run_smoke_test)

These functions are model/data-agnostic and are imported by the training entry scripts.
"""

import os
import sys
import csv
import copy
import random
import datetime
import itertools
import warnings
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

warnings.filterwarnings("ignore")

from utils_common import (
    Logger,
    collate_fn,
    seed_everything,
    save_json,
    save_history_and_plots,
    print_table,
    print_kv_table,
    compute_binary_metrics,
    compute_multiclass_metrics,
    compute_confusion_matrix,
    print_confusion_matrix,
    empty_metrics,
    empty_multiclass_metrics,
    fmt_percent_ratio,
    summarize_task_metrics,
    compute_metrics_by_thresholds,
    compute_ip_metrics_by_shared_thresholds,
    print_threshold_table_compact,
    print_ip_multiclass_metrics,
    pick_best_threshold_row,
    compute_metrics_at_threshold,
    compute_total_risk_stats,
    print_total_risk_stats_table,
    predict_ip_with_shared_threshold,
    safe_copy,
    snapshot_code_dir,
)
from data_pipeline import (
    IP_ID2NAME,
)
from models_zoo import build_optimizer


# =========================
# Loss / Mask
# =========================
def build_task_masks_from_train_groups(train_groups_batch, device, enable_train_group_head_mask=False):
    """Per-head sample-level masks from train_group; when disabled every head sees all samples."""
    bs = len(train_groups_batch)
    if not enable_train_group_head_mask:
        ones = torch.ones(bs, dtype=torch.float32, device=device)
        return ones, ones, ones

    porn_only = {"porn", "porn_borderline"}
    gore_only = {"gore", "gore_borderline"}
    ip_only = {"ip_controlled", "ip_borderline"}
    all_heads = {"normal_other"}

    p_mask, g_mask, i_mask = [], [], []
    for g in train_groups_batch:
        if g in porn_only:
            p_mask.append(1.0); g_mask.append(0.0); i_mask.append(0.0)
        elif g in gore_only:
            p_mask.append(0.0); g_mask.append(1.0); i_mask.append(0.0)
        elif g in ip_only:
            p_mask.append(0.0); g_mask.append(0.0); i_mask.append(1.0)
        elif g in all_heads:
            p_mask.append(1.0); g_mask.append(1.0); i_mask.append(1.0)
        else:
            raise ValueError(f"Unknown train_group for head mask: {g}")

    return (torch.tensor(p_mask, dtype=torch.float32, device=device),
            torch.tensor(g_mask, dtype=torch.float32, device=device),
            torch.tensor(i_mask, dtype=torch.float32, device=device))


def masked_cross_entropy_loss(logits, targets, sample_mask):
    per_sample = F.cross_entropy(logits, targets, reduction="none")
    mask = sample_mask.float()
    valid = mask.sum()
    if valid.item() <= 0:
        return logits.new_tensor(0.0)
    return (per_sample * mask).sum() / valid


# =========================
# History
# =========================
def init_history():
    keys = ["train_iter", "train_iter_loss", "epoch", "train_loss", "val_loss", "test_loss"]
    for split in ["train", "val", "test"]:
        for task in ["porn", "gore", "ip_pos"]:
            for stat in ["acc", "precision", "recall", "f1"]:
                keys.append(f"{split}_{task}_{stat}")
    for split in ["val", "test"]:
        for stat in ["total_risk_recall", "e2e_unsafety", "normal_disturb_rate"]:
            keys.append(f"{split}_{stat}")
    return {k: [] for k in keys}


def append_metrics(history, prefix, metrics):
    for stat in ["acc", "precision", "recall", "f1"]:
        history[f"{prefix}_{stat}"].append(float(metrics[stat]))


def append_epoch_history(history, epoch, train_loss, val_loss, test_loss,
                         train_porn, val_porn, test_porn,
                         train_gore, val_gore, test_gore,
                         train_ip_pos, val_ip_pos, test_ip_pos,
                         val_total_risk_stats=None, test_total_risk_stats=None):
    history["epoch"].append(epoch)
    history["train_loss"].append(float(train_loss))
    history["val_loss"].append(float(val_loss))
    history["test_loss"].append(float(test_loss) if test_loss is not None else float("nan"))

    # test_* may be None when the test set is not evaluated every epoch
    # (eval_test_every_epoch=False); append NaN so the parallel history lists
    # stay length-aligned for plotting.
    _nan_metrics = {"acc": float("nan"), "precision": float("nan"),
                    "recall": float("nan"), "f1": float("nan")}
    for prefix, metrics in {
        "train_porn": train_porn, "val_porn": val_porn, "test_porn": test_porn,
        "train_gore": train_gore, "val_gore": val_gore, "test_gore": test_gore,
        "train_ip_pos": train_ip_pos, "val_ip_pos": val_ip_pos, "test_ip_pos": test_ip_pos,
    }.items():
        append_metrics(history, prefix, metrics if metrics is not None else _nan_metrics)

    for split, stats in [("val", val_total_risk_stats), ("test", test_total_risk_stats)]:
        for k in ["total_risk_recall", "e2e_unsafety", "normal_disturb_rate"]:
            history[f"{split}_{k}"].append(float(stats[k]) if stats else 0.0)


# =========================
# Train
# =========================
def _running_f1(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def train_one_epoch(model, train_loader, optimizer, criterion_bin, criterion_ip, device,
                    epoch, history, global_iter,
                    lambda_porn=1.0, lambda_gore=1.0, lambda_ip=1.0,
                    enable_train_group_head_mask=False, grad_clip_max_norm=0.0,
                    num_porn_classes=2, num_gore_classes=2):
    _porn_3class = (num_porn_classes == 3)
    _gore_3class = (num_gore_classes == 3)
    model.train()
    total_loss = 0.0
    total_count = 0
    empty_batches = 0
    porn_loss_n = gore_loss_n = ip_loss_n = 0

    porn_true, porn_pred = [], []
    gore_true, gore_pred = [], []
    ip_true_epoch, ip_pred_epoch = [], []

    sampled_three_classes, sampled_groups, sampled_ip_groups, sampled_train_groups = [], [], [], []
    train_group_task_counter = defaultdict(lambda: {"samples": 0, "porn": 0, "gore": 0, "ip": 0})

    # incremental counters so the tqdm F1 updates in O(1)
    running_tp = {"porn": 0, "gore": 0, "ip": 0}
    running_fp = {"porn": 0, "gore": 0, "ip": 0}
    running_fn = {"porn": 0, "gore": 0, "ip": 0}
        # in 3-class mode use running accuracy instead of running F1
    running_porn_correct = 0
    running_porn_total = 0
    running_gore_correct = 0
    running_gore_total = 0
            # in 3cls mode IP also uses running accuracy instead of running F1
    running_ip_correct = 0
    running_ip_total = 0

    pbar = tqdm(train_loader, desc=f"Train Epoch {epoch}")
    first_batch_printed = False

    for batch in pbar:
        if batch is None:
            empty_batches += 1
            continue

        inputs = batch["input"].to(device, non_blocking=True)
        porn_labels = batch["porn_label"].to(device, non_blocking=True)
        gore_labels = batch["gore_label"].to(device, non_blocking=True)
        ip_labels = batch["ip_label"].to(device, non_blocking=True)
        meta = batch["meta"]

        train_groups_batch = list(meta["train_group"])
        sampling_groups_batch = list(meta["sampling_group"])
        ip_groups_batch = list(meta["ip_group"])
        three_classes = list(meta["three_class"])

        if not first_batch_printed:
            _pe_status = "ON" if (batch.get("prompt_embeds") is not None) else "OFF"
            _model_cls = type(model).__name__
            if hasattr(model, "module"):
                _model_cls = type(model.module).__name__
            _mode_hint = f", model_class={_model_cls}"
            print(f"[Train Debug] input.shape={tuple(inputs.shape)}, "
                  f"prompt_embeds={_pe_status}{_mode_hint}")
            first_batch_printed = True

        prompt_embeds = batch.get("prompt_embeds")
        if prompt_embeds is not None:
            prompt_embeds = prompt_embeds.to(device, non_blocking=True)
        prompt_mask = batch.get("prompt_mask")
        if prompt_mask is not None:
            prompt_mask = prompt_mask.to(device, non_blocking=True)

        optimizer.zero_grad()
        porn_logits, gore_logits, ip_logits = model(inputs, prompt_embeds=prompt_embeds, prompt_mask=prompt_mask)

        porn_mask, gore_mask, ip_mask = build_task_masks_from_train_groups(
            train_groups_batch, device, enable_train_group_head_mask,
        )
        porn_loss_n += int(porn_mask.sum().item())
        gore_loss_n += int(gore_mask.sum().item())
        ip_loss_n += int(ip_mask.sum().item())

        for i, g in enumerate(train_groups_batch):
            d = train_group_task_counter[g]
            d["samples"] += 1
            d["porn"] += int(porn_mask[i].item())
            d["gore"] += int(gore_mask[i].item())
            d["ip"] += int(ip_mask[i].item())

        porn_loss = masked_cross_entropy_loss(porn_logits, porn_labels, porn_mask)
        gore_loss = masked_cross_entropy_loss(gore_logits, gore_labels, gore_mask)
        ip_loss = masked_cross_entropy_loss(ip_logits, ip_labels, ip_mask)
        loss = lambda_porn * porn_loss + lambda_gore * gore_loss + lambda_ip * ip_loss

        loss.backward()
        if grad_clip_max_norm and grad_clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip_max_norm))
        optimizer.step()

        bs = inputs.size(0)
        total_loss += float(loss.item()) * bs
        total_count += bs

        porn_preds = torch.argmax(porn_logits, dim=1)
        gore_preds = torch.argmax(gore_logits, dim=1)
        ip_preds = torch.argmax(ip_logits, dim=1)

        porn_labels_np = porn_labels.detach().cpu().numpy()
        porn_preds_np = porn_preds.detach().cpu().numpy()
        gore_labels_np = gore_labels.detach().cpu().numpy()
        gore_preds_np = gore_preds.detach().cpu().numpy()
        ip_labels_np = ip_labels.detach().cpu().numpy()
        ip_preds_np = ip_preds.detach().cpu().numpy()

        porn_true.extend(porn_labels_np.tolist()); porn_pred.extend(porn_preds_np.tolist())
        gore_true.extend(gore_labels_np.tolist()); gore_pred.extend(gore_preds_np.tolist())
        ip_true_epoch.extend(ip_labels_np.tolist()); ip_pred_epoch.extend(ip_preds_np.tolist())

        if _porn_3class:
            running_porn_correct += int((porn_labels_np == porn_preds_np).sum())
            running_porn_total += len(porn_labels_np)
        else:
            running_tp["porn"] += int(((porn_labels_np == 1) & (porn_preds_np == 1)).sum())
            running_fp["porn"] += int(((porn_labels_np == 0) & (porn_preds_np == 1)).sum())
            running_fn["porn"] += int(((porn_labels_np == 1) & (porn_preds_np == 0)).sum())

        if _gore_3class:
            running_gore_correct += int((gore_labels_np == gore_preds_np).sum())
            running_gore_total += len(gore_labels_np)
        else:
            running_tp["gore"] += int(((gore_labels_np == 1) & (gore_preds_np == 1)).sum())
            running_fp["gore"] += int(((gore_labels_np == 0) & (gore_preds_np == 1)).sum())
            running_fn["gore"] += int(((gore_labels_np == 1) & (gore_preds_np == 0)).sum())

        if _porn_3class or _gore_3class:
            running_ip_correct += int((ip_labels_np == ip_preds_np).sum())
            running_ip_total += len(ip_labels_np)
        else:
            ip_true_bin_np = np.isin(ip_labels_np, [0, 1, 2, 3, 4]).astype(int)
            ip_pred_bin_np = np.isin(ip_preds_np, [0, 1, 2, 3, 4]).astype(int)
            running_tp["ip"] += int(((ip_true_bin_np == 1) & (ip_pred_bin_np == 1)).sum())
            running_fp["ip"] += int(((ip_true_bin_np == 0) & (ip_pred_bin_np == 1)).sum())
            running_fn["ip"] += int(((ip_true_bin_np == 1) & (ip_pred_bin_np == 0)).sum())

        sampled_three_classes.extend(three_classes)
        sampled_groups.extend(sampling_groups_batch)
        sampled_ip_groups.extend(ip_groups_batch)
        sampled_train_groups.extend(train_groups_batch)

        global_iter += 1
        history["train_iter"].append(global_iter)
        history["train_iter_loss"].append(float(loss.item()))

        _porn_disp = (f"{running_porn_correct / max(1, running_porn_total):.4f}" if _porn_3class
                      else f"{_running_f1(running_tp['porn'], running_fp['porn'], running_fn['porn']):.4f}")
        _gore_disp = (f"{running_gore_correct / max(1, running_gore_total):.4f}" if _gore_3class
                      else f"{_running_f1(running_tp['gore'], running_fp['gore'], running_fn['gore']):.4f}")
        _3cls = _porn_3class or _gore_3class
        _ip_disp = (f"{running_ip_correct / max(1, running_ip_total):.4f}" if _3cls
                    else f"{_running_f1(running_tp['ip'], running_fp['ip'], running_fn['ip']):.4f}")
        pbar.set_postfix({
            "loss": f"{total_loss / max(1, total_count):.4f}",
            "porn_acc" if _porn_3class else "porn_f1": _porn_disp,
            "gore_acc" if _gore_3class else "gore_f1": _gore_disp,
            "ip_acc" if _3cls else "ip_f1": _ip_disp,
        })

    if _porn_3class:
        train_porn = (compute_multiclass_metrics(porn_true, porn_pred, 3,
                                                   class_names=["safe", "borderline", "risk"])
                     if total_count > 0 else empty_multiclass_metrics(3))
    else:
        train_porn = compute_binary_metrics(porn_true, porn_pred) if total_count > 0 else empty_metrics()
    if _gore_3class:
        train_gore = (compute_multiclass_metrics(gore_true, gore_pred, 3,
                                                   class_names=["safe", "borderline", "risk"])
                     if total_count > 0 else empty_multiclass_metrics(3))
    else:
        train_gore = compute_binary_metrics(gore_true, gore_pred) if total_count > 0 else empty_metrics()
    if len(ip_true_epoch) > 0:
        if _porn_3class or _gore_3class:
            train_ip_pos = compute_multiclass_metrics(ip_true_epoch, ip_pred_epoch, 6,
                                                       class_names=[IP_ID2NAME[i] for i in range(6)])
        else:
            ip_true_bin = [1 if x in (0, 1, 2, 3, 4) else 0 for x in ip_true_epoch]
            ip_pred_bin = [1 if x in (0, 1, 2, 3, 4) else 0 for x in ip_pred_epoch]
            train_ip_pos = compute_binary_metrics(ip_true_bin, ip_pred_bin)
    else:
        train_ip_pos = empty_multiclass_metrics(6) if (_porn_3class or _gore_3class) else empty_metrics()

    epoch_porn_n = sum(1 for x in sampled_three_classes if x == "porn")
    epoch_gore_n = sum(1 for x in sampled_three_classes if x == "gore")
    epoch_normal_n = sum(1 for x in sampled_three_classes if x == "normal")
    avg_loss = total_loss / max(1, total_count)

    cur_lrs = [f"{pg['lr']:.2e}" for pg in optimizer.param_groups]
    print_kv_table(
        [("epoch", epoch), ("valid_samples", total_count), ("empty_batches", empty_batches),
         ("sampled_porn", epoch_porn_n), ("sampled_gore", epoch_gore_n),
         ("sampled_normal", epoch_normal_n), ("ip_samples", len(ip_true_epoch)),
         ("porn_loss_samples", porn_loss_n), ("gore_loss_samples", gore_loss_n),
         ("ip_loss_samples", ip_loss_n),
         ("enable_train_group_head_mask", enable_train_group_head_mask),
         ("grad_clip_max_norm", grad_clip_max_norm),
         ("lr_per_param_group", " / ".join(cur_lrs))],
        title=f"Train Epoch {epoch} Summary / Basic",
    )

    grp = Counter(sampled_groups)
    ipgrp = Counter(sampled_ip_groups)
    tgrp = Counter(sampled_train_groups)
    print_table(["sampling_group", "count"], [[k, grp[k]] for k in sorted(grp.keys())],
                title=f"Train Epoch {epoch} Summary / Raw Sampling Groups")
    print_table(["ip_group", "count"], [[k, ipgrp[k]] for k in sorted(ipgrp.keys())],
                title=f"Train Epoch {epoch} Summary / IP Groups")
    print_table(["train_group", "count"], [[k, tgrp[k]] for k in sorted(tgrp.keys())],
                title=f"Train Epoch {epoch} Summary / Train Groups")

    usage_rows = [[g, d["samples"], d["porn"], d["gore"], d["ip"]]
                  for g, d in sorted(train_group_task_counter.items())]
    print_table(["train_group", "samples", "porn_loss_used", "gore_loss_used", "ip_loss_used"],
                usage_rows, title=f"Train Epoch {epoch} Summary / Train Group Task Usage")

    metric_rows = [[d["Task"], d["Acc"], d["Precision"], d["Recall"], d["F1"]] for d in [
        summarize_task_metrics("Porn", train_porn),
        summarize_task_metrics("Gore", train_gore),
        summarize_task_metrics("IP(6-class macro-F1)" if (_porn_3class or _gore_3class) else "IP(1-5 vs Other)", train_ip_pos),
    ]]
    print_table(["Task", "Acc", "Precision", "Recall", "F1"], metric_rows,
                title=f"Train Epoch {epoch} Summary / Metrics", width_limit=28)

    # in 3-class mode also print per-class F1
    if _porn_3class and train_porn.get("multiclass"):
        _pc = train_porn["per_class"]
        _pc_rows = [[_pc[c].get("name", c), _pc[c]["support"],
                     f"{_pc[c]['precision'] * 100:.2f}%", f"{_pc[c]['recall'] * 100:.2f}%",
                     f"{_pc[c]['f1'] * 100:.2f}%"] for c in range(3)]
        print_table(["Porn Class", "Support", "Precision", "Recall", "F1"], _pc_rows,
                    title=f"Train Epoch {epoch} Summary / Porn Per-Class (macro-F1={train_porn['f1']:.4f})", width_limit=28)
    if _gore_3class and train_gore.get("multiclass"):
        _pc = train_gore["per_class"]
        _pc_rows = [[_pc[c].get("name", c), _pc[c]["support"],
                     f"{_pc[c]['precision'] * 100:.2f}%", f"{_pc[c]['recall'] * 100:.2f}%",
                     f"{_pc[c]['f1'] * 100:.2f}%"] for c in range(3)]
        print_table(["Gore Class", "Support", "Precision", "Recall", "F1"], _pc_rows,
                    title=f"Train Epoch {epoch} Summary / Gore Per-Class (macro-F1={train_gore['f1']:.4f})", width_limit=28)
    if (_porn_3class or _gore_3class) and train_ip_pos.get("multiclass"):
        _pc = train_ip_pos["per_class"]
        _pc_rows = [[_pc[c].get("name", c), _pc[c]["support"],
                     f"{_pc[c]['precision'] * 100:.2f}%", f"{_pc[c]['recall'] * 100:.2f}%",
                     f"{_pc[c]['f1'] * 100:.2f}%"] for c in range(6)]
        print_table(["IP Class", "Support", "Precision", "Recall", "F1"], _pc_rows,
                    title=f"Train Epoch {epoch} Summary / IP Per-Class (macro-F1={train_ip_pos['f1']:.4f})", width_limit=28)

    return avg_loss, train_porn, train_gore, train_ip_pos, global_iter



# =========================
# Eval
# =========================
@torch.no_grad()
def evaluate(model, loader, criterion_bin, criterion_ip, device, epoch, split="Val",
             lambda_porn=1.0, lambda_gore=1.0, lambda_ip=1.0, verbose=True,
             scan_threshold=False, num_porn_classes=2, num_gore_classes=2):
    """Evaluation entry.
    scan_threshold=False (default): use argmax predictions, no threshold sweep.
    scan_threshold=True: sweep thresholds and pick the best F1.
    3-class mode (num_porn_classes==3 or num_gore_classes==3) disables scan_threshold automatically and uses argmax + macro-F1.
    """
    _porn_3class = (num_porn_classes == 3)
    _gore_3class = (num_gore_classes == 3)
    if _porn_3class or _gore_3class:
        scan_threshold = False  # 3-class uses argmax + macro-F1, no threshold sweep
    _porn_prob_idx = 2 if _porn_3class else 1
    _gore_prob_idx = 2 if _gore_3class else 1
    model.eval()
    total_loss = 0.0
    total_count = 0
    empty_batches = 0

    porn_true, porn_pred_argmax, porn_prob = [], [], []
    gore_true, gore_pred_argmax, gore_prob = [], [], []
    ip_true_all, ip_pred_all_argmax = [], []
    ip_probs_all = []
    ip_loss_sum = 0.0
    ip_count = 0
    meta_store = defaultdict(list)

    for batch in tqdm(loader, desc=f"{split} Epoch {epoch}"):
        if batch is None:
            empty_batches += 1
            continue

        inputs = batch["input"].to(device, non_blocking=True)
        porn_labels = batch["porn_label"].to(device, non_blocking=True)
        gore_labels = batch["gore_label"].to(device, non_blocking=True)
        ip_labels = batch["ip_label"].to(device, non_blocking=True)
        meta = batch["meta"]

        prompt_embeds = batch.get("prompt_embeds")
        if prompt_embeds is not None:
            prompt_embeds = prompt_embeds.to(device, non_blocking=True)
        prompt_mask = batch.get("prompt_mask")
        if prompt_mask is not None:
            prompt_mask = prompt_mask.to(device, non_blocking=True)

        porn_logits, gore_logits, ip_logits = model(inputs, prompt_embeds=prompt_embeds, prompt_mask=prompt_mask)
        p_loss = criterion_bin(porn_logits, porn_labels)
        g_loss = criterion_bin(gore_logits, gore_labels)
        ip_loss = criterion_ip(ip_logits, ip_labels)
        loss = lambda_porn * p_loss + lambda_gore * g_loss + lambda_ip * ip_loss

        bs = inputs.size(0)
        total_loss += float(loss.item()) * bs
        total_count += bs
        ip_loss_sum += float(ip_loss.item()) * bs
        ip_count += bs

        porn_true.extend(porn_labels.cpu().numpy().tolist())
        porn_pred_argmax.extend(torch.argmax(porn_logits, dim=1).cpu().numpy().tolist())
        porn_prob.extend(torch.softmax(porn_logits, dim=1)[:, _porn_prob_idx].cpu().numpy().tolist())

        gore_true.extend(gore_labels.cpu().numpy().tolist())
        gore_pred_argmax.extend(torch.argmax(gore_logits, dim=1).cpu().numpy().tolist())
        gore_prob.extend(torch.softmax(gore_logits, dim=1)[:, _gore_prob_idx].cpu().numpy().tolist())

        ip_probs_batch = torch.softmax(ip_logits, dim=1)
        ip_true_all.extend(ip_labels.cpu().numpy().tolist())
        ip_pred_all_argmax.extend(torch.argmax(ip_probs_batch, dim=1).cpu().numpy().tolist())
        ip_probs_all.append(ip_probs_batch.cpu().numpy())

        for k, v in meta.items():
            meta_store[k].extend(list(v))

    avg_loss = total_loss / max(1, total_count)

    # merge ip_probs_all into numpy
    ip_probs_all = (np.concatenate(ip_probs_all, axis=0) if ip_probs_all
                    else np.zeros((0, 6), dtype=np.float32))

    threshold_results = {"porn": [], "gore": []}
    ip_threshold_rows = []

    if scan_threshold and total_count > 0:
        # ====== threshold-sweep mode ======
        thresholds = [round(i * 0.05, 2) for i in range(1, 20)]
        threshold_results["porn"] = compute_metrics_by_thresholds(porn_true, porn_prob, thresholds)
        threshold_results["gore"] = compute_metrics_by_thresholds(gore_true, gore_prob, thresholds)
        if verbose:
            porn_best = print_threshold_table_compact("Porn", threshold_results["porn"], split=split)
            gore_best = print_threshold_table_compact("Gore", threshold_results["gore"], split=split)
        else:
            porn_best = pick_best_threshold_row(threshold_results["porn"])
            gore_best = pick_best_threshold_row(threshold_results["gore"])

        if porn_best is not None:
            porn_metrics, porn_pred_best = compute_metrics_at_threshold(porn_true, porn_prob, porn_best["threshold"])
        else:
            porn_metrics, porn_pred_best = empty_metrics(), [0] * len(porn_true)
        if gore_best is not None:
            gore_metrics, gore_pred_best = compute_metrics_at_threshold(gore_true, gore_prob, gore_best["threshold"])
        else:
            gore_metrics, gore_pred_best = empty_metrics(), [0] * len(gore_true)

        best_ip = None
        if len(ip_true_all) > 0:
            ip_threshold_rows = compute_ip_metrics_by_shared_thresholds(
                ip_true_all, ip_probs_all, thresholds=[i / 100 for i in range(5, 100, 5)],
            )
            if verbose:
                best_ip = print_threshold_table_compact("IP(1-5 shared-th)", ip_threshold_rows, split=split)
            else:
                best_ip = pick_best_threshold_row(ip_threshold_rows)

            if best_ip is not None:
                ip_pred_best = predict_ip_with_shared_threshold(ip_probs_all, best_ip["threshold"])
                ip_true_bin = np.isin(np.array(ip_true_all), [0, 1, 2, 3, 4]).astype(int).tolist()
                ip_pred_bin = np.isin(np.array(ip_pred_best), [0, 1, 2, 3, 4]).astype(int).tolist()
                ip_pos_metrics = compute_binary_metrics(ip_true_bin, ip_pred_bin)
            else:
                ip_pred_best = [5] * len(ip_true_all)
                ip_pos_metrics = empty_metrics()
        else:
            ip_pred_best = [5] * len(ip_true_all)
            ip_pos_metrics = empty_metrics()

        if porn_best is None:
            porn_best = {"threshold": 0.5, "f1": porn_metrics["f1"]}
        if gore_best is None:
            gore_best = {"threshold": 0.5, "f1": gore_metrics["f1"]}
        if best_ip is None:
            best_ip = {"threshold": 0.5, "f1": ip_pos_metrics["f1"]}

        total_risk_stats = compute_total_risk_stats(
            porn_true=porn_true, porn_prob=porn_prob, porn_th=porn_best["threshold"],
            gore_true=gore_true, gore_prob=gore_prob, gore_th=gore_best["threshold"],
            ip_true_full=ip_true_all, ip_probs_full=ip_probs_all, ip_th=best_ip["threshold"],
        )

    else:
        # ====== argmax mode (default) ======
        porn_pred_best = porn_pred_argmax
        if _porn_3class:
            porn_metrics = compute_multiclass_metrics(porn_true, porn_pred_argmax, 3,
                                                       class_names=["safe", "borderline", "risk"])
        else:
            porn_metrics = compute_binary_metrics(porn_true, porn_pred_argmax)
        gore_pred_best = gore_pred_argmax
        if _gore_3class:
            gore_metrics = compute_multiclass_metrics(gore_true, gore_pred_argmax, 3,
                                                       class_names=["safe", "borderline", "risk"])
        else:
            gore_metrics = compute_binary_metrics(gore_true, gore_pred_argmax)
        porn_best = {"threshold": -1.0 if _porn_3class else 0.5, "f1": porn_metrics["f1"]}
        gore_best = {"threshold": -1.0 if _gore_3class else 0.5, "f1": gore_metrics["f1"]}

        ip_pred_best = ip_pred_all_argmax
        if len(ip_true_all) > 0:
            if _porn_3class or _gore_3class:
                # 3cls mode: IP also uses macro-F1 (6 classes: 0-4 risk subclasses + 5=other)
                ip_pos_metrics = compute_multiclass_metrics(
                    ip_true_all, ip_pred_best, 6,
                    class_names=[IP_ID2NAME[i] for i in range(6)])
            else:
                ip_true_bin = np.isin(np.array(ip_true_all), [0, 1, 2, 3, 4]).astype(int).tolist()
                ip_pred_bin = np.isin(np.array(ip_pred_best), [0, 1, 2, 3, 4]).astype(int).tolist()
                ip_pos_metrics = compute_binary_metrics(ip_true_bin, ip_pred_bin)
        else:
            ip_pos_metrics = empty_multiclass_metrics(6) if (_porn_3class or _gore_3class) else empty_metrics()
        best_ip = {"threshold": 0.5, "f1": ip_pos_metrics["f1"]}

        # total_risk_stats: computed directly from argmax predictions
        _porn_risk_class = 2 if _porn_3class else 1
        _gore_risk_class = 2 if _gore_3class else 1
        _porn_true_arr = np.array(porn_true).astype(int)
        _gore_true_arr = np.array(gore_true).astype(int)
        _ip_true_arr = np.array(ip_true_all).astype(int)
        _porn_pred_arr = np.array(porn_pred_argmax).astype(int)
        _gore_pred_arr = np.array(gore_pred_argmax).astype(int)
        _ip_pred_arr = np.array(ip_pred_best).astype(int)
        _pred_ip_risk = np.isin(_ip_pred_arr, [0, 1, 2, 3, 4]).astype(int)
        _gt_ip_risk = np.isin(_ip_true_arr, [0, 1, 2, 3, 4]).astype(int)
        _gt_total_risk = ((_porn_true_arr == _porn_risk_class) | (_gore_true_arr == _gore_risk_class) | (_gt_ip_risk == 1)).astype(int)
        _pred_total_risk = ((_porn_pred_arr == _porn_risk_class) | (_gore_pred_arr == _gore_risk_class) | (_pred_ip_risk == 1)).astype(int)
        _tp = int(((_gt_total_risk == 1) & (_pred_total_risk == 1)).sum())
        _fp = int(((_gt_total_risk == 0) & (_pred_total_risk == 1)).sum())
        _fn = int(((_gt_total_risk == 1) & (_pred_total_risk == 0)).sum())
        _tn = int(((_gt_total_risk == 0) & (_pred_total_risk == 0)).sum())
        _total_risk = int((_gt_total_risk == 1).sum())
        _total_normal = int((_gt_total_risk == 0).sum())
        _total_samples = int(len(_gt_total_risk))
        total_risk_stats = {
            "total_samples": _total_samples, "total_risk_samples": _total_risk,
            "total_normal_samples": _total_normal, "recalled_risk_samples": _tp,
            "missed_risk_samples": _fn, "disturbed_normal_samples": _fp,
            "safe_normal_samples": _tn,
            "total_risk_recall": _tp / _total_risk if _total_risk > 0 else 0.0,
            "e2e_unsafety": _fn / _total_samples if _total_samples > 0 else 0.0,
            "normal_disturb_rate": _fp / _total_normal if _total_normal > 0 else 0.0,
            "pred_total_risk_count": int((_pred_total_risk == 1).sum()),
            "gt_total_risk_flags": _gt_total_risk.tolist(),
            "pred_total_risk_flags": _pred_total_risk.tolist(),
        }

    # --- Confusion Matrix & IP verbose print ---
    cm = np.zeros((6, 6), dtype=np.int64)
    if len(ip_true_all) > 0:
        cm = compute_confusion_matrix(ip_true_all, ip_pred_best, num_classes=6)
        if verbose:
            print_ip_multiclass_metrics(ip_true_all, ip_pred_best,
                                         class_names=[IP_ID2NAME[i] for i in range(6)], split=split)
            print_confusion_matrix(cm, [IP_ID2NAME[i] for i in range(6)],
                                    title=f"{split} IP 6-Class Confusion Matrix")

    # --- Summary table ---
    _mode_label = "Best-Threshold" if scan_threshold else "Argmax"
    def _eval_th(x):
        return "N/A" if (x is None or float(x) < 0) else f"{float(x):.2f}"
    def _eval_pr(metrics):
        if metrics.get("multiclass", False):
            return (f"{metrics['precision'] * 100:.2f}%", f"{metrics['recall'] * 100:.2f}%")
        return (fmt_percent_ratio(metrics["precision"], metrics["tp"], metrics["tp"] + metrics["fp"]),
                fmt_percent_ratio(metrics["recall"], metrics["tp"], metrics["tp"] + metrics["fn"]))
    _porn_p, _porn_r = _eval_pr(porn_metrics)
    _gore_p, _gore_r = _eval_pr(gore_metrics)
    _ip_p, _ip_r = _eval_pr(ip_pos_metrics)
    metric_rows = [
        ["Porn", _eval_th(porn_best['threshold']), f"{porn_metrics['acc'] * 100:.2f}%",
         _porn_p, _porn_r, f"{porn_metrics['f1'] * 100:.2f}%"],
        ["Gore", _eval_th(gore_best['threshold']), f"{gore_metrics['acc'] * 100:.2f}%",
         _gore_p, _gore_r, f"{gore_metrics['f1'] * 100:.2f}%"],
        ["IP", _eval_th(best_ip['threshold']), f"{ip_pos_metrics['acc'] * 100:.2f}%",
         _ip_p, _ip_r, f"{ip_pos_metrics['f1'] * 100:.2f}%"],
    ]
    print_table(["Task", "Threshold", "Acc", "Precision", "Recall", "F1"], metric_rows,
                title=f"[{split}] Epoch {epoch} Summary / {_mode_label} Metrics", width_limit=28)
    print_total_risk_stats_table(total_risk_stats, split=split)

    # in 3-class mode also print per-class F1
    if _porn_3class and porn_metrics.get("multiclass"):
        _pc = porn_metrics["per_class"]
        _pc_rows = [[_pc[c].get("name", c), _pc[c]["support"],
                     f"{_pc[c]['precision'] * 100:.2f}%", f"{_pc[c]['recall'] * 100:.2f}%",
                     f"{_pc[c]['f1'] * 100:.2f}%"] for c in range(3)]
        print_table(["Porn Class", "Support", "Precision", "Recall", "F1"], _pc_rows,
                    title=f"[{split}] Epoch {epoch} Porn Per-Class (macro-F1={porn_metrics['f1']:.4f})", width_limit=28)
    if _gore_3class and gore_metrics.get("multiclass"):
        _pc = gore_metrics["per_class"]
        _pc_rows = [[_pc[c].get("name", c), _pc[c]["support"],
                     f"{_pc[c]['precision'] * 100:.2f}%", f"{_pc[c]['recall'] * 100:.2f}%",
                     f"{_pc[c]['f1'] * 100:.2f}%"] for c in range(3)]
        print_table(["Gore Class", "Support", "Precision", "Recall", "F1"], _pc_rows,
                    title=f"[{split}] Epoch {epoch} Gore Per-Class (macro-F1={gore_metrics['f1']:.4f})", width_limit=28)
    if (_porn_3class or _gore_3class) and ip_pos_metrics.get("multiclass"):
        _pc = ip_pos_metrics["per_class"]
        _pc_rows = [[_pc[c].get("name", c), _pc[c]["support"],
                     f"{_pc[c]['precision'] * 100:.2f}%", f"{_pc[c]['recall'] * 100:.2f}%",
                     f"{_pc[c]['f1'] * 100:.2f}%"] for c in range(6)]
        print_table(["IP Class", "Support", "Precision", "Recall", "F1"], _pc_rows,
                    title=f"[{split}] Epoch {epoch} IP Per-Class (macro-F1={ip_pos_metrics['f1']:.4f})", width_limit=28)

    ip_pred_all_best = ip_pred_best if len(ip_true_all) > 0 else [5] * len(porn_true)

    results = []
    n = len(porn_true)
    for i in range(n):
        one = {
            "name": meta_store["name"][i],
            "video_path": meta_store["video_path"][i],
            "predict": meta_store["predict"][i],
            "three_class": meta_store["three_class"][i],
            "sampling_group": meta_store["sampling_group"][i],
            "ip_group": meta_store["ip_group"][i],
            "train_group": meta_store["train_group"][i],
            "manually_modified": bool(meta_store["manually_modified"][i]) if "manually_modified" in meta_store else False,
            "has_ip_label": True,
            "ip_label": int(ip_true_all[i]),
            "ip_pred": int(ip_pred_all_best[i]),
            "ip_best_threshold": float(best_ip["threshold"]),
            "ip_risk_score": float(np.max(ip_probs_all[i][:5])) if len(ip_probs_all) > i else 0.0,
            "ip_risk_pred_best": bool(int(ip_pred_all_best[i]) in (0, 1, 2, 3, 4)),
            "porn_label": int(porn_true[i]),
            "porn_pred": int(porn_pred_best[i]),
            "porn_prob": float(porn_prob[i]),
            "porn_correct": bool(porn_true[i] == porn_pred_best[i]),
            "gore_label": int(gore_true[i]),
            "gore_pred": int(gore_pred_best[i]),
            "gore_prob": float(gore_prob[i]),
            "gore_correct": bool(gore_true[i] == gore_pred_best[i]),
            "pred_total_risk_best": bool(total_risk_stats["pred_total_risk_flags"][i]),
            "gt_total_risk": bool(total_risk_stats["gt_total_risk_flags"][i]),
        }
        for k in ["total_frames", "sampled_indices", "middle_frame_idx", "latent_root", "step"]:
            if k in meta_store and len(meta_store[k]) > i:
                one[k] = meta_store[k][i]
        results.append(one)

    ip_eval_extra = {
        "ip_pos_metrics": ip_pos_metrics,
        "ip_confusion_matrix": cm.tolist(),
        "ip_class_names": [IP_ID2NAME[i] for i in range(6)],
        "ip_labeled_count": int(len(ip_true_all)),
        "ip_avg_loss": float(ip_loss_sum / max(1, ip_count)),
        "ip_risk_threshold_metrics": ip_threshold_rows,
        "ip_risk_best_threshold": float(best_ip["threshold"]) if best_ip is not None else None,
        "ip_risk_best_f1": float(best_ip["f1"]) if best_ip is not None else None,
        "porn_best_threshold": float(porn_best["threshold"]) if porn_best is not None else None,
        "porn_best_f1": float(porn_best["f1"]) if porn_best is not None else None,
        "gore_best_threshold": float(gore_best["threshold"]) if gore_best is not None else None,
        "gore_best_f1": float(gore_best["f1"]) if gore_best is not None else None,
        "total_risk_stats": total_risk_stats,
    }
    return avg_loss, porn_metrics, gore_metrics, ip_eval_extra, results, threshold_results


# =========================
# Badcase
# =========================
def save_badcases(results, save_dir, epoch, max_per_type=100):
    base = os.path.join(save_dir, "badcase", f"epoch_{epoch}")
    porn_dir = os.path.join(base, "porn")
    gore_dir = os.path.join(base, "gore")
    ip_dir = os.path.join(base, "ip")
    risk_dir = os.path.join(base, "total_risk")
    for d in [porn_dir, gore_dir, ip_dir, risk_dir]:
        os.makedirs(d, exist_ok=True)

    porn_bad = [x for x in results if not x["porn_correct"]]
    gore_bad = [x for x in results if not x["gore_correct"]]
    ip_bad = [x for x in results if int(x.get("ip_label", -1)) != int(x.get("ip_pred", -999))]
    missed = [x for x in results if x.get("gt_total_risk", False) and not x.get("pred_total_risk_best", False)]
    false_alarm = [x for x in results if not x.get("gt_total_risk", False) and x.get("pred_total_risk_best", False)]

    save_json(porn_bad, os.path.join(porn_dir, "meta.json"))
    save_json(gore_bad, os.path.join(gore_dir, "meta.json"))
    save_json(ip_bad, os.path.join(ip_dir, "meta.json"))
    save_json({"missed_risk": missed, "false_alarm_normal": false_alarm}, os.path.join(risk_dir, "meta.json"))

    def _copy_sub(bads, dst, name_fn):
        n = 0
        for x in bads[:max_per_type]:
            src = x["video_path"]
            if safe_copy(src, os.path.join(dst, name_fn(x))):
                n += 1
        return n

    porn_saved = _copy_sub(porn_bad, porn_dir, lambda x:
        f"{Path(x['video_path']).stem}__predfield_{str(x['predict']).replace('/', '_')}__"
        f"gt{x['porn_label']}__pred{x['porn_pred']}__prob{x['porn_prob']:.4f}{Path(x['video_path']).suffix or '.mp4'}")
    gore_saved = _copy_sub(gore_bad, gore_dir, lambda x:
        f"{Path(x['video_path']).stem}__predfield_{str(x['predict']).replace('/', '_')}__"
        f"gt{x['gore_label']}__pred{x['gore_pred']}__prob{x['gore_prob']:.4f}{Path(x['video_path']).suffix or '.mp4'}")
    ip_saved = _copy_sub(ip_bad, ip_dir, lambda x:
        f"{Path(x['video_path']).stem}__gt_{IP_ID2NAME.get(int(x['ip_label']), str(x['ip_label']))}__"
        f"pred_{IP_ID2NAME.get(int(x['ip_pred']), str(x['ip_pred']))}{Path(x['video_path']).suffix or '.mp4'}")

    print_table(
        ["epoch", "porn_saved", "gore_saved", "ip_saved", "missed_risk", "false_alarm_normal"],
        [[epoch, f"{porn_saved}/{min(len(porn_bad), max_per_type)}",
          f"{gore_saved}/{min(len(gore_bad), max_per_type)}",
          f"{ip_saved}/{min(len(ip_bad), max_per_type)}",
          len(missed), len(false_alarm)]],
        title="Badcase Saved",
    )


# =========================
# Model Load Info
# =========================
def print_model_load_summary(model):
    """Print model-load info; list missing/unexpected/shape_mismatch keys in detail."""
    info = getattr(model, "load_info", None)
    if info is None:
        print_table(["Field", "Value"], [["load_info", "None"]], title="Model Load Summary")
        return

    total_params = sum(p.numel() for p in model.parameters())
    missing_keys = set(info.get("missing_keys", []))
    shape_mismatch_keys = {item["key"] for item in info.get("skipped_shape_mismatch", [])}
    not_loaded_keys = missing_keys | shape_mismatch_keys
    not_loaded_params = sum(
        p.numel() for name, p in model.named_parameters() if name in not_loaded_keys
    )
    loaded_params = total_params - not_loaded_params
    loaded_ratio = loaded_params / total_params if total_params > 0 else 0.0

    print_table(
        ["Field", "Value"],
        [["loaded", info.get("loaded", False)],
         ["checkpoint_path", info.get("checkpoint_path", "")],
         ["note", info.get("note", "")],
         ["total_params", f"{total_params:,}"],
         ["loaded_params", f"{loaded_params:,}"],
         ["loaded_ratio", f"{loaded_ratio:.2%}"],
         ["missing_keys_count", len(info.get("missing_keys", []))],
         ["unexpected_keys_count", len(info.get("unexpected_keys", []))],
         ["unexpected_keys_after_remap_count", len(info.get("unexpected_keys_after_remap", []))],
         ["shape_mismatch_count", len(info.get("skipped_shape_mismatch", []))]],
        title="Model Load Summary", width_limit=120,
    )

    for header, key in [("Missing Keys", "missing_keys"),
                        ("Unexpected Keys", "unexpected_keys"),
                        ("Unexpected Keys After Remap", "unexpected_keys_after_remap")]:
        items = info.get(key, [])
        if items:
            print_table([header], [[k] for k in items],
                        title=f"Model Load Summary / {header}", width_limit=140)

    skipped = info.get("skipped_shape_mismatch", [])
    if skipped:
        print_table(["Key", "CkptShape", "ModelShape"],
                    [[x["key"], str(x["ckpt_shape"]), str(x["model_shape"])] for x in skipped],
                    title="Model Load Summary / Shape Mismatch", width_limit=140)


def print_pe_mlp_load_summary(model):
    """Print PE-MLP encoder load info (called in PE2 mode)."""
    pf = getattr(model, "prompt_fusion", None)
    if pf is None:
        return
    info = getattr(pf, "pe_mlp_load_info", None)
    if info is None:
        return  # not pe2 mode, or no ckpt loaded

    print_table(
        ["Field", "Value"],
        [["loaded", info.get("loaded", False)],
         ["checkpoint_path", info.get("checkpoint_path", "")],
         ["global_step", info.get("global_step", "?")],
         ["test_loss", _fmt_num(info.get("test_loss", "?"))],
         ["acc_avg", _fmt_num(info.get("acc_avg", "?"))],
         ["input_dim", info.get("input_dim", "?")],
         ["hidden_dim", info.get("hidden_dim", "?")],
         ["num_layers", info.get("num_layers", "?")],
         ["ckpt_total_keys", info.get("total_keys_in_ckpt", 0)],
         ["encoder_keys_loaded", f"{info.get('encoder_keys_loaded', 0)}/{info.get('encoder_keys_total', 0)}"],
         ["encoder_params", f"{info.get('encoder_total_params', 0):,}"],
         ["missing_keys_count", len(info.get("missing_keys", []))],
         ["unexpected_keys_count", len(info.get("unexpected_keys", []))],
         ["note", info.get("note", "")]],
        title="PE-MLP Encoder Load Summary", width_limit=120,
    )

    for header, key in [("Missing Keys", "missing_keys"),
                        ("Unexpected Keys", "unexpected_keys")]:
        items = info.get(key, [])
        if items:
            print_table([header], [[k] for k in items],
                        title=f"PE-MLP Load Summary / {header}", width_limit=140)


def _fmt_num(v):
    """Format a number: floats keep 4 decimals, everything else as-is."""
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


# =========================
# LR Scheduler
# =========================
def build_lr_scheduler(optimizer, config, t_max_override=None):
    """Build the LR scheduler from config['lr_scheduler'].
    Supported: 'none' | 'cosine'
    t_max_override: overrides T_max (used for staged unfreezing).
    """
    sched = str(config.get("lr_scheduler", "none")).lower()
    if sched in ("none", "", "off"):
        return None
    if sched == "cosine":
        t_max = t_max_override if t_max_override is not None else int(config["epochs"])
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=t_max,
            eta_min=float(config.get("lr_min", 0.0)),
        )
    raise ValueError(f"Unknown lr_scheduler: {sched!r}")


# =========================
# Runtime / Save
# =========================
def prepare_output_dirs(ckpt_dir):
    for sub in ["ckpts", "plots", "meta", "badcase"]:
        os.makedirs(os.path.join(ckpt_dir, sub), exist_ok=True)


def setup_logger(ckpt_dir):
    sys.stdout = Logger(os.path.join(ckpt_dir, "log.txt"))


def append_test_metrics_csv(csv_path, epoch, test_porn_metrics, test_gore_metrics, test_ip_eval_extra):
    """Append one row of test metrics to csv_path per epoch (header written on
    first use). 3-class mode outputs macro-avg precision/recall + per-class F1 details (including the 6 IP classes)."""
    def fmt_pct_ratio(num, den):
        num = int(num); den = int(den)
        if den <= 0:
            return f"0.00% ({num}/{den})"
        return f"{num / den * 100:.2f}% ({num}/{den})"

    def fmt_pct(x):
        return f"{float(x) * 100:.2f}%"

    def fmt_th(x):
        if x is None or float(x) < 0:
            return "N/A"
        return f"{float(x):.2f}"

    ip_pos = (test_ip_eval_extra or {}).get("ip_pos_metrics", {}) or {}
    trs = (test_ip_eval_extra or {}).get("total_risk_stats", {}) or {}

    _porn_mc = test_porn_metrics.get("multiclass", False)
    _gore_mc = test_gore_metrics.get("multiclass", False)

    # --- porn columns ---
    if _porn_mc:
        _pc = test_porn_metrics.get("per_class", {})
        porn_cols = {
            "porn_precision": fmt_pct(test_porn_metrics["precision"]),
            "porn_recall":    fmt_pct(test_porn_metrics["recall"]),
            "porn_f1":        fmt_pct(test_porn_metrics["f1"]),
        }
        for c, cname in enumerate(["safe", "borderline", "risk"]):
            _e = _pc.get(c, {})
            porn_cols[f"porn_{cname}_f1"] = fmt_pct(_e.get("f1", 0.0))
            porn_cols[f"porn_{cname}_support"] = str(_e.get("support", 0))
    else:
        porn_cols = {
            "porn_precision": fmt_pct_ratio(test_porn_metrics["tp"], test_porn_metrics["tp"] + test_porn_metrics["fp"]),
            "porn_recall":    fmt_pct_ratio(test_porn_metrics["tp"], test_porn_metrics["tp"] + test_porn_metrics["fn"]),
            "porn_f1":        fmt_pct(test_porn_metrics["f1"]),
        }

    # --- gore columns ---
    if _gore_mc:
        _pc = test_gore_metrics.get("per_class", {})
        gore_cols = {
            "gore_precision": fmt_pct(test_gore_metrics["precision"]),
            "gore_recall":    fmt_pct(test_gore_metrics["recall"]),
            "gore_f1":        fmt_pct(test_gore_metrics["f1"]),
        }
        for c, cname in enumerate(["safe", "borderline", "risk"]):
            _e = _pc.get(c, {})
            gore_cols[f"gore_{cname}_f1"] = fmt_pct(_e.get("f1", 0.0))
            gore_cols[f"gore_{cname}_support"] = str(_e.get("support", 0))
    else:
        gore_cols = {
            "gore_precision": fmt_pct_ratio(test_gore_metrics["tp"], test_gore_metrics["tp"] + test_gore_metrics["fp"]),
            "gore_recall":    fmt_pct_ratio(test_gore_metrics["tp"], test_gore_metrics["tp"] + test_gore_metrics["fn"]),
            "gore_f1":        fmt_pct(test_gore_metrics["f1"]),
        }

    row = {
        "epoch":             int(epoch),
        "porn_threshold":    fmt_th((test_ip_eval_extra or {}).get("porn_best_threshold")),
    }
    row.update(porn_cols)
    row["gore_threshold"] = fmt_th((test_ip_eval_extra or {}).get("gore_best_threshold"))
    row.update(gore_cols)
    # --- IP columns --- (3cls: macro-avg + per-class; binary: binary metrics)
    _ip_mc = ip_pos.get("multiclass", False)
    if _ip_mc:
        _pc = ip_pos.get("per_class", {})
        _ip_names = [IP_ID2NAME[i] for i in range(ip_pos.get("num_classes", 6))]
        ip_cols = {
            "ip_precision": fmt_pct(ip_pos["precision"]),
            "ip_recall":    fmt_pct(ip_pos["recall"]),
            "ip_f1":        fmt_pct(ip_pos["f1"]),
        }
        for c, cname in enumerate(_ip_names):
            _e = _pc.get(c, {})
            ip_cols[f"ip_{cname}_f1"] = fmt_pct(_e.get("f1", 0.0))
            ip_cols[f"ip_{cname}_support"] = str(_e.get("support", 0))
    else:
        ip_cols = {
            "ip_precision": fmt_pct_ratio(ip_pos.get("tp", 0), ip_pos.get("tp", 0) + ip_pos.get("fp", 0)),
            "ip_recall":    fmt_pct_ratio(ip_pos.get("tp", 0), ip_pos.get("tp", 0) + ip_pos.get("fn", 0)),
            "ip_f1":        fmt_pct(ip_pos.get("f1", 0.0)),
        }

    row["ip_threshold"] = fmt_th((test_ip_eval_extra or {}).get("ip_risk_best_threshold"))
    row.update(ip_cols)
    row.update({
        "total_risk_recall":   fmt_pct_ratio(trs.get("recalled_risk_samples", 0), trs.get("total_risk_samples", 0)),
        "e2e_unsafety":        fmt_pct_ratio(trs.get("missed_risk_samples", 0), trs.get("total_samples", 0)),
        "normal_disturb_rate": fmt_pct_ratio(trs.get("disturbed_normal_samples", 0), trs.get("total_normal_samples", 0)),
    })

    write_header = (not os.path.exists(csv_path)) or os.path.getsize(csv_path) == 0
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def save_epoch_artifacts(ckpt_dir, epoch, history, model,
                         val_results, test_results,
                         val_threshold_results=None, test_threshold_results=None,
                         val_ip_eval_extra=None, test_ip_eval_extra=None,
                         test_porn_metrics=None, test_gore_metrics=None):
    """Save all per-epoch artifacts: weights, metrics JSON, plots, badcases."""
    save_history_and_plots(history, os.path.join(ckpt_dir, "plots"), epoch)
    torch.save(model.state_dict(), os.path.join(ckpt_dir, "ckpts", f"model_epoch_{epoch}.pth"))

    save_json(val_results, os.path.join(ckpt_dir, "meta", f"val_results_epoch_{epoch}.json"))
    if test_results is not None:
        save_json(test_results, os.path.join(ckpt_dir, "meta", f"test_results_epoch_{epoch}.json"))
    if val_threshold_results is not None:
        save_json(val_threshold_results, os.path.join(ckpt_dir, "meta", f"val_threshold_metrics_epoch_{epoch}.json"))
    if test_threshold_results is not None:
        save_json(test_threshold_results, os.path.join(ckpt_dir, "meta", f"test_threshold_metrics_epoch_{epoch}.json"))
    if val_ip_eval_extra is not None:
        save_json(val_ip_eval_extra, os.path.join(ckpt_dir, "meta", f"val_ip_metrics_epoch_{epoch}.json"))
    if test_ip_eval_extra is not None:
        save_json(test_ip_eval_extra, os.path.join(ckpt_dir, "meta", f"test_ip_metrics_epoch_{epoch}.json"))

    if test_porn_metrics is not None and test_gore_metrics is not None and test_ip_eval_extra is not None:
        append_test_metrics_csv(
            os.path.join(ckpt_dir, "meta", "test_metrics.csv"),
            epoch=epoch,
            test_porn_metrics=test_porn_metrics,
            test_gore_metrics=test_gore_metrics,
            test_ip_eval_extra=test_ip_eval_extra,
        )

    save_json([x for x in val_results if not x["porn_correct"]],
              os.path.join(ckpt_dir, "meta", f"val_porn_errors_epoch_{epoch}.json"))
    save_json([x for x in val_results if not x["gore_correct"]],
              os.path.join(ckpt_dir, "meta", f"val_gore_errors_epoch_{epoch}.json"))
    if test_results is not None:
        save_json([x for x in test_results if not x["porn_correct"]],
                  os.path.join(ckpt_dir, "meta", f"test_porn_errors_epoch_{epoch}.json"))
        save_json([x for x in test_results if not x["gore_correct"]],
                  os.path.join(ckpt_dir, "meta", f"test_gore_errors_epoch_{epoch}.json"))
        save_badcases(test_results, ckpt_dir, epoch, max_per_type=100)


def update_best_checkpoints(model, ckpt_dir, epoch, val_results, test_results,
                             val_porn, val_gore, val_ip_eval, test_porn, test_gore, test_ip_eval,
                             best_val_avg_f1, best_test_avg_f1):
    """Update the best checkpoints (by avg F1 over the three val/test tasks)."""
    def pick(extra, fallback_metrics, key):
        v = extra.get(key)
        return v if v is not None else fallback_metrics["f1"]

    val_avg = (pick(val_ip_eval, val_porn, "porn_best_f1")
               + pick(val_ip_eval, val_gore, "gore_best_f1")
               + (val_ip_eval.get("ip_risk_best_f1") or val_ip_eval["ip_pos_metrics"]["f1"])) / 3.0

    if val_avg > best_val_avg_f1:
        best_val_avg_f1 = val_avg
        torch.save(model.state_dict(), os.path.join(ckpt_dir, "ckpts", "best_val_avg_f1_model.pth"))
        save_json(val_results, os.path.join(ckpt_dir, "meta", "best_val_results.json"))
        print_table(["epoch", "best_val_avg_f1"], [[epoch, f"{best_val_avg_f1:.4f}"]],
                    title="Best Val Avg F1 Updated")

    # The deployed checkpoint is selected by validation avg-F1. best_test is
    # tracked only when test metrics are explicitly passed in (via
    # eval_test_every_epoch=True).
    if test_results is not None and test_ip_eval is not None:
        test_avg = (pick(test_ip_eval, test_porn, "porn_best_f1")
                    + pick(test_ip_eval, test_gore, "gore_best_f1")
                    + (test_ip_eval.get("ip_risk_best_f1") or test_ip_eval["ip_pos_metrics"]["f1"])) / 3.0
        if test_avg > best_test_avg_f1:
            best_test_avg_f1 = test_avg
            torch.save(model.state_dict(), os.path.join(ckpt_dir, "ckpts", "best_test_avg_f1_model.pth"))
            save_json(test_results, os.path.join(ckpt_dir, "meta", "best_test_results.json"))
            print_table(["epoch", "best_test_avg_f1"], [[epoch, f"{best_test_avg_f1:.4f}"]],
                        title="Best Test Avg F1 Updated")
    return best_val_avg_f1, best_test_avg_f1


# =========================
# Epoch Summary
# =========================
def _epoch_summary_row(epoch, train_loss, val_loss, test_loss,
                       val_porn, val_gore, val_ip_eval,
                       test_porn, test_gore, test_ip_eval):
    def _th(x):
        return "N/A" if (x is None or float(x) < 0) else f"{float(x):.2f}"
    # test_* are None when the test set is not evaluated this epoch
    # (eval_test_every_epoch=False)
    test_off = test_ip_eval is None
    def _f1(m):
        return "N/A" if m is None else f"{m['f1']:.4f}"
    return [
        epoch,
        f"{train_loss:.4f}" if isinstance(train_loss, float) else train_loss,
        f"{val_loss:.4f}",
        "N/A" if test_loss is None else f"{test_loss:.4f}",
        f"{val_porn['f1']:.4f}", f"{val_gore['f1']:.4f}",
        f"{val_ip_eval['ip_pos_metrics']['f1']:.4f}",
        _th(val_ip_eval.get('porn_best_threshold')),
        _th(val_ip_eval.get('gore_best_threshold')),
        _th(val_ip_eval.get('ip_risk_best_threshold')),
        _f1(test_porn), _f1(test_gore),
        "N/A" if test_off else f"{test_ip_eval['ip_pos_metrics']['f1']:.4f}",
        "N/A" if test_off else _th(test_ip_eval.get('porn_best_threshold')),
        "N/A" if test_off else _th(test_ip_eval.get('gore_best_threshold')),
        "N/A" if test_off else _th(test_ip_eval.get('ip_risk_best_threshold')),
        "N/A" if test_off else f"{test_ip_eval['total_risk_stats']['total_risk_recall']:.4f}",
        "N/A" if test_off else f"{test_ip_eval['total_risk_stats']['e2e_unsafety']:.4f}",
        "N/A" if test_off else f"{test_ip_eval['total_risk_stats']['normal_disturb_rate']:.4f}",
    ]


_EPOCH_SUMMARY_HEADERS = [
    "epoch", "train_loss", "val_loss", "test_loss",
    "val_porn_f1", "val_gore_f1", "val_ip_f1",
    "val_p_th", "val_g_th", "val_ip_th",
    "test_porn_f1", "test_gore_f1", "test_ip_f1",
    "test_p_th", "test_g_th", "test_ip_th",
    "test_total_risk_recall", "test_e2e_unsafety", "test_normal_disturb_rate",
]


# =========================
# Smoke Test
# =========================
class _LimitedLoader:
    """Wrap a DataLoader so each epoch yields only the first max_batches batches (smoke-test helper)."""
    def __init__(self, loader, max_batches):
        self._loader = loader
        self.max_batches = int(max_batches)
        for attr in ("dataset", "batch_size", "num_workers", "sampler"):
            if hasattr(loader, attr):
                setattr(self, attr, getattr(loader, attr))

    def __iter__(self):
        return iter(itertools.islice(self._loader, self.max_batches))

    def __len__(self):
        try:
            return min(self.max_batches, len(self._loader))
        except TypeError:
            return self.max_batches


def _snapshot_train_state(model, optimizer, scheduler):
    """Snapshot the training state; restored after the smoke test finishes."""
    return {
        "model": copy.deepcopy(model.state_dict()),
        "optimizer": copy.deepcopy(optimizer.state_dict()),
        "scheduler": copy.deepcopy(scheduler.state_dict()) if scheduler is not None else None,
        "rng_torch": torch.get_rng_state(),
        "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "rng_random": random.getstate(),
        "rng_numpy": np.random.get_state(),
    }


def _restore_train_state(snapshot, model, optimizer, scheduler):
    model.load_state_dict(snapshot["model"])
    optimizer.load_state_dict(snapshot["optimizer"])
    if scheduler is not None and snapshot["scheduler"] is not None:
        scheduler.load_state_dict(snapshot["scheduler"])
    torch.set_rng_state(snapshot["rng_torch"])
    if snapshot["rng_cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(snapshot["rng_cuda"])
    random.setstate(snapshot["rng_random"])
    np.random.set_state(snapshot["rng_numpy"])
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_smoke_test(config, device, ckpt_dir, model, optimizer, scheduler,
                   criterion_bin, criterion_ip,
                   train_loader, val_loader, test_loader,
                   num_batches=5):
    """Before the real run, execute one full epoch (train + val/test eval + all saves)
    on num_batches batches to validate the whole pipeline. All artifacts go to
    the ckpt_dir/_smoke/ subdirectory. During the smoke run model/optimizer/
    scheduler/rng state is temporarily modified and restored afterwards. Any failing step raises immediately and the real training is not started.
    """
    print("=" * 80)
    print(f"[Smoke Test] quick pipeline check before the real run (train/val/test with {num_batches} batches each)")
    print("=" * 80)

    tmp_root = os.path.join(ckpt_dir, "_smoke")
    prepare_output_dirs(tmp_root)
    print(f"[Smoke Test] tmp dir: {tmp_root}")

    snapshot = _snapshot_train_state(model, optimizer, scheduler)
    smoke_train = _LimitedLoader(train_loader, num_batches)
    smoke_val = _LimitedLoader(val_loader, num_batches)
    smoke_test = _LimitedLoader(test_loader, num_batches)

    eval_kw = dict(lambda_porn=config["lambda_porn"], lambda_gore=config["lambda_gore"],
                   lambda_ip=config["lambda_ip"],
                   scan_threshold=bool(config.get("eval_scan_threshold", False)),
                   num_porn_classes=config.get("num_porn_classes", 2),
                   num_gore_classes=config.get("num_gore_classes", 2))

    try:
        history = init_history()
        print("[Smoke Test] ▶ train_one_epoch")
        train_loss, train_porn, train_gore, train_ip_pos, _ = train_one_epoch(
            model=model, train_loader=smoke_train, optimizer=optimizer,
            criterion_bin=criterion_bin, criterion_ip=criterion_ip, device=device,
            epoch=1, history=history, global_iter=0,
            lambda_porn=config["lambda_porn"], lambda_gore=config["lambda_gore"],
            lambda_ip=config["lambda_ip"],
            enable_train_group_head_mask=config.get("enable_train_group_head_mask", False),
            grad_clip_max_norm=float(config.get("grad_clip_max_norm", 0.0)),
            num_porn_classes=config.get("num_porn_classes", 2),
            num_gore_classes=config.get("num_gore_classes", 2),
        )

        print("[Smoke Test] ▶ evaluate(val)")
        v_loss, v_p, v_g, v_ip, v_res, v_thr = evaluate(
            model, smoke_val, criterion_bin, criterion_ip, device, 1, "Val", verbose=False, **eval_kw,
        )

        print("[Smoke Test] ▶ evaluate(test)")
        t_loss, t_p, t_g, t_ip, t_res, t_thr = evaluate(
            model, smoke_test, criterion_bin, criterion_ip, device, 1, "Test", verbose=False, **eval_kw,
        )

        print("[Smoke Test] ▶ append_epoch_history")
        append_epoch_history(
            history, 1, train_loss, v_loss, t_loss,
            train_porn, v_p, t_p, train_gore, v_g, t_g,
            train_ip_pos, v_ip["ip_pos_metrics"], t_ip["ip_pos_metrics"],
            val_total_risk_stats=v_ip.get("total_risk_stats"),
            test_total_risk_stats=t_ip.get("total_risk_stats"),
        )

        print("[Smoke Test] ▶ save_epoch_artifacts")
        save_epoch_artifacts(tmp_root, 1, history, model,
                             val_results=v_res, test_results=t_res,
                             val_threshold_results=v_thr, test_threshold_results=t_thr,
                             val_ip_eval_extra=v_ip, test_ip_eval_extra=t_ip,
                             test_porn_metrics=t_p, test_gore_metrics=t_g)

        print("=" * 80)
        print("[Smoke Test] ✓ pipeline verified, ready for the real run")
        print(f"[Smoke Test] tmp artifacts kept at {tmp_root}")
        print("=" * 80)
    except Exception as e:
        print("=" * 80)
        print(f"[Smoke Test] ✗ failed: {type(e).__name__}: {e}")
        print(f"[Smoke Test] tmp dir kept for debugging: {tmp_root}")
        print("=" * 80)
        raise
    finally:
        _restore_train_state(snapshot, model, optimizer, scheduler)
