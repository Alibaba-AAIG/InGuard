import os
import sys
import json
import shutil
import random
import datetime
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =========================
# Logger / Print
# =========================
class Logger(object):
    def __init__(self, filename="log.txt"):
        self.terminal = sys.stdout
        self.log = None
        try:
            self.log = open(filename, "a", encoding="utf-8")
        except Exception:
            self.log = None

        msg = f"\n{'=' * 18} task started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {'=' * 18}\n"
        try:
            self.terminal.write(msg)
        except Exception:
            pass
        try:
            if self.log is not None:
                self.log.write(msg)
                self.log.flush()
        except Exception:
            self.log = None

    def write(self, message):
        try:
            self.terminal.write(message)
        except Exception:
            pass
        try:
            if self.log is not None:
                self.log.write(message)
                self.log.flush()
        except Exception:
            self.log = None

    def flush(self):
        try:
            self.terminal.flush()
        except Exception:
            pass
        try:
            if self.log is not None:
                self.log.flush()
        except Exception:
            self.log = None

    def isatty(self):
        return getattr(self.terminal, "isatty", lambda: False)()

    @property
    def encoding(self):
        return getattr(self.terminal, "encoding", "utf-8")

    def fileno(self):
        return getattr(self.terminal, "fileno", lambda: -1)()


def print_section(title, width=110, char="="):
    print(f"\n{char * width}")
    print(title)
    print(f"{char * width}")


def _stringify(x):
    if x is None:
        return ""
    if isinstance(x, float):
        return f"{x:.6f}"
    return str(x)


def print_table(headers, rows, title=None, width_limit=40):
    if title:
        print_section(title)
    headers = [_stringify(h) for h in headers]
    rows = [[_stringify(v) for v in row] for row in rows]
    all_rows = [headers] + rows if rows else [headers]

    col_num = len(headers)
    col_widths = []
    for c in range(col_num):
        w = max(len(r[c]) if c < len(r) else 0 for r in all_rows)
        col_widths.append(min(max(w, len(headers[c])), width_limit))

    def fmt_row(row):
        vals = []
        for i, v in enumerate(row):
            v = v if len(v) <= col_widths[i] else v[:col_widths[i] - 3] + "..."
            vals.append(v.ljust(col_widths[i]))
        return " | ".join(vals)

    line = "-+-".join("-" * w for w in col_widths)
    print(fmt_row(headers))
    print(line)
    for row in rows:
        print(fmt_row(row))


def print_kv_table(kvs, title=None, key_name="Key", value_name="Value", width_limit=60):
    print_table([key_name, value_name], [[k, v] for k, v in kvs], title=title, width_limit=width_limit)


# =========================
# Common Utils
# =========================
def snapshot_code_dir(ckpt_dir, include_subdirs=False, allowed_suffixes=None, verbose=True):
    if "__file__" not in globals():
        print("[Code Snapshot] __file__ not found, skip snapshot.")
        return

    if allowed_suffixes is None:
        allowed_suffixes = {".py", ".json", ".jsonl", ".yaml", ".yml", ".txt", ".sh"}

    src_dir = os.path.dirname(os.path.abspath(__file__))
    dst_dir = os.path.join(ckpt_dir, "code_snapshot")
    os.makedirs(dst_dir, exist_ok=True)

    copied_files = 0
    skipped_items = []

    for name in os.listdir(src_dir):
        src_path = os.path.join(src_dir, name)
        dst_path = os.path.join(dst_dir, name)
        try:
            if os.path.isfile(src_path):
                suffix = Path(name).suffix.lower()
                if suffix in allowed_suffixes or suffix == "":
                    shutil.copy2(src_path, dst_path)
                    copied_files += 1
                else:
                    skipped_items.append([name, f"suffix_skipped({suffix})"])
            elif os.path.isdir(src_path):
                if include_subdirs:
                    if os.path.abspath(src_path).startswith(os.path.abspath(ckpt_dir)):
                        skipped_items.append([name, "skip_ckpt_dir"])
                        continue
                    shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
                    copied_files += 1
                else:
                    skipped_items.append([name, "dir_skipped"])
            else:
                skipped_items.append([name, "unknown_type"])
        except Exception as e:
            skipped_items.append([name, f"error: {str(e)}"])

    if verbose:
        print_table(
            ["Field", "Value"],
            [["SourceDir", src_dir], ["SnapshotDir", dst_dir],
             ["IncludeSubdirs", include_subdirs],
             ["AllowedSuffixes", ",".join(sorted(allowed_suffixes))],
             ["CopiedCount", copied_files], ["SkippedCount", len(skipped_items)]],
            title="Code Snapshot Summary", width_limit=120,
        )
        if skipped_items:
            print_table(["Name", "Reason"], skipped_items, title="Code Snapshot Skipped Items", width_limit=140)


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    return torch.utils.data.dataloader.default_collate(batch)


def _to_jsonable(obj):
    if isinstance(obj, torch.Tensor):
        if obj.ndim == 0:
            return obj.item()
        return obj.detach().cpu().tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(obj), f, ensure_ascii=False, indent=2)


def moving_average(values, window=50):
    if len(values) == 0:
        return []
    arr = np.array(values, dtype=np.float32)
    window = max(1, int(window))
    out = []
    for i in range(len(arr)):
        left = max(0, i - window + 1)
        out.append(float(arr[left:i + 1].mean()))
    return out


# =========================
# Plot
# =========================
def plot_iter_loss(history, save_path):
    iters = history.get("train_iter", [])
    losses = history.get("train_iter_loss", [])
    if len(iters) == 0:
        return

    plt.figure(figsize=(10, 5))
    plt.plot(iters, losses, linewidth=1, alpha=0.35, label="iter_loss")
    plt.plot(iters, moving_average(losses, window=50), linewidth=2, label="iter_loss_ma50")
    plt.title("Train Iter Loss")
    plt.xlabel("Global Iter")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_epoch_metrics(history, save_path):
    epochs = history.get("epoch", [])
    if not epochs:
        return

    panels = [
        ("Total Loss", [("train_loss", "train_loss"), ("test_loss", "test_loss")]),
        ("Porn F1", [("train_porn_f1", "train_porn_f1"), ("test_porn_f1", "test_porn_f1")]),
        ("Gore F1", [("train_gore_f1", "train_gore_f1"), ("test_gore_f1", "test_gore_f1")]),
        ("IP(1-5 vs Other) F1", [("train_ip_pos_f1", "train_ip_pos_f1"), ("test_ip_pos_f1", "test_ip_pos_f1")]),
        ("Porn Precision", [("test_porn_precision", "test_porn_precision")]),
        ("Porn Recall", [("test_porn_recall", "test_porn_recall")]),
        ("Gore Precision / Recall", [
            ("test_gore_precision", "test_gore_precision"),
            ("test_gore_recall", "test_gore_recall"),
        ]),
        ("IP Precision", [("test_ip_pos_precision", "test_ip_pos_precision")]),
        ("IP Recall", [("test_ip_pos_recall", "test_ip_pos_recall")]),
        ("Total Risk Recall", [("test_total_risk_recall", "test_total_risk_recall")]),
        ("End-to-End Unsafety", [("test_e2e_unsafety", "test_e2e_unsafety")]),
        ("Normal Traffic Disturb Rate", [("test_normal_disturb_rate", "test_normal_disturb_rate")]),
    ]

    fig, axes = plt.subplots(4, 3, figsize=(18, 18))
    axes = axes.flatten()
    for ax, (title, curves) in zip(axes, panels):
        for key, label in curves:
            ax.plot(epochs, history.get(key, []), marker="o", label=label)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.3)
        ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def save_history_and_plots(history, save_dir, epoch):
    os.makedirs(save_dir, exist_ok=True)
    save_json(history, os.path.join(save_dir, f"history_epoch_{epoch}.json"))
    plot_iter_loss(history, os.path.join(save_dir, f"train_iter_loss_epoch_{epoch}.png"))
    plot_epoch_metrics(history, os.path.join(save_dir, f"epoch_metrics_epoch_{epoch}.png"))


# =========================
# Metrics
# =========================
def compute_binary_metrics(y_true, y_pred):
    y_true = np.array(y_true).astype(int)
    y_pred = np.array(y_pred).astype(int)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    acc = (tp + tn) / max(1, tp + tn + fp + fn)

    return {
        "acc": acc, "precision": precision, "recall": recall, "f1": f1,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "pred_pos": tp + fp, "gt_pos": tp + fn,
    }


def compute_confusion_matrix(y_true, y_pred, num_classes):
    mat = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        t = int(t); p = int(p)
        if 0 <= t < num_classes and 0 <= p < num_classes:
            mat[t, p] += 1
    return mat


def print_confusion_matrix(cm, class_names, title="Confusion Matrix"):
    rows = [[class_names[i]] + [int(v) for v in row] for i, row in enumerate(cm)]
    print_table(["GT\\Pred"] + class_names, rows, title=title, width_limit=20)


def empty_metrics():
    return {
        "acc": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0,
        "tp": 0, "fp": 0, "fn": 0, "tn": 0,
        "pred_pos": 0, "gt_pos": 0,
    }


def compute_multiclass_metrics(y_true, y_pred, num_classes, class_names=None):
    """Evaluate a 3+ class task: returns macro-avg metrics + per-class details.
    The returned dict keeps the keys of compute_binary_metrics
    (acc/precision/recall/f1/tp/fp/fn/tn) and additionally carries a
    per_class dict and a multiclass=True flag.
    """
    y_true = np.array(y_true).astype(int)
    y_pred = np.array(y_pred).astype(int)
    acc = (y_true == y_pred).mean() if len(y_true) > 0 else 0.0

    per_class = {}
    precisions, recalls, f1s = [], [], []
    total_tp = total_fp = total_fn = total_tn = 0
    for c in range(num_classes):
        tp = int(((y_true == c) & (y_pred == c)).sum())
        fp = int(((y_true != c) & (y_pred == c)).sum())
        fn = int(((y_true == c) & (y_pred != c)).sum())
        tn = int(((y_true != c) & (y_pred != c)).sum())
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        per_class[c] = {"precision": p, "recall": r, "f1": f1,
                        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                        "support": int((y_true == c).sum())}
        if class_names:
            per_class[c]["name"] = class_names[c]
        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_tn += tn

    return {
        "acc": acc,
        "precision": float(np.mean(precisions)),
        "recall": float(np.mean(recalls)),
        "f1": float(np.mean(f1s)),
        "tp": total_tp, "fp": total_fp, "fn": total_fn, "tn": total_tn,
        "pred_pos": total_tp + total_fp,
        "gt_pos": total_tp + total_fn,
        "per_class": per_class,
        "multiclass": True,
        "num_classes": num_classes,
    }


def empty_multiclass_metrics(num_classes=3):
    """Empty placeholder compatible with empty_metrics, used by 3-class
    tasks when there is no data."""
    per_class = {}
    for c in range(num_classes):
        per_class[c] = {"precision": 0.0, "recall": 0.0, "f1": 0.0,
                        "tp": 0, "fp": 0, "fn": 0, "tn": 0, "support": 0}
    return {
        "acc": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0,
        "tp": 0, "fp": 0, "fn": 0, "tn": 0,
        "pred_pos": 0, "gt_pos": 0,
        "per_class": per_class,
        "multiclass": True,
        "num_classes": num_classes,
    }


def fmt_percent_ratio(x, num, den):
    return f"{x * 100:.2f}%({num}/{den})"


def summarize_task_metrics(name, metrics):
    if metrics.get("multiclass", False):
        return {
            "Task": name,
            "Acc": f"{metrics['acc'] * 100:.2f}%",
            "Precision": f"{metrics['precision'] * 100:.2f}%",
            "Recall": f"{metrics['recall'] * 100:.2f}%",
            "F1": f"{metrics['f1'] * 100:.2f}%",
        }
    return {
        "Task": name,
        "Acc": f"{metrics['acc'] * 100:.2f}%",
        "Precision": fmt_percent_ratio(metrics["precision"], metrics["tp"], metrics["tp"] + metrics["fp"]),
        "Recall": fmt_percent_ratio(metrics["recall"], metrics["tp"], metrics["tp"] + metrics["fn"]),
        "F1": f"{metrics['f1'] * 100:.2f}%",
    }


def fmt_ratio(num, den):
    val = num / den if den > 0 else 0.0
    return f"{val:.4f}({num}/{den})"


def fmt_f1(tp, fp, fn):
    den = 2 * tp + fp + fn
    val = (2 * tp) / den if den > 0 else 0.0
    return f"{val:.4f}({2 * tp}/{den})"


def compute_multiclass_one_vs_rest_metrics(y_true, y_pred, class_id):
    y_true = np.array(y_true).astype(int)
    y_pred = np.array(y_pred).astype(int)
    true_pos_mask = (y_true == class_id)
    pred_pos_mask = (y_pred == class_id)
    tp = int((true_pos_mask & pred_pos_mask).sum())
    fp = int(((y_true != class_id) & pred_pos_mask).sum())
    fn = int((true_pos_mask & (y_pred != class_id)).sum())
    tn = int(((y_true != class_id) & (y_pred != class_id)).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    acc = (tp + tn) / max(1, tp + tn + fp + fn)
    return {"precision": precision, "recall": recall, "f1": f1, "acc": acc,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def compute_metrics_by_thresholds(y_true, y_prob, thresholds):
    rows = []
    y_true = np.array(y_true).astype(int)
    y_prob = np.array(y_prob).astype(float)
    for th in thresholds:
        y_pred = (y_prob >= th).astype(int)
        m = compute_binary_metrics(y_true, y_pred)
        rows.append({
            "threshold": float(th), "precision": float(m["precision"]),
            "recall": float(m["recall"]), "f1": float(m["f1"]), "acc": float(m["acc"]),
            "tp": int(m["tp"]), "fp": int(m["fp"]), "fn": int(m["fn"]), "tn": int(m["tn"]),
            "pred_pos": int(m["pred_pos"]), "gt_pos": int(m["gt_pos"]),
        })
    return rows


def predict_ip_with_shared_threshold(ip_probs, threshold):
    ip_probs = np.asarray(ip_probs, dtype=np.float32)
    risk_probs = ip_probs[:, :5]
    risk_max = risk_probs.max(axis=1)
    risk_argmax = risk_probs.argmax(axis=1)
    pred = np.where(risk_max >= float(threshold), risk_argmax, 5)
    return pred.astype(int).tolist()


def compute_ip_metrics_by_shared_thresholds(ip_true, ip_probs, thresholds):
    rows = []
    ip_true = np.array(ip_true).astype(int)
    ip_probs = np.array(ip_probs).astype(np.float32)
    ip_true_bin = np.isin(ip_true, [0, 1, 2, 3, 4]).astype(int)
    for th in thresholds:
        ip_pred = np.array(predict_ip_with_shared_threshold(ip_probs, th)).astype(int)
        ip_pred_bin = np.isin(ip_pred, [0, 1, 2, 3, 4]).astype(int)
        m = compute_binary_metrics(ip_true_bin, ip_pred_bin)
        rows.append({
            "threshold": float(th), "precision": float(m["precision"]),
            "recall": float(m["recall"]), "f1": float(m["f1"]), "acc": float(m["acc"]),
            "tp": int(m["tp"]), "fp": int(m["fp"]), "fn": int(m["fn"]), "tn": int(m["tn"]),
            "pred_pos": int(m["pred_pos"]), "gt_pos": int(m["gt_pos"]),
        })
    return rows


def print_threshold_table_compact(task_name, rows, split="Test"):
    if not rows:
        print(f"[{split}] {task_name} threshold metrics: empty")
        return None
    best = None
    table_rows = []
    for r in rows:
        tp, fp, fn, tn = r["tp"], r["fp"], r["fn"], r["tn"]
        table_rows.append([f"{r['threshold']:.2f}", fmt_ratio(tp, tp + fn),
                           fmt_ratio(tp, tp + fp), fmt_f1(tp, fp, fn),
                           fmt_ratio(tp + tn, tp + tn + fp + fn)])
        if best is None or r["f1"] > best["f1"]:
            best = r
    print_table(["TH", "Recall", "Precision", "F1", "Acc"], table_rows,
                title=f"[{split}] {task_name} Threshold-wise Metrics", width_limit=24)
    if best is not None:
        print_table(
            ["Task", "BestTH", "Recall", "Precision", "F1", "Acc"],
            [[task_name, f"{best['threshold']:.2f}",
              fmt_ratio(best["tp"], best["tp"] + best["fn"]),
              fmt_ratio(best["tp"], best["tp"] + best["fp"]),
              fmt_f1(best["tp"], best["fp"], best["fn"]),
              fmt_ratio(best["tp"] + best["tn"], best["tp"] + best["tn"] + best["fp"] + best["fn"])]],
            title=f"[{split}] {task_name} Best Threshold", width_limit=24,
        )
    return best


def print_ip_multiclass_metrics(ip_true, ip_pred, class_names, split="Test"):
    if len(ip_true) == 0:
        print(f"[{split}] IP Multi-class metrics empty")
        return
    ip_true_np = np.array(ip_true).astype(int)
    rows = []
    for cls_id in range(len(class_names)):
        m = compute_multiclass_one_vs_rest_metrics(ip_true, ip_pred, cls_id)
        tp, fp, fn = m["tp"], m["fp"], m["fn"]
        rows.append([class_names[cls_id], fmt_ratio(tp, tp + fn), fmt_ratio(tp, tp + fp),
                     fmt_f1(tp, fp, fn), int((ip_true_np == cls_id).sum())])
    print_table(["Class", "Recall", "Precision", "F1", "Support"], rows,
                title=f"[{split}] IP Multi-class Per-Class Metrics", width_limit=24)


def pick_best_threshold_row(rows):
    if not rows:
        return None
    best = rows[0]
    for r in rows[1:]:
        if r["f1"] > best["f1"]:
            best = r
    return best


def compute_metrics_at_threshold(y_true, y_prob, threshold):
    y_true = np.array(y_true).astype(int)
    y_prob = np.array(y_prob).astype(float)
    y_pred = (y_prob >= float(threshold)).astype(int)
    m = compute_binary_metrics(y_true, y_pred)
    return m, y_pred.tolist()


def compute_total_risk_stats(porn_true, porn_prob, porn_th,
                              gore_true, gore_prob, gore_th,
                              ip_true_full, ip_probs_full, ip_th):
    porn_true = np.array(porn_true).astype(int)
    porn_prob = np.array(porn_prob).astype(float)
    gore_true = np.array(gore_true).astype(int)
    gore_prob = np.array(gore_prob).astype(float)
    ip_true_full = np.array(ip_true_full).astype(int)
    ip_probs_full = np.array(ip_probs_full).astype(np.float32)

    pred_porn = (porn_prob >= float(porn_th)).astype(int)
    pred_gore = (gore_prob >= float(gore_th)).astype(int)
    pred_ip_cls = np.array(predict_ip_with_shared_threshold(ip_probs_full, ip_th)).astype(int)
    pred_ip_risk = np.isin(pred_ip_cls, [0, 1, 2, 3, 4]).astype(int)

    gt_ip_risk = np.isin(ip_true_full, [0, 1, 2, 3, 4]).astype(int)
    gt_total_risk = ((porn_true == 1) | (gore_true == 1) | (gt_ip_risk == 1)).astype(int)
    pred_total_risk = ((pred_porn == 1) | (pred_gore == 1) | (pred_ip_risk == 1)).astype(int)

    tp = int(((gt_total_risk == 1) & (pred_total_risk == 1)).sum())
    fp = int(((gt_total_risk == 0) & (pred_total_risk == 1)).sum())
    fn = int(((gt_total_risk == 1) & (pred_total_risk == 0)).sum())
    tn = int(((gt_total_risk == 0) & (pred_total_risk == 0)).sum())

    total_risk = int((gt_total_risk == 1).sum())
    total_normal = int((gt_total_risk == 0).sum())
    total_samples = int(len(gt_total_risk))

    return {
        "total_samples": total_samples, "total_risk_samples": total_risk,
        "total_normal_samples": total_normal, "recalled_risk_samples": tp,
        "missed_risk_samples": fn, "disturbed_normal_samples": fp,
        "safe_normal_samples": tn,
        "total_risk_recall": tp / total_risk if total_risk > 0 else 0.0,
        "e2e_unsafety": fn / total_samples if total_samples > 0 else 0.0,
        "normal_disturb_rate": fp / total_normal if total_normal > 0 else 0.0,
        "pred_total_risk_count": int((pred_total_risk == 1).sum()),
        "gt_total_risk_flags": gt_total_risk.tolist(),
        "pred_total_risk_flags": pred_total_risk.tolist(),
    }


def print_total_risk_stats_table(stats, split="Test", title_suffix="Best-Threshold Aggregated Risk"):
    rows = [
        ["total_samples", stats["total_samples"]],
        ["total_risk_samples", stats["total_risk_samples"]],
        ["recalled_risk_samples", stats["recalled_risk_samples"]],
        ["missed_risk_samples", stats["missed_risk_samples"]],
        ["total_normal_samples", stats["total_normal_samples"]],
        ["disturbed_normal_samples", stats["disturbed_normal_samples"]],
        ["pred_total_risk_count", stats["pred_total_risk_count"]],
        ["total_risk_recall", f"{stats['total_risk_recall']:.4f}({stats['recalled_risk_samples']}/{max(1, stats['total_risk_samples'])})"],
        ["e2e_unsafety", f"{stats['e2e_unsafety']:.4f}({stats['missed_risk_samples']}/{max(1, stats['total_samples'])})"],
        ["normal_disturb_rate", f"{stats['normal_disturb_rate']:.4f}({stats['disturbed_normal_samples']}/{max(1, stats['total_normal_samples'])})"],
    ]
    print_table(["Metric", "Value"], rows, title=f"[{split}] {title_suffix}", width_limit=40)


# =========================
# File Helper
# =========================
def safe_copy(src, dst):
    try:
        if os.path.exists(src):
            shutil.copy2(src, dst)
            return True
        return False
    except Exception:
        return False
