import os
import re
import csv
import sys
import io
import json
import random
import shutil
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
from tqdm import tqdm

from torchvision.transforms import Normalize

from utils_common import save_json, print_table, print_kv_table

csv.field_size_limit(sys.maxsize)


# =========================
# Label / Group Constants
# =========================
# Internal IP ids (0-4 = the five named controlled IPs, 5 = other)
# Chinese by design: canonical IP concept names shared with the labeling prompts / detector IP head
IP_ID2NAME = {0: "白雪公主", 1: "哆啦A梦", 2: "小黄人", 3: "艾莎", 4: "海绵宝宝", 5: "其它"}
CONTROLLED_IP_IDS = {0, 1, 2, 3, 4}

# CSV ip_copyright raw value -> internal ip_label
# CSV 0=none, 1..5 = the five named controlled IPs, 6=related_ip;
# both 0 and 6 map to "other" (internal id 5)
CSV_IP_TO_INTERNAL = {0: 5, 1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5}

# train_group names (order = bucket priority)
TRAIN_GROUP_KEYS = [
    "porn", "gore", "ip_controlled",
    "porn_borderline", "gore_borderline", "ip_borderline",
    "normal_other",
]


# =========================
# Helpers
# =========================
def adapt_video_to_latent_root(video_path):
    """`.../video/xxx.mp4` -> `.../latents_x1`."""
    p = Path(str(video_path).strip().strip('"').strip("'"))
    parent = p.parent
    return str(parent.parent / "latents_x1") if parent.name == "video" else str(parent / "latents_x1")


def adapt_video_to_prompt_embeds_path(video_path):
    """`.../video/xxx.mp4` or `.../image/xxx.jpg` -> `.../prompt_embeds_forward/xxx.pth`"""
    p = Path(str(video_path).strip().strip('"').strip("'"))
    parent = p.parent
    base = parent.parent if parent.name in ("video", "image") else parent
    return str(base / "prompt_embeds_forward" / f"{p.stem}.pth")


def adapt_to_prompt_embeds_mask_path(prompt_embeds_path):
    """`.../prompt_embeds_forward/xxx.pth` → `.../prompt_embeds_mask_forward/xxx.pth`"""
    if not prompt_embeds_path:
        return None
    return prompt_embeds_path.replace("prompt_embeds_forward", "prompt_embeds_mask_forward")


def adapt_image_to_latent_root(image_path):
    """`.../image/xxx.jpg` -> `.../latents_x1`. Mirror of adapt_video_to_latent_root,
    for image-generation latents (parent dir is 'image' instead of 'video')."""
    p = Path(str(image_path).strip().strip('"').strip("'"))
    parent = p.parent
    return str(parent.parent / "latents_x1") if parent.name == "image" else str(parent / "latents_x1")


def decide_train_group(rp, rv, ri, filename="", category=None):
    """
    Bucketing rules (priority top to bottom):
      1. porn           : CSV pornographic == 2
      2. gore           : CSV violence_gore == 2
      3. ip_controlled  : CSV ip_copyright in {1..5}
      4. porn_borderline: category/filename contains 'porn' (case-insensitive) — treated as borderline risk
      5. gore_borderline: category/filename contains 'gore'
      6. ip_borderline  : category/filename contains the whole token 'ip'
      7. normal_other   : none of the above matched

    Note: the first three rules are label-based (confirmed risk); the last three
    are category-keyword based, because those prompts are borderline by nature.
    The borderline signal is read from `category` (the prompt's intended risk
    category, e.g. from testset.csv/trainset.csv) when provided; otherwise it
    falls back to the filename stem, for backward compatibility with datasets
    whose id embeds the category as a prefix.
    """
    if rp == 2:
        return "porn"
    if rv == 2:
        return "gore"
    if ri in (1, 2, 3, 4, 5):
        return "ip_controlled"
    # borderline detection: prefer the explicit prompt category; fall back to the
    # filename stem (extension stripped, lowercased). "porn" / "gore" are long
    # enough for substring matching; "ip" is only two letters, so match whole
    # tokens to avoid false hits on "script" / "recipe" / "tip".
    src_lower = str(category).lower() if category else Path(str(filename)).stem.lower()
    src_tokens = set(re.split(r'[^a-z]+', src_lower))  # split into alphabetic tokens
    if "porn" in src_lower:
        return "porn_borderline"
    if "gore" in src_lower:
        return "gore_borderline"
    if "ip" in src_tokens:
        return "ip_borderline"
    return "normal_other"


def load_id_to_category(csv_path):
    """Load an {id: category} map from a prompt CSV for borderline bucketing.

    Uses the 'category' column when present, else 'control_category'. Returns
    None when the file is missing or lacks an id/category column (callers then
    fall back to the filename-keyword heuristic in decide_train_group).
    """
    if not csv_path or not os.path.isfile(csv_path):
        return None
    mapping = {}
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        cat_col = "category" if "category" in fields else (
            "control_category" if "control_category" in fields else None)
        if "id" not in fields or cat_col is None:
            return None
        for row in reader:
            rid = (row.get("id") or "").strip()
            if rid:
                mapping[rid] = (row.get(cat_col) or "").strip()
    return mapping


def get_three_class(porn_label, gore_label, num_porn_classes=2, num_gore_classes=2):
    """Three-class bucketing from the porn/gore labels.
    binary (default): label==1 counts as risk -> goes to the risk bucket
    3class: label>=1 (borderline+risk) -> same bucket, keeping borderline samples
    spread evenly across the train/val stratified split
    """
    _porn_thresh = 1 if num_porn_classes == 3 else 1
    _gore_thresh = 1 if num_gore_classes == 3 else 1
    if int(porn_label) >= _porn_thresh:
        return "porn"
    if int(gore_label) >= _gore_thresh:
        return "gore"
    return "normal"


# =========================
# CSV Loader
# =========================
def load_predictions_csv(csv_path, source_name,
                          input_mode="image", file_type="image", verbose=False,
                          label_mode="binary", id_to_category=None):
    """Read a predictions CSV, map labels + bucket, and return a list of item dicts.

    Args:
        csv_path:    path to the predictions CSV.
        source_name: data-source name (for logging and classification).
        input_mode:  model input mode, e.g. 'latent_frames', 'latent_to_image'.
        file_type:   filter on the CSV file_type column, 'image' or 'video'
                     (default 'image').
        verbose:     force printing statistics.
        label_mode:  'binary' (default) -> binarize to 0/1 (2=risk, 0/1=safe);
                     '3class' -> keep the raw 0/1/2 labels (0=safe, 1=borderline, 2=risk)
        id_to_category: optional {id: category} map (id = filename stem). When
                     given, the borderline train_group is decided from the
                     prompt's intended category instead of the filename keyword,
                     making it robust to opaque / prefix-less ids.
    """
    _num_porn = 3 if label_mode == "3class" else 2
    _num_gore = 3 if label_mode == "3class" else 2
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    # Read the whole file into memory before decoding: a streaming csv.DictReader
    # can crash on non-UTF-8 bytes inside quoted fields when a buffer boundary
    # splits them (seen with real predictions.csv files).
    with open(csv_path, "rb") as f:
        text = f.read().decode("utf-8-sig", errors="replace")
    # NUL bytes (\x00) make csv.DictReader raise "_csv.Error: line contains NUL";
    # strip them (usually an encoding artifact or stray content in raw_response).
    text = text.replace("\x00", "")
    rows = list(csv.DictReader(io.StringIO(text)))

    items = []
    skipped = {"not_success": 0, f"not_{file_type}": 0, "missing_field": 0, "bad_label": 0}

    for row in tqdm(rows, desc=f"Parsing {source_name}"):
        if str(row.get("status", "")).strip() != "success":
            skipped["not_success"] += 1
            continue
        if str(row.get("file_type", "")).strip() != file_type:
            skipped[f"not_{file_type}"] += 1
            continue

        file_path = str(row.get("file_path", "")).strip()
        filename = str(row.get("filename", "")).strip()
        if not file_path or not filename:
            skipped["missing_field"] += 1
            continue

        try:
            rp = int(row["pornographic"])
            rv = int(row["violence_gore"])
            ri = int(row["ip_copyright"])
        except Exception:
            skipped["bad_label"] += 1
            continue

        if label_mode == "3class":
            porn_label = rp   # keep 0/1/2
            gore_label = rv   # keep 0/1/2
        else:
            porn_label = 1 if rp == 2 else 0
            gore_label = 1 if rv == 2 else 0
        ip_label = CSV_IP_TO_INTERNAL.get(ri, 5)

        item = {
            # identity / paths
            "name": Path(filename).stem,
            "file_path": file_path,
            "video_path": file_path,
            "filename": filename,
            "source_dir": str(row.get("source_dir", "")),
            "source_name": source_name,

            # model labels (mapped; binary collapses borderline to 0, 3class keeps 0/1/2)
            "porn_label": porn_label,
            "gore_label": gore_label,
            "ip_label": ip_label,
            "has_ip_label": True,

            # raw CSV labels (for badcases / analysis)
            "raw_porn": rp,
            "raw_gore": rv,
            "raw_ip": ri,
            "raw_porn_name": str(row.get("pornographic_name", "")),
            "raw_gore_name": str(row.get("violence_gore_name", "")),
            "raw_ip_name": str(row.get("ip_copyright_name", "")),

            # derived / bucketing
            "three_class": get_three_class(porn_label, gore_label,
                                              num_porn_classes=_num_porn,
                                              num_gore_classes=_num_gore),
            "train_group": decide_train_group(
                rp, rv, ri, filename=filename,
                category=(id_to_category.get(Path(filename).stem) if id_to_category else None)),
            "ip_group": "ip_controlled" if ip_label in CONTROLLED_IP_IDS else "ip_other",
            "sampling_group": Path(str(row.get("source_dir", ""))).name or "unknown",

            # badcase filename (summarizes the three raw labels at a glance)
            "predict": f"p{rp}v{rv}i{ri}",

            # meaningful for the test split only
            "manually_modified": str(row.get("manually_modified", "")).strip().lower() == "true",
        }
        if input_mode in ("latent_frames", "latent_to_video", "latent_to_feat", "latent",
                         "latent_to_image", "latent_to_image_feat"):
            adapt_latent = adapt_image_to_latent_root if file_type == "image" else adapt_video_to_latent_root
            item["latent_root"] = adapt_latent(file_path)

        item["prompt_embeds_path"] = adapt_video_to_prompt_embeds_path(file_path)
        item["prompt_embeds_mask_path"] = adapt_to_prompt_embeds_mask_path(item["prompt_embeds_path"])

        items.append(item)

    if verbose or sum(skipped.values()) > 0:
        print_kv_table(
            [("source", source_name), ("csv_path", csv_path),
             ("loaded", len(items)),
             ("skipped_not_success", skipped["not_success"]),
             (f"skipped_not_{file_type}", skipped[f"not_{file_type}"]),
             ("skipped_missing_field", skipped["missing_field"]),
             ("skipped_bad_label", skipped["bad_label"])],
            title=f"Load CSV / {source_name}", width_limit=120,
        )
    return items


# =========================
# Dedup / Split
# =========================
def remove_train_overlap_with_test(train_items, test_items, verbose=True):
    """Drop training-pool samples whose file_path also appears in the test set
    (leak prevention). Matching is by file_path only: filenames can repeat across
    pools (e.g. same prompt, different seed), so a filename fallback would drop
    many unrelated samples.
    """
    test_paths = set(x["file_path"] for x in test_items)

    kept, removed_by_path = [], 0
    for x in train_items:
        if x["file_path"] in test_paths:
            removed_by_path += 1
            continue
        kept.append(x)

    if verbose:
        print_kv_table(
            [("train_before", len(train_items)),
             ("removed_by_file_path", removed_by_path),
             ("train_after", len(kept))],
            title="Remove Train/Test Overlap",
        )
    return kept


# ===========================================================================
# Prompt-level testset leak check
# ===========================================================================

def _load_prompt_csv(csv_path):
    """Load a prompt CSV (header: id, prompt) and return an {id: prompt} dict."""
    if not csv_path or not os.path.isfile(csv_path):
        return None
    mapping = {}
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        if "id" not in fieldnames or "prompt" not in fieldnames:
            print(f"  [ERROR] prompt CSV missing id/prompt columns: {csv_path}")
            print(f"          actual columns: {fieldnames}")
            return None
        for row in reader:
            rid = (row.get("id") or "").strip()
            prompt = (row.get("prompt") or "").strip()
            if rid and prompt:
                mapping[rid] = prompt
    return mapping


def _normalize_prompt(text):
    """Normalize a prompt for exact comparison: strip + lower + collapse whitespace."""
    return re.sub(r'\s+', ' ', text.strip().lower())


def _char_ngrams(text, n=3):
    """Build the character-level n-gram set."""
    text = _normalize_prompt(text)
    if len(text) < n:
        return {text}
    return {text[i:i+n] for i in range(len(text) - n + 1)}


def check_prompt_testset_leak(train_all, config, ckpt_dir=None, verbose=True):
    """Check test-set leakage by prompt text; remove matching/similar samples
    from the training set.

    Flow:
    1. load train_prompt_csv and test_prompt_csv
    2. build the id -> prompt mapping
    3. check by leak_check_mode:
       - "exact": identical prompts -> flagged as leaked
       - "minhash": Jaccard similarity >= threshold -> flagged as leaked
    4. drop the matched ids from train_all

    Returns: the filtered train_all
    """
    train_prompt_csv = (config.get("train_prompt_csv") or "").strip()
    test_prompt_csv = (config.get("test_prompt_csv") or "").strip()
    mode = config.get("leak_check_mode", "exact")

    # validate paths
    if not train_prompt_csv or not test_prompt_csv:
        if verbose:
            print("  [WARN] check_testset_leak=True but train_prompt_csv or test_prompt_csv is empty; skipping the leak check")
        return train_all

    # load the prompt CSVs
    train_prompts = _load_prompt_csv(train_prompt_csv)
    test_prompts = _load_prompt_csv(test_prompt_csv)
    if train_prompts is None:
        print(f"  [WARN] cannot load the train prompt CSV: {train_prompt_csv}; skipping the leak check")
        return train_all
    if test_prompts is None:
        print(f"  [WARN] cannot load the test prompt CSV: {test_prompt_csv}; skipping the leak check")
        return train_all

    if verbose:
        print(f"  [Leak Check] mode={mode}, train_prompts={len(train_prompts)}, test_prompts={len(test_prompts)}")

    # build the normalized test-prompt set / index
    leaked_train_ids = set()
    leak_details = []  # [{train_id, test_id, train_prompt, test_prompt, score}]

    if mode == "exact":
        # ---- exact matching ----
        # test set: normalized_prompt -> list of test_ids
        test_norm_to_ids = defaultdict(list)
        for tid, tprompt in test_prompts.items():
            test_norm_to_ids[_normalize_prompt(tprompt)].append(tid)

        for train_id, train_prompt in train_prompts.items():
            norm = _normalize_prompt(train_prompt)
            if norm in test_norm_to_ids:
                leaked_train_ids.add(train_id)
                leak_details.append({
                    "train_id": train_id,
                    "test_id": test_norm_to_ids[norm][0],
                    "train_prompt": train_prompt[:200],
                    "test_prompt": test_prompts[test_norm_to_ids[norm][0]][:200],
                    "score": 1.0,
                    "mode": "exact",
                })

    elif mode == "minhash":
        # ---- MinHash approximate matching ----
        threshold = float(config.get("leak_minhash_threshold", 0.8))
        num_perm = int(config.get("leak_minhash_num_perm", 128))
        ngram_n = int(config.get("leak_minhash_ngram", 3))

        try:
            from datasketch import MinHash, MinHashLSH
        except ImportError:
            print("  [WARN] datasketch not installed; falling back to brute-force Jaccard")
            # fallback: brute-force Jaccard
            test_ngrams_list = [(tid, _char_ngrams(tp, ngram_n)) for tid, tp in test_prompts.items()]
            for train_id, train_prompt in train_prompts.items():
                train_ng = _char_ngrams(train_prompt, ngram_n)
                if not train_ng:
                    continue
                for test_id, test_ng in test_ngrams_list:
                    if not test_ng:
                        continue
                    jaccard = len(train_ng & test_ng) / len(train_ng | test_ng)
                    if jaccard >= threshold:
                        leaked_train_ids.add(train_id)
                        leak_details.append({
                            "train_id": train_id,
                            "test_id": test_id,
                            "train_prompt": train_prompt[:200],
                            "test_prompt": test_prompts[test_id][:200],
                            "score": round(jaccard, 4),
                            "mode": "minhash_bruteforce",
                        })
                        break  # one matching test prompt per train prompt is enough
            # skip the datasketch branch
            threshold = None

        if threshold is not None:
            # datasketch available
            lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
            test_minhashes = {}
            for test_id, test_prompt in test_prompts.items():
                m = MinHash(num_perm=num_perm)
                for ng in _char_ngrams(test_prompt, ngram_n):
                    m.update(ng.encode('utf-8'))
                lsh.insert(test_id, m)
                test_minhashes[test_id] = m

            for train_id, train_prompt in train_prompts.items():
                m = MinHash(num_perm=num_perm)
                for ng in _char_ngrams(train_prompt, ngram_n):
                    m.update(ng.encode('utf-8'))
                candidates = lsh.query(m)
                if candidates:
                    # take the most similar candidate
                    best_tid = None
                    best_score = 0.0
                    for cand_id in candidates:
                        score = m.jaccard(test_minhashes[cand_id])
                        if score > best_score:
                            best_score = score
                            best_tid = cand_id
                    if best_score >= threshold:
                        leaked_train_ids.add(train_id)
                        leak_details.append({
                            "train_id": train_id,
                            "test_id": best_tid,
                            "train_prompt": train_prompt[:200],
                            "test_prompt": test_prompts[best_tid][:200],
                            "score": round(best_score, 4),
                            "mode": "minhash",
                        })
    else:
        print(f"  [ERROR] unsupported leak_check_mode={mode!r}; skipping the leak check")
        return train_all

    # drop leaked ids from train_all
    before_count = len(train_all)
    if leaked_train_ids:
        train_all = [x for x in train_all if x.get("name", "") not in leaked_train_ids]

    removed_count = before_count - len(train_all)

    if verbose:
        print_kv_table(
            [("mode", mode),
             ("train_prompt_count", len(train_prompts)),
             ("test_prompt_count", len(test_prompts)),
             ("leaked_prompt_pairs", len(leak_details)),
             ("train_before", before_count),
             ("removed_by_prompt_leak", removed_count),
             ("train_after", len(train_all))],
            title="Prompt Testset Leak Check",
        )
        if leak_details:
            print(f"  first 5 leaked samples:")
            for item in leak_details[:5]:
                print(f"    train_id={item['train_id']} | test_id={item['test_id']} | score={item['score']} | prompt={item['train_prompt'][:80]}...")

    # save the leak report
    if ckpt_dir and leak_details:
        report_path = os.path.join(ckpt_dir, "meta", "prompt_leak_report.json")
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        save_json({
            "mode": mode,
            "threshold": config.get("leak_minhash_threshold") if mode == "minhash" else None,
            "total_leaked": len(leak_details),
            "removed_from_train": removed_count,
            "details": leak_details,
        }, report_path)
        if verbose:
            print(f"  leak report saved: {report_path}")

    return train_all


def split_val_fixed_per_class(items, val_per_class, seed):
    """Split a fixed number per class into val, the rest into train. Buckets are
    disjoint and cover the full set:
      - porn           : three_class == "porn"               (porn_label == 1)
      - gore           : three_class == "gore"               (gore_label == 1)
      - ip_controlled  : three_class == "normal" and ip_label in {0..4}  (true IP positives)
      - normal         : three_class == "normal" and ip_label == 5       (clean / borderline etc.)
    ip_controlled is stratified separately so the val split always contains true IP
    positives (otherwise they are swallowed by the normal bucket and the IP-head
    val metrics become unreliable).
    """
    by_cls = {
        "porn":          [x for x in items if x["three_class"] == "porn"],
        "gore":          [x for x in items if x["three_class"] == "gore"],
        "ip_controlled": [x for x in items if x["three_class"] == "normal"
                          and int(x.get("ip_label", 5)) in (0, 1, 2, 3, 4)],
        "normal":        [x for x in items if x["three_class"] == "normal"
                          and int(x.get("ip_label", 5)) == 5],
    }
    rng = random.Random(seed)
    for lst in by_cls.values():
        rng.shuffle(lst)

    val_list, train_list = [], []
    for c, lst in by_cls.items():
        n = min(val_per_class, len(lst))
        if n < val_per_class:
            print(f"[Warning] val {c} insufficient: need={val_per_class}, got={len(lst)}")
        val_list.extend(lst[:n])
        train_list.extend(lst[n:])
    rng.shuffle(train_list)
    rng.shuffle(val_list)
    return train_list, val_list


# =========================
# Summary
# =========================
def print_final_dataset_summary(train_list, test_list):
    """Print the three core tables at startup:
       1) train set - counts per data pool
       2) test set - counts per data pool
       3) IP-class breakdown - train/test side by side
    """
    train_grp = Counter(x["train_group"] for x in train_list)
    test_grp = Counter(x["train_group"] for x in test_list)
    train_ip = Counter(int(x.get("ip_label", 5)) for x in train_list)
    test_ip = Counter(int(x.get("ip_label", 5)) for x in test_list)

    rows = [[g, train_grp.get(g, 0)] for g in TRAIN_GROUP_KEYS]
    rows.append(["TOTAL", sum(train_grp.values())])
    print_table(["train_group", "count"], rows,
                title="1) train set - counts per data pool", width_limit=30)

    rows = [[g, test_grp.get(g, 0)] for g in TRAIN_GROUP_KEYS]
    rows.append(["TOTAL", sum(test_grp.values())])
    print_table(["train_group", "count"], rows,
                title="2) test set - counts per data pool", width_limit=30)

    rows = [[IP_ID2NAME[i], train_ip.get(i, 0), test_ip.get(i, 0)] for i in range(6)]
    rows.append(["TOTAL", sum(train_ip.values()), sum(test_ip.values())])
    print_table(["IP class", "train", "test"], rows,
                title="3) IP-class breakdown", width_limit=30)


# =========================
# Frame Sampling (shared by Dataset / Latent Viz)
# =========================
def uniform_frame_indices(total, k):
    """Uniformly pick k frame indices along the T=total axis; return all indices when k is invalid or >= total."""
    if k is None or k <= 0 or k >= total:
        return list(range(total))
    if k == 1:
        return [total // 2]
    idx = sorted(set(np.round(np.linspace(0, total - 1, k)).astype(int).tolist()))
    if len(idx) == k:
        return idx
    bins = np.array_split(np.arange(total), k)
    return [int(b[len(b) // 2]) for b in bins if len(b) > 0]


# =========================
# Previews
# =========================
def _save_image_latent_channel_grid(x, save_path):
    """Image latent [C, H, W] -> one viridis heatmap per channel + a grid.
    Column count adapts: 4 columns when C<=16 (16ch -> 4x4); 8 columns when C>16 (32ch -> 4x8).
    Aligned with pretrain_image.py._visualize_latent_grid.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    C = int(x.shape[0])
    ncols = 4 if C <= 16 else 8
    nrows = (C + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.2, nrows * 2.2))
    axes_flat = np.array(axes).reshape(-1)

    for ch_idx in range(C):
        ax = axes_flat[ch_idx]
        ax.imshow(x[ch_idx].numpy(), cmap="viridis", aspect="auto")
        ax.set_title(f"Ch {ch_idx}", fontsize=8)
        ax.axis("off")
    for i in range(C, len(axes_flat)):
        axes_flat[i].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return True


def save_latent_visualization(item, save_dir, dst_filename, step, latent_sample_num_frames):
    """Load a .pth -> visualize the latent -> save a PNG.
    - image latent ([1,C,H,W]): one viridis heatmap per channel + 4xN grid (same as pretrain_image.py)
    - video latent ([1,C,T,H,W]): sample frames -> average over the C dim -> horizontal grayscale strip
    Returns False silently on failure.
    """
    latent_root = item.get("latent_root")
    name = item.get("name")
    if not latent_root or not name:
        return False
    pth = os.path.join(latent_root, str(int(step)), f"{name}.pth")
    if not os.path.exists(pth):
        return False
    try:
        x = torch.load(pth, map_location="cpu").float()
        # drop the batch dim: [1,C,T,H,W] -> [C,T,H,W], [1,C,H,W] -> [C,H,W]
        if x.ndim == 5 and x.shape[0] == 1:
            x = x.squeeze(0)
        if x.ndim == 4 and x.shape[0] == 1:
            x = x.squeeze(0)

        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, dst_filename)

        # ---- image latent: [C, H, W] -> per-channel grid heatmaps ----
        if x.ndim == 3 and x.shape[0] > 0:
            if not getattr(save_latent_visualization, "_shape_printed", False):
                print(f"[Latent Viz][image] first sample shape: {tuple(x.shape)} "
                      f"→ per-channel viridis grid (C={x.shape[0]})")
                save_latent_visualization._shape_printed = True
            return _save_image_latent_channel_grid(x, save_path)

        # ---- video latent: [C, T, H, W] -> original path ----
        if x.ndim != 4 or x.shape[1] <= 0:
            return False
        idx = uniform_frame_indices(x.shape[1], latent_sample_num_frames)
        gray = x[:, idx, :, :].mean(dim=0)  # [n, H, W], averaged over dim=0 (C)

        if not getattr(save_latent_visualization, "_shape_printed", False):
            print(f"[Latent Viz][video] first sample shape after squeeze: {tuple(x.shape)} "
                  f"→ mean over dim=0 (C={x.shape[0]}) "
                  f"→ {len(idx)} frames of ({x.shape[2]}, {x.shape[3]})")
            save_latent_visualization._shape_printed = True

        gmin, gmax = float(gray.min()), float(gray.max())
        if gmax > gmin:
            gray_u8 = ((gray - gmin) / (gmax - gmin) * 255).clamp(0, 255).numpy().astype(np.uint8)
        else:
            gray_u8 = np.zeros(gray.shape, dtype=np.uint8)

        n, h, w = gray_u8.shape
        sep = 2  # 2px white separator between frames
        combined = np.full((h, n * w + max(0, n - 1) * sep), 255, dtype=np.uint8)
        for i in range(n):
            combined[:, i * (w + sep): i * (w + sep) + w] = gray_u8[i]

        from PIL import Image  # lazy import so the module still loads without PIL
        Image.fromarray(combined, mode="L").save(save_path)
        return True
    except Exception:
        return False


def _decode_latent_to_rgb_and_save(item, save_dir, dst_filename, step, model_name, vae_decoder):
    """Load a latent .pth -> VAE decode -> save the decoded RGB image (PNG).
    This is the real RGB image fed into the detector backbone (saved after the [-1,1]->[0,1] mapping).
    Returns False silently on failure.
    """
    latent_root = item.get("latent_root")
    name = item.get("name")
    if not latent_root or not name:
        return False
    pth = os.path.join(latent_root, str(int(step)), f"{name}.pth")
    if not os.path.exists(pth):
        return False
    try:
        x = _load_image_latent_by_model(pth, model_name)  # [C, H, W]
        x_batch = x.unsqueeze(0)  # [1, C, H, W]
        device = next(iter(vae_decoder.vae.parameters())).device
        x_batch = x_batch.to(device)
        with torch.no_grad():
            rgb = vae_decoder(x_batch)  # [1, 3, H_img, W_img] ∈ [-1, 1]
        rgb = ((rgb[0].float().cpu() + 1.0) / 2.0).clamp_(0.0, 1.0)  # [3, H, W] ∈ [0, 1]
        rgb_np = (rgb.permute(1, 2, 0).numpy() * 255).astype(np.uint8)  # [H, W, 3]
        os.makedirs(save_dir, exist_ok=True)
        from PIL import Image
        Image.fromarray(rgb_np, mode="RGB").save(os.path.join(save_dir, dst_filename))
        return True
    except Exception as e:
        print(f"[Preview] decode RGB failed: {e}")
        return False


def generate_previews(config, train_list, test_list, save_dir):
    """Save one preview group each for train/test (val is not saved; little value). Each bucket picks N random items: copy the file;
    latent_frames or latent mode additionally appends a grayscale latent visualization PNG in the same directory.
    latent_to_image mode additionally saves the VAE-decoded RGB image (the real image fed into the detector).
    Directory layout:
      previews/{train,test}/
        porn/  porn_borderline/  gore/  gore_borderline/  normal_other/      <- by train_group
        ip/{白雪公主, 哆啦A梦, 小黄人, 艾莎, 海绵宝宝}                          <- by ip_label 0-4
        ip/ip_related/                                                       <- train_group=='ip_borderline'
    """
    preview_dir = os.path.join(save_dir, "previews")
    os.makedirs(preview_dir, exist_ok=True)
    n_per = int(config.get("preview_examples_num", 8))
    input_mode = config.get("input_mode", "")

    # grayscale latent visualization: supported by latent_frames (video) and latent / latent_to_image / latent_to_image_feat (image) modes
    save_latent = input_mode in ("latent_frames", "latent", "latent_to_image", "latent_to_image_feat")
    if save_latent:
        if input_mode == "latent_frames":
            latent_step = int(config.get("latent_frames_val_step", 8))
            latent_sample_num_frames = config.get("latent_frames_sample_num_frames")
        elif input_mode in ("latent_to_image", "latent_to_image_feat"):
            latent_step = int(config.get("latent_to_image_val_step", 5))
            latent_sample_num_frames = 1
        else:  # "latent" image-latent mode
            latent_step = int(config.get("latent_val_step", 5))
            latent_sample_num_frames = 1  # image latent has T=1
    else:
        latent_step = None
        latent_sample_num_frames = None

    # latent_to_image mode: load a VAE decoder for saving decoded RGB previews
    save_decoded_rgb = (input_mode == "latent_to_image")
    vae_decoder = None
    if save_decoded_rgb:
        vae_path = config.get("vae_pretrained_path", "")
        model_name = config.get("model", "")
        if vae_path and model_name:
            try:
                from models_zoo import ImageVAEDecoderModule
                decode_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                vae_decoder = ImageVAEDecoderModule(
                    model_name=model_name,
                    vae_pretrained_path=vae_path,
                    mode="image",
                    device=decode_device,
                    decode_dtype=torch.bfloat16,
                )
                # move registered buffers (scaling_factor etc.) to the device too
                vae_decoder = vae_decoder.to(decode_device)
                print(f"[Preview] VAE decoder loaded for decoded RGB preview (model={model_name})")
            except Exception as e:
                print(f"[Preview] failed to load the VAE decoder; skipping decoded RGB previews: {e}")
                vae_decoder = None
                save_decoded_rgb = False
        else:
            print("[Preview] vae_pretrained_path or model not configured; skipping decoded RGB previews")
            save_decoded_rgb = False

    def _dump(items, sub_dir, prefix):
        if not items:
            return
        os.makedirs(sub_dir, exist_ok=True)
        for i, item in enumerate(random.sample(items, min(n_per, len(items)))):
            src = item.get("video_path", "")
            stem = Path(src).stem
            ext = Path(src).suffix or ".mp4"
            ip_label = item.get("ip_label", 5)
            # same prompt + different seed videos share one stem; append the source-pool directory
            # segment (e.g. 'train-xxx'), truncated to 30 chars; fallback: second-to-last dir segment
            parts = Path(src).parts
            pool = next((p for p in parts if p.startswith("train-")), parts[-3] if len(parts) >= 3 else "unknown")
            pool_tag = pool[-30:]
            base = (f"{prefix}_{i:02d}_{item.get('three_class', 'unknown')}_"
                    f"ip{ip_label}_{IP_ID2NAME.get(ip_label, '其它')}_{pool_tag}__{stem}")
            if os.path.exists(src):
                try:
                    shutil.copy2(src, os.path.join(sub_dir, f"{base}{ext}"))
                except Exception:
                    pass
            if save_latent:
                save_latent_visualization(
                    item, sub_dir, f"{base}_latent_step{latent_step}.png",
                    step=latent_step, latent_sample_num_frames=latent_sample_num_frames,
                )
            if save_decoded_rgb and vae_decoder is not None:
                _decode_latent_to_rgb_and_save(
                    item, sub_dir, f"{base}_decoded_rgb_step{latent_step}.png",
                    step=latent_step, model_name=config.get("model", ""),
                    vae_decoder=vae_decoder,
                )

    # top-level 5 buckets: positives + matching borderline + normal_other, three task views side by side
    TASK_GROUPS = ["porn", "porn_borderline", "gore", "gore_borderline", "normal_other"]

    for split_name, items in [("train", train_list), ("test", test_list)]:
        for g in TASK_GROUPS:
            _dump([x for x in items if x.get("train_group") == g],
                  os.path.join(preview_dir, split_name, g),
                  f"{split_name}_{g}")
        for ip_label in range(5):
            name = IP_ID2NAME[ip_label]
            _dump([x for x in items if int(x.get("ip_label", 5)) == ip_label],
                  os.path.join(preview_dir, split_name, "ip", name),
                  f"{split_name}_ip_{ip_label}_{name}")
        _dump([x for x in items if x.get("train_group") == "ip_borderline"],
              os.path.join(preview_dir, split_name, "ip", "ip_related"),
              f"{split_name}_ip_related")

    # release the VAE decoder GPU memory
    if vae_decoder is not None:
        import gc
        del vae_decoder
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[Preview] VAE decoder released.")


# =========================
# Sampler
# =========================
class TrainGroupEpochSampler(Sampler):
    """Each epoch draws a fixed number of samples per train_group in sampling_plan; without replacement when the pool is big enough, with replacement otherwise."""

    def __init__(self, train_list, sampling_plan, train_group_field="train_group"):
        self.sampling_plan = sampling_plan
        self.group_to_indices = defaultdict(list)
        for idx, x in enumerate(train_list):
            self.group_to_indices[x.get(train_group_field, "unknown")].append(idx)
        self.total_samples = sum(int(v) for v in sampling_plan.values())

        for g, n in sampling_plan.items():
            pool = len(self.group_to_indices.get(g, []))
            if pool == 0 and int(n) > 0:
                raise RuntimeError(f"Train group '{g}' has no samples, but sampling_plan requires {n}.")

    def __len__(self):
        return self.total_samples

    def __iter__(self):
        all_indices = []
        for g, n in self.sampling_plan.items():
            n = int(n)
            if n <= 0:
                continue
            pool = self.group_to_indices.get(g, [])
            replace = n > len(pool)
            all_indices.extend(np.random.choice(pool, size=n, replace=replace).tolist())
        random.shuffle(all_indices)
        return iter(all_indices)


# =========================
# Transforms
# =========================
# Three resize strategies (shared semantics across datasets):
#   stretch        : direct F.interpolate to (target, target); loses aspect ratio, no crop, no info loss
#   keep_ratio_pad : keep-ratio scaling to the largest size fitting (target, target), pad the rest with 0
#   native         : no forced resize to target; train adds random scale + random crop as augmentation,
#                    eval keeps the original size and only normalizes; output size is not fixed -> needs bs=1, CNN backbones only
# Train on stretch / keep_ratio_pad additionally: first resize to target*train_overscan,
# then random crop back to target + optional horizontal flip. Eval does no crop and no random augmentation.
def _resize_style_single(frame, target, style):
    """frame: a [C, H, W] tensor. target: int (square) or an (H, W) tuple (stretch only).
    native never calls this function."""
    if style == "stretch":
        if isinstance(target, (tuple, list)):
            th, tw = int(target[0]), int(target[1])
        else:
            th = tw = int(target)
        return F.interpolate(frame.unsqueeze(0), size=(th, tw),
                             mode="bilinear", align_corners=False).squeeze(0)
    if style == "keep_ratio_pad":
        # keep_ratio_pad always uses a square target
        target = int(target if not isinstance(target, (tuple, list)) else target[0])
        _, h, w = frame.shape
        scale = min(target / h, target / w)
        new_h = max(1, int(round(h * scale)))
        new_w = max(1, int(round(w * scale)))
        f = F.interpolate(frame.unsqueeze(0), size=(new_h, new_w),
                          mode="bilinear", align_corners=False).squeeze(0)
        pad_h, pad_w = target - new_h, target - new_w
        pad_top, pad_left = pad_h // 2, pad_w // 2
        return F.pad(f, (pad_left, pad_w - pad_left, pad_top, pad_h - pad_top),
                     mode="constant", value=0.0)
    raise ValueError(f"Unsupported preprocess_style: {style}")


def _native_batch_aug(inputs, enable_hflip):
    """Roll scale/crop/flip once for a whole batch of inputs (list of tensors) and apply it together — keeps output sizes consistent.
    Supports both [C,H,W] (middle-frame image) and [C,T,H,W] (video/latent) shapes; H/W are always the last two dims.
    Requires all samples in a batch to share the same original H/W (a data-uniformity precondition); raise a clear error otherwise.
    """
    first = inputs[0]
    h, w = int(first.shape[-2]), int(first.shape[-1])
    for i, x in enumerate(inputs[1:], 1):
        if int(x.shape[-2]) != h or int(x.shape[-1]) != w:
            raise RuntimeError(
                f"preprocess_style='native' requires all samples in a batch to share the same original resolution; "
                f"sample 0 is ({h},{w}), sample {i} is ({int(x.shape[-2])},{int(x.shape[-1])}). "
                f"Check whether the video resolutions in train_csv are really uniform."
            )

    scale = random.uniform(0.9, 1.1)
    if abs(scale - 1.0) > 0.01:
        new_h = max(8, int(round(h * scale)))
        new_w = max(8, int(round(w * scale)))
        resized = []
        for x in inputs:
            if x.ndim == 3:  # [C, H, W]
                x = F.interpolate(x.unsqueeze(0), size=(new_h, new_w),
                                  mode="bilinear", align_corners=False).squeeze(0)
            elif x.ndim == 4:  # [C, T, H, W] -> reshape to [C*T, 1, H, W] for bilinear, then reshape back
                c, t = x.shape[0], x.shape[1]
                x = x.reshape(c * t, 1, h, w)
                x = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)
                x = x.reshape(c, t, new_h, new_w)
            else:
                raise ValueError(f"_native_batch_aug does not support ndim={x.ndim}")
            resized.append(x)
        inputs = resized
        h, w = new_h, new_w

    area_frac = random.uniform(0.85, 1.0)
    side_frac = area_frac ** 0.5
    crop_h = max(8, int(round(h * side_frac)))
    crop_w = max(8, int(round(w * side_frac)))
    top = random.randint(0, max(0, h - crop_h))
    left = random.randint(0, max(0, w - crop_w))
    cropped = []
    for x in inputs:
        if x.ndim == 3:
            x = x[:, top:top + crop_h, left:left + crop_w]
        else:
            x = x[:, :, top:top + crop_h, left:left + crop_w]
        cropped.append(x)
    inputs = cropped

    if enable_hflip and random.random() < 0.5:
        inputs = [torch.flip(x, dims=[-1]) for x in inputs]

    return inputs


def _make_native_train_collate(enable_hflip):
    """Return a collate_fn: roll the spatial aug once per batch, apply it to all inputs, then default_collate.
    eval does not call this (no aug; default_collate works directly).
    """
    def _fn(batch):
        batch = [b for b in batch if b is not None]
        if not batch:
            return None
        inputs = _native_batch_aug([b["input"] for b in batch], enable_hflip=enable_hflip)
        for b, x in zip(batch, inputs):
            b["input"] = x
        return torch.utils.data.dataloader.default_collate(batch)
    return _fn


def _resolve_target_hw(image_size, style, stretch_target_hw):
    """Always return (th, tw). If stretch + stretch_target_hw is set -> use it; otherwise square (image_size, image_size)."""
    if style == "stretch" and stretch_target_hw is not None:
        return int(stretch_target_hw[0]), int(stretch_target_hw[1])
    s = int(image_size)
    return s, s



class ImageTransform:
    """Preprocessing for a single [C, H, W] image."""
    def __init__(self, image_size, style, train, enable_hflip=True, train_overscan=1.05,
                 mean=None, std=None, stretch_target_hw=None):
        self.th, self.tw = _resolve_target_hw(image_size, style, stretch_target_hw)
        self.style = style
        self.train = train
        self.enable_hflip = enable_hflip
        self.overscan = max(1.0, float(train_overscan)) if train else 1.0
        self.norm = Normalize(mean=mean or [0.485, 0.456, 0.406], std=std or [0.229, 0.224, 0.225])

    def __call__(self, image):
        if self.style == "native":
            # native: normalize directly and output the original size; train's spatial aug moves to the collate layer (per batch).
            return self.norm(image)

        inter_h = int(round(self.th * self.overscan))
        inter_w = int(round(self.tw * self.overscan))
        size_arg = (inter_h, inter_w) if self.style == "stretch" else inter_h
        image = _resize_style_single(image, size_arg, self.style)
        if self.train:
            top = random.randint(0, inter_h - self.th) if inter_h > self.th else 0
            left = random.randint(0, inter_w - self.tw) if inter_w > self.tw else 0
            image = image[:, top:top + self.th, left:left + self.tw]
            if self.enable_hflip and random.random() < 0.5:
                image = torch.flip(image, dims=[2])
        return self.norm(image)


# =========================
# Datasets
# =========================
def _load_prompt_embeds(path):
    """Read a prompt_embeds .pth and return a [seq_len, hidden_dim] tensor. Return None on failure.

    Three on-disk formats are supported:
      - bare tensor: [seq_len, D] or [1, seq_len, D]
      - list[tensor]: Z-Image-Turbo format, take data[0] -> [seq_len, D]
      - triple-CFG tensor: [3, seq_len, D] — InternVL-U triple-CFG format, take [0] (full condition)
    """
    if not path or not os.path.exists(path):
        return None
    try:
        data = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(data, list):
            # Z-Image-Turbo format: list[tensor[seq_len, D]]
            t = data[0].float()
        elif isinstance(data, torch.Tensor):
            t = data.float()
        else:
            return None
        if t.ndim == 3:
            if t.shape[0] == 1:
                t = t.squeeze(0)  # [1, L, D] → [L, D]
            elif t.shape[0] == 3:
                # InternVL-U triple CFG: [full, partial, uncond], take the full condition
                t = t[0]  # [3, L, D] → [L, D]
        return t  # [seq_len, D]
    except Exception:
        return None


def _load_prompt_embeds_mask(path):
    """Read a prompt_embeds mask .pth -> a [seq_len] tensor or None.

    The file may store None (meaning no padding) or a tensor [1, seq_len] / [seq_len].
    The triple-CFG full mask [3, seq_len] is supported: take [0] (full condition) -> [seq_len].
    """
    if not path or not os.path.exists(path):
        return None
    try:
        data = torch.load(path, map_location="cpu", weights_only=False)
        if data is None:
            return None
        if isinstance(data, torch.Tensor):
            t = data.float()
            if t.ndim == 2:
                if t.shape[0] == 1:
                    t = t.squeeze(0)  # [1, L] → [L]
                elif t.shape[0] == 3:
                    # InternVL-U triple-CFG mask: take the full condition
                    t = t[0]  # [3, L] → [L]
            return t  # [seq_len]
        return None
    except Exception:
        return None


def _load_prompt_embeds_with_mask(info, fallback_dim=None):
    """Load prompt_embeds and mask together; return (pe_tensor, mask_tensor).

    - pe_tensor: [seq_len, D]; an all-zero fallback is used on load failure
    - mask_tensor: [seq_len]; all-ones on load failure / no mask (all tokens valid)
    - fallback_dim: fallback D dim on load failure; a global default when None
    """
    pe = _load_prompt_embeds(info.get("prompt_embeds_path"))
    if pe is None:
        dim = fallback_dim if fallback_dim is not None else _PROMPT_EMBEDS_FALLBACK_DIM
        pe = torch.zeros(1, dim)  # seq_len=1 is enough; still a zero vector after mean pooling
    mask = _load_prompt_embeds_mask(info.get("prompt_embeds_mask_path"))
    # fallback: flux2-klein-base-9b / internvl-u use the prompt_attention_mask_forward directory name
    if mask is None and info.get("prompt_embeds_mask_path"):
        alt_path = info["prompt_embeds_mask_path"].replace(
            "prompt_embeds_mask_forward", "prompt_attention_mask_forward"
        )
        if alt_path != info["prompt_embeds_mask_path"]:
            mask = _load_prompt_embeds_mask(alt_path)
    if mask is None:
        mask = torch.ones(pe.shape[0])  # all ones -> all tokens valid
    return pe, mask


# the fallback dim is decided by the fallback_dim each dataset passes to _load_prompt_embeds_with_mask
_PROMPT_EMBEDS_FALLBACK_DIM = 4096  # default (Wan text encoder for videos)


def _build_meta(info, extra=None):
    meta = {
        "name": info["name"],
        "video_path": info["video_path"],
        "predict": info["predict"],
        "three_class": info["three_class"],
        "sampling_group": info.get("sampling_group", "unknown"),
        "ip_group": info.get("ip_group", "ip_other"),
        "train_group": info.get("train_group", "unknown"),
        "manually_modified": bool(info.get("manually_modified", False)),
    }
    if extra:
        meta.update(extra)
    return meta


def _build_labels_dict(info):
    return {
        "porn_label": torch.tensor(info["porn_label"], dtype=torch.long),
        "gore_label": torch.tensor(info["gore_label"], dtype=torch.long),
        "ip_label": torch.tensor(info.get("ip_label", 5), dtype=torch.long),
        "has_ip_label": torch.tensor(1, dtype=torch.long),
    }


def load_latent_stats_from_cache(cache_path, expected_chans=None):
    """Reuse per-channel mean/std directly from a saved latent_stats.json, skipping the recomputation.
    cache_path: full path of a latent_stats.json produced by an earlier run
    expected_chans: when given, validate the channel count; raise on mismatch (avoids mixing wan-2.1 <-> wan-2.2 latents)
    Returns (mean: Tensor[C], std: Tensor[C], info: dict)
    """
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"latent_stats cache file does not exist: {cache_path}")
    with open(cache_path, "r", encoding="utf-8") as f:
        info = json.load(f)

    mean_list = info.get("mean_per_channel")
    std_list = info.get("std_per_channel")
    if mean_list is None or std_list is None:
        raise ValueError(f"latent_stats cache is missing mean_per_channel / std_per_channel fields: {cache_path}")
    if len(mean_list) != len(std_list):
        raise ValueError(f"latent_stats cache mean/std channel counts differ: "
                         f"{len(mean_list)} vs {len(std_list)} ({cache_path})")
    if expected_chans is not None and len(mean_list) != int(expected_chans):
        raise ValueError(f"latent_stats cache channel count={len(mean_list)} does not match "
                         f"latent_in_chans={expected_chans} (different VAE?): {cache_path}")

    mean = torch.tensor(mean_list, dtype=torch.float32)
    std = torch.tensor(std_list, dtype=torch.float32)
    info["_loaded_from_cache"] = cache_path
    return mean, std, info


def load_decoded_feat_stats_from_cache(cache_path, expected_chans=None):
    """Reuse per-channel mean/std directly from a saved decoded_feat_stats.json, skipping the recomputation.
    Same structure as load_latent_stats_from_cache; kept as a separate function to avoid semantic confusion (that one is the
    normalized-latent domain, this one is the decoder mid-layer feature domain — completely different distributions).
    """
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"decoded_feat_stats cache file does not exist: {cache_path}")
    with open(cache_path, "r", encoding="utf-8") as f:
        info = json.load(f)

    mean_list = info.get("mean_per_channel")
    std_list = info.get("std_per_channel")
    if mean_list is None or std_list is None:
        raise ValueError(f"decoded_feat_stats cache is missing mean_per_channel / std_per_channel: {cache_path}")
    if len(mean_list) != len(std_list):
        raise ValueError(f"decoded_feat_stats cache mean/std channel counts differ: "
                         f"{len(mean_list)} vs {len(std_list)} ({cache_path})")
    if expected_chans is not None and len(mean_list) != int(expected_chans):
        raise ValueError(f"decoded_feat_stats cache channel count={len(mean_list)} does not match "
                         f"expected={expected_chans} (VAE or hook site changed?): {cache_path}")

    mean = torch.tensor(mean_list, dtype=torch.float32)
    std = torch.tensor(std_list, dtype=torch.float32)
    info["_loaded_from_cache"] = cache_path
    return mean, std, info


# =============================================================
# Image Training Support (for train_image.py)
# =============================================================

# -------------------------
# Helpers
# -------------------------
def _image_latent_spatial_resize(x, target_hw, style):
    """Spatial resize on a 3D latent [C, H, W] (for ImageLatentDataset).
      style='stretch'       : F.interpolate to (th, tw)
      style='keep_ratio_pad': keep-ratio scaling + zero padding
    """
    if target_hw is None:
        return x
    th, tw = int(target_hw[0]), int(target_hw[1])
    if x.shape[-2] == th and x.shape[-1] == tw:
        return x
    x4d = x.unsqueeze(0)  # [1, C, H, W]
    if style == "stretch":
        out = F.interpolate(x4d, size=(th, tw), mode="bilinear", align_corners=False)
        return out.squeeze(0)
    if style == "keep_ratio_pad":
        c, h, w = x.shape
        scale = min(th / h, tw / w)
        new_h = max(1, int(round(h * scale)))
        new_w = max(1, int(round(w * scale)))
        out = F.interpolate(x4d, size=(new_h, new_w), mode="bilinear", align_corners=False)
        pad_h = th - new_h
        pad_w = tw - new_w
        pad_top, pad_left = pad_h // 2, pad_w // 2
        out = F.pad(out, (pad_left, pad_w - pad_left, pad_top, pad_h - pad_top),
                    mode="constant", value=0.0)
        return out.squeeze(0)
    raise ValueError(f"Unsupported image latent resize style: {style!r}")


# -------------------------
# ImageFileDataset
# -------------------------
class ImageFileDataset(Dataset):
    """Dataset loading straight from image files (jpg/png); no video decoding, no frame sampling.
    Reuses the existing ImageTransform.
    """
    def __init__(self, info_list, train=False, image_size=224, style="stretch",
                 enable_hflip=True, train_overscan=1.05, mean=None, std=None,
                 stretch_target_hw=None, enable_prompt_fusion=False,
                 prompt_embeds_dim=None):
        self.info_list = info_list
        self.enable_prompt_fusion = enable_prompt_fusion
        self.prompt_embeds_dim = prompt_embeds_dim
        self.transform = ImageTransform(
            image_size=image_size, style=style, train=train,
            enable_hflip=enable_hflip, train_overscan=train_overscan,
            mean=mean, std=std, stretch_target_hw=stretch_target_hw,
        )

    def __len__(self):
        return len(self.info_list)

    def __getitem__(self, idx):
        info = self.info_list[idx]
        # accept both file_path (image) and video_path (alias field; both carry the same value in load_predictions_csv)
        path = info.get("file_path") or info.get("video_path", "")
        if not os.path.exists(path):
            return None
        try:
            from PIL import Image as PILImage
            img = PILImage.open(path).convert("RGB")
            img_t = torch.from_numpy(np.array(img)).float() / 255.0
            img_t = img_t.permute(2, 0, 1).contiguous()  # [3, H, W]
            img_t = self.transform(img_t)                 # [3, H, W] after transform
        except Exception:
            return None
        result = {
            "input": img_t,
            **_build_labels_dict(info),
            "meta": _build_meta(info),
        }
        if self.enable_prompt_fusion:
            pe, pm = _load_prompt_embeds_with_mask(info, fallback_dim=self.prompt_embeds_dim)
            # masked mean pool directly inside the dataset -> [D], avoiding variable-length seq collate issues
            pm_f = pm.float().unsqueeze(-1)              # [seq_len, 1]
            valid_sum = pm_f.sum(dim=0).clamp(min=1.0)   # [1]
            pooled = (pe.float() * pm_f).sum(dim=0) / valid_sum  # [D]
            result["prompt_embeds"] = pooled
        return result


# -------------------------
# Image Latent Loader (dispatch by model name, hardcoded whitelist)
# -------------------------
def _load_image_latent_by_model(pth_path, model_name):
    """Read an image-latent .pth by model name; always return a 3D tensor [C, H, W].

    Each model has a fixed on-disk layout, so the dispatch is hardcoded here to avoid heuristic squeeze misjudging the channel dim.
    To add a model, append an elif branch below.
      - qwen-image-2512:    saved [1, C, 1, H, W]  -> squeeze(0).squeeze(1) -> [C, H, W]  (C=16)
                            or packed [1, HW, Cx4] (early SAGE-enhancement runs stored it packed) -> unpack separately -> [C, H, W]
      - internvl-u:         saved [1, C, H, W]     -> squeeze(0)            -> [C, H, W]  (C=16, reuses the Qwen VAE; data generation added no T dim)
                               or [1, C, 1, H, W]  -> squeeze(0).squeeze(1) -> [C, H, W]  (5D compatible)
      - z-image-turbo:      saved [1, C, H, W]     -> squeeze(0)            -> [C, H, W]  (C=16)
      - hunyuan-image-2_1:  saved [1, C, H, W]     -> squeeze(0)            -> [C, H, W]  (C=64)
      - flux2-klein-base-9b:saved [1, C, H, W]     -> squeeze(0)            -> [C, H, W]  (C=32)
    """
    x = torch.load(pth_path, map_location="cpu").float()
    if model_name == "qwen-image-2512":
        # standard 5D: [1, C, 1, H, W]  -> squeeze(0).squeeze(1) -> [C, H, W]
        if x.ndim == 5 and x.shape[0] == 1 and x.shape[2] == 1:
            return x.squeeze(0).squeeze(1)            # [C, H, W]
        # 3D packed compat: [1, num_patches, channels_packed]
        # Early SAGE-enhancement runs skipped pipe._unpack_latents and stored the pipeline-internal packed layout.
        # Unpack independently here; identical to QwenImagePipeline._unpack_latents.
        if x.ndim == 3 and x.shape[0] == 1:
            num_patches = x.shape[1]       # e.g., 6889 = 83×83
            ch_packed = x.shape[2]         # e.g., 64 = C×2×2
            hw_half = int(num_patches ** 0.5)
            if hw_half * hw_half != num_patches:
                raise ValueError(
                    f"qwen-image-2512 3D packed: num_patches={num_patches} is not a perfect square, "
                    f"cannot infer the spatial dims: {pth_path}"
                )
            height = 2 * hw_half
            width = 2 * hw_half
            x = x.view(1, hw_half, hw_half, ch_packed // 4, 2, 2)
            x = x.permute(0, 3, 1, 4, 2, 5)
            x = x.reshape(1, ch_packed // 4, 1, height, width)
            return x.squeeze(0).squeeze(1)            # [C, H, W]
        raise ValueError(
            f"qwen-image-2512 expects [1,C,1,H,W] or packed [1,HW,C], got {tuple(x.shape)}: {pth_path}"
        )
    if model_name == "internvl-u":
        # Reuses the Qwen VAE, but the data-generation pipeline added no T=1 dim; saved as 4D [1, C, H, W]
        # Both 5D [1, C, 1, H, W] and 4D [1, C, H, W] on-disk formats are accepted
        if x.ndim == 5 and x.shape[0] == 1 and x.shape[2] == 1:
            return x.squeeze(0).squeeze(1)        # [C, H, W]
        if x.ndim == 4 and x.shape[0] == 1:
            return x.squeeze(0)                   # [C, H, W]
        raise ValueError(
            f"internvl-u expects [1,C,1,H,W] or [1,C,H,W], got {tuple(x.shape)}: {pth_path}"
        )
    if model_name in ("z-image-turbo", "hunyuan-image-2_1", "flux2-klein-base-9b"):
        if x.ndim != 4 or x.shape[0] != 1:
            raise ValueError(
                f"{model_name} expects [1,C,H,W], got {tuple(x.shape)}: {pth_path}"
            )
        return x.squeeze(0)                       # [C, H, W]
    raise ValueError(
        f"Unknown model={model_name!r}; cannot decide the image-latent layout. "
        f"Supported: ['qwen-image-2512', 'internvl-u', 'z-image-turbo', 'hunyuan-image-2_1', 'flux2-klein-base-9b']. "
        f"To add a new model, append an elif branch in _load_image_latent_by_model."
    )


# -------------------------
# ImageLatentDataset
# -------------------------
class ImageLatentDataset(Dataset):
    """Image-latent Dataset: read .pth by model name -> [C,H,W],
    per-channel normalization + spatial resize + optional data augmentation.
    """
    def __init__(self, info_list, model, step_mode="random", fixed_step=5, num_steps=10,
                 train=False, data_aug=False,
                 latent_mean=None, latent_std=None,
                 resize_target=None, resize_style="stretch",
                 enable_prompt_fusion=False, prompt_embeds_dim=None):
        self.info_list = info_list
        self.model = str(model)
        self.step_mode = str(step_mode)
        self.fixed_step = int(fixed_step)
        self.num_steps = int(num_steps)
        self.train = train
        self.data_aug = data_aug
        self.enable_prompt_fusion = enable_prompt_fusion
        self.prompt_embeds_dim = prompt_embeds_dim
        # per-channel normalize stats: view(-1,1,1) for 3D [C,H,W] broadcast
        if latent_mean is not None and latent_std is not None:
            self.latent_mean = latent_mean.detach().float().view(-1, 1, 1)
            self.latent_std  = latent_std.detach().float().view(-1, 1, 1).clamp(min=1e-6)
        else:
            self.latent_mean = None
            self.latent_std  = None
        self.resize_target = (int(resize_target[0]), int(resize_target[1])) if resize_target is not None else None
        self.resize_style = str(resize_style)

    def __len__(self):
        return len(self.info_list)

    def _pick_step(self):
        if self.step_mode == "random":
            return random.randrange(self.num_steps)
        if self.step_mode == "fixed":
            return self.fixed_step
        raise ValueError(f"Unsupported step_mode: {self.step_mode!r} (only 'random' / 'fixed' are supported)")

    def _load_latent(self, pth_path):
        """Load an image-latent .pth -> [C, H, W]; the layout is decided solely by self.model."""
        return _load_image_latent_by_model(pth_path, self.model)

    def _augment(self, x):
        """图像 latent 空间增强（适配 3D [C,H,W]）：
        1. Random HFlip
        2. Random Crop + Resize back (\u88c1 90% \u9762\u79ef\u518d resize \u56de\u539f\u5c3a\u5bf8)
        3. Gaussian Noise (50% \u6982\u7387, sigma=0.01)
        4. Scale Jitter (\xd70.95~1.05)
        """
        # 1. hflip
        if random.random() > 0.5:
            x = torch.flip(x, dims=[-1])
        # 2. random crop + resize back
        if x.shape[-2] >= 32 and x.shape[-1] >= 32:
            c, h, w = x.shape
            ch = max(16, int(h * 0.9))
            cw = max(16, int(w * 0.9))
            top  = random.randint(0, h - ch)
            left = random.randint(0, w - cw)
            x = x[:, top:top + ch, left:left + cw]   # [C, ch, cw]
            x = F.interpolate(x.unsqueeze(0), size=(h, w),
                              mode="bilinear", align_corners=False).squeeze(0)
        # 3. gaussian noise
        if random.random() > 0.5:
            x = x + 0.01 * torch.randn_like(x)
        # 4. scale jitter
        return x * random.uniform(0.95, 1.05)
    
    def __getitem__(self, idx):
        info = self.info_list[idx]
        step = self._pick_step()
        pth = os.path.join(info["latent_root"], str(step), f"{info['name']}.pth")
        if not os.path.exists(pth):
            return None
        try:
            x = self._load_latent(pth)
            # per-channel normalize
            if self.latent_mean is not None:
                if x.shape[0] != self.latent_mean.shape[0]:
                    raise ValueError(
                        f"latent C mismatch: x.shape={tuple(x.shape)}, "
                        f"latent_mean expects C={self.latent_mean.shape[0]}. "
                        f"\u6587\u4ef6: {pth}"
                    )
                x = (x - self.latent_mean) / self.latent_std
            # \u7a7a\u95f4 resize
            if self.resize_target is not None:
                x = _image_latent_spatial_resize(x, self.resize_target, self.resize_style)
            # \u6570\u636e\u589e\u5f3a\uff08\u4ec5 train\uff09
            if self.train and self.data_aug:
                x = self._augment(x)
        except Exception:
            return None
        result = {
            "input": x,
            **_build_labels_dict(info),
            "meta": _build_meta(info, {"latent_step": int(step)}),
        }
        if self.enable_prompt_fusion:
            pe, pm = _load_prompt_embeds_with_mask(info, fallback_dim=self.prompt_embeds_dim)
            # masked mean pool directly inside the dataset -> [D], avoiding variable-length seq collate issues
            pm_f = pm.float().unsqueeze(-1)              # [seq_len, 1]
            valid_sum = pm_f.sum(dim=0).clamp(min=1.0)   # [1]
            pooled = (pe.float() * pm_f).sum(dim=0) / valid_sum  # [D]
            result["prompt_embeds"] = pooled
        return result


# -------------------------
# Image Latent Stats
# -------------------------
def compute_image_latent_stats(train_list, model, num_steps, num_samples=500, seed=42):
    """Compute per-channel mean/std of image latents, used by ImageLatentDataset for normalize.

    Each .pth is parsed to 3D [C,H,W] via _load_image_latent_by_model by model name.
    Returns (mean: Tensor[C], std: Tensor[C], info: dict).
    """
    rng = random.Random(seed)
    n_req = min(int(num_samples), len(train_list))
    sub = rng.sample(train_list, n_req)

    running_sum = None      # float64 [C]
    running_sumsq = None    # float64 [C]
    count = 0               # accumulated element count per channel
    n_loaded, n_failed = 0, 0
    step_hist = Counter()

    for item in tqdm(sub, desc="Computing image latent stats", ncols=80):
        step = rng.randrange(int(num_steps))
        step_hist[step] += 1
        pth = os.path.join(item["latent_root"], str(step), f"{item['name']}.pth")
        if not os.path.exists(pth):
            n_failed += 1
            continue
        try:
            x = _load_image_latent_by_model(pth, model)    # [C, H, W]
            C = x.shape[0]
            if running_sum is None:
                running_sum   = torch.zeros(C, dtype=torch.float64)
                running_sumsq = torch.zeros(C, dtype=torch.float64)
            flat = x.reshape(C, -1).double()
            running_sum   += flat.sum(dim=1)
            running_sumsq += (flat * flat).sum(dim=1)
            count += flat.shape[1]
            n_loaded += 1
        except Exception:
            n_failed += 1
            continue

    if n_loaded == 0 or count == 0:
        raise RuntimeError(
            f"compute_image_latent_stats: no latent loaded "
            f"(model={model!r}, requested={n_req}, failed={n_failed}). "
            f"Check that the latent_root / step paths match the model layout."
        )

    mean64 = running_sum / count
    var64 = (running_sumsq / count) - mean64 * mean64
    var64 = var64.clamp(min=1e-12)
    std64 = var64.sqrt()

    mean = mean64.float()
    std  = std64.float()

    info = {
        "model": str(model),
        "num_samples_requested": n_req,
        "num_loaded": n_loaded,
        "num_failed": n_failed,
        "num_channels": int(mean.numel()),
        "elements_per_channel": int(count),
        "num_steps": int(num_steps),
        "step_histogram": dict(sorted(step_hist.items())),
        "mean_per_channel": [float(v) for v in mean.tolist()],
        "std_per_channel":  [float(v) for v in std.tolist()],
        "mean_overall": float(mean.mean().item()),
        "std_overall":  float(std.mean().item()),
        "mean_abs_max": float(mean.abs().max().item()),
        "std_min":      float(std.min().item()),
        "std_max":      float(std.max().item()),
    }
    return mean, std, info


# -------------------------
# ImageLatentX1RawDataset (shared by latent_to_image / latent_to_image_feat)
# -------------------------
class ImageLatentX1RawDataset(Dataset):
    """Read-only image latent_x1 (normalized domain); no normalize / resize / aug.
    Shared by the latent_to_image / latent_to_image_feat modes:
    all VAE decoding + post-processing happens inside model.forward on the GPU.

    Returns a dict:
        input: Tensor [C, H, W] float32 (raw values in the normalized domain)
        porn_label / gore_label / ip_label / has_ip_label
        meta: with latent_root, step and other fields
    """
    def __init__(self, info_list, model, step_mode="random", fixed_step=5, num_steps=10,
                 enable_prompt_fusion=False, prompt_embeds_dim=None):
        self.info_list = info_list
        self.model = str(model)
        self.step_mode = str(step_mode)
        self.fixed_step = int(fixed_step)
        self.num_steps = int(num_steps)
        self.enable_prompt_fusion = enable_prompt_fusion
        self.prompt_embeds_dim = prompt_embeds_dim

    def __len__(self):
        return len(self.info_list)

    def _pick_step(self):
        if self.step_mode == "random":
            return random.randrange(self.num_steps)
        if self.step_mode == "fixed":
            return self.fixed_step
        raise ValueError(f"Unsupported step_mode: {self.step_mode!r} (only 'random' / 'fixed' are supported)")

    def __getitem__(self, idx):
        info = self.info_list[idx]
        step = self._pick_step()
        pth = os.path.join(info["latent_root"], str(step), f"{info['name']}.pth")
        if not os.path.exists(pth):
            return None
        try:
            x = _load_image_latent_by_model(pth, self.model)  # [C, H, W] float32
        except Exception:
            return None
        result = {
            "input": x,
            **_build_labels_dict(info),
            "meta": _build_meta(info, {"latent_root": info["latent_root"], "step": int(step)}),
        }
        if self.enable_prompt_fusion:
            pe, pm = _load_prompt_embeds_with_mask(info, fallback_dim=self.prompt_embeds_dim)
            pm_f = pm.float().unsqueeze(-1)              # [seq_len, 1]
            valid_sum = pm_f.sum(dim=0).clamp(min=1.0)   # [1]
            pooled = (pe.float() * pm_f).sum(dim=0) / valid_sum  # [D]
            result["prompt_embeds"] = pooled
        return result


# -------------------------
# Image Decoded Feat Stats (for latent_to_image_feat)
# -------------------------
def compute_image_decoded_feat_stats(train_list, vae_extractor, model_name, num_steps,
                                      num_samples=200, seed=42, device=None):
    """Decode N samples with ImageVAEDecoderModule(mode='feat') and compute per-channel mean/std.
    Returns (mean: Tensor[C], std: Tensor[C], info: dict).
    """
    rng = random.Random(seed)
    n_req = min(int(num_samples), len(train_list))
    sub = rng.sample(train_list, n_req)

    running_sum = None
    running_sumsq = None
    count = 0
    n_loaded, n_failed = 0, 0

    for item in tqdm(sub, desc="Computing image decoded feat stats", ncols=80):
        step = rng.randrange(int(num_steps))
        pth = os.path.join(item["latent_root"], str(step), f"{item['name']}.pth")
        if not os.path.exists(pth):
            n_failed += 1
            continue
        try:
            x = _load_image_latent_by_model(pth, model_name)  # [C, H, W]
            x_batch = x.unsqueeze(0).to(device)  # [1, C, H, W]
            feat = vae_extractor(x_batch)  # [1, feat_chans, H_feat, W_feat]
            feat = feat.squeeze(0).float().cpu()  # [feat_chans, H_feat, W_feat]
            C = feat.shape[0]
            if running_sum is None:
                running_sum = torch.zeros(C, dtype=torch.float64)
                running_sumsq = torch.zeros(C, dtype=torch.float64)
            flat = feat.reshape(C, -1).double()
            running_sum += flat.sum(dim=1)
            running_sumsq += (flat * flat).sum(dim=1)
            count += flat.shape[1]
            n_loaded += 1
        except Exception:
            n_failed += 1
            continue

    if n_loaded == 0 or count == 0:
        raise RuntimeError(
            f"compute_image_decoded_feat_stats: no feat computed "
            f"(model={model_name!r}, requested={n_req}, failed={n_failed})"
        )

    mean64 = running_sum / count
    var64 = (running_sumsq / count) - mean64 * mean64
    var64 = var64.clamp(min=1e-12)
    std64 = var64.sqrt()
    mean = mean64.float()
    std = std64.float()

    info = {
        "model": str(model_name),
        "num_samples_requested": n_req,
        "num_loaded": n_loaded,
        "num_failed": n_failed,
        "num_channels": int(mean.numel()),
        "elements_per_channel": int(count),
        "mean_per_channel": [float(v) for v in mean.tolist()],
        "std_per_channel": [float(v) for v in std.tolist()],
        "mean_overall": float(mean.mean().item()),
        "std_overall": float(std.mean().item()),
        "mean_abs_max": float(mean.abs().max().item()),
        "std_min": float(std.min().item()),
        "std_max": float(std.max().item()),
    }
    return mean, std, info


# -------------------------
# Image Dataset / DataLoader Builders
# -------------------------
def build_image_datasets(config, train_list, val_list, test_list, latent_stats=None):
    """Dataset builder for train_image.py; supports input_mode in {'image', 'latent', 'latent_to_image', 'latent_to_image_feat'}."""
    mode = config.get("input_mode", "image")
    epf = bool(config.get("enable_prompt_fusion", False))
    pe_dim = int(config.get("prompt_fusion_text_dim", 4096)) if epf else None

    if mode == "image":
        kw = dict(
            image_size=config.get("image_size", 224),
            style=config.get("image_preprocess_style", "stretch"),
            enable_hflip=bool(config.get("image_enable_hflip", True)),
            train_overscan=float(config.get("image_train_overscan", 1.05)),
            mean=config.get("image_mean", [0.485, 0.456, 0.406]),
            std=config.get("image_std",  [0.229, 0.224, 0.225]),
            stretch_target_hw=config.get("image_stretch_target_hw"),
            enable_prompt_fusion=epf,
            prompt_embeds_dim=pe_dim,
        )
        return (ImageFileDataset(train_list, train=True,  **kw),
                ImageFileDataset(val_list,   train=False, **kw),
                ImageFileDataset(test_list,  train=False, **kw))

    if mode == "latent":
        latent_mean, latent_std = (latent_stats if latent_stats is not None else (None, None))
        resize_hw    = config.get("latent_stretch_target_hw")
        resize_style = config.get("latent_preprocess_style", "stretch")
        resize_target = (int(resize_hw[0]), int(resize_hw[1])) if resize_hw else None
        model_name = str(config["model"])
        train_ds = ImageLatentDataset(
            train_list,
            model=model_name,
            step_mode=config.get("latent_train_step_mode", "random"),
            fixed_step=int(config.get("latent_val_step", 5)),
            num_steps=int(config["num_steps"]),
            train=True, data_aug=bool(config.get("latent_data_aug", True)),
            latent_mean=latent_mean, latent_std=latent_std,
            resize_target=resize_target, resize_style=resize_style,
            enable_prompt_fusion=epf,
            prompt_embeds_dim=pe_dim,
        )
        eval_kw = dict(
            model=model_name,
            step_mode="fixed", fixed_step=int(config.get("latent_val_step", 5)),
            num_steps=int(config["num_steps"]),
            train=False, data_aug=False,
            latent_mean=latent_mean, latent_std=latent_std,
            resize_target=resize_target, resize_style=resize_style,
            enable_prompt_fusion=epf,
            prompt_embeds_dim=pe_dim,
        )
        return (train_ds,
                ImageLatentDataset(val_list,  **eval_kw),
                ImageLatentDataset(test_list, **eval_kw))

    if mode in ("latent_to_image", "latent_to_image_feat"):
        model_name = str(config["model"])
        step_mode_key = "latent_to_image_train_step_mode"
        val_step_key = "latent_to_image_val_step"
        common = dict(
            model=model_name,
            num_steps=int(config["num_steps"]),
            enable_prompt_fusion=epf,
            prompt_embeds_dim=pe_dim,
        )
        train_ds = ImageLatentX1RawDataset(
            train_list,
            step_mode=config.get(step_mode_key, "random"),
            fixed_step=int(config.get(val_step_key, 5)),
            **common,
        )
        val_ds = ImageLatentX1RawDataset(
            val_list,
            step_mode="fixed",
            fixed_step=int(config.get(val_step_key, 5)),
            **common,
        )
        test_ds = ImageLatentX1RawDataset(
            test_list,
            step_mode="fixed",
            fixed_step=int(config.get(val_step_key, 5)),
            **common,
        )
        return train_ds, val_ds, test_ds

    raise ValueError(f"train_image.py does not support input_mode={mode!r}; only 'image' / 'latent' / 'latent_to_image' / 'latent_to_image_feat' are supported")


def build_image_dataloaders(config, train_ds, val_ds, test_ds, train_list, collate_fn):
    """DataLoader builder for train_image.py.
    With native preprocess_style, per-batch spatial aug happens in the collate layer (same as the video branch).
    """
    sampler = TrainGroupEpochSampler(
        train_list=train_list, sampling_plan=config["sampling_plan"],
    )
    mode = config.get("input_mode", "image")
    style_key = "image_preprocess_style" if mode == "image" else "latent_preprocess_style"
    style = config.get(style_key, "stretch")
    hflip = bool(config.get("image_enable_hflip", True)) if mode == "image" else True
    train_collate = _make_native_train_collate(enable_hflip=hflip) if style == "native" else collate_fn
    base = dict(batch_size=config["bs"], num_workers=config["nw"], pin_memory=True)
    return (
        DataLoader(train_ds, sampler=sampler, shuffle=False, collate_fn=train_collate, **base),
        DataLoader(val_ds,   shuffle=False, collate_fn=collate_fn, **base),
        DataLoader(test_ds,  shuffle=False, collate_fn=collate_fn, **base),
    )
