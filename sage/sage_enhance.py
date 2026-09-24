#!/usr/bin/env python
"""
sage_enhance.py — multi-model SAGE enhancement data-collection script (InGuard open-source release)

Supports 5 models: z-image-turbo / qwen-image-2512 / hunyuan-image-2_1 / flux2-klein-base-9b / internvl-u

Runs the full generation pipeline for each alpha value and saves the enhanced prompt_embeds and latent_x1
(Flow Matching one-step estimate); the directory layout matches each model's original save_data output.
These data train / evaluate the InGuard components (the PE-MLP gating model and the latent detector).

══════════════════════════════════════════════════════════════
Enhancement method
══════════════════════════════════════════════════════════════

  sage — SAGE (Soft-gated Asymmetric Guardrail for Embeddings):
         asymmetric soft gating + per-prompt concept selection; only gated risky prompts
         are enhanced. Gating thresholds: porn_T=3, gore_T=2, ip∈{1-5}

══════════════════════════════════════════════════════════════
Top config-block parameters (edit manually before running, or override via CLI args)
══════════════════════════════════════════════════════════════

  MODEL_NAME       generation model name; valid values:
                     "z-image-turbo" / "qwen-image-2512" / "hunyuan-image-2_1"
                     "flux2-klein-base-9b" / "internvl-u"
  NUM_JOBS         total number of shards (with multi-GPU parallelism each card runs one shard)
  TARGET_JOB       current shard index (0 ~ NUM_JOBS-1)
  SKIP_EXISTING    whether to skip already-generated images
  MAX_ROWS         cap on the number of rows processed (0 = unlimited)

  CSV_PATH         gating-prediction CSV path (with id, prompt, pred_* columns)
  ALPHAS           list of alpha values controlling the projection strength (negative = more aggressive projection)

  SEED             random seed
  DEVICE           CUDA device id, e.g. "cuda:0"
  SAVE_FULL_TRACE  whether to save the full intermediate data (latents, velocity, decoded_latents_x1)
                     False = save only latents_x1 (default)
  NUM_WORKERS      background save-thread count

  TAU_TOXIC        SAGE soft-gating temperature (toxic direction), default 0.005
  TAU_SAFE         SAGE soft-gating temperature (safe direction), default 0.02
  PORN_THRESHOLD   gating threshold: pred_porn >= PORN_THRESHOLD counts as risky
  GORE_THRESHOLD   gating threshold: pred_gore >= GORE_THRESHOLD counts as risky
  IP_CTRL_MIN/MAX  IP gating range: pred_ip ∈ [1, 5] counts as risky

══════════════════════════════════════════════════════════════
Optional CLI args (override the top config block)
══════════════════════════════════════════════════════════════

  --model_name       same as MODEL_NAME
  --method           same as METHOD
  --model_path       model path (defaults to the path in MODEL_REGISTRY)
  --csv_path         same as CSV_PATH
  --save_dir         save dir (defaults to the path in MODEL_REGISTRY)
  --alphas           comma-separated alpha values, e.g. "-0.05,-0.01,0.01"
  --num_jobs         same as NUM_JOBS
  --target_job       same as TARGET_JOB
  --max_rows         same as MAX_ROWS
  --skip_existing    same as SKIP_EXISTING
  --shuffle/--no-shuffle  whether to shuffle the CSV order
  --seed             same as SEED
  --width            image width (defaults to the model config)
  --height           image height (defaults to the model config)
  --num_inference_steps  inference steps (defaults to the model config)
  --guidance_scale   guidance strength (defaults to the model config)
  --device           same as DEVICE
  --col_porn/gore/ip  label column names in the CSV
  --tau_toxic/safe   SAGE soft-gating temperatures
  --ckpt_root        gating-model checkpoint root dir
  --auto_csv         auto-detect the best gating result
  --pred_source      gating-prediction source: "prompt" (train_prompt.py) / "text" (a text-encoder
                      training variant not shipped in this repository)
  --encoder_name     text encoder name (required when --pred_source text)

══════════════════════════════════════════════════════════════
Run guide
══════════════════════════════════════════════════════════════

SAGE relies on per-prompt concept selection; the gating model's (PE-MLP) predicted labels decide:
  (a) which prompts are gated in (pred_porn >= 3 OR pred_gore >= 2 OR pred_ip ∈ {1-5})
  (b) which concept groups build P_C for each gated prompt

Steps:
  1. train the gating model:
       python pe_mlp/train_prompt.py --model_name {model_name}
     after training it exports {output_dir}/meta/test_predictions.csv:
       id, prompt, pred_porn, pred_gore, pred_ip

  2a. Option 1 — auto-detect the best gating result (recommended):
       python sage/sage_enhance.py \
         --model_name {model_name} --auto_csv \
         --model_path /path/to/model --ckpt_root /path/to/ckpt_root
     the script auto-searches {ckpt_root}/*/{model_name}/prompt-mlp-multitask-porn-gore-ip/*/,
     reads acc_avg from history.json, and picks the best training run.

  2b. Option 2 — manually specify the CSV:
       CSV_PATH = "/path/to/test_predictions.csv"
       COL_PORN = "pred_porn"
       COL_GORE = "pred_gore"
       COL_IP   = "pred_ip"

  3. run on a cluster (multi-GPU sharding: submit each card separately with --target_job 0,1,2,...)

Output directories:
  {save_dir_base}/
    sage/alpha_-0.10/{image,prompt_embeds_forward,noise_init,latents_x1,sigmas}
    sage/alpha_-0.09/...
    ... (one per alpha)

══════════════════════════════════════════════════════════════
InternVL-U-specific notes
══════════════════════════════════════════════════════════════

  InternVL-U uses the custom InternVLUPipeline (not diffusers); the code lives in the backends/internvlu/ package.
  Differences from the other 4 models:
    - text encoder: a VLM (Qwen2.5, 2048d, 28 layers) instead of a standalone text encoder
    - vlm_select_layer = [-1, -2]: the last 2 layers concatenated → D = 4096
    - decoder_projector = Identity (no projection); SAGE operates directly in the VLM hidden-state space
    - triple CFG: cond / part_cond / uncond (batch=3) instead of the standard 2-way
    - part_cfg_scale = 2.0 (only InternVL-U has this parameter)
    - prompt_embeds: [3, 768, 4096], attention_mask: [3, 768] (bool)

  SAGE injection: hook generation_decoder.prepare_forward_input and
  replace the valid region of the cond row (row 0) with the enhanced embedding.
"""



import os
import sys

# Module layout: this file lives in <repo>/sage/, with the per-model
# adapters in <repo>/sage/backends/ (imported flat).
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
_BACKENDS_DIR = os.path.join(_HERE, "backends")
if os.path.isdir(_BACKENDS_DIR) and _BACKENDS_DIR not in sys.path:
    sys.path.insert(0, _BACKENDS_DIR)

import argparse
import threading
import queue
import traceback
import glob
import json

import torch
import pandas as pd
from tqdm import tqdm

# ---- Pipeline imports ----
try:
    from diffusers import ZImagePipeline
except ImportError:
    ZImagePipeline = None
try:
    from diffusers import QwenImagePipeline
except ImportError:
    QwenImagePipeline = None
try:
    from diffusers import HunyuanImagePipeline
except ImportError:
    HunyuanImagePipeline = None
try:
    from diffusers import Flux2KleinPipeline
except ImportError:
    Flux2KleinPipeline = None

# InternVL-U (not a diffusers pipeline; uses the local internvlu package).
# Lazy import: only import it when load_pipeline actually loads internvl-u, so that scripts
# for the other models do not needlessly load internvlu (and its transformers dependency chain) at import time.

from projection import projection_matrix
from run_zimage import ZImageAdapter
from run_qwen_image import QwenImageAdapter
from run_hunyuan_image import HunyuanImageAdapter
from run_flux2_klein import Flux2KleinAdapter
from run_internvlu import InternVLUAdapter
from toxic_concepts import (
    PORN_CONCEPTS,
    GORE_CONCEPTS,
    IP_CODE_TO_CONCEPT,
)


# ============================================================
# Model Registry
# ============================================================

# RevGen generation-data output root (overridable via INGUARD_OUTPUT_ROOT; see the README "Configuration" section)
_REVGEN_ROOT = os.environ.get("INGUARD_OUTPUT_ROOT", "./outputs/revgen")
# Generation-model root dir (overridable via the INGUARD_MODELS_ROOT env var; see the README "Configuration" section)
_MODELS_ROOT = os.environ.get("INGUARD_MODELS_ROOT", "/path/to/models")


def _model_path(dirname: str, hf_repo: str) -> str:
    """Prefer the local model dir; fall back to the HuggingFace repo id when absent
    (diffusers auto-downloads it into the HF cache on first use)."""
    local = os.path.join(_MODELS_ROOT, dirname)
    return local if os.path.isdir(local) else hf_repo


MODEL_REGISTRY = {
    "z-image-turbo": {
        "_model_name": "z-image-turbo",
        "pipeline_class_name": "ZImagePipeline",
        "adapter_class": ZImageAdapter,
        "model_path": _model_path("Z-Image-Turbo", "Tongyi-MAI/Z-Image-Turbo"),
        "save_dir_base": f"{_REVGEN_ROOT}/z-image-turbo/testset-seed42-1024-9steps_enhanced",
        "steps": 9,
        "width": 1024,
        "height": 1024,
        "guidance_key": "guidance_scale",
        "guidance_value": 0.0,
        # Z-Image: prompt_embeds is a list[Tensor]; no mask
        "uses_list_embeds": True,
        "needs_prompt_mask": False,       # the pipeline does not need prompt_embeds_mask
        "needs_negative_embeds": False,   # guidance_scale=0.0, no CFG
        "needs_byt5": False,
        "target_dtype_attr": "text_encoder",
        # mask save dir name (None = do not save masks)
        "mask_dir_name": None,
    },
    "qwen-image-2512": {
        "_model_name": "qwen-image-2512",
        "pipeline_class_name": "QwenImagePipeline",
        "adapter_class": QwenImageAdapter,
        "model_path": _model_path("Qwen-Image-2512", "Qwen/Qwen-Image-2512"),
        "save_dir_base": f"{_REVGEN_ROOT}/qwen-image-2512/testset-seed42-1328-10steps_enhanced",
        "steps": 10,
        "width": 1328,
        "height": 1328,
        "guidance_key": "true_cfg_scale",
        "guidance_value": 4.0,
        "uses_list_embeds": False,
        "needs_prompt_mask": True,        # the pipeline needs prompt_embeds_mask
        "needs_negative_embeds": True,     # CFG needs negative embeds
        "needs_byt5": False,
        "target_dtype_attr": "text_encoder",
        # the original data has no mask; enhanced data does not save one either
        "mask_dir_name": None,
    },
    "hunyuan-image-2_1": {
        "_model_name": "hunyuan-image-2_1",
        "pipeline_class_name": "HunyuanImagePipeline",
        "adapter_class": HunyuanImageAdapter,
        "model_path": _model_path("HunyuanImage-2.1-Diffusers",
                                  "hunyuanvideo-community/HunyuanImage-2.1-Diffusers"),
        "save_dir_base": f"{_REVGEN_ROOT}/hunyuan-image-2_1/testset-seed42-2048-10steps_enhanced",
        "steps": 10,
        "width": 2048,
        "height": 2048,
        "guidance_key": "distilled_guidance_scale",
        "guidance_value": 3.25,
        "uses_list_embeds": False,
        "needs_prompt_mask": True,
        "needs_negative_embeds": False,   # Hunyuan takes a negative_prompt string, not embeds
        "needs_byt5": True,                # ByT5 side branch
        "target_dtype_attr": "transformer",
        "mask_dir_name": "prompt_embeds_mask_forward",
    },
    "flux2-klein-base-9b": {
        "_model_name": "flux2-klein-base-9b",
        "pipeline_class_name": "Flux2KleinPipeline",
        "adapter_class": Flux2KleinAdapter,
        "model_path": _model_path("FLUX.2-klein-base-9B",
                                  "black-forest-labs/FLUX.2-klein-base-9B"),
        "save_dir_base": f"{_REVGEN_ROOT}/flux2-klein-base-9b/testset-seed42-1024-10steps_enhanced",
        "steps": 10,
        "width": 1024,
        "height": 1024,
        "guidance_key": "guidance_scale",
        "guidance_value": 4.0,
        "uses_list_embeds": False,
        "needs_prompt_mask": False,        # the pipeline does not need prompt_embeds_mask
        "needs_negative_embeds": True,
        "needs_byt5": False,
        "target_dtype_attr": "text_encoder",
        "mask_dir_name": "prompt_attention_mask_forward",
    },
    "internvl-u": {
        "_model_name": "internvl-u",
        "pipeline_class_name": "InternVLUPipeline",
        "adapter_class": InternVLUAdapter,
        "model_path": _model_path("InternVL-U", "InternVL-U/InternVL-U"),
        "save_dir_base": f"{_REVGEN_ROOT}/internvl-u/testset-seed42-1024-20steps_enhanced",
        "steps": 20,
        "width": 1024,
        "height": 1024,
        "guidance_key": "all_cfg_scale",
        "guidance_value": 4.5,
        # InternVL-U specific
        "is_internvlu": True,
        "part_cfg_scale": 2.0,
        # compatibility fields (not actually used in the InternVL-U pipeline call)
        "uses_list_embeds": False,
        "needs_prompt_mask": False,
        "needs_negative_embeds": False,
        "needs_byt5": False,
        "target_dtype_attr": "vlm",
        "mask_dir_name": "prompt_attention_mask_forward",
    },
}


# ============================================================
# Config (edit manually before running)
# ============================================================

# ★★★ Edit the parameters below manually before running ★★★
MODEL_NAME = "qwen-image-2512"
NUM_JOBS = 1
TARGET_JOB = 0
SKIP_EXISTING = True
MAX_ROWS = 0
# ★★★★★★★★★★★★★★★★★★★★★★★

CSV_PATH = "./data/test_predictions.csv"  # gating-prediction CSV (with pred_porn/pred_gore/pred_ip columns)

ALPHAS = [-0.10, -0.09, -0.08, -0.07, -0.06, -0.05, -0.04, -0.03, -0.02, -0.01, 0.0, 0.01]

SEED = 42
DEVICE = "cuda:0"
SAVE_FULL_TRACE = False
NUM_WORKERS = 3

# SAGE soft-gating temperatures
TAU_TOXIC = 0.005
TAU_SAFE = 0.02

# Gating thresholds
PORN_THRESHOLD = 3
GORE_THRESHOLD = 2
IP_CTRL_MIN = 1
IP_CTRL_MAX = 5

# Label column names
COL_PORN = "label_porn_risk_level"
COL_GORE = "label_gore_risk_level"
COL_IP = "label_ip_risk_level"


# ============================================================
# Dir preparation
# ============================================================

def prepare_sub_dirs(save_dir, steps, mask_dir_name=None, save_full_trace=False,
                       extra_base_dirs=None):
    base_dirs = ["image", "noise_init", "prompt_embeds_forward"]
    if mask_dir_name:
        base_dirs.append(mask_dir_name)
    if extra_base_dirs:
        base_dirs.extend(extra_base_dirs)
    step_dirs_minimal = ["latents_x1"]
    step_dirs_full = ["latents", "decoded_latents_x1", "velocity"]

    for d in base_dirs:
        os.makedirs(os.path.join(save_dir, d), exist_ok=True)
    for d in step_dirs_minimal:
        for i in range(steps):
            os.makedirs(os.path.join(save_dir, d, str(i)), exist_ok=True)
    if save_full_trace:
        for d in step_dirs_full:
            for i in range(steps):
                os.makedirs(os.path.join(save_dir, d, str(i)), exist_ok=True)


# ============================================================
# Auto-detect the best gating result
# ============================================================

# min_date: ignore PE-MLP runs dated before this (early trial runs with different configs)
_AUTO_CSV_MIN_DATE = "20260720"


def _compute_encoder_short(encoder_name):
    """Compute the short name from encoder_name (matches the text-MLP training layout)"""
    enc_short = encoder_name.split("/")[-1]
    enc_short = enc_short.replace("multilingual-e5", "me5")
    enc_short = enc_short.replace("bert-base-multilingual", "mbert")
    return enc_short


def _find_latest_run(ckpt_root, parent_name, stage, min_date=_AUTO_CSV_MIN_DATE):
    """
    Scan {ckpt_root}/{date}/{parent_name}/{stage}/{run_dir},
    filter by date, and take the latest run.
    """
    import re
    candidates = []
    for run_dir in glob.glob(os.path.join(ckpt_root, "*", parent_name, stage, "*")):
        if not os.path.isdir(run_dir):
            continue
        rel = os.path.relpath(run_dir, ckpt_root)
        date_part = rel.split(os.sep)[0]
        if not re.match(r"^\d{8}$", date_part) or date_part < min_date:
            continue
        candidates.append(run_dir)
    return sorted(candidates)[-1] if candidates else None


def _read_csv_candidates(candidates, **kwargs):
    """
    Try reading the candidate files in order (already sorted by priority, newest first),
    skipping empty files and corrupted files that are still being written.
    Returns: (DataFrame, file path) or (None, None)
    """
    import pandas as _pd
    for path in candidates:
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            continue
        try:
            df = _pd.read_csv(path, **kwargs)
            if len(df) > 0:
                return df, path
        except Exception as e:
            print(f"  [ReadCSV] skipping unreadable file {os.path.basename(path)}: {e}")
    return None, None


def _parse_pct(s):
    """Parse the 'xx.xx% (n/N)' format → float in 0~1"""
    try:
        return float(str(s).split("%")[0]) / 100.0
    except (ValueError, IndexError):
        return 0.0


def _find_best_step_by_gating(meta_dir, porn_t, gore_t):
    """
    Scan meta/gating_step*.csv and pick the best step at the operating point (porn_t, gore_t).
    Score = OverallRecall - FalseAlarm (higher is better).

    Returns: (best_step_int, best_score, detail_str) or (None, -1, "")
    """
    import pandas as _pd
    import re as _re

    gating_files = sorted(glob.glob(
        os.path.join(meta_dir, "gating_step*.csv")
    ))
    if not gating_files:
        return None, -1.0, ""

    best_step = None
    best_score = -1.0
    best_detail = ""

    for gf in gating_files:
        # extract the step from the file name: gating_step000123.csv → 123
        m = _re.search(r"gating_step(\d+)\.csv", os.path.basename(gf))
        if not m:
            continue
        step = int(m.group(1))

        try:
            df = _pd.read_csv(gf)
        except Exception:
            continue

        row = df[(df["porn_T"] == porn_t) & (df["gore_T"] == gore_t)]
        if row.empty:
            continue
        r = row.iloc[0]

        recall = _parse_pct(r.get("OverallRecall", "0"))
        fa = _parse_pct(r.get("FalseAlarm", "0"))
        hit = _parse_pct(r.get("HitRate", "0"))
        score = recall - fa

        if score > best_score:
            best_score = score
            best_step = step
            best_detail = (
                f"recall={recall:.4f}, fa={fa:.4f}, hit={hit:.4f}, "
                f"score={score:.4f}"
            )

    return best_step, best_score, best_detail


def _find_best_step_by_ckpt(run_dir):
    """
    Scan the checkpoints under run_dir/ckpts/ and pick the step with the highest val acc_avg.
    Prefer scanning best_prompt_model_step*.pth (fewer files, pre-filtered by acc_avg),
    falling back to ckpt_step*.pth.

    Returns: (best_step_int, best_val_acc_avg, detail_str) or (None, None, "")
    """
    import torch as _torch
    import re as _re

    ckpts_dir = os.path.join(run_dir, "ckpts")
    if not os.path.isdir(ckpts_dir):
        return None, None, ""

    # prefer best_prompt_model_step*.pth (fewer files, pre-filtered by acc_avg)
    ckpt_files = sorted(glob.glob(
        os.path.join(ckpts_dir, "best_prompt_model_step*.pth")
    ))
    source_tag = "best_prompt_model"

    if not ckpt_files:
        # fall back to ckpt_step*.pth
        ckpt_files = sorted(glob.glob(
            os.path.join(ckpts_dir, "ckpt_step*.pth")
        ))
        source_tag = "ckpt"

    if not ckpt_files:
        return None, None, ""

    best_step = None
    best_acc = None

    for cf in ckpt_files:
        m = _re.search(r"step(\d+)\.pth$", os.path.basename(cf))
        if not m:
            continue
        step = int(m.group(1))

        try:
            payload = _torch.load(cf, map_location="cpu", weights_only=False)
            acc = payload.get("val_acc_avg", payload.get("acc_avg", None))
            if acc is None:
                continue
            acc = float(acc)
        except Exception as e:
            print(f"  [CkptScan] skipping {os.path.basename(cf)}: {e}")
            continue

        if best_acc is None or acc > best_acc:
            best_acc = acc
            best_step = step

    if best_step is None:
        return None, None, ""

    detail = f"val_acc_avg={best_acc:.6f} ({source_tag})"
    return best_step, best_acc, detail


def find_best_predictions(model_name, ckpt_root, pred_source="prompt", encoder_name=None):
    """
    Auto-search the output dirs of the PE-MLP training runs (prompt / text sources),
    take the latest training run (no cross-run acc_avg comparison),
    and within that run prefer the final version, falling back to the newest step version when absent.

    Search path patterns:
      prompt: {ckpt_root}/{date}/{model_name}/prompt-mlp-multitask-porn-gore-ip/{run}/
      text:   {ckpt_root}/{date}/{encoder_short}/text-mlp-multitask-porn-gore-ip/{run}/

    Returns: (csv_path, run_dir, best_acc_avg, best_step) or None
      - best_acc_avg / best_step: read from history.json (only present once training finished); logging only
    """
    if pred_source == "prompt":
        parent_name = model_name
        stage = "prompt-mlp-multitask-porn-gore-ip"
    elif pred_source == "text":
        if encoder_name is None:
            raise ValueError("--pred_source text requires --encoder_name")
        parent_name = _compute_encoder_short(encoder_name)
        stage = "text-mlp-multitask-porn-gore-ip"
    else:
        raise ValueError(f"unknown pred_source: {pred_source}")

    # 1. take the latest run dir
    run_dir = _find_latest_run(ckpt_root, parent_name, stage)
    if not run_dir:
        print(f"[AutoCSV] training dir not found: {parent_name}/{stage}")
        return None

    meta_dir = os.path.join(run_dir, "meta")
    print(f"[AutoCSV] latest training run: {run_dir}")

    # 2. prefer the final version; when absent, use gating_step*.csv to pick the best step
    final_csv = os.path.join(meta_dir, "test_predictions.csv")
    if os.path.exists(final_csv) and os.path.getsize(final_csv) > 0:
        # training finished; final_best was already picked by train_prompt.py via acc_avg
        used_path = final_csv
        print(f"[AutoCSV] using final_best: test_predictions.csv")

        # read history.json for best_acc / best_step (logging only)
        history_path = os.path.join(meta_dir, "history.json")
        best_acc = -1.0
        best_step = -1
        if os.path.exists(history_path):
            try:
                with open(history_path, "r") as f:
                    history = json.load(f)
                acc_avg_list = history.get("acc_avg", [])
                step_list = history.get("step", [])
                if acc_avg_list:
                    best_idx = max(range(len(acc_avg_list)), key=lambda i: acc_avg_list[i])
                    best_acc = float(acc_avg_list[best_idx])
                    if best_idx < len(step_list):
                        best_step = int(step_list[best_idx])
            except Exception as e:
                print(f"  [WARN] failed to read history.json: {e}")
    else:
        # training unfinished; use gating_step*.csv to pick the best step at the operating point
        print(f"[AutoCSV] final_best not found, scanning gating_step*.csv for the best step...")
        best_step, best_score, gating_detail = _find_best_step_by_gating(
            meta_dir, PORN_THRESHOLD, GORE_THRESHOLD
        )
        if best_step is None:
            # no gating CSV either; use checkpoint val acc_avg to pick the best step
            print(f"[AutoCSV] no gating CSV, scanning checkpoints for the highest val acc_avg...")
            best_step, best_acc_ckpt, ckpt_detail = _find_best_step_by_ckpt(run_dir)
            if best_step is not None:
                used_path = os.path.join(
                    meta_dir, f"test_predictions_step{best_step:06d}.csv"
                )
                if not os.path.exists(used_path) or os.path.getsize(used_path) == 0:
                    print(f"[AutoCSV] test_predictions for step={best_step} not found, falling back to the newest step")
                    best_step = None  # proceed with the newest-step fallback
                else:
                    best_acc = -1.0
                    print(f"[AutoCSV] best step={best_step} ({ckpt_detail})")
                    print(f"[AutoCSV] using: {os.path.basename(used_path)}")
            if best_step is None:
                # no checkpoint either, or its csv is missing; finally fall back to the newest step version
                step_csvs = sorted(
                    glob.glob(os.path.join(meta_dir, "test_predictions_step*.csv")),
                    reverse=True,
                )
                df_tmp, used_path = _read_csv_candidates(step_csvs, dtype={"id": str})
                if used_path is None:
                    print(f"[AutoCSV] no readable test_predictions*.csv in this run")
                    return None
                best_acc = -1.0
                print(f"[AutoCSV] no usable checkpoint, using the newest step version: {os.path.basename(used_path)}")
        else:
            used_path = os.path.join(
                meta_dir, f"test_predictions_step{best_step:06d}.csv"
            )
            if not os.path.exists(used_path) or os.path.getsize(used_path) == 0:
                # the corresponding test_predictions is missing; fall back to the newest
                step_csvs = sorted(
                    glob.glob(os.path.join(meta_dir, "test_predictions_step*.csv")),
                    reverse=True,
                )
                df_tmp, used_path = _read_csv_candidates(step_csvs, dtype={"id": str})
                if used_path is None:
                    print(f"[AutoCSV] no readable test_predictions*.csv in this run")
                    return None
            best_acc = best_score  # use the gating score in place of acc_avg for logging
            print(f"[AutoCSV] best step={best_step} ({gating_detail})")
            print(f"[AutoCSV] using: {os.path.basename(used_path)}")

    return used_path, run_dir, best_acc, best_step


# ============================================================
# Data loading
# ============================================================

def load_csv(csv_path, col_porn, col_gore, col_ip,
             max_rows=0, shuffle=False, shuffle_seed=42):
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str).str.replace("\x00", "", regex=False)

    required = ["id", "prompt", col_porn, col_gore, col_ip]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing columns: {missing}; actual columns: {df.columns.tolist()}")

    df = df.drop_duplicates(subset=["id"])
    df = df.dropna(subset=["id", "prompt"])

    if shuffle:
        df = df.sample(frac=1, random_state=shuffle_seed).reset_index(drop=True)
        print(f"  CSV shuffled (seed={shuffle_seed})")

    if max_rows > 0:
        df = df.head(max_rows)
        print(f"  limiting to the first {max_rows} rows")

    n_benign = int(
        ((df[col_porn] == 0) & (df[col_gore] == 0) & (df[col_ip] == 0)).sum()
    )
    print(f"  Total: {len(df)} (benign={n_benign}, risky={len(df) - n_benign})")
    return df.reset_index(drop=True)


# ============================================================
# Pipeline / VAE
# ============================================================

def load_pipeline(model_name, model_path, device):
    cfg = MODEL_REGISTRY[model_name]
    cls_name = cfg["pipeline_class_name"]

    # InternVL-U (not a diffusers pipeline; uses the custom InternVLUPipeline)
    if cls_name == "InternVLUPipeline":
        try:
            from internvlu import InternVLUPipeline
        except Exception as _e:
            raise ImportError(
                f"InternVLUPipeline unavailable (internvlu package import failed: {type(_e).__name__}: {_e})"
            ) from _e
        print(f"[{model_name}] Loading InternVLUPipeline from {model_path}...")
        pipe = InternVLUPipeline.from_pretrained(
            model_path, torch_dtype=torch.bfloat16
        )
        pipe.to(device)
        return pipe

    cls = {
        "ZImagePipeline": ZImagePipeline,
        "QwenImagePipeline": QwenImagePipeline,
        "HunyuanImagePipeline": HunyuanImagePipeline,
        "Flux2KleinPipeline": Flux2KleinPipeline,
    }[cls_name]
    if cls is None:
        raise ImportError(f"{cls_name} unavailable; please check the diffusers version")
    print(f"[{model_name}] Loading {cls_name} from {model_path}...")
    pipe = cls.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=False
    )
    pipe.to(device)
    return pipe


def get_target_dtype(pipe, model_name):
    attr = MODEL_REGISTRY[model_name]["target_dtype_attr"]
    return getattr(pipe, attr).dtype


# ============================================================
# FLUX2-Klein latent unpack helpers
# (consistent with the implementation in flux2_klein_base_9b_save_data_revgen.py)
# ============================================================

def unpack_latents_with_ids(x, x_ids, height=None, width=None):
    """Standalone version of Flux2KleinPipeline._unpack_latents_with_ids"""
    x_list = []
    for data, pos in zip(x, x_ids):
        _, ch = data.shape
        h_ids = pos[:, 1].to(torch.int64)
        w_ids = pos[:, 2].to(torch.int64)
        h = height if height is not None else int(torch.max(h_ids).item()) + 1
        w = width if width is not None else int(torch.max(w_ids).item()) + 1
        flat_ids = h_ids * w + w_ids
        out = torch.zeros((h * w, ch), device=data.device, dtype=data.dtype)
        out.scatter_(0, flat_ids.unsqueeze(1).expand(-1, ch), data)
        out = out.view(h, w, ch).permute(2, 0, 1)
        x_list.append(out)
    return torch.stack(x_list, dim=0)


def unpatchify_latents(latents):
    """Standalone version of Flux2KleinPipeline._unpatchify_latents"""
    batch_size, num_channels, height, width = latents.shape
    latents = latents.reshape(batch_size, num_channels // 4, 2, 2, height, width)
    latents = latents.permute(0, 1, 4, 2, 5, 3)
    latents = latents.reshape(batch_size, num_channels // 4, height * 2, width * 2)
    return latents


def full_unpack_latents(packed, latent_ids, bn_mean, bn_std, latent_h, latent_w):
    """packed [B, num_patches, C] → unpacked [B, 32, H*2, W*2]"""
    x = unpack_latents_with_ids(packed, latent_ids, latent_h, latent_w)
    x = x * bn_std + bn_mean
    x = unpatchify_latents(x)
    return x


_flux_bn_stats_cache = {}


def get_flux_bn_stats(pipe, height, width):
    """Extract BN statistics from the FLUX2-Klein pipeline (cached, computed once)."""
    key = (height, width)
    if key not in _flux_bn_stats_cache:
        bn_mean = pipe.vae.bn.running_mean.view(1, -1, 1, 1).cpu().float()
        bn_std = torch.sqrt(
            pipe.vae.bn.running_var.view(1, -1, 1, 1) + pipe.vae.config.batch_norm_eps
        ).cpu().float()
        latent_h = 2 * (int(height) // (pipe.vae_scale_factor * 2)) // 2
        latent_w = 2 * (int(width) // (pipe.vae_scale_factor * 2)) // 2
        _flux_bn_stats_cache[key] = (bn_mean, bn_std, latent_h, latent_w)
    return _flux_bn_stats_cache[key]


def decode_latents_to_pil(pipe, latents_4d):
    """Generic VAE decode, compatible with VAEs with/without shift_factor."""
    with torch.no_grad():
        latents_4d = latents_4d.to(pipe.device, dtype=pipe.vae.dtype)
        if hasattr(pipe.vae.config, "shift_factor") and pipe.vae.config.shift_factor is not None:
            latents_4d = (latents_4d / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
        elif hasattr(pipe.vae.config, "scaling_factor"):
            latents_4d = latents_4d / pipe.vae.config.scaling_factor
        image = pipe.vae.decode(latents_4d, return_dict=False)[0]
        image = pipe.image_processor.postprocess(image, output_type="pil")
        return image[0]


# ============================================================
# Post-processing worker
# ============================================================

def post_process_worker(pipe, process_queue, steps, save_full_trace):
    while True:
        item = process_queue.get()
        if item is None:
            process_queue.task_done()
            break

        save_dir = item["save_dir"]
        event_id = item["event_id"]
        final_img = item["final_img"]
        prompt_embeds = item["prompt_embeds"]
        prompt_embeds_mask = item.get("prompt_embeds_mask")
        noise_init = item["noise_init"]
        latent_ids = item.get("latent_ids")
        latents_all_steps = item["latents_all_steps"]
        velocities_all_steps = item["velocities_all_steps"]
        sigmas = item["sigmas"]
        # info needed for unpacking
        model_name = item.get("model_name", "")
        img_height = item.get("height", 0)
        img_width = item.get("width", 0)

        try:
            sigmas_path = os.path.join(save_dir, "sigmas.pth")
            if not os.path.exists(sigmas_path):
                torch.save(sigmas, sigmas_path)

            final_img.save(os.path.join(save_dir, "image", f"{event_id}.jpg"), quality=95)

            if prompt_embeds is not None:
                torch.save(prompt_embeds,
                           os.path.join(save_dir, "prompt_embeds_forward", f"{event_id}.pth"))

            if prompt_embeds_mask is not None:
                mask_dir = os.path.join(save_dir, item["mask_dir_name"])
                os.makedirs(mask_dir, exist_ok=True)
                torch.save(prompt_embeds_mask,
                           os.path.join(mask_dir, f"{event_id}.pth"))

            if noise_init is not None:
                torch.save(noise_init,
                           os.path.join(save_dir, "noise_init", f"{event_id}.pth"))

            # FLUX2-Klein: save latent_ids (consistent with the original save_data)
            if latent_ids is not None:
                torch.save(latent_ids,
                           os.path.join(save_dir, "latent_ids", f"{event_id}.pth"))

            for i in range(len(velocities_all_steps)):
                current_xt = noise_init if i == 0 else latents_all_steps[i - 1]
                v = velocities_all_steps[i]
                sigma_t = sigmas[i]
                latent_x1 = current_xt - sigma_t * v

                # ---- model-specific unpack ----
                # Qwen-Image-2512: the pipeline internally uses packed [1, HW, C*4];
                # the original LDM_safety collection code calls pipe._unpack_latents to expand it into 5D [1, C, 1, H, W].
                # The enhancement code must do the same unpack, otherwise the loading side reports a shape mismatch.
                if model_name == "qwen-image-2512" and latent_x1.ndim == 3:
                    latent_x1 = pipe._unpack_latents(
                        latent_x1, img_height, img_width, pipe.vae_scale_factor)

                # FLUX2-Klein: packed [B, num_patches, C] → unpacked [B, 32, H*2, W*2]
                # (consistent with full_unpack_latents in flux2_klein_base_9b_save_data_revgen.py)
                if model_name == "flux2-klein-base-9b" and latent_x1.ndim == 3:
                    bn_mean, bn_std, lh, lw = get_flux_bn_stats(pipe, img_height, img_width)
                    latent_x1 = full_unpack_latents(
                        latent_x1.float(), latent_ids, bn_mean, bn_std, lh, lw
                    )

                torch.save(latent_x1,
                           os.path.join(save_dir, "latents_x1", str(i), f"{event_id}.pth"))

                if not save_full_trace:
                    continue
                torch.save(latents_all_steps[i],
                           os.path.join(save_dir, "latents", str(i), f"{event_id}.pth"))
                torch.save(v,
                           os.path.join(save_dir, "velocity", str(i), f"{event_id}.pth"))
                # InternVL-U accesses the VAE differently (pipe.image_pipeline); skip full-trace decoding
                if not hasattr(pipe, "image_pipeline"):
                    img_x1 = decode_latents_to_pil(pipe, latent_x1)
                    img_x1.save(
                        os.path.join(save_dir, "decoded_latents_x1", str(i), f"{event_id}.jpg"),
                        quality=90,
                    )

        except Exception as e:
            print(f"[Worker Error] event={event_id}: {e}")
            traceback.print_exc()
        finally:
            process_queue.task_done()


def start_workers(pipe, process_queue, steps, save_full_trace, num_workers=NUM_WORKERS):
    workers = []
    for i in range(num_workers):
        t = threading.Thread(
            target=post_process_worker,
            args=(pipe, process_queue, steps, save_full_trace),
            name=f"Worker-{i}", daemon=True,
        )
        t.start()
        workers.append(t)
    return workers


def stop_workers(process_queue, workers):
    for _ in workers:
        process_queue.put(None)
    process_queue.join()
    for t in workers:
        t.join()


# ============================================================
# Core generation function (generalized; supports 4 models)
# ============================================================

def generate_with_hooks(
    pipe,
    model_cfg,
    prompt_embeds_input,      # list[Tensor] (Z-Image) or Tensor[1,L,D] (others)
    prompt_mask,              # [1,L] or None
    event_id,
    save_dir,
    device,
    process_queue,
    prompt_embeds_to_save,    # list[Tensor] or Tensor (CPU)
    prompt_mask_to_save,     # Tensor or None
    byt5_embeds=None,         # Hunyuan: (tensor, mask) or None
    negative_embeds=None,    # Qwen/FLUX: (tensor, mask_or_None) or None
    steps=10,
    width=1024,
    height=1024,
    guidance_value=4.0,
    seed=42,
    save_full_trace=False,
):
    captured_init = {}
    latents_all_steps = []
    velocities_all_steps = []

    original_prepare_latents = pipe.prepare_latents

    def hooked_prepare_latents(*args, **kwargs):
        ret = original_prepare_latents(*args, **kwargs)
        # FLUX2-Klein's prepare_latents returns a tuple (latents, latent_ids)
        lats = ret[0] if isinstance(ret, (tuple, list)) else ret
        captured_init["noise_init"] = lats.detach().cpu()
        # FLUX2-Klein: also capture latent_ids (used later by full_unpack_latents)
        if isinstance(ret, (tuple, list)) and len(ret) >= 2:
            captured_init["latent_ids"] = ret[1].detach().cpu()
        return ret

    pipe.prepare_latents = hooked_prepare_latents

    original_step = pipe.scheduler.step

    def hooked_step(model_output, timestep, sample, **kwargs):
        velocities_all_steps.append(model_output.detach().cpu())
        return original_step(model_output, timestep, sample, **kwargs)

    pipe.scheduler.step = hooked_step

    # ---- callback: latents capture ----
    tensor_inputs = ["latents"]

    def combined_callback(pipe_obj, step_index, timestep, callback_kwargs):
        latents = callback_kwargs.get("latents")
        if latents is not None:
            latents_all_steps.append(latents.detach().cpu())
        return callback_kwargs

    generator = torch.Generator(device).manual_seed(seed)
    target_dtype = get_target_dtype(pipe, model_cfg.get("_model_name", ""))

    # ---- build model-specific call_kwargs ----
    call_kwargs = dict(
        prompt=None,
        width=width,
        height=height,
        num_inference_steps=steps,
        generator=generator,
        callback_on_step_end=combined_callback,
        callback_on_step_end_tensor_inputs=tensor_inputs,
    )

    if model_cfg["uses_list_embeds"]:
        # Z-Image: prompt_embeds = list[Tensor]
        call_kwargs["prompt_embeds"] = prompt_embeds_input
        call_kwargs["negative_prompt_embeds"] = None
        call_kwargs[model_cfg["guidance_key"]] = guidance_value
    elif model_cfg["needs_prompt_mask"]:
        # Qwen / Hunyuan: prompt_embeds + mask
        call_kwargs["prompt_embeds"] = prompt_embeds_input.to(dtype=target_dtype)
        call_kwargs["prompt_embeds_mask"] = prompt_mask.to(dtype=torch.long)
        call_kwargs[model_cfg["guidance_key"]] = guidance_value
        if model_cfg["needs_byt5"]:
            # Hunyuan: ByT5 side branch
            byt5_e, byt5_m = byt5_embeds
            call_kwargs["prompt_embeds_2"] = byt5_e.to(dtype=target_dtype)
            call_kwargs["prompt_embeds_mask_2"] = byt5_m.to(dtype=torch.long)
            call_kwargs["negative_prompt"] = None
        if model_cfg["needs_negative_embeds"]:
            # Qwen: negative embeds + mask
            neg_e, neg_m = negative_embeds
            call_kwargs["negative_prompt_embeds"] = neg_e.to(dtype=target_dtype)
            call_kwargs["negative_prompt_embeds_mask"] = neg_m.to(dtype=torch.long)
    else:
        # FLUX: prompt_embeds + negative_prompt_embeds
        call_kwargs["prompt_embeds"] = prompt_embeds_input.to(dtype=target_dtype)
        neg_e, _ = negative_embeds
        call_kwargs["negative_prompt_embeds"] = neg_e.to(dtype=target_dtype)
        call_kwargs[model_cfg["guidance_key"]] = guidance_value

    try:
        output = pipe(**call_kwargs)
        data_to_save = {
            "save_dir": save_dir,
            "event_id": event_id,
            "final_img": output.images[0],
            "prompt_embeds": prompt_embeds_to_save,
            "prompt_embeds_mask": prompt_mask_to_save,
            "mask_dir_name": model_cfg.get("mask_dir_name"),
            "noise_init": captured_init.get("noise_init"),
            "latent_ids": captured_init.get("latent_ids"),
            "latents_all_steps": latents_all_steps,
            "velocities_all_steps": velocities_all_steps,
            "sigmas": pipe.scheduler.sigmas.detach().cpu(),
            # model info needed for unpacking (decide per model whether to unpack before saving the latent)
            "model_name": model_cfg.get("_model_name", ""),
            "height": height,
            "width": width,
        }
        process_queue.put(data_to_save)
    finally:
        pipe.prepare_latents = original_prepare_latents
        pipe.scheduler.step = original_step


# ============================================================
# InternVL-U-specific generation + data collection
# ============================================================

def generate_internvlu_with_hooks(
    pipe,
    model_cfg,
    safe_emb,               # [1, 768, 4096] — the SAGE-enhanced cond embedding
    prompt,                 # str — the original prompt (the pipeline does its own 3-way encoding)
    event_id,
    save_dir,
    device,
    process_queue,
    steps=20,
    width=1024,
    height=1024,
    all_cfg_scale=4.5,
    part_cfg_scale=2.0,
    seed=42,
    save_full_trace=False,
):
    """
    InternVL-U-specific generation function.

    Hook strategy (consistent with internvlu_save_data_revgen.py):
      1. hook image_pipeline.prepare_latents → capture noise_init
      2. hook generation_decoder.prepare_forward_input → inject safe_emb + capture prompt_embeds
      3. hook scheduler.step → capture velocity
      4. patch InternVLUDiffusionPipeline.__call__ → inject a callback (latents capture)

    Pipeline call: pipe(prompt=..., generation_mode="image", ...)
    Inside the pipeline: processor(3 variants) → VLM forward → _prepare_diffusion_inputs
                  → image_pipeline(encoder_hidden_states, ...)
                    → prepare_forward_input [hook: inject safe_emb into row 0]
                    → 20-step denoising (triple CFG)
    """
    from internvlu.diffusion.pipeline_internvlu_generation_decoder import (
        InternVLUDiffusionPipeline,
    )

    img_pipe = pipe.image_pipeline
    gen_decoder = img_pipe.generation_decoder

    captured = {
        "noise_init": None,
        "prompt_embeds": None,         # [3, 768, 4096]
        "prompt_attention_mask": None,  # [3, 768]
        "cond_valid_len": 0,
    }
    latents_all_steps = []
    velocities_all_steps = []

    # --- Hook 1: prepare_latents → capture noise_init ---
    original_prepare_latents = img_pipe.prepare_latents

    def hooked_prepare_latents(*args, **kwargs):
        ret = original_prepare_latents(*args, **kwargs)
        # FLUX2-Klein's prepare_latents returns a tuple (latents, latent_ids)
        lats = ret[0] if isinstance(ret, (tuple, list)) else ret
        captured["noise_init"] = lats.detach().cpu()
        return ret

    img_pipe.prepare_latents = hooked_prepare_latents

    # --- Hook 2: prepare_forward_input → inject safe_emb (cond row 0) + capture ---
    original_prepare_forward = gen_decoder.prepare_forward_input

    def hooked_prepare_forward(encoder_hidden_states, **kw):
        result = original_prepare_forward(encoder_hidden_states, **kw)
        enc_hs, attn_mask, img_mask = result
        # enc_hs: [3, 768, 4096], attn_mask: [3, 768]

        # capture the full 3-way batch (including the safe-corrected cond row)
        captured["prompt_embeds"] = enc_hs.detach().cpu()
        captured["prompt_attention_mask"] = attn_mask.detach().cpu()

        # replace the valid region of the cond variant (row 0)
        valid_len = int(attn_mask[0].sum())
        captured["cond_valid_len"] = valid_len
        enc_hs[0, :valid_len, :] = safe_emb[0, :valid_len, :]

        return enc_hs, attn_mask, img_mask

    gen_decoder.prepare_forward_input = hooked_prepare_forward

    # --- Hook 3: scheduler.step → capture velocity ---
    original_scheduler_step = img_pipe.scheduler.step

    def hooked_scheduler_step(model_output, timestep, sample, **kwargs):
        velocities_all_steps.append(model_output.detach().cpu())
        return original_scheduler_step(model_output, timestep, sample, **kwargs)

    img_pipe.scheduler.step = hooked_scheduler_step

    # --- Hook 4: patch __call__ → inject a callback (latents capture) ---
    original_diffusion_call = InternVLUDiffusionPipeline.__call__

    def patched_diffusion_call(self_dp, *args, **kw):
        def _callback(pipe_obj, step_index, timestep, cb_kwargs):
            # capture the latents at each step
            if "latents" in cb_kwargs and cb_kwargs["latents"] is not None:
                latents_all_steps.append(cb_kwargs["latents"].detach().cpu())
            return cb_kwargs

        kw["callback_on_step_end"] = _callback
        kw["callback_on_step_end_tensor_inputs"] = ["latents"]
        return original_diffusion_call(self_dp, *args, **kw)

    InternVLUDiffusionPipeline.__call__ = patched_diffusion_call

    # --- run inference ---
    generator = torch.Generator(device=device).manual_seed(seed)

    try:
        output = pipe(
            prompt=prompt,
            generation_mode="image",
            num_inference_steps=steps,
            all_cfg_scale=all_cfg_scale,
            part_cfg_scale=part_cfg_scale,
            height=height,
            width=width,
            generator=generator,
        )

        sigmas = img_pipe.scheduler.sigmas.detach().cpu()

        data_to_save = {
            "save_dir": save_dir,
            "event_id": event_id,
            "final_img": output.images[0],
            "prompt_embeds": captured["prompt_embeds"],
            "prompt_embeds_mask": captured["prompt_attention_mask"],
            "mask_dir_name": model_cfg.get("mask_dir_name"),
            "noise_init": captured["noise_init"],
            "latents_all_steps": latents_all_steps,
            "velocities_all_steps": velocities_all_steps,
            "sigmas": sigmas,
            "model_name": model_cfg.get("_model_name", ""),
            "height": height,
            "width": width,
        }
        process_queue.put(data_to_save)
    finally:
        img_pipe.prepare_latents = original_prepare_latents
        gen_decoder.prepare_forward_input = original_prepare_forward
        img_pipe.scheduler.step = original_scheduler_step
        InternVLUDiffusionPipeline.__call__ = original_diffusion_call


# ============================================================
# SAGE concept utilities
# ============================================================

def precompute_concept_vectors(adapter, concept_groups=None):
    """Pre-encode concept vectors.

    Args:
        adapter: the model adapter
        concept_groups: optional, dict with keys "porn", "gore", "ip".
            - "porn" / "gore": list[str]
            - "ip": dict {code: list[str]}
            defaults to None, which uses the built-in Chinese-English concept table.
    """
    if concept_groups is None:
        porn_concepts = PORN_CONCEPTS
        gore_concepts = GORE_CONCEPTS
        ip_map = IP_CODE_TO_CONCEPT
    else:
        porn_concepts = concept_groups["porn"]
        gore_concepts = concept_groups["gore"]
        ip_map = concept_groups["ip"]

    print(f"[SAGE] pre-encoding concept vectors (porn={len(porn_concepts)}, gore={len(gore_concepts)}, ip={len(ip_map)}) ...")
    result = {"porn": [], "gore": [], "ip": {}}
    for c in porn_concepts:
        v = adapter.encode_pooled_concept(c)
        if v is not None:
            result["porn"].append(v.float())
    for c in gore_concepts:
        v = adapter.encode_pooled_concept(c)
        if v is not None:
            result["gore"].append(v.float())
    for code, concept_list in ip_map.items():
        vecs = []
        for c in concept_list:
            v = adapter.encode_pooled_concept(c)
            if v is not None:
                vecs.append(v.float())
        if vecs:
            result["ip"][code] = vecs
    n_ip = sum(len(v) for v in result["ip"].values())
    print(f"  porn={len(result['porn'])}, gore={len(result['gore'])}, ip={n_ip} (across {len(result['ip'])} codes)")
    return result


def select_concept_vectors(concept_vecs, porn_level, gore_level, ip_code,
                           concept_names=None):
    """Select the concept vectors of the matching category based on the prompt labels.

    Args:
        concept_names: optional, dict with keys "porn", "gore", "ip".
            Used for log output; defaults to None, which uses the built-in Chinese-English concept table.
    """
    if concept_names is None:
        porn_names = PORN_CONCEPTS
        gore_names = GORE_CONCEPTS
        ip_map = IP_CODE_TO_CONCEPT
    else:
        porn_names = concept_names["porn"]
        gore_names = concept_names["gore"]
        ip_map = concept_names["ip"]

    selected = []
    names = []
    if porn_level > 0:
        selected.extend(concept_vecs["porn"])
        names.extend(porn_names)
    if gore_level > 0:
        selected.extend(concept_vecs["gore"])
        names.extend(gore_names)
    if IP_CTRL_MIN <= ip_code <= IP_CTRL_MAX and ip_code in concept_vecs["ip"]:
        # the ip value is always a list[tensor]
        selected.extend(concept_vecs["ip"][ip_code])
        names.extend(ip_map.get(ip_code, []))
    if not selected:
        return None, None
    return selected, names


def build_P_C(selected_vecs, device):
    C = torch.stack(selected_vecs, dim=0)
    P_C = projection_matrix(C.T)
    dim = P_C.shape[0]
    I_minus_Pc = torch.eye(dim, device=device, dtype=P_C.dtype) - P_C
    return P_C, I_minus_Pc


# ============================================================
# SAGE main flow (generalized)
# ============================================================

def run_sage(adapter, pipe, df, args, model_cfg):
    model_name = args.model_name
    print(f"\n{'=' * 60}")
    print(f"Starting SAGE (model={model_name}), alphas={args.alphas}")
    print(f"  tau_toxic={args.tau_toxic}, tau_safe={args.tau_safe}")
    print(f"  concepts: mixed Chinese and English")
    print(f"{'=' * 60}")

    # concept grouping (mixed Chinese and English)
    concept_groups = {
        "porn": PORN_CONCEPTS,
        "gore": GORE_CONCEPTS,
        "ip": IP_CODE_TO_CONCEPT,
    }
    concept_vecs = precompute_concept_vectors(adapter, concept_groups=concept_groups)
    target_dtype = get_target_dtype(pipe, model_name)

    # pre-compute negative embeds
    negative_embeds_data = None
    if model_cfg["needs_negative_embeds"]:
        if model_name == "qwen-image-2512":
            neg_e, neg_m = adapter._get_negative_embeds("")
            negative_embeds_data = (
                neg_e.to(dtype=target_dtype, device=args.device),
                neg_m.to(dtype=torch.long, device=args.device),
            )
        elif model_name == "flux2-klein-base-9b":
            neg_e = adapter._get_negative_embeds("")
            negative_embeds_data = (
                neg_e.to(dtype=target_dtype, device=args.device),
                None,
            )

    alpha_dirs = {}
    for alpha in args.alphas:
        alpha_str = f"{alpha:.2f}"
        sd = os.path.join(args.save_dir, "sage", f"alpha_{alpha_str}")
        extra_dirs = ["latent_ids"] if model_name == "flux2-klein-base-9b" else None
        prepare_sub_dirs(sd, args.num_inference_steps, model_cfg["mask_dir_name"],
                         SAVE_FULL_TRACE, extra_base_dirs=extra_dirs)
        # FLUX2-Klein: save bn_stats.pth
        if model_name == "flux2-klein-base-9b":
            bn_stats_path = os.path.join(sd, "bn_stats.pth")
            if not os.path.exists(bn_stats_path):
                bn_mean, bn_std, lh, lw = get_flux_bn_stats(pipe, args.height, args.width)
                torch.save({"bn_mean": bn_mean, "bn_std": bn_std,
                            "latent_h": lh, "latent_w": lw}, bn_stats_path)
        alpha_dirs[alpha] = sd

    process_queue = queue.Queue(maxsize=50)
    workers = start_workers(pipe, process_queue, args.num_inference_steps, SAVE_FULL_TRACE)

    n_benign = 0
    n_risky = 0
    n_skipped = 0

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="sage"):
        if idx % args.num_jobs != args.target_job:
            continue

        event_id = str(row["id"])
        prompt = str(row["prompt"])
        porn_level = int(row[args.col_porn])
        gore_level = int(row[args.col_gore])
        ip_code = int(row[args.col_ip])
        is_flagged = (porn_level >= PORN_THRESHOLD or
                      gore_level >= GORE_THRESHOLD or
                      IP_CTRL_MIN <= ip_code <= IP_CTRL_MAX)

        if not is_flagged:
            n_benign += 1
            continue

        if args.skip_existing:
            all_done = all(
                os.path.exists(os.path.join(alpha_dirs[a], "image", f"{event_id}.jpg"))
                for a in args.alphas
            )
            if all_done:
                n_skipped += 1
                continue

        n_risky += 1

        try:
            # ---- alpha-independent pre-computation ----
            full_emb_raw, user_mask = adapter.encode_full_prompt(prompt)
            orig_dtype_emb = full_emb_raw.dtype
            full_emb = full_emb_raw.float()
            user_mask = user_mask.bool()

            if user_mask.sum().item() == 0:
                print(f"  [{event_id}] WARN: no user tokens, skip")
                continue

            selected_vecs, selected_names = select_concept_vectors(
                concept_vecs, porn_level, gore_level, ip_code,
                concept_names=concept_groups,
            )
            if selected_vecs is None:
                print(f"  [{event_id}] WARN: no concept vectors, skip")
                continue

            P_C, I_minus_Pc = build_P_C(selected_vecs, args.device)
            original = full_emb[user_mask]
            n_tokens = original.shape[0]

            p_masked = adapter.encode_pooled_masked_per_token(prompt).float()
            if p_masked.shape[0] != n_tokens:
                m = min(p_masked.shape[0], n_tokens)
                p_masked = p_masked[:m]
                if m < n_tokens:
                    n_tokens = m
                    original = original[:m]

            P_I = projection_matrix(p_masked.T)
            dist_vec = I_minus_Pc @ p_masked.T
            dist = torch.norm(dist_vec, dim=0)

            means = []
            for i in range(n_tokens):
                others = torch.cat((dist[:i], dist[i + 1:]))
                means.append(others.mean() if others.numel() > 0 else dist[i])
            mean_dist = torch.stack(means)
            ratio = dist / (mean_dist + 1e-8)

            new_text_e = (I_minus_Pc @ P_I @ original.T).T  # [N, D]

            # ---- per-prompt mask & byt5 (alpha-independent) ----
            prompt_mask_for_pipe = None
            if model_cfg["needs_prompt_mask"]:
                raw_mask = adapter.get_last_attn_mask()
                if raw_mask is not None:
                    prompt_mask_for_pipe = raw_mask.to(dtype=torch.long, device=args.device)

            prompt_mask_to_save = None
            if model_cfg["mask_dir_name"]:
                raw_mask = adapter.get_last_attn_mask()
                if raw_mask is not None:
                    prompt_mask_to_save = raw_mask.detach().cpu()

            byt5_embeds = None
            if model_cfg["needs_byt5"]:
                byt5_e, byt5_m = adapter._build_byt5_embeds(adapter._last_prompt)
                byt5_embeds = (
                    byt5_e.to(dtype=target_dtype, device=args.device),
                    byt5_m.to(dtype=torch.long, device=args.device),
                )

            # ---- for each alpha ----
            for alpha in args.alphas:
                alpha_str = f"{alpha:.2f}"
                save_dir = alpha_dirs[alpha]

                if args.skip_existing:
                    img_path = os.path.join(save_dir, "image", f"{event_id}.jpg")
                    if os.path.exists(img_path):
                        continue

                # asymmetric soft gating
                delta = ratio - (1.0 + alpha)
                tau_eff = torch.where(
                    delta >= 0,
                    torch.full_like(delta, args.tau_toxic),
                    torch.full_like(delta, args.tau_safe),
                )
                gamma = torch.sigmoid(delta / tau_eff)

                merged = (1.0 - gamma.unsqueeze(1)) * original + gamma.unsqueeze(1) * new_text_e
                safe_full = full_emb.clone()
                safe_full[user_mask] = merged
                safe_emb = safe_full.to(orig_dtype_emb).unsqueeze(0)  # [1, L, D]

                # ---- InternVL-U-specific path ----
                if model_cfg.get("is_internvlu"):
                    safe_emb_gpu = safe_emb.to(dtype=target_dtype, device=args.device)
                    n_trigger = int((gamma > 0.5).sum().item())
                    print(f"  [{event_id}] a={alpha_str} N={n_tokens} trg={n_trigger} "
                          f"g_mean={gamma.mean():.3f}")
                    generate_internvlu_with_hooks(
                        pipe=pipe,
                        model_cfg=model_cfg,
                        safe_emb=safe_emb_gpu,
                        prompt=prompt,
                        event_id=event_id,
                        save_dir=save_dir,
                        device=args.device,
                        process_queue=process_queue,
                        steps=args.num_inference_steps,
                        width=args.width,
                        height=args.height,
                        all_cfg_scale=args.guidance_scale,
                        part_cfg_scale=model_cfg["part_cfg_scale"],
                        seed=args.seed,
                        save_full_trace=SAVE_FULL_TRACE,
                    )
                    continue

                # convert to the model-required format
                if model_cfg["uses_list_embeds"]:
                    safe_emb_gpu = safe_emb.to(dtype=target_dtype, device=args.device)
                    safe_emb_list = adapter._to_list_by_mask(safe_emb_gpu)
                    prompt_embeds_to_save = [t.detach().cpu() for t in safe_emb_list]
                    prompt_embeds_input = safe_emb_list
                else:
                    safe_emb_gpu = safe_emb.to(dtype=target_dtype, device=args.device)
                    prompt_embeds_to_save = safe_emb_gpu.detach().cpu()
                    prompt_embeds_input = safe_emb_gpu

                n_trigger = int((gamma > 0.5).sum().item())
                print(f"  [{event_id}] a={alpha_str} N={n_tokens} trg={n_trigger} "
                      f"g_mean={gamma.mean():.3f}")

                generate_with_hooks(
                    pipe=pipe,
                    model_cfg=model_cfg,
                    prompt_embeds_input=prompt_embeds_input,
                    prompt_mask=prompt_mask_for_pipe,
                    event_id=event_id,
                    save_dir=save_dir,
                    device=args.device,
                    process_queue=process_queue,
                    prompt_embeds_to_save=prompt_embeds_to_save,
                    prompt_mask_to_save=prompt_mask_to_save,
                    byt5_embeds=byt5_embeds,
                    negative_embeds=negative_embeds_data,
                    steps=args.num_inference_steps,
                    width=args.width,
                    height=args.height,
                    guidance_value=args.guidance_scale,
                    seed=args.seed,
                    save_full_trace=SAVE_FULL_TRACE,
                )

        except Exception as e:
            print(f"[ERROR] id={event_id}: {type(e).__name__}: {e}")
            traceback.print_exc()
            continue

    print(f"\n  waiting for the save queue to drain (remaining: {process_queue.qsize()})...")
    stop_workers(process_queue, workers)
    print(f"\n  SAGE done: risky={n_risky}, benign={n_benign} (skip), skipped={n_skipped}")


# ============================================================
# main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Multi-model SAGE enhancement data collection (prompt_embeds + latent_x1)"
    )
    parser.add_argument("--model_name", default=MODEL_NAME,
                        choices=list(MODEL_REGISTRY.keys()),
                        help="generation model name")
    parser.add_argument("--model_path", default=None,
                        help="model path (defaults to the path in MODEL_REGISTRY)")
    parser.add_argument("--csv_path", default=CSV_PATH)
    parser.add_argument("--save_dir", default=None,
                        help="save dir (defaults to the path in MODEL_REGISTRY)")
    parser.add_argument("--alphas", type=str, default=None,
                        help="comma-separated alpha values")
    parser.add_argument("--num_jobs", type=int, default=NUM_JOBS)
    parser.add_argument("--target_job", type=int, default=TARGET_JOB)
    parser.add_argument("--max_rows", type=int, default=MAX_ROWS)
    parser.add_argument("--skip_existing", action="store_true", default=SKIP_EXISTING)
    parser.add_argument("--shuffle", action="store_true", default=True)
    parser.add_argument("--no-shuffle", dest="shuffle", action="store_false")
    parser.add_argument("--shuffle_seed", type=int, default=42)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--width", type=int, default=None,
                        help="image width (defaults to the model config)")
    parser.add_argument("--height", type=int, default=None,
                        help="image height (defaults to the model config)")
    parser.add_argument("--num_inference_steps", type=int, default=None,
                        help="inference steps (defaults to the model config)")
    parser.add_argument("--guidance_scale", type=float, default=None,
                        help="guidance strength (defaults to the model config)")
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--col_porn", default=COL_PORN)
    parser.add_argument("--col_gore", default=COL_GORE)
    parser.add_argument("--col_ip", default=COL_IP)
    # SAGE soft-gating temperatures
    parser.add_argument("--tau_toxic", type=float, default=TAU_TOXIC)
    parser.add_argument("--tau_safe", type=float, default=TAU_SAFE)
    # auto-detect the gating result
    parser.add_argument("--ckpt_root", type=str,
                        default="./outputs/checkpoints",
                        help="gating-model checkpoint root dir")
    parser.add_argument("--auto_csv", action="store_true", default=False,
                        help="auto-detect the best gating result (replaces --csv_path)")
    parser.add_argument("--pred_source", type=str, default="prompt",
                        choices=["prompt", "text"],
                        help="gating-prediction source: prompt=train_prompt.py, text=a text-encoder variant not shipped in this repo")
    parser.add_argument("--encoder_name", type=str, default=None,
                        help="text encoder name (required when --pred_source text)")
    args = parser.parse_args()

    # apply model default config
    model_cfg = dict(MODEL_REGISTRY[args.model_name])
    model_cfg["_model_name"] = args.model_name

    if args.model_path is None:
        args.model_path = model_cfg["model_path"]
    if args.save_dir is None:
        args.save_dir = model_cfg["save_dir_base"]
    if args.width is None:
        args.width = model_cfg["width"]
    if args.height is None:
        args.height = model_cfg["height"]
    if args.num_inference_steps is None:
        args.num_inference_steps = model_cfg["steps"]
    if args.guidance_scale is None:
        args.guidance_scale = model_cfg["guidance_value"]

    args.alphas = (
        [float(x) for x in args.alphas.split(",")] if args.alphas else list(ALPHAS)
    )

    # make sure save_dir exists (os.makedirs may not take effect for deeply nested OSS paths; create it explicitly once first)
    os.makedirs(args.save_dir, exist_ok=True)

    # ---- auto-detect the best gating result ----
    if args.auto_csv:
        result = find_best_predictions(
            model_name=args.model_name,
            ckpt_root=args.ckpt_root,
            pred_source=args.pred_source,
            encoder_name=args.encoder_name,
        )
        if result is None:
            raise FileNotFoundError(
                f"no training run found for {args.model_name} (ckpt_root={args.ckpt_root})"
            )
        csv_path, run_dir, best_acc, best_step = result
        args.csv_path = csv_path
        args.col_porn = "pred_porn"
        args.col_gore = "pred_gore"
        args.col_ip = "pred_ip"
        acc_str = f"{best_acc:.4f}" if best_acc >= 0 else "N/A"
        print(f"[AutoCSV] using training run: {run_dir}")
        print(f"  best_step={best_step}, best_acc_avg={acc_str}")
        print(f"  csv={csv_path}")
        print(f"  col_porn={args.col_porn}, col_gore={args.col_gore}, col_ip={args.col_ip}")

    print(f"Model: {args.model_name}")
    print(f"Alphas: {args.alphas}")
    print(f"Save dir: {args.save_dir}")
    print(f"Device: {args.device}")
    print(f"Steps: {args.num_inference_steps}, Seed: {args.seed}")
    print(f"Resolution: {args.width}x{args.height}")
    print(f"Guidance: {model_cfg['guidance_key']}={args.guidance_scale}")
    print(f"Uses list embeds: {model_cfg['uses_list_embeds']}")
    print(f"Needs prompt mask: {model_cfg['needs_prompt_mask']}")
    print(f"Needs negative embeds: {model_cfg['needs_negative_embeds']}")
    print(f"Needs ByT5: {model_cfg['needs_byt5']}")
    print(f"Mask dir name: {model_cfg['mask_dir_name']}")

    # load models
    pipe = load_pipeline(args.model_name, args.model_path, args.device)

    if model_cfg.get("is_internvlu"):
        adapter = model_cfg["adapter_class"](
            pipe,
            device=args.device,
            height=args.height,
            width=args.width,
            num_inference_steps=args.num_inference_steps,
            all_cfg_scale=args.guidance_scale,
            part_cfg_scale=model_cfg["part_cfg_scale"],
            seed=args.seed,
        )
        # save the VAE config (consistent with internvlu_save_data_revgen.py)
        vae_config_path = os.path.join(args.save_dir, "vae_config.pth")
        if not os.path.exists(vae_config_path):
            torch.save({
                "latents_mean": pipe.image_pipeline.vae.config.latents_mean,
                "latents_std": pipe.image_pipeline.vae.config.latents_std,
                "vae_scale_factor": pipe.image_pipeline.vae_scale_factor,
                "latent_channels": pipe.image_pipeline.latent_channels,
            }, vae_config_path)
            print(f"Saved VAE config to {vae_config_path}")
    else:
        adapter = model_cfg["adapter_class"](pipe, device=args.device)

    # load data
    df = load_csv(
        args.csv_path, args.col_porn, args.col_gore, args.col_ip,
        args.max_rows, args.shuffle, args.shuffle_seed,
    )

    # run
    run_sage(adapter, pipe, df, args, model_cfg)

    print("\nAll done.")


if __name__ == "__main__":
    main()
