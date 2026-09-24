#!/usr/bin/env python3
"""Export all guardrail weights to the fixed layout used by this repository.

Run this where the training checkpoints live (it only reads checkpoints and
copies files; no GPU required, but `torch` must be importable to inspect
PE-MLP payloads).

Usage:
    python scripts/export_weights.py \
        --ckpt_root ./outputs/checkpoints \
        --export_dir /path/to/inner-guardrail/weights

Output layout (consumed by integration/guardrail_pipeline.py):

    weights/
    ├── latent_detector/
    │   ├── z-image-turbo/
    │   │   ├── model.pth                     # D-group: distill pretrain + finetune
    │   │   └── config.json                   # inference-only fields
    │   ├── qwen-image-2512/ ...
    │   ├── hunyuan-image-2_1/ ...
    │   ├── flux2-klein-base-9b/ ...
    │   └── internvl-u/ ...
    ├── pe_mlp/
    │   ├── z-image-turbo/
    │   │   └── model.pth                     # best_prompt_model_stepXXXXXX.pth (val-selected)
    │   ├── qwen-image-2512/ ...
    │   ├── hunyuan-image-2_1/ ...
    │   ├── flux2-klein-base-9b/ ...
    │   └── internvl-u/ ...
    └── manifest.json                          # status + checkpoint file of each model

Selection criteria:
    Latent detector: D-group run (latent + stage-wise distillation pretrain +
                     finetune, 50k pool), checkpoint best_val_avg_f1_model.pth
                     (highest validation avg-F1).
    PE-MLP:         latest run only (run-selection rule `find | sort | tail -1`
                    over run dirs named YYYYMMDD_HHMMSS_*, i.e. the
                    chronologically latest run), then the eval step with the
                    highest validation acc_avg WITHIN that run (criterion
                    `max_val_acc_avg`). The step is resolved via
                    meta/history.json (fast), then train_prompt.log, then a
                    torch.load scan (slow, last resort). At that step we take
                    best_prompt_model_step{N}.pth if present, else
                    ckpt_step{N}.pth — the two files hold the identical
                    payload; best_prompt_model_* is only written when val
                    acc_avg improves, while ckpt_step* is written at every eval.

Checkpoint layout written by the training scripts (under --ckpt_root):
    Latent: {ckpt_root}/{date}/{model}/imagelatent-convnext_base-multitask-porn-gore-ip/
            {tag}/ckpts/best_val_avg_f1_model.pth
            tag = {P}{B}D-v51-224-ft-{PT}-{B}-D-224-distill-full-stage-wise-pool50k
    PE-MLP: {ckpt_root}/{date}/{model}/prompt-mlp-multitask-porn-gore-ip/
            {run}/ckpts/best_prompt_model_step{:06d}.pth
"""

import argparse
import glob
import json
import os
import re
import shutil
import sys

# torch is only needed to read PE-MLP payloads; the latent-detector part of
# this script is torch-free. Import lazily-but-tolerantly so that running the
# latent export in a torch-less environment still works.
try:
    import torch
except ImportError:
    torch = None

MODELS = [
    "z-image-turbo",
    "qwen-image-2512",
    "hunyuan-image-2_1",
    "flux2-klein-base-9b",
    "internvl-u",
]

# model prefix / pretrain tag (run-dir naming used by the training scripts)
MODEL_INFO = {
    "z-image-turbo":       ("Z", "ZP"),
    "qwen-image-2512":     ("Q", "QP"),
    "internvl-u":          ("I", "QP"),
    "hunyuan-image-2_1":   ("H", "HP"),
    "flux2-klein-base-9b": ("F", "FP"),
}


def build_d_group_tag(model_name, backbone="convnext_base", pool="50k"):
    """D-group (latent + distill pretrain) run tag."""
    prefix, ptag = MODEL_INFO[model_name]
    bb = "B" if backbone == "convnext_base" else "T"
    return f"{prefix}{bb}D-v51-224-ft-{ptag}-{bb}-D-224-distill-full-stage-wise-pool{pool}"


def find_latent_ckpt(ckpt_root, model_name, backbone="convnext_base",
                     pool="50k", min_date="20260720"):
    """Locate the D-group latent detector checkpoint. Returns
    (ckpt_path, config_path, run_dir) or (None, None, None)."""
    stage = f"imagelatent-{backbone}-multitask-porn-gore-ip"
    tag = build_d_group_tag(model_name, backbone, pool)
    print(f"  [find] stage={stage}")
    print(f"  [find] pattern={tag}")

    candidates = []
    for run_dir in glob.glob(os.path.join(ckpt_root, "*", model_name, stage, "*")):
        if not os.path.isdir(run_dir):
            continue
        date_part = os.path.relpath(run_dir, ckpt_root).split(os.sep)[0]
        if not date_part.isdigit() or date_part < min_date:
            continue
        if tag not in os.path.basename(run_dir):
            continue
        candidates.append(run_dir)

    if not candidates:
        return None, None, None
    run_dir = sorted(candidates)[-1]

    ckpt_path = os.path.join(run_dir, "ckpts", "best_val_avg_f1_model.pth")
    if not os.path.exists(ckpt_path):
        # Fall back to the model_epoch_*.pth with the largest epoch number.
        # Epoch files are unpadded (model_epoch_{epoch}.pth), so a plain
        # lexicographic sort would rank epoch 10 below epoch 3; extract the
        # number and take the numeric max (matches the Stage1 resolver in
        # latent_detector/train_image.py).
        epoch_ckpts = glob.glob(
            os.path.join(run_dir, "ckpts", "model_epoch_*.pth"))
        if epoch_ckpts:
            def _epoch_num(fp):
                m = re.search(r"model_epoch_(\d+)\.pth$", fp)
                return int(m.group(1)) if m else -1
            ckpt_path = max(epoch_ckpts, key=_epoch_num)
        else:
            return None, None, None

    config_path = os.path.join(run_dir, "meta", "config_snapshot.json")
    if not os.path.exists(config_path):
        config_path = os.path.join(run_dir, "meta", "config.json")
    return ckpt_path, (config_path if os.path.exists(config_path) else None), run_dir


def _read_history_best(run_dir):
    """Tier 1: meta/history.json -> (best_step, max_val_acc_avg) or None.

    train_prompt.py saves history.json after every eval as parallel lists
    {"step": [...], "train_loss": [...], "val_loss": [...], "acc_avg": [...], ...};
    acc_avg is the validation average accuracy that drives model selection.
    """
    p = os.path.join(run_dir, "meta", "history.json")
    if not os.path.exists(p):
        return None
    try:
        with open(p) as f:
            h = json.load(f)
        steps, accs = h.get("step", []), h.get("acc_avg", [])
        if not steps or len(steps) != len(accs):
            return None
        pairs = [(int(s), float(a)) for s, a in zip(steps, accs)
                 if isinstance(a, (int, float))]
        if not pairs:
            return None
        return max(pairs, key=lambda x: x[1])
    except Exception:
        return None


# eval lines in train_prompt.log look like:
#   [Epoch 1 step 2534] step=2534 | train_loss=0.9822 val_loss=0.8066 | ... | val_avg=85.20%
_EVAL_LINE_RE = re.compile(
    r"step=(\d+)\s*\|.*val_avg=([\d.]+)%")


def _read_log_best(run_dir):
    """Tier 2: parse train_prompt.log eval lines -> (best_step, max_val_acc_avg) or None."""
    p = os.path.join(run_dir, "train_prompt.log")
    if not os.path.exists(p):
        return None
    best = None
    try:
        with open(p, errors="replace") as f:
            for line in f:
                m = _EVAL_LINE_RE.search(line)
                if not m:
                    continue
                step, acc = int(m.group(1)), float(m.group(2)) / 100.0
                if best is None or acc > best[1]:
                    best = (step, acc)
    except Exception:
        return None
    return best


def _resolve_ckpt_for_step(run_dir, step):
    """Checkpoint file for eval step N.

    best_prompt_model_step{N}.pth is saved only when acc_avg improves, while
    ckpt_step{N}.pth is saved at every eval step; the two carry the identical
    ckpt_payload. Prefer the former, fall back to the latter.
    """
    for name in (f"best_prompt_model_step{step:06d}.pth",
                 f"ckpt_step{step:06d}.pth"):
        p = os.path.join(run_dir, "ckpts", name)
        if os.path.exists(p):
            return p
    return None


def find_pe_mlp_best(ckpt_root, model_name, min_date="20260701"):
    """Locate the PE-MLP checkpoint with the highest validation acc_avg.

    Run selection (`find ... | sort | tail -1`): among all runs with date >= min_date, use
    ONLY the lexicographically latest run dir (run names start with
    YYYYMMDD_HHMMSS_, so lexicographic order == chronological order).
    max(val acc_avg) is then resolved WITHIN that run, never across runs —
    older runs are typically early tries (different setup / fewer evals),
    and a cross-run global max could silently pick a non-deployed model.

    Resolution chain (fast to slow; the first tier reads a KB-size JSON
    instead of loading hundreds of MB of checkpoints):

        1. meta/history.json -> max(acc_avg) step -> that step's checkpoint
        2. train_prompt.log eval lines -> max(val_avg) step -> ditto
        3. torch.load scan over best_prompt_model_step*.pth / ckpt_step*.pth
           (only when both metadata sources are missing or broken)

    Returns (ckpt_path, best_step, val_acc_avg, run_dir, source, n_scanned)
    or (None, None, None, None, None, n_scanned).
    """
    stage = "prompt-mlp-multitask-porn-gore-ip"
    run_dirs = []
    for run_dir in glob.glob(os.path.join(ckpt_root, "*", model_name, stage, "*")):
        if not os.path.isdir(run_dir):
            continue
        date_part = os.path.relpath(run_dir, ckpt_root).split(os.sep)[0]
        if not date_part.isdigit() or date_part < min_date:
            continue
        run_dirs.append(run_dir)

    if not run_dirs:
        # diagnose: show what actually exists for this model
        stages = sorted({
            os.path.basename(os.path.dirname(d))
            for d in glob.glob(os.path.join(ckpt_root, "*", model_name, "prompt*", "*"))
            if os.path.isdir(d)
        })
        dates = sorted({
            os.path.relpath(d, ckpt_root).split(os.sep)[0]
            for d in glob.glob(os.path.join(ckpt_root, "*", model_name, stage, "*"))
            if os.path.isdir(d)
        })
        print(f"  [diagnose] no run dir matched {{ckpt_root}}/*/{model_name}/{stage}/*"
              f" with date >= {min_date}")
        if dates:
            print(f"  [diagnose] the stage exists under dates: {dates} "
                  f"(try a lower --pe_mlp_min_date)")
        elif stages:
            print(f"  [diagnose] other prompt-* stages exist for {model_name}: {stages}")
        else:
            print(f"  [diagnose] no prompt-* stage dir exists under "
                  f"{{ckpt_root}}/*/{model_name}/ at all")
        return None, None, None, None, None, 0

    # ---- run selection (rule: sort | tail -1) ----
    run_dirs = sorted(run_dirs)
    latest_run = run_dirs[-1]
    if len(run_dirs) > 1:
        print(f"  [runs] {len(run_dirs)} runs >= {min_date} found; "
              f"scanning ONLY the latest (rule: sort | tail -1):")
        for r in run_dirs:
            print(f"    {'->' if r == latest_run else ' .'} "
                  f"{os.path.basename(r)}")

    best = (None, None, None, None, None)  # (ckpt, step, loss, run_dir, source)
    n_scanned = 0
    for run_dir in (latest_run,):
        run_name = os.path.basename(run_dir)

        # ---- Tier 1/2: metadata ----
        got, source = _read_history_best(run_dir), "history.json"
        if got is None:
            got, source = _read_log_best(run_dir), "train_prompt.log"
        if got is not None:
            step, val_acc = got
            ckpt = _resolve_ckpt_for_step(run_dir, step)
            if ckpt is not None:
                n_scanned += 1
                print(f"  [meta] {run_name}: best step {step} "
                      f"(val_acc_avg={val_acc:.4f}) via {source}")
                if best[2] is None or val_acc > best[2]:
                    best = (ckpt, step, val_acc, run_dir, source)
                continue
            print(f"  [warn] {run_name}: best step {step} has no checkpoint "
                  f"on disk; falling back to scan")

        # ---- Tier 3: torch.load scan (rare) ----
        if torch is None:
            print(f"  [warn] {run_name}: metadata missing and torch not "
                  f"importable; skipping this run")
            continue
        ckpts_dir = os.path.join(run_dir, "ckpts")
        if not os.path.isdir(ckpts_dir):
            continue
        files = sorted(glob.glob(
            os.path.join(ckpts_dir, "best_prompt_model_step*.pth")))
        if not files:
            files = sorted(glob.glob(
                os.path.join(ckpts_dir, "ckpt_step*.pth")))
        if files:
            print(f"  [scan] {len(files)} ckpt(s) in {run_name} "
                  f"(slow path: torch.load each)")
        for cf in files:
            m = re.search(r"step(\d+)\.pth$", os.path.basename(cf))
            if not m:
                continue
            step = int(m.group(1))
            payload, last_err = None, None
            # mmap first (lazy tensor pages - big speedup on OSS mounts,
            # torch>=2.1 zipfile ckpts), then plain weights_only, then full
            # load. Payloads are plain dicts of tensors + scalars written by
            # our own train_prompt.py, so weights_only is safe to try first.
            for kwargs in ({"mmap": True, "weights_only": True},
                           {"weights_only": True},
                           {"weights_only": False}):
                try:
                    payload = torch.load(cf, map_location="cpu", **kwargs)
                    break
                except Exception as e:
                    last_err = e
            if payload is None:
                print(f"  [skip] {cf}: {last_err}")
                continue
            if not isinstance(payload, dict):
                continue
            val_acc = payload.get("val_acc_avg", payload.get("acc_avg"))
            if val_acc is None:
                continue
            n_scanned += 1
            if best[2] is None or float(val_acc) > best[2]:
                best = (cf, step, float(val_acc), run_dir, "torch-scan")
    return (*best, n_scanned) if best[0] else (None, None, None, None, None, n_scanned)


def export_latent_detector(ckpt_root, export_dir, backbone, pool, min_date):
    print("\n" + "=" * 60)
    print("=== Latent detector (D-group: distill pretrain + finetune) ===")
    print("=" * 60)
    out_root = os.path.join(export_dir, "latent_detector")
    manifest = {}
    for model in MODELS:
        print(f"\n--- {model} ---")
        ckpt_path, config_path, run_dir = find_latent_ckpt(
            ckpt_root, model, backbone, pool, min_date)
        if ckpt_path is None:
            print("  [MISS] no D-group checkpoint found")
            manifest[model] = {"status": "missing"}
            continue

        print(f"  run:   {run_dir}")
        print(f"  ckpt:  {ckpt_path}")

        model_dir = os.path.join(out_root, model)
        os.makedirs(model_dir, exist_ok=True)

        dst = os.path.join(model_dir, "model.pth")
        shutil.copy2(ckpt_path, dst)
        print(f"  [OK]   -> {dst}")

        config = {
            "backbone": backbone,
            "latent_stem_type": "single",
            "latent_stem_init": "expand_in1k",
            "num_porn_classes": 2,
            "num_gore_classes": 2,
            "latent_stretch_target_hw": None,
            "latent_preprocess_style": "stretch",
            "model_name": model,
            "training_pool": pool,
            "training_method": "Latent + distill pretrain (stage-wise) + finetune",
        }
        if config_path:
            with open(config_path) as f:
                full = json.load(f)
            for k in ("latent_in_chans", "latent_stem_type", "latent_stem_init",
                      "num_porn_classes", "num_gore_classes",
                      "latent_stretch_target_hw", "latent_preprocess_style",
                      "backbone"):
                if full.get(k) is not None:
                    config[k] = full[k]
        if config.get("latent_in_chans") is None:
            print("  [WARN] latent_in_chans not in training config; "
                  "inference will fall back to the per-model default table")

        config_dst = os.path.join(model_dir, "config.json")
        with open(config_dst, "w") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        print(f"  [OK]   -> {config_dst}")

        manifest[model] = {
            "status": "ok",
            "checkpoint": "model.pth",
            "latent_in_chans": config.get("latent_in_chans"),
        }
    return manifest


def export_pe_mlp(ckpt_root, export_dir, min_date):
    print("\n" + "=" * 60)
    print("=== PE-MLP prompt risk classifier (max val acc_avg) ===")
    print("=" * 60)
    out_root = os.path.join(export_dir, "pe_mlp")
    manifest = {}
    if torch is None:
        print("\n[ERROR] PE-MLP export needs `torch` to read checkpoint payloads,")
        print("        but torch is not importable in the current environment.")
        print("        Activate a conda env with torch (e.g. the training env), then re-run:")
        print("            python scripts/export_weights.py --only pe_mlp")
        for model in MODELS:
            manifest[model] = {"status": "error",
                               "reason": "torch not importable in this environment"}
        return manifest
    for model in MODELS:
        print(f"\n--- {model} ---")
        ckpt_path, best_step, val_acc, run_dir, source, n = find_pe_mlp_best(
            ckpt_root, model, min_date)
        if ckpt_path is None:
            print(f"  [MISS] no PE-MLP checkpoint found ({n} payload(s) scanned)")
            manifest[model] = {"status": "missing", "scanned": n}
            continue

        print(f"  run:       {run_dir}")
        print(f"  best step: {best_step}  (val_acc_avg={val_acc:.4f}, {n} scanned)")

        model_dir = os.path.join(out_root, model)
        os.makedirs(model_dir, exist_ok=True)
        dst = os.path.join(model_dir, "model.pth")
        shutil.copy2(ckpt_path, dst)
        print(f"  [OK]       -> {dst}")

        manifest[model] = {
            "status": "ok",
            "checkpoint": "model.pth",
        }
    return manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt_root",
                    default=os.environ.get("INGUARD_CKPT_ROOT", "./outputs/checkpoints"),
                    help="training checkpoint root "
                         "(env override: INGUARD_CKPT_ROOT)")
    ap.add_argument("--export_dir", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "weights"),
        help="output weights/ directory of this repository")
    ap.add_argument("--backbone", default="convnext_base")
    ap.add_argument("--pool", default="50k")
    ap.add_argument("--latent_min_date", default="20260720",
                    help="min date dir for latent detector runs")
    ap.add_argument("--pe_mlp_min_date", default="20260701",
                    help="min date dir (YYYYMMDD) for PE-MLP runs to consider")
    ap.add_argument("--only", choices=["latent", "pe_mlp"], default=None,
                    help="export only one component "
                         "(e.g. --only pe_mlp re-runs just the PE-MLP export; "
                         "results merge into the existing manifest.json)")
    args = ap.parse_args()

    print(f"ckpt_root:  {args.ckpt_root}")
    print(f"export_dir: {args.export_dir}")
    os.makedirs(args.export_dir, exist_ok=True)

    manifest = {}
    if args.only in (None, "latent"):
        manifest["latent_detector"] = export_latent_detector(
            args.ckpt_root, args.export_dir, args.backbone, args.pool,
            args.latent_min_date)
    if args.only in (None, "pe_mlp"):
        manifest["pe_mlp"] = export_pe_mlp(
            args.ckpt_root, args.export_dir, args.pe_mlp_min_date)

    manifest_path = os.path.join(args.export_dir, "manifest.json")
    merged = {}
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path) as f:
                merged = json.load(f)
        except Exception:
            merged = {}
    merged.update(manifest)   # --only runs merge into the existing manifest
    with open(manifest_path, "w") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)
    print("\n" + "=" * 60)
    print(f"manifest -> {manifest_path}")

    ok, missing = [], []
    for comp, models in merged.items():
        for m, info in models.items():
            (ok if info.get("status") == "ok" else missing).append(f"{comp}/{m}")
    if missing:
        print("\n[ATTENTION] exported OK:")
        for m in ok:
            print(f"  + {m}")
        print("\n[ATTENTION] missing / failed components:")
        for m in missing:
            print(f"  - {m}")
        sys.exit(1)
    print(f"All components exported ({len(ok)} ok).")


if __name__ == "__main__":
    main()
