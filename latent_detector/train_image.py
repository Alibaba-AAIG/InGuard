"""
train_image.py — multi-task safety classification training script for images / image latents

Supports four input_mode values:
  image  : read JPG/PNG image files directly -> 3-channel RGB
           → MiddleFrameMultiTaskModel (ConvNeXt/ResNet/ViT, torchvision IN1K_V1)
  latent : read diffusion-model image latent .pth -> latent_in_chans channels
           -> ConvNeXt/ResNet/ViT go through MultiTaskWrapperCNN (stem/conv_proj channel-adapted)
  latent_to_image : read latent .pth -> run the VAE decoder online during training to decode into RGB -> resize + normalize -> backbone -> heads
  latent_to_image_feat : read latent .pth -> VAE decoder early-stop taking intermediate features -> per-ch normalize -> CNN backbone -> heads

Label format, sampler, loss, and evaluation metrics are identical to the video training script;
training-loop functions are imported from train_utils.py for reuse (zero duplicated code).
"""

import os
import sys
import csv
import random
import datetime
import warnings
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

warnings.filterwarnings("ignore")

# --- repo layout: shared training utilities live in <repo>/common/ ---
import os as _os, sys as _sys
_common = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "common")
if _common not in _sys.path:
    _sys.path.insert(0, _common)
from utils_common import (
    Logger,
    collate_fn,
    seed_everything,
    save_json,
    save_history_and_plots,
    print_table,
    print_kv_table,
    compute_binary_metrics,
    empty_metrics,
    snapshot_code_dir,
)
from data_pipeline import (
    IP_ID2NAME,
    load_predictions_csv,
    load_id_to_category,
    print_final_dataset_summary,
    build_image_datasets,
    build_image_dataloaders,
    compute_image_latent_stats,
    load_latent_stats_from_cache,
    split_val_fixed_per_class,
    remove_train_overlap_with_test,
    check_prompt_testset_leak,
    generate_previews,
    compute_image_decoded_feat_stats,
    load_decoded_feat_stats_from_cache,
)
from models_zoo import (
    MiddleFrameMultiTaskModel,
    MultiTaskWrapperCNN,
    build_optimizer,
    make_load_info,
    _build_latent_to_image_model,
    _build_latent_to_image_feat_model,
    VALID_BACKBONES,
    get_backbone_family,
)

# common training utilities
from train_utils import (
    build_task_masks_from_train_groups,
    masked_cross_entropy_loss,
    init_history,
    append_metrics,
    append_epoch_history,
    train_one_epoch,
    evaluate,
    save_badcases,
    save_epoch_artifacts,
    update_best_checkpoints,
    build_lr_scheduler,
    prepare_output_dirs,
    setup_logger,
    run_smoke_test,
    append_test_metrics_csv,
    _epoch_summary_row,
    _EPOCH_SUMMARY_HEADERS,
    print_model_load_summary,
    print_pe_mlp_load_summary,
)


# =========================
# Data Loading (image-specific)
# =========================
def load_and_split_image_data(config, ckpt_dir):
    """Read the image CSV, deduplicate, split train/val, and generate sample previews.
    """
    input_mode = config.get("input_mode", "image")
    label_mode = "3class" if int(config.get("num_porn_classes", 2)) == 3 else "binary"
    # {id: category} maps from the prompt dataset CSVs (testset.csv/trainset.csv),
    # used for borderline bucketing when ids are opaque/prefix-less; empty config
    # -> None -> decide_train_group falls back to the filename keyword.
    train_cat = load_id_to_category(config.get("train_prompt_csv", ""))
    test_cat = load_id_to_category(config.get("test_prompt_csv", ""))
    train_all = load_predictions_csv(
        config["train_csv"], "train_all",
        input_mode=input_mode, file_type="image", verbose=False,
        label_mode=label_mode, id_to_category=train_cat,
    )
    test_list = load_predictions_csv(
        config["test_csv"], "test",
        input_mode=input_mode, file_type="image", verbose=False,
        label_mode=label_mode, id_to_category=test_cat,
    )
    if not train_all:
        raise RuntimeError("No valid train data loaded.")
    if not test_list:
        raise RuntimeError("No valid test data loaded.")

    # ---- leak filtering ----
    # Train/test leakage is already removed at the source: the released trainset.csv drops
    # every sample matching the test set (prompt Jaccard + image pHash), so no runtime
    # filtering is needed. The two online checks below stay available as optional extra
    # defenses (both off by default).
    # (1) remove_train_overlap_with_test (exact file_path dedup) — off by default
    # (2) check_prompt_testset_leak (online prompt text matching) — off by default
    if config.get("remove_train_test_overlap", False):
        train_all = remove_train_overlap_with_test(train_all, test_list, verbose=False)
        if not train_all:
            raise RuntimeError("No valid train data after dedup.")

    if config.get("check_testset_leak", False):
        train_all = check_prompt_testset_leak(train_all, config, ckpt_dir=ckpt_dir, verbose=True)
        if not train_all:
            raise RuntimeError("No valid train data after prompt leak removal.")

    train_list, val_list = split_val_fixed_per_class(
        train_all, val_per_class=config["val_per_class"], seed=config["seed"],
    )

    for name, data in [("train_set.json", train_list),
                       ("val_set.json",   val_list),
                       ("test_set.json",  test_list)]:
        save_json(data, os.path.join(ckpt_dir, "meta", name))

    # generate sample previews (copy image files into previews/ grouped by train_group / IP)
    generate_previews(config, train_list, test_list, ckpt_dir)

    return train_all, train_list, val_list, test_list


# =========================
# Model Builder (image-specific)
# =========================
def build_image_model_optimizer_criterion(config, device):
    """
    input_mode='image'  + CNN/ViT backbone : MiddleFrameMultiTaskModel (3-channel RGB, IN1K pretrained)
    input_mode='latent' + CNN/ViT backbone : MultiTaskWrapperCNN (latent_in_chans channels; stem/conv_proj widened)
    """
    mode     = config.get("input_mode", "image")
    backbone = str(config.get("backbone", "convnext_base"))
    pretrain = bool(config.get("backbone_pretrained", True))
    dropout  = float(config.get("dropout", 0.1))
    freeze   = bool(config.get("freeze_backbone", False))
    num_porn = int(config.get("num_porn_classes", 2))
    num_gore = int(config.get("num_gore_classes", 2))

    # prompt fusion config
    prompt_fusion_cfg = None
    if config.get("enable_prompt_fusion", False):
        prompt_fusion_cfg = {
            "text_dim": config.get("prompt_fusion_text_dim", 2560),
            "dropout": config.get("prompt_fusion_dropout", 0.3),
            "fusion_mode": config.get("prompt_fusion_mode", "concat"),
            "scale_init": config.get("prompt_fusion_scale_init", 0.1),
            # PE2-specific config
            "pe2_mlp_ckpt_path": config.get("pe2_mlp_ckpt_path", ""),
            "pe2_mlp_hidden_dim": config.get("pe2_mlp_hidden_dim", 1024),
            "pe2_mlp_num_layers": config.get("pe2_mlp_num_layers", 3),
            "pe2_mlp_dropout": config.get("pe2_mlp_dropout", 0.1),
        }

    _VALID_BACKBONES = VALID_BACKBONES
    if backbone not in _VALID_BACKBONES:
        raise ValueError(
            f"train_image.py backbone={backbone!r} is not supported.\n"
            f"Supported: {sorted(_VALID_BACKBONES)}"
        )

    if mode == "image":
        # reuse MiddleFrameMultiTaskModel directly (3-channel RGB input, IN1K pretrained)
        model = MiddleFrameMultiTaskModel(
            backbone_name=backbone, pretrained=pretrain,
            dropout=dropout, freeze_backbone=freeze,
            prompt_fusion_cfg=prompt_fusion_cfg,
            num_porn_classes=num_porn, num_gore_classes=num_gore,
        )
        load_info = make_load_info(
            checkpoint_path=config.get("model_ckpt") or None,
            loaded=pretrain,
            note=f"image mode, backbone={backbone}, pretrained={pretrain}",
        )
        # optional: load from a custom ckpt
        ckpt_path = config.get("model_ckpt") or None
        if ckpt_path:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            sd = ckpt.get("state_dict", ckpt)
            sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
            ms = model.state_dict()
            filtered = {k: v for k, v in sd.items()
                        if k in ms and v.shape == ms[k].shape}
            missing, unexpected = model.load_state_dict(filtered, strict=False)
            load_info["loaded"] = True
            load_info["missing_keys"]   = list(missing)
            load_info["unexpected_keys"] = list(unexpected)
            load_info["note"] += f" | ckpt loaded {len(filtered)}/{len(sd)} keys"
        model = model.to(device)
        model.load_info = load_info
        backbone_prefixes = ["backbone."]

    elif mode == "latent":
        # MultiTaskWrapperCNN: stem channel-adapted + 3 heads; the 4D forward handles a single frame [B,C,H,W] directly
        in_chans = int(config.get("latent_in_chans", 16))
        model = MultiTaskWrapperCNN(
            checkpoint_path=config.get("model_ckpt") or None,
            device=device,
            backbone_name=backbone,
            in_chans=in_chans,
            pretrained=pretrain,
            dropout=dropout,
            freeze_backbone=freeze,
            prompt_fusion_cfg=prompt_fusion_cfg,
            stem_type=config.get("latent_stem_type", "single"),
            stem_channels=config.get("latent_stem_channels"),
            stem_init=config.get("latent_stem_init", "expand_in1k"),
            num_porn_classes=num_porn, num_gore_classes=num_gore,
        )
        model = model.to(device)
        model.load_info = model.base_load_info
        backbone_prefixes = ["base.backbone."]

    elif mode == "latent_to_image":
        model = _build_latent_to_image_model(config, device, dropout, freeze,
                                             prompt_fusion_cfg=prompt_fusion_cfg)
        backbone_prefixes = ["backbone."]

    elif mode == "latent_to_image_feat":
        model = _build_latent_to_image_feat_model(config, device, dropout, freeze,
                                                  prompt_fusion_cfg=prompt_fusion_cfg)
        backbone_prefixes = ["backbone."]

    else:
        raise ValueError(
            f"train_image.py does not support input_mode={mode!r}; "
            f"only 'image' / 'latent' / 'latent_to_image' / 'latent_to_image_feat' are supported"
        )

    criterion_bin = torch.nn.CrossEntropyLoss()
    criterion_ip  = torch.nn.CrossEntropyLoss()

    optimizer = build_optimizer(model, config, backbone_prefixes=backbone_prefixes)
    return model, optimizer, criterion_bin, criterion_ip


# =========================
# ckpt directory naming
# =========================
def build_ckpt_dir(config):
    now = datetime.datetime.now()
    today_str = now.strftime("%Y%m%d")
    now_str   = now.strftime("%Y%m%d_%H%M%S")

    mode     = config.get("input_mode", "image")
    backbone = config.get("backbone", "convnext_base")

    # naming convention: with exp_tag given, use the concise "{timestamp}_{exp_tag}" format,
    # otherwise the full parameter-concatenation format
    exp_tag = config.get("exp_tag", "")

    if mode == "image":
        exp_name = f"image-{backbone}-multitask-porn-gore-ip"
    elif mode == "latent":
        exp_name = f"imagelatent-{backbone}-multitask-porn-gore-ip"
    elif mode == "latent_to_image":
        l2i_bb = config.get("latent_to_image_backbone", backbone)
        exp_name = f"latent2image-{l2i_bb}-multitask-porn-gore-ip"
    elif mode == "latent_to_image_feat":
        l2f_bb = config.get("latent_to_image_feat_backbone", backbone)
        exp_name = f"latent2imagefeat-{l2f_bb}-multitask-porn-gore-ip"
    else:
        exp_name = f"image-unknown-{mode}"

    if exp_tag:
        dir_name = f"{now_str}_{exp_tag}"
    else:
        # full format: concatenate all parameters
        plan     = config.get("sampling_plan", {})
        plan_str = "_".join(f"{k}{v}" for k, v in sorted(plan.items()))
        if mode == "image":
            hw = config.get("image_stretch_target_hw")
            extra = f"hw{int(hw[0])}x{int(hw[1])}" if hw else f"img{config.get('image_size', 224)}"
        elif mode == "latent":
            hw = config.get("latent_stretch_target_hw")
            extra = f"hw{int(hw[0])}x{int(hw[1])}" if hw else "noresize"
            extra += f"_step{config.get('latent_val_step', 5)}"
        elif mode == "latent_to_image":
            hw = config.get("latent_to_image_stretch_target_hw")
            extra = f"hw{int(hw[0])}x{int(hw[1])}" if hw else f"img{config.get('latent_to_image_image_size', 224)}"
            extra += f"_step{config.get('latent_to_image_val_step', 5)}"
        elif mode == "latent_to_image_feat":
            extra = f"step{config.get('latent_to_image_val_step', 5)}"
        else:
            extra = ""
        dir_name = (
            f"{now_str}_BS{config['bs']}_LR{config['lr']}_valpc{config['val_per_class']}"
            f"_{plan_str}_{extra}_seed{config['seed']}"
        )

    return os.path.join(
        config["ckpt_root"], today_str, config.get("model", "image-model"), exp_name,
        dir_name,
    )


# =========================
# Staged Unfreeze (image latent + CNN / ViT)
# =========================
def _get_backbone_and_stem(model, config):
    """Return (backbone_module, stem_module, backbone_prefixes) by backbone type.
    - CNN (MultiTaskWrapperCNN): model.base.backbone / stem
      - progressive stem: features[0] (the whole ProgressiveStem) or conv1 (resnet)
      - single stem: features[0][0] or conv1
    - ViT (MultiTaskWrapperCNN): model.base.backbone / stem = conv_proj
    """
    backbone_name = str(config.get("backbone", "convnext_base"))
    family = get_backbone_family(backbone_name)
    if family == "vit":
        backbone = model.base.backbone
        stem = backbone.conv_proj
        prefixes = ["base.backbone."]
    else:
        backbone = model.base.backbone
        stem_type = config.get("latent_stem_type", "single")
        if family in ("convnext", "swin_v2"):
            if stem_type == "progressive":
                stem = backbone.features[0]  # the whole ProgressiveStem module
            else:
                stem = backbone.features[0][0]  # a single Conv2d
        else:
            stem = backbone.conv1  # resnet: both progressive/single live in conv1
        prefixes = ["base.backbone."]
    return backbone, stem, prefixes


def _freeze_backbone_for_staged_image(model, config):
    """Staged unfreezing — stage 1: freeze all backbone layers except the stem; train only stem + heads.
    Supports latent + CNN (MultiTaskWrapperCNN) and latent + ViT (MiddleFrameMultiTaskModel).
    """
    backbone, stem, _ = _get_backbone_and_stem(model, config)

    # first freeze all backbone params
    for param in backbone.parameters():
        param.requires_grad = False

    # unfreeze the stem (the channel-adaptation layer)
    for param in stem.parameters():
        param.requires_grad = True

    # non-backbone layers such as heads / dropout are already trainable; nothing to do
    frozen_count = sum(1 for p in model.parameters() if not p.requires_grad)
    trainable_count = sum(1 for p in model.parameters() if p.requires_grad)
    backbone_name = str(config.get("backbone", "convnext_base"))
    print(f"[Staged Unfreeze] Backbone frozen (except stem). backbone={backbone_name}, "
          f"Frozen: {frozen_count}, Trainable: {trainable_count}")


def _unfreeze_backbone_image(model):
    """Staged unfreezing — stage 2: unfreeze all parameters."""
    for param in model.parameters():
        param.requires_grad = True
    total_trainable = sum(1 for p in model.parameters() if p.requires_grad)
    print(f"[Staged Unfreeze] All parameters unfrozen. Total trainable: {total_trainable}")


# =========================
# Training Loop
# =========================
def run_training_loop(config, device, ckpt_dir, model, optimizer, scheduler,
                      criterion_bin, criterion_ip,
                      train_loader, val_loader, test_loader,
                      train_ds, val_ds, test_ds):
    history = init_history()
    global_iter      = 0
    best_val_avg_f1  = -1.0
    best_test_avg_f1 = -1.0
    grad_clip = float(config.get("grad_clip_max_norm", 0.0))
    eval_kw = dict(lambda_porn=config["lambda_porn"],
                   lambda_gore=config["lambda_gore"],
                   lambda_ip=config["lambda_ip"],
                   scan_threshold=bool(config.get("eval_scan_threshold", False)),
                   num_porn_classes=config.get("num_porn_classes", 2),
                   num_gore_classes=config.get("num_gore_classes", 2))

    # Evaluate the validation set every epoch and select the deployed checkpoint
    # by val avg-F1. Set eval_test_every_epoch=True to also run per-epoch test
    # evaluation for monitoring; the checkpoint is still selected by val avg-F1.
    eval_test_every_epoch = bool(config.get("eval_test_every_epoch", False))

    # ---- staged-unfreeze state (effective only for latent-like modes; supports CNN and ViT backbones) ----
    staged = bool(config.get("staged_unfreeze", False))
    _input_mode = config.get("input_mode", "image")
    if staged and _input_mode not in ("latent", "latent_to_image", "latent_to_image_feat"):
        print(f"[WARN] staged_unfreeze=True only takes effect when input_mode ∈ latent/latent_to_image/latent_to_image_feat; "
              f"current mode={_input_mode!r}; automatically disabled, running full-parameter finetuning.")
        staged = False
    backbone_unfrozen = not staged  # False = currently in the frozen stage
    if staged:
        staged_max_epochs = int(config.get("staged_unfreeze_max_epochs", 5))
        staged_patience = int(config.get("staged_unfreeze_loss_patience", 2))
        staged_threshold = float(config.get("staged_unfreeze_loss_threshold", 0.05))
        staged_loss_history = []
        staged_patience_counter = 0

        _freeze_backbone_for_staged_image(model, config)
        # rebuild the optimizer: include only the currently trainable params
        _, _, backbone_prefixes = _get_backbone_and_stem(model, config)
        optimizer = build_optimizer(model, config, backbone_prefixes=backbone_prefixes)
        # rebuild the scheduler: stage 1 uses staged_max_epochs as T_max
        scheduler = build_lr_scheduler(optimizer, config, t_max_override=staged_max_epochs)

    if config.get("test_before_training", False):
        print("=" * 80)
        print("[Epoch 0] Evaluating model before training starts...")
        print("=" * 80)
        v0 = evaluate(model, val_loader,  criterion_bin, criterion_ip, device, 0, "Val",  verbose=False, **eval_kw)
        v_loss, v_p, v_g, v_ip, v_res, v_thr = v0
        t_loss = t_p = t_g = t_ip = t_res = t_thr = None
        if eval_test_every_epoch:
            t0 = evaluate(model, test_loader, criterion_bin, criterion_ip, device, 0, "Test", verbose=True, **eval_kw)
            t_loss, t_p, t_g, t_ip, t_res, t_thr = t0
        print_table(_EPOCH_SUMMARY_HEADERS,
                    [_epoch_summary_row(0, "N/A", v_loss, t_loss, v_p, v_g, v_ip, t_p, t_g, t_ip)],
                    title="Epoch 0 Summary (Before Training)", width_limit=22)
        save_epoch_artifacts(ckpt_dir, 0, history, model,
                             val_results=v_res, test_results=t_res,
                             val_threshold_results=v_thr, test_threshold_results=t_thr,
                             val_ip_eval_extra=v_ip, test_ip_eval_extra=t_ip,
                             test_porn_metrics=t_p, test_gore_metrics=t_g)

    for epoch in range(1, config["epochs"] + 1):
        train_loss, train_porn, train_gore, train_ip_pos, global_iter = train_one_epoch(
            model=model, train_loader=train_loader, optimizer=optimizer,
            criterion_bin=criterion_bin, criterion_ip=criterion_ip, device=device,
            epoch=epoch, history=history, global_iter=global_iter,
            lambda_porn=config["lambda_porn"],
            lambda_gore=config["lambda_gore"],
            lambda_ip=config["lambda_ip"],
            enable_train_group_head_mask=config.get("enable_train_group_head_mask", False),
            grad_clip_max_norm=grad_clip,
            num_porn_classes=config.get("num_porn_classes", 2),
            num_gore_classes=config.get("num_gore_classes", 2),
        )

        v_loss, v_p, v_g, v_ip, v_res, v_thr = evaluate(
            model, val_loader, criterion_bin, criterion_ip, device, epoch, "Val",
            verbose=False, **eval_kw,
        )
        t_loss = t_p = t_g = t_ip = t_res = t_thr = None
        if eval_test_every_epoch:
            t_loss, t_p, t_g, t_ip, t_res, t_thr = evaluate(
                model, test_loader, criterion_bin, criterion_ip, device, epoch, "Test",
                verbose=True, **eval_kw,
            )

        append_epoch_history(
            history, epoch, train_loss, v_loss, t_loss,
            train_porn, v_p, t_p, train_gore, v_g, t_g,
            train_ip_pos, v_ip["ip_pos_metrics"],
            t_ip["ip_pos_metrics"] if t_ip is not None else None,
            val_total_risk_stats=v_ip.get("total_risk_stats"),
            test_total_risk_stats=t_ip.get("total_risk_stats") if t_ip is not None else None,
        )

        save_epoch_artifacts(ckpt_dir, epoch, history, model,
                             val_results=v_res, test_results=t_res,
                             val_threshold_results=v_thr, test_threshold_results=t_thr,
                             val_ip_eval_extra=v_ip, test_ip_eval_extra=t_ip,
                             test_porn_metrics=t_p, test_gore_metrics=t_g)

        print_table(_EPOCH_SUMMARY_HEADERS,
                    [_epoch_summary_row(epoch, train_loss, v_loss, t_loss, v_p, v_g, v_ip, t_p, t_g, t_ip)],
                    title=f"Epoch {epoch} Summary", width_limit=22)

        best_val_avg_f1, best_test_avg_f1 = update_best_checkpoints(
            model, ckpt_dir, epoch, v_res, t_res,
            v_p, v_g, v_ip, t_p, t_g, t_ip,
            best_val_avg_f1, best_test_avg_f1,
        )

        # ---- staged unfreezing: loss-convergence check + automatic unfreeze ----
        if staged and not backbone_unfrozen:
            staged_loss_history.append(train_loss)
            if len(staged_loss_history) >= 2:
                prev_loss = staged_loss_history[-2]
                relative_drop = (prev_loss - train_loss) / max(abs(prev_loss), 1e-8)
                print(f"[Staged Unfreeze] Epoch {epoch} train_loss={train_loss:.4f} "
                      f"(↓{relative_drop:.1%})")
                if relative_drop < staged_threshold:
                    staged_patience_counter += 1
                    print(f"[Staged Unfreeze] Loss drop below threshold {staged_threshold:.1%} "
                          f"— patience {staged_patience_counter}/{staged_patience}")
                else:
                    staged_patience_counter = 0
            else:
                print(f"[Staged Unfreeze] Epoch {epoch} train_loss={train_loss:.4f}")

            trigger_reason = None
            if staged_patience_counter >= staged_patience:
                trigger_reason = "loss_converged"
            elif epoch >= staged_max_epochs:
                trigger_reason = f"reached_max_epochs ({staged_max_epochs})"

            if trigger_reason is not None:
                remaining_epochs = config["epochs"] - epoch
                print(f"[Staged Unfreeze] \u2605 Unfreezing backbone at epoch {epoch} "
                      f"(trigger: {trigger_reason}). "
                      f"Rebuilding optimizer & scheduler for remaining {remaining_epochs} epochs.")
                _unfreeze_backbone_image(model)
                backbone_unfrozen = True
                _, _, backbone_prefixes = _get_backbone_and_stem(model, config)
                optimizer = build_optimizer(model, config, backbone_prefixes=backbone_prefixes)
                scheduler = build_lr_scheduler(optimizer, config, t_max_override=remaining_epochs)

        if scheduler is not None:
            scheduler.step()

    # ---- Final test evaluation ----
    # After training completes, reload the val-selected checkpoint and report its
    # test performance.
    best_val_ckpt = os.path.join(ckpt_dir, "ckpts", "best_val_avg_f1_model.pth")
    if os.path.exists(best_val_ckpt):
        print("=" * 80)
        print("[Final] Loading best-val checkpoint for a one-shot test evaluation...")
        print("=" * 80)
        model.load_state_dict(torch.load(best_val_ckpt, map_location=device))
        f_loss, f_p, f_g, f_ip, f_res, f_thr = evaluate(
            model, test_loader, criterion_bin, criterion_ip, device,
            config["epochs"], "Test", verbose=True, **eval_kw,
        )
        f_ip_f1 = f_ip["ip_pos_metrics"]["f1"]
        f_avg_f1 = (f_p["f1"] + f_g["f1"] + f_ip_f1) / 3.0
        print_table(
            ["checkpoint", "porn_f1", "gore_f1", "ip_f1", "avg_f1", "test_loss"],
            [["best_val_avg_f1", f"{f_p['f1']:.4f}", f"{f_g['f1']:.4f}",
              f"{f_ip_f1:.4f}", f"{f_avg_f1:.4f}", f"{f_loss:.4f}"]],
            title="Final Test Evaluation (best-val checkpoint)",
        )
        save_json(
            {"checkpoint": "best_val_avg_f1_model.pth", "test_loss": f_loss,
             "avg_f1": f_avg_f1, "porn_metrics": f_p, "gore_metrics": f_g,
             "ip_eval": f_ip},
            os.path.join(ckpt_dir, "meta", "final_test_metrics.json"),
        )
        save_json(f_res, os.path.join(ckpt_dir, "meta", "final_test_results.json"))
        if f_thr is not None:
            save_json(f_thr, os.path.join(ckpt_dir, "meta", "final_test_threshold_metrics.json"))
    else:
        print(f"[Final] best-val checkpoint not found at {best_val_ckpt}; "
              f"skipping final test evaluation.")

    print("Training Finished.")


# =========================
# Data Ablation: Pool Size Limiting
# =========================
def _apply_train_pool_limit(train_list, max_pool, seed=42):
    """Stratified sampling by train_group, truncating the training pool to max_pool items.
    Keeps each group's original ratio; a fixed seed guarantees reproducibility.
    """
    from collections import defaultdict

    # group by train_group
    group_to_items = defaultdict(list)
    for item in train_list:
        g = item.get("train_group", "unknown")
        group_to_items[g].append(item)

    total_original = len(train_list)
    rng = np.random.RandomState(seed)

    # allocate each group's target count proportionally
    group_counts = {g: len(items) for g, items in group_to_items.items()}
    group_targets = {}
    allocated = 0
    sorted_groups = sorted(group_counts.keys())
    for i, g in enumerate(sorted_groups):
        if i == len(sorted_groups) - 1:
            # the last group takes the remainder, avoiding rounding accumulation errors
            group_targets[g] = max_pool - allocated
        else:
            ratio = group_counts[g] / total_original
            n = int(round(ratio * max_pool))
            group_targets[g] = min(n, group_counts[g])  # capped at the actual pool size
            allocated += group_targets[g]

    # perform the sampling
    result = []
    print(f"\n{'='*60}")
    print(f" [Data Ablation] max_train_pool = {max_pool}")
    print(f" original training-set size: {total_original} -> truncated to: {max_pool}")
    print(f"{'-'*60}")
    print(f" {'Group':<20} {'Original':>10} {'Sampled':>10} {'Ratio':>8}")
    print(f" {'-'*18:<20} {'-'*10:>10} {'-'*10:>10} {'-'*8:>8}")

    for g in sorted_groups:
        items = group_to_items[g]
        target = group_targets[g]
        if target >= len(items):
            sampled = items  # capped at the actual pool size
        else:
            indices = rng.choice(len(items), size=target, replace=False)
            sampled = [items[i] for i in sorted(indices)]
        result.extend(sampled)
        print(f" {g:<20} {len(items):>10} {len(sampled):>10} {len(sampled)/len(items)*100:>7.1f}%")

    print(f"{'-'*60}")
    print(f" total: {total_original} -> {len(result)} (truncation ratio {len(result)/total_original*100:.1f}%)")
    print(f"{'='*60}\n")

    return result


# =========================
# Experiment Entry
# =========================
def run_experiment(config):
    seed_everything(config["seed"])
    if config.get("torch_home"):
        os.environ["TORCH_HOME"] = str(config["torch_home"])
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")

    # ---- auto-resolve model_ckpt from pretrain_dir (no need to fill in the ckpt path manually) ----
    # Strategy: prefer backbone_iter100000.pth (the fair-comparison baseline);
    #           if training never reached 100000 iters, take the latest (largest iter number) backbone_iter*.pth
    pretrain_dir = (config.get("pretrain_dir") or "").strip()
    if pretrain_dir and not config.get("model_ckpt"):
        import glob as _glob
        import re as _re
        ckpts_dir_path = os.path.join(pretrain_dir, "ckpts")
        resolved_ckpt = ""

        # 1) first try the fixed iter100000
        target_iter_path = os.path.join(ckpts_dir_path, "backbone_iter100000.pth")
        if os.path.exists(target_iter_path):
            resolved_ckpt = target_iter_path
            print(f"[Auto-resolve] pretrain_dir -> backbone_iter100000.pth (fair-comparison baseline)")
        else:
            # 2) never reached 100000: take the latest (largest iter number) backbone_iter*.pth
            iter_pattern = os.path.join(ckpts_dir_path, "backbone_iter*.pth")
            iter_files = _glob.glob(iter_pattern)
            if iter_files:
                # extract the iter number from the file name and take the maximum
                def _extract_iter(fp):
                    m = _re.search(r'backbone_iter(\d+)\.pth$', fp)
                    return int(m.group(1)) if m else -1
                iter_files_sorted = sorted(iter_files, key=_extract_iter)
                resolved_ckpt = iter_files_sorted[-1]
                latest_iter = _extract_iter(resolved_ckpt)
                print(f"[Auto-resolve] backbone_iter100000.pth does not exist; "
                      f"taking the latest iter={latest_iter} checkpoint")
            else:
                print(f"[WARNING] no backbone_iter*.pth found under pretrain_dir: {ckpts_dir_path}")

        if resolved_ckpt and os.path.exists(resolved_ckpt):
            config["model_ckpt"] = resolved_ckpt
            print(f"[Auto-resolve] pretrain_dir -> model_ckpt: {resolved_ckpt}")
        elif not resolved_ckpt:
            print(f"[WARNING] pretrain_dir specified but no valid ckpt found: {pretrain_dir}")

    # ---- auto-resolve stage1_ckpt_dir: find the best ckpt from the Stage 1 finetune experiment ----
    # Strategy:
    #   1) prefer ckpts/best_val_avg_f1_model.pth (selected by val avg-F1 during training)
    #   2) fall back to the model_epoch_*.pth with the largest epoch number
    # Note: the Stage 1 ckpt has no prompt_fusion params; they go missing when loaded with strict=False (expected behavior)
    # Effective only when enable_prompt_fusion=True (PE/PE2A Stage 2), preventing misuse by the M/N/D groups
    stage1_dir = (config.get("stage1_ckpt_dir") or "").strip()
    if stage1_dir and not config.get("model_ckpt") and config.get("enable_prompt_fusion", False):
        import glob as _glob
        import re as _re
        stage1_ckpts_dir = os.path.join(stage1_dir, "ckpts")
        stage1_resolved = ""

        # 1) preferred: best_val_avg_f1_model.pth (the deployed, val-selected checkpoint)
        best_val_ckpt = os.path.join(stage1_ckpts_dir, "best_val_avg_f1_model.pth")
        if os.path.exists(best_val_ckpt):
            stage1_resolved = best_val_ckpt
            print(f"[Stage1-resolve] using best_val_avg_f1_model.pth")

        # 2) fallback: the largest epoch number
        if not stage1_resolved:
            epoch_pattern = os.path.join(stage1_ckpts_dir, "model_epoch_*.pth")
            epoch_files = _glob.glob(epoch_pattern)
            if epoch_files:
                def _extract_epoch(fp):
                    m = _re.search(r'model_epoch_(\d+)\.pth$', fp)
                    return int(m.group(1)) if m else -1
                epoch_files_sorted = sorted(epoch_files, key=_extract_epoch)
                stage1_resolved = epoch_files_sorted[-1]
                latest_ep = _extract_epoch(stage1_resolved)
                print(f"[Stage1-resolve] falling back to the largest epoch={latest_ep} ckpt")

        if stage1_resolved and os.path.exists(stage1_resolved):
            config["model_ckpt"] = stage1_resolved
            print(f"[Stage1-resolve] stage1_ckpt_dir -> model_ckpt: {stage1_resolved}")
        else:
            print(f"[WARNING] stage1_ckpt_dir specified but no valid ckpt found: {stage1_dir}")

    # ---- auto-resolve pe2_mlp_ckpt_dir: find the best ckpt (by max val acc_avg) from the PE-MLP training dir ----
    # Strategy (aligned with scripts/export_weights.py):
    #   1a) read meta/history.json -> max(acc_avg) -> best_step
    #       try best_prompt_model_step{N}.pth first; if absent, try ckpt_step{N}.pth
    #   1b) fallback: parse train_prompt.log with a regex to extract step/val_avg -> max(val_avg)
    #   2) fallback: glob best_prompt_model_step*.pth -> take the largest step
    pe2_mlp_dir = (config.get("pe2_mlp_ckpt_dir") or "").strip()
    if pe2_mlp_dir and not config.get("pe2_mlp_ckpt_path"):
        import json as _json
        import glob as _glob
        pe2_ckpts_dir = os.path.join(pe2_mlp_dir, "ckpts")
        pe2_resolved = ""

        # 1) read history.json to find max val acc_avg
        # 1a) prefer history.json
        history_json = os.path.join(pe2_mlp_dir, "meta", "history.json")
        if os.path.exists(history_json):
            try:
                with open(history_json) as _f:
                    _h = _json.load(_f)
                _steps = _h.get("step", [])
                _accs = _h.get("acc_avg", [])
                if _steps and _accs:
                    best_idx = max(range(len(_accs)), key=lambda i: _accs[i])
                    best_step = _steps[best_idx]
                    best_acc = _accs[best_idx]
                    # try best_prompt_model_step{N}.pth first (saved only when acc_avg hits a new high);
                    # if absent, try ckpt_step{N}.pth (saved at every eval step, always present)
                    best_ckpt = os.path.join(pe2_ckpts_dir, f"best_prompt_model_step{best_step:06d}.pth")
                    if not os.path.exists(best_ckpt):
                        best_ckpt = os.path.join(pe2_ckpts_dir, f"ckpt_step{best_step:06d}.pth")
                    if os.path.exists(best_ckpt):
                        pe2_resolved = best_ckpt
                        print(f"[PE2-resolve] best step={best_step} (val acc_avg={best_acc:.4f}) -> {best_ckpt}")
                    else:
                        print(f"[PE2-resolve] best step={best_step} (val acc_avg={best_acc:.4f}) but the ckpt file does not exist")
                else:
                    print(f"[PE2-resolve] history.json exists but step/acc_avg is empty")
            except Exception as e:
                print(f"[PE2-resolve] failed to read history.json: {e}")
        else:
            print(f"[PE2-resolve] history.json does not exist: {history_json}")

        # 1b) fallback: parse step/val_avg from train_prompt.log
        if not pe2_resolved:
            import re as _re
            for _log_name in ("train_prompt.log", "train_prompt_text.log"):
                _log_path = os.path.join(pe2_mlp_dir, _log_name)
                if not os.path.exists(_log_path):
                    continue
                try:
                    with open(_log_path, errors="replace") as _lf:
                        _log_text = _lf.read()
                    _pat = _re.compile(
                        r'step=(\d+)\s*\|.*val_avg=([\d.]+)%'
                    )
                    _log_steps, _log_accs = [], []
                    for _m in _pat.finditer(_log_text):
                        _log_steps.append(int(_m.group(1)))
                        _log_accs.append(float(_m.group(2)) / 100.0)
                    if _log_steps and _log_accs:
                        _bi = max(range(len(_log_accs)), key=lambda i: _log_accs[i])
                        best_step = _log_steps[_bi]
                        best_acc = _log_accs[_bi]
                        best_ckpt = os.path.join(pe2_ckpts_dir, f"best_prompt_model_step{best_step:06d}.pth")
                        if not os.path.exists(best_ckpt):
                            best_ckpt = os.path.join(pe2_ckpts_dir, f"ckpt_step{best_step:06d}.pth")
                        if os.path.exists(best_ckpt):
                            pe2_resolved = best_ckpt
                            print(f"[PE2-resolve] (log fallback) best step={best_step} (val acc_avg={best_acc:.4f}) -> {best_ckpt}")
                        else:
                            print(f"[PE2-resolve] (log fallback) best step={best_step} but the ckpt file does not exist")
                    break  # stop after the first log file found
                except Exception as e:
                    print(f"[PE2-resolve] failed to parse {_log_name}: {e}")

        # 2) fallback: glob best_prompt_model_step*.pth and take the largest step
        if not pe2_resolved:
            pattern = os.path.join(pe2_ckpts_dir, "best_prompt_model_step*.pth")
            ckpt_files = _glob.glob(pattern)
            if ckpt_files:
                import re as _re
                def _extract_step(fp):
                    m = _re.search(r'best_prompt_model_step(\d+)\.pth$', fp)
                    return int(m.group(1)) if m else -1
                ckpt_files_sorted = sorted(ckpt_files, key=_extract_step)
                pe2_resolved = ckpt_files_sorted[-1]
                latest_step = _extract_step(pe2_resolved)
                print(f"[PE2-resolve] falling back to the ckpt with the largest step={latest_step}: {pe2_resolved}")

        if pe2_resolved and os.path.exists(pe2_resolved):
            config["pe2_mlp_ckpt_path"] = pe2_resolved
            print(f"[PE2-resolve] pe2_mlp_ckpt_dir -> pe2_mlp_ckpt_path: {pe2_resolved}")
        else:
            print(f"[WARNING] pe2_mlp_ckpt_dir specified but no valid ckpt found: {pe2_mlp_dir}")

    ckpt_dir = build_ckpt_dir(config)
    prepare_output_dirs(ckpt_dir)
    save_json(config, os.path.join(ckpt_dir, "meta", "config_snapshot.json"))
    snapshot_code_dir(ckpt_dir, include_subdirs=False, verbose=False)
    setup_logger(ckpt_dir)

    print_kv_table([(k, v) for k, v in config.items()],
                   title="CONFIG", key_name="Key", value_name="Value", width_limit=120)

    train_all, train_list, val_list, test_list = load_and_split_image_data(config, ckpt_dir)

    # ---- Stage 1 training-pool ID reuse: ensure Stage 2 uses exactly the same training samples as Stage 1 ----
    # Effective only when enable_prompt_fusion=True (PE/PE2A Stage 2), preventing misuse by the M/N/D groups
    # If stage1_ckpt_dir resolved to a model_ckpt, try reading meta/train_pool_ids.csv from the same directory
    stage1_dir_for_pool = (config.get("stage1_ckpt_dir") or "").strip()
    if stage1_dir_for_pool and config.get("enable_prompt_fusion", False):
        stage1_pool_csv = os.path.join(stage1_dir_for_pool, "meta", "train_pool_ids.csv")
        if os.path.exists(stage1_pool_csv):
            with open(stage1_pool_csv, encoding="utf-8") as _pf:
                _pool_reader = csv.DictReader(_pf)
                stage1_ids = set()
                stage1_groups = {}
                for _row in _pool_reader:
                    _tid = _row.get("train_id", "").strip()
                    _tgrp = _row.get("train_group", "unknown").strip()
                    if _tid:
                        stage1_ids.add(_tid)
                        stage1_groups[_tid] = _tgrp
            if stage1_ids:
                _before = len(train_list)
                train_list = [item for item in train_list if item["name"] in stage1_ids]
                _after = len(train_list)
                print(f"\n{'='*60}")
                print(f" [Stage1 Pool Reuse] loading training-pool IDs from stage1_ckpt_dir")
                print(f"  source file: {stage1_pool_csv}")
                print(f"  Stage1 training-pool size: {len(stage1_ids)}")
                print(f"  current train_list: {_before} -> {_after} (after filtering)")
                if _after < len(stage1_ids):
                    _missing = stage1_ids - {item["name"] for item in train_list}
                    print(f"  ⚠️ {len(_missing)} Stage1 IDs not found in the current train_list")
                    print(f"     (the CSV may have changed; check the train_csv config)")
                print(f"{'='*60}\n")
                # flag: skip the later max_train_pool truncation (the Stage 1 pool already exists)
                config["_stage1_pool_loaded"] = True
                # still save a copy of train_pool_ids.csv (should match Stage 1)
            else:
                print(f"[Stage1 Pool Reuse] {stage1_pool_csv} is empty, skipping")
        else:
            print(f"[Stage1 Pool Reuse] {stage1_pool_csv} not found; using normal pool sampling")

    # ---- data ablation: training-pool size limit ----
    max_pool = int(config.get("max_train_pool", 0))
    _stage1_loaded = config.pop("_stage1_pool_loaded", False)
    if _stage1_loaded:
        # the Stage 1 pool is already loaded; skip pool truncation
        print(f"[Data Ablation] Stage 1 training pool loaded; skipping max_train_pool truncation")
    elif max_pool > 0 and len(train_list) > max_pool:
        train_list = _apply_train_pool_limit(train_list, max_pool, seed=int(config["seed"]))
    elif max_pool > 0:
        print(f"[Data Ablation] max_train_pool={max_pool} >= current training-set size ({len(train_list)}); no truncation needed")
    
    # ---- save the training-pool train_id list to meta/train_pool_ids.csv ----
    _pool_csv = os.path.join(ckpt_dir, "meta", "train_pool_ids.csv")
    with open(_pool_csv, "w", newline="", encoding="utf-8") as _f:
        _w = csv.writer(_f)
        _w.writerow(["index", "train_id", "train_group"])
        for _i, _item in enumerate(train_list):
            _w.writerow([_i, _item["name"], _item.get("train_group", "unknown")])
    print(f"[Train Pool] saved {len(train_list)} train_ids -> {_pool_csv}")
    
    print_final_dataset_summary(train_list=train_list, test_list=test_list)

    # ---- latent normalization stats (only for latent mode + normalize enabled) ----
    latent_stats = None
    if config.get("input_mode") == "latent" and bool(config.get("latent_enable_normalize", False)):
        cache_path = (config.get("latent_stats_cache") or "").strip()
        if cache_path:
            latent_mean, latent_std, latent_info = load_latent_stats_from_cache(
                cache_path, expected_chans=int(config["latent_in_chans"]),
            )
            print(f"[Latent Stats] reusing the cache directly: {cache_path}")
        else:
            n_samples = int(config.get("latent_stats_num_samples", 500))
            latent_mean, latent_std, latent_info = compute_image_latent_stats(
                train_list,
                model=str(config["model"]),
                num_steps=int(config["num_steps"]),
                num_samples=n_samples,
                seed=int(config["seed"]),
            )
        save_json(latent_info, os.path.join(ckpt_dir, "meta", "latent_stats.json"))
        print_kv_table(
            [("source",               latent_info.get("_loaded_from_cache", "freshly computed")),
             ("channels",             latent_info["num_channels"]),
             ("mean_overall",         f"{latent_info['mean_overall']:.4f}"),
             ("std_overall",          f"{latent_info['std_overall']:.4f}"),
             ("mean_abs_max",         f"{latent_info['mean_abs_max']:.4f}"),
             ("std_min",              f"{latent_info['std_min']:.4f}"),
             ("std_max",              f"{latent_info['std_max']:.4f}")],
            title="Image Latent Normalization Stats",
            width_limit=60,
        )
        latent_stats = (latent_mean, latent_std)
    elif config.get("input_mode") == "latent":
        print("[Latent Stats] latent_enable_normalize=False; skipping per-channel normalization, using raw latent values")

    train_ds, val_ds, test_ds = build_image_datasets(
        config, train_list, val_list, test_list, latent_stats=latent_stats,
    )
    train_loader, val_loader, test_loader = build_image_dataloaders(
        config, train_ds, val_ds, test_ds, train_list, collate_fn,
    )
    model, optimizer, criterion_bin, criterion_ip = build_image_model_optimizer_criterion(config, device)
    scheduler = build_lr_scheduler(optimizer, config)
    print_model_load_summary(model)
    print_pe_mlp_load_summary(model)

    # ---- latent_to_image / latent_to_image_feat requires vae_pretrained_path ----
    _input_mode_main = config.get("input_mode", "image")
    if _input_mode_main in ("latent_to_image", "latent_to_image_feat"):
        vae_path = config.get("vae_pretrained_path", "")
        if not vae_path:
            raise ValueError(f"input_mode={_input_mode_main!r} requires vae_pretrained_path to be configured")

    # ---- latent_to_image_feat mode: compute decoded-feat per-channel stats before training ----
    if _input_mode_main == "latent_to_image_feat":
        cache_path = (config.get("latent_to_image_feat_stats_cache") or "").strip()
        if cache_path:
            feat_mean, feat_std, feat_info = load_decoded_feat_stats_from_cache(
                cache_path, expected_chans=192,  # qwen-image-2512 WanVAE early-stop outputs 192ch
            )
            print(f"[Image Decoded Feat Stats] reusing the cache directly: {cache_path}")
        else:
            from data_pipeline import compute_image_decoded_feat_stats
            n_samples = int(config.get("latent_to_image_feat_stats_num_samples", 200))
            feat_mean, feat_std, feat_info = compute_image_decoded_feat_stats(
                train_list,
                vae_extractor=model.extractor,
                model_name=str(config["model"]),
                num_steps=int(config["num_steps"]),
                num_samples=n_samples,
                seed=int(config["seed"]),
                device=device,
            )
        save_json(feat_info, os.path.join(ckpt_dir, "meta", "image_decoded_feat_stats.json"))
        print_kv_table(
            [("source", feat_info.get("_loaded_from_cache", "freshly computed")),
             ("channels", feat_info["num_channels"]),
             ("mean_overall", f"{feat_info['mean_overall']:.4f}"),
             ("std_overall", f"{feat_info['std_overall']:.4f}"),
             ("mean_abs_max", f"{feat_info['mean_abs_max']:.4f}"),
             ("std_min", f"{feat_info['std_min']:.4f}"),
             ("std_max", f"{feat_info['std_max']:.4f}")],
            title="Image Decoded Feat Stats (latent_to_image_feat)",
            width_limit=60,
        )
        model.set_feat_stats(feat_mean, feat_std)

    # smoke test
    if bool(config.get("smoke_test_first", True)):
        run_smoke_test(
            config=config, device=device, ckpt_dir=ckpt_dir,
            model=model, optimizer=optimizer, scheduler=scheduler,
            criterion_bin=criterion_bin, criterion_ip=criterion_ip,
            train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
            num_batches=int(config.get("smoke_test_num_batches", 5)),
        )

    run_training_loop(
        config=config, device=device, ckpt_dir=ckpt_dir,
        model=model, optimizer=optimizer, scheduler=scheduler,
        criterion_bin=criterion_bin, criterion_ip=criterion_ip,
        train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
        train_ds=train_ds, val_ds=val_ds, test_ds=test_ds,
    )


# =========================
# CONFIG & Entry
# =========================
if __name__ == "__main__":
    # Note: the CONFIG below targets z-image-turbo. To finetune on another model,
    # edit the fields in place: model / train_csv / test_csv / num_steps /
    # latent_val_step / latent_in_chans / latent_stem_channels (and the same
    # fields in pretrain_image.py, whose stem must match — see README Step 6).

    # Paths can be overridden via environment variables (see the README "Configuration" section):
    #   INGUARD_MODELS_ROOT / INGUARD_OUTPUT_ROOT / INGUARD_CKPT_ROOT
    _MODELS_ROOT = os.environ.get("INGUARD_MODELS_ROOT", "/path/to/models")
    _REVGEN_ROOT = os.environ.get("INGUARD_OUTPUT_ROOT", "./outputs/revgen")
    _CKPT_ROOT   = os.environ.get("INGUARD_CKPT_ROOT", "./outputs/checkpoints")

    CONFIG = {
        # ============================================================
        # 1. runtime / mode selection
        # ============================================================
        "input_mode": "latent",           # "image": read JPG/PNG  |  "latent": read diffusion latent .pth
        "model": "z-image-turbo",       # experiment id, used only for the output directory
        "exp_tag": "",                    # experiment tag; when non-empty the dir is named {timestamp}_{exp_tag}
        "seed": 42,
        "device": "cuda:0",
        "test_before_training": False,
        "smoke_test_first":     True,
        "smoke_test_num_batches": 5,
        "ckpt_root":  _CKPT_ROOT,
        "torch_home": f"{_MODELS_ROOT}/",

        # ============================================================
        # 2. dataset (read the CSV directly; label mapping identical to video training)
        #   pornographic==2 -> porn_label=1  (0/1 -> 0; borderline treated as safe)
        #   violence_gore==2 → gore_label=1 (0/1 → 0)
        #   ip_copyright 1-5 -> ip_label 0-4; 0/6 -> 5 (other)
        # ============================================================
        "train_csv": f"{_REVGEN_ROOT}/z-image-turbo/trainset-seed42-1024-9steps/labels_llm/predictions.csv",
        "test_csv":  f"{_REVGEN_ROOT}/z-image-turbo/testset-seed42-1024-9steps/labels_llm/predictions.csv",

        # ---------- test-set leak checks ----------
        # Train/test leakage is already removed at the source (the released trainset.csv
        # drops every sample matching the test set via prompt Jaccard + image pHash), so the
        # online checks below are optional extras and stay off by default.
        # online prompt leak check (exact match); enable via check_testset_leak=True.
        # Also used for borderline bucketing: point these at the released
        # trainset.csv / testset.csv so the category column drives it (see README
        # Step 6); left empty, borderline falls back to normal_other.
        "train_prompt_csv": "",
        "test_prompt_csv":  "",
        "check_testset_leak": False,
        # exact file_path dedup; enable via remove_train_test_overlap=True
        "remove_train_test_overlap": False,

        "preview_examples_num": 2,              # how many preview images to save per split/class
        "val_per_class": 10,                    # fixed per-class count drawn for val by three_class (porn/gore/normal/ip_controlled)

        # ============================================================
        # 3. training-pool sampling ratios (aligned with video; adjust to the actual data volume)
        # ============================================================
        "sampling_plan": {
            "porn":              2000,
            "gore":              2000,
            "ip_controlled":     2000,
            "porn_borderline":   2000,
            "gore_borderline":   2000,
            "ip_borderline":     2000,
            "normal_other":      2000,
        },

        # ---------- data ablation: training-pool size limit ----------
        # 0 = unlimited (default: full data); >0 = stratified truncation to this total by train_group
        "max_train_pool": 0,

        # ============================================================
        # 4. common training hyperparameters
        # ============================================================
        "epochs":         100,
        "bs":             16,  # 224×224 feature maps are large; lower bs to avoid OOM
        "nw":             8,
        "lr":             1e-4,
        "optimizer_type": "adamw",
        "weight_decay":   0.01,
        "backbone_lr":    5e-5,
        "head_lr":        5e-4,
        "freeze_backbone": False,
        "dropout":         0.1,
        "lr_scheduler":   "cosine",
        "lr_min":         0.0,
        "grad_clip_max_norm": 1.0,

        # multi-task loss weights
        "lambda_porn": 1.0,
        "lambda_gore": 1.0,
        "lambda_ip":   1.0,
        "enable_train_group_head_mask": False,

        # classification-head dim: 2=binary (default), 3=safe/borderline/risk three-way
        # with 3 classes, porn/gore labels keep their raw 0/1/2 values; evaluation uses macro-F1
        "num_porn_classes": 2,
        "num_gore_classes": 2,

        # staged unfreezing (effective in latent mode; supports CNN and ViT backbones; auto-disabled in other modes, running full-parameter finetuning)
        # first freeze the backbone (keeping the stem trainable), training only stem + heads;
        # automatically unfreezes all params once train loss converges or max_epochs is reached.
        "staged_unfreeze":                False,    # master switch; False = start full-parameter finetuning directly
        "staged_unfreeze_max_epochs":     3,        # max epochs of the frozen stage (forced unfreeze when reached)
        "staged_unfreeze_loss_patience":  1,        # trigger unfreeze after N consecutive epochs with no clear loss drop
        "staged_unfreeze_loss_threshold": 0.05,     # relative-drop threshold; (prev-cur)/prev < this counts as "no clear drop"

        # ============================================================
        # 5. Backbone (ConvNeXt / ResNet / ViT, unified torchvision IN1K_V1)
        # ============================================================
        "backbone":            "convnext_base",   # convnext_{tiny,small,base,large} / vit_{b,l}_16 / resnet{18,50,101,152}
        "backbone_pretrained": True,              # torchvision IN1K_V1 pretrained
        "model_ckpt":          "",                # custom ckpt path; empty = start from scratch/IN1K
        # auto-resolve the ckpt from the pretrain working dir (either this or model_ckpt; lower priority than model_ckpt)
        # points to the output dir of pretrain_image.py; automatically finds ckpts/backbone_iter100000.pth;
        # if absent, takes the latest (largest iter number) backbone_iter*.pth
        "pretrain_dir":        "",

        # ============================================================
        # 6. preprocessing params specific to input_mode == "image"
        # ============================================================
        "image_size":             224,
        "image_preprocess_style": "stretch",      # stretch / keep_ratio_pad / native
        "image_stretch_target_hw": [224, 224],    # None = square image_size; [H,W] = rectangular
        "image_mean":  [0.485, 0.456, 0.406],
        "image_std":   [0.229, 0.224, 0.225],
        "image_train_overscan": 1.05,
        "image_enable_hflip":   True,

        # ============================================================
        # 7. params specific to input_mode == "latent"
        # ============================================================
        "latent_in_chans":        16,             # VAE channels: z-image-turbo=16, qwen-image-2512=16, internvl-u=16(reuses the Qwen VAE), flux2-klein-base-9b=32, hunyuan-image-2_1=64
        "num_steps":              9,             # total diffusion steps (a data property)
        "latent_train_step_mode": "random",       # "random" / "fixed"
        "latent_val_step":        3,              # the fixed step used at eval time
        "latent_preprocess_style": "stretch",     # stretch / keep_ratio_pad
        "latent_stretch_target_hw": [224, 224],  # target size in latent space — unified 224×224, aligned with Image/IN1K
        "latent_data_aug":        True,           # training-time augmentation: hflip / random crop+resize / gaussian noise / scale jitter

        # latent per-channel normalization switch: False = no normalization, use raw latent values; True = compute mean/std before training and z-score normalize
        "latent_enable_normalize": False,
        # latent stats cache: empty = computed before training; a path = reused directly (effective only when latent_enable_normalize=True)
        "latent_stats_cache":     "",
        "latent_stats_num_samples": 1000,

        # Stem structure config (must align with pretrain_image.py)
        # stem_type: "single" = one Conv2d(in->128, k=4, s=4) layer | "progressive" = progressive multi-layer stem
        "latent_stem_type": "progressive",
        # stem_channels: per-layer channels in progressive mode [in, ..., out]; progressive only;
        #   first=in_chans, last=backbone stem output channels (convnext_base=128)
        #   z-image-turbo / qwen-image-2512 (16ch): [16, 32, 64, 128]
        #   flux2-klein-base-9b (32ch):             [32, 48, 80, 128]
        #   hunyuan-image-2_1 (64ch):               [64, 80, 100, 128]
        "latent_stem_channels": [16, 32, 64, 128],
        # stem_init: init method (no effect once a pretrain ckpt is loaded; effective only when starting without a ckpt)
        #   "expand_in1k" = reuse the first 3 IN1K ch + mean fill (meaningful for single only)
        #   "trunc_normal" = truncated normal std=0.02 (recommended for progressive)
        #   "kaiming" = Kaiming normal
        "latent_stem_init": "trunc_normal",

        # ============================================================
        # 8. Prompt Fusion (fuse text embeddings into the vision network)
        # z-image-turbo text_dim=2560; other models may differ, adjust as needed
        # masked mean pooling is done in the dataset layer, so no prompt_max_seq_len is needed
        # ============================================================
        "enable_prompt_fusion":       False,   # master switch
        "prompt_fusion_text_dim":     2560,    # z=2560, q=3584, i=4096, h=3584, f=12288
        "prompt_fusion_dropout":      0.3,     # probability of zeroing the whole text_feat during training
        "prompt_fusion_mode":         "concat", # "concat" | "add" (for two-stage)
        "prompt_fusion_scale_init":  0.1,     # initial scale value for add mode

        # ============================================================
        # 8-B. Stage 2 two-stage training config (used together with enable_prompt_fusion=True + stage1_ckpt_dir)
        # ============================================================
        # points to the Stage 1 finetune output dir; automatically finds the best ckpt:
        #   1) ckpts/best_val_avg_f1_model.pth (selected by val avg-F1)
        #   2) the model_epoch_*.pth with the largest epoch number
        # the Stage 1 ckpt has no prompt_fusion params; they go missing automatically when loaded with strict=False (expected behavior)
        "stage1_ckpt_dir":            "",     # Stage 1 finetune output dir path

        # ============================================================
        # 8-C. PE2 scheme config (effective when prompt_fusion_mode="pe2")
        # ============================================================
        # points to the PE-MLP training dir (the output of train_prompt.py); automatically finds the best ckpt:
        #   1) read meta/history.json -> max(acc_avg) -> best_step
        #      try best_prompt_model_step{N}.pth first; if absent, try ckpt_step{N}.pth
        #   2) fallback: glob best_prompt_model_step*.pth -> largest step (only when history.json does not exist)
        # the PE-MLP encoder (proj+proj_ln+blocks) is frozen after loading; only pe2_proj + fusion_scale are trained
        "pe2_mlp_ckpt_dir":           "",     # PE-MLP training dir path
        "pe2_mlp_hidden_dim":         1024,    # PE-MLP hidden dim (align with train_prompt.py --hidden_dim)
        "pe2_mlp_num_layers":         3,       # PE-MLP block count (align with train_prompt.py --num_layers)
        "pe2_mlp_dropout":            0.1,     # PE-MLP dropout (align with train_prompt.py --dropout)

        # ============================================================
        # 9. params specific to input_mode == "latent_to_image"
        # the Dataset reads only latent_x1; the model runs the full-image VAE decoder internally to decode into RGB,
        # resize + ImageNet normalize → backbone → heads.
        # similar to video latent_to_video, but without the temporal dimension.
        # ============================================================
        "latent_to_image_backbone":             "convnext_base",  # convnext_{tiny,small,base,large} / vit_{b,l}_16 / resnet{18,50,101,152}
        "latent_to_image_backbone_pretrained":   True,
        "latent_to_image_image_size":            224,
        "latent_to_image_stretch_target_hw":     [224, 224],       # None = square image_size
        "latent_to_image_preprocess_style":      "stretch",        # stretch / keep_ratio_pad
        "latent_to_image_mean":                  [0.485, 0.456, 0.406],
        "latent_to_image_std":                   [0.229, 0.224, 0.225],
        "latent_to_image_enable_hflip":          True,
        "latent_to_image_train_step_mode":       "random",
        "latent_to_image_val_step":              3,

        # ============================================================
        # 10. params specific to input_mode == "latent_to_image_feat"
        # the Dataset reads only latent_x1; the model's internal VAE decoder stops early and takes intermediate features,
        # per-channel normalize → CNN backbone → heads.
        # currently only qwen-image-2512 is supported (WanVAE early-stop 192ch).
        # ============================================================
        "latent_to_image_feat_backbone":             "convnext_base",  # CNN only
        "latent_to_image_feat_backbone_pretrained":   True,
        "latent_to_image_feat_model_ckpt":            "",
        "latent_to_image_feat_data_aug":              False,
        "latent_to_image_feat_stats_cache":           "",              # decoded feat per-channel mean/std json
        "latent_to_image_feat_stats_num_samples":     200,

        # ============================================================
        # 11. VAE decoding (loaded only when input_mode ∈ {latent_to_image, latent_to_image_feat})
        # ============================================================
        # model Pipeline path (ZImagePipeline / QwenImagePipeline / HunyuanImagePipeline /
        # Flux2KleinPipeline); local dir takes priority, default falls back to a HuggingFace repo id (auto-download)
        "vae_pretrained_path":     (f"{_MODELS_ROOT}/Z-Image-Turbo"
                                    if os.path.isdir(f"{_MODELS_ROOT}/Z-Image-Turbo")
                                    else "Tongyi-MAI/Z-Image-Turbo"),
        "vae_decode_dtype":        "bfloat16",   # bf16 / fp32
    }

    run_experiment(CONFIG)
