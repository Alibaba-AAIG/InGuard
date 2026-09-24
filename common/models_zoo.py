import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as tvm



def make_load_info(checkpoint_path=None, loaded=False, note=""):
    return {
        "checkpoint_path": checkpoint_path,
        "loaded": bool(loaded),
        "note": str(note),
        "missing_keys": [],
        "unexpected_keys": [],
        "unexpected_keys_after_remap": [],
        "skipped_shape_mismatch": [],
    }


# =========================
# Shared MLP head
# =========================
class MLPHead(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=None, dropout=0.0):
        super().__init__()
        hidden_dim = in_dim if hidden_dim is None else int(hidden_dim)
        layers = [nn.Linear(in_dim, hidden_dim), nn.GELU()]
        if dropout and float(dropout) > 0:
            layers.append(nn.Dropout(float(dropout)))
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class PE2MLPEncoder(nn.Module):
    """PE-MLP encoder: structurally identical to the encoder part of PromptMultiTaskMLP in train_prompt.py.

    Structure: proj(input_dim->hidden_dim) + GELU + LayerNorm -> N x (Linear+LN+GELU+Dropout+Residual)
    The classification heads (head_porn/gore/ip) are NOT part of this module.

    Purpose: in the PE2 scheme, load the trained PE-MLP encoder weights and freeze them.
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

    def forward(self, x):
        x = self.proj_ln(F.gelu(self.proj(x)))
        for block in self.blocks:
            x = x + block(x)
        return x


class PromptFusionModule(nn.Module):
    """Fusion module for visual features and the prompt text embedding.

    fusion_mode='concat':
        prompt_embeds [B, seq_len, H] → masked mean pool → proj → text_feat [B, D]
        concat(visual_feat, text_feat) → fusion_proj → [B, D]
        The PE can fully overwrite the visual semantics ("tail wagging the dog" risk).

    fusion_mode='add' (recommended for two-stage training):
        text_feat = proj(PE)  → [B, D]
        fused = visual_feat + scale * text_feat
        The PE can only correct/perturb; it cannot override the main visual signal.
        scale is a learnable parameter initialized small (e.g. 0.1) so the PE contributes little at first.

    fusion_mode='pe2' (the PE2 scheme):
        prompt_embeds → frozen PE-MLP encoder → [B, 1024] → pe2_proj → [B, D]
        fused = visual_feat + scale * pe2_proj(encoder(PE))
        The PE-MLP encoder weights are loaded from the train_prompt best ckpt and frozen;
        pe2_proj (1024->D) is randomly initialized and trained; fusion_scale is trainable.

    During training, text_feat is zeroed-out entirely with probability dropout_rate to prevent over-reliance on text.
    """
    def __init__(self, visual_dim, text_hidden_dim=4096, dropout_rate=0.3,
                 fusion_mode="concat", scale_init=0.1,
                 pe2_mlp_ckpt_path=None, pe2_mlp_hidden_dim=1024,
                 pe2_mlp_num_layers=3, pe2_mlp_dropout=0.1):
        super().__init__()
        self.fusion_mode = fusion_mode
        self.dropout_rate = dropout_rate

        if fusion_mode == "pe2":
            # ---- PE2 scheme ----
            # 1. frozen PE-MLP encoder (loaded from the train_prompt best ckpt)
            self.pe_mlp_encoder = PE2MLPEncoder(
                input_dim=text_hidden_dim,
                hidden_dim=pe2_mlp_hidden_dim,
                num_layers=pe2_mlp_num_layers,
                dropout=pe2_mlp_dropout,
            )
            self.pe_mlp_load_info = None  # no load info by default; set inside _load_pe_mlp_encoder
            if pe2_mlp_ckpt_path:
                self._load_pe_mlp_encoder(pe2_mlp_ckpt_path)
            # freeze the encoder parameters
            for p in self.pe_mlp_encoder.parameters():
                p.requires_grad = False
            # 2. trainable projection layer (1024 -> visual_dim)
            self.text_proj = nn.Sequential(
                nn.Linear(pe2_mlp_hidden_dim, visual_dim),
                nn.GELU(),
            )
            # 3. learnable scale
            self.fusion_scale = nn.Parameter(torch.tensor(scale_init))
        else:
            # ---- concat / add modes (original logic unchanged) ----
            self.text_proj = nn.Sequential(
                nn.Linear(text_hidden_dim, visual_dim),
                nn.GELU(),
            )
            if fusion_mode == "concat":
                self.fusion_proj = nn.Sequential(
                    nn.Linear(visual_dim * 2, visual_dim),
                    nn.GELU(),
                )
            elif fusion_mode == "add":
                self.fusion_scale = nn.Parameter(torch.tensor(scale_init))
            else:
                raise ValueError(f"Unknown fusion_mode: {fusion_mode!r}, expected 'concat', 'add', or 'pe2'")

    def _load_pe_mlp_encoder(self, ckpt_path):
        """Load PE-MLP encoder weights from the train_prompt checkpoint.

        ckpt payload structure (from train_prompt.py):
            model_state_dict: PromptMultiTaskMLP.state_dict()
                proj.weight/bias, proj_ln.weight/bias, blocks.{0,1,2}.{0,1}.weight/bias
                head_porn/gore/ip.* (not needed; filtered out)
        """
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = ckpt.get("model_state_dict", ckpt)
        ckpt_global_step = ckpt.get("global_step", "?")
        ckpt_acc_avg = ckpt.get("acc_avg", "?")
        ckpt_test_loss = ckpt.get("test_loss", "?")
        ckpt_input_dim = ckpt.get("input_dim", "?")
        ckpt_hidden_dim = ckpt.get("hidden_dim", "?")
        ckpt_num_layers = ckpt.get("num_layers", "?")
        # filter: keep only the encoder part (proj, proj_ln, blocks); drop the classification heads
        encoder_sd = {k: v for k, v in sd.items()
                      if k.startswith("proj.") or k.startswith("proj_ln.") or k.startswith("blocks.")}
        missing, unexpected = self.pe_mlp_encoder.load_state_dict(encoder_sd, strict=False)
        loaded = len(encoder_sd) - len(missing)
        # store structured load info for print_pe_mlp_load_summary
        encoder_total = sum(p.numel() for p in self.pe_mlp_encoder.parameters())
        self.pe_mlp_load_info = {
            "checkpoint_path": ckpt_path,
            "loaded": True,
            "global_step": ckpt_global_step,
            "test_loss": ckpt_test_loss,
            "acc_avg": ckpt_acc_avg,
            "input_dim": ckpt_input_dim,
            "hidden_dim": ckpt_hidden_dim,
            "num_layers": ckpt_num_layers,
            "total_keys_in_ckpt": len(sd),
            "encoder_keys_loaded": loaded,
            "encoder_keys_total": len(encoder_sd),
            "missing_keys": list(missing),
            "unexpected_keys": list(unexpected),
            "encoder_total_params": encoder_total,
            "note": f"PE-MLP encoder (proj+proj_ln+blocks) loaded, head_* filtered out",
        }
        print(f"[PE2] PE-MLP encoder loaded from {ckpt_path}")
        print(f"  loaded {loaded}/{len(encoder_sd)} keys, missing={len(missing)}, unexpected={len(unexpected)}")
        if missing:
            print(f"  [WARN] missing keys: {missing[:5]}{'...' if len(missing) > 5 else ''}")

    def forward(self, visual_feat, prompt_embeds, training=False, prompt_mask=None):
        # prompt_embeds: [B, D] (pre-pooled) or [B, seq_len, H] (needs pooling)
        # prompt_mask: [B, seq_len] (1=valid, 0=padding) or None
        if prompt_embeds.ndim == 2:
            # masked mean pooling already done in the dataset; use directly
            text_feat = prompt_embeds.float()
        elif prompt_mask is not None:
            mask_f = prompt_mask.float().unsqueeze(-1)  # [B, seq_len, 1]
            valid_sum = mask_f.sum(dim=1).clamp(min=1.0)  # [B, 1]
            text_feat = (prompt_embeds.float() * mask_f).sum(dim=1) / valid_sum  # [B, H]
        else:
            text_feat = prompt_embeds.float().mean(dim=1)  # [B, H]

        if self.fusion_mode == "pe2":
            # PE-MLP encoder is frozen; do not track gradients
            with torch.no_grad():
                text_feat = self.pe_mlp_encoder(text_feat)

        text_feat = self.text_proj(text_feat)
        if training and self.dropout_rate > 0:
            mask = (torch.rand(text_feat.size(0), 1, device=text_feat.device) > self.dropout_rate).float()
            text_feat = text_feat * mask

        if self.fusion_mode == "concat":
            return self.fusion_proj(torch.cat([visual_feat, text_feat], dim=-1))
        else:  # add / pe2
            return visual_feat + self.fusion_scale * text_feat


def _maybe_build_prompt_fusion(prompt_fusion_cfg, visual_dim):
    if not prompt_fusion_cfg:
        return None
    return PromptFusionModule(
        visual_dim=visual_dim,
        text_hidden_dim=prompt_fusion_cfg.get("text_dim", 4096),
        dropout_rate=prompt_fusion_cfg.get("dropout", 0.3),
        fusion_mode=prompt_fusion_cfg.get("fusion_mode", "concat"),
        scale_init=prompt_fusion_cfg.get("scale_init", 0.1),
        pe2_mlp_ckpt_path=prompt_fusion_cfg.get("pe2_mlp_ckpt_path"),
        pe2_mlp_hidden_dim=prompt_fusion_cfg.get("pe2_mlp_hidden_dim", 1024),
        pe2_mlp_num_layers=prompt_fusion_cfg.get("pe2_mlp_num_layers", 3),
        pe2_mlp_dropout=prompt_fusion_cfg.get("pe2_mlp_dropout", 0.1),
    )


def _apply_prompt_fusion(prompt_fusion, visual_feat, prompt_embeds, training, prompt_mask=None):
    if prompt_fusion is not None and prompt_embeds is not None:
        return prompt_fusion(visual_feat, prompt_embeds, training, prompt_mask=prompt_mask)
    return visual_feat



# =========================
# Middle-frame Multi-task (convnext_* / vit_* / resnet*)
# =========================

# torchvision pretrained weight mapping (IN1K_V1). Backbones not listed raise NotImplementedError.
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

# =========================
# Backbone family registry (single source of truth)
# =========================
# backbone_name → family: convnext / vit / swin_v2 / resnet
BACKBONE_FAMILY = {
    "convnext_tiny": "convnext",  "convnext_small": "convnext",
    "convnext_base": "convnext", "convnext_large": "convnext",
    "vit_b_16": "vit",           "vit_l_16": "vit",
    "swin_v2_t": "swin_v2",      "swin_v2_s": "swin_v2",
    "swin_v2_b": "swin_v2",
    "resnet18": "resnet",        "resnet50": "resnet",
    "resnet101": "resnet",       "resnet152": "resnet",
}

# family -> default resolution
BACKBONE_RESOLUTION = {
    "convnext": 224, "vit": 224, "swin_v2": 256, "resnet": 224,
}

# valid backbone set (auto-derived from BACKBONE_FAMILY)
VALID_BACKBONES = set(BACKBONE_FAMILY.keys())


def get_backbone_family(name):
    """Return the backbone family name: convnext / vit / swin_v2 / resnet / unknown"""
    return BACKBONE_FAMILY.get(name, "unknown")


def _build_middle_frame_backbone(name, pretrained, freeze_backbone):
    """Factory: return (backbone_module, feat_dim) by name; the backbone itself already has its final
    classification layer replaced by Identity, so callers can do backbone(x) directly to get [B, feat_dim] features.
    Supports convnext_{tiny,small,base,large} / vit_{b,l}_16 / swin_v2_{t,s,b} / resnet{18,50,101,152}.
    Everything loads through official torchvision; no timm dependency.
    """
    if name not in _TVM_WEIGHTS:
        raise ValueError(f"Unknown backbone_middle_frame: {name!r}. "
                         f"supported: {list(_TVM_WEIGHTS.keys())}")

    ctor_name, weights_attr = _TVM_WEIGHTS[name]
    weights = getattr(tvm, weights_attr).IMAGENET1K_V1 if pretrained else None
    m = getattr(tvm, ctor_name)(weights=weights)

    family = get_backbone_family(name)
    if family == "convnext":
        feat_dim = m.classifier[2].in_features
        m.classifier[2] = nn.Identity()        # keep LayerNorm2d + Flatten; only drop the Linear
    elif family == "vit":
        feat_dim = m.heads[0].in_features      # heads = Sequential(Linear)
        m.heads = nn.Identity()               # forward(x) directly gives [B, feat_dim] (CLS token)
    elif family == "swin_v2":
        feat_dim = m.head.in_features
        m.head = nn.Identity()                # forward(x) directly gives [B, feat_dim]
    else:                                       # resnet*
        feat_dim = m.fc.in_features
        m.fc = nn.Identity()                    # forward(x) directly gives [B, feat_dim]

    if freeze_backbone:
        for p in m.parameters():
            p.requires_grad = False
    return m, feat_dim


class MiddleFrameMultiTaskModel(nn.Module):
    """Middle-frame classification model; uniformly attaches 3 MLP heads (porn / gore / ip).
    Supported backbones: convnext_{tiny,small,base,large} / vit_{b,l}_16 / resnet{18,50,101,152}.
    Each backbone's final classification layer is already replaced by Identity in the factory, so forward(x) -> [B, feat_dim].
    """
    def __init__(self, backbone_name, pretrained, dropout=0.1, freeze_backbone=False,
                 prompt_fusion_cfg=None, num_porn_classes=2, num_gore_classes=2):
        super().__init__()
        self.backbone_name = backbone_name
        self.backbone, feat_dim = _build_middle_frame_backbone(backbone_name, pretrained, freeze_backbone)
        self.feat_dim = feat_dim
        self.dropout = nn.Dropout(dropout)
        self.head_porn = MLPHead(feat_dim, num_porn_classes, hidden_dim=feat_dim, dropout=0.0)
        self.head_gore = MLPHead(feat_dim, num_gore_classes, hidden_dim=feat_dim, dropout=0.0)
        self.head_ip   = MLPHead(feat_dim, 6, hidden_dim=feat_dim, dropout=0.0)
        self.prompt_fusion = _maybe_build_prompt_fusion(prompt_fusion_cfg, feat_dim)

    def forward(self, x, prompt_embeds=None, prompt_mask=None):
        if x.ndim != 4:
            raise ValueError(f"Expected 4D input [B,C,H,W], got {tuple(x.shape)}")
        feat = self.dropout(self.backbone(x))
        feat = _apply_prompt_fusion(self.prompt_fusion, feat, prompt_embeds, self.training, prompt_mask=prompt_mask)
        return self.head_porn(feat), self.head_gore(feat), self.head_ip(feat)


def expand_conv_in_chans_weight(weight, new_in_chans):
    """Expand the input-channel weights of any conv first layer: from old in_chans to new_in_chans.
    The first min(3, old) channels reuse the old weights; extra channels are filled with the mean of the first 3 channels.
    Works for the ViT conv_proj and CNN stems (resnet conv1 / convnext features[0][0]).
    """
    out_c, old_in_chans, k1, k2 = weight.shape
    if old_in_chans == new_in_chans:
        return weight
    new_weight = torch.zeros((out_c, new_in_chans, k1, k2), dtype=weight.dtype, device=weight.device)
    if new_in_chans >= 3:
        if old_in_chans >= 3:
            new_weight[:, :3] = weight[:, :3]
            if new_in_chans > 3:
                mean = weight[:, :3].mean(dim=1, keepdim=True)
                new_weight[:, 3:] = mean.repeat(1, new_in_chans - 3, 1, 1)
        else:
            new_weight[:, :old_in_chans] = weight
            if new_in_chans > old_in_chans:
                mean = weight.mean(dim=1, keepdim=True)
                new_weight[:, old_in_chans:] = mean.repeat(1, new_in_chans - old_in_chans, 1, 1)
    else:
        new_weight[:] = weight[:, :new_in_chans]
    return new_weight


# =========================
# Latent + CNN (no adapter; the CNN consumes the latent directly; stem channels expanded via expand_conv_in_chans_weight)
# =========================


class ProgressiveStem(nn.Module):
    """Progressive multi-layer stem: stepwise channel expansion + stepwise downsampling.

    stem_channels example: [16, 32, 64, 128]
      - first layer:  Conv(16->32, k=3, s=1, p=1) + LN + GELU   (keeps resolution)
      - middle layer: Conv(32->64, k=3, s=2, p=1) + LN + GELU   (2x downsample)
      - last layer:   Conv(64->128, k=3, s=2, p=1) + LN          (2x downsample, no activation)
    Total downsampling = 4x; output matches the single stem [B, out_ch, H/4, W/4].

    Args:
        stem_channels: per-layer channel list, e.g. [16, 32, 64, 128].
                       First element = input channels, last = output channels.
        stem_init: init scheme, "trunc_normal" | "kaiming".
    """
    def __init__(self, stem_channels, stem_init="trunc_normal"):
        super().__init__()
        assert len(stem_channels) >= 2, "stem_channels needs at least [in_ch, out_ch]"
        layers = nn.ModuleList()
        num_transitions = len(stem_channels) - 1
        for i in range(num_transitions):
            in_c = stem_channels[i]
            out_c = stem_channels[i + 1]
            # first layer stride=1 keeps resolution; later layers stride=2 downsample
            stride = 1 if i == 0 else 2
            conv = nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1, bias=False)
            norm = nn.LayerNorm(out_c)  # channel-last LayerNorm; permuted in forward
            # the last layer has no activation (aligned with the ConvNeXt stem: LN right after stem, no GELU)
            if i < num_transitions - 1:
                layers.append(nn.ModuleDict({"conv": conv, "norm": norm, "act": nn.GELU()}))
            else:
                layers.append(nn.ModuleDict({"conv": conv, "norm": norm}))
        self.layers = layers
        self.out_channels = stem_channels[-1]
        self._init_weights(stem_init)

    def _init_weights(self, stem_init):
        for layer_dict in self.layers:
            conv = layer_dict["conv"]
            if stem_init == "trunc_normal":
                nn.init.trunc_normal_(conv.weight, std=0.02)
            elif stem_init == "kaiming":
                nn.init.kaiming_normal_(conv.weight, mode="fan_out", nonlinearity="relu")
            else:
                raise ValueError(f"ProgressiveStem does not support stem_init={stem_init!r}")

    def forward(self, x):
        # x: [B, C, H, W]
        for layer_dict in self.layers:
            x = layer_dict["conv"](x)              # [B, out_c, H', W']
            # LayerNorm needs channel-last: [B, H', W', C]
            x = x.permute(0, 2, 3, 1)
            x = layer_dict["norm"](x)
            x = x.permute(0, 3, 1, 2)              # back to [B, C, H', W']
            if "act" in layer_dict:
                x = layer_dict["act"](x)
        return x


def _init_single_stem(conv, stem_init, in_chans, old_weight, old_bias):
    """Single-layer stem init strategies.
    Args:
        conv: the newly created Conv2d
        stem_init: "expand_in1k" | "trunc_normal" | "kaiming"
        in_chans: target input channel count
        old_weight: the original IN1K 3ch stem weights (expand_in1k only)
        old_bias: the original IN1K stem bias (expand_in1k only)
    """
    with torch.no_grad():
        if stem_init == "expand_in1k":
            conv.weight.copy_(expand_conv_in_chans_weight(old_weight, in_chans))
            if old_bias is not None and conv.bias is not None:
                conv.bias.copy_(old_bias)
        elif stem_init == "trunc_normal":
            nn.init.trunc_normal_(conv.weight, std=0.02)
            if conv.bias is not None:
                nn.init.zeros_(conv.bias)
        elif stem_init == "kaiming":
            nn.init.kaiming_normal_(conv.weight, mode="fan_out", nonlinearity="relu")
            if conv.bias is not None:
                nn.init.zeros_(conv.bias)
        else:
            raise ValueError(f"unsupported stem_init={stem_init!r}")


def _build_cnn_for_latent(name, in_chans, pretrained,
                           stem_type="single", stem_channels=None, stem_init="expand_in1k"):
    """Build a backbone for latents (supports ConvNeXt / ViT / ResNet).

    Args:
        name: backbone name (convnext_base / vit_b_16 / resnet50, etc.)
        in_chans: input channel count
        pretrained: whether to load IN1K pretrained weights
        stem_type: "single" = one Conv2d stem | "progressive" = progressive multi-layer stem
                   ViT only supports "single" (replaces conv_proj)
        stem_channels: per-layer channel list in progressive mode, e.g. [16, 32, 64, 128].
                       Only effective with stem_type="progressive".
        stem_init: init scheme:
                   - "expand_in1k": reuse IN1K weights + mean fill (only meaningful for single)
                   - "trunc_normal": truncated normal, std=0.02
                   - "kaiming": Kaiming normal

    Returns:
        (backbone_module, feat_dim):
            - stem_type="single": backbone is the full torchvision model (stem/conv_proj already replaced)
            - stem_type="progressive": backbone is nn.Sequential(stem, rest_of_backbone)
    """
    if name not in _TVM_WEIGHTS:
        raise ValueError(f"latent CNN backbone does not support {name!r}; supported: {list(_TVM_WEIGHTS.keys())}")

    ctor_name, weights_attr = _TVM_WEIGHTS[name]
    weights = getattr(tvm, weights_attr).IMAGENET1K_V1 if pretrained else None
    m = getattr(tvm, ctor_name)(weights=weights)

    family = get_backbone_family(name)

    if stem_type == "progressive":
        # --- progressive stem: replace the original stem with ProgressiveStem ---
        if family == "vit":
            raise ValueError("ViT backbone does not support stem_type='progressive'; use stem_type='single'")
        if stem_channels is None:
            stem_channels = [in_chans, 32, 64, 128]
        assert stem_channels[0] == in_chans, (
            f"stem_channels[0]={stem_channels[0]} must equal in_chans={in_chans}")
        prog_stem = ProgressiveStem(stem_channels, stem_init=stem_init)
        expected_out_ch = stem_channels[-1]

        if family == "convnext":
            # ConvNeXt stem = features[0] = Sequential(Conv2d, LayerNorm2d)
            # original stem output channels = features[0][0].out_channels (usually 128)
            orig_out_ch = m.features[0][0].out_channels
            assert expected_out_ch == orig_out_ch, (
                f"stem_channels[-1]={expected_out_ch} must equal the backbone stem output channels {orig_out_ch}")
            # replace features[0] with the progressive stem (with LN)
            # note: the original features[0] = Sequential(Conv2d, LayerNorm2d); our ProgressiveStem already has a built-in LN
            # ConvNeXt features[0][1] is LayerNorm2d and the original flow is Conv->LN
            # ProgressiveStem's last layer already has LN, so we replace features[0] wholesale
            m.features[0] = prog_stem
            feat_dim = m.classifier[2].in_features
            m.classifier[2] = nn.Identity()
        elif family == "swin_v2":
            # SwinV2 stem = features[0] = Sequential(Conv2d, Permute, LayerNorm)
            # original stem output channels = features[0][0].out_channels (T/S=96, B=128)
            orig_out_ch = m.features[0][0].out_channels
            assert expected_out_ch == orig_out_ch, (
                f"stem_channels[-1]={expected_out_ch} must equal the backbone stem output channels {orig_out_ch}")
            m.features[0] = prog_stem
            feat_dim = m.head.in_features
            m.head = nn.Identity()
        else:  # resnet*
            orig_out_ch = m.conv1.out_channels  # usually 64
            assert expected_out_ch == orig_out_ch, (
                f"stem_channels[-1]={expected_out_ch} must equal the backbone conv1 output channels {orig_out_ch}")
            # ResNet: conv1 -> bn1 -> relu -> maxpool
            # the progressive stem already has LN + GELU; replace conv1 + bn1 + relu
            m.conv1 = prog_stem
            m.bn1 = nn.Identity()
            m.relu = nn.Identity()
            feat_dim = m.fc.in_features
            m.fc = nn.Identity()

    elif stem_type == "single":
        # --- single stem: current logic ---
        if family == "convnext":
            old_stem = m.features[0][0]
            new_stem = nn.Conv2d(
                in_chans, old_stem.out_channels,
                kernel_size=old_stem.kernel_size, stride=old_stem.stride,
                padding=old_stem.padding, bias=(old_stem.bias is not None),
            )
            _init_single_stem(new_stem, stem_init, in_chans,
                              old_stem.weight.data, old_stem.bias.data if old_stem.bias is not None else None)
            m.features[0][0] = new_stem
            feat_dim = m.classifier[2].in_features
            m.classifier[2] = nn.Identity()
        elif family == "swin_v2":
            # SwinV2: stem = features[0][0] (Conv2d patch embed)
            old_stem = m.features[0][0]
            new_stem = nn.Conv2d(
                in_chans, old_stem.out_channels,
                kernel_size=old_stem.kernel_size, stride=old_stem.stride,
                padding=old_stem.padding, bias=(old_stem.bias is not None),
            )
            _init_single_stem(new_stem, stem_init, in_chans,
                              old_stem.weight.data, old_stem.bias.data if old_stem.bias is not None else None)
            m.features[0][0] = new_stem
            feat_dim = m.head.in_features
            m.head = nn.Identity()
        elif family == "vit":
            # ViT: replace the conv_proj (patch embedding) input channels
            old_conv = m.conv_proj
            new_conv = nn.Conv2d(
                in_chans, old_conv.out_channels,
                kernel_size=old_conv.kernel_size, stride=old_conv.stride,
                padding=old_conv.padding, bias=(old_conv.bias is not None),
            )
            _init_single_stem(new_conv, stem_init, in_chans,
                              old_conv.weight.data, old_conv.bias.data if old_conv.bias is not None else None)
            m.conv_proj = new_conv
            feat_dim = m.heads[0].in_features
            m.heads = nn.Identity()
        else:  # resnet*
            old_stem = m.conv1
            new_stem = nn.Conv2d(
                in_chans, old_stem.out_channels,
                kernel_size=old_stem.kernel_size, stride=old_stem.stride,
                padding=old_stem.padding, bias=(old_stem.bias is not None),
            )
            _init_single_stem(new_stem, stem_init, in_chans,
                              old_stem.weight.data, old_stem.bias.data if old_stem.bias is not None else None)
            m.conv1 = new_stem
            feat_dim = m.fc.in_features
            m.fc = nn.Identity()
    else:
        raise ValueError(f"unsupported stem_type={stem_type!r}; options: 'single' / 'progressive'")

    return m, feat_dim


class LatentCNNMultiTask(nn.Module):
    """Latent + backbone base: backbone (stem/conv_proj re-channelled) + 3 MLP heads. No adapter.
    Supports ConvNeXt / ViT / ResNet. forward(x) takes a single [B, C, H, W] frame and returns [B, feat_dim].

    Args:
        stem_type: "single" | "progressive" (ViT supports single only)
        stem_channels: only effective in progressive mode, e.g. [16, 32, 64, 128]
        stem_init: "expand_in1k" | "trunc_normal" | "kaiming"
    """
    def __init__(self, backbone_name, in_chans, pretrained,
                 stem_type="single", stem_channels=None, stem_init="expand_in1k",
                 num_porn_classes=2, num_gore_classes=2):
        super().__init__()
        self.backbone_name = backbone_name
        self.stem_type = stem_type
        self.backbone, feat_dim = _build_cnn_for_latent(
            backbone_name, in_chans, pretrained,
            stem_type=stem_type, stem_channels=stem_channels, stem_init=stem_init,
        )
        self.feat_dim = feat_dim
        # heads use dropout=0.0 internally: dropout is controlled uniformly by the outer MultiTaskWrapperCNN.dropout (avoids double dropout)
        self.head_porn = MLPHead(feat_dim, num_porn_classes, hidden_dim=feat_dim, dropout=0.0)
        self.head_gore = MLPHead(feat_dim, num_gore_classes, hidden_dim=feat_dim, dropout=0.0)
        self.head_ip = MLPHead(feat_dim, 6, hidden_dim=feat_dim, dropout=0.0)

    def extract_feat(self, x):
        # x: [B, C, H, W] → backbone(x) → [B, feat_dim]
        return self.backbone(x)


# =========================
# Stage-wise feature extraction for ConvNeXt distillation
# =========================
def extract_convnext_stage_features(model, x):
    """Extract intermediate per-stage features from a ConvNeXt model.

    ConvNeXt features structure (torchvision):
        features[0]: stem (Conv/ProgressiveStem + LN)  → [B, 128, H/4, W/4]
        features[1]: stage 1 blocks                     → [B, 128, H/4, W/4]
        features[2]: downsample 1→2                     → [B, 256, H/8, W/8]
        features[3]: stage 2 blocks                     → [B, 256, H/8, W/8]
        features[4]: downsample 2→3                     → [B, 512, H/16, W/16]
        features[5]: stage 3 blocks                     → [B, 512, H/16, W/16]
        features[6]: downsample 3→4                     → [B, 1024, H/32, W/32]
        features[7]: stage 4 blocks                     → [B, 1024, H/32, W/32]

    When teacher and student both take 128x128 inputs, the per-stage spatial resolutions align exactly:
        stage1: 128ch, 32×32
        stage2: 256ch, 16×16
        stage3: 512ch, 8×8
        stage4: 1024ch, 4×4

    Args:
        model: a ConvNeXt model (teacher or student backbone).
               Must have model.features (nn.Sequential), model.avgpool, model.classifier.
        x: input tensor [B, C, H, W]

    Returns:
        dict: {
            "stage1": [B, 128, H/4, W/4],
            "stage2": [B, 256, H/8, W/8],
            "stage3": [B, 512, H/16, W/16],
            "stage4": [B, 1024, H/32, W/32],
            "gap":    [B, feat_dim],
        }
    """
    feats = {}
    x = model.features[0](x)   # stem
    x = model.features[1](x)   # stage 1 blocks
    feats["stage1"] = x

    x = model.features[2](x)   # downsample
    x = model.features[3](x)   # stage 2 blocks
    feats["stage2"] = x

    x = model.features[4](x)   # downsample
    x = model.features[5](x)   # stage 3 blocks
    feats["stage3"] = x

    x = model.features[6](x)   # downsample
    x = model.features[7](x)   # stage 4 blocks
    feats["stage4"] = x

    # GAP → LayerNorm → Flatten
    x = model.avgpool(x)       # [B, C, 1, 1]
    x = model.classifier[0](x) # LayerNorm2d
    x = model.classifier[1](x) # Flatten → [B, feat_dim]
    feats["gap"] = x

    return feats


# =========================
# Stage-wise feature extraction for ViT distillation
# =========================
def extract_vit_stage_features(model, x):
    """Extract intermediate per-stage features from a torchvision ViT model.

    The transformer blocks are split evenly into 4 groups; features are taken at the end of each group.
    ViT-B/16: 12 blocks → 3 blocks/group
    ViT-L/16: 24 blocks → 6 blocks/group

    Same return format as extract_convnext_stage_features; downstream distillation losses need no change.

    Args:
        model: a torchvision ViT model (teacher or student backbone).
               Must have model.conv_proj, model.class_token, model.encoder (with pos_embedding,
               dropout, layers, ln).
        x: input tensor [B, C, H, W] (teacher: 3ch RGB; student: latent_ch channels)

    Returns:
        dict: {
            "stage1": [B, D, H/P, W/P],
            "stage2": [B, D, H/P, W/P],
            "stage3": [B, D, H/P, W/P],
            "stage4": [B, D, H/P, W/P],
            "gap":    [B, D],
        }
        where D = embed_dim, P = patch_size (16), H/P = W/P = 14 (for 224x224 inputs).
    """
    feats = {}
    B = x.shape[0]

    # Patch embedding: [B, C, H, W] → [B, D, H/P, W/P] → [B, N, D]
    x = model.conv_proj(x)
    H, W = x.shape[2], x.shape[3]
    D = x.shape[1]
    x = x.reshape(B, D, -1).permute(0, 2, 1)  # [B, N, D]

    # Prepend CLS token
    cls_token = model.class_token.expand(B, -1, -1)  # [B, 1, D]
    x = torch.cat([cls_token, x], dim=1)  # [B, N+1, D]

    # Add positional embedding + encoder dropout
    x = x + model.encoder.pos_embedding
    x = model.encoder.dropout(x)

    # Iterate through transformer blocks in 4 groups
    n_blocks = len(model.encoder.layers)
    group_size = n_blocks // 4
    for i in range(n_blocks):
        x = model.encoder.layers[i](x)
        if (i + 1) % group_size == 0:
            stage_idx = (i + 1) // group_size
            # Remove CLS token, reshape to spatial format [B, D, H, W]
            feat = x[:, 1:, :]  # [B, N, D]
            feat = feat.permute(0, 2, 1).reshape(B, D, H, W)
            feats[f"stage{stage_idx}"] = feat

    # Final LayerNorm + CLS token as gap
    x = model.encoder.ln(x)
    feats["gap"] = x[:, 0]  # [B, D]

    return feats


# =========================
# Stage-wise feature extraction for SwinV2 distillation
# =========================
def extract_swin_stage_features(model, x):
    """Extract intermediate per-stage features from a torchvision SwinV2 model.

    SwinV2 features structure (torchvision):
        features[0]: PatchEmbed (Conv2d + Permute + LayerNorm) → [B, H/4, W/4, C] (BHWC)
        features[1]: Stage 1 blocks                     → [B, H/4, W/4, C] (BHWC)
        features[2]: PatchMergingV2 (downsample 1→2)    → [B, H/8, W/8, 2C] (BHWC)
        features[3]: Stage 2 blocks                     → [B, H/8, W/8, 2C] (BHWC)
        features[4]: PatchMergingV2 (downsample 2→3)    → [B, H/16, W/16, 4C] (BHWC)
        features[5]: Stage 3 blocks                     → [B, H/16, W/16, 4C] (BHWC)
        features[6]: PatchMergingV2 (downsample 3→4)    → [B, H/32, W/32, 8C] (BHWC)
        features[7]: Stage 4 blocks                     → [B, H/32, W/32, 8C] (BHWC)

    Same return format as ConvNeXt's extract_convnext_stage_features (BCHW);
    downstream distillation losses need no change.

    Args:
        model: a SwinV2 model (teacher or student backbone).
               Must have model.features (nn.Sequential), model.norm, model.permute, model.avgpool.
        x: input tensor [B, C, H, W]

    Returns:
        dict: {
            "stage1": [B, C, H/4, W/4],
            "stage2": [B, 2C, H/8, W/8],
            "stage3": [B, 4C, H/16, W/16],
            "stage4": [B, 8C, H/32, W/32],
            "gap":    [B, 8C],
        }
        SwinV2-T/S: C=96, 8C=768; SwinV2-B: C=128, 8C=1024
    """
    feats = {}
    x = model.features[0](x)   # patch embed → BHWC
    x = model.features[1](x)   # stage 1 blocks
    feats["stage1"] = x.permute(0, 3, 1, 2)  # BHWC → BCHW

    x = model.features[2](x)   # downsample
    x = model.features[3](x)   # stage 2 blocks
    feats["stage2"] = x.permute(0, 3, 1, 2)

    x = model.features[4](x)   # downsample
    x = model.features[5](x)   # stage 3 blocks
    feats["stage3"] = x.permute(0, 3, 1, 2)

    x = model.features[6](x)   # downsample
    x = model.features[7](x)   # stage 4 blocks
    feats["stage4"] = x.permute(0, 3, 1, 2)

    # final norm + permute + avgpool + flatten
    x = model.norm(x)           # LayerNorm
    x = model.permute(x)        # BHWC → BCHW
    x = model.avgpool(x)        # [B, 8C, 1, 1]
    x = model.flatten(x)        # [B, 8C]
    feats["gap"] = x

    return feats


def extract_stage_features(model, x):
    """Unified dispatch: pick the ConvNeXt / ViT / SwinV2 stage-feature extractor by model type.
    Shared by the distillation pretraining teacher/student so callers need not branch on backbone type.
    """
    if hasattr(model, 'encoder') and hasattr(model.encoder, 'layers'):
        # torchvision ViT
        return extract_vit_stage_features(model, x)
    elif hasattr(model, 'permute') and hasattr(model, 'norm'):
        # torchvision SwinV2 (has permute and norm attributes, unlike ConvNeXt)
        return extract_swin_stage_features(model, x)
    else:
        # ConvNeXt (torchvision)
        return extract_convnext_stage_features(model, x)


# =========================
# Teacher backbone for distillation (frozen, 3ch RGB, IN1K pretrained)
# =========================
def build_teacher_backbone(backbone_name="convnext_base", pretrained=True, ckpt_path=None):
    """Build a frozen teacher backbone (3ch RGB).
    Returns (module, feat_dim); all parameters have requires_grad=False.

    Args:
        backbone_name: teacher network (convnext_base / vit_b_16, etc.)
        pretrained: whether to load IN1K pretrained weights
        ckpt_path: optional custom ckpt path. When given, it overrides the IN1K weights after loading.
                   Two formats are supported:
                     - {"state_dict": {...}}  (backbone-only, keys like backbone.xxx)
                     - a raw state_dict

    module supports two call styles:
      - module(x) -> [B, feat_dim]  (final features after GAP / CLS token)
      - extract_stage_features(module, x) -> dict of stage features
        (for stage-wise distillation; teacher_image_size must match target_hw for spatial alignment)
    """
    if backbone_name not in _TVM_WEIGHTS:
        raise ValueError(f"teacher backbone does not support {backbone_name!r}; supported: {list(_TVM_WEIGHTS.keys())}")

    ctor_name, weights_attr = _TVM_WEIGHTS[backbone_name]
    weights = getattr(tvm, weights_attr).IMAGENET1K_V1 if pretrained else None
    m = getattr(tvm, ctor_name)(weights=weights)

    family = get_backbone_family(backbone_name)
    if family == "convnext":
        feat_dim = m.classifier[2].in_features
        m.classifier[2] = nn.Identity()
    elif family == "vit":
        feat_dim = m.heads[0].in_features
        m.heads = nn.Identity()
    elif family == "swin_v2":
        feat_dim = m.head.in_features
        m.head = nn.Identity()
    else:  # resnet*
        feat_dim = m.fc.in_features
        m.fc = nn.Identity()

    # load the custom ckpt (safety-domain teacher etc.)
    if ckpt_path:
        import torch as _torch
        print(f"  [Teacher] loading custom ckpt: {ckpt_path}")
        ckpt = _torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt
        # handle the key prefix: backbone.features.0.xxx -> features.0.xxx
        cleaned = {}
        for k, v in state_dict.items():
            if k.startswith("backbone."):
                cleaned[k[len("backbone."):]] = v
            else:
                cleaned[k] = v
        missing, unexpected = m.load_state_dict(cleaned, strict=False)
        if missing:
            print(f"  [Teacher] Missing keys ({len(missing)}): {missing[:5]}...")
        if unexpected:
            print(f"  [Teacher] Unexpected keys ({len(unexpected)}): {unexpected[:5]}...")
        print(f"  [Teacher] ckpt loaded")

    # Freeze all parameters
    for p in m.parameters():
        p.requires_grad = False
    m.eval()

    return m, feat_dim

def get_latent_cnn_model(backbone_name, in_chans, pretrained, device,
                          checkpoint_path=None, freeze_backbone=False,
                          stem_type="single", stem_channels=None, stem_init="expand_in1k",
                          num_porn_classes=2, num_gore_classes=2):
    """Build a LatentCNNMultiTask + load the user's own ckpt (if any) + apply the freeze strategy.
    Note: torchvision already loads IN1K pretrained weights at model creation (when pretrained=True),
    so even if checkpoint_path is incompatible (e.g. feeding a ViT-latent ckpt to a CNN),
    the backbone still has usable IN1K weights — only the ckpt step updates nothing.
    """
    model = LatentCNNMultiTask(
        backbone_name=backbone_name, in_chans=in_chans, pretrained=pretrained,
        stem_type=stem_type, stem_channels=stem_channels, stem_init=stem_init,
        num_porn_classes=num_porn_classes, num_gore_classes=num_gore_classes,
    )
    note_parts = [f"backbone={backbone_name}", f"IN1K_pretrained={pretrained}",
                  f"stem_type={stem_type}"]
    load_info = make_load_info(
        checkpoint_path=checkpoint_path,
        loaded=bool(pretrained),
        note=f"latent cnn ({', '.join(note_parts)})",
    )

    if checkpoint_path:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
        state_dict = {(k[7:] if k.startswith("module.") else k): v for k, v in state_dict.items()}

        # strip the base. prefix (if the ckpt was saved from MultiTaskWrapperCNN)
        model_keys = set(model.state_dict().keys())
        if len(set(state_dict.keys()) & model_keys) == 0:
            stripped = {(k[5:] if k.startswith("base.") else k): v for k, v in state_dict.items()}
            if len(set(stripped.keys()) & model_keys) > 0:
                state_dict = stripped

        model_state = model.state_dict()
        filtered, skipped = {}, []
        for k, v in state_dict.items():
            if k not in model_state:
                continue
            # single-stem input channels may differ -> auto-expand
            # note: progressive-stem keys (features.0.layers.*.conv.weight) are skipped when shapes differ (no expansion)
            is_single_stem = k.endswith("conv1.weight") or k.endswith("features.0.0.weight") or k.endswith("conv_proj.weight")
            if is_single_stem and v.shape != model_state[k].shape:
                v = expand_conv_in_chans_weight(v, in_chans)
            if v.shape == model_state[k].shape:
                filtered[k] = v
            else:
                skipped.append((k, tuple(v.shape), tuple(model_state[k].shape)))

        if not filtered:
            # ckpt does not match the CNN structure at all (typical: feeding a ViT-latent ckpt to a CNN)
            # skip load_state_dict (avoids dumping hundreds of backbone keys into the missing_keys report)
            print(f"⚠️ [Latent CNN Load] model_ckpt has no key matching the CNN model structure "
                  f"(checked {len(state_dict)} keys); falling back to IN1K pretrained weights. "
                  f"If you have no CNN-latent ckpt of your own, set model_ckpt to an empty string.")
            load_info["note"] += f" | model_ckpt mismatch ({len(state_dict)} keys, 0 matched); ignored"
        else:
            missing, unexpected = model.load_state_dict(filtered, strict=False)
            # heads are always newly created (no pretrained carries them); filter them out of missing to reduce noise
            real_missing = [k for k in missing if not k.startswith("head_")]
            load_info["loaded"] = True
            load_info["missing_keys"] = real_missing
            load_info["unexpected_keys"] = list(unexpected)
            load_info["skipped_shape_mismatch"] = [
                {"key": k, "ckpt_shape": cs, "model_shape": ms} for k, cs, ms in skipped
            ]
            load_info["note"] += f" | model_ckpt loaded {len(filtered)}/{len(state_dict)} keys"

    if freeze_backbone:
        for p in model.parameters():
            p.requires_grad = False
        # the stem had its channels changed; keep it trainable
        if stem_type == "progressive":
            # the progressive stem lives at backbone.features[0] (convnext) or backbone.conv1 (resnet)
            # ViT has no progressive-stem support; already rejected in _build_cnn_for_latent
            bb_family = get_backbone_family(backbone_name)
            if bb_family in ("convnext", "swin_v2"):
                stem_module = model.backbone.features[0]
            else:  # resnet
                stem_module = model.backbone.conv1
        else:
            # single stem
            bb_family = get_backbone_family(backbone_name)
            if bb_family in ("convnext", "swin_v2"):
                stem_module = model.backbone.features[0][0]
            elif bb_family == "vit":
                stem_module = model.backbone.conv_proj
            else:  # resnet
                stem_module = model.backbone.conv1
        for p in stem_module.parameters():
            p.requires_grad = True
        # heads always stay trainable
        for h in (model.head_porn, model.head_gore, model.head_ip):
            for p in h.parameters():
                p.requires_grad = True
        load_info["note"] += " | backbone frozen except stem / heads"

    return model.to(device), load_info


class MultiTaskWrapperCNN(nn.Module):
    """Multi-task wrapper for the latent + CNN path:
        CNN(stem re-channelled) -> dropout -> (optional prompt_fusion) -> 3 heads
    Input: a single [B, C, H, W] latent frame; the CNN accepts any H/W.
    """
    def __init__(self, checkpoint_path, device, backbone_name, in_chans, pretrained,
                 dropout=0.1, freeze_backbone=False,
                 prompt_fusion_cfg=None,
                 stem_type="single", stem_channels=None, stem_init="expand_in1k",
                 num_porn_classes=2, num_gore_classes=2, **kwargs):
        super().__init__()
        self.base, self.base_load_info = get_latent_cnn_model(
            backbone_name=backbone_name, in_chans=in_chans, pretrained=pretrained,
            device=device, checkpoint_path=checkpoint_path, freeze_backbone=freeze_backbone,
            stem_type=stem_type, stem_channels=stem_channels, stem_init=stem_init,
            num_porn_classes=num_porn_classes, num_gore_classes=num_gore_classes,
        )
        self.feat_dim = self.base.feat_dim
        self.dropout = nn.Dropout(dropout)
        self.prompt_fusion = _maybe_build_prompt_fusion(prompt_fusion_cfg, self.feat_dim)

    def forward(self, x, prompt_embeds=None, prompt_mask=None):
        # x: [B, C, H, W] single-frame latent
        if x.ndim != 4:
            raise ValueError(f"MultiTaskWrapperCNN only supports 4D input [B,C,H,W], got {tuple(x.shape)}")
        feat = self.dropout(self.base.extract_feat(x))
        feat = _apply_prompt_fusion(self.prompt_fusion, feat, prompt_embeds, self.training, prompt_mask=prompt_mask)
        return self.base.head_porn(feat), self.base.head_gore(feat), self.base.head_ip(feat)



# =========================
# Optimizer
# =========================
def build_optimizer(model, config, backbone_prefixes=None, adapter_prefixes=None):
    """3-way parameter grouping: backbone / adapter / head.
    adapter_prefixes=None degrades to 2-way (same as the old behavior); adapter_lr=None makes adapters reuse head_lr.
    """
    base_lr = config["lr"]
    optimizer_type = config.get("optimizer_type", "adamw").lower()
    weight_decay = config.get("weight_decay", 0.05)
    backbone_prefixes = backbone_prefixes or []
    adapter_prefixes = adapter_prefixes or []

    backbone_params, adapter_params, head_params = [], [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(name.startswith(p) for p in backbone_prefixes):
            backbone_params.append(param)
        elif any(name.startswith(p) for p in adapter_prefixes):
            adapter_params.append(param)
        else:
            head_params.append(param)

    backbone_lr = config.get("backbone_lr", base_lr)
    head_lr = config.get("head_lr", base_lr)
    adapter_lr = config.get("latent_frames_adapter_lr", None)
    if adapter_lr is None:
        adapter_lr = head_lr

    param_groups = []
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": backbone_lr})
    if adapter_params:
        param_groups.append({"params": adapter_params, "lr": float(adapter_lr)})
    if head_params:
        param_groups.append({"params": head_params, "lr": head_lr})
    if not param_groups:
        raise ValueError("No trainable parameters found in model.")

    if optimizer_type == "adam":
        return optim.Adam(param_groups, lr=base_lr, weight_decay=weight_decay)
    if optimizer_type == "adamw":
        return optim.AdamW(param_groups, lr=base_lr, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer_type: {optimizer_type}")


# =========================
# Shared constants for the Wan VAE Decoder
# =========================
_VAE_DTYPE_MAP = {
    "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
    "float32": torch.float32,   "fp32": torch.float32,
    "float16": torch.float16,   "fp16": torch.float16,
}


def _resolve_vae_dtype(name):
    key = str(name).lower()
    if key not in _VAE_DTYPE_MAP:
        raise ValueError(f"vae_decode_dtype={name!r} is not supported; options: {list(_VAE_DTYPE_MAP)}")
    return _VAE_DTYPE_MAP[key]


# Constant for the Wan VAE Decoder's internal causal padding (source: CACHE_T = 2).
# The qwen early-stop path of ImageVAEDecoderModule needs it.
_WAN_VAE_CACHE_T = 2


# =========================
# Image VAE Decoder (shared by latent_to_image / latent_to_image_feat)
# =========================

# VAE latent channel counts per image model (used to decide in_chans in feat mode)
_IMAGE_VAE_LATENT_CHANS = {
    "z-image-turbo": 16,
    "qwen-image-2512": 16,
    "internvl-u": 16,       # reuses the Qwen VAE
    "hunyuan-image-2_1": 64,
    "flux2-klein-base-9b": 32,
}


class ImageVAEDecoderModule(nn.Module):
    """Image VAE decoder with multi-model dispatch.

    mode='image': full decode -> RGB [B, 3, H_img, W_img] in [-1, 1]
    mode='feat':  early-stop mid layer -> features (only qwen-image-2512 currently; output [B, 192, H_feat, W_feat])

    Per-model denormalization formulas (latent_x1 -> sample domain):
      z-image-turbo:      l_sample = latent_x1 / scaling_factor + shift_factor
      qwen-image-2512:    l_sample = latent_x1 * latents_std + latents_mean  (5D: [B,C,1,H,W])
      internvl-u:         l_sample = latent_x1 * latents_std + latents_mean  (5D: [B,C,1,H,W], reuses the Qwen VAE)
      hunyuan-image-2_1:  l_sample = latent_x1 / scaling_factor
      flux2-klein-base-9b: l_sample = latent_x1 (no scaling; decode directly)

    VAE weights are loaded in fp32, eval(), all parameters requires_grad=False;
    object.__setattr__ bypasses nn.Module submodule registration so state_dict / parameters stay clean.
    """
    _SUPPORTED_MODELS = ("z-image-turbo", "qwen-image-2512", "internvl-u", "hunyuan-image-2_1", "flux2-klein-base-9b")
    _FEAT_SUPPORTED = ("qwen-image-2512",)  # feat mode currently supports the WanVAE architecture only

    def __init__(self, model_name, vae_pretrained_path, mode, device, decode_dtype=torch.bfloat16):
        super().__init__()
        if model_name not in self._SUPPORTED_MODELS:
            raise ValueError(f"ImageVAEDecoderModule does not support model={model_name!r}; "
                             f"options: {self._SUPPORTED_MODELS}")
        if mode not in ("image", "feat"):
            raise ValueError(f"ImageVAEDecoderModule mode must be 'image'/'feat', got {mode!r}")
        if mode == "feat" and model_name not in self._FEAT_SUPPORTED:
            raise ValueError(
                f"ImageVAEDecoderModule mode='feat' currently supports only {self._FEAT_SUPPORTED}; "
                f"got model={model_name!r}. "
                f"Early-stop decoding for the other models' VAE decoders is not implemented yet."
            )

        self.model_name = model_name
        self.mode = mode
        self.decode_dtype = decode_dtype

        # deferred import so training other modes has no hard dependency
        vae = self._load_vae(model_name, vae_pretrained_path)
        vae.eval()
        for p in vae.parameters():
            p.requires_grad = False
        # bypass nn.Module registration so VAE parameters do not pollute state_dict
        object.__setattr__(self, "vae", vae)

        # register the denormalization params as buffers (they follow the device)
        self._register_denorm_params(model_name, vae, device)
        # move the VAE to the device manually (it is not an nn.Module submodule)
        self.vae.to(device)

    def _load_vae(self, model_name, path):
        """Load the VAE from the pipeline by model name and release the other components."""
        import gc
        if model_name == "z-image-turbo":
            from diffusers import ZImagePipeline
            pipe = ZImagePipeline.from_pretrained(path, torch_dtype=torch.float32, low_cpu_mem_usage=False)
            vae = pipe.vae
            del pipe; gc.collect()
        elif model_name in ("qwen-image-2512", "internvl-u"):
            # internvl-u reuses the Qwen VAE; the loading logic is identical
            from diffusers import QwenImagePipeline
            pipe = QwenImagePipeline.from_pretrained(path, torch_dtype=torch.float32)
            vae = pipe.vae
            del pipe; gc.collect()
        elif model_name == "hunyuan-image-2_1":
            from diffusers import HunyuanImagePipeline
            pipe = HunyuanImagePipeline.from_pretrained(path, torch_dtype=torch.float32)
            vae = pipe.vae
            del pipe; gc.collect()
        elif model_name == "flux2-klein-base-9b":
            from diffusers import Flux2KleinPipeline
            pipe = Flux2KleinPipeline.from_pretrained(path, torch_dtype=torch.float32)
            vae = pipe.vae
            del pipe; gc.collect()
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        return vae

    def _register_denorm_params(self, model_name, vae, device):
        """Read denormalization params from vae.config and register them as buffers."""
        if model_name == "z-image-turbo":
            sf = torch.tensor([vae.config.scaling_factor], dtype=torch.float32)
            sh = torch.tensor([vae.config.shift_factor], dtype=torch.float32)
            self.register_buffer("scaling_factor", sf)
            self.register_buffer("shift_factor", sh)
        elif model_name in ("qwen-image-2512", "internvl-u"):
            # internvl-u reuses the Qwen VAE; the denormalization params are identical
            z_dim = int(vae.config.z_dim)
            mean = torch.tensor(vae.config.latents_mean).view(1, z_dim, 1, 1, 1).float()
            std = torch.tensor(vae.config.latents_std).view(1, z_dim, 1, 1, 1).float()
            self.register_buffer("latents_mean", mean)
            self.register_buffer("latents_std", std)
        elif model_name == "hunyuan-image-2_1":
            sf = torch.tensor([vae.config.scaling_factor], dtype=torch.float32)
            self.register_buffer("scaling_factor", sf)
        elif model_name == "flux2-klein-base-9b":
            pass  # no scaling

    def _denormalize(self, latent_x1):
        """latent_x1 (normalized domain) -> sample domain (the input VAE decode expects)."""
        if self.model_name == "z-image-turbo":
            return latent_x1 / self.scaling_factor + self.shift_factor
        elif self.model_name in ("qwen-image-2512", "internvl-u"):
            # internvl-u reuses the Qwen VAE; the denormalization logic is identical
            # input [B, C, H, W] must be expanded to [B, C, 1, H, W] for the 3D VAE
            x = latent_x1.unsqueeze(2)  # [B, C, 1, H, W]
            return x * self.latents_std + self.latents_mean
        elif self.model_name == "hunyuan-image-2_1":
            return latent_x1 / self.scaling_factor
        elif self.model_name == "flux2-klein-base-9b":
            return latent_x1  # no extra scaling
        raise ValueError(f"Unknown model: {self.model_name}")

    def _autocast_ctx(self, ref_device):
        if ref_device.type == "cuda" and self.decode_dtype != torch.float32:
            return torch.autocast(device_type="cuda", dtype=self.decode_dtype, enabled=True)
        return torch.autocast(device_type="cpu", enabled=False)

    def _decode_to_up_blocks1_qwen(self, l_sample):
        """qwen-image-2512 early-stop decode (reuses the WanVAE logic, T=1 single frame).
        Input l_sample: [B, C, 1, H, W]; output: [B, 192, H_feat, W_feat]
        """
        # qwen-image-2512's VAE is AutoencoderKLWan, identical to the video version
        _, _, num_frame, _, _ = l_sample.shape
        self.vae.clear_cache()
        x = self.vae.post_quant_conv(l_sample)
        decoder = self.vae.decoder
        feat_cache = self.vae._feat_map

        outputs = []
        for i in range(num_frame):
            self.vae._conv_idx = [0]
            feat_idx = self.vae._conv_idx
            chunk = x[:, :, i:i + 1, :, :]

            idx = feat_idx[0]
            cache_x = chunk[:, :, -_WAN_VAE_CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat(
                    [feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x],
                    dim=2,
                )
            h = decoder.conv_in(chunk, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1

            first_chunk = (i == 0)
            h = decoder.mid_block(h, feat_cache=feat_cache, feat_idx=feat_idx)
            h = decoder.up_blocks[0](h, feat_cache=feat_cache, feat_idx=feat_idx, first_chunk=first_chunk)
            h = decoder.up_blocks[1](h, feat_cache=feat_cache, feat_idx=feat_idx, first_chunk=first_chunk)
            outputs.append(h)

        feat = torch.cat(outputs, dim=2)  # [B, 192, T_out, H_feat, W_feat]
        self.vae.clear_cache()
        # with T=1, T_out=1; squeeze out the time dim
        return feat.squeeze(2)  # [B, 192, H_feat, W_feat]

    def forward(self, latent_x1):
        """latent_x1: [B, C, H, W] (normalized domain, float32).
        mode='image' returns [B, 3, H_img, W_img] in [-1, 1].
        mode='feat' returns [B, feat_chans, H_feat, W_feat].
        """
        l_sample = self._denormalize(latent_x1)
        l_sample = l_sample.to(dtype=self.decode_dtype)

        if self.mode == "image":
            with torch.no_grad(), self._autocast_ctx(l_sample.device):
                out = self.vae.decode(l_sample, return_dict=False)[0]
            # qwen-image-2512 outputs 5D [B,3,1,H,W]; squeeze needed
            if self.model_name == "qwen-image-2512" and out.ndim == 5:
                out = out[:, :, 0]  # [B, 3, H, W]
            return out

        # mode == "feat"
        if self.model_name == "qwen-image-2512":
            with torch.no_grad(), self._autocast_ctx(l_sample.device):
                return self._decode_to_up_blocks1_qwen(l_sample)

        raise RuntimeError(f"feat mode not implemented for {self.model_name}")


# =========================
# latent_to_image Multi-task (image version)
# =========================
class LatentToImageMultiTask(nn.Module):
    """latent_x1 → ImageVAEDecoderModule(mode='image') → RGB → resize + normalize → backbone → 3 heads.

    The extractor is held via object.__setattr__, staying out of state_dict / named_parameters.
    Handles single-frame 4D input [B, C, H, W] only; no temporal dim.
    """
    def __init__(self, vae_extractor, backbone, feat_dim,
                 target_hw, preprocess_style, mean, std, enable_hflip,
                 dropout, prompt_fusion_cfg=None,
                 num_porn_classes=2, num_gore_classes=2):
        super().__init__()
        object.__setattr__(self, "extractor", vae_extractor)

        self.backbone = backbone
        self.feat_dim = feat_dim
        self.target_hw = (int(target_hw[0]), int(target_hw[1])) if target_hw is not None else None
        self.preprocess_style = str(preprocess_style)
        self.enable_hflip = bool(enable_hflip)

        self.register_buffer("img_mean", torch.tensor(mean).float().view(1, 3, 1, 1))
        self.register_buffer("img_std", torch.tensor(std).float().view(1, 3, 1, 1))

        self.dropout = nn.Dropout(dropout)
        self.head_porn = MLPHead(feat_dim, num_porn_classes, hidden_dim=feat_dim, dropout=0.0)
        self.head_gore = MLPHead(feat_dim, num_gore_classes, hidden_dim=feat_dim, dropout=0.0)
        self.head_ip = MLPHead(feat_dim, 6, hidden_dim=feat_dim, dropout=0.0)
        self.prompt_fusion = _maybe_build_prompt_fusion(prompt_fusion_cfg, feat_dim)

    def _spatial_resize(self, rgb):
        """rgb: [B, 3, H, W] in [0, 1]; resized to self.target_hw per self.preprocess_style."""
        if self.target_hw is None:
            return rgb
        th, tw = self.target_hw
        _, _, h, w = rgb.shape
        if h == th and w == tw:
            return rgb
        if self.preprocess_style == "stretch":
            rgb = F.interpolate(rgb, size=(th, tw), mode="bilinear", align_corners=False)
        elif self.preprocess_style == "keep_ratio_pad":
            scale = min(th / h, tw / w)
            new_h = max(1, int(round(h * scale)))
            new_w = max(1, int(round(w * scale)))
            rgb = F.interpolate(rgb, size=(new_h, new_w), mode="bilinear", align_corners=False)
            pad_h = th - new_h
            pad_w = tw - new_w
            pad_top, pad_left = pad_h // 2, pad_w // 2
            rgb = F.pad(rgb, (pad_left, pad_w - pad_left, pad_top, pad_h - pad_top))
        else:
            raise ValueError(f"latent_to_image_preprocess_style does not support {self.preprocess_style!r}")
        return rgb

    def forward(self, latent_x1, prompt_embeds=None, prompt_mask=None):
        rgb = self.extractor(latent_x1)  # [B, 3, H, W] ∈ [-1, 1]
        rgb = ((rgb.float() + 1.0) / 2.0).clamp_(0.0, 1.0)  # → [0, 1]
        rgb = self._spatial_resize(rgb)
        if self.training and self.enable_hflip and torch.rand(1).item() < 0.5:
            rgb = torch.flip(rgb, dims=[-1])
        rgb = (rgb - self.img_mean) / self.img_std  # ImageNet normalize
        feat = self.backbone(rgb)  # [B, feat_dim]
        feat = self.dropout(feat)
        feat = _apply_prompt_fusion(self.prompt_fusion, feat, prompt_embeds, self.training, prompt_mask=prompt_mask)
        return self.head_porn(feat), self.head_gore(feat), self.head_ip(feat)


# =========================
# latent_to_image_feat Multi-task (image version)
# =========================
class LatentToImageFeatMultiTask(nn.Module):
    """latent_x1 → ImageVAEDecoderModule(mode='feat') → [B, feat_chans, H, W]
       → per-channel normalize → backbone (stem in_chans=feat_chans) → 3 heads.

    The extractor is held via object.__setattr__, staying out of state_dict.
    feat_stats (per-channel mean/std) are computed by run_experiment before training and injected via set_feat_stats().
    Unlike the video LatentToFeatMultiTask: no temporal dim.
    """
    def __init__(self, vae_extractor, backbone, feat_dim, in_chans,
                 dropout, data_aug=False, prompt_fusion_cfg=None,
                 num_porn_classes=2, num_gore_classes=2):
        super().__init__()
        object.__setattr__(self, "extractor", vae_extractor)

        self.backbone = backbone
        self.feat_dim = feat_dim
        self.in_chans = int(in_chans)
        self.data_aug = bool(data_aug)

        # feat_mean/std placeholders; real values filled by set_feat_stats
        self.register_buffer("feat_mean", torch.zeros(1, self.in_chans, 1, 1))
        self.register_buffer("feat_std", torch.ones(1, self.in_chans, 1, 1))
        self._has_feat_stats = False

        self.dropout = nn.Dropout(dropout)
        self.head_porn = MLPHead(feat_dim, num_porn_classes, hidden_dim=feat_dim, dropout=0.0)
        self.head_gore = MLPHead(feat_dim, num_gore_classes, hidden_dim=feat_dim, dropout=0.0)
        self.head_ip = MLPHead(feat_dim, 6, hidden_dim=feat_dim, dropout=0.0)
        self.prompt_fusion = _maybe_build_prompt_fusion(prompt_fusion_cfg, feat_dim)

    def set_feat_stats(self, mean, std):
        """Injected by the train flow after computing stats; mean/std are [in_chans] tensors."""
        device = self.feat_mean.device
        mean_t = mean.detach().float().view(1, self.in_chans, 1, 1).to(device)
        std_t = std.detach().float().view(1, self.in_chans, 1, 1).clamp_min(1e-6).to(device)
        self.feat_mean.copy_(mean_t)
        self.feat_std.copy_(std_t)
        self._has_feat_stats = True

    def forward(self, latent_x1, prompt_embeds=None, prompt_mask=None):
        if not self._has_feat_stats:
            raise RuntimeError(
                "LatentToImageFeatMultiTask: feat_stats not initialized; "
                "run_experiment should call model.set_feat_stats(mean, std) before training"
            )
        feat = self.extractor(latent_x1).float()  # [B, feat_chans, H, W]
        feat = (feat - self.feat_mean) / self.feat_std
        if self.training and self.data_aug and torch.rand(1).item() < 0.5:
            feat = torch.flip(feat, dims=[-1])
        feat_out = self.backbone(feat)  # [B, feat_dim]
        feat_out = self.dropout(feat_out)
        feat_out = _apply_prompt_fusion(self.prompt_fusion, feat_out, prompt_embeds, self.training, prompt_mask=prompt_mask)
        return self.head_porn(feat_out), self.head_gore(feat_out), self.head_ip(feat_out)


# =========================
# Build helpers for image latent_to_image / latent_to_image_feat
# =========================
def _build_latent_to_image_model(config, device, dropout, freeze_backbone, prompt_fusion_cfg=None):
    model_name = str(config["model"])
    backbone_name = str(config.get("latent_to_image_backbone", "convnext_base"))
    raw_pretrained = bool(config.get("latent_to_image_backbone_pretrained", True))

    backbone, feat_dim = _build_middle_frame_backbone(backbone_name, raw_pretrained, freeze_backbone)
    backbone = backbone.to(device)

    vae_dtype = _resolve_vae_dtype(config.get("vae_decode_dtype", "bfloat16"))
    vae_extractor = ImageVAEDecoderModule(
        model_name=model_name,
        vae_pretrained_path=config["vae_pretrained_path"],
        mode="image", device=device, decode_dtype=vae_dtype,
    )

    # target_hw
    stretch_hw = config.get("latent_to_image_stretch_target_hw")
    if stretch_hw:
        target_hw = (int(stretch_hw[0]), int(stretch_hw[1]))
    else:
        s = int(config.get("latent_to_image_image_size", 224))
        target_hw = (s, s)

    model = LatentToImageMultiTask(
        vae_extractor=vae_extractor,
        backbone=backbone, feat_dim=feat_dim,
        target_hw=target_hw,
        preprocess_style=str(config.get("latent_to_image_preprocess_style", "stretch")),
        mean=config.get("latent_to_image_mean", [0.485, 0.456, 0.406]),
        std=config.get("latent_to_image_std", [0.229, 0.224, 0.225]),
        enable_hflip=bool(config.get("latent_to_image_enable_hflip", True)),
        dropout=dropout,
        prompt_fusion_cfg=prompt_fusion_cfg,
        num_porn_classes=int(config.get("num_porn_classes", 2)),
        num_gore_classes=int(config.get("num_gore_classes", 2)),
    )

    load_info = make_load_info(
        checkpoint_path=f"torchvision/{backbone_name}" if raw_pretrained else None,
        loaded=raw_pretrained,
        note=f"latent_to_image backbone={backbone_name}, pretrained={raw_pretrained}",
    )
    model.load_info = load_info
    return model.to(device)


def _build_latent_to_image_feat_model(config, device, dropout, freeze_backbone, prompt_fusion_cfg=None):
    model_name = str(config["model"])
    backbone_name = str(config.get("latent_to_image_feat_backbone", "convnext_base"))
    pretrained = bool(config.get("latent_to_image_feat_backbone_pretrained", True))

    # qwen-image-2512 uses the WanVAE early stop -> 192 channels
    # other models may differ in the future; dispatch by model here
    if model_name == "qwen-image-2512":
        in_chans = 192
    else:
        raise ValueError(
            f"latent_to_image_feat currently supports only model='qwen-image-2512' (WanVAE early stop, 192ch); "
            f"got model={model_name!r}"
        )

    backbone, feat_dim = _build_cnn_for_latent(backbone_name, in_chans=in_chans, pretrained=pretrained)
    backbone = backbone.to(device)

    vae_dtype = _resolve_vae_dtype(config.get("vae_decode_dtype", "bfloat16"))
    vae_extractor = ImageVAEDecoderModule(
        model_name=model_name,
        vae_pretrained_path=config["vae_pretrained_path"],
        mode="feat", device=device, decode_dtype=vae_dtype,
    )

    model = LatentToImageFeatMultiTask(
        vae_extractor=vae_extractor,
        backbone=backbone, feat_dim=feat_dim, in_chans=in_chans,
        dropout=dropout,
        data_aug=bool(config.get("latent_to_image_feat_data_aug", False)),
        prompt_fusion_cfg=prompt_fusion_cfg,
        num_porn_classes=int(config.get("num_porn_classes", 2)),
        num_gore_classes=int(config.get("num_gore_classes", 2)),
    )

    # optional ckpt loading
    ckpt_path = config.get("latent_to_image_feat_model_ckpt") or None
    if ckpt_path:
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            sd = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
            sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
            model_state = model.state_dict()
            filtered = {k: v for k, v in sd.items()
                        if k in model_state and tuple(v.shape) == tuple(model_state[k].shape)}
            missing, unexpected = model.load_state_dict(filtered, strict=False)
            load_info = make_load_info(
                checkpoint_path=ckpt_path, loaded=bool(filtered),
                note=f"latent_to_image_feat backbone={backbone_name}, ckpt {len(filtered)}/{len(sd)} matched",
            )
            load_info["missing_keys"] = [k for k in missing if not k.startswith("head_")]
            load_info["unexpected_keys"] = list(unexpected)
        except Exception as e:
            load_info = make_load_info(checkpoint_path=ckpt_path, loaded=False,
                                        note=f"latent_to_image_feat ckpt load failed: {e}")
    else:
        load_info = make_load_info(
            checkpoint_path=f"torchvision/{backbone_name}" if pretrained else None,
            loaded=pretrained,
            note=f"latent_to_image_feat backbone={backbone_name}, IN1K_pretrained={pretrained}",
        )

    if freeze_backbone:
        for p in model.backbone.parameters():
            p.requires_grad = False
        if backbone_name.startswith("convnext"):
            stem = model.backbone.features[0][0]
        elif backbone_name.startswith("vit"):
            stem = model.backbone.conv_proj
        else:
            stem = model.backbone.conv1
        for p in stem.parameters():
            p.requires_grad = True
        load_info["note"] += " | backbone frozen except stem / heads"

    model.load_info = load_info
    return model.to(device)
