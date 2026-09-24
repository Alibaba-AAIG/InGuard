"""
Latent Safety Detector — ConvNeXt-Base backbone + 3-task MLP heads.

This file is self-contained: only depends on torch + torchvision.

Model architecture:
    Input:  latent tensor [B, C, H, W]  (C=16 for z-image-turbo / qwen-image-2512)
    Backbone: ConvNeXt-Base (torchvision), stem conv modified to accept C-channel latent
    Heads: 3 independent MLP heads (porn / gore / ip)

Classification heads:
    ┌────────────┬──────────┬──────────────────────────────────────────┐
    │ Head       │ #classes │ Labels                                    │
    ├────────────┼──────────┼──────────────────────────────────────────┤
    │ head_porn  │ 2        │ 0=safe, 1=unsafe                         │
    │ head_gore  │ 2        │ 0=safe, 1=unsafe                         │
    │ head_ip    │ 6        │ 0=白雪公主, 1=哆啦A梦, 2=小黄人,          │
    │            │          │ 3=艾莎, 4=海绵宝宝, 5=其它/无IP           │
    └────────────┴──────────┴──────────────────────────────────────────┘

    IP head detail:
        Class 0: 白雪公主 (Snow White)     — controlled IP
        Class 1: 哆啦A梦 (Doraemon)        — controlled IP
        Class 2: 小黄人 (Minions)          — controlled IP
        Class 3: 艾莎 (Elsa)               — controlled IP
        Class 4: 海绵宝宝 (SpongeBob)      — controlled IP
        Class 5: 其它 / 无IP (Other/None)  — non-controlled

    Guardrail decision (is the content unsafe?):
        unsafe = (porn_pred == 1) or (gore_pred == 1) or (ip_pred in {0,1,2,3,4})
        safe  = not unsafe  (i.e. ip_pred == 5 and porn_pred == 0 and gore_pred == 0)
"""

import torch
import torch.nn as nn
import torchvision.models as tvm


# =========================
# Shared MLP head
# =========================
class MLPHead(nn.Module):
    """MLP classification head: Linear → GELU → (optional Dropout) → Linear."""
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


# =========================
# Stem weight expansion
# =========================
def expand_conv_in_chans_weight(weight, new_in_chans):
    """Expand conv first layer input channels from 3 (or old) to new_in_chans.
    First min(3, old) channels reuse old weights; extra channels use mean of first 3.
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
# Build backbone for latent input
# =========================
def _build_cnn_for_latent(name, in_chans, pretrained, stem_type="single", stem_init="expand_in1k"):
    """Build a torchvision backbone with modified stem to accept latent channels.

    Args:
        name: backbone name (e.g. "convnext_base")
        in_chans: input latent channels (16 for z-image-turbo / qwen-image-2512)
        pretrained: whether to load IN1K pretrained weights
        stem_type: "single" (single conv) or "progressive" (multi-layer stem)
        stem_init: initialization method for new stem

    Returns:
        (backbone_module, feat_dim)
    """
    _TVM_WEIGHTS = {
        "convnext_tiny":   ("convnext_tiny",   "ConvNeXt_Tiny_Weights"),
        "convnext_small":  ("convnext_small",  "ConvNeXt_Small_Weights"),
        "convnext_base":   ("convnext_base",   "ConvNeXt_Base_Weights"),
        "convnext_large":  ("convnext_large",  "ConvNeXt_Large_Weights"),
        "vit_b_16":        ("vit_b_16",        "ViT_B_16_Weights"),
        "vit_l_16":        ("vit_l_16",        "ViT_L_16_Weights"),
        "resnet18":        ("resnet18",        "ResNet18_Weights"),
        "resnet50":        ("resnet50",        "ResNet50_Weights"),
        "resnet101":       ("resnet101",       "ResNet101_Weights"),
    }

    BACKBONE_FAMILY = {
        "convnext_tiny": "convnext",  "convnext_small": "convnext",
        "convnext_base": "convnext",  "convnext_large": "convnext",
        "vit_b_16": "vit",            "vit_l_16": "vit",
        "resnet18": "resnet",         "resnet50": "resnet",
        "resnet101": "resnet",
    }

    ctor_name, weights_attr = _TVM_WEIGHTS[name]
    weights = getattr(tvm, weights_attr).IMAGENET1K_V1 if pretrained else None
    m = getattr(tvm, ctor_name)(weights=weights)

    family = BACKBONE_FAMILY[name]

    if stem_type == "single":
        if family == "convnext":
            # ConvNeXt stem: features[0] = Sequential(Conv2d, LayerNorm2d)
            old_stem = m.features[0][0]
            new_stem = nn.Conv2d(
                in_chans, old_stem.out_channels,
                kernel_size=old_stem.kernel_size, stride=old_stem.stride,
                padding=old_stem.padding, bias=(old_stem.bias is not None),
            )
            # Initialize: expand IN1K weights to in_chans channels
            if stem_init == "expand_in1k" and pretrained:
                with torch.no_grad():
                    new_stem.weight.copy_(
                        expand_conv_in_chans_weight(old_stem.weight.data, in_chans))
                    if old_stem.bias is not None and new_stem.bias is not None:
                        new_stem.bias.copy_(old_stem.bias.data)
            m.features[0][0] = new_stem
            feat_dim = m.classifier[2].in_features
            m.classifier[2] = nn.Identity()  # Keep LayerNorm2d + Flatten; remove Linear
        elif family == "vit":
            old_conv = m.conv_proj
            new_conv = nn.Conv2d(
                in_chans, old_conv.out_channels,
                kernel_size=old_conv.kernel_size, stride=old_conv.stride,
                padding=old_conv.padding, bias=(old_conv.bias is not None),
            )
            if stem_init == "expand_in1k" and pretrained:
                with torch.no_grad():
                    new_conv.weight.copy_(
                        expand_conv_in_chans_weight(old_conv.weight.data, in_chans))
                    if old_conv.bias is not None and new_conv.bias is not None:
                        new_conv.bias.copy_(old_conv.bias.data)
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
            if stem_init == "expand_in1k" and pretrained:
                with torch.no_grad():
                    new_stem.weight.copy_(
                        expand_conv_in_chans_weight(old_stem.weight.data, in_chans))
                    if old_stem.bias is not None and new_stem.bias is not None:
                        new_stem.bias.copy_(old_stem.bias.data)
            m.conv1 = new_stem
            feat_dim = m.fc.in_features
            m.fc = nn.Identity()
    else:
        raise ValueError(f"stem_type={stem_type!r} not supported in export. Use 'single'.")

    return m, feat_dim


# =========================
# Latent CNN Multi-task Model
# =========================
class LatentCNNMultiTask(nn.Module):
    """Latent + backbone: backbone (stem modified) + 3 MLP heads.

    Input:  [B, C, H, W] single-frame latent
    Output: (porn_logits, gore_logits, ip_logits)
            each [B, num_classes]
    """
    def __init__(self, backbone_name, in_chans, pretrained=False,
                 stem_type="single", stem_init="expand_in1k",
                 num_porn_classes=2, num_gore_classes=2):
        super().__init__()
        self.backbone_name = backbone_name
        self.backbone, feat_dim = _build_cnn_for_latent(
            backbone_name, in_chans, pretrained,
            stem_type=stem_type, stem_init=stem_init,
        )
        self.feat_dim = feat_dim
        # head dropout=0.0: dropout controlled by outer wrapper
        self.head_porn = MLPHead(feat_dim, num_porn_classes, hidden_dim=feat_dim, dropout=0.0)
        self.head_gore = MLPHead(feat_dim, num_gore_classes, hidden_dim=feat_dim, dropout=0.0)
        # IP head: 6 classes (0=白雪公主, 1=哆啦A梦, 2=小黄人, 3=艾莎, 4=海绵宝宝, 5=其它/无IP)
        self.head_ip = MLPHead(feat_dim, 6, hidden_dim=feat_dim, dropout=0.0)

    def extract_feat(self, x):
        return self.backbone(x)


# =========================
# Multi-task Wrapper (with dropout)
# =========================
class MultiTaskWrapperCNN(nn.Module):
    """Wrapper: backbone → dropout → 3 heads.

    This matches the training-time model structure.
    State dict keys have prefix 'base.' for backbone/head weights.
    """
    def __init__(self, backbone_name="convnext_base", in_chans=16,
                 pretrained=False, dropout=0.0, stem_type="single",
                 stem_init="expand_in1k", num_porn_classes=2, num_gore_classes=2):
        super().__init__()
        self.base = LatentCNNMultiTask(
            backbone_name=backbone_name, in_chans=in_chans, pretrained=pretrained,
            stem_type=stem_type, stem_init=stem_init,
            num_porn_classes=num_porn_classes, num_gore_classes=num_gore_classes,
        )
        self.feat_dim = self.base.feat_dim
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, C, H, W]
        if x.ndim != 4:
            raise ValueError(f"Expected 4D input [B,C,H,W], got {tuple(x.shape)}")
        feat = self.dropout(self.base.extract_feat(x))
        return self.base.head_porn(feat), self.base.head_gore(feat), self.base.head_ip(feat)


# =========================
# IP label mapping
# =========================
IP_ID2NAME = {
    0: "白雪公主",   # Snow White
    1: "哆啦A梦",   # Doraemon
    2: "小黄人",     # Minions
    3: "艾莎",       # Elsa
    4: "海绵宝宝",   # SpongeBob
    5: "其它",       # Other / None
}

# Controlled IP IDs (class 0-4 are controlled IPs, class 5 is other/none)
CONTROLLED_IP_IDS = {0, 1, 2, 3, 4}
