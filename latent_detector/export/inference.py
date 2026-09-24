"""
Latent Safety Detector — Loading & Inference.

Quick start:
    from inference import LatentDetector

    # Load detector for z-image-turbo
    detector = LatentDetector("z-image-turbo", device="cuda")

    # Run inference on a latent tensor [B, 16, H, W]
    result = detector.predict(latent_batch)
    # result = {
    #     "porn_prob": float,       # P(porn unsafe)
    #     "gore_prob": float,       # P(gore unsafe)
    #     "ip_probs": list[6],     # P for each IP class
    #     "porn_pred": int,         # 0=safe, 1=unsafe
    #     "gore_pred": int,         # 0=safe, 1=unsafe
    #     "ip_pred": int,           # 0-4=controlled IP, 5=other/none
    #     "is_unsafe": bool,        # overall unsafe decision
    # }

    # Or predict a single latent loaded from .pth file
    result = detector.predict_from_file("path/to/latent.pth")
"""

import json
import os

import torch
import torch.nn.functional as F

from model import (
    MultiTaskWrapperCNN,
    IP_ID2NAME,
    CONTROLLED_IP_IDS,
)


class LatentDetector:
    """Load and run inference with the Latent Safety Detector.

    Args:
        model_name: one of "z-image-turbo", "qwen-image-2512", "hunyuan-image-2_1",
                    "flux2-klein-base-9b", "internvl-u"
        ckpt_dir: directory containing model.pth and config.json.
                  If None, uses the default path relative to this file.
        device: "cuda" or "cpu"
    """

    # Default latent channels for each model (overridden by config.json
    # "latent_chans" when the exported checkpoint carries it)
    LATENT_CHANS = {
        "z-image-turbo": 16,
        "qwen-image-2512": 16,
        "hunyuan-image-2_1": 64,
        "flux2-klein-base-9b": 32,
        "internvl-u": 16,
    }

    def __init__(self, model_name, ckpt_dir=None, device="cuda"):
        assert model_name in self.LATENT_CHANS, \
            f"Unsupported model: {model_name}. Supported: {list(self.LATENT_CHANS.keys())}"

        self.model_name = model_name
        self.device = device

        # Default checkpoint directory
        if ckpt_dir is None:
            ckpt_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), model_name)

        ckpt_path = os.path.join(ckpt_dir, "model.pth")
        config_path = os.path.join(ckpt_dir, "config.json")

        # Load config
        if os.path.exists(config_path):
            with open(config_path) as f:
                self.config = json.load(f)
        else:
            self.config = {}

        # Model hyperparameters (from config or defaults)
        # exported config.json uses "latent_in_chans" (see scripts/export_weights.py)
        self.in_chans = int(self.config.get(
            "latent_in_chans",
            self.config.get("latent_chans", self.LATENT_CHANS[model_name])))
        backbone = self.config.get("backbone", "convnext_base")
        stem_type = self.config.get("latent_stem_type", "single")
        stem_init = self.config.get("latent_stem_init", "expand_in1k")
        num_porn = self.config.get("num_porn_classes", 2)
        num_gore = self.config.get("num_gore_classes", 2)
        resize_hw = self.config.get("latent_stretch_target_hw") or self.config.get("resize_target")

        self.resize_target = (int(resize_hw[0]), int(resize_hw[1])) if resize_hw else None
        self.resize_style = self.config.get(
            "latent_preprocess_style", self.config.get("resize_style", "stretch"))

        # Build model (pretrained=False: no IN1K download needed)
        self.model = MultiTaskWrapperCNN(
            backbone_name=backbone,
            in_chans=self.in_chans,
            pretrained=False,
            dropout=0.0,  # inference: no dropout
            stem_type=stem_type,
            stem_init=stem_init,
            num_porn_classes=num_porn,
            num_gore_classes=num_gore,
        )

        # Load checkpoint
        print(f"[LatentDetector] Loading checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt

        # Strip "module." prefix (DDP)
        state_dict = {(k[7:] if k.startswith("module.") else k): v
                      for k, v in state_dict.items()}

        # Strip "base." prefix if checkpoint was saved as MultiTaskWrapperCNN
        model_keys = set(self.model.state_dict().keys())
        if len(set(state_dict.keys()) & model_keys) == 0:
            stripped = {(k[5:] if k.startswith("base.") else k): v
                        for k, v in state_dict.items()}
            if len(set(stripped.keys()) & model_keys) > 0:
                state_dict = stripped

        # Load with shape matching
        model_state = self.model.state_dict()
        filtered = {}
        skipped = []
        for k, v in state_dict.items():
            if k not in model_state:
                continue
            if v.shape == model_state[k].shape:
                filtered[k] = v
            else:
                # Auto-expand stem channels if needed
                is_stem = (k.endswith("conv1.weight")
                           or k.endswith("features.0.0.weight")
                           or k.endswith("conv_proj.weight"))
                if is_stem and v.shape != model_state[k].shape:
                    from model import expand_conv_in_chans_weight
                    v = expand_conv_in_chans_weight(v, self.in_chans)
                    if v.shape == model_state[k].shape:
                        filtered[k] = v
                    else:
                        skipped.append((k, tuple(v.shape), tuple(model_state[k].shape)))
                else:
                    skipped.append((k, tuple(v.shape), tuple(model_state[k].shape)))

        missing, unexpected = self.model.load_state_dict(filtered, strict=False)
        real_missing = [k for k in missing if not k.startswith("head_")]
        if real_missing:
            print(f"[LatentDetector] WARNING: missing keys: {real_missing[:5]}...")
        if skipped:
            print(f"[LatentDetector] WARNING: skipped {len(skipped)} shape-mismatched keys")

        self.model = self.model.to(device)
        self.model.eval()
        print(f"[LatentDetector] Model loaded successfully. "
              f"backbone={backbone}, in_chans={self.in_chans}, "
              f"resize={self.resize_target}")

    def _preprocess_latent(self, latent):
        """Preprocess latent: resize if needed.

        Args:
            latent: [C, H, W] or [B, C, H, W] tensor

        Returns:
            [B, C, H', W'] tensor ready for model input
        """
        if latent.ndim == 3:
            latent = latent.unsqueeze(0)  # [1, C, H, W]
        elif latent.ndim == 4:
            pass
        else:
            raise ValueError(f"Expected 3D or 4D latent, got {latent.ndim}D")

        if self.resize_target is not None:
            h, w = self.resize_target
            if latent.shape[-2] != h or latent.shape[-1] != w:
                # Bilinear resize (stretch mode)
                latent = F.interpolate(
                    latent, size=(h, w), mode="bilinear", align_corners=False)

        return latent

    @torch.no_grad()
    def predict(self, latent):
        """Run inference on a latent batch.

        Args:
            latent: [C, H, W] or [B, C, H, W] tensor

        Returns:
            dict with keys:
                porn_prob: float        — P(porn=1)
                gore_prob: float        — P(gore=1)
                ip_probs: list[float]   — P for each of 6 IP classes
                porn_pred: int          — 0=safe, 1=unsafe
                gore_pred: int          — 0=safe, 1=unsafe
                ip_pred: int            — 0-4=controlled IP, 5=other/none
                ip_name: str            — IP name (e.g. "白雪公主")
                is_unsafe: bool          — True if any head predicts unsafe
        """
        x = self._preprocess_latent(latent).to(self.device)
        x = x.float()

        porn_logits, gore_logits, ip_logits = self.model(x)

        # Take first sample (batch=1 inference)
        porn_prob = torch.softmax(porn_logits, dim=-1)[0, 1].item()
        gore_prob = torch.softmax(gore_logits, dim=-1)[0, 1].item()
        ip_probs = torch.softmax(ip_logits, dim=-1)[0].tolist()

        # Predictions
        porn_pred = 1 if porn_prob >= 0.5 else 0
        gore_pred = 1 if gore_prob >= 0.5 else 0
        # IP: argmax over first 5 classes (controlled IPs), class 5 = other/none
        ip_pred_raw = int(max(range(5), key=lambda i: ip_probs[i]))
        ip_risk_score = ip_probs[ip_pred_raw]
        ip_pred = ip_pred_raw if ip_risk_score >= 0.5 else 5

        is_unsafe = (porn_pred == 1) or (gore_pred == 1) or (ip_pred in CONTROLLED_IP_IDS)

        return {
            "porn_prob": porn_prob,
            "gore_prob": gore_prob,
            "ip_probs": ip_probs,
            "porn_pred": porn_pred,
            "gore_pred": gore_pred,
            "ip_pred": ip_pred,
            "ip_name": IP_ID2NAME.get(ip_pred, "未知"),
            "is_unsafe": is_unsafe,
        }

    @torch.no_grad()
    def predict_batch(self, latent_batch):
        """Run inference on a batch of latents.

        Args:
            latent_batch: [B, C, H, W] tensor

        Returns:
            list of dict (one per sample)
        """
        x = self._preprocess_latent(latent_batch).to(self.device)
        x = x.float()

        porn_logits, gore_logits, ip_logits = self.model(x)
        porn_probs = torch.softmax(porn_logits, dim=-1)[:, 1]
        gore_probs = torch.softmax(gore_logits, dim=-1)[:, 1]
        ip_probs_all = torch.softmax(ip_logits, dim=-1)

        results = []
        for i in range(x.shape[0]):
            pp = porn_probs[i].item()
            gp = gore_probs[i].item()
            ip_p = ip_probs_all[i].tolist()
            porn_pred = 1 if pp >= 0.5 else 0
            gore_pred = 1 if gp >= 0.5 else 0
            ip_pred_raw = int(max(range(5), key=lambda j: ip_p[j]))
            ip_pred = ip_pred_raw if ip_p[ip_pred_raw] >= 0.5 else 5
            results.append({
                "porn_prob": pp,
                "gore_prob": gp,
                "ip_probs": ip_p,
                "porn_pred": porn_pred,
                "gore_pred": gore_pred,
                "ip_pred": ip_pred,
                "ip_name": IP_ID2NAME.get(ip_pred, "未知"),
                "is_unsafe": (porn_pred == 1) or (gore_pred == 1) or (ip_pred in CONTROLLED_IP_IDS),
            })
        return results

    @torch.no_grad()
    def predict_from_file(self, pth_path):
        """Load a latent .pth file and run inference.

        Args:
            pth_path: path to .pth file containing a latent tensor

        Returns:
            dict (same as predict)
        """
        latent = torch.load(pth_path, map_location="cpu", weights_only=False)
        # Handle various saved formats
        if isinstance(latent, dict):
            # Try common keys
            for key in ("latent", "l_sample", "x", "data"):
                if key in latent:
                    latent = latent[key]
                    break
        if latent.ndim == 5 and latent.shape[0] == 1 and latent.shape[2] == 1:
            # qwen-image-2512 format: [1, C, 1, H, W] → squeeze
            latent = latent.squeeze(0).squeeze(0)  # [C, H, W]
        elif latent.ndim == 4 and latent.shape[0] == 1:
            latent = latent.squeeze(0)  # [C, H, W]
        return self.predict(latent)


# =========================
# CLI entry point
# =========================
if __name__ == "__main__":
    import sys

    model_name = sys.argv[1] if len(sys.argv) > 1 else "z-image-turbo"
    pth_path = sys.argv[2] if len(sys.argv) > 2 else None

    detector = LatentDetector(model_name, device="cuda" if torch.cuda.is_available() else "cpu")

    if pth_path:
        result = detector.predict_from_file(pth_path)
        print(f"\n=== Inference Result ({model_name}) ===")
        print(f"  Input: {pth_path}")
        print(f"  Porn:  pred={result['porn_pred']}  prob={result['porn_prob']:.4f}")
        print(f"  Gore:  pred={result['gore_pred']}  prob={result['gore_prob']:.4f}")
        print(f"  IP:    pred={result['ip_pred']} ({result['ip_name']})  probs={[f'{p:.4f}' for p in result['ip_probs']]}")
        print(f"  Unsafe: {result['is_unsafe']}")
    else:
        print(f"\nModel loaded. Use detector.predict(latent_tensor) to run inference.")
        print(f"Example: result = detector.predict_from_file('path/to/latent.pth')")
