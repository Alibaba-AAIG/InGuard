# Latent Safety Detector

ConvNeXt-Base backbone + 3-task MLP heads, trained on latent space of diffusion models.

## Models

| Model | Backbone | Latent Channels | Training Data | Method |
|-------|----------|----------------|---------------|--------|
| z-image-turbo | ConvNeXt-Base | 16 | 50k | Latent + distill pretrain + finetune |
| qwen-image-2512 | ConvNeXt-Base | 16 | 50k | Latent + distill pretrain + finetune |
| hunyuan-image-2_1 | ConvNeXt-Base | 64 | 50k | Latent + distill pretrain + finetune |
| flux2-klein-base-9b | ConvNeXt-Base | 32 | 50k | Latent + distill pretrain + finetune |
| internvl-u | ConvNeXt-Base | 16 | 50k | Latent + distill pretrain + finetune |

## Classification Heads

### head_porn (2 classes)
- 0 = Safe
- 1 = Unsafe (pornographic content)

### head_gore (2 classes)
- 0 = Safe
- 1 = Unsafe (gore/violence content)

### head_ip (6 classes)
| Class | Name | Type |
|-------|------|------|
| 0 | 白雪公主 (Snow White) | Controlled IP |
| 1 | 哆啦A梦 (Doraemon) | Controlled IP |
| 2 | 小黄人 (Minions) | Controlled IP |
| 3 | 艾莎 (Elsa) | Controlled IP |
| 4 | 海绵宝宝 (SpongeBob) | Controlled IP |
| 5 | 其它 / 无IP (Other/None) | Non-controlled |

### Guardrail Decision
```
is_unsafe = (porn_pred == 1) or (gore_pred == 1) or (ip_pred in {0, 1, 2, 3, 4})
```

## Quick Start

```python
from inference import LatentDetector

# Load detector
detector = LatentDetector("z-image-turbo", device="cuda")

# Predict from a latent tensor [C, H, W] or [B, C, H, W]
result = detector.predict(latent_tensor)

# Predict from a .pth file
result = detector.predict_from_file("path/to/latent.pth")

print(result)
# {
#     "porn_prob": 0.02,
#     "gore_prob": 0.01,
#     "ip_probs": [0.01, 0.02, 0.85, 0.01, 0.01, 0.10],
#     "porn_pred": 0,
#     "gore_pred": 0,
#     "ip_pred": 2,           # 小黄人
#     "ip_name": "小黄人",
#     "is_unsafe": True,       # IP detected
# }
```

## Latent Input Format

- **z-image-turbo / internvl-u**: `[16, H, W]` or `[1, 16, H, W]`
- **qwen-image-2512**: `[16, H, W]` or `[1, 16, 1, H, W]` (5D, squeeze needed)
- **hunyuan-image-2_1**: `[64, H, W]` or `[1, 64, H, W]`
- **flux2-klein-base-9b**: unpacked `[32, H, W]` or `[1, 32, H, W]` (patchified → unpacked VAE latent; see `full_unpack_latents` in `sage/sage_enhance.py`)

The detector handles these formats automatically.

## Dependencies

```
torch
torchvision
```

## Files

```
├── model.py              # Model architecture (self-contained)
├── inference.py           # Loading & inference API
├── z-image-turbo/         # model.pth + config.json
├── qwen-image-2512/
├── hunyuan-image-2_1/
├── flux2-klein-base-9b/
├── internvl-u/
└── README.md
```

Weight files are populated by `scripts/export_weights.py` at the repository
root (or downloaded from the release).
