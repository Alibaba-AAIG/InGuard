"""
VAE encoding script: compress images into latents and save them using the Qwen-Image-2512 VAE Encoder.

The output latent distribution matches the latents_x1 saved during diffusion inference.

latents_x1 storage format: [B, C, 1, H, W] (5D, F=1 means a single-frame image)
Normalization formula (encode → latent_x1):
    latent_raw = vae.encode(img)  → [B, C, 1, H, W]
    latent_x1 = (latent_raw - latents_mean) * latents_std
    where latents_std = 1.0 / torch.tensor(vae.config.latents_std)

Inverse decoding formula (for verification):
    img = vae.decode(latent_x1 / latents_std + latents_mean)[:, :, 0]

Usage:
    python qwen_image_2512_vae_encode.py \
        --input_dir /path/to/images \
        --output_dir /path/to/output \
        --model_path /path/to/Qwen-Image-2512 \
        --device cuda:0
"""
import os
import argparse

import gc
import time

import torch
from PIL import Image
from tqdm import tqdm


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


def list_image_files(input_dir):
    """List all image files in the input directory"""
    files = []
    for f in sorted(os.listdir(input_dir)):
        ext = os.path.splitext(f)[1].lower()
        if ext in IMAGE_EXTENSIONS:
            files.append(f)
    return files


def load_vae(model_path, device):
    """Load the VAE part of QwenImagePipeline, releasing the non-VAE components afterwards to save memory"""
    from diffusers import QwenImagePipeline

    print(f"Loading QwenImagePipeline (VAE only) from {model_path} ...")
    pipe = QwenImagePipeline.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
    )
    vae = pipe.vae.to(device)
    image_processor = pipe.image_processor
    vae_scale_factor = pipe.vae_scale_factor

    # QwenImage VAE denormalization params (same construction as the save_data scripts)
    latents_mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1).to(device, vae.dtype)
    latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1).to(device, vae.dtype)

    # free the CPU memory held by the transformer / text_encoder and other big components
    del pipe
    gc.collect()

    print(f"VAE loaded on {device}, dtype={vae.dtype}")
    print(f"  vae_scale_factor={vae_scale_factor}")
    print(f"  latents_mean shape={latents_mean.shape}")
    print(f"  latents_std (reciprocal) shape={latents_std.shape}")
    return vae, image_processor, vae_scale_factor, latents_mean, latents_std


def encode_image(image_path, vae, image_processor, device,
                 latents_mean, latents_std, target_height, target_width):
    """
    Load the image → preprocess → VAE encode → normalize → return the latent.
    Output shape: [1, C, 1, H_latent, W_latent]
    """
    img = Image.open(image_path).convert("RGB")

    if target_height is not None and target_width is not None:
        img = img.resize((target_width, target_height), Image.LANCZOS)

    pixel_values = image_processor.preprocess(
        img, height=img.height, width=img.width
    )
    pixel_values = pixel_values.to(device=device, dtype=vae.dtype)

    # the QwenImage VAE expects 5D input [B, C, F, H, W], F=1 means a single frame
    if pixel_values.dim() == 4:
        pixel_values = pixel_values.unsqueeze(2)  # [B, C, H, W] → [B, C, 1, H, W]

    with torch.no_grad():
        latent_dist = vae.encode(pixel_values).latent_dist
        latent = latent_dist.sample()  # [B, C, 1, H_latent, W_latent]

        # normalize to the same distribution as latents_x1
        latent = (latent - latents_mean) * latents_std

    return latent.cpu()


def main():
    parser = argparse.ArgumentParser(description="Qwen-Image-2512 VAE Encode: image → latent (aligned with the latents_x1 distribution)")
    parser.add_argument("--input_dir", required=True, help="input image folder path")
    parser.add_argument("--output_dir", required=True, help="output .pth folder path")
    parser.add_argument("--model_path", required=True,
                        help="QwenImagePipeline model path (a local dir or a HuggingFace repo id, auto-downloaded in the latter case)")
    parser.add_argument("--device", default="cuda:0", help="GPU device")
    parser.add_argument("--height", type=int, default=None,
                        help="target height (default: no resize; keep the original size)")
    parser.add_argument("--width", type=int, default=None,
                        help="target width (default: no resize; keep the original size)")
    parser.add_argument("--skip_existing", action="store_true",
                        help="skip existing output files")
    parser.add_argument("--watch", action="store_true",
                        help="watch mode: after processing the current files, keep scanning for new ones")
    parser.add_argument("--scan_interval", type=int, default=10,
                        help="scan interval in watch mode (seconds, default 10)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    vae, image_processor, vae_scale_factor, latents_mean, latents_std = load_vae(args.model_path, args.device)

    image_files = list_image_files(args.input_dir)
    print(f"Found {len(image_files)} images in {args.input_dir}")

    n_done = n_skip = n_err = 0
    for filename in tqdm(image_files):
        stem = os.path.splitext(filename)[0]
        output_path = os.path.join(args.output_dir, f"{stem}.pth")

        if args.skip_existing and os.path.exists(output_path):
            n_skip += 1
            continue

        try:
            image_path = os.path.join(args.input_dir, filename)
            latent = encode_image(
                image_path, vae, image_processor, args.device,
                latents_mean, latents_std,
                target_height=args.height, target_width=args.width,
            )
            torch.save(latent, output_path)
            n_done += 1
        except Exception as e:
            n_err += 1
            print(f"[Error] {filename}: {e}")

    print(f"\nFirst pass done. encoded={n_done}, skipped={n_skip}, errors={n_err}")

    if not args.watch:
        return

    # --- Watch mode: keep scanning for new files ---
    print(f"[Watch] entering watch mode, scanning for new files every {args.scan_interval} seconds ...")
    while True:
        time.sleep(args.scan_interval)
        current_files = list_image_files(args.input_dir)
        new_files = [
            f for f in current_files
            if not os.path.exists(os.path.join(args.output_dir, f"{os.path.splitext(f)[0]}.pth"))
        ]
        if not new_files:
            continue

        print(f"\n[Watch] found {len(new_files)} new files, processing ...")
        n_done_round = n_err_round = 0
        for filename in tqdm(new_files):
            stem = os.path.splitext(filename)[0]
            output_path = os.path.join(args.output_dir, f"{stem}.pth")
            try:
                image_path = os.path.join(args.input_dir, filename)
                latent = encode_image(
                    image_path, vae, image_processor, args.device,
                    latents_mean, latents_std,
                    target_height=args.height, target_width=args.width,
                )
                torch.save(latent, output_path)
                n_done_round += 1
            except Exception as e:
                n_err_round += 1
                print(f"[Error] {filename}: {e}")
        print(f"[Watch] round done: encoded={n_done_round}, errors={n_err_round}")


if __name__ == "__main__":
    main()
