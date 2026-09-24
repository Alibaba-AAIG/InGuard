"""
InternVL-U VAE encoding script: compress images into latents and save them.

The output latent distribution matches the latents_x1 saved during diffusion inference (the normalized latent space).
Format: [1, 16, H/32, W/32]

Normalization formula:
    image_latents = vae.encode(image)
    z = (image_latents - latents_mean) / latents_std

De-normalization (for decoding):
    image_latents = z / (1/latents_std) + latents_mean
    image = vae.decode(image_latents)

Usage:
    python internvlu_vae_encode.py \
        --input_dir /path/to/images \
        --output_dir /path/to/output \
        --model_path /path/to/InternVL-U \
        --device cuda:0
"""

import os
import sys
import gc
import time
import argparse

import torch
from PIL import Image
from tqdm import tqdm

# Use the in-repo InternVL-U pipeline package (sage/backends/internvlu/, not diffusers)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sage", "backends"))

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


def list_image_files(input_dir):
    """List all image files in the input directory"""
    files = []
    for f in sorted(os.listdir(input_dir)):
        ext = os.path.splitext(f)[1].lower()
        if ext in IMAGE_EXTENSIONS:
            files.append(f)
    return files


def load_vae_components(model_path, device, dtype=torch.bfloat16):
    """
    Load the InternVL-U VAE and the related normalization params.
    Only the VAE-related components are kept to save GPU memory.
    """
    from internvlu import InternVLUPipeline

    print(f"Loading InternVLUPipeline (VAE only) from {model_path} ...")
    pipe = InternVLUPipeline.from_pretrained(model_path, torch_dtype=dtype)

    # extract the VAE-related components
    img_pipe = pipe.image_pipeline
    vae = img_pipe.vae.to(device)
    vae_scale_factor = img_pipe.vae_scale_factor
    latent_channels = img_pipe.latent_channels

    # extract the normalization params
    latents_mean = torch.tensor(vae.config.latents_mean).view(1, latent_channels, 1, 1, 1)
    latents_std = torch.tensor(vae.config.latents_std).view(1, latent_channels, 1, 1, 1)

    # free the remaining components
    del pipe
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"VAE loaded on {device}, dtype={vae.dtype}")
    print(f"  vae_scale_factor = {vae_scale_factor}")
    print(f"  latent_channels  = {latent_channels}")
    print(f"  latents_mean shape = {latents_mean.shape}")

    return vae, latents_mean, latents_std, vae_scale_factor


def encode_image(image_path, vae, latents_mean, latents_std, device, dtype,
                 target_height=None, target_width=None):
    """
    Load the image → resize → normalize to [-1,1] → VAE encode → latent normalize.
    Output shape: [1, 16, H/32, W/32]
    """
    img = Image.open(image_path).convert("RGB")

    if target_height is not None and target_width is not None:
        img = img.resize((target_width, target_height), Image.LANCZOS)

    # image → tensor in [-1, 1]
    import torchvision.transforms as T
    transform = T.Compose([
        T.ToTensor(),           # [0, 1]
        T.Normalize([0.5]*3, [0.5]*3),  # [-1, 1]
    ])
    pixel_values = transform(img).unsqueeze(0)  # [1, 3, H, W]
    pixel_values = pixel_values.unsqueeze(2)    # [1, 3, T=1, H, W] for video VAE

    pixel_values = pixel_values.to(device=device, dtype=dtype)

    with torch.no_grad():
        # AutoencoderDC encode
        encoder_output = vae.encode(pixel_values)
        if hasattr(encoder_output, "latent_dist"):
            image_latents = encoder_output.latent_dist.mode()
        elif hasattr(encoder_output, "latents"):
            image_latents = encoder_output.latents
        else:
            image_latents = encoder_output[0] if isinstance(encoder_output, tuple) else encoder_output

        # normalize to the diffusion working space
        mean = latents_mean.to(device=device, dtype=dtype)
        std = latents_std.to(device=device, dtype=dtype)
        z = (image_latents - mean) / std
        z = z.squeeze(2)  # drop the time dim [1, 16, H/32, W/32]

    return z.cpu()


def main():
    parser = argparse.ArgumentParser(
        description="InternVL-U VAE Encode: image → latent (aligned with the latents_x1 distribution)"
    )
    parser.add_argument("--input_dir", required=True, help="input image folder path")
    parser.add_argument("--output_dir", required=True, help="output .pth folder path")
    parser.add_argument("--model_path", required=True,
                        help="InternVL-U model path (a local dir or a HuggingFace repo id, auto-downloaded in the latter case)")
    parser.add_argument("--device", default="cuda:0", help="GPU device")
    parser.add_argument("--height", type=int, default=None,
                        help="target height (default: no resize; keep the original size)")
    parser.add_argument("--width", type=int, default=None,
                        help="target width (default: no resize; keep the original size)")
    parser.add_argument("--skip_existing", action="store_true",
                        help="skip existing output files")
    parser.add_argument("--watch", action="store_true",
                        help="watch mode")
    parser.add_argument("--scan_interval", type=int, default=10,
                        help="scan interval in watch mode (seconds)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    vae, latents_mean, latents_std, vae_scale_factor = load_vae_components(
        args.model_path, args.device
    )

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
                image_path, vae, latents_mean, latents_std,
                args.device, vae.dtype,
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

    # --- Watch mode ---
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
        for filename in tqdm(new_files):
            stem = os.path.splitext(filename)[0]
            output_path = os.path.join(args.output_dir, f"{stem}.pth")
            try:
                image_path = os.path.join(args.input_dir, filename)
                latent = encode_image(
                    image_path, vae, latents_mean, latents_std,
                    args.device, vae.dtype,
                    target_height=args.height, target_width=args.width,
                )
                torch.save(latent, output_path)
            except Exception as e:
                print(f"[Error] {filename}: {e}")


if __name__ == "__main__":
    main()
