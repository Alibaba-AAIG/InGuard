#!/usr/bin/env python3
"""build_latent_cache.py — Pre-scan OpenImages latent directories and save file list caches.

Usage:
    python3 build_latent_cache.py            # build all 4 models (skip existing)
    python3 build_latent_cache.py --force    # force rebuild all

Cache file: {latent_dir}/_latent_files.cache.txt
Format: one relative path per line (e.g. "abc123.pth" or "0/abc123.pth")

The cache is read by pretrain_image.py build_manifest() and run_distill_pretrain()
to skip the slow os.listdir() scan on OSS/network filesystems.
"""
import os
import sys
import time

# Paths can be overridden via env vars (see the README "Configuration" section):
#   INGUARD_DATA_ROOT — OpenImages data root (should contain open-images-v7/train/latents*)
_DATA_ROOT = os.environ.get("INGUARD_DATA_ROOT", "/path/to/data")

LATENT_DIRS = {
    'z': f'{_DATA_ROOT}/open-images-v7/train/latents',
    'q': f'{_DATA_ROOT}/open-images-v7/train/latents_qwen_image_2512',
    'h': f'{_DATA_ROOT}/open-images-v7/train/latents_hunyuan_image_2_1',
    'f': f'{_DATA_ROOT}/open-images-v7/train/latents_flux2_klein_base_9b',
}


def scan_latent_dir(latent_dir):
    """Scan latent directory for .pth files. Returns list of relative paths (from latent_dir).

    Supports two layouts:
      - Flat: latent_dir/*.pth → relative path = "filename.pth"
      - Subdirectory: latent_dir/0/*.pth, latent_dir/1/*.pth → relative path = "0/filename.pth"
    """
    # Check top-level first
    try:
        top_files = [f for f in os.listdir(latent_dir) if f.endswith(".pth")]
    except OSError as e:
        print(f"  [ERROR] cannot listdir: {e}")
        return []

    if top_files:
        return top_files  # flat mode: just filenames

    # Subdirectory mode
    rel_paths = []
    subdirs = sorted([d for d in os.listdir(latent_dir)
                      if os.path.isdir(os.path.join(latent_dir, d))])
    for sub in subdirs:
        sub_path = os.path.join(latent_dir, sub)
        for f in os.listdir(sub_path):
            if f.endswith(".pth"):
                rel_paths.append(f"{sub}/{f}")
    return rel_paths


def build_cache(latent_dir, force=False):
    """Scan latent_dir and save _latent_files.cache.txt. Returns file count."""
    cache_path = os.path.join(latent_dir, "_latent_files.cache.txt")

    if os.path.exists(cache_path) and not force:
        with open(cache_path) as f:
            count = sum(1 for line in f if line.strip())
        print(f"  Cache exists: {cache_path} ({count:,} files, use --force to rebuild)")
        return count

    print(f"  Scanning: {latent_dir}")
    sys.stdout.flush()
    t0 = time.time()
    rel_paths = scan_latent_dir(latent_dir)
    elapsed = time.time() - t0

    print(f"  Found {len(rel_paths):,} .pth files ({elapsed:.1f}s)")

    if not rel_paths:
        print("  [WARN] No .pth files found, skipping cache write")
        return 0

    try:
        with open(cache_path, "w") as f:
            f.write("\n".join(rel_paths))
        size_mb = os.path.getsize(cache_path) / 1024 / 1024
        print(f"  Cache saved: {cache_path} ({size_mb:.1f} MB)")
    except OSError as e:
        print(f"  [ERROR] Cannot write cache: {e}")

    return len(rel_paths)


def main():
    force = "--force" in sys.argv
    print("=" * 60)
    print("Building latent cache for all 4 models" + (" (force rebuild)" if force else ""))
    print("=" * 60)
    total = 0
    for key, latent_dir in LATENT_DIRS.items():
        print(f"\n[{key.upper()}] {latent_dir}")
        if not os.path.isdir(latent_dir):
            print(f"  [SKIP] Directory not found")
            continue
        count = build_cache(latent_dir, force=force)
        total += count
    print(f"\n{'=' * 60}")
    print(f"Total: {total:,} .pth files cached")


if __name__ == "__main__":
    main()
