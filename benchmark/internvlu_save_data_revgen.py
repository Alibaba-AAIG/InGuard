"""
InternVL-U safety intermediate-feature collection script

What it does: runs InternVL-U image generation per prompt and captures diffusion intermediates:
  - noise_init: the initial random noise
  - prompt_embeds: the projected VLM hidden states (triple-CFG conditions)
  - prompt_attention_mask: the matching attention mask
  - latents_x1: the per-step x_0 estimate (= x_t - sigma_t * velocity)
  - sigmas: the global sigma schedule
  - image: the final generated image

Formula (Flow Matching):
    x_0_estimate = x_t - sigma_t * velocity

Directory layout:
    SAVE_DIR/
    ├── image/{event_id}.jpg
    ├── noise_init/{event_id}.pth
    ├── prompt_embeds_forward/{event_id}.pth
    ├── prompt_attention_mask_forward/{event_id}.pth
    ├── latents_x1/{0..N-1}/{event_id}.pth
    ├── sigmas.pth
    └── vae_config.pth

Usage:
    python internvlu_save_data_revgen.py                          # trainset split (default)
    INGUARD_SPLIT=testset python internvlu_save_data_revgen.py   # testset split
"""

import os
import sys
import threading
import queue
import traceback

import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm

# Use the in-repo InternVL-U pipeline package (sage/backends/internvlu/, not diffusers).
# Note: on some environments with flash-attn preinstalled it conflicts with the registration; if you hit an error, first run
#   pip uninstall -y flash-attn flash-attn-3 flash-attn-interface
# (the package automatically falls back to plain attention when flash-attn is missing; no reinstall needed).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sage", "backends"))
from internvlu import InternVLUPipeline


# ================= 1. Config =================

# Paths can be overridden via env vars (see the README "Configuration" section):
#   INGUARD_MODELS_ROOT / INGUARD_DATA_ROOT / INGUARD_OUTPUT_ROOT / INGUARD_DEVICE
_MODELS_ROOT = os.environ.get("INGUARD_MODELS_ROOT", "/path/to/models")
_DATA_ROOT   = os.environ.get("INGUARD_DATA_ROOT", "/path/to/data")
_OUT_ROOT    = os.environ.get("INGUARD_OUTPUT_ROOT", "./outputs/revgen")
# Prefer the local model dir; fall back to the HuggingFace repo id when absent (diffusers
# auto-downloads it on first use; the component classes come from the in-repo internvlu port, no official package needed)
MODEL_PATH = f"{_MODELS_ROOT}/InternVL-U" if os.path.isdir(f"{_MODELS_ROOT}/InternVL-U") else "InternVL-U/InternVL-U"

# Split selection: both splits run the exact same generation loop; only the
# source CSV and the output directory differ. Set INGUARD_SPLIT=testset to
# generate the evaluation split (default: trainset).
SPLIT = os.environ.get("INGUARD_SPLIT", "trainset")
assert SPLIT in ("trainset", "testset")
SOURCE_CSV_PATH = f"{_DATA_ROOT}/RevGen/{SPLIT}.csv"

HEIGHT, WIDTH = 1024, 1024
STEPS = 20
ALL_CFG_SCALE = 4.5
PART_CFG_SCALE = 2.0
SEED = 42
DEVICE = os.environ.get("INGUARD_DEVICE", "cuda:0")
DTYPE = torch.bfloat16

# True  = save all intermediate features (latents_x1, latents, velocity, decoded_latents_x1)
# False = save only the core data (latents_x1, noise_init, prompt_embeds, sigmas)
#         per-step latents / velocity are not saved
SAVE_FULL_TRACE = False

SAVE_DIR = f"{_OUT_ROOT}/internvl-u/{SPLIT}-seed{SEED}-{HEIGHT}-{STEPS}steps"

# Background save queue
PROCESS_QUEUE_MAXSIZE = 20
NUM_WORKERS = 3
process_queue = queue.Queue(maxsize=PROCESS_QUEUE_MAXSIZE)


# ================= 2. Helpers =================

def prepare_sub_dirs(save_dir, steps, save_full_trace):
    """Create the output directory layout"""
    base_dirs = [
        "image",
        "noise_init",
        "prompt_embeds_forward",
        "prompt_attention_mask_forward",
    ]
    step_dirs_minimal = ["latents_x1"]
    step_dirs_full = ["latents", "velocity", "decoded_latents_x1"]

    for d in base_dirs:
        os.makedirs(os.path.join(save_dir, d), exist_ok=True)

    for d in step_dirs_minimal:
        for i in range(steps):
            os.makedirs(os.path.join(save_dir, d, str(i)), exist_ok=True)

    if save_full_trace:
        for d in step_dirs_full:
            for i in range(steps):
                os.makedirs(os.path.join(save_dir, d, str(i)), exist_ok=True)

    print(f"All directories prepared. (SAVE_FULL_TRACE={save_full_trace})")


def load_and_prepare_data():
    """Load and deduplicate the CSV data"""
    print(f"Loading data from {SOURCE_CSV_PATH}...")

    df = pd.read_csv(
        SOURCE_CSV_PATH,
        usecols=["id", "prompt"],
        dtype=str,
    ).rename(columns={"id": "event_id"})

    initial_count = len(df)
    df = df.drop_duplicates(subset=["event_id"])
    df = df.drop_duplicates(subset=["prompt"])
    df = df.dropna(subset=["event_id", "prompt"])
    final_count = len(df)
    print(f"Raw rows: {initial_count}, Unique tasks: {final_count}")

    return df.reset_index(drop=True)


def load_pipeline(model_path: str):
    """Load the InternVL-U pipeline"""
    print(f"Loading InternVLUPipeline from {model_path}...")
    pipe = InternVLUPipeline.from_pretrained(
        model_path,
        torch_dtype=DTYPE,
    )
    pipe.to(DEVICE)
    print("Pipeline loaded.")
    print(f"  vae_scale_factor = {pipe.image_pipeline.vae_scale_factor}")
    print(f"  latent_channels  = {pipe.image_pipeline.latent_channels}")
    return pipe


# ================= 3. Post-processing worker =================

def post_process_worker(pipe, save_dir, save_full_trace):
    """Background thread: saves inference results to disk"""
    sigmas_path = os.path.join(save_dir, "sigmas.pth")

    while True:
        item = process_queue.get()
        try:
            if item is None:
                return

            event_id = item["event_id"]
            final_img = item["final_img"]
            prompt_embeds = item["prompt_embeds"]
            prompt_attention_mask = item["prompt_attention_mask"]
            noise_init = item["noise_init"]
            latents_all_steps = item["latents_all_steps"]
            velocities_all_steps = item["velocities_all_steps"]
            sigmas = item["sigmas"]

            # 0. save the global sigmas (idempotent)
            if sigmas is not None and not os.path.exists(sigmas_path):
                torch.save(sigmas, sigmas_path)

            # 1. save the final generated image
            final_img.save(
                os.path.join(save_dir, "image", f"{event_id}.jpg"), quality=95
            )

            # 2. save prompt_embeds and attention_mask
            if prompt_embeds is not None:
                torch.save(
                    prompt_embeds,
                    os.path.join(save_dir, "prompt_embeds_forward", f"{event_id}.pth"),
                )
            if prompt_attention_mask is not None:
                torch.save(
                    prompt_attention_mask,
                    os.path.join(save_dir, "prompt_attention_mask_forward", f"{event_id}.pth"),
                )

            # 3. save the initial noise
            if noise_init is not None:
                torch.save(
                    noise_init,
                    os.path.join(save_dir, "noise_init", f"{event_id}.pth"),
                )

            # 4. compute and save the per-step latents_x1
            n_steps = len(velocities_all_steps)
            for i in range(n_steps):
                # x_t: the current step input
                current_xt = noise_init if i == 0 else latents_all_steps[i - 1]
                v = velocities_all_steps[i]
                sigma_t = float(sigmas[i])

                # x_0 = x_t - sigma_t * v (Flow Matching)
                latent_x1 = current_xt - sigma_t * v

                torch.save(
                    latent_x1,
                    os.path.join(save_dir, "latents_x1", str(i), f"{event_id}.pth"),
                )

                if not save_full_trace:
                    continue

                # ===== the following only runs when SAVE_FULL_TRACE=True =====
                torch.save(
                    latents_all_steps[i],
                    os.path.join(save_dir, "latents", str(i), f"{event_id}.pth"),
                )
                torch.save(
                    v,
                    os.path.join(save_dir, "velocity", str(i), f"{event_id}.pth"),
                )

                # decode x1 into an image
                try:
                    with torch.no_grad():
                        img_pipe = pipe.image_pipeline
                        decoded = img_pipe.latents_to_pixels(
                            latent_x1.to(DEVICE, dtype=DTYPE)
                        )
                        decoded_np = (
                            ((127.5 * decoded + 128.0) / 255).clamp(0, 1)[0]
                            .cpu().float().numpy().transpose(1, 2, 0) * 255
                        ).astype("uint8")
                        img_x1 = Image.fromarray(decoded_np)
                        img_x1.save(
                            os.path.join(save_dir, "decoded_latents_x1", str(i), f"{event_id}.jpg"),
                            quality=90,
                        )
                except Exception as e:
                    print(f"  [Worker] x1 decode failed at step {i}: {e}")

            print(
                f"[Worker] saved event_id={event_id}, "
                f"prompt_embeds={None if prompt_embeds is None else tuple(prompt_embeds.shape)}, "
                f"noise_init={None if noise_init is None else tuple(noise_init.shape)}, "
                f"steps={n_steps}"
            )

        except Exception as e:
            eid = item.get("event_id", "UNKNOWN") if isinstance(item, dict) else "UNKNOWN"
            print(f"[Worker] Error processing {eid}: {e}")
            traceback.print_exc()
        finally:
            process_queue.task_done()


# ================= 4. Inference main logic =================

def generate_and_save_data(pipe, prompt, event_id, height, width, steps, seed):
    """
    Run one InternVL-U image generation, capturing intermediate features via hooks.
    The per-step latents are captured via a class-level callback patch.

    InternVLUPipeline.__call__ → _generate_image() → image_pipeline(...)
    image_pipeline is an InternVLUDiffusionPipeline whose __call__ supports callback_on_step_end.
    But the outer _generate_image() does not pass the callback argument through.

    Solution: patch image_pipeline's __call__ method to inject the callback.
    """
    from internvlu.diffusion.pipeline_internvlu_generation_decoder import (
        InternVLUDiffusionPipeline,
    )

    img_pipe = pipe.image_pipeline
    captured = {
        "noise_init": None,
        "prompt_embeds": None,
        "prompt_attention_mask": None,
    }
    latents_all_steps = []
    velocities_all_steps = []

    # --- Hook: prepare_latents ---
    original_prepare_latents = img_pipe.prepare_latents

    def hooked_prepare_latents(*args, **kwargs):
        lats = original_prepare_latents(*args, **kwargs)
        captured["noise_init"] = lats.detach().cpu()
        return lats

    img_pipe.prepare_latents = hooked_prepare_latents

    # --- Hook: prepare_forward_input ---
    gen_decoder = img_pipe.generation_decoder
    original_prepare_forward = gen_decoder.prepare_forward_input

    def hooked_prepare_forward(*args, **kwargs):
        result = original_prepare_forward(*args, **kwargs)
        enc_hs, attn_mask, _ = result
        captured["prompt_embeds"] = enc_hs.detach().cpu()
        captured["prompt_attention_mask"] = attn_mask.detach().cpu()
        return result

    gen_decoder.prepare_forward_input = hooked_prepare_forward

    # --- Hook: scheduler.step ---
    original_scheduler_step = img_pipe.scheduler.step

    def hooked_scheduler_step(model_output, timestep, sample, **kwargs):
        velocities_all_steps.append(model_output.detach().cpu())
        return original_scheduler_step(model_output, timestep, sample, **kwargs)

    img_pipe.scheduler.step = hooked_scheduler_step

    # --- Patch: inject the callback ---
    original_diffusion_call = InternVLUDiffusionPipeline.__call__

    def patched_diffusion_call(self, *args, **kw):
        def _callback(pipe_obj, step_index, timestep, cb_kwargs):
            if "latents" in cb_kwargs and cb_kwargs["latents"] is not None:
                latents_all_steps.append(cb_kwargs["latents"].detach().cpu())
            return cb_kwargs

        kw["callback_on_step_end"] = _callback
        kw["callback_on_step_end_tensor_inputs"] = ["latents"]
        return original_diffusion_call(self, *args, **kw)

    InternVLUDiffusionPipeline.__call__ = patched_diffusion_call

    # --- run inference ---
    generator = torch.Generator(device=DEVICE).manual_seed(seed)

    try:
        output = pipe(
            prompt=prompt,
            generation_mode="image",
            num_inference_steps=steps,
            all_cfg_scale=ALL_CFG_SCALE,
            part_cfg_scale=PART_CFG_SCALE,
            height=height,
            width=width,
            generator=generator,
        )

        sigmas = (
            img_pipe.scheduler.sigmas.detach().cpu()
            if hasattr(img_pipe.scheduler, "sigmas") and img_pipe.scheduler.sigmas is not None
            else None
        )

        data_to_save = {
            "event_id": event_id,
            "final_img": output.images[0],
            "prompt_embeds": captured["prompt_embeds"],
            "prompt_attention_mask": captured["prompt_attention_mask"],
            "noise_init": captured["noise_init"],
            "latents_all_steps": latents_all_steps,
            "velocities_all_steps": velocities_all_steps,
            "sigmas": sigmas,
        }
        process_queue.put(data_to_save)

    finally:
        img_pipe.prepare_latents = original_prepare_latents
        gen_decoder.prepare_forward_input = original_prepare_forward
        img_pipe.scheduler.step = original_scheduler_step
        InternVLUDiffusionPipeline.__call__ = original_diffusion_call


# ================= 5. Main =================

def main(num_jobs=1, target_job=0):
    prepare_sub_dirs(SAVE_DIR, STEPS, SAVE_FULL_TRACE)
    df_tasks = load_and_prepare_data()

    print(f"rows to process: {len(df_tasks)}")
    print(f"parallel workers: {NUM_WORKERS}")
    print(f"SAVE_FULL_TRACE: {SAVE_FULL_TRACE}")

    pipe = load_pipeline(MODEL_PATH)

    # save the VAE config (latents_mean, latents_std) for downstream use
    vae_config_path = os.path.join(SAVE_DIR, "vae_config.pth")
    if not os.path.exists(vae_config_path):
        torch.save(
            {
                "latents_mean": pipe.image_pipeline.vae.config.latents_mean,
                "latents_std": pipe.image_pipeline.vae.config.latents_std,
                "vae_scale_factor": pipe.image_pipeline.vae_scale_factor,
                "latent_channels": pipe.image_pipeline.latent_channels,
            },
            vae_config_path,
        )
        print(f"Saved VAE config to {vae_config_path}")

    # start the background save threads
    worker_threads = []
    for i in range(NUM_WORKERS):
        t = threading.Thread(
            target=post_process_worker,
            args=(pipe, SAVE_DIR, SAVE_FULL_TRACE),
            name=f"Worker-{i}",
            daemon=True,
        )
        t.start()
        worker_threads.append(t)

    # inference loop
    for index, row in tqdm(df_tasks.iterrows(), total=len(df_tasks)):
        if index % num_jobs != target_job:
            continue

        event_id, prompt = row["event_id"], row["prompt"]
        show_prompt = prompt[:80] + "..." if len(prompt) > 80 else prompt
        print(f"[{index}] event_id={event_id}, prompt={show_prompt}")

        try:
            generate_and_save_data(
                pipe=pipe,
                prompt=prompt,
                event_id=event_id,
                height=HEIGHT,
                width=WIDTH,
                steps=STEPS,
                seed=SEED,
            )
        except Exception as e:
            print(f"Generation failed for {event_id}: {e}")
            traceback.print_exc()
            continue

    # shut down safely
    print(f"\nInference finished; waiting for the background save queue to drain (remaining: {process_queue.qsize()})...")
    process_queue.join()

    for _ in range(NUM_WORKERS):
        process_queue.put(None)
    for t in worker_threads:
        t.join()

    print("All tasks saved successfully.")


if __name__ == "__main__":
    num_jobs = 1
    target_job = 0
    main(num_jobs, target_job)
