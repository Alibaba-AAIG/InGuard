import os
import threading
import queue

import numpy as np
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
from diffusers import Flux2KleinPipeline


# ================= 1. Config =================
# Paths can be overridden via env vars (see the README "Configuration" section):
#   INGUARD_MODELS_ROOT / INGUARD_DATA_ROOT / INGUARD_OUTPUT_ROOT / INGUARD_DEVICE
_MODELS_ROOT = os.environ.get("INGUARD_MODELS_ROOT", "/path/to/models")
_DATA_ROOT   = os.environ.get("INGUARD_DATA_ROOT", "/path/to/data")
_OUT_ROOT    = os.environ.get("INGUARD_OUTPUT_ROOT", "./outputs/revgen")
# Prefer the local model dir; fall back to the HuggingFace repo id when absent (diffusers auto-downloads it on first use)
MODEL_PATH = f"{_MODELS_ROOT}/FLUX.2-klein-base-9B/" if os.path.isdir(f"{_MODELS_ROOT}/FLUX.2-klein-base-9B/") else "black-forest-labs/FLUX.2-klein-base-9B"

# Split selection: both splits run the exact same generation loop; only the
# source CSV and the output directory differ. Set INGUARD_SPLIT=testset to
# generate the evaluation split (default: trainset).
SPLIT = os.environ.get("INGUARD_SPLIT", "trainset")
assert SPLIT in ("trainset", "testset")
SOURCE_CSV_PATH = f"{_DATA_ROOT}/RevGen/{SPLIT}.csv"

ASPECT_RATIO = (1024, 1024)
STEPS = 10
GUIDANCE_SCALE = 4.0
SEED = 42
DEVICE = os.environ.get("INGUARD_DEVICE", "cuda:0")

# True  = save all intermediate features (image, latents_x1, latents, velocity, decoded_latents_x1, ...), i.e. the original behavior
# False = save only the core data (image, latents_x1, noise_init, latent_ids, prompt_embeds, sigmas)
#         per-step latents / velocity / decoded_latents_x1 are not saved
SAVE_FULL_TRACE = False

SAVE_DIR = f"{_OUT_ROOT}/flux2-klein-base-9b/{SPLIT}-seed{SEED}-1024-{STEPS}steps"

process_queue = queue.Queue(maxsize=50)
NUM_WORKERS = 3

VIS_FIRST_N = 10


# ================= Helpers (independent of the pipeline) =================
def unpack_latents_with_ids(x, x_ids, height=None, width=None):
    """Standalone version of Flux2KleinPipeline._unpack_latents_with_ids"""
    x_list = []
    for data, pos in zip(x, x_ids):
        _, ch = data.shape
        h_ids = pos[:, 1].to(torch.int64)
        w_ids = pos[:, 2].to(torch.int64)
        h = height if height is not None else int(torch.max(h_ids).item()) + 1
        w = width if width is not None else int(torch.max(w_ids).item()) + 1
        flat_ids = h_ids * w + w_ids
        out = torch.zeros((h * w, ch), device=data.device, dtype=data.dtype)
        out.scatter_(0, flat_ids.unsqueeze(1).expand(-1, ch), data)
        out = out.view(h, w, ch).permute(2, 0, 1)
        x_list.append(out)
    return torch.stack(x_list, dim=0)


def unpatchify_latents(latents):
    """Standalone version of Flux2KleinPipeline._unpatchify_latents"""
    batch_size, num_channels, height, width = latents.shape
    latents = latents.reshape(batch_size, num_channels // 4, 2, 2, height, width)
    latents = latents.permute(0, 1, 4, 2, 5, 3)
    latents = latents.reshape(batch_size, num_channels // 4, height * 2, width * 2)
    return latents


def full_unpack_latents(packed, latent_ids, bn_mean, bn_std, latent_h, latent_w):
    """packed [B, num_patches, C] → unpacked [B, 32, H*2, W*2]"""
    x = unpack_latents_with_ids(packed, latent_ids, latent_h, latent_w)
    x = x * bn_std + bn_mean
    x = unpatchify_latents(x)
    return x


def save_channel_mean_vis(tensor_4d, save_path):
    """Take the channel mean of [1, C, H, W], normalize, and save as a grayscale image"""
    vis = tensor_4d[0].mean(dim=0)
    v_min, v_max = vis.min(), vis.max()
    vis = (vis - v_min) / (v_max - v_min + 1e-8)
    vis_uint8 = (vis.numpy() * 255).astype(np.uint8)
    Image.fromarray(vis_uint8, mode='L').save(save_path, quality=90)


# ================= 2. Data & model loading =================
def prepare_sub_dirs(save_dir, steps, save_full_trace):
    base_dirs = [
        "image",
        "noise_init",
        "latent_ids",
        "prompt_embeds_forward",
        "prompt_attention_mask_forward",
    ]
    step_dirs_minimal = ["latents_x1", "latents_x1_vis"]
    step_dirs_full = ["latents", "decoded_latents_x1", "velocity"]

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
    print(f"Loading data from {SOURCE_CSV_PATH}...")

    df = pd.read_csv(
        SOURCE_CSV_PATH,
        usecols=['id', 'prompt'],
        dtype=str
    ).rename(columns={'id': 'event_id'})

    initial_count = len(df)
    df = df.drop_duplicates(subset=['event_id'])
    df = df.drop_duplicates(subset=['prompt'])
    df = df.dropna(subset=['event_id', 'prompt'])
    final_count = len(df)
    print(f"Raw rows: {initial_count}, Unique tasks: {final_count}")

    return df.reset_index(drop=True)


def load_pipeline(model_path: str):
    print(f"Loading Flux2KleinPipeline from {model_path}...")
    pipe = Flux2KleinPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
    )
    pipe.to(DEVICE)

    # -------- model config validation --------
    assert not pipe.config.is_distilled, \
        f"model is_distilled={pipe.config.is_distilled}, expected False (non-distilled). " \
        f"Please confirm you are using the 9B base model, not the 4B distilled one."
    will_do_cfg = GUIDANCE_SCALE > 1 and not pipe.config.is_distilled
    assert will_do_cfg, \
        f"GUIDANCE_SCALE={GUIDANCE_SCALE} combined with is_distilled={pipe.config.is_distilled} " \
        f"will not enable CFG; please check the config."
    print(f"Pipeline loaded. is_distilled={pipe.config.is_distilled}, do_classifier_free_guidance={will_do_cfg}")
    return pipe


def decode_flux2_latents_to_pil(pipe, latents_unpacked):
    """VAE-decode directly from already-unpacked latents [B, 32, 128, 128]"""
    with torch.no_grad():
        l = latents_unpacked.to(pipe.device, dtype=pipe.vae.dtype)
        decoded_raw = pipe.vae.decode(l, return_dict=False)[0]
        img = pipe.image_processor.postprocess(decoded_raw, output_type="pil")[0]
        return img


# ================= 3. Post-processing worker =================
def post_process_worker(pipe, save_dir, save_full_trace, bn_mean, bn_std, latent_h, latent_w):
    sigmas_path = os.path.join(save_dir, "sigmas.pth")
    while True:
        item = process_queue.get()
        if item is None:
            process_queue.task_done()
            break

        event_id = item["event_id"]
        final_img = item["final_img"]
        prompt_embeds = item["prompt_embeds"]
        attention_mask = item["attention_mask"]
        noise_init = item["noise_init"]
        latent_ids = item["latent_ids"]
        latents_all_steps = item["latents_all_steps"]
        velocities_all_steps = item["velocities_all_steps"]
        sigmas = item["sigmas"]
        do_vis = item["do_vis"]

        try:
            if not os.path.exists(sigmas_path):
                torch.save(sigmas, sigmas_path)

            final_img.save(os.path.join(save_dir, "image", f"{event_id}.jpg"), quality=95)

            if prompt_embeds is not None:
                torch.save(prompt_embeds, os.path.join(save_dir, "prompt_embeds_forward", f"{event_id}.pth"))
            if attention_mask is not None:
                torch.save(attention_mask, os.path.join(save_dir, "prompt_attention_mask_forward", f"{event_id}.pth"))

            if noise_init is not None:
                torch.save(noise_init, os.path.join(save_dir, "noise_init", f"{event_id}.pth"))
            if latent_ids is not None:
                torch.save(latent_ids, os.path.join(save_dir, "latent_ids", f"{event_id}.pth"))

            for i in range(len(velocities_all_steps)):
                current_xt = noise_init if i == 0 else latents_all_steps[i - 1]
                v = velocities_all_steps[i]
                sigma_t = sigmas[i]

                # x1 = x_t - sigma_t * v (in packed space)
                latent_x1_packed = current_xt - sigma_t * v

                # unpack → BN denorm → unpatchify → [1, 32, 128, 128]
                latent_x1 = full_unpack_latents(
                    latent_x1_packed.float(), latent_ids, bn_mean, bn_std, latent_h, latent_w
                )

                torch.save(latent_x1, os.path.join(save_dir, "latents_x1", str(i), f"{event_id}.pth"))

                if do_vis:
                    save_channel_mean_vis(
                        latent_x1,
                        os.path.join(save_dir, "latents_x1_vis", str(i), f"{event_id}.jpg"),
                    )

                if not save_full_trace:
                    continue

                # ===== the following only runs when SAVE_FULL_TRACE=True =====
                latent_step = full_unpack_latents(
                    latents_all_steps[i].float(), latent_ids, bn_mean, bn_std, latent_h, latent_w
                )
                torch.save(latent_step, os.path.join(save_dir, "latents", str(i), f"{event_id}.pth"))

                velocity = full_unpack_latents(
                    v.float(), latent_ids,
                    torch.zeros_like(bn_mean), torch.ones_like(bn_std),
                    latent_h, latent_w
                )
                torch.save(velocity, os.path.join(save_dir, "velocity", str(i), f"{event_id}.pth"))

                img_x1 = decode_flux2_latents_to_pil(pipe, latent_x1)
                img_x1.save(
                    os.path.join(save_dir, "decoded_latents_x1", str(i), f"{event_id}.jpg"),
                    quality=90,
                )

        except Exception as e:
            print(f"Error processing event {event_id}: {e}")
        finally:
            process_queue.task_done()


# ================= 4. Inference main logic =================
def generate_and_save_data(pipe, prompt, event_id, width, height, steps, guidance_scale, generator, do_vis):
    captured_init = {}
    captured_embeddings = {}
    latents_all_steps = []
    velocities_all_steps = []

    # -------- get attention_mask via the tokenizer --------
    messages = [{"role": "user", "content": prompt}]
    text = pipe.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    tokenized = pipe.tokenizer(
        text, return_tensors="pt", padding="max_length", truncation=True, max_length=512,
    )
    captured_embeddings["attention_mask"] = tokenized["attention_mask"].cpu()

    # -------- hook encode_prompt --------
    # The 9B non-distilled version calls encode_prompt twice (positive + negative); capture only the first (positive) call
    original_encode_prompt = pipe.encode_prompt

    def hooked_encode_prompt(*args, **kwargs):
        result = original_encode_prompt(*args, **kwargs)
        if isinstance(result, tuple) and len(result) >= 2:
            pe = result[0]
            if pe is not None and "prompt_embeds" not in captured_embeddings:
                captured_embeddings["prompt_embeds"] = pe.detach().cpu()
        return result

    pipe.encode_prompt = hooked_encode_prompt

    # -------- hook prepare_latents --------
    original_prepare_latents = pipe.prepare_latents

    def hooked_prepare_latents(*args, **kwargs):
        lats, ids = original_prepare_latents(*args, **kwargs)
        captured_init["noise_init"] = lats.detach().cpu()
        captured_init["latent_ids"] = ids.detach().cpu()
        return lats, ids

    pipe.prepare_latents = hooked_prepare_latents

    # -------- hook scheduler.step to capture the CFG-merged velocity --------
    # The 9B non-distilled version uses true CFG: the transformer runs 2 forwards per step (cond + uncond),
    # and the merged noise_pred is passed to scheduler.step as model_output.
    # So hooking scheduler.step directly yields the merged velocity.
    original_scheduler_step = pipe.scheduler.step

    def hooked_scheduler_step(model_output, *args, **kwargs):
        velocities_all_steps.append(model_output.detach().cpu())
        return original_scheduler_step(model_output, *args, **kwargs)

    pipe.scheduler.step = hooked_scheduler_step

    # -------- callback saves the per-step latents --------
    def store_intermediate_callback(pipe_obj, step_index, timestep, callback_kwargs):
        latents = callback_kwargs.get("latents")
        latents_all_steps.append(latents.detach().cpu())
        return callback_kwargs

    try:
        output = pipe(
            prompt=prompt,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            generator=generator,
            callback_on_step_end=store_intermediate_callback,
            callback_on_step_end_tensor_inputs=["latents"],
        )

        # -------- runtime integrity checks --------
        assert "prompt_embeds" in captured_embeddings, \
            f"[{event_id}] encode_prompt hook did not capture prompt_embeds"
        assert "noise_init" in captured_init, \
            f"[{event_id}] prepare_latents hook did not capture noise_init"
        assert "latent_ids" in captured_init, \
            f"[{event_id}] prepare_latents hook did not capture latent_ids"

        assert len(velocities_all_steps) == steps, \
            f"[{event_id}] velocities count {len(velocities_all_steps)} != steps {steps}; " \
            f"the scheduler.step hook may have been called the wrong number of times (under CFG it should still be once per step)"
        assert len(latents_all_steps) == steps, \
            f"[{event_id}] latents count {len(latents_all_steps)} != steps {steps}; " \
            f"the callback may not have fired correctly"

        noise_init = captured_init["noise_init"]
        assert velocities_all_steps[0].shape == noise_init.shape, \
            f"[{event_id}] velocity shape {velocities_all_steps[0].shape} != noise_init shape {noise_init.shape}, " \
            f"format mismatch (all should be packed [B, num_patches, C])"
        assert latents_all_steps[0].shape == noise_init.shape, \
            f"[{event_id}] latents shape {latents_all_steps[0].shape} != noise_init shape {noise_init.shape}"

        pe = captured_embeddings["prompt_embeds"]
        print("prompt_embeds:", pe.shape)

        data_to_save = {
            "event_id": event_id,
            "final_img": output.images[0],
            "prompt_embeds": pe,
            "attention_mask": captured_embeddings.get("attention_mask"),
            "noise_init": noise_init,
            "latent_ids": captured_init["latent_ids"],
            "latents_all_steps": latents_all_steps,
            "velocities_all_steps": velocities_all_steps,
            "sigmas": pipe.scheduler.sigmas.detach().cpu(),
            "do_vis": do_vis,
        }
        process_queue.put(data_to_save)

    finally:
        pipe.encode_prompt = original_encode_prompt
        pipe.prepare_latents = original_prepare_latents
        pipe.scheduler.step = original_scheduler_step


# ================= 5. main =================
def main(num_jobs=1, target_job=0):
    prepare_sub_dirs(SAVE_DIR, STEPS, SAVE_FULL_TRACE)
    df_tasks = load_and_prepare_data()
    print(f"rows to process: {len(df_tasks)}, parallel workers: {NUM_WORKERS}, SAVE_FULL_TRACE={SAVE_FULL_TRACE}")

    pipe = load_pipeline(MODEL_PATH)
    width, height = ASPECT_RATIO

    bn_mean = pipe.vae.bn.running_mean.view(1, -1, 1, 1).cpu().float()
    bn_std = torch.sqrt(
        pipe.vae.bn.running_var.view(1, -1, 1, 1) + pipe.vae.config.batch_norm_eps
    ).cpu().float()
    latent_h = 2 * (int(height) // (pipe.vae_scale_factor * 2)) // 2
    latent_w = 2 * (int(width) // (pipe.vae_scale_factor * 2)) // 2

    bn_stats_path = os.path.join(SAVE_DIR, "bn_stats.pth")
    if not os.path.exists(bn_stats_path):
        torch.save({
            "bn_mean": bn_mean,
            "bn_std": bn_std,
            "latent_h": latent_h,
            "latent_w": latent_w,
        }, bn_stats_path)
        print(f"Saved BN stats: bn_mean={bn_mean.shape}, bn_std={bn_std.shape}, latent_h={latent_h}, latent_w={latent_w}")

    worker_threads = []
    for i in range(NUM_WORKERS):
        t = threading.Thread(
            target=post_process_worker,
            args=(pipe, SAVE_DIR, SAVE_FULL_TRACE, bn_mean, bn_std, latent_h, latent_w),
            name=f"Worker-{i}",
            daemon=True,
        )
        t.start()
        worker_threads.append(t)

    vis_count = 0
    for index, row in tqdm(df_tasks.iterrows(), total=len(df_tasks)):
        if index % num_jobs != target_job:
            continue

        event_id, prompt = row["event_id"], row["prompt"]
        show_prompt = prompt[:80] + "..." if len(prompt) > 80 else prompt
        print(f"[{index}] event_id={event_id}, prompt={show_prompt}")

        generator = torch.Generator(DEVICE).manual_seed(SEED)
        do_vis = vis_count < VIS_FIRST_N
        vis_count += 1

        try:
            generate_and_save_data(
                pipe=pipe,
                prompt=prompt,
                event_id=event_id,
                width=width,
                height=height,
                steps=STEPS,
                guidance_scale=GUIDANCE_SCALE,
                generator=generator,
                do_vis=do_vis,
            )
        except Exception as e:
            print(f"Generation failed for event_id={event_id}, error={e}")
            continue

    print(f"\nInference finished; waiting for the background save queue (remaining: {process_queue.qsize()})...")

    for _ in range(NUM_WORKERS):
        process_queue.put(None)

    process_queue.join()

    for t in worker_threads:
        t.join()

    print("All tasks saved successfully.")


if __name__ == "__main__":
    num_jobs = 1
    target_job = 0
    main(num_jobs, target_job)
