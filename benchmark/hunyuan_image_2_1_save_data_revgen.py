import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from diffusers import HunyuanImagePipeline
import threading
import queue
import traceback


# ================= 1. Config =================
# Paths can be overridden via env vars (see the README "Configuration" section):
#   INGUARD_MODELS_ROOT / INGUARD_DATA_ROOT / INGUARD_OUTPUT_ROOT / INGUARD_DEVICE
_MODELS_ROOT = os.environ.get("INGUARD_MODELS_ROOT", "/path/to/models")
_DATA_ROOT   = os.environ.get("INGUARD_DATA_ROOT", "/path/to/data")
_OUT_ROOT    = os.environ.get("INGUARD_OUTPUT_ROOT", "./outputs/revgen")
# Prefer the local model dir; fall back to the HuggingFace repo id when absent (diffusers auto-downloads it on first use)
MODEL_PATH = f"{_MODELS_ROOT}/HunyuanImage-2.1-Diffusers" if os.path.isdir(f"{_MODELS_ROOT}/HunyuanImage-2.1-Diffusers") else "hunyuanvideo-community/HunyuanImage-2.1-Diffusers"

# Split selection: both splits run the exact same generation loop; only the
# source CSV and the output directory differ. Set INGUARD_SPLIT=testset to
# generate the evaluation split (default: trainset).
SPLIT = os.environ.get("INGUARD_SPLIT", "trainset")
assert SPLIT in ("trainset", "testset")
SOURCE_CSV_PATH = f"{_DATA_ROOT}/RevGen/{SPLIT}.csv"

HEIGHT, WIDTH = 2048, 2048
STEPS = 10
DISTILLED_GUIDANCE_SCALE = 3.25
SEED = 42
DEVICE = os.environ.get("INGUARD_DEVICE", "cuda:0")
DTYPE = torch.bfloat16

# True  = save all intermediate features (image, latents_x1, latents, velocity, decoded_latents_x1)
# False = save only the core data (image, latents_x1, noise_init, prompt_embeds, sigmas)
#         per-step latents / velocity / decoded_latents_x1 are not saved
SAVE_FULL_TRACE = False

SAVE_DIR = f"{_OUT_ROOT}/hunyuan-image-2_1/{SPLIT}-seed{SEED}-2048-{STEPS}steps"

PROCESS_QUEUE_MAXSIZE = 20
NUM_WORKERS = 3

process_queue = queue.Queue(maxsize=PROCESS_QUEUE_MAXSIZE)


# ================= Helpers =================
def prepare_sub_dirs(save_dir, steps, save_full_trace):
    base_dirs = [
        "image",
        "noise_init",
        "prompt_embeds_forward",
        "prompt_embeds_mask_forward",
        "prompt_embeds_2_forward",
        "prompt_embeds_mask_2_forward",
    ]
    step_dirs_minimal = ["latents_x1"]
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
    print(f"Loading HunyuanImagePipeline from {model_path} ...")

    pipe = HunyuanImagePipeline.from_pretrained(
        model_path,
        torch_dtype=DTYPE,
    )
    pipe.to(DEVICE)

    print("Pipeline loaded.")
    print(f"pipe.vae.dtype = {pipe.vae.dtype}")
    print(f"pipe.vae.config.scaling_factor = {pipe.vae.config.scaling_factor}")
    print(f"pipe.vae.config.spatial_compression_ratio = {pipe.vae.config.spatial_compression_ratio}")
    if getattr(pipe, "transformer", None) is not None:
        print(f"pipe.transformer.dtype = {pipe.transformer.dtype}")
        print(f"pipe.transformer.config.in_channels = {pipe.transformer.config.in_channels}")
        print(f"pipe.transformer.config.guidance_embeds = {pipe.transformer.config.guidance_embeds}")
        print(f"pipe.transformer.config.use_meanflow = {pipe.transformer.config.use_meanflow}")

    return pipe


def decode_latents_to_pil(pipe, latents_4d):
    """HunyuanImage VAE decode: latents / scaling_factor → vae.decode → postprocess"""
    with torch.no_grad():
        l = latents_4d.to(pipe.device, dtype=pipe.vae.dtype)
        l = l / pipe.vae.config.scaling_factor
        image = pipe.vae.decode(l, return_dict=False)[0]
        image = pipe.image_processor.postprocess(image, output_type="pil")
        return image[0]


# ================= 2. Post-processing worker =================
def post_process_worker(pipe, save_dir, save_full_trace):
    sigmas_path = os.path.join(save_dir, "sigmas.pth")
    while True:
        item = process_queue.get()
        try:
            if item is None:
                return

            event_id = item["event_id"]
            final_img = item["final_img"]
            prompt_embeds = item.get("prompt_embeds")
            prompt_embeds_mask = item.get("prompt_embeds_mask")
            prompt_embeds_2 = item.get("prompt_embeds_2")
            prompt_embeds_mask_2 = item.get("prompt_embeds_mask_2")
            noise_init = item.get("noise_init")
            latents_all_steps = item.get("latents_all_steps", [])
            velocities_all_steps = item.get("velocities_all_steps", [])
            sigmas = item.get("sigmas")

            # 0. save the global sigmas on first flush (idempotent)
            if sigmas is not None and not os.path.exists(sigmas_path):
                torch.save(sigmas, sigmas_path)

            # 1. save the final generated image
            final_img.save(os.path.join(save_dir, "image", f"{event_id}.jpg"), quality=95)

            # 2. save the embedding
            if prompt_embeds is not None:
                torch.save(prompt_embeds, os.path.join(save_dir, "prompt_embeds_forward", f"{event_id}.pth"))
            if prompt_embeds_mask is not None:
                torch.save(prompt_embeds_mask, os.path.join(save_dir, "prompt_embeds_mask_forward", f"{event_id}.pth"))
            if prompt_embeds_2 is not None:
                torch.save(prompt_embeds_2, os.path.join(save_dir, "prompt_embeds_2_forward", f"{event_id}.pth"))
            if prompt_embeds_mask_2 is not None:
                torch.save(prompt_embeds_mask_2, os.path.join(save_dir, "prompt_embeds_mask_2_forward", f"{event_id}.pth"))

            # 3. save the initial noise
            if noise_init is not None:
                torch.save(noise_init, os.path.join(save_dir, "noise_init", f"{event_id}.pth"))

            # 4. save each step's intermediate state
            n_steps = min(len(latents_all_steps), len(velocities_all_steps))
            if sigmas is not None:
                n_steps = min(n_steps, len(sigmas))

            for i in range(n_steps):
                current_xt = noise_init if i == 0 else latents_all_steps[i - 1]
                v = velocities_all_steps[i]
                sigma_t = sigmas[i] if sigmas is not None else None

                # latents_x1 is always saved
                if current_xt is not None and sigma_t is not None:
                    latent_x1 = current_xt - sigma_t * v
                    torch.save(
                        latent_x1,
                        os.path.join(save_dir, "latents_x1", str(i), f"{event_id}.pth")
                    )
                else:
                    latent_x1 = None

                if not save_full_trace:
                    continue

                # ===== the following only runs when SAVE_FULL_TRACE=True =====
                torch.save(
                    latents_all_steps[i],
                    os.path.join(save_dir, "latents", str(i), f"{event_id}.pth")
                )
                torch.save(
                    v,
                    os.path.join(save_dir, "velocity", str(i), f"{event_id}.pth")
                )

                if latent_x1 is not None:
                    img_x1 = decode_latents_to_pil(pipe, latent_x1)
                    img_x1.save(
                        os.path.join(save_dir, "decoded_latents_x1", str(i), f"{event_id}.jpg"),
                        quality=90,
                    )

            print(
                f"[Worker] saved event_id={event_id}, "
                f"prompt_embeds={None if prompt_embeds is None else tuple(prompt_embeds.shape)}, "
                f"noise_init={None if noise_init is None else tuple(noise_init.shape)}, "
                f"latents_steps={len(latents_all_steps)}, velocities_steps={len(velocities_all_steps)}, "
                f"SAVE_FULL_TRACE={save_full_trace}"
            )

        except Exception as e:
            event_id = item["event_id"] if item is not None and isinstance(item, dict) and "event_id" in item else "UNKNOWN"
            print(f"[Worker] Error processing event {event_id}: {e}")
            traceback.print_exc()
        finally:
            process_queue.task_done()


# ================= 3. Inference main logic =================
def generate_and_save_data(pipe, prompt, event_id, height, width, steps, guidance_scale, generator):
    captured = {
        "prompt_embeds": None,
        "prompt_embeds_mask": None,
        "prompt_embeds_2": None,
        "prompt_embeds_mask_2": None,
        "noise_init": None,
    }

    latents_all_steps = []
    velocities_all_steps = []

    # --- hook encode_prompt ---
    original_encode_prompt = pipe.encode_prompt

    def hooked_encode_prompt(*args, **kwargs):
        outputs = original_encode_prompt(*args, **kwargs)
        # returns: (prompt_embeds, prompt_embeds_mask, prompt_embeds_2, prompt_embeds_mask_2)
        pe, pe_mask, pe_2, pe_mask_2 = outputs

        current_prompt = kwargs.get("prompt", args[0] if len(args) > 0 else None)
        if current_prompt == prompt or (
            isinstance(current_prompt, list) and len(current_prompt) == 1 and current_prompt[0] == prompt
        ):
            captured["prompt_embeds"] = pe.detach().cpu() if pe is not None else None
            captured["prompt_embeds_mask"] = pe_mask.detach().cpu() if pe_mask is not None else None
            captured["prompt_embeds_2"] = pe_2.detach().cpu() if pe_2 is not None else None
            captured["prompt_embeds_mask_2"] = pe_mask_2.detach().cpu() if pe_mask_2 is not None else None

        return outputs

    pipe.encode_prompt = hooked_encode_prompt

    # --- hook prepare_latents ---
    original_prepare_latents = pipe.prepare_latents

    def hooked_prepare_latents(*args, **kwargs):
        lats = original_prepare_latents(*args, **kwargs)
        captured["noise_init"] = lats.detach().cpu()
        return lats

    pipe.prepare_latents = hooked_prepare_latents

    # --- hook scheduler.step ---
    original_scheduler_step = pipe.scheduler.step

    def hooked_scheduler_step(model_output, timestep, sample, **kwargs):
        velocities_all_steps.append(model_output.detach().cpu())
        return original_scheduler_step(model_output, timestep, sample, **kwargs)

    pipe.scheduler.step = hooked_scheduler_step

    # --- callback: grab the post-step latents ---
    def store_intermediate_callback(pipe_obj, step_index, timestep, callback_kwargs):
        if "latents" in callback_kwargs and callback_kwargs["latents"] is not None:
            latents_all_steps.append(callback_kwargs["latents"].detach().cpu())
        return callback_kwargs

    try:
        output = pipe(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=steps,
            distilled_guidance_scale=guidance_scale,
            generator=generator,
            callback_on_step_end=store_intermediate_callback,
            callback_on_step_end_tensor_inputs=["latents"],
            output_type="pil",
            return_dict=True,
        )

        sigmas = getattr(pipe.scheduler, "sigmas", None)
        if sigmas is not None:
            sigmas = sigmas.detach().cpu() if torch.is_tensor(sigmas) else torch.tensor(sigmas)

        data_to_save = {
            "event_id": event_id,
            "final_img": output.images[0],
            "prompt_embeds": captured["prompt_embeds"],
            "prompt_embeds_mask": captured["prompt_embeds_mask"],
            "prompt_embeds_2": captured["prompt_embeds_2"],
            "prompt_embeds_mask_2": captured["prompt_embeds_mask_2"],
            "noise_init": captured["noise_init"],
            "latents_all_steps": latents_all_steps,
            "velocities_all_steps": velocities_all_steps,
            "sigmas": sigmas,
        }

        process_queue.put(data_to_save)

    finally:
        pipe.encode_prompt = original_encode_prompt
        pipe.prepare_latents = original_prepare_latents
        pipe.scheduler.step = original_scheduler_step


# ================= 4. Main =================
def main(num_jobs=1, target_job=0):
    prepare_sub_dirs(SAVE_DIR, STEPS, SAVE_FULL_TRACE)
    df_tasks = load_and_prepare_data()

    print(f"rows to process: {len(df_tasks)}")
    print(f"parallel workers: {NUM_WORKERS}")
    print(f"Queue maxsize: {PROCESS_QUEUE_MAXSIZE}")
    print(f"SAVE_FULL_TRACE: {SAVE_FULL_TRACE}")

    pipe = load_pipeline(MODEL_PATH)

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

    for index, row in tqdm(df_tasks.iterrows(), total=len(df_tasks)):
        if index % num_jobs != target_job:
            continue

        event_id, prompt = row["event_id"], row["prompt"]
        show_prompt = prompt[:80] + "..." if len(prompt) > 80 else prompt
        print(f"[{index}] event_id={event_id}, prompt={show_prompt}")

        generator = torch.Generator(device=DEVICE).manual_seed(SEED)

        try:
            generate_and_save_data(
                pipe=pipe,
                prompt=prompt,
                event_id=event_id,
                height=HEIGHT,
                width=WIDTH,
                steps=STEPS,
                guidance_scale=DISTILLED_GUIDANCE_SCALE,
                generator=generator,
            )
        except Exception as e:
            print(f"Generation failed for {event_id}: {e}")
            traceback.print_exc()
            continue

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
