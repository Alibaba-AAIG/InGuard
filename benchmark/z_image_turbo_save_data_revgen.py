import os
import threading
import queue

import torch
import pandas as pd
from tqdm import tqdm
from diffusers import ZImagePipeline


# ================= 1. Config =================
# Paths can be overridden via env vars (see the README "Configuration" section):
#   INGUARD_MODELS_ROOT / INGUARD_DATA_ROOT / INGUARD_OUTPUT_ROOT / INGUARD_DEVICE
_MODELS_ROOT = os.environ.get("INGUARD_MODELS_ROOT", "/path/to/models")
_DATA_ROOT   = os.environ.get("INGUARD_DATA_ROOT", "/path/to/data")
_OUT_ROOT    = os.environ.get("INGUARD_OUTPUT_ROOT", "./outputs/revgen")
# Prefer the local model dir; fall back to the HuggingFace repo id when absent (diffusers auto-downloads it on first use)
MODEL_PATH = f"{_MODELS_ROOT}/Z-Image-Turbo" if os.path.isdir(f"{_MODELS_ROOT}/Z-Image-Turbo") else "Tongyi-MAI/Z-Image-Turbo"

# Split selection: both splits run the exact same generation loop; only the
# source CSV and the output directory differ. Set INGUARD_SPLIT=testset to
# generate the evaluation split (default: trainset).
SPLIT = os.environ.get("INGUARD_SPLIT", "trainset")
assert SPLIT in ("trainset", "testset")
SOURCE_CSV_PATH = f"{_DATA_ROOT}/RevGen/{SPLIT}.csv"

ASPECT_RATIO = (1024, 1024)
STEPS = 9   # the official example uses 9
GUIDANCE_SCALE = 0.0   # 0.0 is recommended for Turbo models
SEED = 42
DEVICE = os.environ.get("INGUARD_DEVICE", "cuda:0")

# True  = save all intermediate features (image, latents_x1, latents, velocity, decoded_latents_x1, ...), i.e. the original behavior
# False = save only the core data (image, latents_x1, noise_init, prompt_embeds, sigmas)
#         per-step latents / velocity / decoded_latents_x1 are not saved
SAVE_FULL_TRACE = False

SAVE_DIR = f"{_OUT_ROOT}/z-image-turbo/{SPLIT}-seed{SEED}-1024-{STEPS}steps"

# Queue setup: passes data from the main inference loop to the post-processing threads
process_queue = queue.Queue(maxsize=50)
NUM_WORKERS = 3 # spawn 3 background threads to speed up saving and VAE decoding


def prepare_sub_dirs(save_dir, steps, save_full_trace):
    base_dirs = [
        "image",
        "noise_init",
        "prompt_embeds_forward",
        # "prompt_embeds_mask_forward",  # ZImage has no explicit mask
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
    
    # header=0 skips the first row
    # usecols=[0, 1] takes only the first two columns
    # names=['event_id', 'prompt'] names these two columns manually, regardless of the original header
    df = pd.read_csv(
        SOURCE_CSV_PATH,
        usecols=['id', 'prompt'],
        dtype=str
    ).rename(columns={'id': 'event_id'})
    
    # 1. print the raw count (optional)
    initial_count = len(df)
    
    # 2. deduplicate event_id
    df = df.drop_duplicates(subset=['event_id'])
    
    # 3. deduplicate prompt
    df = df.drop_duplicates(subset=['prompt'])
    
    # 4. drop rows with null values (guards against trailing empty rows at the end of the file)
    df = df.dropna(subset=['event_id', 'prompt'])
    
    final_count = len(df)
    print(f"Raw rows: {initial_count}, Unique tasks: {final_count}")
    
    return df.reset_index(drop=True)


def load_pipeline(model_path: str):
    print(f"Loading ZImagePipeline from {model_path}...")
    pipe = ZImagePipeline.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=False,
    )
    pipe.to(DEVICE)
    return pipe


def decode_zimage_latents_to_pil(pipe, latents_4d: torch.Tensor):
    """
    Decode 4D latents into a PIL image, following the final decoding logic in the ZImagePipeline source.
    latents_4d: [B, C, H, W]
    """
    with torch.no_grad():
        latents_4d = latents_4d.to(pipe.device, dtype=pipe.vae.dtype)
        latents_4d = (latents_4d / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
        image = pipe.vae.decode(latents_4d, return_dict=False)[0]
        image = pipe.image_processor.postprocess(image, output_type="pil")
        return image[0]


# ================= 2. Post-processing worker =================
def post_process_worker(pipe, save_dir, save_full_trace):
    sigmas_path = os.path.join(save_dir, "sigmas.pth")
    while True:
        item = process_queue.get()
        if item is None:
            process_queue.task_done()
            break

        event_id = item["event_id"]
        final_img = item["final_img"]
        prompt_embeds = item["prompt_embeds"]
        # prompt_embeds_mask = item["prompt_embeds_mask"]
        noise_init = item["noise_init"]
        latents_all_steps = item["latents_all_steps"]
        velocities_all_steps = item["velocities_all_steps"]
        sigmas = item["sigmas"]

        try:
            # 0. save the global sigmas on first flush (idempotent; redundant writes from multiple workers are harmless)
            if not os.path.exists(sigmas_path):
                torch.save(sigmas, sigmas_path)

            # 1. save the final generated image
            final_img.save(os.path.join(save_dir, "image", f"{event_id}.jpg"), quality=95)

            # 2. save the embedding
            if prompt_embeds is not None:
                torch.save(prompt_embeds, os.path.join(save_dir, "prompt_embeds_forward", f"{event_id}.pth"))

            # 3. save the initial noise
            if noise_init is not None:
                torch.save(noise_init, os.path.join(save_dir, "noise_init", f"{event_id}.pth"))

            # 4. save each step's intermediate data
            for i in range(len(velocities_all_steps)):
                current_xt = noise_init if i == 0 else latents_all_steps[i - 1]
                v = velocities_all_steps[i]
                sigma_t = sigmas[i]

                # same formula as the original: x1 = x_t - sigma_t * v
                latent_x1 = current_xt - sigma_t * v

                # latents_x1 is always saved (ZImage latents are already [B, C, H, W]; no unpack needed)
                torch.save(latent_x1, os.path.join(save_dir, "latents_x1", str(i), f"{event_id}.pth"))

                if not save_full_trace:
                    continue

                # ===== the following only runs when SAVE_FULL_TRACE=True =====
                torch.save(latents_all_steps[i], os.path.join(save_dir, "latents", str(i), f"{event_id}.pth"))
                torch.save(v, os.path.join(save_dir, "velocity", str(i), f"{event_id}.pth"))

                # decode x1
                img_x1 = decode_zimage_latents_to_pil(pipe, latent_x1)
                img_x1.save(
                    os.path.join(save_dir, "decoded_latents_x1", str(i), f"{event_id}.jpg"),
                    quality=90,
                )

        except Exception as e:
            print(f"Error processing event {event_id}: {e}")
        finally:
            process_queue.task_done()


# ================= 3. Inference main logic =================
def generate_and_save_data(pipe, prompt, event_id, width, height, steps, guidance_scale, generator):
    captured_init = {}
    captured_embeddings = {}
    latents_all_steps = []
    velocities_all_steps = []

    # -------- hook encode_prompt --------
    original_encode_prompt = pipe.encode_prompt

    def hooked_encode_prompt(*args, **kwargs):
        outputs = original_encode_prompt(*args, **kwargs)

        current_prompt = kwargs.get("prompt", None)
        if current_prompt is None and len(args) > 0:
            current_prompt = args[0]

        # ZImage encode_prompt returns: (prompt_embeds, negative_prompt_embeds)
        # save the positive prompt_embeds here
        if current_prompt == prompt:
            if isinstance(outputs, (tuple, list)) and len(outputs) == 2:
                pe, _ = outputs

                # pe is a list[Tensor]; each tensor has variable length
                if pe is not None:
                    captured_embeddings["prompt_embeds"] = [
                        x.detach().cpu() if torch.is_tensor(x) else x for x in pe
                    ]

                # # ZImage has no explicit prompt_embeds_mask
                # captured_embeddings["prompt_embeds_mask"] = None

        return outputs

    pipe.encode_prompt = hooked_encode_prompt

    # -------- hook prepare_latents --------
    original_prepare_latents = pipe.prepare_latents

    def hooked_prepare_latents(*args, **kwargs):
        lats = original_prepare_latents(*args, **kwargs)
        captured_init["noise_init"] = lats.detach().cpu()
        return lats

    pipe.prepare_latents = hooked_prepare_latents

    # -------- hook scheduler.step --------
    original_step = pipe.scheduler.step

    def hooked_step(model_output, timestep, sample, **kwargs):
        # model_output here is exactly the noise_pred passed to scheduler.step
        velocities_all_steps.append(model_output.detach().cpu())
        return original_step(model_output, timestep, sample, **kwargs)

    pipe.scheduler.step = hooked_step

    # -------- callback saves the per-step latents --------
    def store_intermediate_callback(pipe_obj, step_index, timestep, callback_kwargs):
        latents = callback_kwargs.get("latents")
        latents_all_steps.append(latents.detach().cpu())
        return callback_kwargs

    try:
        output = pipe(
            prompt=prompt,
            negative_prompt="",
            height=height,
            width=width,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=generator,
            callback_on_step_end=store_intermediate_callback,
            callback_on_step_end_tensor_inputs=["latents"],
        )

        pe = captured_embeddings.get("prompt_embeds", None)
        if pe is not None and isinstance(pe, list) and len(pe) > 0:
            print("prompt_embeds[0]:", pe[0].shape)

        data_to_save = {
            "event_id": event_id,
            "final_img": output.images[0],
            "prompt_embeds": captured_embeddings.get("prompt_embeds"),
            # "prompt_embeds_mask": captured_embeddings.get("prompt_embeds_mask"),
            "noise_init": captured_init.get("noise_init"),
            "latents_all_steps": latents_all_steps,
            "velocities_all_steps": velocities_all_steps,
            "sigmas": pipe.scheduler.sigmas.detach().cpu(),
        }
        process_queue.put(data_to_save)

    finally:
        pipe.encode_prompt = original_encode_prompt
        pipe.prepare_latents = original_prepare_latents
        pipe.scheduler.step = original_step


# ================= 4. main =================
def main(num_jobs=16, target_job=0):
    prepare_sub_dirs(SAVE_DIR, STEPS, SAVE_FULL_TRACE)
    df_tasks = load_and_prepare_data()
    print(f"rows to process: {len(df_tasks)}, parallel workers: {NUM_WORKERS}, SAVE_FULL_TRACE={SAVE_FULL_TRACE}")

    pipe = load_pipeline(MODEL_PATH)
    width, height = ASPECT_RATIO

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
        print(index, event_id, prompt)

        generator = torch.Generator(DEVICE).manual_seed(SEED)

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
