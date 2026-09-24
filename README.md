# InGuard: Towards Generalized Inner Guardrail for Safe Text-to-Image Generation

InGuard is an **in-pipeline safety framework** for text-to-image (T2I) diffusion
models. Instead of wrapping the model with external prompt blockers and
post-hoc image filters, InGuard enforces safety **inside the generation loop**,
using only the model's own internal representations:

1. **PE-MLP** — a lightweight MLP risk classifier operating directly on the
   model's text-encoder embeddings, grading each prompt into unsafe / risky /
   benign levels for three risk dimensions (porn, gore, IP).
2. **SAGE** (Soft-gated Asymmetric Guardrail for Embeddings) — an
   embedding-space enhancement module that repairs risky prompts by projecting
   them away from concept subspaces, aiming to return safe images that
   preserve the original request instead of rejecting it outright.
3. **Latent detector** — a ConvNeXt-Base classifier that inspects the latent
   features produced by the Flow Matching one-step estimate at an intermediate
   denoising step, so unsafe content is caught **mid-generation** and the
   remaining steps are skipped.

```
              prompt
                │
        ┌───────▼────────┐
        │  1. PE-MLP     │  prompt risk classification on text-encoder
        │  risk levels   │  embeddings (porn / gore / IP, argmax)
        └───────┬────────┘
        red ──► block (no generation)
        ip / borderline ──► 2. SAGE embedding enhancement
        white ──────────────► (no enhancement)
                │
        ┌───────▼────────┐
        │ 3. Latent      │  one-step estimate x_t − σ_t·v_t computed
        │    detector    │  inside the denoising loop at detect_step,
        │                │  fed to a ConvNeXt-Base latent classifier
        └───────┬────────┘
        unsafe ──► block (abort remaining steps, save compute)
        safe   ──► return image + full decision record
```

Compared with a conventional outer guardrail (external prompt classifier +
post-hoc image classifier), InGuard matches or exceeds end-to-end safety
(97.9–98.8%) while cutting benign disturbance by 57.5–73.5%, using ~3.7× fewer
parameters and saving up to 55.6% of the denoising compute when content is
intercepted mid-generation.

## Contents

- [Supported models](#supported-models)
- [Installation](#installation)
- [Configuration](#configuration)
- [Resources](#resources)
  - [RevGen dataset](#revgen-dataset)
  - [Released weights](#released-weights)
- [Quick start (inference with released weights)](#quick-start-inference-with-released-weights)
- [Routing logic](#routing-logic)
- [Training from scratch](#training-from-scratch)
- [Inference with your own checkpoints](#inference-with-your-own-checkpoints)
- [Tests](#tests)
- [Repository layout](#repository-layout)
- [Notes](#notes)
- [License](#license)
- [Citation](#citation)

## Supported models

All loaded through `diffusers` (InternVL-U ships in a separate repository —
see `sage/backends/internvlu/`):

| Model | Denoising steps | detect_step | Compute saved on interception |
|---|---|---|---|
| Z-Image-Turbo | 9  | 3 | 55.6% |
| Qwen-Image-2512 | 10 | 4 | 50.0% |
| HunyuanImage-2.1 | 10 | 4 | 50.0% |
| FLUX.2-klein-base-9B | 10 | 4 | 50.0% |
| InternVL-U | 20 | 8 | 55.0% |

## Installation

```bash
pip install -r requirements.txt
```

Requires Python ≥ 3.9, PyTorch ≥ 2.1, and a recent `diffusers` (the four
diffusers pipelines above are recent additions). GPU is required for
generation and recommended for training; the exported guardrail heads also
run on CPU.

## Configuration

Training, benchmark, and evaluation scripts read their default paths from
environment variables. Defaults are placeholders (`/path/to/...`) — export
the ones you need before running anything:

| Variable | Default | Meaning |
|---|---|---|
| `INGUARD_MODELS_ROOT` | `/path/to/models` | Root of the five T2I generation models |
| `INGUARD_DATA_ROOT` | `/path/to/data` | Root of the RevGen CSVs and OpenImages |
| `INGUARD_OUTPUT_ROOT` | `./outputs/revgen` | Root of generated benchmark data |
| `INGUARD_CKPT_ROOT` | `./outputs/checkpoints` | Root of training checkpoints |
| `INGUARD_DEVICE` | `cuda:0` | Default CUDA device |
| `INGUARD_OSS_MOUNT` / `INGUARD_OSS_URL` | placeholders | Optional: mount-path → HTTP-URL mapping used only by the VLM image-labeling feature |

Example:

```bash
export INGUARD_MODELS_ROOT=/data/models
export INGUARD_DATA_ROOT=/data/datasets          # RevGen CSVs live in /data/datasets/RevGen
export INGUARD_OUTPUT_ROOT=/data/outputs/revgen
export INGUARD_CKPT_ROOT=/data/outputs/checkpoints
```

**T2I model resolution:** a model directory placed at
`$INGUARD_MODELS_ROOT/<name>` (or symlinked there by the smoke test) is used
as-is; if it is absent, the scripts fall back to the public HuggingFace
repo id and diffusers downloads the model into the HF cache on first use —
no manual download required:

| Model | Local dir under `INGUARD_MODELS_ROOT` | HF repo id (fallback) |
|---|---|---|
| Z-Image-Turbo | `Z-Image-Turbo` | `Tongyi-MAI/Z-Image-Turbo` |
| Qwen-Image-2512 | `Qwen-Image-2512` | `Qwen/Qwen-Image-2512` |
| HunyuanImage-2.1 | `HunyuanImage-2.1-Diffusers` | `hunyuanvideo-community/HunyuanImage-2.1-Diffusers` |
| FLUX.2-klein-base-9B | `FLUX.2-klein-base-9B` | `black-forest-labs/FLUX.2-klein-base-9B` |
| InternVL-U | `InternVL-U` | `InternVL-U/InternVL-U` |

Some repositories may require a one-time login or license acceptance
(`hf auth login`); users behind a restricted network can set `HF_ENDPOINT`
to a mirror. The InternVL-U pipeline components are provided by the bundled
port under `sage/backends/internvlu/` — nothing extra to install.

Most scripts expose their defaults as CLI flags (`--help` lists them). The
two latent-detector training scripts are the exception — they are configured
by editing the CONFIG dict at the bottom of the file (see
[Step 6](#step-6--train-the-latent-detector-latent_detector)).

## Resources

### RevGen dataset

**RevGen** (RevGen Safety Benchmark) is the prompt dataset used to train and
evaluate the guardrail: prompts spanning porn, gore, and IP risk dimensions in
Chinese and English, each produced by reverse-generating a prompt from a real
image with a VLM (image → VLM → prompt). It is released as four flat CSV
files:

| File | Content | Consumed by |
|---|---|---|
| `trainset.csv` | Training-split prompts (raw, unlabeled) | Benchmark data generation (train split) |
| `testset.csv` | Test-split prompts (raw, unlabeled) | Benchmark data generation (test split) |
| `trainset_labeled.csv` | Training split + human prompt-level risk labels (`label_porn_risk_level` / `label_gore_risk_level` / `label_ip_risk_level`) | PE-MLP training |
| `testset_labeled.csv` | Test split + the same labels | PE-MLP evaluation, prompt-side ground truth in system-level evaluation |

The training split ships pre-deleaked: prompts matching the test set
(prompt Jaccard + image pHash) were removed before release, so the training
scripts need no runtime leak filtering.

Download and unpack it so that the four CSVs sit directly under
`$INGUARD_DATA_ROOT/RevGen/`:

```bash
hf download Alibaba-AAIG/RevGen --repo-type dataset \
    --local-dir $INGUARD_DATA_ROOT/RevGen
```

The dataset is mirrored on [ModelScope](https://modelscope.cn/datasets/Alibaba-AAIG/RevGen).
RevGen is released under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)
(research and non-commercial use only), separately from the Apache-2.0 code and weights.

> Note: per-image labels (`labels_llm/predictions.csv`, used to train the
> latent detector) are **not** part of the CSV release — they are produced by
> labeling the generated images with a VLM API. See
> [Step 2 of Training from scratch](#training-from-scratch).

### Released weights

The deployed guardrail heads for all five models (PE-MLP heads + latent
detectors) are released as a single archive matching the `weights/` layout:

```
weights/
├── latent_detector/<model>/model.pth + config.json
├── pe_mlp/<model>/model.pth
└── manifest.json        # per-model status + latent input channels
```

```bash
hf download Alibaba-AAIG/InGuard --local-dir ./weights
```

The weights are mirrored on [ModelScope](https://modelscope.cn/models/Alibaba-AAIG/InGuard).

Not included in the release (by design): the Phase-A distillation-pretrained
backbone (retrain it with the provided scripts if you need it from scratch),
and the five T2I generation models themselves (either place them under
`INGUARD_MODELS_ROOT`, or rely on the automatic HuggingFace download — see
the model-resolution table above).

## Quick start (inference with released weights)

After [installation](#installation) and downloading the
[released weights](#released-weights) into `weights/`:

```python
from integration.guardrail_pipeline import GuardrailPipeline

guardrail = GuardrailPipeline(
    "z-image-turbo",
    model_path="/path/to/Z-Image-Turbo",
    device="cuda",
)

result = guardrail.generate("a scenic photo of mountains at sunset", seed=42)

result.decision        # "blocked_prompt" | "enhanced" | "passed" | "blocked_latent"
result.image           # PIL.Image, or None when blocked
result.tier            # "red" | "ip" | "borderline" | "white"
result.risk_levels     # (porn, gore, ip) PE-MLP argmax levels
result.alpha_used      # SAGE strength applied (None if not enhanced)
result.detector_result # latent detector output (probs + preds) if stage 3 ran
result.steps_saved     # denoising steps skipped by early termination
```

Or from the command line:

```bash
python integration/example.py \
    --model z-image-turbo \
    --model_path /path/to/Z-Image-Turbo \
    --prompt "a scenic photo of mountains at sunset"
```

For an interactive session (type prompts in the terminal, images are saved to
`interactive_output/`, blocked prompts report the blocking stage and whether
SAGE was applied):

```bash
python integration/interactive.py \
    --model z-image-turbo \
    --model_path /path/to/Z-Image-Turbo
```

All five models in the table above work with `example.py` / `interactive.py`
and `GuardrailPipeline`. For InternVL-U, point `--model_path` at the local
model repository and mind the `transformers==4.52.3` pin (see
[Notes](#notes)); the SAGE-enhanced embedding is injected by hooking the
pipeline's `prepare_forward_input` (its triple-CFG encoding is kept intact).

## Routing logic

Every prompt is classified into one of four tiers:

| Tier | Condition (PE-MLP levels) | Action |
|---|---|---|
| red | porn ≥ τ_p or gore ≥ τ_g | block the prompt, no generation |
| ip | ip ∈ [1, 5] | SAGE with α_i, then latent detection |
| borderline | porn ≥ 3 or gore ≥ 2 (below red) | SAGE with α_p / α_g, then latent detection |
| white | everything else | no SAGE; latent detection on the original latent (fallback mode) |

The latent detector runs for every non-red tier at `detect_step`. SAGE concept
groups follow the highest-priority triggered category (porn > gore > ip);
borderline prompts merge the porn/gore concept subspaces, ip prompts use the
IP subspace with the prompt's detected character.

All thresholds and strengths are fixed per model in `configs/<model>.json`:
risk thresholds τ_p/τ_g, per-category SAGE strengths α_p/α_g/α_i, asymmetric
soft-gating temperatures, and the deployment step. `detect_step` uses 0-based
step indexing; e.g. `z-image-turbo` checks after the 4th of 9 denoising
steps, skipping the remaining 5.

## Training from scratch

The complete workflow, from raw prompts to a deployed guardrail (steps in
brackets map to the sections below):

```
prompt stream : RevGen CSVs ─► (1) data generation ─► (3) PE-MLP training ─► (4) SAGE enhanced data
latent stream : (1) data generation ─► (2) image labels ─┐
                OpenImages ─► (5) VAE encode ─► (6a) Phase-A pretrain ─► (6b) Phase-B finetune ◄┘
deployment    : (3) PE-MLP + (6) latent detector ─► (7) export weights ─► (8) verify ─► inference / evaluation
```

### Step 1 — Generate per-model data (`benchmark/`)

For each of the four diffusers models, run the generation script once per
split (it reads `$INGUARD_DATA_ROOT/RevGen/{trainset,testset}.csv` selected
by `INGUARD_SPLIT` and writes under `$INGUARD_OUTPUT_ROOT`):

```bash
# (a) Test split (evaluation data): prompt → full generation loop, saving
#     image / prompt_embeds_forward / noise_init / latents_x1 / sigmas.
INGUARD_SPLIT=testset python benchmark/z_image_turbo_save_data_revgen.py

# (b) Guardrail training split: same script, same layout (INGUARD_SPLIT
#     defaults to trainset).
python benchmark/z_image_turbo_save_data_revgen.py
```

> Sharding note: every `*_save_data_revgen.py` ends with a `__main__` block
> whose `num_jobs` / `target_job` split the prompt list across GPUs (sample
> `index % num_jobs == target_job` is kept). The checked-in defaults
> (`num_jobs = 1`, `target_job = 0`) run the full split on a single GPU; to
> shard across N cards, set `num_jobs = N, target_job = 0..N-1` at the bottom
> of the script and submit each card separately.

Output layout (consumed by all training scripts), written to
`$INGUARD_OUTPUT_ROOT/<model>/{trainset,testset}-seed42-<res>-<steps>steps`:

```
<save_dir>/
├── image/{id}.jpg                    # generated image
├── prompt_embeds_forward/{id}.pth    # text-encoder embeddings (PE-MLP input)
├── noise_init/{id}.pth               # initial noise
├── latents_x1/{step}/{id}.pth        # one-step estimate at each step
└── sigmas.pth                        # sigma schedule (shared)
```

Each of the four diffusers models has its own `<model>_save_data_revgen.py` /
`<model>_vae_encode.py` under `benchmark/` (replace
the `z_image_turbo` prefix with `qwen_image_2512`, `hunyuan_image_2_1`,
`flux2_klein_base_9b`; per-model resolution and steps are fixed inside the
scripts). InternVL-U (not a diffusers pipeline) ships the same script family
with the `internvlu_` prefix; its scripts load the bundled pipeline package
from `sage/backends/internvlu/` and require a recent `transformers` (see
[Notes](#notes)).

### Step 2 — Label the generated images (VLM API)

The latent detector trains on **per-image** safety labels
(`labels_llm/predictions.csv` inside each generated split directory), which
are produced by a Qwen-VL API (DashScope) labeling pass over the generated
images — they are not part of the CSV release. Labeling is incremental:
already-labeled images are skipped, and the result is cached permanently.

Provide an API key via `--vlm_api_key` or the `DASHSCOPE_API_KEY`
environment variable (**no key is ever stored in this repository**).

```bash
# Recommended: batch mode — auto-discovers every
# {model}/{split}-seed*-*/image directory under the data root and labels
# each into the sibling labels_llm/predictions.csv
python scripts/label_images.py \
    --data_root ./outputs/revgen \
    --model_names z-image-turbo \
    --splits trainset,testset

# or label a single directory explicitly
python scripts/label_images.py \
    --image_dir ./outputs/revgen/z-image-turbo/trainset-seed42-1024-9steps/image
```

Labeling is incremental: already-labeled images are skipped, and an existing
`predictions.csv` is never re-generated. Run it once per split directory —
Phase-B latent-detector training (Step 6) reads `labels_llm/predictions.csv`
directly, and nothing else in the training flow labels on demand.

### Step 3 — Train the PE-MLP risk classifier (`pe_mlp/`)

PE-MLP pools the saved `prompt_embeds_forward` (masked mean pooling) and
trains three classification heads (porn / gore / IP) on the human
prompt-level labels:

```bash
python pe_mlp/train_prompt.py \
    --model_name z-image-turbo \
    --train_csv $INGUARD_DATA_ROOT/RevGen/trainset_labeled.csv \
    --test_csv  $INGUARD_DATA_ROOT/RevGen/testset_labeled.csv \
    --ckpt_root ./outputs/checkpoints
```

`--train_data_dir` / `--test_data_dir` default to the directories generated
in Step 1 (per model), so only the CSV paths may need overriding. Training
writes checkpoints and
`meta/test_predictions.csv` (id, prompt, pred_porn, pred_gore, pred_ip) —
the gating predictions SAGE consumes next.

### Step 4 — Generate SAGE-enhanced data (`sage/`)

SAGE enhances only prompts flagged as risky by PE-MLP, so it needs the gating
predictions from Step 3:

```bash
# auto-discover the best PE-MLP run under --ckpt_root (recommended)
python sage/sage_enhance.py \
    --model_name z-image-turbo \
    --model_path /path/to/Z-Image-Turbo \
    --ckpt_root ./outputs/checkpoints --auto_csv

# or point at the predictions CSV directly
python sage/sage_enhance.py --model_name z-image-turbo \
    --model_path /path/to/Z-Image-Turbo \
    --csv_path /path/to/test_predictions.csv
```

Output is written to `{save_dir_base}/sage/alpha_<α>/...` with the same
layout as Step 1, and provides the SAGE-enhanced images used to evaluate
disturbance and end-to-end safety. Use `--target_job k --num_jobs K` to shard
across GPUs.

### Step 5 — Prepare Phase-A pretraining data

Phase A distills a frozen RGB teacher into a latent-input backbone using
OpenImages VAE latents. OpenImages is **not** redistributed — download it
yourself and encode it with the model's VAE (the `<model>_vae_encode.py`
family shown in Step 1):

```bash
python benchmark/z_image_turbo_vae_encode.py \
    --input_dir $INGUARD_DATA_ROOT/open-images-v7/train/images \
    --output_dir $INGUARD_DATA_ROOT/open-images-v7/train/latents \
    --model_path /path/to/Z-Image-Turbo

# optional: prescan latent directories into a cache file (speeds up
# manifest building on network filesystems)
python latent_detector/build_latent_cache.py [--force]
```

The expected output directory names (consumed by `build_latent_cache.py`
and the `latent_dir` entry of `pretrain_image.py`'s CONFIG block) are:

| model | OpenImages latent dir (under `$INGUARD_DATA_ROOT`) |
| --- | --- |
| z-image-turbo | `open-images-v7/train/latents` |
| qwen-image-2512 | `open-images-v7/train/latents_qwen_image_2512` |
| hunyuan-image-2_1 | `open-images-v7/train/latents_hunyuan_image_2_1` |
| flux2-klein-base-9b | `open-images-v7/train/latents_flux2_klein_base_9b` |

Encode the other models with their own `*_vae_encode.py` scripts (same CLI).

### Step 6 — Train the latent detector (`latent_detector/`)

```bash
# Phase A: distillation pretraining on OpenImages VAE latents
#   (stage-wise cosine distillation from a frozen RGB ConvNeXt-Base teacher;
#    configure paths in the __main__ config block at the bottom of the file)
python latent_detector/pretrain_image.py

# Phase B: multi-task finetuning on the guardrail training set
#   (labels = labels_llm/predictions.csv from Step 2;
#    separate backbone/head learning rates; config block at the bottom)
python latent_detector/train_image.py
```

Both scripts take no CLI flags — they are configured by editing the CONFIG
dict at the bottom of the file (paths, backbone, lr, epochs); running the
script starts training immediately with the CONFIG as-is. The default
CONFIG targets z-image-turbo; for another model also update `model`,
`train_csv` / `test_csv`, `num_steps` / `latent_val_step`, and
`latent_in_chans` (16 for Z-Image-Turbo / Qwen-Image-2512 / InternVL-U,
32 for FLUX.2-klein-base-9B, 64 for HunyuanImage-2.1). To initialize Phase B
from a Phase-A pretrained backbone, point `pretrain_dir` at the
`pretrain_image.py` output directory (it picks `backbone_iter100000.pth`
when present — the fair-comparison baseline — otherwise the largest-iter
`backbone_iter*.pth`; empty = start from torchvision IN1K weights).

The released dataset ids are random opaque strings, so *borderline* bucketing
(which merges the porn/gore concept subspaces during training) can no longer be
inferred from the id. To restore it, point `train_prompt_csv` / `test_prompt_csv`
at the released `trainset.csv` / `testset.csv` so the category column drives the
bucketing; left empty, borderline samples fall back to `normal_other` (harmless —
the affected consumers are off by default and only change stratification
granularity).

### Step 7 — Export deployment weights (`scripts/`)

```bash
python scripts/export_weights.py \
    --ckpt_root ./outputs/checkpoints \
    --export_dir ./weights
```

This locates, for every model, the distillation-pretrained latent detector
checkpoint (`best_val_avg_f1_model.pth`, selected by validation avg-F1) and the
PE-MLP checkpoint with the highest validation accuracy, renames each to
`model.pth`, and copies them into the fixed `weights/` layout described
in [Released weights](#released-weights), together with a `manifest.json`
recording the deployed configuration (checkpoint name and `latent_in_chans`).

### Step 8 — Verify the deployment

```bash
# 1. files + manifest consistency (seconds, no torch needed)
python scripts/verify_pipeline.py --model z-image-turbo --stage quick

# 2. load both guardrail heads + random-input forward (torch, CPU ok)
python scripts/verify_pipeline.py --model z-image-turbo --stage components

# 3. full chain on real prompts covering the four routing tiers
python scripts/verify_pipeline.py --model z-image-turbo --stage e2e \
    --model_path /path/to/Z-Image-Turbo
```

The `quick` stage cross-checks the exported checkpoints against the deployed
configuration, so a wrong export is caught before any GPU time is spent.

## Inference with your own checkpoints

Once your own training run is exported into `weights/` (Step 7), inference is
identical to the [Quick start](#quick-start-inference-with-released-weights)
— `GuardrailPipeline` loads whatever sits in `weights/`, whether downloaded
or exported by you. `python scripts/verify_pipeline.py --stage quick` will
confirm the consistency of your export.

## Tests

Lightweight unit tests (standard library `unittest`, no extra dependencies;
`tests/test_pe_mlp.py` needs a CPU-only torch):

```bash
python -m unittest discover -s tests -v
```

- `tests/test_routing.py` — the four-tier routing of Algorithm 1: an
  exhaustive truth table (5 model configs × 6 porn × 6 gore × 8 IP levels =
  1440 cases) against an independent reference implementation, plus the
  enhancement gate, category priority, and detection routing.
- `tests/test_pe_mlp.py` — PE-MLP checkpoint loading (including the
  `weights_only` fallback path), forward output ranges, and masked pooling.
- `tests/test_internvlu_pipeline.py` — the InternVL-U generation path
  (offline mocks): SAGE-embedding injection via the `prepare_forward_input`
  hook, per-step latent detection with early termination, hook restoration,
  and the `generate()` routing branch.
- `tests/test_verify_quick.py` — the `quick` stage of
  `scripts/verify_pipeline.py` (file presence + manifest consistency) against
  a synthetic `weights/` tree, healthy and broken.

### Automated smoke test (training-side reproduction)

`scripts/smoke_test.py` chains the checks above with the training-side
smoke path (data layout -> unit tests -> `verify_pipeline` -> a 1-epoch
PE-MLP train) on a machine that already holds the RevGen CSVs and the
benchmark data:

```bash
cp scripts/local_paths.env.example scripts/local_paths.env
# edit scripts/local_paths.env: fill in the real server paths
python scripts/smoke_test.py --stage all
```

`scripts/local_paths.env` is **gitignored** — real paths never enter the
repository. The `link` stage adapts an existing data layout to the open-source
one with symlinks only (no copying): CSVs under their original names,
generated dirs with an optional version infix (e.g. `testset-vX-Y-seed42-...`), and
T2I models stored under other names (per-model full-path overrides). Each
stage can also be run alone (`--stage env|link|tests|verify|pe`).

## Repository layout

```
├── configs/               # per-model deployed hyperparameters (JSON)
├── integration/           # end-to-end pipeline (the main entry point)
│   ├── guardrail_pipeline.py   # PE-MLP → SAGE → latent detection, in-loop
│   ├── routing.py              # tier classification + GuardrailResult
│   ├── pe_mlp_infer.py         # PE-MLP online inference wrapper
│   ├── example.py              # CLI usage example (single prompt)
│   └── interactive.py          # terminal REPL demo (type prompts, get images/decisions)
├── sage/                  # SAGE embedding enhancement
│   ├── sage_enhance.py         # SAGE: soft-gated asymmetric projection
│   ├── projection.py           # shared projection-matrix utility
│   ├── toxic_concepts.py       # concept groups (porn / gore / IP)
│   └── backends/               # per-model adapters (ModelAdapter)
│       ├── base.py             # adapter interface
│       ├── run_zimage.py / run_qwen_image.py / run_hunyuan_image.py
│       ├── run_flux2_klein.py / run_internvlu.py
│       └── internvlu/          # InternVL-U pipeline (non-diffusers)
├── latent_detector/       # latent safety detector
│   ├── export/                 # self-contained inference (model + weights)
│   ├── pretrain_image.py       # Phase-A distillation pretraining
│   ├── train_image.py          # Phase-B multi-task finetuning
│   ├── build_latent_cache.py   # latent manifest cache
├── pe_mlp/                 # PE-MLP prompt risk classifier
│   └── train_prompt.py
├── benchmark/              # per-model data generation (save_data,
│                           # vae_encode)
├── common/                 # shared training utilities
├── scripts/
│   ├── export_weights.py  # training→deployment weight export (see Step 7)
│   ├── label_images.py    # VLM image-labeling CLI (see Step 2)
│   └── verify_pipeline.py # staged deployment verification
├── tests/                 # unit tests (routing / PE-MLP / InternVL-U pipeline / verify-quick)
└── weights/                # populated by export / download (gitignored)
```

## Notes

- InternVL-U is not a diffusers pipeline; its adapter and pipeline code live
  in `sage/backends/internvlu/` and pin `transformers==4.52.3` (the processor
  imports `InternVLImagesKwargs`, removed in transformers 5.x — the four
  diffusers models are unaffected). On some
  environments with a preinstalled `flash-attn`, importing the InternVL-U
  pipeline raises a registration conflict — uninstall the conflicting
  package first (`pip uninstall -y flash-attn flash-attn-3
  flash-attn-interface`); the pipeline automatically falls back to plain
  attention when flash-attn is absent.
- Training and benchmark scripts default to placeholder paths
  (`/path/to/models/...`, `/path/to/data/...`). Every path can be overridden
  via CLI flags or the [INGUARD_* environment variables](#configuration).
- The VLM labeling feature (Step 2) calls the DashScope API and requires an
  API key (`--vlm_api_key` or `DASHSCOPE_API_KEY`). No key is stored anywhere
  in this repository.

## License

InGuard code and released guardrail weights are provided under the
[Apache License 2.0](LICENSE). Bundled third-party components retain their
original terms; see [Third-Party Notices](THIRD_PARTY_NOTICES.md). The RevGen
dataset is separately released under CC BY-NC 4.0.

## Citation

If you use InGuard in your research, please cite our [paper](https://arxiv.org/abs/2609.27620):

```bibtex
@article{wang2026inguard,
  title   = {InGuard: Towards Generalized Inner Guardrail for
             Safe Text-to-Image Generation},
  author  = {Wang, Zeyu and Li, Xiaodan and Li, Zhiwen and
             Chen, Yuefeng and Xue, Hui},
  journal = {arXiv preprint arXiv:2609.27620},
  year    = {2026},
  doi     = {10.48550/arXiv.2609.27620},
  url     = {https://arxiv.org/abs/2609.27620}
}
```
