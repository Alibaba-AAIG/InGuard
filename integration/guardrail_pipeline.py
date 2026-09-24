"""End-to-end Inner-Guardrail pipeline: PE-MLP + SAGE + Latent detector.

Wraps an image-generation pipeline (Z-Image-Turbo, Qwen-Image-2512,
HunyuanImage-2.1, FLUX.2-Klein-base-9B via diffusers; InternVL-U via the
bundled port in sage/backends/internvlu/) with the three-stage inner guardrail:

    Stage 1  PE-MLP prompt risk classification   -> integration/pe_mlp_infer.py
    Stage 2  SAGE embedding enhancement          -> sage/sage_enhance.py
    Stage 3  Latent safety detection at an early
             denoising step, on the fly          -> latent_detector/export/

Routing follows Algorithm 1 (Appendix A.3 of the paper): each prompt is mapped
to one of four tiers -- red (block before generation), ip (SAGE with alpha_i),
borderline (SAGE with alpha_p / alpha_g), or white (no SAGE) -- and every
non-red tier additionally passes the latent detector at ``detect_step`` (the
"w/ fallback" deployment), where the one-step estimate

    latent_x1 = x_t - sigma_t * v_t

is computed inside the denoising hook and fed directly to the detector,
without any disk round-trip.

``detect_step`` uses the same 0-based step indexing as the offline evaluation
artifacts (``latents_x1/<step>/`` directories produced by the data pipeline),
i.e. the deployed step~3 for Z-Image-Turbo (paper Sec. 5.3) corresponds to
``"detect_step": 3`` in configs/z-image-turbo.json. The reported 50-56%
denoising-compute savings equal (num_steps - 1 - detect_step) / num_steps.

Usage:
    from integration.guardrail_pipeline import GuardrailPipeline

    guardrail = GuardrailPipeline(
        "z-image-turbo",
        model_path="/path/to/Tongyi-MAI/Z-Image-Turbo",
        device="cuda",
    )
    result = guardrail.generate("a scenic photo of mountains", seed=42)

    result.decision        # "blocked_prompt" | "enhanced" | "passed" | "blocked_latent"
    result.image           # PIL.Image, or None when blocked
    result.tier            # "red" | "ip" | "borderline" | "white"
    result.risk_levels     # (porn, gore, ip) PE-MLP argmax levels
    result.detector_result # latent detector output dict (stage 3, if run)
    result.steps_saved     # denoising steps skipped by early termination

InternVL-U is not a diffusers pipeline; it loads through the bundled port in
sage/backends/internvlu/ (see load_pipeline in sage/sage_enhance.py). Its
pipeline does its own triple-CFG prompt encoding (processor -> VLM ->
prepare_forward_input), so instead of passing prompt_embeds as call arguments
the SAGE-enhanced cond embedding is injected by hooking prepare_forward_input
(row 0 of the 3-way batch); latent detection hooks the scheduler exactly as
above (see _generate_internvlu_with_detection).
"""

import json
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _sub in ("integration", "sage", "sage/backends", "latent_detector/export"):
    _p = os.path.join(_REPO_ROOT, _sub)
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

import torch

from routing import GuardrailResult, classify_tier, sage_route, needs_detection
from pe_mlp_infer import PEMLPRiskClassifier
from inference import LatentDetector
from projection import projection_matrix
from sage_enhance import (
    MODEL_REGISTRY,
    get_target_dtype,
    load_pipeline,
    precompute_concept_vectors,
    select_concept_vectors,
    build_P_C,
    get_flux_bn_stats,
    full_unpack_latents,
)
from toxic_concepts import PORN_CONCEPTS, GORE_CONCEPTS, IP_CODE_TO_CONCEPT


DEFAULT_WEIGHTS_ROOT = os.path.join(_REPO_ROOT, "weights")


class _LatentBlocked(Exception):
    """Raised inside the denoising loop when the latent detector fires."""

    def __init__(self, detector_result):
        super().__init__("latent detector: unsafe")
        self.detector_result = detector_result


class GuardrailPipeline:
    """Diffusers pipeline wrapped with the inner guardrail (PE-MLP + SAGE +
    latent detector). All hyperparameters (thresholds, alphas, detect_step)
    come from configs/<model_name>.json, which mirrors the paper's deployed
    configuration (Table threshold_params + Sec. 5.3)."""

    def __init__(self, model_name, model_path, weights_root=None, device="cuda",
                 config_path=None):
        if model_name not in MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model {model_name!r}. Supported: {sorted(MODEL_REGISTRY)}")

        if config_path is None:
            config_path = os.path.join(_REPO_ROOT, "configs", f"{model_name}.json")
        with open(config_path) as f:
            self.cfg = json.load(f)

        self.model_name = model_name
        self.device = device
        self.model_cfg = MODEL_REGISTRY[model_name]

        # ---- generation pipeline ----
        if model_path is None:
            model_path = self.model_cfg["model_path"]
        self.pipe = load_pipeline(model_name, model_path, device)
        self.target_dtype = get_target_dtype(self.pipe, model_name)

        # ---- text adapter (per-model encoding quirks) ----
        if self.model_cfg.get("is_internvlu"):
            # InternVL-U: the adapter drives the processor itself, so it needs
            # the generation geometry and CFG scales up front (mirrors the
            # construction in sage_enhance.py main()).
            self.adapter = self.model_cfg["adapter_class"](
                self.pipe,
                device=device,
                height=self.cfg["height"],
                width=self.cfg["width"],
                num_inference_steps=self.cfg["num_steps"],
                all_cfg_scale=self.cfg["guidance_value"],
                part_cfg_scale=self.cfg["part_cfg_scale"],
            )
        else:
            self.adapter = self.model_cfg["adapter_class"](self.pipe, device)

        # ---- stage 1: PE-MLP prompt risk classifier ----
        weights_root = weights_root or DEFAULT_WEIGHTS_ROOT
        pe_ckpt = os.path.join(weights_root, "pe_mlp", model_name, "model.pth")
        if not os.path.exists(pe_ckpt):
            raise FileNotFoundError(
                f"PE-MLP checkpoint not found: {pe_ckpt}\n"
                "Run scripts/export_weights.py on the training server (or "
                "download the released weights) to populate "
                "weights/pe_mlp/<model>/model.pth.")
        self.prompt_classifier = PEMLPRiskClassifier(pe_ckpt, device=device)

        # ---- stage 3: latent safety detector ----
        det_dir = os.path.join(weights_root, "latent_detector", model_name)
        if not os.path.exists(os.path.join(det_dir, "model.pth")):
            raise FileNotFoundError(
                f"Latent detector checkpoint not found: {det_dir}\n"
                "Run scripts/export_weights.py on the training server (or "
                "download the released weights) to populate "
                "weights/latent_detector/<model>/.")
        self.detector = LatentDetector(model_name, ckpt_dir=det_dir, device=device)

        # ---- stage 2: SAGE concept vectors (encoded once, reused per prompt) ----
        self._concept_groups = {
            "porn": PORN_CONCEPTS,
            "gore": GORE_CONCEPTS,
            "ip": IP_CODE_TO_CONCEPT,
        }
        self.concept_vecs = precompute_concept_vectors(
            self.adapter, concept_groups=self._concept_groups)

        # ---- per-model precomputed negative embeddings (Qwen / FLUX CFG) ----
        self._negative_embeds = None
        if self.model_cfg["needs_negative_embeds"]:
            if model_name == "qwen-image-2512":
                neg_e, neg_m = self.adapter._get_negative_embeds("")
                self._negative_embeds = (
                    neg_e.to(dtype=self.target_dtype, device=device),
                    neg_m.to(dtype=torch.long, device=device),
                )
            elif model_name == "flux2-klein-base-9b":
                neg_e = self.adapter._get_negative_embeds("")
                self._negative_embeds = (
                    neg_e.to(dtype=self.target_dtype, device=device),
                    None,
                )

    # ------------------------------------------------------------------
    # Stage 2: SAGE (asymmetric soft-gated projection)
    # ------------------------------------------------------------------

    def _build_sage_embedding(self, prompt, full_emb, user_mask, porn_level,
                              gore_level, ip_level, alpha):
        """Apply SAGE to one prompt's embeddings.

        Mirrors run_sage() in sage/sage_enhance.py line-by-line: concept
        projection, per-token ratio, asymmetric soft gating, and the merged
        embedding write-back. Returns (safe_full, n_trigger, concept_names),
        or (None, 0, ()) when no concept vector is available.
        """
        selected_vecs, selected_names = select_concept_vectors(
            self.concept_vecs, porn_level, gore_level, ip_level,
            concept_names=self._concept_groups)
        if selected_vecs is None:
            return None, 0, ()

        P_C, I_minus_Pc = build_P_C(selected_vecs, self.device)
        original = full_emb[user_mask]                     # [N, D] float32
        n_tokens = original.shape[0]

        p_masked = self.adapter.encode_pooled_masked_per_token(prompt).float()
        if p_masked.shape[0] != n_tokens:                  # defensive, as in run_sage
            m = min(p_masked.shape[0], n_tokens)
            p_masked = p_masked[:m]
            n_tokens = m
            original = original[:m]

        P_I = projection_matrix(p_masked.T)
        dist = torch.norm(I_minus_Pc @ p_masked.T, dim=0)

        means = []
        for i in range(n_tokens):
            others = torch.cat((dist[:i], dist[i + 1:]))
            means.append(others.mean() if others.numel() > 0 else dist[i])
        mean_dist = torch.stack(means)
        ratio = dist / (mean_dist + 1e-8)

        new_text_e = (I_minus_Pc @ P_I @ original.T).T     # [N, D]

        # asymmetric soft gating (TAU_TOXIC / TAU_SAFE)
        delta = ratio - (1.0 + alpha)
        tau_eff = torch.where(
            delta >= 0,
            torch.full_like(delta, self.cfg["tau_toxic"]),
            torch.full_like(delta, self.cfg["tau_safe"]),
        )
        gamma = torch.sigmoid(delta / tau_eff)
        merged = (1.0 - gamma.unsqueeze(1)) * original + gamma.unsqueeze(1) * new_text_e

        safe_full = full_emb.clone()
        mask_idx = user_mask.nonzero(as_tuple=True)[0][:n_tokens]
        safe_full[mask_idx] = merged

        return safe_full, int((gamma > 0.5).sum().item()), tuple(selected_names)

    # ------------------------------------------------------------------
    # Stage 3: in-loop latent detection
    # ------------------------------------------------------------------

    def _detector_input(self, latent_x1, latent_ids, height, width):
        """Convert the one-step latent estimate to the detector's [B, C, H, W]
        input, applying the same per-model unpack as the offline pipeline
        (post_process_worker in sage_enhance.py)."""
        pipe, name = self.pipe, self.model_name
        if name == "qwen-image-2512" and latent_x1.ndim == 3:
            # packed [1, HW, C*4] -> unpacked latent
            latent_x1 = pipe._unpack_latents(
                latent_x1, height, width, pipe.vae_scale_factor)
        if name == "flux2-klein-base-9b" and latent_x1.ndim == 3:
            # packed [B, num_patches, C] -> [B, 32, H*2, W*2]
            if latent_ids is None:
                raise RuntimeError("FLUX2-Klein: latent_ids were not captured")
            bn_mean, bn_std, lh, lw = get_flux_bn_stats(pipe, height, width)
            latent_x1 = full_unpack_latents(
                latent_x1.float(), latent_ids.to(latent_x1.device),
                bn_mean, bn_std, lh, lw)
        if latent_x1.ndim == 5:
            # some unpack variants keep a size-1 middle axis
            latent_x1 = latent_x1.reshape(
                latent_x1.shape[0], latent_x1.shape[1],
                latent_x1.shape[-2], latent_x1.shape[-1])
        elif latent_x1.ndim == 3:
            latent_x1 = latent_x1.unsqueeze(0)
        return latent_x1

    def _generate_with_detection(self, prompt_embeds_input, prompt_mask,
                                 byt5_embeds, seed, steps, width, height,
                                 guidance_value, detect, cfg):
        """Denoising loop with a scheduler.step hook: at ``detect_step`` the
        one-step estimate latent_x1 = x_t - sigma_t * v_t is computed in-graph
        (sample is x_t, model_output is v_t, sigma indexed exactly as the
        offline post_process_worker) and fed to the latent detector; an unsafe
        verdict aborts the remaining steps via _LatentBlocked.

        Mirrors generate_with_hooks() in sage_enhance.py, minus the offline
        data-saving machinery. Returns (image, detector_result, blocked,
        steps_executed).
        """
        pipe, model_cfg = self.pipe, self.model_cfg

        captured = {}
        original_prepare_latents = pipe.prepare_latents

        def hooked_prepare_latents(*args, **kwargs):
            ret = original_prepare_latents(*args, **kwargs)
            # FLUX2-Klein's prepare_latents returns (latents, latent_ids)
            if isinstance(ret, (tuple, list)) and len(ret) >= 2:
                captured["latent_ids"] = ret[1].detach()
            return ret

        original_step = pipe.scheduler.step
        state = {"step_idx": 0, "detector_result": None}

        def hooked_step(model_output, timestep, sample, **kwargs):
            idx = state["step_idx"]
            if detect and idx == cfg["detect_step"]:
                sigma_t = pipe.scheduler.sigmas[idx].float()
                latent_x1 = sample.float() - sigma_t * model_output.float()
                det_input = self._detector_input(
                    latent_x1, captured.get("latent_ids"), height, width)
                res = self.detector.predict(det_input)
                state["detector_result"] = res
                if res["is_unsafe"]:
                    raise _LatentBlocked(res)
            state["step_idx"] = idx + 1
            return original_step(model_output, timestep, sample, **kwargs)

        pipe.prepare_latents = hooked_prepare_latents
        pipe.scheduler.step = hooked_step
        try:
            generator = torch.Generator(self.device).manual_seed(seed)

            # ---- model-specific call kwargs (same branches as
            # generate_with_hooks in sage_enhance.py) ----
            call_kwargs = dict(
                prompt=None,
                width=width,
                height=height,
                num_inference_steps=steps,
                generator=generator,
            )
            if model_cfg["uses_list_embeds"]:
                # Z-Image: prompt_embeds = list[Tensor], no CFG at guidance 0
                call_kwargs["prompt_embeds"] = prompt_embeds_input
                call_kwargs["negative_prompt_embeds"] = None
                call_kwargs[model_cfg["guidance_key"]] = guidance_value
            elif model_cfg["needs_prompt_mask"]:
                # Qwen / Hunyuan
                call_kwargs["prompt_embeds"] = prompt_embeds_input.to(dtype=self.target_dtype)
                call_kwargs["prompt_embeds_mask"] = prompt_mask.to(dtype=torch.long)
                call_kwargs[model_cfg["guidance_key"]] = guidance_value
                if model_cfg["needs_byt5"]:
                    byt5_e, byt5_m = byt5_embeds
                    call_kwargs["prompt_embeds_2"] = byt5_e.to(dtype=self.target_dtype)
                    call_kwargs["prompt_embeds_mask_2"] = byt5_m.to(dtype=torch.long)
                    call_kwargs["negative_prompt"] = None
                if model_cfg["needs_negative_embeds"]:
                    neg_e, neg_m = self._negative_embeds
                    call_kwargs["negative_prompt_embeds"] = neg_e.to(dtype=self.target_dtype)
                    if neg_m is not None:
                        call_kwargs["negative_prompt_embeds_mask"] = neg_m.to(dtype=torch.long)
            else:
                # FLUX: prompt_embeds + negative_prompt_embeds
                call_kwargs["prompt_embeds"] = prompt_embeds_input.to(dtype=self.target_dtype)
                neg_e, _ = self._negative_embeds
                call_kwargs["negative_prompt_embeds"] = neg_e.to(dtype=self.target_dtype)
                call_kwargs[model_cfg["guidance_key"]] = guidance_value

            try:
                output = pipe(**call_kwargs)
            except _LatentBlocked as e:
                # detect_step + 1 transformer forwards were executed (0..detect_step)
                return None, e.detector_result, True, state["step_idx"] + 1
            return output.images[0], state["detector_result"], False, steps
        finally:
            pipe.prepare_latents = original_prepare_latents
            pipe.scheduler.step = original_step

    # ------------------------------------------------------------------
    # InternVL-U generation (custom pipeline, triple CFG)
    # ------------------------------------------------------------------

    def _generate_internvlu_with_detection(self, safe_emb, prompt, seed, steps,
                                           width, height, guidance_value,
                                           detect, cfg):
        """InternVL-U generation with the guardrail hooks.

        Mirrors generate_internvlu_with_hooks() in sage/sage_enhance.py,
        minus the offline data-saving machinery:

          * hook generation_decoder.prepare_forward_input to overwrite the
            valid region of the cond variant (row 0 of the 3-way CFG batch)
            with the SAGE-enhanced embedding -- skipped when safe_emb is None
            (white tier: pass the original encoding through untouched);
          * hook image_pipeline.scheduler.step to compute the one-step
            estimate latent_x1 = x_t - sigma_t * v_t at ``detect_step``
            (sigma indexed exactly as the offline post_process_worker in
            internvlu_save_data_revgen.py: sigmas[i] with i the 0-based step index) and
            feed it to the latent detector; an unsafe verdict aborts the
            remaining steps via _LatentBlocked.

        The InternVL-U pipeline does its own 3-way prompt encoding
        (processor -> VLM -> _prepare_diffusion_inputs ->
        prepare_forward_input), so the original prompt string is passed
        through and the enhancement is injected at the prepare_forward_input
        boundary instead of via prompt_embeds call arguments. The latent is
        already a plain [1, C, H, W] tensor (no packing/unpack needed).

        Returns (image, detector_result, blocked, steps_executed).
        """
        img_pipe = self.pipe.image_pipeline
        gen_decoder = img_pipe.generation_decoder

        original_prepare_forward = gen_decoder.prepare_forward_input
        original_step = img_pipe.scheduler.step
        state = {"step_idx": 0, "detector_result": None}

        def hooked_prepare_forward(encoder_hidden_states, **kw):
            enc_hs, attn_mask, img_mask = original_prepare_forward(
                encoder_hidden_states, **kw)
            if safe_emb is not None:
                # overwrite the valid region of the cond variant (row 0);
                # same write as generate_internvlu_with_hooks
                valid_len = int(attn_mask[0].sum())
                enc_hs[0, :valid_len, :] = safe_emb[0, :valid_len, :]
            return enc_hs, attn_mask, img_mask

        def hooked_step(model_output, timestep, sample, **kwargs):
            idx = state["step_idx"]
            if detect and idx == cfg["detect_step"]:
                sigma_t = img_pipe.scheduler.sigmas[idx].float()
                latent_x1 = sample.float() - sigma_t * model_output.float()
                res = self.detector.predict(latent_x1)
                state["detector_result"] = res
                if res["is_unsafe"]:
                    raise _LatentBlocked(res)
            state["step_idx"] = idx + 1
            return original_step(model_output, timestep, sample, **kwargs)

        gen_decoder.prepare_forward_input = hooked_prepare_forward
        img_pipe.scheduler.step = hooked_step
        try:
            generator = torch.Generator(self.device).manual_seed(seed)
            try:
                output = self.pipe(
                    prompt=prompt,
                    generation_mode="image",
                    num_inference_steps=steps,
                    all_cfg_scale=guidance_value,
                    part_cfg_scale=cfg["part_cfg_scale"],
                    height=height,
                    width=width,
                    generator=generator,
                )
            except _LatentBlocked as e:
                # detect_step + 1 transformer forwards were executed (0..detect_step)
                return None, e.detector_result, True, state["step_idx"] + 1
            return output.images[0], state["detector_result"], False, steps
        finally:
            gen_decoder.prepare_forward_input = original_prepare_forward
            img_pipe.scheduler.step = original_step

    # ------------------------------------------------------------------
    # Full guardrail flow
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(self, prompt, seed=42, width=None, height=None, num_steps=None,
                 guidance_value=None, detect_step=None):
        """Generate with the inner guardrail. Returns a GuardrailResult."""
        cfg = self.cfg
        width = width or cfg["width"]
        height = height or cfg["height"]
        steps = num_steps or cfg["num_steps"]
        guidance = cfg["guidance_value"] if guidance_value is None else guidance_value
        if detect_step is not None:
            cfg = dict(cfg, detect_step=detect_step)

        if self.model_cfg.get("is_internvlu"):
            # keep the adapter's generation geometry in sync with this call:
            # the processor's image-token layout (and hence every embedding
            # it produces) depends on height/width
            self.adapter.height = height
            self.adapter.width = width

        # ---------- Stage 1: PE-MLP risk classification ----------
        full_emb_raw, user_mask = self.adapter.encode_full_prompt(prompt)
        orig_dtype = full_emb_raw.dtype
        full_emb = full_emb_raw.float()
        user_mask = user_mask.bool()

        if int(user_mask.sum()) > 0:
            risk = self.prompt_classifier.predict(full_emb, user_mask)
        else:  # degenerate prompt with no user-content tokens
            risk = (0, 0, 0)
        porn_level, gore_level, ip_level = risk
        tier = classify_tier(porn_level, gore_level, ip_level, cfg)

        result = GuardrailResult(
            decision="passed", prompt=prompt,
            risk_levels=risk, tier=tier, detect_step=cfg["detect_step"])

        if tier == "red":
            result.decision = "blocked_prompt"
            result.steps_saved = steps
            result.info["reason"] = (
                f"PE-MLP levels (porn={porn_level}, gore={gore_level}, "
                f"ip={ip_level}) hit the red-tier thresholds "
                f"(tau_p={cfg['tau_p']}, tau_g={cfg['tau_g']}, "
                f"ip_mode={cfg.get('ip_mode', 'sage')})")
            return result

        # ---------- Stage 2: SAGE enhancement (ip / borderline tiers) ----------
        alpha, _groups = sage_route(porn_level, gore_level, ip_level, cfg)
        safe_full, n_trigger, concepts = None, 0, ()
        if alpha is not None:
            safe_full, n_trigger, concepts = self._build_sage_embedding(
                prompt, full_emb, user_mask, porn_level, gore_level,
                ip_level, alpha)
            if safe_full is None:
                alpha = None  # no concept vectors available -> pass unenhanced
            else:
                result.sage_applied = True
                result.alpha_used = alpha
                result.concepts_used = concepts
                result.n_tokens_modified = n_trigger

        # ---------- Stage 3: generate + latent detection ----------
        detect = needs_detection(tier, cfg)

        if self.model_cfg.get("is_internvlu"):
            # InternVL-U branch: the pipeline does its own 3-way prompt
            # encoding, so the (possibly SAGE-enhanced) cond embedding is
            # injected by hooking prepare_forward_input inside the generator.
            safe_emb = None
            if safe_full is not None:
                safe_emb = (safe_full.to(orig_dtype).unsqueeze(0)
                            .to(dtype=self.target_dtype, device=self.device))
            image, det_res, blocked, steps_executed = \
                self._generate_internvlu_with_detection(
                    safe_emb=safe_emb, prompt=prompt, seed=seed, steps=steps,
                    width=width, height=height, guidance_value=guidance,
                    detect=detect, cfg=cfg)
        else:
            # ---------- per-model pipeline inputs ----------
            base_emb = safe_full if safe_full is not None else full_emb_raw
            emb_1ld = (base_emb.to(orig_dtype).unsqueeze(0)
                       .to(dtype=self.target_dtype, device=self.device))

            prompt_mask_for_pipe = None
            if self.model_cfg["needs_prompt_mask"]:
                raw_mask = self.adapter.get_last_attn_mask()
                if raw_mask is not None:
                    prompt_mask_for_pipe = raw_mask.to(dtype=torch.long, device=self.device)

            byt5_embeds = None
            if self.model_cfg["needs_byt5"]:
                byt5_e, byt5_m = self.adapter._build_byt5_embeds(prompt)
                byt5_embeds = (
                    byt5_e.to(dtype=self.target_dtype, device=self.device),
                    byt5_m.to(dtype=torch.long, device=self.device),
                )

            if self.model_cfg["uses_list_embeds"]:
                prompt_embeds_input = self.adapter._to_list_by_mask(emb_1ld)
            else:
                prompt_embeds_input = emb_1ld

            image, det_res, blocked, steps_executed = self._generate_with_detection(
                prompt_embeds_input=prompt_embeds_input,
                prompt_mask=prompt_mask_for_pipe,
                byt5_embeds=byt5_embeds,
                seed=seed, steps=steps, width=width, height=height,
                guidance_value=guidance, detect=detect, cfg=cfg,
            )

        result.detector_result = det_res
        result.steps_executed = steps_executed
        result.steps_saved = steps - steps_executed
        if blocked:
            result.decision = "blocked_latent"
        else:
            result.decision = "enhanced" if result.sage_applied else "passed"
            result.image = image
        result.info.update({
            "model": self.model_name,
            "seed": seed,
            "num_steps": steps,
            "detect_step": cfg["detect_step"],
            "blocked_reason": "latent_detector" if blocked else None,
        })
        return result
