"""Unit tests for the InternVL-U branch of GuardrailPipeline
(integration/guardrail_pipeline.py).

Everything runs offline: no model weights, no GPU, and no import of the
internvlu package. The custom pipeline / detector / adapter are replaced by
fakes that mimic the real call signatures (see sage/backends/internvlu/ and
generate_internvlu_with_hooks in sage/sage_enhance.py):

  * the fake InternVLUPipeline.__call__ replays the real internal order
    (prepare_forward_input -> per-step scheduler.step), so the two hooks
    installed by _generate_internvlu_with_detection fire exactly as they
    would in production;
  * the latent fed to the detector is checked numerically against
    latent_x1 = x_t - sigma_t * v_t with the 0-based sigma indexing of the
    offline post_process_worker (internvlu_save_data_revgen.py);
  * generate() is driven end-to-end with a white-tier and an ip-tier prompt
    to prove the routing/SAGE stage-1/2 logic is shared with the diffusers
    models and only stage 3 branches.

Run (no torch CUDA needed):
    python -m unittest discover -s tests -v
"""

import json
import os
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_TESTS_DIR)
for _sub in ("integration", "sage", "sage/backends", "latent_detector/export"):
    _p = os.path.join(_REPO_ROOT, _sub)
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

import guardrail_pipeline as gp  # noqa: E402
from guardrail_pipeline import GuardrailPipeline  # noqa: E402
from routing import classify_tier, sage_route  # noqa: E402

STEPS = 20
DETECT_STEP = 8
LATENT_H, LATENT_W = 32, 32
D = 64          # embedding width (real model: 4096; shrunk for speed)
SEQ = 768       # padded sequence length of the InternVL-U diffusion decoder
N_USER = 5      # user tokens in the fake prompt encoding


def _load_cfg():
    with open(os.path.join(_REPO_ROOT, "configs", "internvl-u.json")) as f:
        return json.load(f)


class FakeScheduler:
    def __init__(self, steps):
        # DPMSolver-style sigmas: len = steps + 1, decreasing
        self.sigmas = torch.linspace(1.0, 0.0, steps + 1)
        self.step_calls = []

    def step(self, model_output, timestep, sample, **kwargs):
        self.step_calls.append((model_output, timestep, sample))
        return sample  # identity update: x_{t+1} = x_t


class FakeGenerationDecoder:
    def __init__(self):
        self.prepare_calls = []

    def prepare_forward_input(self, encoder_hidden_states, **kw):
        enc_hs = encoder_hidden_states.clone()
        attn_mask = torch.zeros(3, SEQ, dtype=torch.bool)
        attn_mask[:, :SEQ - 10] = True   # valid_len = SEQ - 10
        img_mask = None
        # record the RETURNED tensor: the hook modifies it in place, so
        # prepare_calls[-1] reflects the post-hook state
        self.prepare_calls.append(enc_hs)
        return enc_hs, attn_mask, img_mask


class FakeImagePipeline:
    def __init__(self, steps):
        self.generation_decoder = FakeGenerationDecoder()
        self.scheduler = FakeScheduler(steps)


class FakeInternVLUPipeline:
    """Replays the real InternVLUPipeline.__call__ internal order for
    generation_mode="image": encode (cond/part/uncond) -> prepare_forward_input
    (hooked) -> denoising loop with scheduler.step (hooked)."""

    def __init__(self, steps=STEPS):
        self.image_pipeline = FakeImagePipeline(steps)
        self.call_kwargs = None

    def __call__(self, prompt, generation_mode="text", **kwargs):
        assert generation_mode == "image"
        self.call_kwargs = dict(prompt=prompt, generation_mode=generation_mode,
                                **kwargs)
        img_pipe = self.image_pipeline
        # processor -> VLM -> _prepare_diffusion_inputs -> (hooked)
        raw_enc = torch.randn(3, SEQ, D)
        enc_hs, attn_mask, img_mask = (
            img_pipe.generation_decoder.prepare_forward_input(raw_enc))
        # denoising loop: latents stay all-zero (identity scheduler), the
        # velocity at step i is (i+1) * ones so latent_x1 is predictable
        latents = torch.zeros(1, 16, LATENT_H, LATENT_W)
        for i in range(kwargs["num_inference_steps"]):
            velocity = torch.full_like(latents, float(i + 1))
            latents = img_pipe.scheduler.step(velocity, None, latents)
        image = SimpleNamespace(size=(1024, 1024))
        return SimpleNamespace(images=[image])


class FakeDetector:
    def __init__(self, unsafe=False):
        self.unsafe = unsafe
        self.seen = []

    def predict(self, latent):
        self.seen.append(latent.clone())
        return {
            "porn_prob": 0.1, "gore_prob": 0.1,
            "ip_probs": [0.0] * 6,
            "porn_pred": 0, "gore_pred": 0, "ip_pred": 5,
            "ip_name": "",
            "is_unsafe": self.unsafe,
        }


def _make_guardrail(detector):
    """Build a GuardrailPipeline shell around the fake InternVL-U pipeline,
    bypassing __init__ (no weights / model loading)."""
    g = object.__new__(GuardrailPipeline)
    g.pipe = FakeInternVLUPipeline()
    g.detector = detector
    g.device = "cpu"
    g.model_name = "internvl-u"
    g.model_cfg = dict(gp.MODEL_REGISTRY["internvl-u"])
    g.cfg = _load_cfg()
    g.target_dtype = torch.float32
    return g


class TestGenerateInternvluWithDetection(unittest.TestCase):
    """_generate_internvlu_with_detection: hook installation, detector input,
    early abort, and hook restoration."""

    def setUp(self):
        self.detector = FakeDetector(unsafe=False)
        self.g = _make_guardrail(self.detector)
        self.cfg = dict(self.g.cfg, detect_step=DETECT_STEP)

    def _run(self, safe_emb=None):
        return self.g._generate_internvlu_with_detection(
            safe_emb=safe_emb, prompt="a castle on a hill", seed=42,
            steps=STEPS, width=1024, height=1024, guidance_value=4.5,
            detect=True, cfg=self.cfg)

    def test_safe_generation_runs_all_steps(self):
        image, det_res, blocked, steps_executed = self._run()
        self.assertFalse(blocked)
        self.assertIsNotNone(image)
        self.assertIsNotNone(det_res)
        self.assertEqual(steps_executed, STEPS)
        # exactly one detector call, at detect_step
        self.assertEqual(len(self.detector.seen), 1)
        # 20 scheduler.step calls happened (the full denoising loop)
        self.assertEqual(len(self.g.pipe.image_pipeline.scheduler.step_calls), STEPS)

    def test_detector_input_matches_offline_formula(self):
        """latent_x1 must equal x_t - sigmas[detect_step] * v_t with the same
        0-based sigma indexing as the offline post_process_worker."""
        self._run()
        x_t = torch.zeros(1, 16, LATENT_H, LATENT_W)          # identity scheduler
        v_t = torch.full((1, 16, LATENT_H, LATENT_W), float(DETECT_STEP + 1))
        sigma_t = float(torch.linspace(1.0, 0.0, STEPS + 1)[DETECT_STEP])
        expected = x_t - sigma_t * v_t
        torch.testing.assert_close(self.detector.seen[0], expected)
        # 4D plain latent, already unpacked (no packing/unpack for InternVL-U)
        self.assertEqual(self.detector.seen[0].shape, (1, 16, LATENT_H, LATENT_W))

    def test_unsafe_generation_aborts_at_detect_step(self):
        self.detector.unsafe = True
        image, det_res, blocked, steps_executed = self._run()
        self.assertTrue(blocked)
        self.assertIsNone(image)
        self.assertTrue(det_res["is_unsafe"])
        # detect_step + 1 transformer forwards executed (steps 0..detect_step)
        self.assertEqual(steps_executed, DETECT_STEP + 1)
        # the loop aborted: transformer forwards 0..detect_step ran, but the
        # scheduler.step for step detect_step never executed (the hook raises
        # before calling the original), so only detect_step step-calls ran
        self.assertEqual(
            len(self.g.pipe.image_pipeline.scheduler.step_calls), DETECT_STEP)

    def test_prepare_forward_injects_safe_emb_into_cond_row(self):
        safe_emb = torch.randn(1, SEQ, D)
        self._run(safe_emb=safe_emb)
        dec = self.g.pipe.image_pipeline.generation_decoder
        self.assertEqual(len(dec.prepare_calls), 1)
        # after the hook, row 0 (cond) valid region == safe_emb; rows 1/2 untouched
        out = dec.prepare_calls[0]
        valid_len = SEQ - 10
        torch.testing.assert_close(out[0, :valid_len], safe_emb[0, :valid_len])
        self.assertFalse(torch.allclose(out[0], safe_emb[0]))  # padding not written

    def test_prepare_forward_passthrough_when_safe_emb_none(self):
        # when safe_emb is None the hook must not write anything: run twice
        # with the same raw input and the recorded outputs must be identical
        # to what the untouched original produces (compare against a direct
        # call to the original method)
        self._run(safe_emb=None)
        dec = self.g.pipe.image_pipeline.generation_decoder
        self.assertEqual(len(dec.prepare_calls), 1)
        self.assertEqual(dec.prepare_calls[0].shape, (3, SEQ, D))

        raw = torch.randn(3, SEQ, D)
        expected_hs, _, _ = FakeGenerationDecoder.prepare_forward_input(
            dec, raw)
        got_hs, _, _ = dec.prepare_forward_input(raw)
        torch.testing.assert_close(got_hs, expected_hs)

    def test_hooks_restored_after_call(self):
        img_pipe = self.g.pipe.image_pipeline
        dec = img_pipe.generation_decoder
        self._run()
        # attribute lookups mint a fresh bound-method object each time, so
        # compare the underlying functions: the restored callable must be the
        # original method, not a leftover hook closure
        self.assertIs(dec.prepare_forward_input.__func__,
                      FakeGenerationDecoder.prepare_forward_input)
        self.assertIs(img_pipe.scheduler.step.__func__, FakeScheduler.step)
        # restored even when the detector aborts the run
        self.detector.unsafe = True
        self._run()
        self.assertIs(dec.prepare_forward_input.__func__,
                      FakeGenerationDecoder.prepare_forward_input)
        self.assertIs(img_pipe.scheduler.step.__func__, FakeScheduler.step)
        # and the restored originals are fully functional
        raw = torch.randn(3, SEQ, D)
        dec.prepare_forward_input(raw)
        img_pipe.scheduler.step(torch.zeros(1), None, torch.zeros(1))

    def test_pipeline_call_kwargs(self):
        self._run()
        kw = self.g.pipe.call_kwargs
        self.assertEqual(kw["prompt"], "a castle on a hill")
        self.assertEqual(kw["generation_mode"], "image")
        self.assertEqual(kw["num_inference_steps"], STEPS)
        self.assertEqual(kw["all_cfg_scale"], 4.5)
        self.assertEqual(kw["part_cfg_scale"], self.cfg["part_cfg_scale"])
        self.assertEqual(kw["height"], 1024)
        self.assertEqual(kw["width"], 1024)


class _FakeAdapter:
    """Mimics InternVLUAdapter.encode_full_prompt /
    encode_pooled_masked_per_token (shapes shrunk to D=64)."""

    def __init__(self):
        self.height = 1024
        self.width = 1024
        self._last_attn_mask = torch.ones(1, SEQ, dtype=torch.long)

    def encode_full_prompt(self, prompt):
        full = torch.randn(SEQ, D)
        user_mask = torch.zeros(SEQ, dtype=torch.bool)
        user_mask[3:3 + N_USER] = True
        return full, user_mask

    def encode_pooled_masked_per_token(self, prompt):
        return torch.randn(N_USER, D)

    def get_last_attn_mask(self):
        return self._last_attn_mask


class _FakeClassifier:
    def __init__(self, risk):
        self.risk = risk

    def predict(self, full_emb, user_mask):
        return self.risk


class TestGenerateRoutingInternvlu(unittest.TestCase):
    """generate(): the internvl-u branch of stage 3, driven end-to-end with
    fakes (stage 1/2 use the real routing/SAGE code paths)."""

    def setUp(self):
        self.cfg = _load_cfg()
        self.calls = []

    def _make(self, risk):
        g = object.__new__(GuardrailPipeline)
        g.pipe = None  # unused: stage 3 is stubbed
        g.device = "cpu"
        g.model_name = "internvl-u"
        g.model_cfg = dict(gp.MODEL_REGISTRY["internvl-u"])
        g.cfg = self.cfg
        g.target_dtype = torch.float32
        g.adapter = _FakeAdapter()
        g.prompt_classifier = _FakeClassifier(risk)
        g._concept_groups = {
            "porn": ["a"], "gore": ["b"],
            "ip": {i: [f"c{i}"] for i in range(1, 6)},
        }
        g.concept_vecs = {}

        def stub(safe_emb, prompt, seed, steps, width, height,
                 guidance_value, detect, cfg):
            self.calls.append(dict(safe_emb=safe_emb, detect=detect))
            return (SimpleNamespace(size=(1, 1)), {"is_unsafe": False},
                    False, cfg["num_steps"])

        g._generate_internvlu_with_detection = stub
        return g

    def test_white_tier_calls_internvlu_branch_without_sage(self):
        g = self._make(risk=(0, 0, 0))
        tier = classify_tier(0, 0, 0, self.cfg)
        self.assertEqual(tier, "white")
        result = g.generate("a scenic photo of mountains", seed=42)
        self.assertEqual(len(self.calls), 1)
        self.assertIsNone(self.calls[0]["safe_emb"])
        self.assertTrue(self.calls[0]["detect"])   # fallback: white is detected
        self.assertEqual(result.decision, "passed")
        self.assertFalse(result.sage_applied)
        self.assertIsNone(result.alpha_used)
        self.assertIsNotNone(result.image)
        # adapter geometry synced with this generate() call
        self.assertEqual(g.adapter.height, self.cfg["height"])
        self.assertEqual(g.adapter.width, self.cfg["width"])
        self.assertEqual(result.info["model"], "internvl-u")

    def test_ip_tier_passes_enhanced_embedding(self):
        g = self._make(risk=(0, 0, 1))
        tier = classify_tier(0, 0, 1, self.cfg)
        self.assertEqual(tier, "ip")
        alpha, _ = sage_route(0, 0, 1, self.cfg)
        self.assertIsNotNone(alpha)
        # patch the two sage_enhance helpers in the guardrail_pipeline
        # namespace: small deterministic projection subspaces
        vecs = [torch.zeros(D), torch.zeros(D)]
        p_c = torch.zeros(D, D)
        i_minus_pc = torch.eye(D)
        with mock.patch.object(gp, "select_concept_vectors",
                               return_value=(vecs, ("ip",))), \
             mock.patch.object(gp, "build_P_C",
                               return_value=(p_c, i_minus_pc)):
            result = g.generate("mario riding a kart", seed=42)
        self.assertEqual(len(self.calls), 1)
        safe_emb = self.calls[0]["safe_emb"]
        self.assertIsNotNone(safe_emb)
        self.assertEqual(safe_emb.shape, (1, SEQ, D))
        self.assertTrue(result.sage_applied)
        self.assertEqual(result.alpha_used, alpha)
        self.assertEqual(result.decision, "enhanced")
        self.assertIn("ip", result.concepts_used)

    def test_red_tier_never_starts_generation(self):
        g = self._make(risk=(5, 5, 3))
        self.assertEqual(classify_tier(5, 5, 3, self.cfg), "red")
        result = g.generate("obviously red prompt", seed=42)
        self.assertEqual(result.decision, "blocked_prompt")
        self.assertEqual(self.calls, [])          # stage 3 stub never invoked
        self.assertEqual(result.steps_saved, self.cfg["num_steps"])
        self.assertIsNone(result.image)


if __name__ == "__main__":
    unittest.main()
