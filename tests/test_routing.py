"""Unit tests for the four-tier guardrail routing (integration/routing.py).

Covers Algorithm 1 of the paper:
  - an exhaustive truth table over all risk-level combinations for every
    model config (5 models x 6 porn x 6 gore x 8 ip = 1440 cases) against
    an independent reference implementation of the tier semantics
  - the enhancement gate (white-tier prompts are never SAGE-enhanced)
  - SAGE category priority (porn > gore > ip) and alpha/concept selection
  - latent-detection routing per tier

Run (no torch needed, pure logic):
    python -m unittest discover -s tests -v
"""

import json
import os
import sys
import unittest

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_TESTS_DIR)
sys.path.insert(0, os.path.join(_REPO_ROOT, "integration"))

from routing import (  # noqa: E402
    GuardrailResult,
    classify_tier,
    needs_detection,
    sage_route,
)

MODEL_NAMES = ["z-image-turbo", "qwen-image-2512", "hunyuan-image-2_1",
               "flux2-klein-base-9b", "internvl-u"]

# porn/gore levels are 0-5; ip levels are 0-7 (0 none, 1-5 controlled IPs,
# 6-7 benign variants)
ALL_LEVELS = [(p, g, i) for p in range(6) for g in range(6) for i in range(8)]


def load_cfg(model):
    with open(os.path.join(_REPO_ROOT, "configs", f"{model}.json")) as f:
        return json.load(f)


def reference_tier(p, g, i, cfg):
    """Independent re-derivation of Algorithm 1 tier semantics.

    Written from the paper's description, deliberately not a copy of
    classify_tier(): a prompt is red when either risk category reaches its
    block threshold (or IP=direct mode fires); otherwise a controlled-IP hit
    routes to the ip tier; remaining prompts above an enhancement threshold
    are borderline; the rest are white.
    """
    ip_lo, ip_hi = cfg["borderline_ip"]
    is_ip = ip_lo <= i <= ip_hi

    red = p >= cfg["tau_p"] or g >= cfg["tau_g"]
    if cfg.get("ip_mode", "sage") == "direct" and is_ip:
        red = True
    if red:
        return "red"
    if is_ip:
        return "ip"
    if p >= cfg["borderline_porn"] or g >= cfg["borderline_gore"]:
        return "borderline"
    return "white"


class TestClassifyTierTruthTable(unittest.TestCase):
    """Exhaustive comparison of classify_tier() against reference_tier()."""

    def test_truth_table_all_models(self):
        n = 0
        for model in MODEL_NAMES:
            cfg = load_cfg(model)
            for p, g, i in ALL_LEVELS:
                got = classify_tier(p, g, i, cfg)
                want = reference_tier(p, g, i, cfg)
                self.assertEqual(
                    got, want,
                    f"{model}: (porn={p}, gore={g}, ip={i}) -> "
                    f"classify_tier={got}, reference={want}")
                n += 1
        self.assertEqual(n, len(MODEL_NAMES) * 6 * 6 * 8)

    def test_ip_direct_mode_blocks(self):
        cfg = load_cfg("z-image-turbo")
        cfg = dict(cfg, ip_mode="direct")
        self.assertEqual(classify_tier(0, 0, 3, cfg), "red")

    def test_ip_benign_variants_are_white(self):
        # ip 6-7 are benign variants: outside borderline_ip -> never "ip" tier
        cfg = load_cfg("z-image-turbo")
        for i in (6, 7):
            self.assertEqual(classify_tier(0, 0, i, cfg), "white")

    def test_benign_all_zero_is_white(self):
        for model in MODEL_NAMES:
            self.assertEqual(classify_tier(0, 0, 0, load_cfg(model)), "white")

    def test_worst_porn_gore_is_red(self):
        for model in MODEL_NAMES:
            cfg = load_cfg(model)
            self.assertEqual(classify_tier(5, 0, 0, cfg), "red")
            self.assertEqual(classify_tier(0, 5, 0, cfg), "red")


class TestSageRoute(unittest.TestCase):
    """SAGE enhancement gate, category priority, and alpha selection."""

    def test_white_tier_never_enhanced(self):
        # Regression: below every enhancement threshold -> (None, ()).
        # (Historically the gate was missing and white prompts were
        # erroneously enhanced.)
        for model in MODEL_NAMES:
            cfg = load_cfg(model)
            for p in range(cfg["borderline_porn"]):
                for g in range(cfg["borderline_gore"]):
                    for i in (0, 6, 7):  # no controlled-IP hit
                        alpha, groups = sage_route(p, g, i, cfg)
                        self.assertIsNone(alpha)
                        self.assertEqual(groups, ())
                        self.assertEqual(classify_tier(p, g, i, cfg), "white",
                                         f"{model}: expected white tier")

    def test_borderline_alpha_and_groups(self):
        cfg = load_cfg("z-image-turbo")
        # porn-only borderline -> alpha_p, porn concepts
        self.assertEqual(sage_route(3, 0, 0, cfg), (cfg["alpha_p"], ("porn",)))
        # gore-only borderline -> alpha_g, gore concepts
        self.assertEqual(sage_route(0, 2, 0, cfg), (cfg["alpha_g"], ("gore",)))
        # porn takes priority over gore (paper: highest-priority category)
        self.assertEqual(sage_route(3, 2, 0, cfg), (cfg["alpha_p"], ("porn", "gore")))

    def test_ip_tier(self):
        for model in MODEL_NAMES:
            cfg = load_cfg(model)
            self.assertEqual(sage_route(0, 0, 3, cfg), (cfg["alpha_i"], ("ip",)))
            # ip + porn/gore risk: porn/gore concepts win (ip tier is for
            # ip-only risk)
            alpha, groups = sage_route(3, 0, 3, cfg)
            self.assertEqual(alpha, cfg["alpha_p"])
            self.assertEqual(groups, ("porn",))

    def test_red_tier_is_rejected_by_gate_caller(self):
        # classify_tier sends red prompts straight to block; sage_route is
        # only called for non-red tiers. Still, its gate stays consistent:
        # tau >= borderline for every model, so a red prompt that reaches
        # sage_route would be enhanced (the pipeline never does).
        for model in MODEL_NAMES:
            cfg = load_cfg(model)
            self.assertGreaterEqual(cfg["tau_p"], cfg["borderline_porn"])
            self.assertGreaterEqual(cfg["tau_g"], cfg["borderline_gore"])


class TestNeedsDetection(unittest.TestCase):
    """Latent-detection routing per tier ("w/ fallback" deployment)."""

    def test_per_tier(self):
        cfg = load_cfg("z-image-turbo")
        self.assertFalse(needs_detection("red", cfg))
        self.assertTrue(needs_detection("ip", cfg))
        self.assertTrue(needs_detection("borderline", cfg))
        # white tier: governed by the fallback flag
        self.assertTrue(needs_detection("white", cfg))
        cfg_nofb = dict(cfg, fallback=False)
        self.assertFalse(needs_detection("white", cfg_nofb))


class TestGuardrailResult(unittest.TestCase):

    def test_blocked_property(self):
        for decision, blocked in (("blocked_prompt", True),
                                  ("blocked_latent", True),
                                  ("enhanced", False),
                                  ("passed", False)):
            r = GuardrailResult(decision=decision, prompt="x")
            self.assertEqual(r.blocked, blocked)
            self.assertEqual(r.steps_saved, 0)  # default


if __name__ == "__main__":
    unittest.main()
