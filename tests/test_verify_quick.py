"""Unit tests for the quick verification stage of scripts/verify_pipeline.py.

stage_quick() checks weight-file presence and the manifest (no torch,
seconds). These tests build a fake weights/ tree in a temp dir — both a
healthy one that must pass every check, and broken variants (missing file,
wrong manifest fields) that must produce failures.

Run (no torch needed):
    python -m unittest discover -s tests -v
"""

import json
import os
import sys
import tempfile
import unittest

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_TESTS_DIR)
sys.path.insert(0, os.path.join(_REPO_ROOT, "scripts"))

from verify_pipeline import (  # noqa: E402
    LATENT_EXPECTED_CHANS,
    stage_quick,
)

MODEL = "z-image-turbo"


def build_fake_weights(root, model=MODEL, mutate=None):
    """Create a minimal weights/ tree that mirrors a healthy export.

    Layout expected by stage_quick():
        <root>/pe_mlp/<model>/model.pth
        <root>/latent_detector/<model>/model.pth
        <root>/latent_detector/<model>/config.json
        <root>/manifest.json
    """
    pe_dir = os.path.join(root, "pe_mlp", model)
    lat_dir = os.path.join(root, "latent_detector", model)
    os.makedirs(pe_dir)
    os.makedirs(lat_dir)
    # stage_quick only checks presence; empty placeholder files suffice
    open(os.path.join(pe_dir, "model.pth"), "w").close()
    open(os.path.join(lat_dir, "model.pth"), "w").close()
    with open(os.path.join(lat_dir, "config.json"), "w") as f:
        json.dump({"latent_in_chans": LATENT_EXPECTED_CHANS[model]}, f)

    manifest = {
        "pe_mlp": {
            model: {
                "status": "ok",
                "checkpoint": "model.pth",
            }
        },
        "latent_detector": {
            model: {
                "status": "ok",
                "checkpoint": "model.pth",
                "latent_in_chans": LATENT_EXPECTED_CHANS[model],
            }
        },
    }
    if mutate:
        mutate(manifest, root)
    with open(os.path.join(root, "manifest.json"), "w") as f:
        json.dump(manifest, f)
    return root


class TestStageQuickPass(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.weights_root = build_fake_weights(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_healthy_tree_has_no_failures(self):
        rep = stage_quick(MODEL, self.weights_root)
        self.assertEqual(rep.failures, [])

    def test_chans_falls_back_to_config_json(self):
        """latent_in_chans may be absent from the manifest: stage_quick must
        fall back to the exported config.json next to the checkpoint."""
        def drop_chans(manifest, root):
            del manifest["latent_detector"][MODEL]["latent_in_chans"]

        root2 = build_fake_weights(os.path.join(self._tmp.name, "w2"),
                                   mutate=drop_chans)
        rep = stage_quick(MODEL, root2)
        self.assertEqual(rep.failures, [])


class TestStageQuickFailures(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self._tmp.cleanup()

    def _weights(self, mutate=None):
        return build_fake_weights(self._tmp.name, mutate=mutate)

    def test_missing_pe_mlp_ckpt(self):
        root = self._weights()
        os.remove(os.path.join(root, "pe_mlp", MODEL, "model.pth"))
        rep = stage_quick(MODEL, root)
        self.assertIn("pe_mlp ckpt exists", rep.failures)

    def test_missing_manifest(self):
        root = self._weights()
        os.remove(os.path.join(root, "manifest.json"))
        rep = stage_quick(MODEL, root)
        self.assertIn("manifest.json exists", rep.failures)

    def test_pe_mlp_status_not_ok(self):
        def not_ok(manifest, root):
            manifest["pe_mlp"][MODEL]["status"] = "missing_source"
        rep = stage_quick(MODEL, self._weights(mutate=not_ok))
        self.assertIn("manifest pe_mlp status=ok", rep.failures)

    def test_wrong_latent_chans(self):
        def wrong_chans(manifest, root):
            # wrong value in the manifest takes precedence over config.json
            manifest["latent_detector"][MODEL]["latent_in_chans"] = 999
        rep = stage_quick(MODEL, self._weights(mutate=wrong_chans))
        self.assertTrue(any("latent_in_chans" in f for f in rep.failures))


class TestExpectedTables(unittest.TestCase):
    """The channel table covers all five shipped model configs."""

    def test_tables_cover_all_configs(self):
        models = {m[:-5] for m in os.listdir(os.path.join(_REPO_ROOT, "configs"))
                  if m.endswith(".json")}
        self.assertEqual(set(LATENT_EXPECTED_CHANS), models)


if __name__ == "__main__":
    unittest.main()
