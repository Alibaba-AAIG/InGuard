"""Unit tests for the PE-MLP risk classifier (integration/pe_mlp_infer.py).

Builds a small randomly-initialized checkpoint in the exact format written
by pe_mlp/train_prompt.py, then checks loading (both torch.load paths),
forward output ranges, and the masked pooling helper.

Run (CPU torch only, no weights checkout needed):
    python -m unittest discover -s tests -v
"""

import os
import sys
import tempfile
import unittest

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_TESTS_DIR)
sys.path.insert(0, os.path.join(_REPO_ROOT, "integration"))

import torch  # noqa: E402

from pe_mlp_infer import (  # noqa: E402
    PEMLPRiskClassifier,
    PromptMultiTaskMLP,
    masked_mean_pool,
)

INPUT_DIM = 32
HIDDEN_DIM = 64
NUM_LAYERS = 2


def _save_ckpt(path, with_numpy_scalar=False):
    """Write a random-init checkpoint in train_prompt.py's payload format."""
    torch.manual_seed(0)
    model = PromptMultiTaskMLP(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM,
                               num_layers=NUM_LAYERS, dropout=0.1)
    payload = {
        "model_state_dict": model.state_dict(),
        "input_dim": INPUT_DIM,
        "hidden_dim": HIDDEN_DIM,
        "num_layers": NUM_LAYERS,
        "dropout": 0.1,
        "model_name": "unit-test",
        "global_step": 1234,
    }
    if with_numpy_scalar:
        # train_prompt.py saved some scalars as numpy types, which the
        # PyTorch 2.6+ weights-only unpickler rejects -> exercises the
        # fallback loader path in PEMLPRiskClassifier.__init__.
        import numpy as np
        payload["test_loss"] = np.float64(0.5)
    torch.save(payload, path)
    return model


class TestPEMLPLoadAndPredict(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.ckpt = os.path.join(self._tmp.name, "best.pth")
        self.ref_model = _save_ckpt(self.ckpt)
        self.clf = PEMLPRiskClassifier(self.ckpt, device="cpu")

    def tearDown(self):
        self._tmp.cleanup()

    def test_hyperparams_restored(self):
        self.assertEqual(self.clf.input_dim, INPUT_DIM)
        self.assertEqual(self.clf.model.proj.in_features, INPUT_DIM)
        self.assertEqual(self.clf.model.proj.out_features, HIDDEN_DIM)
        self.assertEqual(len(self.clf.model.blocks), NUM_LAYERS)
        self.assertEqual(self.clf.source_step, 1234)
        # eval mode: dropout is a no-op at inference
        self.assertFalse(self.clf.model.training)

    def test_predict_output_ranges(self):
        torch.manual_seed(1)
        seq = 16
        embeds = torch.randn(seq, INPUT_DIM)
        pred = self.clf.predict(embeds, torch.ones(seq, dtype=torch.bool))
        self.assertEqual(len(pred), 3)
        self.assertTrue(all(isinstance(v, int) for v in pred))
        p, g, i = pred
        self.assertTrue(0 <= p <= 5)
        self.assertTrue(0 <= g <= 5)
        self.assertTrue(0 <= i <= 7)  # ip head has 8 classes (incl. 6-7 benign)

    def test_predict_matches_reference_model(self):
        """predict() must reproduce the reference model's argmax exactly."""
        torch.manual_seed(2)
        embeds = torch.randn(8, INPUT_DIM)
        mask = torch.tensor([1, 1, 1, 1, 0, 0, 0, 0], dtype=torch.bool)
        p, g, i = self.clf.predict(embeds, mask)

        self.ref_model.eval()
        with torch.no_grad():
            pooled = masked_mean_pool(embeds, mask).unsqueeze(0)
            lp, lg, li = self.ref_model(pooled)
        self.assertEqual((p, g, i),
                         (int(lp.argmax()), int(lg.argmax()), int(li.argmax())))

    def test_predict_without_mask_pools_everything(self):
        torch.manual_seed(3)
        embeds = torch.randn(8, INPUT_DIM)
        no_mask = self.clf.predict(embeds, None)
        all_ones = self.clf.predict(embeds, torch.ones(8, dtype=torch.bool))
        self.assertEqual(no_mask, all_ones)

    def test_load_fallback_for_numpy_payload(self):
        """Payloads with numpy scalars must load via the weights_only=False
        fallback (trusted source: our own training artifacts)."""
        ckpt_np = os.path.join(self._tmp.name, "best_numpy.pth")
        _save_ckpt(ckpt_np, with_numpy_scalar=True)
        clf = PEMLPRiskClassifier(ckpt_np, device="cpu")
        self.assertEqual(clf.input_dim, INPUT_DIM)


class TestMaskedMeanPool(unittest.TestCase):

    def test_shapes(self):
        embeds = torch.randn(7, INPUT_DIM)
        self.assertEqual(masked_mean_pool(embeds).shape, (INPUT_DIM,))
        self.assertEqual(
            masked_mean_pool(embeds, torch.ones(7, dtype=torch.bool)).shape,
            (INPUT_DIM,))

    def test_mask_excludes_tokens(self):
        embeds = torch.randn(5, INPUT_DIM)
        mask = torch.tensor([1, 1, 0, 0, 0], dtype=torch.bool)
        pooled = masked_mean_pool(embeds, mask)
        expected = (embeds[0] + embeds[1]) / 2
        self.assertTrue(torch.allclose(pooled, expected, atol=1e-6))

    def test_all_zero_mask_does_not_divide_by_zero(self):
        embeds = torch.randn(4, INPUT_DIM)
        pooled = masked_mean_pool(embeds, torch.zeros(4, dtype=torch.bool))
        self.assertTrue(torch.isfinite(pooled).all())

    def test_casts_to_float(self):
        embeds = torch.randn(3, INPUT_DIM, dtype=torch.float16)
        pooled = masked_mean_pool(embeds)
        self.assertEqual(pooled.dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
