"""Model adapter interface for SAGE.

SAGE operates on the text-encoder embeddings of the protected T2I model, so
every backbone needs a thin adapter that exposes its encoder and sampling
loop.  The adapter absorbs all model-specific details (chat templates,
attention masks, CFG variants, latent packing, ...); SAGE itself is model
agnostic.  Each model only has to implement the three abstract methods below.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch


class ModelAdapter(ABC):
    """Adapter interface between a T2I pipeline and SAGE.

    Implementations live next to this file (one per model):
    ``run_zimage.py``, ``run_qwen_image.py``, ``run_hunyuan_image.py``,
    ``run_flux2_klein.py`` and ``run_internvlu.py``.
    """

    device: str = "cuda"

    # ---------- encoding ----------
    @abstractmethod
    def encode_full_prompt(self, prompt: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode a full prompt (including chat template / special tokens / padding).

        Returns:
            full_emb : [L, D] full token-level embeddings.
            user_mask: [L] bool, True only for user-content tokens (chat
                template / EOS / PAD positions are False).
        """

    @abstractmethod
    def encode_pooled_concept(self, concept_text: str) -> torch.Tensor:
        """Encode a single concept text into one pooled D-dim vector.

        Returns: [D]
        """

    @abstractmethod
    def encode_pooled_masked_per_token(self, prompt: str) -> torch.Tensor:
        """Leave-one-out token-mask encoding of the prompt.

        Each masked sequence is pooled into one D-dim vector, giving N vectors
        where N == ``encode_full_prompt(...)``'s ``user_mask.sum()``.

        Returns: [N, D]
        """

    # ---------- optional ----------
    def get_last_attn_mask(self) -> Optional[torch.Tensor]:
        """Return the attention mask cached by the last ``encode_full_prompt``.

        Some pipelines (e.g. QwenImagePipeline) require the attention mask to
        be passed alongside ``prompt_embeds``.  Models that do not need it
        simply keep the default ``None``.
        """
        return None
