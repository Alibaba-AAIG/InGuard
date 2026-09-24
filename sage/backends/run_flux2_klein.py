"""
SAGE adapter for FLUX.2-klein.


Pipeline: diffusers.Flux2KleinPipeline
Text encoder: Qwen3 (with chat template)
Layer selection: hidden_states[(9, 18, 27)], 3 layers concatenated (same as Flux2KleinPipeline._get_qwen3_prompt_embeds)
Chat-template rule-of-thumb slicing: 3 in front, 9 at the tail
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from typing import List, Optional, Tuple

import torch

from base import ModelAdapter


class Flux2KleinAdapter(ModelAdapter):
    """FLUX.2-klein SAGE adapter."""

    HIDDEN_LAYERS: Tuple[int, ...] = (9, 18, 27)

    def __init__(self, pipe, device: str = "cuda", max_sequence_length: int = 512):
        self.pipe = pipe
        self.device = device
        self.max_sequence_length = max_sequence_length
        # cache the attention mask for the enhancement data-collection script to save
        self._last_attn_mask: Optional[torch.Tensor] = None
        # dynamically detect the chat-template prefix/suffix lengths (replacing the old hardcoded 3/9)
        self.chat_prefix_len, self.chat_suffix_len = self._detect_chat_template_lens()
        print(f"[Flux2KleinAdapter] chat_prefix_len={self.chat_prefix_len}, "
              f"chat_suffix_len={self.chat_suffix_len} (dynamically detected)")

    def _detect_chat_template_lens(self) -> Tuple[int, int]:
        """Run chat_template + tokenize with two different prompts and take the common prefix/suffix token counts."""
        text_a = self._build_chat_text("x")
        text_b = self._build_chat_text("y z")
        ids_a = self.pipe.tokenizer(text_a, add_special_tokens=False).input_ids
        ids_b = self.pipe.tokenizer(text_b, add_special_tokens=False).input_ids
        p = 0
        max_p = min(len(ids_a), len(ids_b))
        while p < max_p and ids_a[p] == ids_b[p]:
            p += 1
        s = 0
        while (s < len(ids_a) - p) and (s < len(ids_b) - p) and ids_a[-1 - s] == ids_b[-1 - s]:
            s += 1
        return p, s

    # ---------- internal utilities ----------
    def _build_chat_text(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        return self.pipe.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )

    def _encode_token_level(self, prompts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns [B, L, D_concat], [B, L], where D_concat = len(HIDDEN_LAYERS) * hidden_size."""
        chat_texts = [self._build_chat_text(p) for p in prompts]
        tok = self.pipe.tokenizer(
            chat_texts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_sequence_length,
        )
        input_ids = tok["input_ids"].to(self.device)
        attn = tok["attention_mask"].to(self.device)

        with torch.no_grad():
            out = self.pipe.text_encoder(
                input_ids=input_ids,
                attention_mask=attn,
                output_hidden_states=True,
                use_cache=False,
            )
        stack = torch.stack([out.hidden_states[k] for k in self.HIDDEN_LAYERS], dim=1)
        # [B, num_layers, L, D] → [B, L, num_layers * D]
        b, n_l, l, d = stack.shape
        embeds = stack.permute(0, 2, 1, 3).reshape(b, l, n_l * d).to(
            dtype=self.pipe.text_encoder.dtype, device=self.device
        )
        return embeds, attn.long()

    def _user_content_mask(self, attention_mask: torch.Tensor) -> torch.Tensor:
        if attention_mask.dim() == 1:
            attention_mask = attention_mask.unsqueeze(0)
            squeeze_back = True
        else:
            squeeze_back = False
        user_mask = torch.zeros_like(attention_mask, dtype=torch.bool)
        for b in range(attention_mask.shape[0]):
            valid_idx = torch.where(attention_mask[b].bool())[0]
            if len(valid_idx) <= self.chat_prefix_len + self.chat_suffix_len:
                continue
            content_idx = valid_idx[self.chat_prefix_len : -self.chat_suffix_len]
            user_mask[b, content_idx] = True
        return user_mask[0] if squeeze_back else user_mask

    # ---------- interface implementation ----------
    def encode_full_prompt(self, prompt: str) -> Tuple[torch.Tensor, torch.Tensor]:
        embeds, attn = self._encode_token_level([prompt])
        self._last_attn_mask = attn.long()  # cached for get_last_attn_mask()
        return embeds[0], self._user_content_mask(attn)[0]

    def get_last_attn_mask(self) -> Optional[torch.Tensor]:
        return self._last_attn_mask

    def encode_pooled_concept(self, concept_text: str) -> Optional[torch.Tensor]:
        embeds, attn = self._encode_token_level([concept_text])
        user_mask = self._user_content_mask(attn)[0]
        if user_mask.sum().item() == 0:
            return None
        return embeds[0][user_mask].float().mean(dim=0)

    def encode_pooled_masked_per_token(self, prompt: str) -> torch.Tensor:
        chat_text = self._build_chat_text(prompt)
        tok = self.pipe.tokenizer(
            chat_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_sequence_length,
        )
        input_ids = tok["input_ids"].to(self.device)
        attn = tok["attention_mask"].to(self.device)
        user_pos = self._user_content_mask(attn)[0]
        positions = torch.where(user_pos)[0].tolist()
        n = len(positions)
        d_concat = len(self.HIDDEN_LAYERS) * self.pipe.text_encoder.config.hidden_size
        if n == 0:
            return torch.zeros(0, d_concat, device=self.device)

        masked_ids = input_ids.repeat(n, 1)
        masked_attn = attn.repeat(n, 1)
        for i, pos in enumerate(positions):
            masked_ids[i, pos] = 0

        with torch.no_grad():
            out = self.pipe.text_encoder(
                input_ids=masked_ids,
                attention_mask=masked_attn,
                output_hidden_states=True,
                use_cache=False,
            )
        stack = torch.stack([out.hidden_states[k] for k in self.HIDDEN_LAYERS], dim=1)
        b, n_l, l, d = stack.shape
        hidden = stack.permute(0, 2, 1, 3).reshape(b, l, n_l * d).to(
            dtype=self.pipe.text_encoder.dtype, device=self.device
        )

        pooled = []
        for i in range(n):
            row_user_mask = self._user_content_mask(masked_attn[i:i + 1])[0]
            valid = hidden[i][row_user_mask]
            if valid.shape[0] == 0:
                pooled.append(torch.zeros(hidden.shape[-1], device=self.device))
            else:
                pooled.append(valid.mean(dim=0))
        return torch.stack(pooled, dim=0).float()

    def _get_negative_embeds(self, negative_prompt: str = "") -> torch.Tensor:
        embeds, _ = self._encode_token_level([negative_prompt])
        return embeds.float()

    # ---------- inference ----------
