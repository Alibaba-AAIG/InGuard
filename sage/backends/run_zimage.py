"""
SAGE adapter for Z-Image-Turbo.


Pipeline: diffusers.ZImagePipeline
Text encoder: Qwen3
Key source facts (verified against diffusers 0.38.0 ZImagePipeline._encode_prompt):
  1. apply_chat_template(..., enable_thinking=True)
     -> the chat-template suffix contains only <|im_end|> + newline + <|im_start|>assistant + newline (~5 tokens),
        not the extra tokens produced by enable_thinking=False.
  2. take text_encoder(..., output_hidden_states=True).hidden_states[-2]
  3. the official _encode_prompt returns list[Tensor], each entry sliced to its valid length by attention_mask.
     ZImagePipeline.__call__ expects prompt_embeds to also be list[Tensor], so after SAGE
     finishes the projection it must slice back to a list by attention_mask before passing it to the pipeline.
  4. the true token counts of the chat-template prefix/suffix are obtained automatically by
     tokenizing at construction time (not hardcoded).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from typing import List, Optional, Tuple

import torch

from base import ModelAdapter


class ZImageAdapter(ModelAdapter):
    """Z-Image-Turbo SAGE adapter."""

    def __init__(self, pipe, device: str = "cuda", max_sequence_length: int = 512):
        self.pipe = pipe
        self.device = device
        self.max_sequence_length = max_sequence_length
        # cache the attention_mask of the last encode_full_prompt; at generation time it slices
        # the projected fixed-length embedding back into list[Tensor] to feed ZImagePipeline.
        self._last_attn_mask: Optional[torch.Tensor] = None
        # dynamically detect the chat-template prefix/suffix lengths (replacing the old hardcoded 3/5)
        self.chat_prefix_len, self.chat_suffix_len = self._detect_chat_template_lens()
        print(f"[ZImageAdapter] chat_prefix_len={self.chat_prefix_len}, "
              f"chat_suffix_len={self.chat_suffix_len} (dynamically detected)")

    def _detect_chat_template_lens(self) -> Tuple[int, int]:
        """
        Run chat_template + tokenize with two different prompts and take the common prefix/suffix token counts.
        The result is the true prefix/suffix length of the chat template in the token sequence.
        """
        text_a = self._build_chat_text("x")
        text_b = self._build_chat_text("y z")
        ids_a = self.pipe.tokenizer(text_a, add_special_tokens=False).input_ids
        ids_b = self.pipe.tokenizer(text_b, add_special_tokens=False).input_ids
        # common prefix
        p = 0
        max_p = min(len(ids_a), len(ids_b))
        while p < max_p and ids_a[p] == ids_b[p]:
            p += 1
        # common suffix (must not cross p)
        s = 0
        while (s < len(ids_a) - p) and (s < len(ids_b) - p) and ids_a[-1 - s] == ids_b[-1 - s]:
            s += 1
        return p, s

    # ---------- internal utilities ----------
    def _build_chat_text(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        return self.pipe.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
        )

    def _encode_token_level(self, prompts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Takes hidden_states[-2] exactly like ZImagePipeline._encode_prompt, but keeps
        the fixed-length [B, L_max, D] tensor and attention_mask for the SAGE projection.
        Does not call pipe._encode_prompt (it slices back to a list by attention_mask, which hurts matrix projection).
        """
        chat_texts = [self._build_chat_text(p) for p in prompts]
        tok = self.pipe.tokenizer(
            chat_texts,
            padding="max_length",
            max_length=self.max_sequence_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = tok["input_ids"].to(self.device)
        attn = tok["attention_mask"].to(self.device)

        with torch.no_grad():
            out = self.pipe.text_encoder(
                input_ids=input_ids,
                attention_mask=attn.bool(),
                output_hidden_states=True,
            )
            embeds = out.hidden_states[-2]  # [B, L_max, D]
        return embeds, attn.long()

    def _user_content_mask(self, attention_mask: torch.Tensor) -> torch.Tensor:
        """Further strip the chat-template prefix/suffix from the attention_mask."""
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
        embeds, attn = self._encode_token_level([prompt])  # [1, L_max, D], [1, L_max]
        # cache the attn_mask; at generation time it slices the projected tensor back into list[Tensor]
        self._last_attn_mask = attn
        full_emb = embeds[0]
        user_mask = self._user_content_mask(attn)[0]
        return full_emb, user_mask

    def encode_pooled_concept(self, concept_text: str) -> Optional[torch.Tensor]:
        embeds, attn = self._encode_token_level([concept_text])
        emb = embeds[0]
        user_mask = self._user_content_mask(attn)[0]
        if user_mask.sum().item() == 0:
            return None
        return emb[user_mask].float().mean(dim=0)  # [D]

    def encode_pooled_masked_per_token(self, prompt: str) -> torch.Tensor:
        """
        Leave-one-out masks each user token of the prompt, re-encodes, and mean-pools into 1 vector.
        Returns: [N, D]
        """
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
        if n == 0:
            return torch.zeros(0, self.pipe.text_encoder.config.hidden_size, device=self.device)

        masked_ids = input_ids.repeat(n, 1)
        masked_attn = attn.repeat(n, 1)
        for i, pos in enumerate(positions):
            masked_ids[i, pos] = 0  # 0 as the mask

        with torch.no_grad():
            out = self.pipe.text_encoder(
                input_ids=masked_ids,
                attention_mask=masked_attn.bool(),
                output_hidden_states=True,
            )
            hidden = out.hidden_states[-2]

        # mean-pool the user-content part of each masked sequence → 1 D-dim vector
        pooled = []
        for i in range(n):
            row_user_mask = self._user_content_mask(masked_attn[i:i + 1])[0]
            valid = hidden[i][row_user_mask]  # [Lv, D]
            if valid.shape[0] == 0:
                pooled.append(torch.zeros(hidden.shape[-1], device=self.device))
            else:
                pooled.append(valid.mean(dim=0))
        return torch.stack(pooled, dim=0).float()  # [N, D]

    # ---------- inference ----------
    def _to_list_by_mask(self, prompt_embeds: torch.Tensor) -> List[torch.Tensor]:
        """
        ZImagePipeline.__call__ expects prompt_embeds as list[Tensor], each entry sliced to
        its valid length by attention_mask. Slice [1, L_max, D] back by self._last_attn_mask.
        """
        if self._last_attn_mask is None:
            raise RuntimeError("ZImageAdapter._last_attn_mask is empty; call encode_full_prompt first.")
        attn = self._last_attn_mask.bool()  # [1, L_max]
        out = []
        for i in range(prompt_embeds.shape[0]):
            valid_idx = attn[i] if attn.shape[0] == prompt_embeds.shape[0] else attn[0]
            out.append(prompt_embeds[i][valid_idx])
        return out

    def _encode_negative_list(self, negative_prompt: str = "") -> List[torch.Tensor]:
        """Return list[Tensor] for the negative prompt, exactly like ZImagePipeline._encode_prompt."""
        embeds, attn = self._encode_token_level([negative_prompt])
        attn_b = attn.bool()
        return [embeds[0][attn_b[0]]]
