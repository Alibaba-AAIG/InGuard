"""
SAGE adapter for Qwen-Image-2512.


Pipeline: diffusers.QwenImagePipeline
Text encoder: Qwen2.5-VL (text side)
Characteristics:
  - pipe._get_qwen_prompt_embeds() already applies prompt_template_encode internally and drops the first 34
    chat-template tokens, outputting prompt_embeds + attention_mask (mask=1 means user content + the ~5 tail tokens).
  - pipeline inference requires passing prompt_embeds and prompt_embeds_mask together (same for the negative).
  - for leave-one-out masks we do not enter _get_qwen_prompt_embeds; instead we directly take the tokenizer output,
    wrap it with the chat template, mask the user span, run text_encoder.hidden_states[-1], and apply
    the same [drop_idx:] slicing rule as _get_qwen_prompt_embeds on the output.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from typing import Optional, Tuple

import torch

from base import ModelAdapter


class QwenImageAdapter(ModelAdapter):
    """Qwen-Image-2512 SAGE adapter."""

    def __init__(self, pipe, device: str = "cuda"):
        self.pipe = pipe
        self.device = device
        # same as official
        self.template = pipe.prompt_template_encode
        self.drop_idx = pipe.prompt_template_encode_start_idx
        self.tokenizer_max_length = pipe.tokenizer_max_length

        # cache the attention mask for the pipeline
        self._last_attn_mask: Optional[torch.Tensor] = None

        # dynamically detect the tail length (the token count of the template tail <|im_end|>\n<|im_start|>assistant\n part, replacing the old hardcoded 5)
        self.tail_len = self._detect_tail_len()
        print(f"[QwenImageAdapter] drop_idx={self.drop_idx}, "
              f"tail_len={self.tail_len} (dynamically detected)")

    def _detect_tail_len(self) -> int:
        """
        Tokenize template.format("x") / template.format("y z") separately and
        take the common suffix length — the true token count remaining at the template tail (after the drop).
        """
        text_a = self.template.format("x")
        text_b = self.template.format("y z")
        ids_a = self.pipe.tokenizer(text_a, add_special_tokens=False).input_ids
        ids_b = self.pipe.tokenizer(text_b, add_special_tokens=False).input_ids
        s = 0
        while (s < len(ids_a)) and (s < len(ids_b)) and ids_a[-1 - s] == ids_b[-1 - s]:
            s += 1
        return s

    # ---------- internal ----------
    def _encode_via_pipeline(self, prompt: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Goes through the official pipe._get_qwen_prompt_embeds; the output embeds and mask already have the chat-template prefix dropped.
        prompt_embeds: [B, Lv, D], attention_mask: [B, Lv]
        """
        with torch.no_grad():
            embeds, mask = self.pipe._get_qwen_prompt_embeds([prompt], self.device)
        return embeds, mask

    def _user_content_mask_from_attn(self, attn_mask: torch.Tensor) -> torch.Tensor:
        """
        Further strip the tail chat-template tokens from the attn_mask output of _get_qwen_prompt_embeds.
        attn_mask: [L] or [B, L]
        """
        if attn_mask.dim() == 1:
            attn_mask = attn_mask.unsqueeze(0)
            squeeze_back = True
        else:
            squeeze_back = False
        user_mask = torch.zeros_like(attn_mask, dtype=torch.bool)
        for b in range(attn_mask.shape[0]):
            valid_idx = torch.where(attn_mask[b].bool())[0]
            if len(valid_idx) <= self.tail_len:
                continue
            content_idx = valid_idx[: -self.tail_len]
            user_mask[b, content_idx] = True
        return user_mask[0] if squeeze_back else user_mask

    # ---------- interface implementation ----------
    def encode_full_prompt(self, prompt: str) -> Tuple[torch.Tensor, torch.Tensor]:
        embeds, attn = self._encode_via_pipeline(prompt)
        # cache the attention mask for pipeline reuse
        self._last_attn_mask = attn.long()
        full_emb = embeds[0]                           # [Lv, D]
        user_mask = self._user_content_mask_from_attn(attn[0])
        return full_emb, user_mask

    def get_last_attn_mask(self) -> Optional[torch.Tensor]:
        return self._last_attn_mask

    def encode_pooled_concept(self, concept_text: str) -> Optional[torch.Tensor]:
        embeds, attn = self._encode_via_pipeline(concept_text)
        user_mask = self._user_content_mask_from_attn(attn[0])
        if user_mask.sum().item() == 0:
            return None
        return embeds[0][user_mask].float().mean(dim=0)

    def encode_pooled_masked_per_token(self, prompt: str) -> torch.Tensor:
        """
        leave-one-out mask implementation. To align with the _get_qwen_prompt_embeds output we need:
          1) wrap the text with prompt_template_encode
          2) tokenize -> mask the token positions of the user span
          3) text_encoder(output_hidden_states=True).hidden_states[-1]
          4) extract the valid part by attention_mask, then drop the first drop_idx tokens
        """
        # 1) the full template text
        full_text = self.template.format(prompt)
        tok = self.pipe.tokenizer(
            [full_text],
            max_length=self.tokenizer_max_length + self.drop_idx,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)
        input_ids = tok.input_ids        # [1, L_full]
        attn = tok.attention_mask        # [1, L_full]

        # 2) the user-content token span: within [drop_idx, valid_len - tail_len)
        valid_len = int(attn[0].sum().item())
        start = self.drop_idx
        end = valid_len - self.tail_len
        positions = list(range(start, max(start, end)))
        n = len(positions)

        d = self.pipe.text_encoder.config.hidden_size
        if n == 0:
            return torch.zeros(0, d, device=self.device)

        masked_ids = input_ids.repeat(n, 1)
        masked_attn = attn.repeat(n, 1)
        for i, pos in enumerate(positions):
            masked_ids[i, pos] = 0  # same as the official implementation: 0 as the mask token id

        with torch.no_grad():
            out = self.pipe.text_encoder(
                input_ids=masked_ids,
                attention_mask=masked_attn,
                output_hidden_states=True,
            )
            hidden = out.hidden_states[-1]  # [n, L_full, D]

        # 3) slice like the official code: first take the attention_mask valid part, then [drop_idx:]
        pooled = []
        for i in range(n):
            valid_mask_i = masked_attn[i].bool()
            valid_hidden_i = hidden[i][valid_mask_i]    # [valid_len_i, D]
            content_i = valid_hidden_i[self.drop_idx:]  # drop the chat-template prefix
            # then drop the tail_len at the end
            if content_i.shape[0] <= self.tail_len:
                pooled.append(torch.zeros(d, device=self.device))
                continue
            content_i = content_i[: -self.tail_len]     # keep only the user tokens
            if content_i.shape[0] == 0:
                pooled.append(torch.zeros(d, device=self.device))
            else:
                pooled.append(content_i.float().mean(dim=0))
        return torch.stack(pooled, dim=0).float()  # [N, D]

    # ---------- inference ----------
    def _get_negative_embeds(self, negative_prompt: str = "") -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            neg_embeds, neg_mask = self.pipe._get_qwen_prompt_embeds([negative_prompt], self.device)
        return neg_embeds, neg_mask.long()
