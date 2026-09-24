"""
SAGE adapter for HunyuanImage-2.1.


Pipeline: diffusers.HunyuanImagePipeline
Text encoder (primary):    Qwen2.5-VL-7B-Instruct  -> hidden_states[-3]
Text encoder (secondary):  ByT5  -> only renders glyphs; unrelated to "safety", SAGE does not touch it

Source facts (diffusers 0.38.0 hunyuan_image/pipeline_hunyuanimage.py L223-258):
  1. template = "<|im_start|>system\\n...:<|im_end|>\\n<|im_start|>user\\n{}<|im_end|>"
     Note: unlike ZImage, **the tail is only <|im_end|>** — there is no
     "<|im_start|>assistant\n" — so after dropping the system prefix (drop_idx=34),
     the suffix is just 1 <|im_end|> token.
  2. drop_idx = pipe.prompt_template_encode_start_idx = 34
  3. hidden_state_skip_layer = 2 → hidden_states[-3]
  4. _get_qwen_prompt_embeds already drops the prefix via [:, drop_idx:]; here we only need to strip the tail suffix
  5. the ByT5 path runs _get_byt5_prompt_embeds + extract_glyph_text inside the adapter;
     at generation time both embeds are passed to pipe.__call__ together (bypassing the
     check_inputs restriction that forbids passing prompt + prompt_embeds simultaneously).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from typing import Optional, Tuple

import torch

from base import ModelAdapter


class HunyuanImageAdapter(ModelAdapter):
    """HunyuanImage-2.1 SAGE adapter."""

    HIDDEN_STATE_SKIP_LAYER = 2  # take hidden_states[-3], same as official

    def __init__(self, pipe, device: str = "cuda"):
        self.pipe = pipe
        self.device = device

        # same as official
        self.template = pipe.prompt_template_encode
        self.drop_idx = pipe.prompt_template_encode_start_idx
        self.tokenizer_max_length = pipe.tokenizer_max_length

        # cache the attention mask and the raw prompt: the ByT5 side path needs the prompt string at generation time
        self._last_attn_mask: Optional[torch.Tensor] = None
        self._last_prompt: Optional[str] = None

        # dynamically detect tail_len (how many tokens remain at the template tail after the drop)
        # for HunyuanImage this is usually 1 (<|im_end|>), but still detected dynamically in case of tokenizer differences
        self.tail_len = self._detect_tail_len()
        print(f"[HunyuanImageAdapter] drop_idx={self.drop_idx}, "
              f"tail_len={self.tail_len} (dynamically detected)")

    # ---------- internal utilities ----------
    def _detect_tail_len(self) -> int:
        """Tokenize template.format("x") / template.format("y z") and take the common suffix length."""
        text_a = self.template.format("x")
        text_b = self.template.format("y z")
        ids_a = self.pipe.tokenizer(text_a, add_special_tokens=False).input_ids
        ids_b = self.pipe.tokenizer(text_b, add_special_tokens=False).input_ids
        s = 0
        while (s < len(ids_a)) and (s < len(ids_b)) and ids_a[-1 - s] == ids_b[-1 - s]:
            s += 1
        return s

    def _encode_via_pipeline(self, prompt: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Goes through the official pipe._get_qwen_prompt_embeds; the output
        prompt_embeds + attention_mask already have the chat-template prefix dropped.
        """
        with torch.no_grad():
            embeds, mask = self.pipe._get_qwen_prompt_embeds(
                tokenizer=self.pipe.tokenizer,
                text_encoder=self.pipe.text_encoder,
                prompt=[prompt],
                device=self.device,
                tokenizer_max_length=self.tokenizer_max_length,
                template=self.template,
                drop_idx=self.drop_idx,
                hidden_state_skip_layer=self.HIDDEN_STATE_SKIP_LAYER,
            )
        return embeds, mask

    def _user_content_mask_from_attn(self, attn_mask: torch.Tensor) -> torch.Tensor:
        """Further strip the tail suffix from the attn_mask output of _get_qwen_prompt_embeds."""
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
            content_idx = valid_idx[: -self.tail_len] if self.tail_len > 0 else valid_idx
            user_mask[b, content_idx] = True
        return user_mask[0] if squeeze_back else user_mask

    # ---------- interface implementation ----------
    def encode_full_prompt(self, prompt: str) -> Tuple[torch.Tensor, torch.Tensor]:
        embeds, attn = self._encode_via_pipeline(prompt)
        self._last_attn_mask = attn.long()
        self._last_prompt = prompt
        full_emb = embeds[0]                                  # [Lv, D]
        user_mask = self._user_content_mask_from_attn(attn[0])
        return full_emb, user_mask

    def get_last_attn_mask(self) -> Optional[torch.Tensor]:
        return self._last_attn_mask

    def encode_pooled_concept(self, concept_text: str) -> Optional[torch.Tensor]:
        embeds, attn = self._encode_via_pipeline(concept_text)
        user_mask = self._user_content_mask_from_attn(attn[0])
        if user_mask.sum().item() == 0:
            return None
        return embeds[0][user_mask].float().mean(dim=0)  # [D]

    def encode_pooled_masked_per_token(self, prompt: str) -> torch.Tensor:
        """
        leave-one-out mask implementation. To align with the _get_qwen_prompt_embeds output:
          1) wrap the prompt with self.template
          2) tokenize, then leave-one-out mask the user-token positions
          3) text_encoder(output_hidden_states=True).hidden_states[-3]
          4) take the valid part by attention_mask → drop the prefix via [drop_idx:] → then strip the tail_len suffix
        """
        full_text = self.template.format(prompt)
        tok = self.pipe.tokenizer(
            [full_text],
            max_length=self.tokenizer_max_length + self.drop_idx,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)
        input_ids = tok.input_ids
        attn = tok.attention_mask

        valid_len = int(attn[0].sum().item())
        start = self.drop_idx
        end = valid_len - self.tail_len
        positions = list(range(start, max(start, end)))
        n = len(positions)

        # Qwen2_5_VLForConditionalGeneration's top-level config has no hidden_size;
        # the field lives in the sub-config text_config — fall back to it.
        cfg = self.pipe.text_encoder.config
        d = getattr(cfg, "hidden_size", None)
        if d is None and hasattr(cfg, "text_config"):
            d = cfg.text_config.hidden_size
        if n == 0:
            return torch.zeros(0, d, device=self.device)

        masked_ids = input_ids.repeat(n, 1)
        masked_attn = attn.repeat(n, 1)
        for i, pos in enumerate(positions):
            masked_ids[i, pos] = 0  # 0 as the mask token id

        with torch.no_grad():
            out = self.pipe.text_encoder(
                input_ids=masked_ids,
                attention_mask=masked_attn,
                output_hidden_states=True,
            )
            hidden = out.hidden_states[-(self.HIDDEN_STATE_SKIP_LAYER + 1)]  # hidden_states[-3]

        pooled = []
        for i in range(n):
            valid_mask_i = masked_attn[i].bool()
            valid_hidden_i = hidden[i][valid_mask_i]    # [valid_len_i, D]
            content_i = valid_hidden_i[self.drop_idx:]  # drop the prefix
            if content_i.shape[0] <= self.tail_len:
                pooled.append(torch.zeros(d, device=self.device))
                continue
            if self.tail_len > 0:
                content_i = content_i[: -self.tail_len]
            if content_i.shape[0] == 0:
                pooled.append(torch.zeros(d, device=self.device))
            else:
                pooled.append(content_i.float().mean(dim=0))
        return torch.stack(pooled, dim=0).float()  # [N, D]

    # ---------- inference ----------
    def _build_byt5_embeds(self, prompt: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Manually reproduces the ByT5 side-path logic of HunyuanImagePipeline.encode_prompt (batch=1 only):
          - extract_glyph_text(prompt) -> the quoted glyph text / None
          - returns zeros when None, otherwise goes through _get_byt5_prompt_embeds
        """
        # lazy import to keep the top-level imports clean
        from diffusers.pipelines.hunyuan_image.pipeline_hunyuanimage import extract_glyph_text

        glyph_text = extract_glyph_text(prompt)
        if glyph_text is None:
            embeds = torch.zeros(
                (1, self.pipe.tokenizer_2_max_length, self.pipe.text_encoder_2.config.d_model),
                device=self.device,
            )
            mask = torch.zeros(
                (1, self.pipe.tokenizer_2_max_length), device=self.device, dtype=torch.int64
            )
            return embeds, mask

        with torch.no_grad():
            embeds, mask = self.pipe._get_byt5_prompt_embeds(
                tokenizer=self.pipe.tokenizer_2,
                text_encoder=self.pipe.text_encoder_2,
                prompt=glyph_text,
                device=self.device,
                tokenizer_max_length=self.pipe.tokenizer_2_max_length,
            )
        return embeds, mask
