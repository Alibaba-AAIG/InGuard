"""
SAGE × InternVL-U inference entry (adapter + standalone script).

InternVL-U adapter for SAGE. Implements the ModelAdapter interface,
plugging the InternVL-U VLM-based generation pipeline into the SAGE framework.

══════════════════════════════════════════════════════════════
Architecture overview
══════════════════════════════════════════════════════════════

Pipeline: InternVLUPipeline (custom, not diffusers; the code lives in the local internvlu/ package)
Text encoder: VLM (Qwen2.5, hidden_size=2048, 28 layers)

Key source facts (verified against the actual model config):
  1. vlm_select_layer = [-1, -2]: the last 2 layers concatenated → D = 2 × 2048 = 4096
  2. decoder_projector = Identity (no projection); SAGE operates directly in the VLM hidden-state space
  3. triple CFG: cond / part_cond / uncond; batch dim = 3
  4. the processor creates 3 variants → VLM forward → _prepare_diffusion_inputs (state_mask extraction)
     → prepare_forward_input (pad to max(selected_len, 768) = 768)
  5. the state_mask-selected span: from the 2nd <|im_start|> to <img> (inclusive)
     structure: [<|im_start|>user\n, {USER_TEXT}, <|im_end|>\n<|im_start|>assistant\n<img>]
  6. prompt_embeds shape = [3, 768, 4096], attention_mask shape = [3, 768]
  7. SAGE modifies only the user-token positions of the cond variant (row 0)

══════════════════════════════════════════════════════════════
InternVLUAdapter constructor parameters
══════════════════════════════════════════════════════════════

  pipe                 a loaded InternVLUPipeline instance
  device               CUDA device id, e.g. "cuda"
  height               image height (default 1024)
  width                image width (default 1024)
  num_inference_steps  denoising steps (default 20)
  all_cfg_scale        full CFG strength (default 4.5)
  part_cfg_scale       partial CFG strength (default 2.0)
  seed                 random seed (default 42)

══════════════════════════════════════════════════════════════
ModelAdapter interface implementation
══════════════════════════════════════════════════════════════

  encode_full_prompt(prompt) → ([768, 4096], [768] bool)
      encodes the cond variant: processor → VLM forward → prepare_forward_input
      returns the full embedding and the user-token mask

  encode_pooled_concept(concept_text) → [4096] or None
      encodes a single toxic concept, pooling the user-token positions

  encode_pooled_masked_per_token(prompt) → [N, 4096]
      leave-one-out masking: re-encode with each user token masked in turn
      N = the number of user tokens; each returns a pooled vector
      note: this runs N VLM forwards and is slow

  get_last_attn_mask() → [1, 768] or None
      returns the attention mask of the last encode_full_prompt

══════════════════════════════════════════════════════════════
Notes
══════════════════════════════════════════════════════════════

  This file provides only InternVLUAdapter (reused by sage/sage_enhance.py and
  integration/guardrail_pipeline.py). For full data collection and end-to-end
  inference use sage/sage_enhance.py and integration/.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from typing import List, Optional, Tuple

import torch

from base import ModelAdapter


class InternVLUAdapter(ModelAdapter):
    """InternVL-U SAGE adapter."""

    def __init__(
        self,
        pipe,
        device: str = "cuda",
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 20,
        all_cfg_scale: float = 4.5,
        part_cfg_scale: float = 2.0,
        seed: int = 42,
    ):
        self.pipe = pipe
        self.device = device
        self.height = height
        self.width = width
        self.num_inference_steps = num_inference_steps
        self.all_cfg_scale = all_cfg_scale
        self.part_cfg_scale = part_cfg_scale
        self.seed = seed

        # module shortcuts
        self.vlm = pipe.vlm
        self.tokenizer = pipe.tokenizer
        self.processor_ = pipe.processor
        self.gen_decoder = pipe.generation_decoder
        self.gen_cfg = self.gen_decoder.config

        # special token IDs
        self.im_start_id = self.vlm.im_start_token_id   # 151644
        self.im_end_id = self.vlm.im_end_token_id        # 151645
        self.img_start_id = self.vlm.img_start_token_id  # 151669

        # dynamically detect the prefix/suffix lengths inside the state_mask-selected span
        self.selected_prefix_len, self.selected_suffix_len = self._detect_selected_prefix_suffix()
        print(f"[InternVLUAdapter] selected_prefix_len={self.selected_prefix_len}, "
              f"selected_suffix_len={self.selected_suffix_len}")

        # caches
        self._last_attn_mask: Optional[torch.Tensor] = None
        self._last_prompt: Optional[str] = None

    # ================================================================
    # internal utilities
    # ================================================================

    def _get_selected_token_ids(self, prompt: str) -> List[int]:
        """Get the token IDs of the state_mask-selected span (for prefix/suffix detection)."""
        inputs = self.processor_(
            prompt=prompt,
            generation_mode="image",
            padding=True,
            return_tensors="pt",
            height=self.height,
            width=self.width,
        )
        input_ids = inputs["input_ids"][0]  # the cond variant, [L]

        # find the 2nd <|im_start|> and the <img> after it
        im_start_positions = (input_ids == self.im_start_id).nonzero(as_tuple=True)[0]
        img_start_positions = (input_ids == self.img_start_id).nonzero(as_tuple=True)[0]

        if len(im_start_positions) < 2 or len(img_start_positions) < 1:
            return input_ids.tolist()

        second_im_start = im_start_positions[1].item()
        img_starts_after = img_start_positions[img_start_positions >= second_im_start]
        if len(img_starts_after) == 0:
            return input_ids.tolist()
        img_start_pos = img_starts_after[0].item()

        return input_ids[second_im_start:img_start_pos + 1].tolist()

    def _detect_selected_prefix_suffix(self) -> Tuple[int, int]:
        """Detect the common prefix/suffix lengths inside the state_mask-selected span using two different prompts."""
        ids_a = self._get_selected_token_ids("x")
        ids_b = self._get_selected_token_ids("y z")

        p = 0
        while p < min(len(ids_a), len(ids_b)) and ids_a[p] == ids_b[p]:
            p += 1

        s = 0
        while (s < len(ids_a) - p and s < len(ids_b) - p
               and ids_a[-1 - s] == ids_b[-1 - s]):
            s += 1

        return p, s

    def _build_cond_inputs(self, prompt: str) -> dict:
        """Run the processor to get the 3 variants; extract the cond variant (index 0)."""
        inputs = self.processor_(
            prompt=prompt,
            generation_mode="image",
            padding=True,
            return_tensors="pt",
            height=self.height,
            width=self.width,
        )
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(self.device)

        # take only the cond variant
        return {
            "input_ids": inputs["input_ids"][0:1],          # [1, L]
            "attention_mask": inputs["attention_mask"][0:1], # [1, L]
            "generation_flags": torch.tensor([1], dtype=torch.long, device=self.device),
            "image_grid_thw_gen": inputs["image_grid_thw_gen"][0:1],  # [1, 3]
            "pixel_values": None,
            "pixel_values_gen": None,
        }

    def _encode_and_pad(self, cond_inputs: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        VLM forward → _prepare_diffusion_inputs → prepare_forward_input
        Returns:
            encoder_hidden_states: [1, 768, 4096]
            attention_masks: [1, 768] bool
        """
        # VLM forward
        outputs = self.vlm.generate_hidden_states(
            input_ids=cond_inputs["input_ids"],
            attention_mask=cond_inputs["attention_mask"],
            pixel_values=cond_inputs.get("pixel_values"),
        )
        vlm_hidden_states = outputs.hidden_states

        # _prepare_diffusion_inputs (state_mask extraction)
        diffusion_inputs = self.pipe._prepare_diffusion_inputs(
            input_ids=cond_inputs["input_ids"],
            attention_mask=cond_inputs["attention_mask"],
            pixel_values=cond_inputs.get("pixel_values"),
            pixel_values_gen=cond_inputs.get("pixel_values_gen"),
            image_grid_thw_gen=cond_inputs["image_grid_thw_gen"],
            generation_flags=cond_inputs["generation_flags"],
            vlm_hidden_states=vlm_hidden_states,
        )

        # prepare_forward_input (padding + identity projector)
        encoder_hidden_states, attention_masks, _ = self.gen_decoder.prepare_forward_input(
            diffusion_inputs["encoder_hidden_states"],
            encoder_image_token_mask=diffusion_inputs.get("encoder_image_token_mask"),
        )

        return encoder_hidden_states, attention_masks

    def _build_user_mask(self, valid_len: int) -> torch.Tensor:
        """
        Marks the user-token positions in the [768] space.
        Structure: [prefix (selected_prefix_len), user_text, suffix (selected_suffix_len), padding]
        """
        user_mask = torch.zeros(768, dtype=torch.bool, device=self.device)
        user_start = self.selected_prefix_len
        user_end = valid_len - self.selected_suffix_len
        if user_end > user_start:
            user_mask[user_start:user_end] = True
        return user_mask

    def _get_user_token_positions(self, input_ids: torch.Tensor) -> List[int]:
        """Get the absolute positions of the user tokens in input_ids (for the leave-one-out mask)."""
        ids = input_ids[0]  # [L]
        im_start_positions = (ids == self.im_start_id).nonzero(as_tuple=True)[0]
        img_start_positions = (ids == self.img_start_id).nonzero(as_tuple=True)[0]

        second_im_start = im_start_positions[1].item()
        img_start_pos = img_start_positions[img_start_positions >= second_im_start][0].item()

        selected_start = second_im_start
        selected_end = img_start_pos + 1  # inclusive

        user_start = selected_start + self.selected_prefix_len
        user_end = selected_end - self.selected_suffix_len
        return list(range(user_start, user_end))

    # ================================================================
    # ModelAdapter interface implementation
    # ================================================================

    def encode_full_prompt(self, prompt: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encodes the full prompt (cond variant); returns [768, D] + user_mask [768].
        """
        cond_inputs = self._build_cond_inputs(prompt)
        enc_hs, attn_masks = self._encode_and_pad(cond_inputs)

        full_emb = enc_hs[0]  # [768, 4096]
        valid_len = int(attn_masks[0].sum())
        user_mask = self._build_user_mask(valid_len)

        # cache
        self._last_attn_mask = attn_masks  # [1, 768]
        self._last_prompt = prompt

        return full_emb, user_mask

    def encode_pooled_concept(self, concept_text: str) -> Optional[torch.Tensor]:
        """Encode a single concept; returns a [D] pooled vector."""
        cond_inputs = self._build_cond_inputs(concept_text)
        enc_hs, attn_masks = self._encode_and_pad(cond_inputs)

        full_emb = enc_hs[0]  # [768, 4096]
        valid_len = int(attn_masks[0].sum())
        user_mask = self._build_user_mask(valid_len)

        if user_mask.sum().item() == 0:
            return None
        return full_emb[user_mask].float().mean(dim=0)  # [4096]

    def encode_pooled_masked_per_token(self, prompt: str) -> torch.Tensor:
        """
        Leave-one-out masks each user token of the prompt, re-encodes, and pools.
        Returns: [N, D]
        """
        cond_inputs = self._build_cond_inputs(prompt)
        input_ids = cond_inputs["input_ids"]  # [1, L]

        user_positions = self._get_user_token_positions(input_ids)
        n = len(user_positions)
        if n == 0:
            return torch.zeros(0, 4096, device=self.device)

        pooled = []
        for pos in user_positions:
            masked_inputs = dict(cond_inputs)
            masked_ids = input_ids.clone()
            masked_ids[0, pos] = 0  # the mask token (same as official SAGE)
            masked_inputs["input_ids"] = masked_ids

            enc_hs, attn_masks = self._encode_and_pad(masked_inputs)
            full_emb = enc_hs[0]  # [768, 4096]
            valid_len = int(attn_masks[0].sum())
            mask = self._build_user_mask(valid_len)

            if mask.sum().item() == 0:
                pooled.append(torch.zeros(4096, device=self.device))
            else:
                pooled.append(full_emb[mask].float().mean(dim=0))

        return torch.stack(pooled, dim=0)  # [N, 4096]

    def get_last_attn_mask(self) -> Optional[torch.Tensor]:
        """Return the attention mask [1, 768] of the last encode_full_prompt."""
        return self._last_attn_mask

    # ================================================================
    # inference
    # ================================================================
