"""Online inference wrapper for the PE-MLP prompt risk classifier.

Loads a checkpoint produced by pe_mlp/train_prompt.py and maps a prompt's
token embeddings to three risk levels via argmax over the task heads:

    porn_level, gore_level : 0-5   (0 = safe, higher = more severe)
    ip_level               : 0-7   (0 = none, 1-5 = controlled IPs, 6-7 benign)

The model architecture below is a verbatim copy of PromptMultiTaskMLP in
pe_mlp/train_prompt.py (L307-337). Keep the two definitions in sync, or
load state dicts will silently mismatch.

Checkpoint format (see train_prompt.py L1302-1317):
    {'model_state_dict': ..., 'input_dim': int, 'hidden_dim': int,
     'num_layers': int, 'dropout': float, 'model_name': str, ...}
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PromptMultiTaskMLP(nn.Module):
    """Prompt embeddings multi-task classification MLP.
    Structure: Projection -> N x (Linear + LN + GELU + Dropout + Residual) -> three heads.
    """

    def __init__(self, input_dim, hidden_dim=1024, num_layers=3, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.proj_ln = nn.LayerNorm(hidden_dim)

        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            for _ in range(num_layers)
        ])

        self.head_porn = nn.Linear(hidden_dim, 6)   # 0-5
        self.head_gore = nn.Linear(hidden_dim, 6)   # 0-5
        self.head_ip = nn.Linear(hidden_dim, 8)     # 0-7

    def forward(self, x):
        """x: [B, input_dim] -> (logits_porn, logits_gore, logits_ip)"""
        x = self.proj_ln(F.gelu(self.proj(x)))
        for block in self.blocks:
            x = x + block(x)
        return self.head_porn(x), self.head_gore(x), self.head_ip(x)


def masked_mean_pool(embeds: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
    """[seq_len, D] (+ optional [seq_len] 0/1 mask) -> [D]."""
    embeds = embeds.float()
    if mask is None:
        return embeds.mean(dim=0)
    mask_f = mask.float().unsqueeze(-1)
    valid_sum = mask_f.sum(dim=0).clamp(min=1.0)
    return (embeds * mask_f).sum(dim=0) / valid_sum


class PEMLPRiskClassifier:
    """Loads a PE-MLP checkpoint and classifies prompt embeddings online."""

    def __init__(self, ckpt_path: str, device: str = "cuda"):
        # Payloads are plain dicts of tensors + scalars written by our own
        # train_prompt.py, but some scalar fields (e.g. test_loss) were saved
        # as numpy types, which the PyTorch 2.6+ weights-only unpickler
        # rejects. Try the safe loader first, fall back to a full load
        # (trusted source: our own training artifacts).
        try:
            payload = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        except Exception:
            payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.input_dim = int(payload["input_dim"])
        self.model = PromptMultiTaskMLP(
            input_dim=self.input_dim,
            hidden_dim=int(payload.get("hidden_dim", 1024)),
            num_layers=int(payload.get("num_layers", 3)),
            dropout=float(payload.get("dropout", 0.1)),
        )
        self.model.load_state_dict(payload["model_state_dict"])
        self.model.to(device).eval()
        self.device = device
        self.source_step = payload.get("global_step")

    @torch.no_grad()
    def predict(self, prompt_embeds: torch.Tensor, user_mask: torch.Tensor = None):
        """Classify one prompt.

        Args:
            prompt_embeds: [seq_len, D] token embeddings from the LDM text encoder
                (float, any device; user-content tokens selected by user_mask).
            user_mask: [seq_len] bool/0-1 mask of user content tokens, or None to
                pool over all tokens. PE-MLP was trained on pooled *user content*
                embeddings, so pass the adapter's user mask for exact parity.

        Returns:
            (porn_level, gore_level, ip_level): three ints from argmax.
        """
        pooled = masked_mean_pool(prompt_embeds, user_mask)          # [D]
        pooled = pooled.unsqueeze(0).to(self.device)                 # [1, D]
        logits_porn, logits_gore, logits_ip = self.model(pooled)
        return (
            int(logits_porn.argmax(dim=1).item()),
            int(logits_gore.argmax(dim=1).item()),
            int(logits_ip.argmax(dim=1).item()),
        )
