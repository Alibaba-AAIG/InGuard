"""Tiered guardrail routing (online version of Algorithm 1 in the paper).

Maps PE-MLP risk levels to one of four tiers and decides the action chain:

    red        -> block the prompt directly (no generation)
    ip         -> SAGE with alpha_i on IP concepts, then latent detection
    borderline -> SAGE with alpha_{p/g} (highest-priority category) on porn/gore
                  concepts, then latent detection
    white      -> no SAGE; with fallback=True, latent detection on the original
                  latent (the "w/ fallback" variant used in the paper)

Risk-level semantics (PE-MLP heads, argmax):
    porn / gore : 0-5 (0 = safe, higher = more severe)
    ip          : 0-7 (0 = none, 1-5 = controlled IPs, 6-7 = benign variants)
"""

from dataclasses import dataclass, field
from typing import Any, Optional, Tuple


@dataclass
class GuardrailResult:
    """Return value of GuardrailPipeline.generate()."""

    decision: str                 # "blocked_prompt" | "enhanced" | "passed" | "blocked_latent"
    prompt: str
    image: Optional[Any] = None   # PIL.Image or None (None when blocked)
    # --- stage 1: PE-MLP risk classification ---
    risk_levels: Optional[Tuple[int, int, int]] = None   # (porn, gore, ip)
    tier: Optional[str] = None                            # "red" | "ip" | "borderline" | "white"
    # --- stage 2: SAGE enhancement ---
    sage_applied: bool = False
    alpha_used: Optional[float] = None
    concepts_used: Tuple[str, ...] = ()                  # selected concept groups
    n_tokens_modified: Optional[int] = None              # tokens with gamma > 0.5
    # --- stage 3: latent detection ---
    detector_result: Optional[dict] = None               # LatentDetector.predict() output
    detect_step: Optional[int] = None
    steps_executed: Optional[int] = None                 # denoising steps actually run
    steps_saved: int = 0                                 # steps skipped by early termination
    info: dict = field(default_factory=dict)

    @property
    def blocked(self) -> bool:
        return self.decision in ("blocked_prompt", "blocked_latent")


def classify_tier(porn_level: int, gore_level: int, ip_level: int, cfg: dict) -> str:
    """Classify a prompt into one of four tiers.

    cfg keys: tau_p, tau_g, borderline_porn, borderline_gore, borderline_ip.
    """
    ip_lo, ip_hi = cfg["borderline_ip"]
    is_ip = ip_lo <= ip_level <= ip_hi
    # IP=direct mode routes controlled-IP prompts to the red tier; the paper's
    # deployed configuration uses IP=SAGE-enh. (Algorithm 1, Appendix A.3).
    ip_blocks = (cfg.get("ip_mode", "sage") == "direct") and is_ip

    if porn_level >= cfg["tau_p"] or gore_level >= cfg["tau_g"] or ip_blocks:
        return "red"
    if is_ip:
        return "ip"
    if porn_level >= cfg["borderline_porn"] or gore_level >= cfg["borderline_gore"]:
        return "borderline"
    return "white"


def sage_route(porn_level: int, gore_level: int, ip_level: int, cfg: dict):
    """Pick SAGE alpha and concept groups for ip / borderline tiers.

    Mirrors the paper's multi-category routing: the concept subspace merges all
    triggered groups, and the sensitivity is taken from the highest-priority
    triggered category (porn > gore > ip).
    """
    ip_lo, ip_hi = cfg["borderline_ip"]
    is_ip = ip_lo <= ip_level <= ip_hi

    # Enhancement gate (Algorithm 1, risky-tier condition): only prompts with
    # porn >= beta_p, gore >= beta_g, or IP in beta_ip are SAGE-enhanced.
    # White-tier prompts (e.g. porn=1 or gore=1, below the enhancement
    # thresholds) pass through unenhanced.
    if not (porn_level >= cfg["borderline_porn"]
            or gore_level >= cfg["borderline_gore"] or is_ip):
        return None, ()

    # Borderline tier: porn/gore concepts participate; ip concepts do not
    # (ip-risk prompts are routed to the ip tier instead).
    if porn_level > 0 or gore_level > 0:
        groups = []
        if porn_level > 0:
            groups.append("porn")
        if gore_level > 0:
            groups.append("gore")
        alpha = cfg["alpha_p"] if porn_level > 0 else cfg["alpha_g"]
        return alpha, tuple(groups)

    # IP tier (ip-only risk)
    if is_ip:
        return cfg["alpha_i"], ("ip",)

    return None, ()


def needs_detection(tier: str, cfg: dict) -> bool:
    """Latent detection runs for every non-red tier in the paper's deployed
    configuration ("w/ fallback"): ip and borderline tiers check the SAGE-enhanced
    latent, white tier checks the original latent."""
    if tier in ("ip", "borderline"):
        return True
    if tier == "white":
        return bool(cfg.get("fallback", True))
    return False
