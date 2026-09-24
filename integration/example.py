"""Minimal usage example for the end-to-end inner-guardrail pipeline.

Run (from the repository root):

    python integration/example.py \
        --model z-image-turbo \
        --model_path /path/to/Tongyi-MAI/Z-Image-Turbo \
        --prompt "a scenic photo of mountains at sunset"

Expected decisions:
    blocked_prompt  red tier: PE-MLP blocked the prompt before generation
    enhanced        SAGE-enhanced, latent detection passed, image returned
    passed          white tier (no SAGE), latent detection passed, image returned
    blocked_latent  latent detector fired at detect_step; generation aborted
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from guardrail_pipeline import GuardrailPipeline


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True,
                    choices=["z-image-turbo", "qwen-image-2512",
                             "hunyuan-image-2_1", "flux2-klein-base-9b",
                             "internvl-u"])
    ap.add_argument("--model_path", required=True,
                    help="Local path of the generation-pipeline weights "
                         "(diffusers pipeline or InternVL-U repository)")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--weights_root", default=None,
                    help="Root of exported guardrail weights "
                         "(default: <repo>/weights)")
    ap.add_argument("--out", default="guardrail_output.png",
                    help="Where to save the returned image")
    args = ap.parse_args()

    guardrail = GuardrailPipeline(
        args.model, args.model_path,
        weights_root=args.weights_root, device=args.device)

    result = guardrail.generate(args.prompt, seed=args.seed)

    info = {
        "decision": result.decision,
        "tier": result.tier,
        "risk_levels": {"porn": result.risk_levels[0],
                        "gore": result.risk_levels[1],
                        "ip": result.risk_levels[2]},
        "sage_applied": result.sage_applied,
        "alpha_used": result.alpha_used,
        "concepts_used": list(result.concepts_used),
        "n_tokens_modified": result.n_tokens_modified,
        "detect_step": result.detect_step,
        "steps_executed": result.steps_executed,
        "steps_saved": result.steps_saved,
        "detector_result": result.detector_result,
        "info": result.info,
    }
    print(json.dumps(info, indent=2, ensure_ascii=False, default=str))

    if result.image is not None:
        result.image.save(args.out)
        print(f"image saved to: {args.out}")
    else:
        print("no image returned (blocked)")


if __name__ == "__main__":
    main()
