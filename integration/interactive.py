"""Interactive end-to-end demo of the inner guardrail.

Loads the generation model plus all three InGuard components (PE-MLP risk
classification -> SAGE enhancement -> latent detection), then repeatedly reads
a prompt from the terminal and reports the pipeline decision:

    passed          white tier, no SAGE, latent detection passed -> image
    enhanced        SAGE applied, latent detection passed -> image
    blocked_prompt  red tier: PE-MLP blocked the prompt before generation
    blocked_latent  latent detector fired at detect_step; generation aborted

Run (from the repository root):

    python integration/interactive.py \
        --model z-image-turbo \
        --model_path /path/to/Tongyi-MAI/Z-Image-Turbo

Commands inside the session:
    <any text>   generate with that prompt
    seed <n>     switch the random seed
    q / quit / exit / Ctrl-D        leave the session (empty lines are ignored)

Images are saved to ./interactive_output/<index>_<decision>.png
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from guardrail_pipeline import GuardrailPipeline


def _print_report(result, out_path=None):
    """Human-readable one-shot report of a GuardrailResult."""
    sage = "yes" if result.sage_applied else "no"
    p, g, i = result.risk_levels

    if result.decision == "blocked_prompt":
        print(f"  [blocked] stage = Stage 1 PE-MLP (before generation, red tier rejected)")
        print(f"  SAGE applied: {sage} (red tier never enters SAGE)")
        print(f"  risk levels: porn={p}, gore={g}, ip={i} (tier={result.tier})")
        print(f"  denoising never started, saved {result.steps_saved} steps of compute")
    elif result.decision == "blocked_latent":
        print(f"  [blocked] stage = Stage 3 latent detector (hit at step {result.detect_step})")
        print(f"  SAGE applied: {sage}")
        print(f"  risk levels: porn={p}, gore={g}, ip={i} (tier={result.tier})")
        print(f"  aborted after {result.steps_executed} steps, saved {result.steps_saved} steps of compute")
        if result.detector_result:
            print(f"  detector output: {result.detector_result}")
    else:
        label = "SAGE enhanced" if result.sage_applied else "not enhanced (white tier, no SAGE needed)"
        print(f"  [passed] {label}, latent detection passed, image generated")
        print(f"  risk levels: porn={p}, gore={g}, ip={i} (tier={result.tier})")
        if result.sage_applied:
            print(f"  SAGE: alpha={result.alpha_used}, "
                  f"tokens modified={result.n_tokens_modified}, "
                  f"concept groups={list(result.concepts_used)}")
        if out_path:
            print(f"  image saved to: {out_path}")


def main():
    ap = argparse.ArgumentParser(
        description="Interactive end-to-end demo (type prompts in the terminal)")
    ap.add_argument("--model", required=True,
                    choices=["z-image-turbo", "qwen-image-2512",
                             "hunyuan-image-2_1", "flux2-klein-base-9b",
                             "internvl-u"])
    ap.add_argument("--model_path", required=True,
                    help="Local path of the generation-pipeline weights "
                         "(diffusers pipeline or InternVL-U repository)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--weights_root", default=None,
                    help="Root of exported guardrail weights (default: <repo>/weights)")
    ap.add_argument("--out_dir", default="interactive_output")
    args = ap.parse_args()

    print(f"Loading {args.model} + InGuard components (PE-MLP / SAGE / latent detector)...")
    guardrail = GuardrailPipeline(
        args.model, args.model_path,
        weights_root=args.weights_root, device=args.device)
    os.makedirs(args.out_dir, exist_ok=True)

    seed = args.seed
    n_done = 0
    # Loading the model can take minutes; keys pressed meanwhile (especially
    # Enter) stay in the stdin buffer and would make the first input() return
    # an empty line immediately. Flush stdin before entering the REPL.
    try:
        import termios
        termios.tcflush(sys.stdin, termios.TCIFLUSH)
    except Exception:
        pass  # not a tty (pipe/redirection); nothing to flush
    print(f"\nModel ready. Type a prompt to generate; seed={seed}; "
          f"type q/quit/exit to leave, 'seed <n>' to change the seed.\n")

    while True:
        try:
            prompt = input("prompt> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if prompt in ("q", "quit", "exit"):
            print("Bye.")
            break
        if not prompt:
            # Empty lines are most likely accidental Enters or leftover buffer
            # from loading; ignore instead of exiting.
            print("  (empty line ignored; type q/quit/exit to leave)")
            continue
        if prompt.startswith("seed ") or prompt == "seed":
            parts = prompt.split()
            if len(parts) == 2 and parts[1].lstrip("-").isdigit():
                seed = int(parts[1])
                print(f"  seed switched to {seed}\n")
            else:
                print(f"  usage: seed <integer>; current seed={seed}\n")
            continue

        n_done += 1
        print(f"\n--- [{n_done}] seed={seed} ---")
        try:
            result = guardrail.generate(prompt, seed=seed)
        except Exception as e:
            print(f"  [error] {type(e).__name__}: {e}\n")
            continue

        out_path = None
        if result.image is not None:
            out_path = os.path.join(
                args.out_dir, f"{n_done:03d}_{result.decision}.png")
            result.image.save(out_path)
        _print_report(result, out_path)
        print()


if __name__ == "__main__":
    main()
