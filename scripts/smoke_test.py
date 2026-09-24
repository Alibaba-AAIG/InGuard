#!/usr/bin/env python3
"""Automated readiness smoke test for the InGuard repository.

Loads real server paths from scripts/local_paths.env (gitignored; see
local_paths.env.example), then runs verification in stages — cheap first:

    python scripts/smoke_test.py --stage env      # paths exist + env loaded
    python scripts/smoke_test.py --stage link     # symlink RevGen layout
    python scripts/smoke_test.py --stage tests    # unit tests (CPU)
    python scripts/smoke_test.py --stage verify   # verify_pipeline quick+components
    python scripts/smoke_test.py --stage pe       # PE-MLP 1-epoch smoke train
    python scripts/smoke_test.py --stage all      # env -> link -> tests -> verify -> pe

Stage `link` adapts an existing data layout to the open-source one WITHOUT
copying anything:
  - the four RevGen CSVs (any original names) -> $INGUARD_DATA_ROOT/RevGen/
    under their standard names
  - generated benchmark dirs (e.g. testset-seed42-1024-9steps; a version
    infix in the name is also accepted) -> $INGUARD_OUTPUT_ROOT/<model>/
    {train,test}set-seed42-<res>-<steps>steps
  - T2I models stored under other names / locations (per-model full-path
    env vars in local_paths.env) -> $INGUARD_MODELS_ROOT/<MODEL_REGISTRY
    name>, plus an optional torch hub cache and the InternVL-U pipeline
    code dir

Stage `pe` truncates the labeled CSVs to --pe_rows rows (header kept) so a
full training entry is exercised in seconds, then restores nothing (the
truncated copies live in a temp dir).

No repository code is modified by any stage.
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from glob import glob

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ENV_FILE = os.path.join(_REPO_ROOT, "scripts", "local_paths.env")

# (res, steps) per model — must match the benchmark save_data scripts
MODEL_RES_STEPS = {
    "z-image-turbo": (1024, 9),
    "qwen-image-2512": (1328, 10),
    "internvl-u": (1024, 20),
    "hunyuan-image-2_1": (2048, 10),
    "flux2-klein-base-9b": (1024, 10),
}

MODEL_DIRS = {  # expected under INGUARD_MODELS_ROOT (MODEL_REGISTRY names)
    "z-image-turbo": "Z-Image-Turbo",
    "qwen-image-2512": "Qwen-Image-2512",
    "internvl-u": "InternVL-U",
    "hunyuan-image-2_1": "HunyuanImage-2.1-Diffusers",
    "flux2-klein-base-9b": "FLUX.2-klein-base-9B",
}

# Standard model dir -> env var holding the FULL path of that model on
# this machine. The open-source code expects {INGUARD_MODELS_ROOT}/
# <standard name> (see MODEL_REGISTRY); if a model lives somewhere else
# (e.g. under an HF-style org/repo name), set its env var in
# scripts/local_paths.env and `--stage link` creates the symlink. Unset
# vars are skipped — models already placed under INGUARD_MODELS_ROOT with
# the standard names need no linking at all.
MODEL_LINKS = {
    "Z-Image-Turbo":              "SERVER_MODEL_Z_IMAGE",
    "Qwen-Image-2512":            "SERVER_MODEL_QWEN",
    "HunyuanImage-2.1-Diffusers": "SERVER_MODEL_HUNYUAN",
    "FLUX.2-klein-base-9B":       "SERVER_MODEL_FLUX",
    "InternVL-U":                 "SERVER_MODEL_INTERNVLU",
}

CSV_LINKS = [
    ("SERVER_CSV_TRAINSET", "trainset.csv"),
    ("SERVER_CSV_TESTSET", "testset.csv"),
    ("SERVER_CSV_TRAIN_LABELED", "trainset_labeled.csv"),
    ("SERVER_CSV_TEST_LABELED", "testset_labeled.csv"),
]

SMOKE_MODEL = "z-image-turbo"


def _ok(msg):
    print(f"  [PASS] {msg}")


def _fail(msg, hint=""):
    print(f"  [FAIL] {msg}")
    if hint:
        print(f"         {hint}")
    return False


def _note(msg):
    print(f"  [..]   {msg}")


# ---------------------------------------------------------------- env loading

def load_env():
    """Export local_paths.env into os.environ (existing values win).

    Values may reference earlier lines via ${VAR} (expanded against the
    variables exported above, then against the inherited environment)."""
    if not os.path.isfile(_ENV_FILE):
        return False, f"{_ENV_FILE} not found"
    exported = 0
    with open(_ENV_FILE) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key, val = key.strip(), val.strip()
            val = val.split(" #")[0].strip()  # strip trailing comments
            if not key or not val:
                continue
            if key not in os.environ:
                os.environ[key] = os.path.expandvars(os.path.expanduser(val))
                exported += 1
    return True, f"{exported} variables exported (existing env wins)"


def _p(name):
    return os.path.expanduser(os.environ.get(name, ""))


# ---------------------------------------------------------------- stage: env

def stage_env():
    print("=== stage env: load local_paths.env + path existence ===")
    ok = load_env()
    if not ok[0]:
        return _fail("local_paths.env missing",
                     "cp scripts/local_paths.env.example scripts/local_paths.env "
                     "and fill in real paths")
    _ok(ok[1])

    all_good = True

    models_root = _p("INGUARD_MODELS_ROOT")
    if os.path.isdir(models_root):
        _ok(f"INGUARD_MODELS_ROOT exists: {models_root}")
        for model, sub in MODEL_DIRS.items():
            d = os.path.join(models_root, sub)
            if os.path.isdir(d):
                _ok(f"  model dir: {sub}")
            else:
                _note(f"model dir missing: {sub} — run `--stage link` to link a "
                      f"local copy, or leave it: the model configs fall back to the "
                      f"HuggingFace repo id (auto-download on first use)")
    else:
        _note(f"INGUARD_MODELS_ROOT not a dir ({models_root!r}) — either link "
              f"local model copies via `--stage link`, or leave it: the model "
              f"configs fall back to the HuggingFace repo id (auto-download)")

    ckpt_root = _p("INGUARD_CKPT_ROOT")
    if os.path.isdir(ckpt_root):
        _ok(f"INGUARD_CKPT_ROOT exists: {ckpt_root}")
    else:
        all_good &= _fail(f"INGUARD_CKPT_ROOT not a dir: {ckpt_root!r}")

    # RevGen CSVs (standard names) under INGUARD_DATA_ROOT
    data_root = _p("INGUARD_DATA_ROOT")
    revgen = os.path.join(data_root, "RevGen")
    if os.path.isdir(revgen):
        for _, std in CSV_LINKS:
            f = os.path.join(revgen, std)
            if os.path.isfile(f):
                _ok(f"RevGen CSV: {std}")
            else:
                all_good &= _fail(f"RevGen CSV missing: {f}",
                                  "run `--stage link` or place the CSVs manually")
    else:
        _note(f"RevGen dir not present yet ({revgen}) — run `--stage link`")

    # weights/ exported by export_weights.py (only checked if populated)
    weights = os.path.join(_REPO_ROOT, "weights")
    if os.path.isfile(os.path.join(weights, "manifest.json")):
        _ok("weights/manifest.json present")
    else:
        all_good &= _fail("weights/manifest.json missing",
                          "run scripts/export_weights.py (or download released weights)")

    # OpenImages (optional, Phase-A only)
    oi = _p("SERVER_OPENIMAGES_ROOT")
    if oi and os.path.isdir(os.path.join(oi, "train")):
        _ok(f"OpenImages root ok: {oi}")
    else:
        _note(f"OpenImages not found at {oi!r} (only needed for Phase-A pretraining)")

    return all_good


# ---------------------------------------------------------------- stage: link

def _find_gen_dir(server_gen_root, model, split, res, steps):
    """Newest matching generated dir for (model, split), version infix optional."""
    model_dir = os.path.join(server_gen_root, model)
    if not os.path.isdir(model_dir):
        return None
    # exact open-source name first, then any version infix (vX-Y, ...)
    patterns = [
        f"{split}-seed42-{res}-{steps}steps",
        f"{split}*v*-seed42-{res}-{steps}steps",
    ]
    for pat in patterns:
        matches = sorted(glob(os.path.join(model_dir, pat)))
        if matches:
            return matches[-1]  # lexicographic max = newest version
    return None


def stage_link():
    print("=== stage link: build RevGen symlink layout (no copying) ===")
    load_env()
    all_good = True

    # ---- CSVs -> $INGUARD_DATA_ROOT/RevGen/<standard name> ----
    revgen = os.path.join(_p("INGUARD_DATA_ROOT"), "RevGen")
    os.makedirs(revgen, exist_ok=True)
    for env_key, std in CSV_LINKS:
        src = _p(env_key)
        dst = os.path.join(revgen, std)
        if os.path.isfile(dst) or os.path.islink(dst):
            _ok(f"already present: RevGen/{std}")
            continue
        if not src or not os.path.isfile(src):
            all_good &= _fail(f"{env_key} not set / not a file: {src!r}")
            continue
        os.symlink(os.path.abspath(src), dst)
        _ok(f"linked RevGen/{std} -> {src}")

    # ---- generated dirs -> $INGUARD_OUTPUT_ROOT/<model>/<std name> ----
    out_root = _p("INGUARD_OUTPUT_ROOT")
    server_gen_root = _p("SERVER_GEN_ROOT")
    if not server_gen_root or not os.path.isdir(server_gen_root):
        all_good &= _fail(f"SERVER_GEN_ROOT not a dir: {server_gen_root!r}")
    else:
        os.makedirs(out_root, exist_ok=True)
        for model, (res, steps) in MODEL_RES_STEPS.items():
            for split in ("trainset", "testset"):
                dst = os.path.join(out_root, model,
                                   f"{split}-seed42-{res}-{steps}steps")
                if os.path.isdir(dst):
                    _ok(f"already present: {model}/{os.path.basename(dst)}")
                    continue
                src = _find_gen_dir(server_gen_root, model, split, res, steps)
                if src is None:
                    _note(f"no generated dir for {model}/{split} "
                          f"(skip if this model is not on this server)")
                    continue
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                os.symlink(os.path.abspath(src), dst)
                _ok(f"linked {model}/{os.path.basename(dst)} -> {src}")

    # ---- T2I model dirs -> $INGUARD_MODELS_ROOT/<standard name> ----
    # The training / data-generation scripts build model paths as
    # f"{INGUARD_MODELS_ROOT}/<standard name>" (see MODEL_REGISTRY), so point
    # INGUARD_MODELS_ROOT at a LOCAL disk and link the standard names to the
    # server-side vendor dirs (no copying of multi-GB checkpoints) — i.e.
    # wherever the downloaded models happen to live.
    models_root = _p("INGUARD_MODELS_ROOT")
    if not models_root:
        all_good &= _fail("INGUARD_MODELS_ROOT not set")
    else:
        os.makedirs(models_root, exist_ok=True)
        for std, env_key in MODEL_LINKS.items():
            dst = os.path.join(models_root, std)
            if os.path.isdir(dst):
                _ok(f"already present: models/{std}")
                continue
            src = _p(env_key)  # full path from local_paths.env
            if not src:
                _note(f"models/{std} not linked — set {env_key} in local_paths.env "
                      f"if this model lives elsewhere (skip if not on this machine)")
                continue
            if not os.path.isdir(src):
                all_good &= _fail(f"{env_key} not a dir: {src!r}")
                continue
            os.symlink(os.path.abspath(src), dst)
            _ok(f"linked models/{std} -> {src}")

        # torch hub cache (latent_detector CONFIGs default TORCH_HOME to
        # f"{INGUARD_MODELS_ROOT}/" -> $TORCH_HOME/hub/checkpoints/*.pth);
        # link an existing cache so IN1K weights do not re-download. Optional.
        hub_dst = os.path.join(models_root, "hub")
        hub_src = _p("SERVER_TORCH_HUB")
        if os.path.isdir(hub_dst):
            _ok("already present: models/hub (torch cache)")
        elif os.path.isdir(hub_src):
            os.symlink(os.path.abspath(hub_src), hub_dst)
            _ok(f"linked models/hub -> {hub_src} (torch cache)")
        else:
            _note(f"no torch cache at {hub_src!r} — IN1K weights will download")

    # InternVL-U pipeline code dir: expected at {_MODELS_ROOT}/../InternVL-U-main
    # (the historical code_dir layout of the data-generation scripts). OPTIONAL —
    # the repo ships its own port (sage/backends/internvlu) which every script
    # falls back to.
    code_dst = os.path.join(os.path.dirname(models_root), "InternVL-U-main")
    code_src = _p("SERVER_INTERNVLU_CODE")
    if os.path.isdir(code_dst):
        _ok("already present: ../InternVL-U-main (pipeline code)")
    elif code_src and os.path.isdir(code_src):
        os.symlink(os.path.abspath(code_src), code_dst)
        _ok(f"linked ../InternVL-U-main -> {code_src} (pipeline code)")
    else:
        _note(f"InternVL-U-main not at {code_src!r} (only needed for internvl-u)")

    # ---- report predictions.csv availability (latent-detector training label)
    for model in MODEL_RES_STEPS:
        res, steps = MODEL_RES_STEPS[model]
        for split in ("trainset", "testset"):
            d = os.path.join(out_root, model,
                             f"{split}-seed42-{res}-{steps}steps",
                             "labels_llm", "predictions.csv")
            if os.path.isfile(d):
                _ok(f"labels ready: {model}/{split} predictions.csv")

    if all_good:
        _note(f"symlink layout root: {out_root} (INGUARD_OUTPUT_ROOT)")
    return all_good


# ---------------------------------------------------------------- stage: tests

def stage_tests():
    print("=== stage tests: unit tests (CPU) ===")
    r = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s",
                        os.path.join(_REPO_ROOT, "tests")],
                       cwd=_REPO_ROOT)
    return r.returncode == 0


# ---------------------------------------------------------------- stage: verify

def stage_verify():
    print("=== stage verify: verify_pipeline quick + components ===")
    load_env()
    all_good = True
    for st in ("quick", "components"):
        r = subprocess.run(
            [sys.executable, os.path.join(_REPO_ROOT, "scripts", "verify_pipeline.py"),
             "--model", SMOKE_MODEL, "--stage", st, "--device", _p("INGUARD_DEVICE")],
            cwd=_REPO_ROOT)
        all_good &= (r.returncode == 0)
    return all_good


# ---------------------------------------------------------------- stage: pe

def _truncate_csv(src, dst, n_rows):
    """Copy the first n_rows data rows (header kept), BOM stripped."""
    import csv
    with open(src, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))
    with open(dst, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows[: n_rows + 1])
    return len(rows) - 1


def stage_pe(pe_rows):
    print(f"=== stage pe: PE-MLP smoke training ({pe_rows} rows, 1 epoch) ===")
    load_env()
    all_good = True

    revgen = os.path.join(_p("INGUARD_DATA_ROOT"), "RevGen")
    train_labeled = os.path.join(revgen, "trainset_labeled.csv")
    test_labeled = os.path.join(revgen, "testset_labeled.csv")
    for name, f in (("trainset_labeled.csv", train_labeled),
                    ("testset_labeled.csv", test_labeled)):
        if not os.path.isfile(f):
            return _fail(f"{name} missing under {revgen}",
                         "run `--stage link` first, or place the CSVs manually")

    tmp = tempfile.mkdtemp(prefix="inguard_smoke_")
    try:
        small_train = os.path.join(tmp, "trainset_labeled.csv")
        small_test = os.path.join(tmp, "testset_labeled.csv")
        n1 = _truncate_csv(train_labeled, small_train, pe_rows)
        n2 = _truncate_csv(test_labeled, small_test, pe_rows)
        _ok(f"truncated CSVs: train {n1} rows, test {n2} rows -> {tmp}")

        cmd = [sys.executable, os.path.join(_REPO_ROOT, "pe_mlp", "train_prompt.py"),
               "--model_name", SMOKE_MODEL,
               "--train_csv", small_train,
               "--test_csv", small_test,
               "--epochs", "1"]
        _note("running: " + " ".join(cmd))
        r = subprocess.run(cmd, cwd=_REPO_ROOT)
        if r.returncode != 0:
            return _fail("train_prompt.py exited non-zero")
        _ok("PE-MLP smoke training finished")

        # confirm a run dir + meta/test_predictions.csv appeared
        ckpt_root = _p("INGUARD_CKPT_ROOT")
        hits = sorted(glob(os.path.join(ckpt_root, "*", SMOKE_MODEL,
                                        "prompt-mlp-*", "*")))
        _note(f"check output under {ckpt_root}/<date>/{SMOKE_MODEL}/"
              f"prompt-mlp-*/  (newest: {hits[-1] if hits else 'none found'})")
        return True
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", default="all",
                    choices=["env", "link", "tests", "verify", "pe", "all"])
    ap.add_argument("--pe_rows", type=int, default=200,
                    help="rows kept in the truncated CSVs for --stage pe")
    args = ap.parse_args()

    # "all" re-runs env after link: the first pass reports the RevGen/
    # models layout as missing (it is created by link), the re-run is the
    # verdict that counts (results dict keeps the last value per stage).
    stages = (["env", "link", "env", "tests", "verify", "pe"]
              if args.stage == "all" else [args.stage])
    results = {}
    for st in stages:
        if st == "env":
            results[st] = stage_env()
        elif st == "link":
            results[st] = stage_link()
        elif st == "tests":
            results[st] = stage_tests()
        elif st == "verify":
            results[st] = stage_verify()
        elif st == "pe":
            results[st] = stage_pe(args.pe_rows)
        print()

    print("=" * 60)
    for st, ok in results.items():
        print(f"  {st:8s} : {'PASS' if ok else 'FAIL'}")
    print("=" * 60)
    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
