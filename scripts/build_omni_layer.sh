#!/usr/bin/env bash
# build_omni_layer.sh — turn a built plain-vLLM ROCm bundle into a vLLM-Omni bundle.
#
# vLLM-Omni is a pure-Python layer (one py3-none-any wheel) that rides on the
# base vLLM + torch + Triton already in the bundle, so the omni artifact is the
# SAME relocatable bundle plus: vllm-omni, its pure-Python runtime deps, an
# ABI-matched torchaudio, the multimodal deps the base trim drops
# (timm / opencv / peft), and a dedicated `vllm-omni-server` launcher.
#
# This is intentionally a thin layer invoked AFTER the base "Install vLLM +
# PyTorch" step in build-vllm-rocm.yml — it does not rebuild the base. Keeping
# the lean LLM bundle and the heavier omni bundle as separate release artifacts
# is deliberate (see README): omni pulls ~1GB of deps and pins Python <3.14
# that plain-LLM users should not carry.
#
# Required env (exported by the workflow's earlier steps):
#   VLLM_ROOT   — bundle root (contains bin/, lib/python3.X/site-packages)
#   TORCH_INDEX — the same ROCm torch index the base install used (for a
#                 torchaudio built against the bundled torch's ABI)
# Optional env:
#   VLLM_OMNI_VER — vllm-omni version to install. Default: auto-match the base
#                   vLLM major.minor (vllm-omni REQUIRES the same major.minor,
#                   or its import-time patch layer misbehaves).
#
# Note on fa3-fwd: it is a declared vllm-omni dependency (flash-attn-3 forward,
# used only by the diffusion-attention path for Cosmos3 / image+video models).
# It installs cleanly here — it ships a cp39-abi3 manylinux x86_64 wheel, so it
# is NOT cp314-blocked — but whether its kernels run on ROCm is unverified; AR
# omni models (Qwen-Omni) never invoke it.
set -euo pipefail

: "${VLLM_ROOT:?VLLM_ROOT must point at the built bundle}"
PYBIN="$(ls "$VLLM_ROOT"/bin/python3.1? | head -1)"
SP="$(echo "$VLLM_ROOT"/lib/python3.*/site-packages)"
export PYTHONNOUSERSITE=1

echo "== omni layer: bundle=$VLLM_ROOT python=$("$PYBIN" --version)"

# --- pip is trimmed from the bundle; bootstrap if absent --------------------
if ! "$PYBIN" -m pip --version >/dev/null 2>&1; then
  echo "== bootstrapping pip (trimmed from base bundle)"
  curl -fsSL -o /tmp/get-pip.py https://bootstrap.pypa.io/get-pip.py
  "$PYBIN" /tmp/get-pip.py
fi

# --- resolve versions --------------------------------------------------------
BASE_VLLM="$("$PYBIN" -c 'import importlib.metadata as m; print(m.version("vllm"))')"
BASE_MM="$(echo "$BASE_VLLM" | sed -E 's/^([0-9]+\.[0-9]+).*/\1/')"          # e.g. 0.23
TORCH_VER="$("$PYBIN" -c 'import importlib.metadata as m; print(m.version("torch"))')"
: "${VLLM_OMNI_VER:=$("$PYBIN" - <<PY
import json,urllib.request
base="${BASE_MM}"
d=json.load(urllib.request.urlopen("https://pypi.org/pypi/vllm-omni/json"))
# newest release whose major.minor == base vLLM major.minor (incl. rc)
cands=[v for v in d["releases"] if v.split("rc")[0].rsplit(".",1)[0]==base]
print(sorted(cands)[-1] if cands else "")
PY
)}"
[ -n "$VLLM_OMNI_VER" ] || { echo "::error::no vllm-omni release matches base vLLM $BASE_MM"; exit 1; }
echo "== base vLLM=$BASE_VLLM (mm=$BASE_MM) torch=$TORCH_VER -> vllm-omni==$VLLM_OMNI_VER"

# Protect the GPU-coupled stack so dep resolution can never swap the ROCm
# torch/vllm/triton/transformers for a PyPI build. safetensors/accelerate are
# left to float (diffusers 0.38 needs safetensors>=0.8).
CONSTRAINTS=/tmp/omni-constraints.txt
"$PYBIN" - > "$CONSTRAINTS" <<'PY'
import importlib.metadata as m
for p in ("torch","vllm","pytorch-triton-rocm","triton","torchvision","torchaudio","transformers","numpy"):
    try: print(f"{p}=={m.version(p)}")
    except Exception: pass
PY

# --- ABI-matched torchaudio (omni api_server imports it eagerly) ------------
# Not in the base bundle. Pull the build that matches the bundled torch
# version+stamp from the same ROCm index the base used.
if ! "$PYBIN" -c 'import torchaudio' >/dev/null 2>&1; then
  echo "== installing torchaudio==$TORCH_VER from $TORCH_INDEX"
  "$PYBIN" -m pip install -c "$CONSTRAINTS" \
    --index-url "${TORCH_INDEX:?set TORCH_INDEX to the base torch ROCm index}" \
    --extra-index-url https://pypi.org/simple/ \
    "torchaudio==$TORCH_VER"
fi

# --- vllm-omni + runtime deps ------------------------------------------------
# Two-step install so --ignore-requires-python is scoped to vllm-omni ONLY.
#
# 1. vllm-omni itself: every release pins Python <3.14, but the qualified nightly
#    base is cp314. The cap is conservative (validated to run on 3.14) and this
#    is a bundle we control + qualify, so we override it with
#    --ignore-requires-python + --no-deps.
# 2. Its runtime deps (+ the multimodal extras the base trim drops), installed
#    WITHOUT that flag. Passed to the whole resolve, the flag strips the
#    Python-version filter from every transitive dep too, so pip picked
#    numba 0.62 (no cp314 wheel -> sdist -> setup.py refuses 3.14) over
#    numba 0.63 (has a cp314 wheel). Scoping it here is the same pattern the
#    base install uses for amd-quark.
echo "== installing vllm-omni==$VLLM_OMNI_VER (no-deps)"
"$PYBIN" -m pip install --no-deps --ignore-requires-python "vllm-omni==$VLLM_OMNI_VER"

# Read vllm-omni's core (non-extra) runtime deps from its metadata so this tracks
# the installed version instead of a hand-maintained list.
mapfile -t OMNI_REQS < <("$PYBIN" - "$SP" <<'PY'
import glob, os, sys
sp = sys.argv[1]
metas = glob.glob(os.path.join(sp, "vllm_omni-*.dist-info", "METADATA"))
if not metas:
    sys.exit("vllm_omni dist-info not found after install")
for line in open(metas[0], encoding="utf-8", errors="replace"):
    if not line.startswith("Requires-Dist:"):
        continue
    spec = line.split(":", 1)[1].strip()
    if "extra ==" in spec:                  # skip optional extras (dev/demo/…)
        continue
    spec = spec.split(";", 1)[0].strip()    # drop any environment marker
    if spec:
        print(spec)
PY
)
[ "${#OMNI_REQS[@]}" -gt 0 ] || { echo "::error::no vllm-omni runtime deps parsed"; exit 1; }

# Multimodal deps the base bundle trims but omni model loading needs.
OMNI_EXTRAS=(timm opencv-python-headless peft)

echo "== installing ${#OMNI_REQS[@]} vllm-omni deps + ${#OMNI_EXTRAS[@]} multimodal extras"
"$PYBIN" -m pip install -c "$CONSTRAINTS" "${OMNI_REQS[@]}" "${OMNI_EXTRAS[@]}"

# --- re-strip pip (we bootstrapped it above) to match the base bundle --------
# The base Strip step removes pip; we re-added it only to install this layer.
# Remove it again so the omni bundle stays lean and passes qualification T0.4
# (which flags an SP/pip dir as bundle bloat). Deps are installed; pip is done.
rm -rf "$SP"/pip "$SP"/pip-*.dist-info "$SP"/wheel "$SP"/wheel-*.dist-info 2>/dev/null || true

# --- omni launcher -----------------------------------------------------------
# Mirrors bin/vllm-server's env (LD_LIBRARY_PATH/PYTHONPATH/clang) but execs the
# vllm-omni CLI. The bundled bin/vllm has a hardcoded #!/opt/vllm shebang that
# breaks under relocation, so we always go through `python3 -m`.
cat > "$VLLM_ROOT/bin/vllm-omni-server" <<'LAUNCHER_EOF'
#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$(dirname "$SCRIPT_DIR")"
SP="$(echo "$VENV_DIR"/lib/python3.*/site-packages)"
for libdir in "$SP"/_rocm_sdk_*/lib "$SP"/torch/lib; do
  [ -d "$libdir" ] && LD_LIBRARY_PATH="${libdir}:${LD_LIBRARY_PATH:-}"
done
LD_LIBRARY_PATH="$SP/_rocm_sdk_core/lib/llvm/lib:${LD_LIBRARY_PATH}"
export LD_LIBRARY_PATH
export PYTHONPATH="$SP/_rocm_sdk_core/share/amd_smi:${PYTHONPATH:-}"
export FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
CLANG="$SP/_rocm_sdk_core/lib/llvm/bin/clang"
[ -f "$CLANG" ] && chmod +x "$CLANG" "$CLANG"-* 2>/dev/null
export CC="$CLANG"
exec "$SCRIPT_DIR/python3" -m vllm_omni.entrypoints.cli.main "$@"
LAUNCHER_EOF
chmod +x "$VLLM_ROOT/bin/vllm-omni-server"

# --- verify the layer imports under the launcher env ------------------------
echo "== verifying vllm_omni import"
LD_LIBRARY_PATH="$SP/_rocm_sdk_core/lib/llvm/lib" \
PYTHONPATH="$SP/_rocm_sdk_core/share/amd_smi" \
  "$VLLM_ROOT/bin/vllm-omni-server" --help >/dev/null 2>&1 \
  || { echo "::error::vllm-omni-server failed to import"; exit 1; }

echo "== omni layer complete: vllm-omni==$VLLM_OMNI_VER on vLLM $BASE_VLLM"
