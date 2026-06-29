# RESOLVED — `hipErrorInvalidImage` on gfx1151 was OUR CI dropping hidden `.kpack` code objects

**Status:** Fixed in `build-vllm-rocm.yml` (`include-hidden-files: true` on the
build-job `upload-artifact`). This was **not** an upstream AMD bug, not the
runner/box, not the OS kernel, and **not** a vLLM 0.23.1 / cp314 regression.
Earlier drafts of this file blamed AMD — that conclusion was wrong and has been
retracted.

## What actually happened

AMD's ROCm `torch` / `rocm-sdk` wheels ship the per-GPU device code objects in
**hidden dot-directories**:

- `torch/.kpack/torch_gfx1151.kpack` (~50 MB — torch's own gfx1151 kernels)
- `_rocm_sdk_libraries/.kpack/{blas,fft,rand,rccl}_lib_gfx1151.kpack`
- `_rocm_sdk_libraries/.devel_links/gfx1151.json`

`actions/upload-artifact@v4` **excludes hidden files by default** (it only emits
a warning). The build job uploaded `path: $VLLM_ROOT/` without
`include-hidden-files: true`, so every `.kpack/` directory was silently dropped
from the artifact. Everything downstream (the qualify job's `download-artifact`,
and the GitHub release tarball) inherited a bundle with **no gfx1151 device
images**.

The host-side `libtorch_hip.so` is unaffected (byte-identical md5 to a faithful
install), so the bundle still:

- imports torch and vLLM,
- enumerates the GPU (`torch.cuda.get_device_name()` → "AMD Radeon 8060S"),

…but dies on the **first GPU kernel launch** (`torch.zeros` →
`vllm/v1/utils.py:125`) with:

```
torch.AcceleratorError: CUDA error: device kernel image is invalid  (hipErrorInvalidImage)
```

because there is no gfx1151 code object for the loader to bind.

## How it was isolated (two gfx1151 boxes)

On a clean gfx1151 box (`/home/user/work/repro`):

| Environment | `torch.zeros` on cuda | vLLM `generate` |
|---|---|---|
| Faithful `pip install torch[device-gfx1151]` + `amd-torch-device-gfx1151` (clean venv) | ✅ | ✅ |
| Downloaded runner bundle (cp314 / 0.23.1 / a20260624) | ❌ `hipErrorInvalidImage` | ❌ |
| Same bundle, `torch/.kpack/torch_gfx1151.kpack` + 4 library kpacks restored | ✅ | ✅ "Paris" |

Bisection: `torch/`, `_rocm_sdk_core/`, and `_rocm_sdk_libraries/` are
byte-identical between the failing bundle and the working clean install **except**
the missing `.kpack` / `.devel_links` dot-dirs. Restoring only those makes the
exact same wheel set serve inference on the exact same hardware.

This is precisely CLAUDE.md inviolable principle #3: *"Size-trimming must never
delete a file required to load or execute."* The Strip step in the workflow never
touched `.kpack`; the deletion was entirely `upload-artifact`'s hidden-file
exclusion.

## Fix

```yaml
- name: Upload build artifacts
  uses: actions/upload-artifact@v4
  with:
    name: vllm-ubuntu-rocm-${{ matrix.gfx_target }}-x64
    path: ${{ env.VLLM_ROOT }}/
    include-hidden-files: true   # keep the .kpack gfx1151 code objects
    ...
```

## Guard against regression

Qualification should fail loudly if the gfx code objects are absent rather than
only catching it at tier2 inference. Cheap tier0/tier1 check to add: assert that
`torch/.kpack/torch_*.kpack` and `_rocm_sdk_libraries/.kpack/*_<gfx>.kpack` exist
in the downloaded bundle (and are non-empty) before hardware tests run.

## Separate, unrelated finding (still valid)

The dev_lab runner's earlier `6.17.0-1023-oem` kernel genuinely failed even the
known-good cp313/0.21.1 set; updating to `6.17.0-22-generic` fixed that. That was
a real kernel issue, distinct from this artifact bug.
