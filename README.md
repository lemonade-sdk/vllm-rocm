# vllm-rocm

<a href="https://github.com/lemonade-sdk/vllm-rocm/releases/latest" title="Download the latest release">
  <img src="https://img.shields.io/github/v/release/lemonade-sdk/vllm-rocm?logo=github&logoColor=white" alt="GitHub release (latest by date)" />
</a>
<a href="https://github.com/lemonade-sdk/vllm-rocm/releases/latest" title="View latest release date">
  <img src="https://img.shields.io/github/release-date/lemonade-sdk/vllm-rocm?logo=github&logoColor=white" alt="Latest release date" />
</a>
<a href="LICENSE" title="View license">
  <img src="https://img.shields.io/github/license/lemonade-sdk/vllm-rocm?logo=opensourceinitiative&logoColor=white" alt="License" />
</a>
<a href="https://github.com/ROCm/ROCm" title="Powered by ROCm 7.12">
  <img src="https://img.shields.io/badge/ROCm-7.12-blue?logo=amd&logoColor=white" alt="ROCm 7.12" />
</a>
<a href="https://github.com/vllm-project/vllm" title="Powered by vLLM">
  <img src="https://img.shields.io/badge/Powered%20by-vLLM-blue" alt="Powered by vLLM" />
</a>
<a href="#-supported-devices" title="Platform support">
  <img src="https://img.shields.io/badge/OS-Ubuntu-0078D6?logo=ubuntu&logoColor=white" alt="Platform: Ubuntu" />
</a>

We provide portable builds of **vLLM** with **AMD ROCm 7.12** acceleration. Each release is a self-contained archive that bundles a relocatable CPython interpreter, vLLM, PyTorch, and all required ROCm user-space libraries as pip packages — no system Python, PyTorch, or ROCm install required. Our automated pipeline targets integration with [**Lemonade**](https://github.com/lemonade-sdk/lemonade).

> [!IMPORTANT]
> **Early Development**: This project is in active development. ROCm support for consumer AMD GPUs (RDNA) in vLLM is experimental. We welcome issue reports and contributions.

## Supported Devices

| GPU Target | Architecture | Devices |
|------------|-------------|---------|
| **gfx1151** | STX Halo APU | Ryzen AI MAX+ Pro 395 |
| **gfx1150** | STX Point APU | Ryzen AI 300 |
| **gfx120X** | RDNA4 GPUs | RX 9070 XT, RX 9070, RX 9060 XT, RX 9060 |
| **gfx110X** | RDNA3 GPUs | RX 7900 XTX/XT/GRE, RX 7800 XT, RX 7700 XT, RX 7600 XT/7600 |

**All builds include ROCm 7.12 user-space built-in** — no separate ROCm installation required. You still need a Linux kernel with a working amdgpu driver for your GPU; for gfx1151 specifically this means kernel 6.18.4+ (see [Lemonade's gfx1151 notes](https://lemonade-server.ai/gfx1151_linux.html)).

## Quick Start

1. **Download** both parts of the build for your GPU from the [latest release](https://github.com/lemonade-sdk/vllm-rocm/releases/latest). Releases are split into `.part00.tar.gz` + `.part01.tar.gz` because each build exceeds GitHub's 2 GB per-asset limit.
2. **Extract** the archive (concatenate the parts and pipe into tar):
   ```bash
   mkdir -p ~/vllm-rocm
   cat vllm0.19.0-rocm7.12.0-gfx1151-x64.part00.tar.gz \
       vllm0.19.0-rocm7.12.0-gfx1151-x64.part01.tar.gz \
     | tar xz -C ~/vllm-rocm
   ```
3. **Run** the server:
   ```bash
   ~/vllm-rocm/bin/vllm-server --model meta-llama/Llama-3.2-1B --port 8000
   ```
4. **Test** with curl:
   ```bash
   curl http://localhost:8000/v1/completions \
     -H "Content-Type: application/json" \
     -d '{"model": "meta-llama/Llama-3.2-1B", "prompt": "Hello", "max_tokens": 50}'
   ```

> **Lemonade Integration**: These builds are designed to work as a backend for [**Lemonade**](https://github.com/lemonade-sdk/lemonade), which manages downloading, launching, and routing requests to vLLM automatically.

## What's Included

Each release archive extracts to a relocatable CPython 3.12 distribution with all deps pre-installed into `site-packages`:

```
bin/
  vllm-server                 # Launcher shim (sets LD_LIBRARY_PATH, execs api_server)
  python3.12                  # Bundled CPython interpreter (python-build-standalone)
lib/
  libpython3.12.so            # Python runtime
  python3.12/
    site-packages/
      vllm/                   # pip-installed from wheels.vllm.ai/rocm/
      torch/                  # pip-installed from repo.amd.com/rocm/whl/<arch>/
      _rocm_sdk_core/lib/     # ROCm core user-space (hip, hsa, comgr, clang, llvm)
      _rocm_sdk_libraries_gfx<arch>/lib/
                              # Per-arch ROCm math libs (rocblas, hipblas, rccl, MIOpen, ...)
      transformers/, numpy/, ...  # Python deps
```

The top-level `lib/` holds the Python stdlib and `libpython3.12.so`; ROCm libraries (e.g. `libamdhip64.so`, `librocblas.so`) live under the bundled site-packages. The `bin/vllm-server` shim puts those directories on `LD_LIBRARY_PATH` before exec-ing `python3 -m vllm.entrypoints.openai.api_server`.

## Automated Builds

Our GitHub Actions workflow:
- Downloads a relocatable **CPython 3.12** from [`astral-sh/python-build-standalone`](https://github.com/astral-sh/python-build-standalone)
- Installs **PyTorch ROCm** from AMD's pip index (`https://repo.amd.com/rocm/whl/<target>/`)
- Installs **vLLM ROCm** (pre-built wheel) from AMD's vLLM wheel index (`https://wheels.vllm.ai/rocm/`), which pulls the matching `rocm-sdk-core` and `rocm-sdk-libraries-gfx<target>` wheels as transitive deps
- Generates a `bin/vllm-server` shim that wires up `LD_LIBRARY_PATH` / `PYTHONPATH` at startup
- Tars the result, splits it into `< 2 GB` parts, and tests on self-hosted AMD GPU hardware before releasing

| GPU Target | Ubuntu |
|------------|--------|
| **gfx1151** | [![Download](https://img.shields.io/badge/Download-Ubuntu%20gfx1151-blue)](https://github.com/lemonade-sdk/vllm-rocm/releases/latest) |
| **gfx1150** | [![Download](https://img.shields.io/badge/Download-Ubuntu%20gfx1150-blue)](https://github.com/lemonade-sdk/vllm-rocm/releases/latest) |
| **gfx120X** | [![Download](https://img.shields.io/badge/Download-Ubuntu%20gfx120X-blue)](https://github.com/lemonade-sdk/vllm-rocm/releases/latest) |
| **gfx110X** | [![Download](https://img.shields.io/badge/Download-Ubuntu%20gfx110X-blue)](https://github.com/lemonade-sdk/vllm-rocm/releases/latest) |

> **Linux (gfx1150/APU):** OOM despite free VRAM? Add `ttm.pages_limit=12582912` (48 GB) to the kernel cmdline (e.g. GRUB), run `update-grub`, then reboot. See [TheRock FAQ](https://github.com/ROCm/TheRock/blob/main/docs/faq.md#gfx1151-strix-halo-specific-questions).

## Dependencies

### Runtime (bundled in the release)
- **[vLLM](https://github.com/vllm-project/vllm)** — high-throughput LLM serving engine (ROCm wheel from `wheels.vllm.ai/rocm/`)
- **[PyTorch](https://pytorch.org/)** — tensor compute (ROCm wheel from `repo.amd.com/rocm/whl/<target>/`)
- **[ROCm SDK wheels](https://github.com/ROCm/TheRock)** — AMD's pip-packaged ROCm user-space (`rocm-sdk-core`, `rocm-sdk-libraries-gfx<target>`, published alongside via [TheRock](https://github.com/ROCm/TheRock))
- **[python-build-standalone](https://github.com/astral-sh/python-build-standalone)** — relocatable CPython 3.12

### Build (CI only)
- **Ubuntu 22.04** GitHub Actions runner
- `pip` (no `cmake`, `ninja`, or `patchelf` involved — everything comes from pre-built wheels)

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
