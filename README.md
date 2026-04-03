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
<a href="https://github.com/ROCm/ROCm" title="Powered by ROCm 7.0">
  <img src="https://img.shields.io/badge/ROCm-7.0-blue?logo=amd&logoColor=white" alt="ROCm 7.0" />
</a>
<a href="https://github.com/vllm-project/vllm" title="Powered by vLLM">
  <img src="https://img.shields.io/badge/Powered%20by-vLLM-blue" alt="Powered by vLLM" />
</a>
<a href="#-supported-devices" title="Platform support">
  <img src="https://img.shields.io/badge/OS-Ubuntu-0078D6?logo=ubuntu&logoColor=white" alt="Platform: Ubuntu" />
</a>

We provide portable builds of **vLLM** with **AMD ROCm 7** acceleration based on TheRock. Each release is a self-contained archive containing a bundled Python environment, vLLM, PyTorch ROCm, and all required ROCm runtime libraries. Our automated pipeline targets integration with [**Lemonade**](https://github.com/lemonade-sdk/lemonade).

> [!IMPORTANT]
> **Early Development**: This project is in active development. ROCm support for consumer AMD GPUs (RDNA) in vLLM is experimental. We welcome issue reports and contributions.

## Supported Devices

| GPU Target | Architecture | Devices |
|------------|-------------|---------|
| **gfx1151** | STX Halo APU | Ryzen AI MAX+ Pro 395 |
| **gfx1150** | STX Point APU | Ryzen AI 300 |
| **gfx120X** | RDNA4 GPUs | RX 9070 XT, RX 9070, RX 9060 XT, RX 9060 |
| **gfx110X** | RDNA3 GPUs | RX 7900 XTX/XT/GRE, RX 7800 XT, RX 7700 XT, RX 7600 XT/7600 |

**All builds include ROCm 7 runtime built-in** — no separate ROCm installation required!

## Quick Start

1. **Download** the build for your GPU from the [latest release](https://github.com/lemonade-sdk/vllm-rocm/releases/latest)
2. **Extract** the archive:
   ```bash
   tar xzf vllm-b1000-ubuntu-rocm-gfx1151-x64.tar.gz -C ~/vllm-rocm
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

Each release archive contains a complete, portable environment:

```
bin/
  vllm-server     # Launcher script (entry point)
  python3.11      # Bundled Python interpreter
lib/
  libamdhip64.so  # ROCm runtime (HIP)
  librocblas.so   # ROCm BLAS
  libhipblas.so   # HIP BLAS
  ...             # All required ROCm shared libraries
  rocblas/library/ # rocBLAS kernel files
  python3.11/site-packages/
    vllm/          # vLLM package
    torch/         # PyTorch ROCm
    ...            # All Python dependencies
```

No external Python, PyTorch, or ROCm installation is needed.

## Automated Builds

Our GitHub Actions workflow:
- Downloads the latest **ROCm 7 nightly** from TheRock
- Installs **PyTorch ROCm** from the official pip index
- Builds **vLLM from source** with architecture-specific HIP kernels
- Bundles everything with `patchelf --set-rpath` for portability
- Tests on self-hosted AMD GPU hardware before releasing

| GPU Target | Ubuntu |
|------------|--------|
| **gfx1151** | [![Download](https://img.shields.io/badge/Download-Ubuntu%20gfx1151-blue)](https://github.com/lemonade-sdk/vllm-rocm/releases/latest) |
| **gfx1150** | [![Download](https://img.shields.io/badge/Download-Ubuntu%20gfx1150-blue)](https://github.com/lemonade-sdk/vllm-rocm/releases/latest) |
| **gfx120X** | [![Download](https://img.shields.io/badge/Download-Ubuntu%20gfx120X-blue)](https://github.com/lemonade-sdk/vllm-rocm/releases/latest) |
| **gfx110X** | [![Download](https://img.shields.io/badge/Download-Ubuntu%20gfx110X-blue)](https://github.com/lemonade-sdk/vllm-rocm/releases/latest) |

> **Linux (gfx1150/APU):** OOM despite free VRAM? Add `ttm.pages_limit=12582912` (48 GB) to the kernel cmdline (e.g. GRUB), run `update-grub`, then reboot. See [TheRock FAQ](https://github.com/ROCm/TheRock/blob/main/docs/faq.md#gfx1151-strix-halo-specific-questions).

## Dependencies

### Runtime
- **[vLLM](https://github.com/vllm-project/vllm)** — High-throughput LLM serving engine
- **[PyTorch](https://pytorch.org/)** — Tensor computation framework (ROCm build)
- **[ROCm (TheRock)](https://github.com/ROCm/TheRock)** — AMD GPU compute platform

### Build (CI only)
- **Ubuntu 22.04** GitHub Actions runner
- **Python 3.11** from deadsnakes PPA
- **CMake**, **Ninja**, **patchelf**

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
