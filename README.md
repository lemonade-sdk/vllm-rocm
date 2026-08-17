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
<a href="https://github.com/ROCm/ROCm" title="Powered by ROCm 7.13">
  <img src="https://img.shields.io/badge/ROCm-7.13-blue?logo=amd&logoColor=white" alt="ROCm 7.13" />
</a>
<a href="https://github.com/vllm-project/vllm" title="Powered by vLLM">
  <img src="https://img.shields.io/badge/Powered%20by-vLLM-blue" alt="Powered by vLLM" />
</a>
<a href="#-supported-devices" title="Platform support">
  <img src="https://img.shields.io/badge/OS-Ubuntu-0078D6?logo=ubuntu&logoColor=white" alt="Platform: Ubuntu" />
</a>

We provide portable builds of **vLLM** with **AMD ROCm 7.13** acceleration (the stable channel; the nightly channel tracks newer ROCm). Each release is a self-contained archive that bundles a relocatable CPython interpreter, vLLM, PyTorch, and all required ROCm user-space libraries as pip packages — no system Python, PyTorch, or ROCm install required. Our automated pipeline targets integration with [**Lemonade**](https://github.com/lemonade-sdk/lemonade).

> [!IMPORTANT]
> **Early Development**: This project is in active development. ROCm support for consumer AMD GPUs (RDNA) in vLLM is experimental. We welcome issue reports and contributions.

## Supported Devices

| GPU Target | Architecture | Devices |
|------------|-------------|---------|
| **gfx1151** | STX Halo APU | Ryzen AI MAX+ Pro 395 |
| **gfx1150** | STX Point APU | Ryzen AI 300 |
| **gfx120X** | RDNA4 GPUs | RX 9070 XT, RX 9070, RX 9060 XT, RX 9060 |
| **gfx110X** | RDNA3 GPUs | RX 7900 XTX/XT/GRE, RX 7800 XT, RX 7700 XT, RX 7600 XT/7600 |
| **gfx942** † | CDNA3 (Instinct) | MI300X, MI300A, MI325X |
| **gfx950** † | CDNA4 (Instinct) | MI350X, MI355X |

> **Data-center CDNA targets (gfx942 / gfx950) are built on demand only.** They are
> not in the default/scheduled build matrix — they are produced solely by a manual
> `workflow_dispatch` of **Build vLLM + ROCm** with the target listed in `gfx_target`.
> They are also **not hardware-qualified in this repo** (only gfx1151 has a self-hosted
> GPU runner), so CDNA builds are build-verified but released without a hardware
> qualification.

**All builds include the ROCm user-space built-in** — no separate ROCm installation required. You still need a Linux kernel with a working amdgpu driver for your GPU; for gfx1151 specifically this means kernel 6.18.4+ (see [Lemonade's gfx1151 notes](https://lemonade-server.ai/gfx1151_linux.html)).

## Quick Start

1. **Download** both parts of the build for your GPU from the [latest release](https://github.com/lemonade-sdk/vllm-rocm/releases/latest). Releases are split into `.part01-of-02.tar.gz` + `.part02-of-02.tar.gz` because each build exceeds GitHub's 2 GB per-asset limit.
2. **Extract** the archive (concatenate the parts and pipe into tar):
   ```bash
   mkdir -p ~/vllm-rocm
   cat vllm0.22.1-rocm7.13.0-gfx1151-x64.part01-of-02.tar.gz \
       vllm0.22.1-rocm7.13.0-gfx1151-x64.part02-of-02.tar.gz \
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

Each release archive extracts to a relocatable CPython distribution — CPython 3.13 on the stable channel, 3.14 on nightly — with all deps pre-installed into `site-packages` (the tree below shows a 3.13 stable bundle):

```
bin/
  vllm-server                 # Launcher shim (sets LD_LIBRARY_PATH, execs api_server)
  python3.13                  # Bundled CPython interpreter (python-build-standalone)
lib/
  libpython3.13.so            # Python runtime
  python3.13/
    site-packages/
      vllm/                   # pip-installed from AMD's vLLM index (rocm.frameworks.amd.com/whl/<suffix>/)
      torch/                  # pip-installed from repo.amd.com/rocm/whl/<suffix>/
      _rocm_sdk_core/lib/     # ROCm core user-space (hip, hsa, comgr, clang, llvm)
      _rocm_sdk_libraries_gfx<arch>/lib/
                              # Per-arch ROCm math libs (rocblas, hipblas, rccl, MIOpen, ...)
      transformers/, numpy/, ...  # Python deps
```

The top-level `lib/` holds the Python stdlib and `libpython3.NN.so`; ROCm libraries (e.g. `libamdhip64.so`, `librocblas.so`) live under the bundled site-packages. The `bin/vllm-server` shim puts those directories on `LD_LIBRARY_PATH` before exec-ing `python3 -m vllm.entrypoints.openai.api_server`.

## Automated Builds

Our GitHub Actions workflow:
- Downloads a relocatable **CPython** from [`astral-sh/python-build-standalone`](https://github.com/astral-sh/python-build-standalone) (3.13 for the stable channel, 3.14 for nightly)
- Installs **PyTorch ROCm** from AMD's pip index (`https://repo.amd.com/rocm/whl/<suffix>/` for stable; `https://rocm.nightlies.amd.com/whl-multi-arch` for nightly)
- Installs **vLLM ROCm** (pre-built wheel) from AMD's vLLM wheel index (`https://rocm.frameworks.amd.com/whl/<suffix>/` for stable; `https://rocm.frameworks-nightlies.amd.com/whl/device-all-rdna` — or `device-all-cdna` for gfx942/gfx950 — for nightly), which pulls the matching `rocm-sdk-core` and `rocm-sdk-libraries-gfx<arch>` wheels as transitive deps
  - These are pip **index** URLs (passed to `--index-url`/`--extra-index-url`); pip appends `/vllm/` and `/torch/` itself. On stable, `<suffix>` is AMD's per-target aisle, which is *not* always the bare gfx target: `gfx110X`→`gfx110X-all`, `gfx120X`→`gfx120X-all`, `gfx942`→`gfx94X-dcgpu`, `gfx950`→`gfx950-dcgpu` (`gfx1151`/`gfx1150` map verbatim). Nightly instead uses AMD's single universal-family wheel (`device-all-rdna` / `device-all-cdna`), not a per-target aisle.
- Generates a `bin/vllm-server` shim that wires up `LD_LIBRARY_PATH` / `PYTHONPATH` at startup
- Runs a **16-test qualification** on the **gfx1151** build on self-hosted AMD GPU hardware (Strix Halo) — Tier 0 static bundle checks, Tier 1 hardware smoke, Tier 2 functional inference — aggregates the results into a qualification report, and gates all releases on the build being promotable (see [`scripts/qualify`](scripts/qualify/README.md))
- Tars the result, splits it into `< 2 GB` parts, and publishes the release

| GPU Target | Ubuntu |
|------------|--------|
| **gfx1151** | [![Download](https://img.shields.io/badge/Download-Ubuntu%20gfx1151-blue)](https://github.com/lemonade-sdk/vllm-rocm/releases/latest) |
| **gfx1150** | [![Download](https://img.shields.io/badge/Download-Ubuntu%20gfx1150-blue)](https://github.com/lemonade-sdk/vllm-rocm/releases/latest) |
| **gfx120X** | [![Download](https://img.shields.io/badge/Download-Ubuntu%20gfx120X-blue)](https://github.com/lemonade-sdk/vllm-rocm/releases/latest) |
| **gfx110X** | [![Download](https://img.shields.io/badge/Download-Ubuntu%20gfx110X-blue)](https://github.com/lemonade-sdk/vllm-rocm/releases/latest) |

> **Linux (gfx1150/APU):** OOM despite free VRAM? Add `ttm.pages_limit=12582912` (48 GB) to the kernel cmdline (e.g. GRUB), run `update-grub`, then reboot. See [TheRock FAQ](https://github.com/ROCm/TheRock/blob/main/docs/faq.md#gfx1151-strix-halo-specific-questions).

### vLLM-Omni variant (experimental)

[vLLM-Omni](https://github.com/vllm-project/vllm-omni) serves omni / any-to-any
multimodal models (Qwen-Omni, Cosmos3, …). It is a **pure-Python layer** on top
of the same base vLLM + PyTorch + Triton this repo already bundles, so the omni
build is the base bundle plus `vllm-omni`, its runtime deps, an ABI-matched
`torchaudio`, and a `bin/vllm-omni-server` launcher — see
[`scripts/build_omni_layer.sh`](scripts/build_omni_layer.sh).

It ships as a **separate release artifact** (tag `vllm-omni<ver>-rocm<ver>-<gfx>`),
not folded into the lean LLM bundle — omni pulls ~1 GB of extra deps that
plain-LLM users should not carry. It **auto-builds daily** for gfx1151 (a second
`schedule` cron at 16:30 UTC) — builds on the current nightly base, qualifies
via `tier2_omni`, and publishes the `vllm-omni*` release on green — and can also
be built on demand from the **Build vLLM + ROCm** workflow with the `omni: true`
dispatch input (`vllm-omni`'s version is auto-matched to the base vLLM
major.minor). Run a serving model with
`bin/vllm-omni-server serve <model> --omni --deploy-config <single-gpu.yaml>`;
on a single-GPU box you must supply a deploy config that colocates all stages on
device 0 (the upstream defaults target multi-GPU hosts).

## Dependencies

### Runtime (bundled in the release)
- **[vLLM](https://github.com/vllm-project/vllm)** — high-throughput LLM serving engine (ROCm wheel from AMD's `rocm.frameworks.amd.com` / `rocm.frameworks-nightlies.amd.com` index)
- **[PyTorch](https://pytorch.org/)** — tensor compute (ROCm wheel from `repo.amd.com/rocm/whl/<suffix>/` / `rocm.nightlies.amd.com/whl-multi-arch`)
- **[ROCm SDK wheels](https://github.com/ROCm/TheRock)** — AMD's pip-packaged ROCm user-space (`rocm-sdk-core`, `rocm-sdk-libraries-gfx<target>`, published alongside via [TheRock](https://github.com/ROCm/TheRock))
- **[python-build-standalone](https://github.com/astral-sh/python-build-standalone)** — relocatable CPython (3.13 stable / 3.14 nightly)

### Build (CI only)
- **Ubuntu 22.04** GitHub Actions runner
- `pip` (no `cmake`, `ninja`, or `patchelf` involved — everything comes from pre-built wheels)

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
