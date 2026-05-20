# Build qualification suite

Tiered tests that gate a vllm-rocm build before its release tag is promoted from
`prerelease` to a full release. Each tier emits a dashboard-friendly JSON
fragment; `aggregate.py` merges them into one qualification record per build
target, decides promotion, and appends the record to the `build-results` branch
ledger (`results/ledger.jsonl`).

| Tier | Where it runs | Catches | Gating |
|------|---------------|---------|--------|
| 0 static | hosted build runner (no GPU) | torch ABI / version-pin mismatch, missing native exts, broken launcher | yes |
| 1 smoke | gfx1151 self-hosted | native ext won't dlopen, platform import crash, GPU not visible | yes |
| 2 inference | gfx1151 self-hosted | server won't boot, broken Triton JIT, dead endpoints | yes |
| 3 lemonade | gfx1151 self-hosted (reusable workflow in lemonade) | install path, recipe registry, real-model load+chat | yes |

Promotion is **per target**. gfx1151 must pass tiers 0-3. The other targets
(gfx1150 / gfx110X / gfx120X) have no hardware here, so they pass tier 0 only and
are recorded `hardware_validated: false`.

## Channels

`channel` is a **run-level** parameter (not a matrix axis) — stable and nightly
have different triggers and different upstream sources, so they run as separate
workflow runs. The qualification suite is identical for both, and **both promote
only on green**.

| Channel | Source | Tag suffix | Notes |
|---------|--------|-----------|-------|
| **stable** | AMD's matched vLLM + PyTorch (`rocm.frameworks.amd.com` + `repo.amd.com`) | `…-stable` | self-consistent; lags upstream vLLM |
| **nightly** | latest vLLM (`wheels.vllm.ai/rocm`) + latest AMD ROCm PyTorch | `…-nightly` | bleeding edge; may be red when latest+latest are ABI-incompatible — reported, never patched |

Every qualification record carries `build.channel`, so the dashboard can show a
stable column and a nightly column per target. Consumers select a channel by tag
suffix (`…-stable` / `…-nightly`), **not** GitHub's single "latest" pointer
(which can't represent two channels), mirroring lemonade's
`vllm.rocm-stable` / `vllm.rocm-nightly` keys.

## What each tier looks for

- **T0.1** vLLM's `Requires-Dist: torch==` release == the bundled torch release.
- **T0.2** every undefined `c10::`/`at::`/`torch::` symbol in `_C.abi3.so` /
  `_rocm_C.abi3.so` is defined by a bundled torch/ROCm lib.
- **T0.3** DT_NEEDED sonames resolve in-bundle (warn). **T0.4** required files +
  launcher syntax. **T0.6** bundled amdsmi present.
- **T1.1** `import vllm._C, vllm._rocm_C`. **T1.2** `from vllm.platforms import
  current_platform`. **T1.3** torch.cuda sees the GPU + gcnArchName. **T1.4**
  amdsmi ASIC read (warn). **T1.5** `vllm-server --help`.
- **T2.1** server boots. **T2.2** non-empty completion. **T2.3** greedy
  determinism. **T2.4** chat. **T2.5** streaming.
- **T3.n** lemonade installs the candidate and each hot vLLM model loads + chats;
  tokens/sec and TTFT captured as metrics.

## Self-hosted runner setup (gfx1151)

The HW tiers target a runner labelled **`self-hosted, stx-halo, Linux`** (same
labels lemonade already uses). One GPU → run **one job at a time** on it.

1. **GPU group membership is the #1 correctness requirement.** The runner's
   service user must be in **both** `render` and `video`:
   ```bash
   sudo usermod -aG render,video <runner-user>
   ```
   Then **fully restart the runner service** (a new login is required — `id
   <user>` shows the group DB, not the groups of the already-running session).
   Without this, `torch.cuda.is_available()` is False and the bundled amdsmi
   throws `AMDSMI_STATUS_FILE_ERROR`, which manifests as a misleading
   `vllm.platforms` import error. (This — not a vLLM bug — was the root of two of
   the three failure modes seen with the 0.21.0 release.)
2. **Devices** readable by that user: `/dev/kfd`, `/dev/dri/card*`,
   `/dev/dri/renderD*`.
3. **Kernel/driver**: amdgpu with gfx1151 support (kernel 6.18.4+, or a backport
   with the CWSR fix).
4. **Tools**: `git`, `curl`, system `python3` (tier scripts are stdlib-only).
   The bundle ships its own Python/torch for inference.
5. **Disk**: ~3.2 GB per bundle + model weights (8 GB+). Allow 100 GB+ free for
   the work dir and the HF cache.
6. **Network**: `huggingface.co` (weights) and `github.com` (release assets).
7. **HF token (recommended)**: set `HF_TOKEN` in the runner environment (or as a
   secret) to avoid Hub rate limits during weight downloads.

## Cross-repo wiring (one-time)

Tier 3 is a reusable workflow that lives in **lemonade**, so vllm-rocm must be
allowed to call it:

- In **lemonade** repo → Settings → Actions → General → *Access*: allow access
  from repositories in the `lemonade-sdk` org (so vllm-rocm can `uses:` it).
- Make the gfx1151 runner an **org runner** (or register it to both repos) so
  both `hw-qualify` (vllm-rocm) and the reusable `validate` (lemonade) can use it.
- The `tier3` job currently references the reusable workflow at
  `@feat/vllm-qualification-reusable`. After that branch merges to lemonade
  `main`, change the ref to `@main`.

## Triggering a run

1. Push branch `feat/build-qualification` (vllm-rocm) and
   `feat/vllm-qualification-reusable` (lemonade).
2. vllm-rocm → Actions → **Build vLLM + ROCm** → *Run workflow*. Pick the
   `channel` (`nightly` for latest vLLM, `stable` for AMD's matched set). For a
   fast first pass set `gfx_target = gfx1151` and `create_release = true`.
   Scheduled runs default to `nightly`; `stable` is run on an AMD release (or
   manual dispatch with `channel=stable`).
3. Flow: `build-ubuntu` → Tier 0 → `publish-prerelease` → `hw-qualify`
   (Tier 1+2) → `tier3` (lemonade) → `aggregate` (promote on pass) → `ledger`.
4. Review: the per-job **Step Summary** table, the `qualification-record-*`
   artifact, and `results/ledger.jsonl` on the `build-results` branch.

## Running tiers locally

```bash
# Tier 0 (no GPU)
python3 scripts/qualify/tier0_static.py --bundle-root /opt/vllm --gfx-target gfx1151

# Tier 1/2 (need GPU + render+video groups)
python3 scripts/qualify/tier1_smoke.py  --bundle-root ./vllm-install --gfx-target gfx1151
python3 scripts/qualify/tier2_inference.py --bundle-root ./vllm-install --gfx-target gfx1151

# Aggregate fragments -> record + promotion decision
python3 scripts/qualify/aggregate.py --fragments-dir ./frags --gfx-target gfx1151 \
    --require-tiers tier0,tier1,tier2,tier3 --hardware-validated
```
