# Build qualification suite

Tiered tests that gate whether a vllm-rocm build is released at all: a target's
release is created only when its qualification tiers pass. Each tier emits a
dashboard-friendly JSON fragment; `aggregate.py` merges them into one
qualification record per build target and decides promotion. The
`publish-qualification` job then publishes that record — **pass or fail** — to
the `qualification-data` branch feed (`qualification/index.json`; see
[`../../docs/qualification-feed.md`](../../docs/qualification-feed.md)).

| Tier | Where it runs | Catches | Gating |
|------|---------------|---------|--------|
| 0 static | gfx1151 self-hosted (in the `qualify` job; needs no GPU) | torch ABI / version-pin mismatch, missing native exts, dropped gfx code-object packs, broken launcher | yes |
| 1 smoke | gfx1151 self-hosted | native ext won't dlopen, platform import crash, GPU not visible | yes |
| 2 inference | gfx1151 self-hosted | server won't boot, broken Triton JIT, dead endpoints | yes |

The `qualify` job **runs 16 tests**:
- Tier 0 static (T0.1–T0.6, 6) + 
- Tier 1 hardware smoke (T1.1–T1.5, 5) + 
- Tier 2 functional inference (T2.1–T2.5, 5).

> **16 run, 15 published.** The published dashboard feed reports a fixed **15**:
> [`scripts/publish_qualification.py`](../publish_qualification.py) folds **T0.5**
> (the `.kpack` code-object-pack check) out of the per-test `results` map. T0.5 still
> runs and still **gates** (a T0.5 failure fails the job and blocks the release) — it
> just isn't surfaced in the feed. Separately, **T0.1** reports `skip` on the
> **nightly** channel (the universal-RDNA wheel declares no `Requires-Dist: torch`,
> so there's no pin to check); on **stable** it's a real check. See
> [`../../docs/qualification-feed.md`](../../docs/qualification-feed.md).

The three tiers all run in the single `qualify` job on the gfx1151 box; `aggregate.py
--require-tiers tier0,tier1,tier2 --fail-on-no-promote` is the one gate that
decides pass/fail for the whole job.

**Tier 3 (Lemonade integration) is intentionally NOT run here.** Under the
producer/consumer split, lemonade validates integration on adoption via its own
[`validate_vllm.yml`](https://github.com/lemonade-sdk/lemonade/blob/main/.github/workflows/validate_vllm.yml)
(it installs the candidate `vllm-rocm` release into a lemonade build and runs
`test/validate_vllm.py` on its own self-hosted gfx1151 runner). This repo's suite
only ever **measures** the bundle statically and standalone (tiers 0–2).

> Only **gfx1151** has a self-hosted GPU runner, so it is the only target the
`qualify` job tests. The other targets (gfx1150 / gfx110X / gfx120X) are built
and released alongside a green gfx1151 run, but are not themselves qualified and
do not appear in the feed. (Data-center CDNA targets gfx942 / gfx950 are
buildable on demand but are likewise unqualified here.)

## Release gating (no prerelease)

There is **no prerelease-then-promote step and no `publish-prerelease` job.** The
`create-release` job runs only when `build-ubuntu` succeeded **and** `qualify`
passed (or was skipped because gfx1151 was not in the build set); it then creates
a full GitHub release directly. A **failed** qualification publishes no release
anywhere — the result is recorded only on the `qualification-data` feed (which
logs every run, pass or fail), and the gfx1151 report is also attached to the
release as a convenience when one is cut. Lemonade only auto-discovers full
releases.

## Channels

`channel` is a **run-level** parameter (not a matrix axis) — stable and nightly
have different triggers and different upstream sources, so they run as separate
workflow runs. The qualification suite is identical for both, and **both are
released only on green**.

| Channel | Source | Notes |
|---------|--------|-------|
| **stable** | AMD's matched per-gfx vLLM (`rocm.frameworks.amd.com/whl/<gfx>`) + PyTorch (`repo.amd.com/rocm/whl/<gfx>`); cp313 | self-consistent; lags upstream vLLM |
| **nightly** | AMD's universal-RDNA nightly vLLM (`rocm.frameworks-nightlies.amd.com/whl/device-all-rdna`) + the AMD ROCm PyTorch carrying the same `rocm7.X.0a<DATE>` stamp (`rocm.nightlies.amd.com/whl-multi-arch`); cp314 | bleeding edge; may be red when latest+latest are ABI-incompatible — reported, never patched |

Every qualification record carries `build.channel`, so the dashboard can show a
stable column and a nightly column per target. Release tags carry **no** channel
suffix — the scheme is `vllm{version}-rocm{version}-{gfx_target}` (omni builds:
`vllm-omni{version}-rocm{version}-{gfx_target}`); the channel is distinguishable
by the version string (nightly wheels carry a `.devN+rocm7.X.0a<DATE>` stamp) and
is authoritative in the feed's `channel` field. Lemonade currently consumes a
single `vllm.rocm` pin in `backend_versions.json` (with per-arch overrides for
CDNA), not separate channel keys.

## What each tier looks for

- **T0.1** vLLM's `Requires-Dist: torch==` release == the bundled torch release.
- **T0.2** every undefined `c10::`/`at::`/`torch::` symbol in `_C.abi3.so` /
  `_rocm_C.abi3.so` is defined by a bundled torch/ROCm lib.
- **T0.3** DT_NEEDED sonames resolve in-bundle (warn). **T0.4** required files +
  launcher syntax. **T0.5** every `.kpack` gfx code-object pack the AMD device
  wheels declare in their dist-info RECORD exists on disk and is non-empty
  (catches hidden-dot-dir packs dropped at artifact upload). **T0.6** bundled
  amdsmi present (warn).
- **T1.1** `import vllm._C, vllm._rocm_C`. **T1.2** `from vllm.platforms import
  current_platform`. **T1.3** torch.cuda sees the GPU + gcnArchName. **T1.4**
  amdsmi ASIC read (warn). **T1.5** `vllm-server --help`.
- **T2.1** server boots. **T2.2** non-empty completion. **T2.3** greedy
  determinism. **T2.4** chat. **T2.5** streaming.
- **Omni variant** (`tier2_omni.py`, run *instead of* `tier2_inference.py` on
  builds made with the workflow's `omni: true` input): boots `vllm-omni-server`
  on `Qwen2.5-Omni-3B` with a single-GPU deploy config
  (`deploy/qwen2_5_omni_1gpu.yaml`) and checks **T2.1** omni server boot,
  **T2.2** chat completion, **T2.3** streaming. Emits the same `tier2` fragment,
  so promotion (`--require-tiers tier0,tier1,tier2`) is unchanged.

## Self-hosted runner setup (gfx1151)

The build and qualify jobs target the **`dev_lab` runner group** with the
`[self-hosted, Linux]` labels (targeting the group, not bare labels, keeps runs
off Linux boxes in other visible groups). One GPU → run **one job at a time** on
it.

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
5. **Disk**: ~3.3 GB per bundle (extracts to ~11 GB) + model weights. Allow
   100 GB+ free for the work dir and the HF cache.
6. **Network**: `huggingface.co` (weights) and `github.com` (release assets).
7. **HF token (recommended)**: set `HF_TOKEN` in the runner environment (or as a
   secret) to avoid Hub rate limits during weight downloads.

## Triggering a run

1. vllm-rocm → Actions → **Build vLLM + ROCm** → *Run workflow*. Pick the
   `channel` (`nightly` for AMD's latest, `stable` for AMD's matched set). For a
   fast pass set `gfx_target = gfx1151` and `create_release = true`. Scheduled
   runs default to `nightly`; `stable` is run on demand (dispatch with
   `channel=stable`, vLLM version via the `stable_vllm_ver` input).
2. Flow: `detect-nightly` (dedup poll) → `prepare-matrix` → `build-ubuntu`
   (per-target repackage) → `qualify` (tier0 + tier1 + tier2 + `aggregate`, gates
   the job) → `publish-qualification` (feed record, pass or fail) +
   `create-release` (gated on qualify passing/skipped).
3. Review: the per-job **Step Summary** table, the `qualification-gfx1151`
   artifact, and `qualification/index.json` on the `qualification-data` branch.

## Running tiers locally

```bash
# Tier 0 (no GPU)
python3 scripts/qualify/tier0_static.py --bundle-root /opt/vllm --gfx-target gfx1151

# Tier 1/2 (need GPU + render+video groups)
python3 scripts/qualify/tier1_smoke.py  --bundle-root ./vllm-install --gfx-target gfx1151
python3 scripts/qualify/tier2_inference.py --bundle-root ./vllm-install --gfx-target gfx1151

# Aggregate fragments -> record + promotion decision
python3 scripts/qualify/aggregate.py --fragments-dir ./frags --gfx-target gfx1151 \
    --require-tiers tier0,tier1,tier2 --hardware-validated
```
