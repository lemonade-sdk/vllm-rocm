# CLAUDE.md — vllm-rocm

## What this repo is

`vllm-rocm` is a **pure repackager + qualifier** of ROCm-based vLLM wheels. It
does two things and nothing else:

1. **Repackage** an upstream wheel set into a self-contained, relocatable
   archive (bundled CPython + the wheels + ROCm user-space libs) so it can be
   dropped in as a Lemonade backend with no system Python/ROCm.
2. **Qualify** that archive on real hardware and report the result.

It is **not** a place to fix, patch, or work around upstream bugs.

## Channels

Builds are produced on two channels (mirroring `llamacpp-rocm` /
lemonade's `rocm_channel`). The **qualification suite is identical** for both,
and **both promote prerelease → release only on a green qualification.**

| Channel | What it is | Source | Expectation |
|---------|-----------|--------|-------------|
| **stable** | Pure repackage of AMD's matched, self-consistent set | AMD's published per-gfx vLLM (`rocm.frameworks.amd.com/whl/<gfx>`) + PyTorch (`repo.amd.com/rocm/whl/<gfx>`) | Should pass; lags upstream vLLM |
| **nightly** | AMD's official nightly, date-stamp-matched by us | AMD's universal-RDNA nightly vLLM (`rocm.frameworks-nightlies.amd.com/whl/device-all-rdna`) + the AMD ROCm PyTorch carrying the same `rocm7.X.0a<DATE>` build stamp (`rocm.nightlies.amd.com/whl-multi-arch`) | Bleeding edge; **may legitimately be red** when the newest vLLM + ROCm aren't yet usable on the target GPU |

A red nightly is a correct outcome, not a bug to fix: it reports that the
newest AMD vLLM + ROCm aren't yet usable together on the target hardware. It
stays a prerelease until it goes green on its own.

## Inviolable principles

1. **Do not modify upstream wheel contents.** Repackage AMD's published
   artifacts as-is into a portable layout. No patching binaries, no editing
   sources, no substituting component versions to "make a broken combination
   work."

2. **Do not attempt to fix a broken upstream release.** If AMD publishes a
   wheel that fails to load or run, the qualification suite **reports it as
   broken** and the release stays a prerelease. We never carry a workaround.
   A red qualification is a correct, useful outcome — it tells AMD (and us)
   the release is not usable, with evidence.

3. **Repackaging must be faithful.** The portable archive must contain
   everything the wheel needs at runtime. Size-trimming must never delete a
   file required to load or execute (for example, the versioned `clang-NN`
   that Triton's runtime JIT execs). When in doubt, keep it. Removing a file
   the wheel shipped is *corrupting* the wheel, which is different from — and
   not permitted by — "don't fix upstream."

4. **Single source per channel; never reconcile to make it pass.** Each
   channel repackages one defined source (stable = AMD's matched set; nightly =
   vLLM project wheels + latest ROCm torch). Do not pin or swap a component
   version to force an incompatible combination to load — that is the
   forbidden "fix." If the channel's components don't work together,
   qualification reports it red. (Nightly composing "latest + latest" is the
   channel's *definition*, not a reconciliation; if they mismatch, that is the
   true, reported result.)

5. **Qualification gates promotion — both channels, green-only.** Every build
   (stable and nightly) is published as a `prerelease` first and promoted to a
   full release only when its target's qualification tiers pass. Failing builds
   remain downloadable prereleases with their qualification report attached.
   Lemonade only auto-discovers full releases.

6. **Automation.** **nightly** runs automatically on a daily schedule: a
   `detect-nightly` job polls AMD's `device-all-rdna` index and dedups against
   the published feed, so the pipeline only does real work when a genuinely new
   nightly wheel appears. **stable** runs on demand (`workflow_dispatch` with
   `channel=stable`, vLLM version via the `stable_vllm_ver` input); auto-trigger
   on a new AMD stable wheel is future work. (Scheduled runs only fire from the
   default branch — this must be on `main` to run nightly.)

## Source of truth (per channel)

- **stable** repackages AMD's own per-gfx matched set: AMD vLLM
  (`rocm.frameworks.amd.com/whl/<gfx>`) against AMD PyTorch
  (`repo.amd.com/rocm/whl/<gfx>`). AMD builds the vLLM wheel against that
  PyTorch, so the set is intended to be self-consistent.
- **nightly** repackages AMD's official **universal-RDNA** nightly vLLM
  (`rocm.frameworks-nightlies.amd.com/whl/device-all-rdna` — one wheel covering
  gfx1100..gfx1201) paired with the AMD ROCm PyTorch carrying the **same ROCm
  build stamp** (`rocm7.X.0a<DATE>`) from `rocm.nightlies.amd.com/whl-multi-arch`.
  Pairing torch to the vLLM wheel's stamp is the channel's *definition* (it
  selects the matched component, by construction ABI-consistent) — not a
  reconciliation to force an incompatible combination. This is the channel that
  surfaces breakage on the newest stack (e.g. a vLLM↔torch C++-ABI skew, or GPU
  kernels invalid for a target arch); the qualification suite reports it red and
  it stays a prerelease. We never patch or swap a component to dodge that.

## Qualification suite

See `scripts/qualify/README.md`. Tiers 0-2 (static → hardware smoke →
standalone inference) emit dashboard-friendly JSON records that accumulate on
the `qualification-data` branch (see `docs/qualification-feed.md`). Tier 3
(Lemonade integration) is intentionally **not** here — under the producer/
consumer split, lemonade validates integration on adoption via its own
`validate_vllm.yml`. The suite only ever **measures** the bundle; it must never
alter it to pass.

## What NOT to do here

- Do not patch vLLM, PyTorch, Triton, or ROCm.
- Do not pin/swap component versions to dodge an upstream incompatibility.
- Do not delete files during trim that the runtime needs.
- Do not promote a release that did not pass qualification.
