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
| **stable** | Pure repackage of AMD's matched, self-consistent set | AMD's published vLLM + PyTorch (`rocm.frameworks.amd.com` / `repo.amd.com`) | Should pass; lags upstream vLLM |
| **nightly** | Latest vLLM on latest ROCm, composed by us | vLLM project ROCm wheels (`wheels.vllm.ai/rocm`) + latest AMD ROCm PyTorch | Bleeding edge; **may legitimately be red** when latest vLLM and latest ROCm are incompatible |

A red nightly is a correct outcome, not a bug to fix: it reports that the
newest vLLM + newest ROCm don't yet work together. It stays a prerelease until
it goes green on its own.

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

6. **Automation.** Packaging + qualification runs automatically: **stable**
   whenever AMD publishes a new vLLM/ROCm wheel, **nightly** on a schedule
   tracking the latest vLLM + ROCm. No manual step for either.

## Source of truth (per channel)

- **stable** repackages AMD's own matched set: AMD vLLM (`rocm.frameworks.amd.com`)
  resolved against AMD PyTorch (`repo.amd.com`). Because AMD builds the vLLM
  wheel against that PyTorch, the set is self-consistent — no reinstall, no
  pinning, no ABI reconciliation.
- **nightly** repackages the vLLM project's ROCm wheel (`wheels.vllm.ai/rocm`)
  on the latest AMD ROCm PyTorch. This is the channel that surfaces
  incompatibilities (e.g. `vllm0.21.0` linked against `c10::hip` while the
  latest AMD torch exposes `c10::cuda`). We do **not** pin around it; the
  qualification suite reports nightly red and it stays a prerelease.

## Qualification suite

See `scripts/qualify/README.md`. Tiers 0-3 (static → hardware smoke →
standalone inference → Lemonade integration) emit dashboard-friendly JSON
records that accumulate on the `build-results` branch. The suite only ever
**measures** the bundle; it must never alter it to pass.

## What NOT to do here

- Do not patch vLLM, PyTorch, Triton, or ROCm.
- Do not pin/swap component versions to dodge an upstream incompatibility.
- Do not delete files during trim that the runtime needs.
- Do not promote a release that did not pass qualification.
