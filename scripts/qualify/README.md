# vLLM-ROCm qualification harness

Runs against an assembled bundle (`/opt/vllm`-style tree) on a self-hosted AMD
GPU runner and decides whether the build is promotable to a release. Invoked by
the `qualify` job in `.github/workflows/build-vllm-rocm.yml`.

## The 15 tests

| Tier | Module | Tests |
|------|--------|-------|
| **Tier 0** — static bundle verification (no GPU) | `tier0_static` | T0.1 torch version-pin · T0.2 native-ext symbol satisfaction · T0.3 DT_NEEDED resolution · T0.4 structural manifest · T0.6 amdsmi path sanity |
| **Tier 1** — hardware smoke | `tier1_smoke` | T1.1 native dlopen · T1.2 platform import · T1.3 device visibility · T1.4 amdsmi gcn-arch · T1.5 launcher `--help` |
| **Tier 2** — functional inference | `tier2_inference` | T2.1 server boot · T2.2 completion · T2.3 greedy determinism · T2.4 chat · T2.5 streaming |

Supporting modules: `aggregate` (merges tier fragments → `qualification-report.json`,
writes the job-summary table, emits `promote`/`overall`/`tag` outputs, decides
promotion via `--require-tiers`), `report` (schema + `TierReport` helper),
`bundle` (replicates the `bin/vllm-server` runtime env for probe subprocesses).
`tier3_from_validate` builds a Lemonade-integration fragment and is **not** part
of the 15-test gate.

Each tier writes a JSON fragment (`tierN-report.json`) and is run without
`--fail-on-error`, so all 15 tests execute even when one fails. `aggregate
--fail-on-no-promote` is the single gate that fails the CI job (and blocks the
release) when the build is not promotable.

## ⚠️ Sourceless bytecode

The `.py` sources for this harness were lost; only the compiled `.pyc`
(Python 3.12) survive and are committed here as **sourceless modules**. A 3.12
interpreter — including the bundle's own `bin/python3` (3.12.x) — executes them
directly, and `import report`/`import bundle` resolve from the sibling `.pyc`.

This is a stopgap so the qualification flow works. **The original `.py` sources
should be recovered and committed** to make the harness reviewable and robust
across Python versions.

## Running locally

```bash
PY=/path/to/bundle/bin/python3   # a 3.12 interpreter
$PY scripts/qualify/tier0_static.pyc  --bundle-root /path/to/bundle --gfx-target gfx1151 --output /tmp/frag/tier0-report.json
$PY scripts/qualify/tier1_smoke.pyc   --bundle-root /path/to/bundle --gfx-target gfx1151 --output /tmp/frag/tier1-report.json
$PY scripts/qualify/tier2_inference.pyc --bundle-root /path/to/bundle --gfx-target gfx1151 --output /tmp/frag/tier2-report.json
$PY scripts/qualify/aggregate.pyc --fragments-dir /tmp/frag --gfx-target gfx1151 \
    --require-tiers tier0,tier1,tier2 --output /tmp/frag/qualification-report.json --fail-on-no-promote
```
