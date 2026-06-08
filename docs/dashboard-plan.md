# Executive dashboard — plan

The dashboard is a **separate project** (outside this repo) that visualizes vLLM-ROCm
build qualification. This doc is the plan/spec it builds against; the data contract it
consumes is [`qualification-feed.md`](qualification-feed.md).

## What it answers

For each build: **how many of the 15 tests passed/failed, and did the build qualify to
release?** — plus the trend of that over time, and which tests fail most often.

## Data source

One poll of the feed is enough for the whole dashboard:

```
GET https://raw.githubusercontent.com/lemonade-sdk/vllm-rocm/qualification-data/qualification/index.json
```

`reports/<build_id>.json` is fetched lazily only when a user drills into a build. The
dashboard is a pure consumer — it never writes back.

## Views and the fields that drive them

| View | Type | Reads (from `index.json`) |
|------|------|----------------------------|
| **Fleet status tiles** (per target) | status card: qualified ✓/✗, `7/15`, version | `latest/<target>.json` (or newest entry per `gfx_target`) |
| **Qualification rate over time** | line / % | `qualified` + `generated_at` bucketed by day/week |
| **Pass/fail per build** | stacked bar, time x-axis | `tests.{passed,failed,warned,skipped}` |
| **Trend annotated with versions** | line + markers | above + `vllm_version` / `torch_version` change points |
| **Failure Pareto** (which tests fail most) | horizontal bar | `results` aggregated across builds → fail-count per test id |
| **Tier heatmap** (builds × 15 tests) | grid | `results` per build |
| **Build drill-down** | detail panel | `report_url` → full report (`tiers[].tests[].error/details`) |

The first four need only the compact counts; the failure-analysis views need the
`results` map — which is why it lives in the index (avoids an N+1 fetch of every
report).

## Reference layout

```
┌──────────────────────────────────────────────────────────────────┐
│  vLLM-ROCm Build Qualification                          gfx1151 ▾  │
├───────────────┬───────────────┬───────────────┬──────────────────┤
│ gfx1151  ✗    │ gfx1150  —    │ gfx120X  —    │ gfx110X  —       │  tiles: latest/
│ 7/15  not qual│ no hw test    │ no hw test    │ no hw test       │
├───────────────┴───────────────┴───────────────┴──────────────────┤
│  Qualification rate (30d)            │  Pass/fail per build       │
│  100% ┤      ╭─╮                     │  15 ┤ █▆ █ ▆█ ▆▆█ ▂        │  index.builds[]
│   50% ┤ ╭──╮ │ ╰──╴                  │     ┤ ▂▂ ▂ ▂▂ ▂▂▂ █        │
│    0% ┼─╯  ╰─╯                       │     └────────────────────  │
├──────────────────────────────────────────────────────────────────┤
│  Failure Pareto                      │  builds × tests heatmap    │
│  T0.2 ████████  T2.1 ██████          │   ▓▓░░░ ░░▓░░ ...          │  results{}
│  T1.1 ███████   T0.1 █████           │   (red = fail/missing)     │
└──────────────────────────────────────────────────────────────────┘
```

## Suggested mechanics

- **Refresh:** poll `index.json` every 1–5 min (raw URL has ~5 min CDN cache; no point
  polling faster). Optionally show `updated_at` as "data as of …".
- **Color:** `qualified=true` green / `false` red on tiles; per-cell color from
  `results` (`pass`→green, `fail`/`missing`→red, `warn`→amber, `skip`→grey).
- **Time axis:** `generated_at` (ISO-8601 UTC). Sort/bucket on it.
- **Drill-down:** link `run_url` (the CI run) and `report_url` (full JSON).
- **Schema guard:** read `schema_version`; render a "feed updated" notice instead of
  breaking on an unknown future version.

## Caveats to render honestly

- **gfx1151 only** has hardware coverage today — other tiles are "no hw test", not
  "passing". Don't imply green where there's no data.
- A **single build is one point** — trend views need history to accumulate on the
  branch before they're meaningful.
- `total` is always 15 and **`missing` counts as failed**, so a crashed tier reads as
  red, never as a smaller-but-green total.

## Not in scope here

The dashboard's framework/hosting is the separate project's call. Anything that reads
the documented feed works — a static SPA fetching the raw URL, a server that ingests
the JSONL, etc. If write volume or querying outgrows a git branch, swap the transport
(Pages/bucket/DB) and keep this contract.
