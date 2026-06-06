# Qualification feed — published contract

Every hardware qualification run publishes its result to the **`qualification-data`**
branch of this repo, in a stable layout an external dashboard can poll. This is the
contract; treat it as versioned (`schema_version`).

## Why a branch (not releases)

Releases only exist for builds that **pass** and are cut **per gfx target**, and the
Releases API is rate-limited to 60 req/hr unauthenticated. The dashboard needs an
**append-only log of every run, pass or fail** — so the feed is independent of
releases. (Qualified builds *also* get their report attached to the release as a
convenience; that copy is not the dashboard's source of truth.)

The branch is the current **transport**. The **contract below is transport-agnostic**:
the same files/schema can later move to GitHub Pages, a bucket, or a DB without the
dashboard changing.

## Layout (paths on the `qualification-data` branch)

```
qualification/
  index.json                 # primary feed: one compact row per build
  reports/<build_id>.json    # full qualification report, immutable (drill-down)
  latest/<gfx_target>.json   # newest full report for that target
  SCHEMA.md                  # a copy of this contract, on the branch
```

Public raw base URL (no auth, ~5 min CDN cache):

```
https://raw.githubusercontent.com/lemonade-sdk/vllm-rocm/qualification-data/qualification/
```

So the dashboard's primary fetch is:

```
GET .../qualification/index.json
```

## `build_id` and de-duplication

```
build_id = "<run_id>-<gfx_target>-<channel>"   e.g. 27054168201-gfx1151-stable
```

Deterministic and unique per build. Publishing is an **idempotent upsert keyed on
`build_id`**: the per-build file is named by it (a re-run overwrites, never
duplicates), and the index row is replaced in place. `run_attempt` is recorded as a
field so re-runs are visible, but they don't fork the record — the dashboard answers
"did *this build* qualify?" with the latest attempt.

## `index.json`

```jsonc
{
  "schema_version": 1,
  "updated_at": "2026-06-06T06:04:11Z",      // generated_at of the newest build
  "builds": [                                  // newest first
    {
      "build_id": "27054168201-gfx1151-stable",
      "tag": "vllm0.22.1-gfx1151",
      "gfx_target": "gfx1151",
      "channel": "stable",
      "generated_at": "2026-06-06T06:04:11Z",  // ISO-8601 UTC, the time axis
      "run_id": "27054168201",
      "run_attempt": 1,
      "run_url": "https://github.com/lemonade-sdk/vllm-rocm/actions/runs/27054168201",
      "vllm_version": "0.22.1+rocm722",
      "torch_version": "2.11.0+rocm7.13.0",
      "rocm_version": "7.13.0",
      "tests": { "passed": 7, "failed": 8, "warned": 0, "skipped": 0, "total": 15 },
      "qualified": false,                      // == promoted; "did it qualify to release"
      "overall": "fail",                       // pass | fail | warn
      "blocked_by": ["tier0=fail", "tier1=fail", "tier2 missing"],
      "results": {                             // per-test status across the 15-test suite
        "T0.1": "fail", "T0.2": "fail", "T0.3": "pass", "T0.4": "pass", "T0.6": "pass",
        "T1.1": "fail", "T1.2": "pass", "T1.3": "pass", "T1.4": "pass", "T1.5": "pass",
        "T2.1": "missing", "T2.2": "missing", "T2.3": "missing", "T2.4": "missing", "T2.5": "missing"
      },
      "report_url": "reports/27054168201-gfx1151-stable.json"
    }
  ]
}
```

### Field notes

- **`tests.total` is always 15** (the canonical suite). A tier that produced no
  fragment (e.g. Tier 2 when the server failed to boot) has its tests marked
  `"missing"` in `results` and **counted as `failed`**. So `passed + failed +
  warned + skipped == 15` always, and the counts never look misleadingly small.
- **`results`** values are one of `pass | fail | warn | skip | missing`. Test ids are
  the canonical 15: T0.1–T0.4, T0.6, T1.1–T1.5, T2.1–T2.5. This lets the dashboard
  compute failure-frequency and heatmap views from `index.json` alone (no N+1 fetch).
- **`qualified`** is the release gate (`true` only if every required tier passed).
- **`report_url`** is relative to the feed base — open it for the full per-test
  detail (errors, unresolved symbols, timings).

## `reports/<build_id>.json`

The verbatim `qualification-report.json` from `scripts/qualify/aggregate`
(`kind: "qualification_record"`): `build{}` metadata, `overall`, `promoted`,
raw `counts`, and `tiers{}` with every test's `status` / `gating` / `error` /
`details`. The drill-down source; immutable.

## `latest/<gfx_target>.json`

The full report for the most recent build of that target — for status tiles / badges
that want "current state of gfx1151" without scanning the index.

## Coverage caveat

Only **gfx1151** has a self-hosted GPU runner today, so only gfx1151 builds appear in
the feed. Other targets will appear if/when they get a runner (or if Tier 0 — which
needs no GPU — is run for them, giving partial static coverage). The schema already
carries `gfx_target`, so multi-target is a data change, not a contract change.

## Versioning

Bump `schema_version` on any breaking change to `index.json`/entry shape. Additive
fields are non-breaking. The dashboard should read `schema_version` and degrade
gracefully on unknown future versions.
