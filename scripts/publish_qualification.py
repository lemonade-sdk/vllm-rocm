#!/usr/bin/env python3
"""Transform a qualification report into the dashboard feed and upsert it into
the `qualification-data` branch layout.

Reads the full `qualification-report.json` emitted by `scripts/qualify/aggregate`
and writes, under --data-dir (the `qualification/` dir on the data branch):

  index.json                 compact one-row-per-build feed (the dashboard poll)
  reports/<build_id>.json    the full report, immutable, for drill-down
  latest/<gfx_target>.json   newest full report for that target

The index entry is keyed by build_id, so re-running this for the same build is
an idempotent upsert (no duplicate rows/files). Counts are normalized to the
canonical 15-test suite: a tier that produced no fragment (e.g. Tier 2 when the
server failed to boot) has its tests marked "missing" and folded into `failed`,
so the dashboard never shows a misleadingly small total.

See docs/qualification-feed.md for the published contract.
"""
import argparse
import json
from pathlib import Path

SCHEMA_VERSION = 1

# The canonical 15-test suite. Tier 0 intentionally skips T0.5.
TIER_TESTS = {
    "tier0": ["T0.1", "T0.2", "T0.3", "T0.4", "T0.6"],
    "tier1": ["T1.1", "T1.2", "T1.3", "T1.4", "T1.5"],
    "tier2": ["T2.1", "T2.2", "T2.3", "T2.4", "T2.5"],
}
TOTAL_TESTS = sum(len(v) for v in TIER_TESTS.values())  # 15


def _int(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def build_entry(report, run_url):
    """Derive the compact, dashboard-facing index entry from a full report."""
    b = report.get("build", {})
    tiers = report.get("tiers", {})

    # Per-test status map across the canonical suite; absent tiers -> "missing".
    results = {}
    for tier, ids in TIER_TESTS.items():
        by_id = {t.get("id"): t for t in tiers.get(tier, {}).get("tests", [])}
        for tid in ids:
            results[tid] = by_id.get(tid, {}).get("status", "missing")

    passed = sum(1 for s in results.values() if s == "pass")
    warned = sum(1 for s in results.values() if s == "warn")
    skipped = sum(1 for s in results.values() if s == "skip")
    failed = TOTAL_TESTS - passed - warned - skipped  # missing folds into failed

    return {
        "build_id": b.get("build_id"),
        "tag": b.get("candidate_tag"),
        "gfx_target": b.get("gfx_target"),
        "channel": b.get("channel"),
        "generated_at": report.get("generated_at"),
        "run_id": b.get("run_id"),
        "run_attempt": _int(b.get("run_attempt")),
        "run_url": run_url,
        "vllm_version": b.get("vllm_version"),
        "torch_version": b.get("torch_version"),
        "rocm_version": b.get("rocm_version"),
        "tests": {
            "passed": passed,
            "failed": failed,
            "warned": warned,
            "skipped": skipped,
            "total": TOTAL_TESTS,
        },
        "qualified": bool(report.get("promoted")),
        "overall": report.get("overall"),
        "blocked_by": report.get("promotion", {}).get("blocked_by", []),
        "results": results,
        "report_url": f"reports/{b.get('build_id')}.json",
    }


def upsert_index(index_path, entry):
    """Replace-or-append the entry by build_id, newest first."""
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
    else:
        index = {"schema_version": SCHEMA_VERSION, "updated_at": None, "builds": []}

    builds = [b for b in index.get("builds", []) if b.get("build_id") != entry["build_id"]]
    builds.append(entry)
    builds.sort(key=lambda b: b.get("generated_at") or "", reverse=True)

    index["schema_version"] = SCHEMA_VERSION
    index["updated_at"] = entry.get("generated_at")
    index["builds"] = builds
    return index


def main():
    ap = argparse.ArgumentParser(description="Publish a qualification report to the data-branch feed")
    ap.add_argument("--report", required=True, help="path to qualification-report.json")
    ap.add_argument("--data-dir", required=True, help="the qualification/ dir on the data branch")
    ap.add_argument("--run-url", default="", help="URL of the CI run that produced the report")
    args = ap.parse_args()

    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    entry = build_entry(report, args.run_url)
    if not entry["build_id"]:
        raise SystemExit("report has no build.build_id; cannot publish")

    data_dir = Path(args.data_dir)
    (data_dir / "reports").mkdir(parents=True, exist_ok=True)
    (data_dir / "latest").mkdir(parents=True, exist_ok=True)

    full = json.dumps(report, indent=2)
    (data_dir / "reports" / f"{entry['build_id']}.json").write_text(full, encoding="utf-8")
    if entry["gfx_target"]:
        (data_dir / "latest" / f"{entry['gfx_target']}.json").write_text(full, encoding="utf-8")

    index = upsert_index(data_dir / "index.json", entry)
    (data_dir / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")

    print(
        f"Published {entry['build_id']}: "
        f"{entry['tests']['passed']}/{entry['tests']['total']} passed, "
        f"qualified={entry['qualified']} ({len(index['builds'])} builds in index)"
    )


if __name__ == "__main__":
    main()
