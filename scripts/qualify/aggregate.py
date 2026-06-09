#!/usr/bin/env python3
"""
Aggregate per-tier fragments for one build target into a qualification record,
decide whether the target may be promoted from prerelease to a full release,
append the record to the durable JSONL ledger, and emit a GitHub Step Summary.

The qualification record is the unit a downstream dashboard consumes: one JSON
object per (build, target) with a top-level rollup plus every tier's full test
list for drill-down.

Promotion rule (per target):
  - The tiers named in --require-tiers must all be present AND status == pass.
  - gfx1151 (has hardware): require tier0,tier1,tier2. (Lemonade-integration
    coverage lives in lemonade's own adoption gate, not here.)
  - gfx1150/110X/120X (no hardware): require tier0 only; the record is flagged
    hardware_validated=false so consumers know it was build-verified only.

Usage:
    python3 aggregate.py --fragments-dir ./fragments --gfx-target gfx1151 \
        --candidate-tag vllm0.21.0-rocm7.13.0-gfx1151 \
        --require-tiers tier0,tier1,tier2 \
        --output qualification-report.json --ledger results/ledger.jsonl
"""

import argparse
import glob
import json
import os

import report


def load_fragments(fragments_dir):
    fragments = []
    for path in sorted(glob.glob(os.path.join(fragments_dir, "*.json"))):
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
        if data.get("kind") == "tier_fragment":
            fragments.append(data)
    return fragments


def decide_promotion(record, require_tiers):
    reasons = []
    tiers = record.get("tiers", {})
    for tier in require_tiers:
        if tier not in tiers:
            reasons.append(f"{tier} missing")
        elif tiers[tier]["status"] != report.STATUS_PASS:
            reasons.append(f"{tier}={tiers[tier]['status']}")
    promote = not reasons
    return promote, reasons


def write_step_summary(record, promote, reasons):
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not path:
        return
    meta = record["build"]
    lines = [
        f"## Qualification — {meta.get('candidate_tag') or meta.get('gfx_target')}",
        "",
        f"- **Target:** `{meta.get('gfx_target')}`  **Channel:** `{meta.get('channel')}`",
        f"- **vLLM:** `{meta.get('vllm_version')}`  **torch:** `{meta.get('torch_version')}`",
        f"- **Overall:** **{record['overall'].upper()}**  "
        f"**Promote:** **{'YES' if promote else 'NO'}**",
        f"- **Hardware-validated:** {meta.get('hardware_validated')}",
    ]
    if reasons:
        lines.append(f"- **Blocked by:** {', '.join(reasons)}")
    lines += ["", "| Tier | Status | pass | fail | warn | skip |", "|---|---|---|---|---|---|"]
    for tier in sorted(record["tiers"]):
        info = record["tiers"][tier]
        counts = info.get("counts", {})
        lines.append(
            f"| {tier} {info.get('tier_name', '')} | {info['status'].upper()} | "
            f"{counts.get('pass', 0)} | {counts.get('fail', 0)} | "
            f"{counts.get('warn', 0)} | {counts.get('skip', 0)} |"
        )
    # List failing/warning tests for quick triage.
    problems = []
    for tier in sorted(record["tiers"]):
        for test in record["tiers"][tier].get("tests", []):
            if test["status"] in (report.STATUS_FAIL, report.STATUS_WARN):
                problems.append(
                    f"- `{test['id']}` **{test['status']}** — "
                    f"{test['name']}: {test.get('error') or ''}"
                )
    if problems:
        lines += ["", "### Findings", *problems]
    with open(path, "a", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def emit_outputs(promote, tag, overall):
    path = os.environ.get("GITHUB_OUTPUT")
    if not path:
        return
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(f"promote={'true' if promote else 'false'}\n")
        handle.write(f"overall={overall}\n")
        if tag:
            handle.write(f"tag={tag}\n")


def main():
    parser = argparse.ArgumentParser(description="Aggregate tier fragments")
    parser.add_argument("--fragments-dir", required=True)
    parser.add_argument("--gfx-target", required=True)
    parser.add_argument("--channel", default=None, choices=[None, "stable", "nightly"])
    parser.add_argument("--candidate-tag", default=None)
    parser.add_argument(
        "--require-tiers",
        default="tier0",
        help="Comma-separated tiers that must pass to promote.",
    )
    parser.add_argument(
        "--hardware-validated",
        action="store_true",
        help="Mark the record as validated on real hardware.",
    )
    parser.add_argument("--output", default="qualification-report.json")
    parser.add_argument("--ledger", default=None)
    parser.add_argument(
        "--fail-on-no-promote",
        action="store_true",
        help="Exit non-zero if the target is not promotable (after writing all outputs).",
    )
    args = parser.parse_args()

    require_tiers = [t.strip() for t in args.require_tiers.split(",") if t.strip()]

    fragments = load_fragments(args.fragments_dir)
    extra = {
        "gfx_target": args.gfx_target,
        "channel": args.channel,
        "candidate_tag": args.candidate_tag,
        "hardware_validated": bool(args.hardware_validated),
    }
    record = report.merge_fragments(fragments, extra_meta=extra)

    promote, reasons = decide_promotion(record, require_tiers)
    record["promoted"] = promote
    record["promotion"] = {"required_tiers": require_tiers, "blocked_by": reasons}
    # Deterministic, authoritative "why blocked" summary. An optional LLM pass
    # (scripts/qualify/enrich_summary.py) may later rewrite only its prose.
    record["summary"] = report.build_summary(record)

    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2)
        handle.write("\n")

    if args.ledger:
        os.makedirs(os.path.dirname(args.ledger) or ".", exist_ok=True)
        with open(args.ledger, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")

    write_step_summary(record, promote, reasons)
    emit_outputs(promote, args.candidate_tag, record["overall"])

    print(f"target={args.gfx_target} overall={record['overall']} "
          f"promote={promote} blocked_by={reasons}")
    print(f"Wrote {args.output}" + (f" and appended {args.ledger}" if args.ledger else ""))

    if args.fail_on_no_promote and not promote:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
