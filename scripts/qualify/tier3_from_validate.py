#!/usr/bin/env python3
"""
Convert lemonade's validate_vllm.py results into a Tier 3 fragment.

The lemonade integration test (run via the reusable workflow in the lemonade
repo) emits a list of per-model results. This script wraps those into the same
tier-fragment schema every other tier uses, so the aggregation step and the
downstream dashboard see Tier 3 identically to Tiers 0-2.

Each hot model becomes one test:
  T3.<n>  <model id>  — pass/fail from the load+chat result, with tokens/sec and
                        time-to-first-token captured as numeric metrics.

Usage:
    python3 tier3_from_validate.py --validate-json vllm_validation_rocm.json \
        --gfx-target gfx1151 --candidate-tag <tag> --output tier3-gfx1151.json
"""

import argparse
import json
import os

import report


def as_number(value):
    if isinstance(value, (int, float)):
        return value
    return None


def main():
    parser = argparse.ArgumentParser(description="Build a Tier 3 fragment")
    parser.add_argument("--validate-json", required=True)
    parser.add_argument("--gfx-target", required=True)
    parser.add_argument("--candidate-tag", default=None)
    parser.add_argument("--vllm-version", default=None)
    parser.add_argument("--torch-version", default=None)
    parser.add_argument("--rocm-version", default=None)
    parser.add_argument("--lemonade-ref", default=None)
    parser.add_argument("--run-id", default=os.environ.get("GITHUB_RUN_ID"))
    parser.add_argument("--run-attempt", default=os.environ.get("GITHUB_RUN_ATTEMPT"))
    parser.add_argument("--output", default="tier3-report.json")
    args = parser.parse_args()

    meta = report.build_meta(
        gfx_target=args.gfx_target,
        vllm_version=args.vllm_version,
        torch_version=args.torch_version,
        rocm_version=args.rocm_version,
        candidate_tag=args.candidate_tag,
        lemonade_ref=args.lemonade_ref,
        hardware_validated=True,
        run_id=args.run_id,
        run_attempt=args.run_attempt,
    )
    tier = report.TierReport("tier3", meta)

    if not os.path.isfile(args.validate_json):
        tier.add(
            "T3.0",
            "lemonade validation results present",
            report.STATUS_FAIL,
            error=f"missing {args.validate_json} (validation did not produce output)",
        )
    else:
        with open(args.validate_json, encoding="utf-8") as handle:
            results = json.load(handle)
        if not results:
            tier.add(
                "T3.0",
                "lemonade validation produced results",
                report.STATUS_FAIL,
                error="empty results list (no hot vllm models tested)",
            )
        for index, result in enumerate(results, start=1):
            model = result.get("model", f"model-{index}")
            passed = bool(result.get("pass"))
            details = {
                "response_sample": str(result.get("response", ""))[:200],
                "input_tokens": result.get("input_tokens"),
                "output_tokens": result.get("output_tokens"),
                "metrics": {
                    "tokens_per_second": as_number(result.get("tokens_per_second")),
                    "time_to_first_token": as_number(
                        result.get("time_to_first_token")
                    ),
                },
            }
            tier.add(
                f"T3.{index}",
                f"lemonade load+chat: {model}",
                report.STATUS_PASS if passed else report.STATUS_FAIL,
                error=None if passed else str(result.get("response", ""))[:200],
                details=details,
            )

    tier.write(args.output)
    tier.print_summary()
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
