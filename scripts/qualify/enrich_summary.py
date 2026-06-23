#!/usr/bin/env python3
"""Optional LLM enrichment of the qualification report's `summary` block.

The deterministic summary fields written by aggregate.py are AUTHORITATIVE and
always present. When an ANTHROPIC_API_KEY is available, this step rewrites only
the human-facing PROSE (`title`, `recommended_action`, `root_causes` wording)
with a small, schema-constrained model pass. It NEVER changes `blocking_tests`
or any machine-readable field, and it is strictly best-effort: any problem (no
key, missing package, API error, schema mismatch, malformed report) leaves the
deterministic summary untouched and exits 0. It must never fail the job.

Run AFTER aggregate.py, on a runner with internet access and the `anthropic`
package installed (e.g. the hosted publish-qualification job — NOT the GPU box's
stripped portable Python). Mutates the report file in place.

    python3 enrich_summary.py --report qualification-report.json \
        --model claude-haiku-4-5
"""
import argparse
import json
import os
from pathlib import Path

MODEL_DEFAULT = "claude-haiku-4-5"

SYSTEM = (
    "You write a concise, accurate failure summary for a vLLM-ROCm build "
    "qualification dashboard. You are given the machine-computed summary plus "
    "the raw failing tests. Improve ONLY the wording of `title`, `root_causes`, "
    "and `recommended_action`. Ground every statement in the provided test "
    "errors — never speculate or invent a cause. Name the concrete versions, "
    "symbols, or components involved. `title` must be <= 70 characters."
)


def _failing_tests(report):
    out = []
    for tier in sorted(report.get("tiers", {})):
        for test in report["tiers"][tier].get("tests", []):
            if test.get("status") in ("fail", "warn"):
                out.append(
                    {
                        "id": test.get("id"),
                        "tier": tier,
                        "name": test.get("name"),
                        "gating": test.get("gating"),
                        "status": test.get("status"),
                        "error": test.get("error"),
                        "details": test.get("details"),
                    }
                )
    return out


def main():
    ap = argparse.ArgumentParser(description="Best-effort LLM summary enrichment")
    ap.add_argument("--report", required=True)
    ap.add_argument("--model", default=MODEL_DEFAULT)
    args = ap.parse_args()

    path = Path(args.report)
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        print(f"::warning::enrich: cannot read {path}: {exc}", flush=True)
        return

    det = report.get("summary")
    if not det:
        print("::warning::enrich: no deterministic summary present; skipping", flush=True)
        return
    if report.get("promoted"):
        print("enrich: build promoted; keeping deterministic summary", flush=True)
        return
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("enrich: ANTHROPIC_API_KEY not set; keeping deterministic summary", flush=True)
        return

    try:
        import anthropic
        from pydantic import BaseModel
    except Exception as exc:  # noqa: BLE001
        print(f"::warning::enrich: anthropic/pydantic unavailable ({exc})", flush=True)
        return

    class Summary(BaseModel):
        title: str
        root_causes: list[str]
        recommended_action: str

    build = report.get("build", {})
    payload = {
        "build": {
            k: build.get(k)
            for k in (
                "gfx_target",
                "channel",
                "vllm_version",
                "torch_version",
                "rocm_version",
                "candidate_tag",
            )
        },
        "overall": report.get("overall"),
        "blocked_by": report.get("promotion", {}).get("blocked_by", []),
        "deterministic_summary": det,
        "failing_tests": _failing_tests(report),
    }

    try:
        client = anthropic.Anthropic()
        resp = client.messages.parse(
            model=args.model,
            max_tokens=1024,
            system=SYSTEM,
            messages=[{"role": "user", "content": json.dumps(payload, indent=2)}],
            output_format=Summary,
        )
        parsed = resp.parsed_output
        if parsed is None:
            print("::warning::enrich: model returned no parsed output", flush=True)
            return
    except Exception as exc:  # noqa: BLE001
        print(f"::warning::enrich: API/parse failed ({exc})", flush=True)
        return

    report["summary"] = {
        "title": parsed.title.strip() or det.get("title"),
        "root_causes": [c.strip() for c in parsed.root_causes if c.strip()]
        or det.get("root_causes", []),
        "recommended_action": parsed.recommended_action.strip()
        or det.get("recommended_action"),
        # AUTHORITATIVE — never taken from the model.
        "blocking_tests": det.get("blocking_tests", []),
        "source": "llm",
        "model": args.model,
        "deterministic_fallback": det,
    }
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"enrich: rewrote summary via {args.model}: {report['summary']['title']}", flush=True)


if __name__ == "__main__":
    main()
