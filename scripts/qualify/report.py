#!/usr/bin/env python3
"""
Shared reporting for the vLLM-ROCm build qualification suite.

Every qualification tier (tier0 static, tier1 smoke, tier2 inference, tier3
lemonade integration) emits a JSON *fragment* with an identical schema. A
later aggregation step merges the per-tier fragments for one build target into
a single qualification record.

The schema is intentionally flat and stable so it can be consumed directly by a
downstream dashboard:

  - Every test has a stable `id` (e.g. "T0.1") that never changes meaning, so a
    dashboard can chart one check across builds and detect regressions.
  - `status` is always one of the STATUS_* enum values below.
  - Numeric measurements live under `details`/`metrics` as numbers (never
    strings) so they can be plotted without parsing.
  - Timestamps are ISO-8601 UTC with a trailing "Z".
  - `counts` gives a ready-made pass/fail/warn/skip rollup per fragment.

This module is stdlib-only so it can run under the bundle's stripped-down
portable Python on a build runner with no extra packages installed.
"""

import datetime
import json
import time
import traceback

SCHEMA_VERSION = 1

STATUS_PASS = "pass"
STATUS_FAIL = "fail"
STATUS_WARN = "warn"
STATUS_SKIP = "skip"

# Severity order used for rollups. Higher wins.
_SEVERITY = {STATUS_SKIP: 0, STATUS_PASS: 1, STATUS_WARN: 2, STATUS_FAIL: 3}

TIER_NAMES = {
    "tier0": "Static bundle verification",
    "tier1": "Hardware smoke",
    "tier2": "Standalone functional inference",
    "tier3": "Lemonade integration",
}


def now_iso():
    """Current UTC time as an ISO-8601 string with a trailing 'Z'."""
    return (
        datetime.datetime.now(datetime.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def build_meta(
    gfx_target,
    channel=None,
    vllm_version=None,
    torch_version=None,
    rocm_version=None,
    candidate_tag=None,
    lemonade_ref=None,
    hardware_validated=False,
    run_id=None,
    run_attempt=None,
):
    """Assemble the per-build metadata block shared by every fragment."""
    build_id = run_id or "local"
    for part in (gfx_target, channel):
        if part:
            build_id = f"{build_id}-{part}"
    return {
        "build_id": build_id,
        "run_id": run_id,
        "run_attempt": run_attempt,
        "gfx_target": gfx_target,
        "channel": channel,
        "vllm_version": vllm_version,
        "torch_version": torch_version,
        "rocm_version": rocm_version,
        "candidate_tag": candidate_tag,
        "lemonade_ref": lemonade_ref,
        "hardware_validated": bool(hardware_validated),
    }


class TierReport:
    """Collects test results for one tier and serializes a JSON fragment."""

    def __init__(self, tier, meta):
        if tier not in TIER_NAMES:
            raise ValueError(f"unknown tier {tier!r}")
        self.tier = tier
        self.meta = meta
        self.tests = []
        self._started = time.monotonic()
        self._generated_at = now_iso()

    def add(
        self,
        test_id,
        name,
        status,
        gating=True,
        duration_ms=None,
        error=None,
        details=None,
    ):
        """Record a single test result."""
        if status not in _SEVERITY:
            raise ValueError(f"invalid status {status!r}")
        self.tests.append(
            {
                "id": test_id,
                "name": name,
                "tier": self.tier,
                "status": status,
                "gating": bool(gating),
                "duration_ms": duration_ms,
                "error": error,
                "details": details or {},
            }
        )
        return status

    def run(self, test_id, name, fn, gating=True):
        """Run a check function, timing it and trapping exceptions.

        `fn` must return one of:
          - a status string, or
          - (status, error) , or
          - (status, error, details).
        Any raised exception is recorded as a FAIL with the traceback in
        details, so one broken check never aborts the rest of the tier.
        """
        start = time.monotonic()
        try:
            result = fn()
            if isinstance(result, str):
                status, error, details = result, None, None
            elif len(result) == 2:
                (status, error), details = result, None
            else:
                status, error, details = result
        except Exception as exc:  # noqa: BLE001 - we want to capture everything
            status = STATUS_FAIL
            error = f"{type(exc).__name__}: {exc}"
            details = {"traceback": traceback.format_exc()}
        duration_ms = int((time.monotonic() - start) * 1000)
        return self.add(
            test_id,
            name,
            status,
            gating=gating,
            duration_ms=duration_ms,
            error=error,
            details=details,
        )

    def counts(self):
        out = {
            STATUS_PASS: 0,
            STATUS_FAIL: 0,
            STATUS_WARN: 0,
            STATUS_SKIP: 0,
            "total": len(self.tests),
        }
        for test in self.tests:
            out[test["status"]] += 1
        return out

    def rollup_status(self):
        """Tier status = worst severity among *gating* tests."""
        worst = STATUS_PASS
        for test in self.tests:
            if not test["gating"]:
                continue
            if _SEVERITY[test["status"]] > _SEVERITY[worst]:
                worst = test["status"]
        return worst

    def to_dict(self):
        return {
            "schema_version": SCHEMA_VERSION,
            "kind": "tier_fragment",
            "tier": self.tier,
            "tier_name": TIER_NAMES[self.tier],
            "status": self.rollup_status(),
            "generated_at": self._generated_at,
            "duration_ms": int((time.monotonic() - self._started) * 1000),
            "build": self.meta,
            "counts": self.counts(),
            "tests": self.tests,
        }

    def write(self, path):
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2)
            handle.write("\n")

    def print_summary(self):
        """Human-readable summary to stdout (also useful in CI logs)."""
        rollup = self.rollup_status()
        print(f"\n=== {self.tier} ({TIER_NAMES[self.tier]}): {rollup.upper()} ===")
        for test in self.tests:
            mark = test["status"].upper()
            gate = "" if test["gating"] else " (non-gating)"
            line = f"  [{mark}] {test['id']} {test['name']}{gate}"
            if test["error"]:
                line += f" — {test['error']}"
            print(line)
        counts = self.counts()
        print(
            f"  totals: {counts[STATUS_PASS]} pass, {counts[STATUS_FAIL]} fail, "
            f"{counts[STATUS_WARN]} warn, {counts[STATUS_SKIP]} skip"
        )


def merge_fragments(fragments, extra_meta=None):
    """Merge per-tier fragments for one build target into one record.

    Used by the aggregation step. The merged record keeps every tier's full
    test list (for drill-down) plus a top-level rollup the dashboard can index.
    """
    fragments = sorted(fragments, key=lambda frag: frag.get("tier", ""))
    meta = {}
    for frag in fragments:
        meta.update(frag.get("build", {}))
    if extra_meta:
        meta.update(extra_meta)

    tiers = {}
    overall = STATUS_PASS
    totals = {STATUS_PASS: 0, STATUS_FAIL: 0, STATUS_WARN: 0, STATUS_SKIP: 0, "total": 0}
    for frag in fragments:
        tiers[frag["tier"]] = {
            "status": frag.get("status", STATUS_SKIP),
            "tier_name": frag.get("tier_name"),
            "counts": frag.get("counts", {}),
            "tests": frag.get("tests", []),
        }
        if _SEVERITY[frag.get("status", STATUS_SKIP)] > _SEVERITY[overall]:
            overall = frag.get("status", STATUS_SKIP)
        for key, value in frag.get("counts", {}).items():
            totals[key] = totals.get(key, 0) + value

    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "qualification_record",
        "generated_at": now_iso(),
        "build": meta,
        "overall": overall,
        "promoted": False,
        "counts": totals,
        "tiers": tiers,
    }


# Signature rules for the deterministic "why blocked" summary. Each rule is
# (predicate over the lowercased combined error text, title, recommended_action).
# Ordered most-upstream-cause first so the title names the root, not a symptom.
_SUMMARY_RULES = [
    (
        lambda s: "requires torch" in s
        or "version-pin" in s
        or ("torch" in s and "but bundle ships" in s),
        "Blocked by torch version mismatch",
        "Bundle a torch build matching the exact version vLLM was compiled "
        "against (the matched ROCm release). Do not pin around the mismatch.",
    ),
    (
        lambda s: "undefined symbol" in s or "abi symbol" in s or "c10::" in s,
        "Blocked by torch ABI mismatch",
        "Pair the vLLM wheel with the torch build its native extensions were "
        "linked against (same ROCm version), or rebuild the extensions.",
    ),
    (
        lambda s: "enginecore" in s
        or "engine core" in s
        or "failed to start" in s
        or "importerror" in s,
        "Blocked by vLLM server startup failure",
        "Resolve the import/startup error (often a downstream torch ABI break) "
        "so functional inference can run.",
    ),
]


def build_summary(record):
    """Deterministic 'why blocked' summary derived from the merged record.

    Authoritative and stdlib-only — computed every run with no network. An
    optional LLM step may later rewrite the prose (`title`, `recommended_action`,
    `root_causes` wording) but must preserve `blocking_tests` and the verdict.
    """
    tiers = record.get("tiers", {})
    blocked_by = record.get("promotion", {}).get("blocked_by", [])

    failing = []
    for tier in sorted(tiers):
        for test in tiers[tier].get("tests", []):
            if test.get("gating") and test.get("status") == STATUS_FAIL:
                failing.append(test)
    blocking_tests = [t["id"] for t in failing if t.get("id")]

    if record.get("promoted"):
        return {
            "title": "Promoted — all required tiers passed",
            "root_causes": [],
            "recommended_action": "None — the build qualified and can be released.",
            "blocking_tests": [],
            "source": "deterministic",
        }

    root_causes, seen = [], set()
    for test in failing:
        err = (test.get("error") or "").strip()
        if err and err not in seen:
            seen.add(err)
            root_causes.append(err)
    # Required tiers that produced no fragment (e.g. server never booted).
    for reason in blocked_by:
        if reason.endswith("missing"):
            tier = reason.split()[0]
            root_causes.append(
                f"{tier} ({TIER_NAMES.get(tier, tier)}) produced no results — "
                "a prerequisite tier likely failed before it could run."
            )

    combined = " ".join((t.get("error") or "") for t in failing).lower()
    title = action = None
    for predicate, rule_title, rule_action in _SUMMARY_RULES:
        if predicate(combined):
            title, action = rule_title, rule_action
            break
    if title is None:
        if blocking_tests:
            title = f"Qualification failed ({len(blocking_tests)} gating check(s) failed)"
            action = "Inspect the failing tests below and the attached run logs."
        else:
            title = "Qualification incomplete — required tier(s) did not run"
            action = (
                "A required tier produced no results; check whether an earlier "
                "tier failed and prevented it from running."
            )

    return {
        "title": title,
        "root_causes": root_causes,
        "recommended_action": action,
        "blocking_tests": blocking_tests,
        "source": "deterministic",
    }
