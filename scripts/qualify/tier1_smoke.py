#!/usr/bin/env python3
"""
Tier 1 — hardware smoke test (requires a real AMD GPU, gfx1151 runner).

Runs the candidate bundle directly (no lemonade) to confirm the native stack
loads and sees the GPU. Each probe runs as an isolated subprocess so a crash is
recorded rather than fatal.

Checks:
  T1.1  native dlopen        — import vllm._C and vllm._rocm_C (gating).
  T1.2  platform import      — from vllm.platforms import current_platform
                               (gating; catches the amdsmi circular-import bug).
  T1.3  device visibility    — torch.cuda sees the GPU; gcnArchName matches
                               the target (gating).
  T1.4  amdsmi gcn-arch      — bundled amdsmi can read ASIC info (NON-gating;
                               currently fails on gfx1151, recorded as data).
  T1.5  launcher --help      — bin/vllm-server --help exits 0 (gating; exercises
                               the real launcher import path).

Usage:
    python3 tier1_smoke.py --bundle-root ./vllm-install --gfx-target gfx1151 \
        --output tier1-gfx1151.json
"""

import argparse
import glob
import json
import os
import subprocess
import sys

import bundle
import report

PROBE_NATIVE = "import vllm._C, vllm._rocm_C; print('NATIVE_OK')"

PROBE_PLATFORM = (
    "from vllm.platforms import current_platform; "
    "print('PLATFORM:' + str(current_platform.get_device_name()))"
)

PROBE_DEVICE = r"""
import json, torch
info = {"available": bool(torch.cuda.is_available()),
        "count": int(torch.cuda.device_count())}
if info["available"] and info["count"] > 0:
    info["name"] = torch.cuda.get_device_name(0)
    info["gcnArchName"] = torch.cuda.get_device_properties(0).gcnArchName
print("DEVICE:" + json.dumps(info))
"""

PROBE_AMDSMI = r"""
import amdsmi
amdsmi.amdsmi_init()
h = amdsmi.amdsmi_get_processor_handles()[0]
asic = amdsmi.amdsmi_get_gpu_asic_info(h)
print("AMDSMI:" + str(asic.get("target_graphics_version", "")))
"""


def _tail(text, lines=8):
    return "\n".join(text.strip().splitlines()[-lines:])


def check_native(root):
    rc, out, err = bundle.run_bundle_python(root, PROBE_NATIVE, timeout=120)
    if rc == 0 and "NATIVE_OK" in out:
        return (report.STATUS_PASS, None, {})
    return (
        report.STATUS_FAIL,
        f"native extension import failed (rc={rc})",
        {"stderr_tail": _tail(err)},
    )


def check_platform(root):
    rc, out, err = bundle.run_bundle_python(root, PROBE_PLATFORM, timeout=120)
    name = bundle.extract_tagged(out, "PLATFORM:")
    if rc == 0 and name is not None:
        return (report.STATUS_PASS, None, {"device_name": name})
    return (
        report.STATUS_FAIL,
        f"vllm.platforms import failed (rc={rc})",
        {"stderr_tail": _tail(err)},
    )


def check_device(root, gfx_target):
    rc, out, err = bundle.run_bundle_python(root, PROBE_DEVICE, timeout=120)
    payload = bundle.extract_tagged(out, "DEVICE:")
    if rc != 0 or payload is None:
        return (
            report.STATUS_FAIL,
            f"torch.cuda probe failed (rc={rc})",
            {"stderr_tail": _tail(err)},
        )
    info = json.loads(payload)
    if not info.get("available") or info.get("count", 0) < 1:
        return (report.STATUS_FAIL, "no GPU visible to torch.cuda", info)
    arch = info.get("gcnArchName", "")
    # Concrete targets (gfx1151/gfx1150) should match exactly; umbrella targets
    # (gfx110X/gfx120X) are not run in Tier 1, so only flag a hard mismatch.
    if gfx_target.startswith("gfx") and "X" not in gfx_target:
        if gfx_target not in arch:
            return (
                report.STATUS_FAIL,
                f"gcnArchName '{arch}' does not match target {gfx_target}",
                info,
            )
    return (report.STATUS_PASS, None, info)


def check_amdsmi(root):
    rc, out, err = bundle.run_bundle_python(root, PROBE_AMDSMI, timeout=60)
    arch = bundle.extract_tagged(out, "AMDSMI:")
    if rc == 0 and arch:
        return (report.STATUS_PASS, None, {"target_graphics_version": arch})
    return (
        report.STATUS_WARN,
        "bundled amdsmi could not read ASIC info (known gfx1151 issue)",
        {"stderr_tail": _tail(err)},
    )


def check_launcher(root):
    launcher = os.path.join(root, "bin", "vllm-server")
    try:
        proc = subprocess.run(
            [launcher, "--help"],
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return (report.STATUS_FAIL, "vllm-server --help timed out", {})
    if proc.returncode == 0:
        return (report.STATUS_PASS, None, {})
    return (
        report.STATUS_FAIL,
        f"vllm-server --help exited {proc.returncode}",
        {"stderr_tail": _tail(proc.stderr)},
    )


def detect_versions(root):
    sp = bundle.find_site_packages(root)

    def ver(project):
        for meta in glob.glob(os.path.join(sp, f"{project}-*.dist-info", "METADATA")):
            with open(meta, encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    if line.startswith("Version:"):
                        return line.split(":", 1)[1].strip()
        return None

    torch_v = ver("torch")
    rocm_v = None
    if torch_v and "rocm" in torch_v:
        import re

        match = re.search(r"rocm([\d.]+)", torch_v)
        rocm_v = match.group(1) if match else None
    return ver("vllm"), torch_v, rocm_v


def main():
    parser = argparse.ArgumentParser(description="Tier 1 hardware smoke test")
    parser.add_argument("--bundle-root", required=True)
    parser.add_argument("--gfx-target", required=True)
    parser.add_argument("--channel", default=None, choices=[None, "stable", "nightly"])
    parser.add_argument("--candidate-tag", default=None)
    parser.add_argument("--run-id", default=os.environ.get("GITHUB_RUN_ID"))
    parser.add_argument("--run-attempt", default=os.environ.get("GITHUB_RUN_ATTEMPT"))
    parser.add_argument("--output", default="tier1-report.json")
    parser.add_argument("--fail-on-error", action="store_true")
    args = parser.parse_args()

    root = os.path.abspath(args.bundle_root)
    vllm_v, torch_v, rocm_v = detect_versions(root)

    meta = report.build_meta(
        gfx_target=args.gfx_target,
        channel=args.channel,
        vllm_version=vllm_v,
        torch_version=torch_v,
        rocm_version=rocm_v,
        candidate_tag=args.candidate_tag,
        hardware_validated=True,
        run_id=args.run_id,
        run_attempt=args.run_attempt,
    )

    tier = report.TierReport("tier1", meta)
    tier.run("T1.1", "native dlopen", lambda: check_native(root))
    tier.run("T1.2", "platform import", lambda: check_platform(root))
    tier.run("T1.3", "device visibility", lambda: check_device(root, args.gfx_target))
    tier.run("T1.4", "amdsmi gcn-arch", lambda: check_amdsmi(root), gating=False)
    tier.run("T1.5", "launcher --help", lambda: check_launcher(root))

    tier.write(args.output)
    tier.print_summary()
    print(f"\nWrote {args.output}")

    if args.fail_on_error and tier.rollup_status() == report.STATUS_FAIL:
        sys.exit(1)


if __name__ == "__main__":
    main()
