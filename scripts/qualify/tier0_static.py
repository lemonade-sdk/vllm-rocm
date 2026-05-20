#!/usr/bin/env python3
"""
Tier 0 — static bundle verification (no GPU required).

Runs on the build runner immediately after the portable vLLM bundle is
assembled, before it is uploaded/released. Catches the class of packaging bug
that shipped in vllm0.21.0-rocm7.13.0, where the build force-reinstalled an
AMD torch whose C++ ABI (c10::cuda::*) no longer matched the symbols vLLM's
prebuilt _C.abi3.so was linked against (c10::hip::*).

Checks:
  T0.1  torch version-pin consistency  — vLLM's Requires-Dist torch pin vs the
        torch actually bundled (release segment must match).
  T0.2  native-ext symbol satisfaction — every undefined c10::/at::/torch::
        symbol in vllm/_C.abi3.so and _rocm_C.abi3.so is defined by some
        bundled torch/ROCm library.
  T0.3  DT_NEEDED resolution           — each NEEDED soname of the native exts
        is either a base system lib or present in the bundle (non-gating warn).
  T0.4  structural manifest            — required files exist; launcher parses.
  T0.6  amdsmi path sanity             — bundled amdsmi package is present.

Usage:
    python tier0_static.py --bundle-root /opt/vllm \
        --gfx-target gfx1151 --output tier0-report.json
"""

import argparse
import glob
import os
import re
import subprocess
import sys

import report

# Sonames we expect to inherit from the host base system rather than the bundle.
BASE_LIBS = {
    "libc.so.6",
    "libm.so.6",
    "libdl.so.2",
    "librt.so.1",
    "libpthread.so.0",
    "libstdc++.so.6",
    "libgcc_s.so.1",
    "ld-linux-x86-64.so.2",
    "libutil.so.1",
    "libresolv.so.2",
}

# Mangled-name fragments that identify the torch/c10/at C++ namespaces. These
# are the symbols whose ABI drifts between torch builds, so they are what T0.2
# verifies. (e.g. _ZN3c103hip19getCurrentHIPStreamEa contains "3c10".)
TORCH_NS_FRAGMENTS = ("3c10", "2at", "5torch")

NATIVE_EXTS = ("vllm/_C.abi3.so", "vllm/_rocm_C.abi3.so")


def find_site_packages(root):
    matches = sorted(glob.glob(os.path.join(root, "lib", "python3*", "site-packages")))
    if not matches:
        raise RuntimeError(f"no site-packages found under {root}/lib/python3*/")
    return matches[0]


def read_metadata_field(metadata_path, field):
    with open(metadata_path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith(field + ":"):
                return line.split(":", 1)[1].strip()
    return None


def dist_info_version(site_packages, project):
    matches = glob.glob(os.path.join(site_packages, f"{project}-*.dist-info", "METADATA"))
    for path in matches:
        version = read_metadata_field(path, "Version")
        if version:
            return version
    return None


def vllm_torch_requirement(site_packages):
    """Return the torch version vLLM pins via Requires-Dist, e.g. '2.10.0+git…'."""
    matches = glob.glob(os.path.join(site_packages, "vllm-*.dist-info", "METADATA"))
    for path in matches:
        with open(path, encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if line.startswith("Requires-Dist:"):
                    req = line.split(":", 1)[1].strip()
                    match = re.match(r"torch\s*==\s*([^\s;]+)", req)
                    if match:
                        return match.group(1)
    return None


def release_segment(version):
    """PEP 440 release segment only, e.g. '2.10.0+git8514f05' -> '2.10.0'."""
    if not version:
        return None
    match = re.match(r"(\d+(?:\.\d+)*)", version)
    return match.group(1) if match else None


def rocm_from_torch(torch_version):
    if not torch_version:
        return None
    match = re.search(r"rocm([\d.]+)", torch_version)
    return match.group(1) if match else None


def _run(cmd):
    return subprocess.run(
        cmd, capture_output=True, text=True, check=False
    )


def nm_symbols(so_path, mode):
    """Return the set of symbols. mode='undef' -> undefined, 'def' -> defined."""
    flag = "-u" if mode == "undef" else "--defined-only"
    proc = _run(["nm", "-D", flag, so_path])
    syms = set()
    for line in proc.stdout.splitlines():
        parts = line.split()
        if parts:
            syms.add(parts[-1])
    return syms


def readelf_needed(so_path):
    proc = _run(["readelf", "-d", so_path])
    needed = []
    for line in proc.stdout.splitlines():
        if "(NEEDED)" in line:
            match = re.search(r"\[([^\]]+)\]", line)
            if match:
                needed.append(match.group(1))
    return needed


def demangle(symbols):
    if not symbols:
        return {}
    proc = _run(["c++filt"] + list(symbols))
    if proc.returncode != 0:
        return {s: s for s in symbols}
    out = proc.stdout.splitlines()
    return dict(zip(symbols, out))


# --------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------


def check_torch_pin(sp):
    required = vllm_torch_requirement(sp)
    installed = dist_info_version(sp, "torch")
    req_rel = release_segment(required)
    inst_rel = release_segment(installed)
    details = {
        "vllm_requires_torch": required,
        "bundled_torch": installed,
        "required_release": req_rel,
        "bundled_release": inst_rel,
    }
    if not req_rel or not inst_rel:
        return (
            report.STATUS_WARN,
            "could not determine torch versions from metadata",
            details,
        )
    if req_rel != inst_rel:
        return (
            report.STATUS_FAIL,
            f"vLLM requires torch=={req_rel} but bundle ships {inst_rel}",
            details,
        )
    return (report.STATUS_PASS, None, details)


def _bundled_defined_symbols(sp):
    lib_globs = [
        os.path.join(sp, "torch", "lib", "*.so"),
        os.path.join(sp, "_rocm_sdk_*", "lib", "*.so"),
    ]
    defined = set()
    libs = []
    for pattern in lib_globs:
        for so in glob.glob(pattern):
            libs.append(so)
            defined |= nm_symbols(so, "def")
    return defined, libs


def check_symbol_satisfaction(sp):
    exts = [os.path.join(sp, rel) for rel in NATIVE_EXTS]
    present = [p for p in exts if os.path.exists(p)]
    if not present:
        return (report.STATUS_SKIP, "no native extensions found (see T0.4)", {})

    defined, libs = _bundled_defined_symbols(sp)
    if not defined:
        return (
            report.STATUS_FAIL,
            "no defined symbols found in bundled torch/ROCm libs",
            {"searched_libs": len(libs)},
        )

    checked = 0
    misses = []
    per_ext = {}
    for ext in present:
        undef = nm_symbols(ext, "undef")
        torch_undef = [
            s for s in undef if any(frag in s for frag in TORCH_NS_FRAGMENTS)
        ]
        checked += len(torch_undef)
        ext_misses = sorted(s for s in torch_undef if s not in defined)
        per_ext[os.path.basename(ext)] = {
            "torch_symbols_checked": len(torch_undef),
            "missing": len(ext_misses),
        }
        misses.extend(ext_misses)

    details = {
        "libs_searched": len(libs),
        "torch_symbols_checked": checked,
        "missing_count": len(misses),
        "per_ext": per_ext,
    }
    if misses:
        demangled = demangle(misses)
        details["missing_symbols"] = [
            {"mangled": s, "demangled": demangled.get(s, s)} for s in misses
        ]
        sample = demangled.get(misses[0], misses[0])
        return (
            report.STATUS_FAIL,
            f"{len(misses)} torch ABI symbol(s) unresolved, e.g. {sample}",
            details,
        )
    return (report.STATUS_PASS, None, details)


def check_dt_needed(sp, root):
    search_dirs = [
        os.path.join(sp, "torch", "lib"),
        os.path.join(root, "lib"),
    ]
    search_dirs += glob.glob(os.path.join(sp, "_rocm_sdk_*", "lib"))
    search_dirs += glob.glob(os.path.join(sp, "_rocm_sdk_*", "lib", "llvm", "lib"))

    def resolvable(soname):
        if soname in BASE_LIBS:
            return True
        for directory in search_dirs:
            if os.path.exists(os.path.join(directory, soname)):
                return True
        return False

    exts = [os.path.join(sp, rel) for rel in NATIVE_EXTS]
    present = [p for p in exts if os.path.exists(p)]
    unresolved = {}
    for ext in present:
        missing = [n for n in readelf_needed(ext) if not resolvable(n)]
        if missing:
            unresolved[os.path.basename(ext)] = missing

    details = {"search_dirs": len(search_dirs), "unresolved": unresolved}
    if unresolved:
        flat = sorted({n for names in unresolved.values() for n in names})
        return (
            report.STATUS_WARN,
            f"NEEDED soname(s) not found in bundle: {', '.join(flat)}",
            details,
        )
    return (report.STATUS_PASS, None, details)


def check_structural_manifest(sp, root):
    required = {
        "vllm-server launcher": os.path.join(root, "bin", "vllm-server"),
        "vllm/_C.abi3.so": os.path.join(sp, "vllm", "_C.abi3.so"),
        "vllm/_rocm_C.abi3.so": os.path.join(sp, "vllm", "_rocm_C.abi3.so"),
    }
    missing = [name for name, path in required.items() if not os.path.exists(path)]

    if not glob.glob(os.path.join(root, "bin", "python3*")):
        missing.append("bin/python3*")
    if not glob.glob(os.path.join(sp, "_rocm_sdk_core", "lib", "*.so*")):
        missing.append("_rocm_sdk_core/lib/*.so")

    details = {"missing": missing, "warnings": []}

    launcher = required["vllm-server launcher"]
    if os.path.exists(launcher):
        syntax = _run(["bash", "-n", launcher])
        if syntax.returncode != 0:
            missing.append("vllm-server (bash syntax error)")
            details["launcher_stderr"] = syntax.stderr.strip()

    if os.path.isdir(os.path.join(sp, "pip")):
        details["warnings"].append("pip present in site-packages (bundle bloat)")

    if missing:
        return (report.STATUS_FAIL, f"missing: {', '.join(missing)}", details)
    if details["warnings"]:
        return (report.STATUS_WARN, "; ".join(details["warnings"]), details)
    return (report.STATUS_PASS, None, details)


def check_amdsmi_path(sp):
    amd_smi = os.path.join(sp, "_rocm_sdk_core", "share", "amd_smi")
    details = {"path": amd_smi, "exists": os.path.isdir(amd_smi)}
    if not details["exists"]:
        return (
            report.STATUS_WARN,
            "bundled amdsmi not found at _rocm_sdk_core/share/amd_smi",
            details,
        )
    return (report.STATUS_PASS, None, details)


def main():
    parser = argparse.ArgumentParser(description="Tier 0 static bundle verification")
    parser.add_argument(
        "--bundle-root",
        default="/opt/vllm",
        help="Root of the assembled bundle (contains lib/python3*/site-packages)",
    )
    parser.add_argument("--gfx-target", required=True)
    parser.add_argument("--channel", default=None, choices=[None, "stable", "nightly"])
    parser.add_argument("--candidate-tag", default=None)
    parser.add_argument("--lemonade-ref", default=None)
    parser.add_argument("--run-id", default=os.environ.get("GITHUB_RUN_ID"))
    parser.add_argument(
        "--run-attempt", default=os.environ.get("GITHUB_RUN_ATTEMPT")
    )
    parser.add_argument("--output", default="tier0-report.json")
    parser.add_argument(
        "--fail-on-error",
        action="store_true",
        help="Exit non-zero if the tier rolls up to fail (gating mode).",
    )
    args = parser.parse_args()

    root = os.path.abspath(args.bundle_root)
    sp = find_site_packages(root)

    torch_version = dist_info_version(sp, "torch")
    meta = report.build_meta(
        gfx_target=args.gfx_target,
        channel=args.channel,
        vllm_version=dist_info_version(sp, "vllm"),
        torch_version=torch_version,
        rocm_version=rocm_from_torch(torch_version),
        candidate_tag=args.candidate_tag,
        lemonade_ref=args.lemonade_ref,
        hardware_validated=False,
        run_id=args.run_id,
        run_attempt=args.run_attempt,
    )

    tier = report.TierReport("tier0", meta)
    tier.run("T0.1", "torch version-pin consistency", lambda: check_torch_pin(sp))
    tier.run(
        "T0.2",
        "native-ext symbol satisfaction",
        lambda: check_symbol_satisfaction(sp),
    )
    tier.run(
        "T0.3",
        "DT_NEEDED resolution",
        lambda: check_dt_needed(sp, root),
        gating=False,
    )
    tier.run(
        "T0.4",
        "structural manifest",
        lambda: check_structural_manifest(sp, root),
    )
    tier.run(
        "T0.6", "amdsmi path sanity", lambda: check_amdsmi_path(sp), gating=False
    )

    tier.write(args.output)
    tier.print_summary()
    print(f"\nWrote {args.output}")

    if args.fail_on_error and tier.rollup_status() == report.STATUS_FAIL:
        sys.exit(1)


if __name__ == "__main__":
    main()
