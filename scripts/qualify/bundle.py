#!/usr/bin/env python3
"""
Shared helpers for locating and invoking a portable vLLM-ROCm bundle.

The bundle layout (produced by build-vllm-rocm.yml) is:

    <root>/bin/python3            portable interpreter
    <root>/bin/vllm-server        bash launcher (sets LD_LIBRARY_PATH etc.)
    <root>/lib/python3*/site-packages/{vllm,torch,_rocm_sdk_*}

Tier 1/2 run probes inside the bundle's interpreter as *subprocesses* so a
segfault or hard crash in one probe (a real risk with mismatched native
extensions) is contained and recorded rather than taking down the tier runner.
"""

import glob
import os
import subprocess


def find_site_packages(root):
    matches = sorted(glob.glob(os.path.join(root, "lib", "python3*", "site-packages")))
    if not matches:
        raise RuntimeError(f"no site-packages found under {root}/lib/python3*/")
    return matches[0]


def python_bin(root):
    matches = sorted(glob.glob(os.path.join(root, "bin", "python3*")))
    # Prefer the bare "python3" symlink if present.
    for match in matches:
        if os.path.basename(match) == "python3":
            return match
    if matches:
        return matches[0]
    raise RuntimeError(f"no python3 found under {root}/bin/")


def launcher_env(root):
    """Replicate the environment that bin/vllm-server sets up."""
    sp = find_site_packages(root)
    lib_dirs = []
    lib_dirs += glob.glob(os.path.join(sp, "_rocm_sdk_*", "lib"))
    torch_lib = os.path.join(sp, "torch", "lib")
    if os.path.isdir(torch_lib):
        lib_dirs.append(torch_lib)
    llvm_lib = os.path.join(sp, "_rocm_sdk_core", "lib", "llvm", "lib")
    if os.path.isdir(llvm_lib):
        lib_dirs.insert(0, llvm_lib)

    env = dict(os.environ)
    existing = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = ":".join(lib_dirs + ([existing] if existing else []))

    amd_smi = os.path.join(sp, "_rocm_sdk_core", "share", "amd_smi")
    pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = ":".join([amd_smi] + ([pythonpath] if pythonpath else []))

    env["FLASH_ATTENTION_TRITON_AMD_ENABLE"] = "TRUE"
    return env


def run_bundle_python(root, code, timeout=120):
    """Run `code` in the bundle interpreter. Returns (rc, stdout, stderr)."""
    proc = subprocess.run(
        [python_bin(root), "-c", code],
        env=launcher_env(root),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    return proc.returncode, proc.stdout, proc.stderr


def extract_tagged(stdout, tag):
    """Pull the payload from a line like 'TAG:payload' emitted by a probe."""
    for line in stdout.splitlines():
        if line.startswith(tag):
            return line[len(tag):]
    return None
