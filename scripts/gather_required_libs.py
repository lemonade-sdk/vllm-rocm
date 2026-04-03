#!/usr/bin/env python3
"""
Discover and copy ROCm shared libraries required by the vLLM server.

Iteratively runs the vllm-server launcher, detects missing .so errors,
finds the library inside the ROCm installation, and copies it to the
destination directory. Repeat until no more missing-library errors.

Usage:
    python gather_required_libs.py --rocm-dir /opt/rocm --dest-dir /opt/vllm/lib
"""

import subprocess
import os
import shutil
import argparse


def find_lib_in_rocm(libname, rocm_dir):
    """Walk the ROCm tree to locate *libname*."""
    for root, _, files in os.walk(rocm_dir):
        if libname in files:
            return os.path.join(root, libname)
    raise RuntimeError(f"Could not find {libname} in {rocm_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Gather required ROCm libraries for vllm-server"
    )
    parser.add_argument(
        "--rocm-dir",
        default="/opt/rocm",
        help="Path to ROCm installation directory (default: /opt/rocm)",
    )
    parser.add_argument(
        "--dest-dir",
        default="/opt/vllm/lib",
        help="Destination directory for libraries (default: /opt/vllm/lib)",
    )
    parser.add_argument(
        "--vllm-server",
        default="/opt/vllm/bin/vllm-server",
        help="Path to vllm-server launcher (default: /opt/vllm/bin/vllm-server)",
    )

    args = parser.parse_args()

    rocm_dir = args.rocm_dir
    dest_dir = args.dest_dir
    binary = args.vllm_server

    os.makedirs(dest_dir, exist_ok=True)

    # Quick sanity import — triggers .so loading without needing a GPU
    cmd = [binary, "--help"]
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = dest_dir + ":" + env.get("LD_LIBRARY_PATH", "")

    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    print(f"Initial stderr: {result.stderr[:500]}")

    iterations = 0
    while "error while loading shared libraries" in result.stderr:
        so_file = result.stderr.split("shared libraries: ")[1].split(": ")[0]
        so_file_path = find_lib_in_rocm(so_file, rocm_dir)
        shutil.copy2(so_file_path, dest_dir)
        print(f"Copied {so_file_path} -> {dest_dir}")
        result = subprocess.run(cmd, capture_output=True, text=True, env=env)
        iterations += 1
        if iterations > 100:
            print("Too many iterations, aborting")
            break

    if "error while loading shared libraries" not in result.stderr:
        print(f"All libraries resolved after {iterations} copies")
    else:
        print(f"Still have errors after {iterations} copies: {result.stderr[:300]}")


if __name__ == "__main__":
    main()
