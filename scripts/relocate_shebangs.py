#!/usr/bin/env python3
"""Make a bundle's console-script shebangs relocatable.

pip writes the interpreter it was invoked with into every console script it
generates, as an absolute path. In a bundle that is built in one place and
extracted somewhere else, that path does not exist on the consumer's machine,
so execve() fails. The error names the script rather than the interpreter, so
it reads as a missing file:

    FileNotFoundError: [Errno 2] No such file or directory: '<bundle>/bin/hipconfig'

...on a file that is present and executable.

This rewrites the shebang of scripts whose interpreter points into the bundle
so that it resolves relative to the script instead:

    #!/bin/sh
    '''exec' "$(dirname "$(readlink -f "$0")")/python3" "$0" "$@"
    ' '''

That is valid sh (line 1-2 exec the sibling interpreter) and a no-op string
expression in Python (lines 2-3 are one triple-quoted string), so the scripts
still parse and run unchanged.

Exits non-zero if it rewrites nothing, or if any script still references the
build-time interpreter afterwards. A build that silently rewrites zero files
would otherwise ship a broken bundle with a green pipeline.
"""

from __future__ import annotations

import argparse
import os
import re
import stat
import subprocess
import sys
from pathlib import Path

TRAMPOLINE = (
    b"#!/bin/sh\n"
    b"'''exec' \"$(dirname \"$(readlink -f \"$0\")\")/python3\" \"$0\" \"$@\"\n"
    b"' '''\n"
)


def interpreter_prefixes(bindir: Path) -> set[bytes]:
    """Both spellings of the bin dir.

    pip records the interpreter path it was invoked with, which may be the
    literal path or its resolution if the build root contains a symlinked
    component. Match either.
    """
    return {str(bindir).encode(), str(bindir.resolve()).encode()}


# python, python3, python3.14 -- but NOT python3-config or any other helper
# that merely starts with "python".
_PYTHON_BASENAME = re.compile(rb"^/python[0-9]*(\.[0-9]+)*$")


def targets_bundle_python(first_line: bytes, prefixes: set[bytes]) -> bool:
    """True if this shebang names a Python interpreter inside the bundle.

    The interpreter basename is matched exactly, not by prefix: a shebang
    naming `.../bin/python3-config`, or any other bundle-local helper, must not
    be rewritten to run under python3, which would change its semantics.
    """
    line = first_line.split(b"\n", 1)[0].rstrip(b"\r")
    if not line.startswith(b"#!"):
        return False
    fields = line[2:].split()
    if not fields:
        return False
    interpreter = fields[0]
    for prefix in prefixes:
        if interpreter.startswith(prefix):
            return bool(_PYTHON_BASENAME.match(interpreter[len(prefix):]))
    return False


def relocate(bindir: Path) -> tuple[int, list[str]]:
    """Rewrite matching shebangs. Returns (count, names)."""
    prefixes = interpreter_prefixes(bindir)
    rewritten: list[str] = []

    for path in sorted(bindir.iterdir()):
        if not path.is_file() or path.is_symlink():
            continue
        try:
            with path.open("rb") as handle:
                first = handle.readline()
        except OSError:
            continue
        if not targets_bundle_python(first, prefixes):
            continue

        # partition, not split-and-index: a file consisting of only a shebang
        # has no newline, and indexing would raise and fail the whole build.
        _, _, rest = path.read_bytes().partition(b"\n")

        mode = path.stat().st_mode
        path.write_bytes(TRAMPOLINE + rest)
        os.chmod(path, stat.S_IMODE(mode))
        rewritten.append(path.name)

    return len(rewritten), rewritten


def audit(bindir: Path) -> list[str]:
    """Names of scripts still pointing at an absolute interpreter in the bundle.

    Catches a partial rewrite, which fail-on-zero cannot see.
    """
    prefixes = interpreter_prefixes(bindir)
    stragglers = []
    for path in sorted(bindir.iterdir()):
        if not path.is_file() or path.is_symlink():
            continue
        try:
            with path.open("rb") as handle:
                first = handle.readline()
        except OSError:
            continue
        if any(first.startswith(b"#!" + p) for p in prefixes):
            stragglers.append(path.name)
    return stragglers


def smoke_test(bindir: Path) -> str | None:
    """Execute a throwaway script carrying the trampoline; return an error or None.

    This cannot prove relocatability in-job, because the build path still
    exists. It proves the trampoline itself parses and execs, which is what
    stands between a mangled heredoc and every consumer of the release.
    """
    probe = bindir / ".relocate-probe"
    try:
        probe.write_bytes(TRAMPOLINE + b"print('PROBE_OK')\n")
        os.chmod(probe, 0o755)
        run = subprocess.run([str(probe)], capture_output=True, text=True, timeout=60)
        if run.returncode != 0 or "PROBE_OK" not in run.stdout:
            return f"probe exited {run.returncode}: {run.stderr.strip()[:200]}"
        return None
    except OSError as exc:
        return f"probe could not run: {exc}"
    finally:
        probe.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bindir", type=Path, help="the bundle's bin/ directory")
    parser.add_argument(
        "--allow-zero",
        action="store_true",
        help="do not fail when nothing needed rewriting (for reruns, not CI)",
    )
    args = parser.parse_args()

    bindir = args.bindir
    if not bindir.is_dir():
        print(f"error: not a directory: {bindir}", file=sys.stderr)
        return 2

    count, names = relocate(bindir)
    print(f"rewrote {count} console-script shebangs")

    if count == 0 and not args.allow_zero:
        print(
            "error: rewrote nothing. Either the interpreter path pip recorded no "
            "longer matches this bin directory, or the wrong directory was passed. "
            "Failing rather than shipping a bundle whose console scripts were never "
            "checked.",
            file=sys.stderr,
        )
        return 1

    stragglers = audit(bindir)
    if stragglers:
        print(
            "error: these scripts still reference the build-time interpreter: "
            + ", ".join(stragglers),
            file=sys.stderr,
        )
        return 1

    failure = smoke_test(bindir)
    if failure is not None:
        print(f"error: the rewritten shebang does not execute -- {failure}", file=sys.stderr)
        return 1

    print(f"verified: trampoline executes in {bindir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
