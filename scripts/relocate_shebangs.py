#!/usr/bin/env python3
"""Make a bundle's console-script shebangs relocatable.

pip records the interpreter it was invoked with in every console script it
generates, as an absolute path. In a bundle built in one place and extracted
somewhere else that path does not exist, so execve() fails. The error names the
script rather than the interpreter, so it reads as a missing file:

    FileNotFoundError: [Errno 2] No such file or directory: '<bundle>/bin/hipconfig'

...on a file that is present and executable.

For each console script whose interpreter is a Python inside the bundle, the
shebang is replaced with:

    #!/bin/sh
    '''exec' "$(dirname "$(readlink -f "$0")")/<interpreter>" "$0" "$@"
    ' '''

which is valid sh (it execs the sibling interpreter) and a no-op string
expression in Python, so the script still parses and runs. The interpreter
basename is carried over from the original shebang rather than assumed, so a
bundle built against python3.13 keeps python3.13.

Fails the build if nothing was rewritten and nothing had been rewritten
previously, if a script that should have been rewritten was not, or if the
rewritten form does not execute. A silent no-op here ships a broken bundle
behind a green pipeline.

Linux-oriented: the trampoline uses `readlink -f`.
"""

from __future__ import annotations

import argparse
import os
import re
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

# python, python3, python3.13, and suffixed builds (python3.13t free-threaded,
# python3.10m). Anchored so bundle-local helpers such as python3-config, whose
# names merely start with "python", are not treated as interpreters.
PYTHON_NAME = re.compile(rb"^python[0-9]*(\.[0-9]+)*[a-z]?$")

# A coding declaration is honoured only on line 1 or 2, and the trampoline
# pushes line 2 down. Displacing a utf-8 declaration is a no-op because Python 3
# already defaults to utf-8 -- and every pip console script carries exactly that
# line, so treating it as a blocker would skip the entire bundle. Only a
# declaration of some OTHER encoding actually changes how the file is read.
CODING_COOKIE = re.compile(rb"^[ \t\f]*#.*coding[:=][ \t]*([-_.a-zA-Z0-9]+)")
DEFAULT_ENCODINGS = {b"utf-8", b"utf8", b"ascii", b"us-ascii"}


def displaces_a_real_encoding(second_line: bytes) -> bool:
    """True if line 2 declares an encoding that is not Python 3's default."""
    match = CODING_COOKIE.match(second_line)
    if not match:
        return False
    return match.group(1).lower().replace(b"_", b"-") not in DEFAULT_ENCODINGS

RELOCATED_FIRST_LINE = b"#!/bin/sh\n"
RELOCATED_MARKER = b"readlink -f"


def trampoline(interpreter_name: bytes) -> bytes:
    """The replacement shebang, carrying the original interpreter's basename."""
    return (
        b"#!/bin/sh\n"
        b"'''exec' \"$(dirname \"$(readlink -f \"$0\")\")/"
        + interpreter_name
        + b"\" \"$0\" \"$@\"\n"
        b"' '''\n"
    )


def interpreter_prefixes(bindir: Path) -> set[bytes]:
    """Both spellings of the bin dir: pip may record either the literal path or
    its resolution, if the build root contains a symlinked component."""
    return {str(bindir).encode(), str(bindir.resolve()).encode()}


def shebang_fields(first_line: bytes) -> list[bytes] | None:
    """Whitespace-separated shebang fields, or None if this is not a shebang."""
    line = first_line.split(b"\n", 1)[0].rstrip(b"\r")
    if not line.startswith(b"#!"):
        return None
    fields = line[2:].split()
    return fields or None


def bundle_interpreter(fields: list[bytes], prefixes: set[bytes]) -> bytes | None:
    """The interpreter basename if it is a Python inside the bundle, else None.

    The path is normalised first, so a `..`-containing spelling of the same
    interpreter is recognised rather than skipped and later reported missed.
    The trailing separator makes this a path-boundary test, so a sibling
    directory such as `<bin>foo` cannot match.
    """
    normalised = os.path.normpath(fields[0])
    for prefix in prefixes:
        boundary = prefix if prefix.endswith(b"/") else prefix + b"/"
        if not normalised.startswith(boundary):
            continue
        name = normalised[len(boundary):]
        if b"/" in name:
            return None
        return name if PYTHON_NAME.match(name) else None
    return None


def already_relocated(path: Path) -> bool:
    with path.open("rb") as handle:
        if handle.readline() != RELOCATED_FIRST_LINE:
            return False
        return RELOCATED_MARKER in handle.readline()


def candidates(bindir: Path) -> list[Path]:
    return [p for p in sorted(bindir.iterdir()) if p.is_file() and not p.is_symlink()]


def relocate(bindir: Path) -> tuple[list[str], list[str], list[str]]:
    """Rewrite matching shebangs.

    Returns (rewritten, refused_carrying_args, skipped_coding_cookie).
    """
    prefixes = interpreter_prefixes(bindir)
    rewritten: list[str] = []
    with_args: list[str] = []
    cookies: list[str] = []

    for path in candidates(bindir):
        try:
            raw = path.read_bytes()
        except OSError:
            continue

        first, sep, rest = raw.partition(b"\n")
        fields = shebang_fields(first)
        if fields is None:
            continue
        name = bundle_interpreter(fields, prefixes)
        if name is None:
            continue

        if len(fields) > 1:
            # e.g. `#!<bundle>/bin/python3 -s`. Dropping the flags would change
            # behaviour silently, so refuse rather than guess.
            with_args.append(path.name)
            continue

        second = rest.split(b"\n", 1)[0] if sep else b""
        if displaces_a_real_encoding(second):
            cookies.append(path.name)
            continue

        mode = path.stat().st_mode
        path.write_bytes(trampoline(name) + rest)
        os.chmod(path, stat.S_IMODE(mode))
        rewritten.append(path.name)

    return rewritten, with_args, cookies


def stragglers(bindir: Path, excused: set[str]) -> list[str]:
    """Scripts that should have been rewritten but still name a build-time path.

    Uses the SAME predicate as the rewriter, so a file the rewriter deliberately
    left alone is not reported. `excused` carries the names it declined on
    purpose this run.
    """
    prefixes = interpreter_prefixes(bindir)
    found = []
    for path in candidates(bindir):
        if path.name in excused:
            continue
        try:
            with path.open("rb") as handle:
                first = handle.readline()
        except OSError:
            continue
        fields = shebang_fields(first)
        if fields is None:
            continue
        if bundle_interpreter(fields, prefixes) is not None:
            found.append(path.name)
    return found


def count_relocated(bindir: Path) -> int:
    total = 0
    for path in candidates(bindir):
        try:
            if already_relocated(path):
                total += 1
        except OSError:
            continue
    return total


def smoke_test(bindir: Path, interpreter_name: bytes) -> str | None:
    """Execute a throwaway script carrying the trampoline; return an error or None.

    Proves the trampoline form parses and execs in this directory. It does NOT
    prove any particular rewritten script still behaves correctly.
    """
    fd, name = tempfile.mkstemp(prefix=".relocate-probe-", dir=bindir)
    probe = Path(name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(trampoline(interpreter_name) + b"print('PROBE_OK')\n")
        os.chmod(probe, 0o700)
        run = subprocess.run([str(probe)], capture_output=True, text=True, timeout=120)
        if run.returncode != 0 or "PROBE_OK" not in run.stdout:
            return f"probe exited {run.returncode}: {run.stderr.strip()[:200]}"
        return None
    except (OSError, subprocess.SubprocessError) as exc:
        return f"probe could not run: {exc}"
    finally:
        try:
            probe.unlink()
        except OSError:
            pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bindir", type=Path, help="the bundle's bin/ directory")
    parser.add_argument(
        "--allow-nothing-to-do",
        action="store_true",
        help="tolerate a run with no rewrites and nothing previously rewritten",
    )
    args = parser.parse_args()

    bindir = args.bindir
    if not bindir.is_dir():
        print(f"error: not a directory: {bindir}", file=sys.stderr)
        return 2

    rewritten, with_args, cookies = relocate(bindir)
    print(f"rewrote {len(rewritten)} console-script shebangs")
    for entry in cookies:
        print(f"  skipped (non-utf-8 coding declaration on line 2): {entry}")

    if with_args:
        for entry in with_args:
            print(f"  refused (shebang carries arguments): {entry}", file=sys.stderr)
        print(
            "error: refusing to rewrite a shebang carrying interpreter arguments, "
            "because they would be dropped. Handle these explicitly.",
            file=sys.stderr,
        )
        return 1

    left = stragglers(bindir, excused=set(cookies))
    if left:
        print(
            "error: these scripts should have been rewritten but still name a "
            "build-time interpreter: " + ", ".join(left),
            file=sys.stderr,
        )
        return 1

    if not rewritten:
        previously = count_relocated(bindir)
        if previously:
            print(f"nothing to do: {previously} script(s) already relocated")
            return 0
        if not args.allow_nothing_to_do:
            print(
                "error: rewrote nothing and found nothing previously rewritten. "
                "Either the interpreter path pip recorded no longer matches this "
                "bin directory, or the wrong directory was passed. Failing rather "
                "than shipping a bundle whose console scripts were never checked.",
                file=sys.stderr,
            )
            return 1
        print("warning: nothing to do, continuing at your request")
        return 0

    interpreter = b"python3"
    for path in candidates(bindir):
        if path.name == rewritten[0]:
            with path.open("rb") as handle:
                handle.readline()
                # the two closing parens are separated by a quote -- )")/ -- so anchor on )/
                match = re.search(rb'\)/([^"]+)"', handle.readline())
            if match:
                interpreter = match.group(1)
            break

    failure = smoke_test(bindir, interpreter)
    if failure is not None:
        print(f"error: the rewritten shebang does not execute -- {failure}", file=sys.stderr)
        return 1

    print(f"verified: trampoline execs {interpreter.decode()} in {bindir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
