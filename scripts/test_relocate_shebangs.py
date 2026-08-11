#!/usr/bin/env python3
"""Tests for relocate_shebangs.py. Run: python3 scripts/test_relocate_shebangs.py"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parent / "relocate_shebangs.py"


class Base(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.bindir = self.tmp / "bundle" / "bin"
        self.bindir.mkdir(parents=True)
        # a stand-in interpreter so rewritten scripts can actually run
        shutil.copy(sys.executable, self.bindir / "python3")
        os.chmod(self.bindir / "python3", 0o755)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def write(self, name: str, body: bytes, mode: int = 0o755) -> Path:
        p = self.bindir / name
        p.write_bytes(body)
        os.chmod(p, mode)
        return p

    def console_script(self, name: str, tail: bytes = b"print('ok')\n") -> Path:
        return self.write(name, b"#!" + str(self.bindir).encode() + b"/python3\n" + tail)

    def run_cli(self, *extra: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, str(SCRIPT), str(self.bindir), *extra],
            capture_output=True,
            text=True,
        )


class TestRewrite(Base):
    def test_rewrites_and_script_still_runs_after_relocation(self) -> None:
        self.console_script("demo", b"import sys; print('DEMO', sys.argv[1:])\n")
        self.assertEqual(self.run_cli().returncode, 0)

        moved = self.tmp / "moved"
        (self.tmp / "bundle").rename(moved)
        out = subprocess.run(
            [str(moved / "bin" / "demo"), "a", "b"], capture_output=True, text=True
        )
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn("DEMO", out.stdout)
        self.assertIn("'a', 'b'", out.stdout)

    def test_body_is_preserved_byte_for_byte(self) -> None:
        tail = b"# comment\nimport os\nprint(os.name)\n"
        p = self.console_script("keeps", tail)
        self.run_cli()
        self.assertTrue(p.read_bytes().endswith(tail))

    def test_mode_is_preserved(self) -> None:
        p = self.console_script("moded")
        os.chmod(p, 0o750)
        self.run_cli()
        self.assertEqual(p.stat().st_mode & 0o777, 0o750)

    def test_shebang_only_file_is_rewritten_not_crashed_on(self) -> None:
        """A file that is nothing but a shebang has no newline to split on.

        An earlier revision indexed [1] after a split and raised here, which
        would have failed the whole build step.
        """
        p = self.write("nonewline", b"#!" + str(self.bindir).encode() + b"/python3")
        r = self.run_cli()
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertTrue(p.read_bytes().startswith(b"#!/bin/sh\n"))

    def test_idempotent_second_run_rewrites_zero(self) -> None:
        self.console_script("twice")
        self.assertEqual(self.run_cli().returncode, 0)
        again = self.run_cli("--allow-zero")
        self.assertEqual(again.returncode, 0)
        self.assertIn("rewrote 0", again.stdout)


class TestLeavesAlone(Base):
    def test_foreign_interpreter_untouched(self) -> None:
        p = self.write("systempy", b"#!/usr/bin/python3\nprint(1)\n")
        before = p.read_bytes()
        self.console_script("real")  # so the run has something to do
        self.run_cli()
        self.assertEqual(p.read_bytes(), before)

    def test_non_python_in_bundle_untouched(self) -> None:
        """A bundle-local helper that is not python must keep its interpreter."""
        p = self.write(
            "python3-config", b"#!" + str(self.bindir).encode() + b"/python3-config\nx\n"
        )
        before = p.read_bytes()
        self.console_script("real")
        self.run_cli()
        self.assertEqual(p.read_bytes(), before)

    def test_shell_script_untouched(self) -> None:
        p = self.write("plainsh", b"#!/bin/sh\necho hi\n")
        before = p.read_bytes()
        self.console_script("real")
        self.run_cli()
        self.assertEqual(p.read_bytes(), before)

    def test_symlink_untouched(self) -> None:
        self.console_script("real")
        link = self.bindir / "alias"
        link.symlink_to(self.bindir / "real")
        self.run_cli()
        self.assertTrue(link.is_symlink())


class TestSmokeTest(Base):
    def test_reports_when_the_trampoline_cannot_execute(self) -> None:
        """If the interpreter beside the scripts is unusable, say so."""
        self.console_script("real")
        (self.bindir / "python3").unlink()
        (self.bindir / "python3").write_text("not an interpreter\n")
        os.chmod(self.bindir / "python3", 0o755)
        r = self.run_cli()
        self.assertEqual(r.returncode, 1)
        self.assertIn("does not execute", r.stderr)

    def test_probe_is_cleaned_up(self) -> None:
        self.console_script("real")
        self.assertEqual(self.run_cli().returncode, 0)
        self.assertFalse((self.bindir / ".relocate-probe").exists())


class TestFailsLoudly(Base):
    def test_zero_rewrites_is_an_error(self) -> None:
        self.write("plainsh", b"#!/bin/sh\necho hi\n")
        r = self.run_cli()
        self.assertEqual(r.returncode, 1)
        self.assertIn("rewrote nothing", r.stderr)

    def test_missing_directory_is_an_error(self) -> None:
        r = subprocess.run(
            [sys.executable, str(SCRIPT), str(self.tmp / "nope")],
            capture_output=True,
            text=True,
        )
        self.assertEqual(r.returncode, 2)

    def test_audit_catches_a_partial_rewrite(self) -> None:
        """A straggler left pointing at the build interpreter must fail the run."""
        self.console_script("real")
        straggler = self.bindir / "straggler"
        r = self.run_cli()
        self.assertEqual(r.returncode, 0, r.stderr)
        # a path the matcher genuinely misses: it points into the bundle but the
        # interpreter basename is not a plain python (a ..-containing spelling)
        straggler.write_bytes(
            b"#!" + str(self.bindir).encode() + b"/../bin/python3\nx\n"
        )
        os.chmod(straggler, 0o755)
        r2 = self.run_cli("--allow-zero")
        self.assertEqual(r2.returncode, 1)
        self.assertIn("still reference the build-time interpreter", r2.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
