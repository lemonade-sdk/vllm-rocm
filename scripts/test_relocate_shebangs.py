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
sys.path.insert(0, str(SCRIPT.parent))
import relocate_shebangs as rs  # noqa: E402


class Base(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.bindir = self.tmp / "bundle" / "bin"
        self.bindir.mkdir(parents=True)
        shutil.copy(sys.executable, self.bindir / "python3")
        os.chmod(self.bindir / "python3", 0o755)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def write(self, name: str, body: bytes, mode: int = 0o755) -> Path:
        p = self.bindir / name
        p.write_bytes(body)
        os.chmod(p, mode)
        return p

    def script(self, name: str, interp: bytes = b"python3", tail: bytes = b"print('ok')\n") -> Path:
        return self.write(name, b"#!" + str(self.bindir).encode() + b"/" + interp + b"\n" + tail)

    def run_cli(self, *extra: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, str(SCRIPT), str(self.bindir), *extra],
            capture_output=True, text=True,
        )


class TestRewrite(Base):
    def test_relocates_and_still_runs_after_the_tree_moves(self) -> None:
        self.script("demo", tail=b"import sys; print('DEMO', sys.argv[1:])\n")
        self.assertEqual(self.run_cli().returncode, 0)
        moved = self.tmp / "moved"
        (self.tmp / "bundle").rename(moved)
        out = subprocess.run([str(moved / "bin" / "demo"), "a"], capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn("DEMO", out.stdout)

    def test_body_preserved_and_mode_preserved(self) -> None:
        tail = b"# c\nimport os\nprint(os.name)\n"
        p = self.script("keeps", tail=tail)
        os.chmod(p, 0o750)
        self.run_cli()
        self.assertTrue(p.read_bytes().endswith(tail))
        self.assertEqual(p.stat().st_mode & 0o777, 0o750)

    def test_shebang_only_file_does_not_crash(self) -> None:
        p = self.write("nonewline", b"#!" + str(self.bindir).encode() + b"/python3")
        r = self.run_cli()
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertTrue(p.read_bytes().startswith(b"#!/bin/sh\n"))

    def test_interpreter_basename_is_preserved_not_hardcoded(self) -> None:
        """qwen: a bundle built against python3.13 must keep python3.13."""
        shutil.copy(sys.executable, self.bindir / "python3.13")
        os.chmod(self.bindir / "python3.13", 0o755)
        p = self.script("versioned", interp=b"python3.13")
        r = self.run_cli()
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn(b"/python3.13\"", p.read_bytes())

    def test_dotdot_spelling_is_normalised_and_rewritten(self) -> None:
        """A ..-containing path names the same interpreter; rewrite it."""
        p = self.write(
            "dotdot",
            b"#!" + str(self.bindir).encode() + b"/../bin/python3\nprint(1)\n",
        )
        r = self.run_cli()
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertTrue(p.read_bytes().startswith(b"#!/bin/sh\n"))


class TestLeavesAlone(Base):
    def _unchanged(self, p: Path, before: bytes, r: subprocess.CompletedProcess) -> None:
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(p.read_bytes(), before)

    def test_non_python_bundle_helper_untouched_AND_run_succeeds(self) -> None:
        """deepseek+qwen blocker: the audit must not punish a deliberate skip."""
        p = self.write("python3-config", b"#!" + str(self.bindir).encode() + b"/python3-config\nx\n")
        before = p.read_bytes()
        self.script("real")
        self._unchanged(p, before, self.run_cli())

    def test_foreign_interpreter_untouched(self) -> None:
        p = self.write("systempy", b"#!/usr/bin/python3\nprint(1)\n")
        before = p.read_bytes()
        self.script("real")
        self._unchanged(p, before, self.run_cli())

    def test_shell_script_untouched(self) -> None:
        p = self.write("plainsh", b"#!/bin/sh\necho hi\n")
        before = p.read_bytes()
        self.script("real")
        self._unchanged(p, before, self.run_cli())

    def test_symlink_untouched(self) -> None:
        self.script("real")
        link = self.bindir / "alias"
        link.symlink_to(self.bindir / "real")
        self.run_cli()
        self.assertTrue(link.is_symlink())

    def test_non_utf8_coding_cookie_is_skipped(self) -> None:
        """qwen: the trampoline displaces line 2, so a real encoding is honoured."""
        p = self.write(
            "cookie",
            b"#!" + str(self.bindir).encode() + b"/python3\n# -*- coding: latin-1 -*-\nx=1\n",
        )
        before = p.read_bytes()
        self.script("real")
        r = self.run_cli()
        self._unchanged(p, before, r)
        self.assertIn("coding declaration", r.stdout)

    def test_utf8_cookie_IS_rewritten(self) -> None:
        """EVERY real pip console script carries this line.

        An earlier revision skipped any coding declaration, which skipped the
        entire bundle and failed the build. Only a non-default encoding matters,
        because Python 3 already defaults to utf-8.
        """
        p = self.write(
            "pipshaped",
            b"#!" + str(self.bindir).encode()
            + b"/python3\n# -*- coding: utf-8 -*-\nimport sys\nprint('ok')\n",
        )
        r = self.run_cli()
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertTrue(p.read_bytes().startswith(b"#!/bin/sh\n"))
        self.assertIn(b"coding: utf-8", p.read_bytes())

    def test_sibling_directory_prefix_does_not_match(self) -> None:
        """qwen: <bin>foo must not satisfy a <bin> prefix test."""
        fields = [str(self.bindir).encode() + b"foo/python3"]
        self.assertIsNone(rs.bundle_interpreter(fields, rs.interpreter_prefixes(self.bindir)))


class TestFailsLoudly(Base):
    def test_shebang_with_arguments_is_refused(self) -> None:
        """qwen: dropping `-s` would silently change behaviour."""
        self.write("withargs", b"#!" + str(self.bindir).encode() + b"/python3 -s\nprint(1)\n")
        r = self.run_cli()
        self.assertEqual(r.returncode, 1)
        self.assertIn("carrying interpreter arguments", r.stderr)

    def test_genuine_no_op_fails(self) -> None:
        self.write("plainsh", b"#!/bin/sh\necho hi\n")
        r = self.run_cli()
        self.assertEqual(r.returncode, 1)
        self.assertIn("rewrote nothing", r.stderr)

    def test_missing_directory_is_an_error(self) -> None:
        r = subprocess.run(
            [sys.executable, str(SCRIPT), str(self.tmp / "nope")], capture_output=True, text=True
        )
        self.assertEqual(r.returncode, 2)

    def test_broken_interpreter_is_reported(self) -> None:
        self.script("real")
        (self.bindir / "python3").unlink()
        (self.bindir / "python3").write_text("not an interpreter\n")
        os.chmod(self.bindir / "python3", 0o755)
        r = self.run_cli()
        self.assertEqual(r.returncode, 1)
        self.assertIn("does not execute", r.stderr)

    def test_straggler_invariant_reports(self) -> None:
        """The audit still fires if a should-have-been file survives a pass."""
        self.write("missed", b"#!" + str(self.bindir).encode() + b"/python3\nx\n")
        found = rs.stragglers(self.bindir, excused=set())
        self.assertIn("missed", found)


class TestRerun(Base):
    def test_second_run_is_nothing_to_do_not_a_failure(self) -> None:
        """qwen: an idempotent re-run must not be indistinguishable from a no-op."""
        self.script("twice")
        self.assertEqual(self.run_cli().returncode, 0)
        again = self.run_cli()
        self.assertEqual(again.returncode, 0, again.stderr)
        self.assertIn("already relocated", again.stdout)

    def test_probe_leaves_no_residue(self) -> None:
        self.script("real")
        self.assertEqual(self.run_cli().returncode, 0)
        leftovers = [p.name for p in self.bindir.iterdir() if p.name.startswith(".relocate-probe")]
        self.assertEqual(leftovers, [])


if __name__ == "__main__":
    unittest.main(verbosity=1)
