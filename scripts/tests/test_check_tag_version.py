#!/usr/bin/env python3
"""Tests for scripts/check_tag_version.py (M38 release guard).

These build a synthetic repo tree in a temp dir so they do not depend on the
live manifest versions (which change on every release bump). Run directly or
via `python3 -m unittest`:

    python3 scripts/tests/test_check_tag_version.py
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from check_tag_version import (  # noqa: E402
    check,
    manifest_version,
    parse_tag,
    strip_ref,
)


def write_repo(root, cargo="0.4.0", pysolver="0.4.0", pyomo="0.4.0"):
    """Lay down the three manifests check_tag_version reads."""
    with open(os.path.join(root, "Cargo.toml"), "w") as fh:
        fh.write(
            "[workspace]\n"
            'members = ["crates/*"]\n\n'
            "[workspace.package]\n"
            f'version = "{cargo}"\n'
            'edition = "2021"\n'
        )
    py_dir = os.path.join(root, "python")
    os.makedirs(py_dir)
    with open(os.path.join(py_dir, "pyproject.toml"), "w") as fh:
        fh.write(f'[project]\nname = "pounce-solver"\nversion = "{pysolver}"\n')
    pyomo_dir = os.path.join(root, "pyomo-pounce")
    os.makedirs(pyomo_dir)
    with open(os.path.join(pyomo_dir, "pyproject.toml"), "w") as fh:
        fh.write(f'[project]\nname = "pyomo-pounce"\nversion = "{pyomo}"\n')


class StripRefTest(unittest.TestCase):
    def test_strips_refs_tags_prefix(self):
        self.assertEqual(strip_ref("refs/tags/v0.5.0"), "v0.5.0")

    def test_bare_tag_passes_through(self):
        self.assertEqual(strip_ref("v0.5.0"), "v0.5.0")


class ParseTagTest(unittest.TestCase):
    def test_bare_v_is_crates(self):
        self.assertEqual(
            parse_tag("v0.5.0"),
            ("crates.io workspace", "Cargo.toml", "0.5.0"),
        )

    def test_python_prefix_wins_over_bare_v(self):
        # Longest-prefix-first: must route to the wheel manifest, not Cargo.
        self.assertEqual(
            parse_tag("python-v1.2.3"),
            ("pounce-solver (PyPI)", "python/pyproject.toml", "1.2.3"),
        )

    def test_pyomo_prefix_routes_to_pyomo(self):
        self.assertEqual(
            parse_tag("pyomo-pounce-v9.9.9"),
            ("pyomo-pounce (PyPI)", "pyomo-pounce/pyproject.toml", "9.9.9"),
        )

    def test_prerelease_suffix_accepted(self):
        self.assertEqual(parse_tag("v0.5.0-rc.1")[2], "0.5.0-rc.1")

    def test_non_version_suffix_rejected(self):
        self.assertIsNone(parse_tag("v-not-a-version"))
        self.assertIsNone(parse_tag("vlatest"))

    def test_unknown_prefix_rejected(self):
        self.assertIsNone(parse_tag("release-0.5.0"))


class ManifestVersionTest(unittest.TestCase):
    def test_reads_top_level_version(self):
        text = '[workspace.package]\nversion = "0.4.0"\n'
        self.assertEqual(manifest_version(text), "0.4.0")

    def test_ignores_indented_dependency_versions(self):
        # An indented `version =` inside a dep table must not be picked up.
        text = (
            "[workspace.package]\n"
            'version = "0.4.0"\n\n'
            "[workspace.dependencies]\n"
            '    serde = { version = "1.0" }\n'
        )
        self.assertEqual(manifest_version(text), "0.4.0")

    def test_missing_version_is_none(self):
        self.assertIsNone(manifest_version("[project]\nname = 'x'\n"))


class CheckTest(unittest.TestCase):
    def test_matching_tag_passes(self):
        with tempfile.TemporaryDirectory() as root:
            write_repo(root, cargo="0.4.0")
            code, msg = check("refs/tags/v0.4.0", repo_root=root)
            self.assertEqual(code, 0, msg)

    def test_m38_mismatch_fails(self):
        # The headline bug: tag ahead of the manifest.
        with tempfile.TemporaryDirectory() as root:
            write_repo(root, cargo="0.4.0")
            code, msg = check("refs/tags/v0.5.0", repo_root=root)
            self.assertEqual(code, 2)
            self.assertIn("MISMATCH", msg)
            self.assertIn("0.5.0", msg)
            self.assertIn("0.4.0", msg)

    def test_python_tag_checks_wheel_manifest(self):
        with tempfile.TemporaryDirectory() as root:
            # Cargo at 0.4.0 but the wheel bumped to 0.5.0; a python-v0.5.0
            # tag must validate against the wheel manifest and pass.
            write_repo(root, cargo="0.4.0", pysolver="0.5.0")
            code, msg = check("refs/tags/python-v0.5.0", repo_root=root)
            self.assertEqual(code, 0, msg)

    def test_python_tag_mismatch_fails_even_if_cargo_matches(self):
        with tempfile.TemporaryDirectory() as root:
            write_repo(root, cargo="0.5.0", pysolver="0.4.0")
            code, _ = check("refs/tags/python-v0.5.0", repo_root=root)
            self.assertEqual(code, 2)

    def test_pyomo_tag_checks_pyomo_manifest(self):
        with tempfile.TemporaryDirectory() as root:
            write_repo(root, pyomo="0.4.0")
            code, _ = check("refs/tags/pyomo-pounce-v0.4.0", repo_root=root)
            self.assertEqual(code, 0)

    def test_unrecognized_tag_returns_3(self):
        with tempfile.TemporaryDirectory() as root:
            write_repo(root)
            code, _ = check("refs/tags/nightly", repo_root=root)
            self.assertEqual(code, 3)

    def test_missing_manifest_returns_4(self):
        with tempfile.TemporaryDirectory() as root:
            # No manifests written at all.
            code, _ = check("v0.4.0", repo_root=root)
            self.assertEqual(code, 4)


if __name__ == "__main__":
    unittest.main()
