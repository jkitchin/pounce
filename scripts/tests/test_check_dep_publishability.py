#!/usr/bin/env python3
"""Tests for scripts/check_dep_publishability.py (H14 regression guard).

These exercise the detection logic against synthetic `cargo metadata` documents
so they do not depend on the live workspace state (which is itself blocked by
the feral git pin today). Run directly or via `python3 -m unittest`:

    python3 scripts/tests/test_check_dep_publishability.py
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from check_dep_publishability import find_blockers  # noqa: E402

REGISTRY = "registry+https://github.com/rust-lang/crates.io-index"


def pkg(name, deps, publish=None):
    p = {"id": name + "-id", "name": name, "dependencies": deps}
    if publish is not None:
        p["publish"] = publish
    return p


def dep(name, req="^1.0", source=REGISTRY, kind=None):
    return {"name": name, "req": req, "source": source, "kind": kind}


def metadata(packages):
    return {
        "packages": packages,
        "workspace_members": [p["id"] for p in packages],
    }


class FindBlockersTest(unittest.TestCase):
    def test_clean_workspace_has_no_blockers(self):
        # Internal path dep (source None, real version req) + registry dep.
        m = metadata(
            [
                pkg(
                    "a",
                    [
                        dep("b", req="^0.4.0", source=None),
                        dep("serde", req="1.0", source=REGISTRY),
                    ],
                ),
                pkg("b", []),
            ]
        )
        self.assertEqual(find_blockers(m), [])

    def test_git_dependency_is_flagged(self):
        m = metadata(
            [
                pkg(
                    "a",
                    [dep("feral", req="*", source="git+https://x/feral.git?rev=abc")],
                )
            ]
        )
        blockers = find_blockers(m)
        self.assertEqual(len(blockers), 1)
        crate, name, reason = blockers[0]
        self.assertEqual((crate, name), ("a", "feral"))
        self.assertIn("git dependency", reason)

    def test_wildcard_version_is_flagged(self):
        # A path/registry dep that lost its version requirement surfaces as '*'.
        m = metadata([pkg("a", [dep("b", req="*", source=None)])])
        blockers = find_blockers(m)
        self.assertEqual(len(blockers), 1)
        self.assertIn("version", blockers[0][2])

    def test_dev_dependency_git_dep_is_ignored(self):
        # dev-dependencies are stripped on publish, so they never block.
        m = metadata(
            [
                pkg(
                    "a",
                    [
                        dep(
                            "criterion",
                            req="*",
                            source="git+https://x/criterion.git",
                            kind="dev",
                        )
                    ],
                )
            ]
        )
        self.assertEqual(find_blockers(m), [])

    def test_build_dependency_git_dep_is_flagged(self):
        m = metadata(
            [
                pkg(
                    "a",
                    [dep("gen", req="*", source="git+https://x/gen.git", kind="build")],
                )
            ]
        )
        self.assertEqual(len(find_blockers(m)), 1)

    def test_publish_false_crate_is_not_checked(self):
        # A non-published crate (publish = []) may carry git deps freely.
        m = metadata(
            [
                pkg(
                    "internal",
                    [dep("feral", req="*", source="git+https://x/feral.git")],
                    publish=[],
                )
            ]
        )
        self.assertEqual(find_blockers(m), [])

    def test_restrict_to_limits_checked_crates(self):
        m = metadata(
            [
                pkg("a", [dep("feral", req="*", source="git+https://x/feral.git")]),
                pkg("b", [dep("serde", req="1.0")]),
            ]
        )
        # Restricting to a clean crate hides the blocker in the other one.
        self.assertEqual(find_blockers(m, restrict_to={"b"}), [])
        self.assertEqual(len(find_blockers(m, restrict_to={"a"})), 1)


if __name__ == "__main__":
    unittest.main()
