#!/usr/bin/env python3
"""Fail if any publishable workspace crate has a dependency that `cargo publish`
cannot upload — the gap behind the H14 release bug.

`cargo publish` rewrites every path/git dependency to a crates.io version
requirement and refuses to publish a crate whose dependency lacks one (or pins a
wildcard `*`). A git dependency is doubly broken: even when a same-named crate
exists on crates.io, the published crate would silently depend on the *registry*
version, shipping different code than was built and tested locally. The original
release tooling checked versions, membership, and topological order, but never
this — so a `vX.Y.Z` tag would publish the leading crates and then hard-fail
mid-batch at the first crate carrying such a dependency, leaving an
irreversible, un-rollback-able partial release on crates.io.

A dependency is a publish BLOCKER when it is a normal/build dependency
(dev-dependencies are stripped on publish, so they never block) of a publishable
crate AND either:

  * its version requirement is a bare wildcard (``req == "*"``) — cargo refuses
    wildcard requirements on crates.io, and a path/git dep with no ``version =``
    surfaces here as ``*``; or
  * its source is a git source (``git+...``) — the git spec is dropped on
    publish, so the upload needs a registry version that may not exist and, if it
    does, points at different code.

Internal workspace path deps are *not* flagged: they carry a real version
requirement (e.g. ``^0.4.0``) from ``[workspace.dependencies]`` and a registry
version is published for each, in topological order, ahead of its dependents.

Usage:
    cargo metadata --format-version 1 | \\
        python3 scripts/check_dep_publishability.py [name ...]

If crate names are given (the explicit publish list), only those are checked;
otherwise every publishable workspace member is checked. Exits 0 when clean, 1
when any blocker is found (printing one line per blocker).
"""

import json
import sys


def _is_git_source(source):
    return bool(source) and source.startswith("git+")


def find_blockers(metadata, restrict_to=None):
    """Return a list of (crate, dep, reason) for publish-blocking dependencies.

    ``metadata`` is the parsed ``cargo metadata --format-version 1`` document.
    ``restrict_to`` optionally limits the crates checked to that set of names
    (intersected with the publishable members); ``None`` checks all publishable
    members.
    """
    id2pkg = {p["id"]: p for p in metadata["packages"]}
    workspace = set(metadata["workspace_members"])

    # Publishable = workspace member without `publish = false`, which cargo
    # metadata reports as an empty publish list.
    publishable = {}
    for pid in workspace:
        pkg = id2pkg[pid]
        if pkg.get("publish") == []:
            continue
        publishable[pkg["name"]] = pkg

    names = set(publishable)
    if restrict_to is not None:
        names &= set(restrict_to)

    blockers = []
    for name in sorted(names):
        pkg = publishable[name]
        for dep in pkg["dependencies"]:
            # dev-dependencies are dropped from the published manifest, so they
            # never block a publish. Normal (None) and build deps do.
            if dep.get("kind") not in (None, "build"):
                continue
            source = dep.get("source") or ""
            req = dep.get("req", "*")
            if _is_git_source(source):
                blockers.append(
                    (
                        name,
                        dep["name"],
                        "git dependency (req %r, source %s) — cargo publish "
                        "drops the git spec and needs a crates.io version"
                        % (req, source),
                    )
                )
            elif req == "*":
                blockers.append(
                    (
                        name,
                        dep["name"],
                        "wildcard/no version requirement (req '*') — cargo "
                        "refuses to publish a crate whose dependency lacks a "
                        "version",
                    )
                )
    return blockers


def main(argv):
    restrict_to = argv[1:] or None
    metadata = json.load(sys.stdin)
    blockers = find_blockers(metadata, restrict_to)
    if not blockers:
        print(
            "  OK — every publishable crate's dependencies carry a crates.io "
            "version requirement"
        )
        return 0
    print("  UNPUBLISHABLE dependencies (cargo publish would fail mid-batch):")
    for crate, dep, reason in blockers:
        print("    - %s -> %s: %s" % (crate, dep, reason))
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
