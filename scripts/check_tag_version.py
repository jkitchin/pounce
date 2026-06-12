#!/usr/bin/env python3
"""Release guard: a pushed tag's version must match the manifest it publishes.

POUNCE cuts a release by pushing a tag, and each release workflow keys off a
distinct tag prefix:

    v<X.Y.Z>               -> crates.io       (root Cargo.toml [workspace.package])
    python-v<X.Y.Z>        -> PyPI pounce-solver (python/pyproject.toml)
    pyomo-pounce-v<X.Y.Z>  -> PyPI pyomo-pounce  (pyomo-pounce/pyproject.toml)

Nothing previously compared the tag against the manifest. A `v0.5.0` tag with
the manifests still at 0.4.0 made the crates.io publish a silent green no-op
(scripts/publish-crates.sh sees every crate already live at 0.4.0 and skips
it) and the PyPI publish ship 0.4.0 under a 0.5.0 release. This guard fails the
release workflow up front when the two disagree.

`scripts/check-release-consistency.sh` checks the three *manifests* agree with
each other; this script checks the *tag* agrees with them. They are
complementary: the former runs on every PR, the latter at tag time.

Usage:
    check_tag_version.py <tag-or-ref>

Accepts a bare tag (`v0.5.0`) or a full ref (`refs/tags/v0.5.0`). Exit codes:
    0  tag version matches the manifest
    2  version mismatch (the M38 failure)
    3  unrecognized tag (no known release prefix / not a version)
    4  manifest could not be read or had no version

Prefix dispatch is longest-prefix-first so `pyomo-pounce-v` and `python-v`
take precedence over the bare `v`.
"""

import os
import re
import sys

# (tag prefix, human label, manifest path relative to repo root). Ordered
# longest-prefix-first: a `pyomo-pounce-v…` / `python-v…` tag must not be
# misread as a bare `v…` crates tag.
SURFACES = [
    ("pyomo-pounce-v", "pyomo-pounce (PyPI)", "pyomo-pounce/pyproject.toml"),
    ("python-v", "pounce-solver (PyPI)", "python/pyproject.toml"),
    ("v", "crates.io workspace", "Cargo.toml"),
]

# A release version: X.Y.Z, optionally with a -prerelease / +build suffix.
_VERSION_RE = re.compile(r"^\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?$")

# First top-of-line `version = "..."`. In Cargo.toml that is the
# [workspace.package] version (other `version =` keys are indented inside
# dependency tables); in a pyproject.toml it is the [project] version. This
# mirrors the extraction in scripts/check-release-consistency.sh.
_MANIFEST_VERSION_RE = re.compile(r'^version\s*=\s*"([^"]+)"', re.MULTILINE)


def strip_ref(ref):
    """`refs/tags/v1.2.3` -> `v1.2.3`; a bare tag passes through unchanged."""
    prefix = "refs/tags/"
    return ref[len(prefix):] if ref.startswith(prefix) else ref


def parse_tag(tag):
    """Map a tag to (label, manifest_path, version), or None if unrecognized.

    Returns None when the tag carries no known release prefix or the part
    after the prefix is not a valid X.Y.Z version (so a stray `vfoo` or a
    non-release tag is rejected rather than silently matched).
    """
    for prefix, label, manifest in SURFACES:
        if tag.startswith(prefix):
            version = tag[len(prefix):]
            if _VERSION_RE.match(version):
                return label, manifest, version
            return None
    return None


def manifest_version(text):
    """Extract the first top-of-line `version = "..."`, or None."""
    m = _MANIFEST_VERSION_RE.search(text)
    return m.group(1) if m else None


def check(ref, repo_root="."):
    """Validate a tag/ref against its manifest.

    Returns (exit_code, message). exit_code 0 means the tag matches.
    """
    tag = strip_ref(ref)
    parsed = parse_tag(tag)
    if parsed is None:
        return 3, (
            f"check_tag_version: tag {tag!r} is not a recognized release tag "
            "(expected v<X.Y.Z>, python-v<X.Y.Z>, or pyomo-pounce-v<X.Y.Z>)."
        )
    label, manifest_rel, tag_version = parsed
    path = os.path.join(repo_root, manifest_rel)
    try:
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
    except OSError as e:
        return 4, f"check_tag_version: cannot read {manifest_rel}: {e}"
    mver = manifest_version(text)
    if mver is None:
        return 4, f"check_tag_version: no `version = \"...\"` found in {manifest_rel}"
    if mver != tag_version:
        return 2, (
            f"check_tag_version: TAG/MANIFEST MISMATCH for {label}.\n"
            f"  tag {tag!r} declares version {tag_version}\n"
            f"  {manifest_rel} is at {mver}\n"
            "  Tagging without bumping the manifest publishes the WRONG version "
            "(crates.io silently no-ops as 'already published'; PyPI ships the "
            "stale version). Bump the manifest to match the tag, or retag."
        )
    return 0, (
        f"check_tag_version: OK — tag {tag!r} matches {manifest_rel} "
        f"at {mver} ({label})."
    )


def main(argv):
    if len(argv) != 2:
        print("usage: check_tag_version.py <tag-or-ref>", file=sys.stderr)
        return 64
    repo_root = os.environ.get("GITHUB_WORKSPACE", ".")
    code, message = check(argv[1], repo_root=repo_root)
    stream = sys.stdout if code == 0 else sys.stderr
    print(message, file=stream)
    return code


if __name__ == "__main__":
    sys.exit(main(sys.argv))
