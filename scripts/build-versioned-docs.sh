#!/usr/bin/env bash
# Build the POUNCE docs as a multi-version site for GitHub Pages.
#
# Layout produced (single artifact, served under the Pages base path):
#   <out>/               STABLE  — the latest v* release tag (default landing)
#   <out>/dev/           main    — the current working tree
#   <out>/vX.Y.Z/        one archived build per v* release tag
#   <out>/versions.json  manifest the selector reads (copied into every book)
#   <out>/versions.js    the selector (copied into every book)
#   <out>/ask.js         the docs assistant (copied into every book)
#   <out>/ask.css        its styling (copied into every book)
#   <out>/ask-index.json its retrieval corpus (ONE copy, at the site root)
#
# The version selector (docs/assets/versions.js) is injected into EVERY built
# book — including archived tag builds whose source predates the selector — by
# copying the current versions.js in and adding a <script> tag to every page.
# The assistant (docs/assets/ask.js) rides exactly the same mechanism, and for
# the same reason: shipped only via book.toml's `additional-js` it would reach
# `dev/` and nothing else until the next release, because the site root is
# built from the newest *tag*, whose source predates this change.
#
# The retrieval index is the one shared file. It is built from the working
# tree's docs/src plus (optionally) a wiki clone, so it describes CURRENT
# docs no matter which version's book is displaying it — which is why each
# page gets an explicit `data-index` path back to the single root copy rather
# than a per-book copy. One 1.3 MB file, not one per archived tag.
# docs/src/ask.md tells readers that the assistant answers about current docs.
#
# Set POUNCE_WIKI_DIR to a `pounce.wiki` clone to index the wiki too; without
# it the index is book-only and the build still succeeds (see docs.yml, which
# clones the wiki, and scripts/build-docs-index.py).
#
# Requirements: git, mdbook, python3. Local runs need the tags present:
#   git fetch --tags && scripts/build-versioned-docs.sh ./site
# Then serve under the real base path to exercise the selector, e.g.:
#   mkdir -p _serve/pounce && cp -a site/. _serve/pounce/
#   python3 -m http.server -d _serve 8000   # http://localhost:8000/pounce/
#
# Usage: scripts/build-versioned-docs.sh [output-dir]   (default: ./site)

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

OUT="${1:-$PWD/site}"
# Make OUT absolute (mdbook --dest-dir is relative to the book, not cwd).
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"
rm -rf "${OUT:?}"/*

VERSIONS_JS="$PWD/docs/assets/versions.js"
[[ -f "$VERSIONS_JS" ]] || { echo "missing $VERSIONS_JS" >&2; exit 1; }

ASK_JS="$PWD/docs/assets/ask.js"
ASK_CSS="$PWD/docs/assets/ask.css"
for f in "$ASK_JS" "$ASK_CSS"; do
  [[ -f "$f" ]] || { echo "missing $f" >&2; exit 1; }
done

# Release tags only: vX.Y.Z (excludes python-v* / pyomo-pounce-v*). sort -V so
# the highest is last -> STABLE. A new release tag appears here automatically.
TAGS="$(git tag -l 'v*' | grep -E '^v[0-9]+\.[0-9]+\.[0-9]+$' | sort -V || true)"
[[ -n "$TAGS" ]] || { echo "no v* release tags found" >&2; exit 1; }
STABLE="$(printf '%s\n' "$TAGS" | tail -1)"

echo "build-versioned-docs: out=$OUT stable=$STABLE"
# shellcheck disable=SC2086  # intentional split: list tags space-separated
echo "  tags: $(printf '%s ' $TAGS)"

# --- dev (current working tree) -------------------------------------------
echo "[dev] building from working tree"
mdbook build docs --dest-dir "$OUT/dev"

# --- archived tag builds via worktrees ------------------------------------
for tag in $TAGS; do
  wt="$(mktemp -d)"
  git worktree add --detach "$wt" "$tag" >/dev/null 2>&1
  if [[ -f "$wt/docs/book.toml" ]]; then
    echo "[$tag] building"
    mdbook build "$wt/docs" --dest-dir "$OUT/$tag"
  else
    echo "[$tag] no docs/book.toml — skipping"
  fi
  git worktree remove --force "$wt"
done

# --- stable -> site root (byte-identical to its archived copy) -------------
if [[ ! -d "$OUT/$STABLE" ]]; then
  echo "stable build $OUT/$STABLE missing" >&2; exit 1
fi
echo "[root] copying stable ($STABLE) to site root"
cp -a "$OUT/$STABLE/." "$OUT/"

# --- assistant retrieval index (one copy, shared by every version) ---------
# Built from the working tree, i.e. main: the assistant answers about current
# docs even when displayed inside an archived book. Non-fatal if the wiki
# clone is absent — the index is then book-only rather than the build failing
# on a second repository being unreachable.
echo "[ask] building retrieval index (wiki: ${POUNCE_WIKI_DIR:-none})"
ASK_INDEX_ARGS=(-o "$OUT/ask-index.json" --book-src "$PWD/docs/src")
if [[ -n "${POUNCE_WIKI_DIR:-}" ]]; then
  ASK_INDEX_ARGS+=(--wiki "$POUNCE_WIKI_DIR")
fi
python3 scripts/build-docs-index.py "${ASK_INDEX_ARGS[@]}"

# --- selector + assistant: assets, manifest, per-page <script> injection ---
echo "[selector] writing versions.json, copying assets, injecting tags"
OUT="$OUT" STABLE="$STABLE" TAGS="$TAGS" VERSIONS_JS="$VERSIONS_JS" \
ASK_JS="$ASK_JS" ASK_CSS="$ASK_CSS" python3 - <<'PY'
import os, re, json, shutil

out = os.environ["OUT"]
stable = os.environ["STABLE"]
tags = os.environ["TAGS"].split()
versions_js = os.environ["VERSIONS_JS"]
ask_js = os.environ["ASK_JS"]
ask_css = os.environ["ASK_CSS"]
ask_index = os.path.join(out, "ask-index.json")

# Manifest: dev first, then tags newest -> oldest.
entries = [{"id": "dev", "label": "dev (unreleased)", "path": "dev", "kind": "dev"}]
for t in sorted(tags, key=lambda s: [int(x) for x in s[1:].split(".")], reverse=True):
    if t == stable:
        entries.append({"id": t, "label": t + " (stable)", "path": "", "kind": "stable"})
    else:
        entries.append({"id": t, "label": t, "path": t, "kind": "archived"})
manifest = json.dumps({"schema": 1, "stable": stable, "versions": entries}, indent=2) + "\n"

# Book roots that need versions.json + versions.js + injected <script> tags:
#   the site root (stable), dev/, and each tag dir.
book_roots = [out, os.path.join(out, "dev")]
book_roots += [os.path.join(out, t) for t in tags if os.path.isdir(os.path.join(out, t))]

P2R = re.compile(r'const\s+path_to_root\s*=\s*"([^"]*)"')

def inject(html_path):
    with open(html_path, "r", encoding="utf-8") as f:
        html = f.read()
    if "</head>" not in html:
        return
    m = P2R.search(html)
    p2r = m.group(1) if m else ""
    tags_html = ""
    if 'id="pounce-versions"' not in html:
        tags_html += ('<script defer src="%sversions.js" id="pounce-versions" '
                      'data-p2r="%s"></script>\n') % (p2r, p2r)
    if 'id="pounce-ask"' not in html:
        # p2r only reaches this *version's* root, and two of the assistant's
        # paths are site-root-relative, not version-relative:
        #
        #   data-index  one index file, shared by every version
        #   data-root   where citations point — the stable book at the site
        #               root, because the index describes current docs. An
        #               archived book resolving them against its own root
        #               yields 404s for pages and anchors added since.
        #
        # Both are computed from this page's real directory rather than
        # derived from p2r, which cannot express "up out of this version".
        page_dir = os.path.dirname(html_path)
        rel_index = os.path.relpath(ask_index, page_dir).replace(os.sep, "/")
        rel_root = os.path.relpath(out, page_dir).replace(os.sep, "/")
        if rel_root == ".":
            rel_root = ""
        else:
            rel_root += "/"
        tags_html += ('<script defer src="%sask.js" id="pounce-ask" '
                      'data-p2r="%s" data-index="%s" data-root="%s"></script>\n'
                      ) % (p2r, p2r, rel_index, rel_root)
    if not tags_html:
        return
    html = html.replace("</head>", tags_html + "</head>", 1)
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)

for root in book_roots:
    with open(os.path.join(root, "versions.json"), "w", encoding="utf-8") as f:
        f.write(manifest)
    shutil.copyfile(versions_js, os.path.join(root, "versions.js"))
    shutil.copyfile(ask_js, os.path.join(root, "ask.js"))
    shutil.copyfile(ask_css, os.path.join(root, "ask.css"))
    for dirpath, _dirs, files in os.walk(root):
        for name in files:
            if name.endswith(".html"):
                inject(os.path.join(dirpath, name))

print("  versions: " + ", ".join(e["id"] for e in entries))
print("  injected into %d book root(s)" % len(book_roots))
PY

echo "build-versioned-docs: done -> $OUT"
