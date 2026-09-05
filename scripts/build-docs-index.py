#!/usr/bin/env python3
"""Build the retrieval index the in-browser docs assistant searches.

The assistant (docs/assets/ask.js) is retrieval-augmented: it looks up
passages in this index and hands them to a WebLLM model running in the
reader's own browser. The model never sees the corpus, only the passages a
question retrieves, so the index *is* the assistant's knowledge — a page
missing here is a page the assistant cannot answer about.

Two corpora, deliberately handled differently:

  * ``docs/src/**/*.md`` — the rendered book. Citations are relative links
    into the book itself (``options.html#barrier-parameter``).
  * the GitHub wiki (``--wiki <dir>``, a clone of ``pounce.wiki``) — indexed
    but **not** rendered into the book. Citations are absolute links back to
    github.com/jkitchin/pounce/wiki. The wiki holds the longer-form measured
    guidance (tuning, bad starts, bound relaxation, multistart), which is
    exactly the material a reader asks a question about and exactly what a
    reference manual does not carry, so leaving it out would make the
    assistant blind to the best answers the project has.

Chunking is by heading section, because a heading is the finest granularity
that mdBook gives a stable anchor — so every retrieved passage can be
deep-linked rather than dumping the reader at the top of a 900-line page.
Sections longer than MAX_CHARS are split at paragraph boundaries and each
piece re-carries the heading trail, so a passage is self-describing even
when it is the fourth slice of one section.

Usage:
    scripts/build-docs-index.py [-o OUT] [--wiki DIR] [--book-src DIR]

Output is JSON on the schema documented in ``docs/src/ask.md``. Keys are
short (``t``/``h``/``u``/``k``/``x``) because this file is downloaded by
every reader who opens the assistant.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

# Passage sizing. MAX_CHARS is a retrieval choice, not a model-context one:
# short enough that a hit is specific, long enough that a worked example or a
# full option description survives in one piece. MIN_CHARS drops the "### Foo"
# heading-only stubs that would otherwise win on a title match and carry no
# answer.
MAX_CHARS = 1600
MIN_CHARS = 40

# A long fenced block is bad context — it crowds out prose in the model's
# window and rarely contains the sentence that answers the question. Keep
# enough to show the shape of the call.
MAX_CODE_LINES = 40

WIKI_BASE = "https://github.com/jkitchin/pounce/wiki/"

# mdBook directives ({{#include ...}}, {{#playground ...}}) expand at render
# time; the raw form is noise in an index.
RE_MDBOOK_DIRECTIVE = re.compile(r"\{\{#[^}]*\}\}")
RE_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
RE_HTML_TAG = re.compile(r"<[^>\n]{1,200}>")
RE_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
RE_FENCE = re.compile(r"^\s*(```+|~~~+)")


def strip_inline_markup(text: str) -> str:
    """Reduce heading markup to the text mdBook renders (and slugifies)."""
    text = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", text)  # images
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)  # links
    text = re.sub(r"\[([^\]]*)\]\[[^\]]*\]", r"\1", text)  # ref links
    text = text.replace("`", "")
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"\*([^*]+)\*", r"\1", text)
    text = re.sub(r"__([^_]+)__", r"\1", text)
    return text.strip()


def slugify(heading: str) -> str:
    """mdBook's heading-anchor algorithm.

    mdBook lowercases the rendered heading text, keeps only alphanumerics,
    spaces, `-` and `_`, then turns spaces into `-`. Reimplemented rather
    than guessed at: an anchor that does not match is a citation link that
    silently lands at the top of the page, which looks like it worked.
    """
    slug = strip_inline_markup(heading).lower()
    slug = "".join(c for c in slug if c.isalnum() or c in " -_")
    slug = slug.replace(" ", "-")
    return slug


def uniquify(slug: str, seen: dict[str, int]) -> str:
    """Match mdBook's duplicate-anchor suffixing (`foo`, `foo-1`, `foo-2`)."""
    if slug not in seen:
        seen[slug] = 0
        return slug
    seen[slug] += 1
    return "%s-%d" % (slug, seen[slug])


def clean_body(lines: list[str]) -> str:
    """Normalize a section body for indexing.

    Fenced code is kept (an option name or a CLI flag is often *only* in a
    code block) but capped at MAX_CODE_LINES.
    """
    out: list[str] = []
    fence: str | None = None
    code_lines = 0
    for line in lines:
        m = RE_FENCE.match(line)
        if m:
            marker = m.group(1)
            if fence is None:
                fence = marker
                code_lines = 0
                out.append(line)
            elif line.strip().startswith(fence):
                fence = None
                out.append(line)
            else:
                out.append(line)
            continue
        if fence is not None:
            code_lines += 1
            if code_lines <= MAX_CODE_LINES:
                out.append(line)
            elif code_lines == MAX_CODE_LINES + 1:
                out.append("    … (%d more lines)" % (len(lines) - MAX_CODE_LINES))
            continue
        out.append(line)

    text = "\n".join(out)
    text = RE_HTML_COMMENT.sub(" ", text)
    text = RE_MDBOOK_DIRECTIVE.sub(" ", text)
    text = RE_HTML_TAG.sub(" ", text)
    # Collapse runs of blank lines but keep paragraph breaks: the splitter
    # below uses them as the only safe place to cut a long section.
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = "\n".join(ln.rstrip() for ln in text.split("\n"))
    return text.strip()


def _pack(units: list[str], sep: str, max_chars: int) -> list[str]:
    """Greedily pack units into runs no longer than max_chars."""
    parts: list[str] = []
    buf = ""
    for unit in units:
        if not buf:
            buf = unit
        elif len(buf) + len(sep) + len(unit) <= max_chars:
            buf += sep + unit
        else:
            parts.append(buf)
            buf = unit
    if buf:
        parts.append(buf)
    return parts


def split_long(text: str, max_chars: int = MAX_CHARS) -> list[str]:
    """Split an over-long section into passages.

    Paragraph boundaries first. A markdown *table* is one paragraph — its
    rows are single-newline separated — and the option tables in this book
    run to eleven thousand characters, so a paragraph pass alone leaves
    chunks that would swamp a small model's whole context window with one
    hit. Over-long paragraphs therefore get a second, line-level pass.

    A single line longer than max_chars is still emitted whole: cutting
    mid-sentence costs more than the overrun.
    """
    if len(text) <= max_chars:
        return [text]
    parts: list[str] = []
    for para in _pack(text.split("\n\n"), "\n\n", max_chars):
        if len(para) <= max_chars:
            parts.append(para)
        else:
            parts.extend(_pack(para.split("\n"), "\n", max_chars))
    return parts


def parse_sections(md: str) -> list[tuple[str, str, str]]:
    """Split markdown into (heading_trail, anchor, body) triples.

    The trail is the enclosing heading path ("Solver Options › Scaling"), so
    a retrieved passage names its own context without the reader opening the
    page. Content before the first heading is attributed to the page itself
    with an empty anchor.
    """
    lines = md.split("\n")
    sections: list[tuple[str, str, str]] = []
    seen_slugs: dict[str, int] = {}
    stack: list[tuple[int, str]] = []  # (level, text)
    cur_trail = ""
    cur_anchor = ""
    buf: list[str] = []
    fence: str | None = None

    def flush() -> None:
        body = clean_body(buf)
        if body:
            sections.append((cur_trail, cur_anchor, body))

    for line in lines:
        m = RE_FENCE.match(line)
        if m:
            marker = m.group(1)
            if fence is None:
                fence = marker
            elif line.strip().startswith(fence):
                fence = None
        # A `#` inside a fence is a shell comment, not a heading.
        h = None if fence is not None else RE_HEADING.match(line)
        if h:
            flush()
            buf = []
            level = len(h.group(1))
            text = strip_inline_markup(h.group(2))
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, text))
            cur_trail = " › ".join(t for _lvl, t in stack)
            cur_anchor = uniquify(slugify(h.group(2)), seen_slugs)
        else:
            buf.append(line)
    flush()
    return sections


def chunks_for(md: str, title: str, url_base: str, kind: str) -> list[dict]:
    """Turn one document into indexable chunks."""
    out: list[dict] = []
    for trail, anchor, body in parse_sections(md):
        heading = trail or title
        url = url_base + ("#" + anchor if anchor else "")
        for piece in split_long(body):
            if len(piece) < MIN_CHARS:
                continue
            out.append(
                {
                    "t": title,
                    "h": heading,
                    "u": url,
                    "k": kind,
                    "x": piece,
                }
            )
    return out


def page_title(md: str, fallback: str) -> str:
    for line in md.split("\n"):
        m = RE_HEADING.match(line)
        if m and len(m.group(1)) == 1:
            return strip_inline_markup(m.group(2))
    return fallback


def collect_book(src_dir: str) -> list[dict]:
    chunks: list[dict] = []
    paths = sorted(glob.glob(os.path.join(src_dir, "**", "*.md"), recursive=True))
    for path in paths:
        rel = os.path.relpath(path, src_dir)
        if rel == "SUMMARY.md":
            continue
        with open(path, encoding="utf-8") as f:
            md = f.read()
        title = page_title(md, os.path.splitext(os.path.basename(rel))[0])
        url_base = rel[: -len(".md")] + ".html"
        chunks.extend(chunks_for(md, title, url_base.replace(os.sep, "/"), "book"))
    return chunks


def collect_wiki(wiki_dir: str) -> list[dict]:
    """Index a `pounce.wiki` clone.

    Wiki filenames are the page names with spaces as hyphens, which is also
    the URL form — so the citation URL is the basename, no slugification.

    `_Sidebar` / `_Footer` are navigation chrome. `Home` is too: it is a
    one-paragraph blurb per wiki page, so it matches a term from every topic
    at once and, being short, wins on BM25 length normalization — measured, it
    was the top hit for "my solve fails because of where it started" ahead of
    the page named "Recovering from a bad start". Everything it links to is
    indexed on its own, so dropping it costs no content.
    """
    chunks: list[dict] = []
    paths = sorted(glob.glob(os.path.join(wiki_dir, "*.md")))
    for path in paths:
        name = os.path.splitext(os.path.basename(path))[0]
        if name.startswith("_") or name == "Home":
            continue
        with open(path, encoding="utf-8") as f:
            md = f.read()
        title = page_title(md, name.replace("-", " "))
        chunks.extend(chunks_for(md, title, WIKI_BASE + name, "wiki"))
    return chunks


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-o", "--out", default="ask-index.json", help="output JSON path")
    ap.add_argument("--book-src", default="docs/src", help="mdBook src directory")
    ap.add_argument(
        "--wiki",
        default=None,
        help="path to a pounce.wiki clone; omitted means book-only (a local "
        "build without network access still produces a usable index)",
    )
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    if not os.path.isdir(args.book_src):
        print("build-docs-index: no such book src: %s" % args.book_src, file=sys.stderr)
        return 1

    chunks = collect_book(args.book_src)
    n_book = len(chunks)

    n_wiki = 0
    if args.wiki:
        if os.path.isdir(args.wiki):
            wiki_chunks = collect_wiki(args.wiki)
            n_wiki = len(wiki_chunks)
            chunks.extend(wiki_chunks)
        else:
            # Not fatal: the wiki is a separate repository, and a docs build
            # must not fail because that clone did not happen. The assistant
            # degrades to book-only, and the count below says so out loud.
            print(
                "build-docs-index: WARNING wiki dir not found, indexing book only: %s"
                % args.wiki,
                file=sys.stderr,
            )

    for i, c in enumerate(chunks):
        c["i"] = i

    doc = {
        "schema": 1,
        "counts": {"book": n_book, "wiki": n_wiki, "total": len(chunks)},
        "wiki_base": WIKI_BASE,
        "chunks": chunks,
    }

    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, separators=(",", ":"))
        f.write("\n")

    if not args.quiet:
        size = os.path.getsize(args.out)
        print(
            "build-docs-index: %d chunks (%d book, %d wiki) -> %s (%.0f KB)"
            % (len(chunks), n_book, n_wiki, args.out, size / 1024.0)
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
