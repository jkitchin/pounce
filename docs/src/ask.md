# Ask POUNCE: the in-browser docs assistant

The **Ask** button in the menu bar opens a question box for these docs. It has
two halves, and they work independently:

| | What it does | What it costs |
|---|---|---|
| **Search** | Ranks passages from this book *and* the [project wiki](https://github.com/jkitchin/pounce/wiki), deep-linked to the exact heading | one ~1.3 MB index download, on first use |
| **Answer** | Writes those passages up into prose, with citations | a language model download, hundreds of MB, only if you ask for it |

Search always works. The answer half is strictly opt-in: nothing is downloaded
until you pick a model and click **Load model**. If you never do, you still get
ranked, linked passages, which is most of the value on a reference manual.

Nothing you type leaves your browser. There is no API key, no server, and no
telemetry — the assistant is a static JavaScript file, and the only third-party
request it ever makes is fetching model weights from the WebLLM CDN after you
click.

## What it searches

The index covers every page of this book **and** the five long-form wiki
pages, which are not part of the book:

- [Tuning POUNCE per problem](https://github.com/jkitchin/pounce/wiki/Tuning-POUNCE-per-problem)
- [Recovering from a bad start](https://github.com/jkitchin/pounce/wiki/Recovering-from-a-bad-start)
- [A hair's width times a cliff](https://github.com/jkitchin/pounce/wiki/A-hairs-width-times-a-cliff)
- [Why multistart misses solutions](https://github.com/jkitchin/pounce/wiki/Why-multistart-misses-solutions)

That inclusion is the point of the feature as much as the model is. The wiki
carries the measured guidance — which of the 441 options are worth setting,
what to do when a solve dies at its starting point, why bound relaxation
costs what it costs — and a reader asking "why is my solve slow" wants those
pages, not a reference entry. Wiki hits are marked `wiki` in the source list
and link out to GitHub.

Passages are cut at heading boundaries, so every citation lands on the section
that answered rather than at the top of a long page.

## Requirements for written answers

| | |
|---|---|
| **Browser** | WebGPU: Chrome/Edge 113+, Safari 26+, Firefox with WebGPU enabled |
| **Download** | 0.9–2.3 GB depending on the model, cached by the browser afterwards |
| **Memory** | roughly the model's size in GPU memory while loaded |

Without WebGPU the panel says so and stays in search-only mode.

Three models are offered, smallest first — Llama 3.2 1B, Qwen 2.5 1.5B, and
Llama 3.2 3B. The smallest is the default because a 0.9 GB download that
answers beats a 2.3 GB one you abandon; the larger ones follow instructions
and citation formatting more reliably. If your GPU reports `shader-f16` you
get the half-precision builds, which are about a third smaller.

## How much to trust it

The model only ever sees the five passages your question retrieved, and it is
told to answer from them alone and to cite each claim. That keeps it far
closer to the docs than an unaided chatbot — but a 1B model reading correct
passages can still summarize them wrongly.

**Treat the answer as a routing device and the passages as the source.** Every
answer is shown with the passages it was built from, in rank order, linked to
the heading they came from. When the two disagree, the passage is right.

The retrieval half is worth stating plainly too: it is
[BM25](https://en.wikipedia.org/wiki/Okapi_BM25) over words, not embeddings.
It is excellent at exact names — an option like `bound_relax_factor`, or a
solver status string copied straight out of your log — and weaker at questions
phrased with no term in common with the docs. If a question comes back empty,
search for the option name or the exact status string.

(Only one identifier is named in that sentence, on purpose. This page is in the
index like any other, and an earlier draft that listed several real option and
status names in one short paragraph became the top hit for each of them — ahead
of the reference pages that actually define them. A short passage dense in rare
terms is exactly what BM25 rewards.)

## For maintainers

| Piece | File |
|---|---|
| Index builder | `scripts/build-docs-index.py` |
| Panel, retrieval, WebLLM glue | `docs/assets/ask.js`, `docs/assets/ask.css` |
| Staging and injection | `scripts/build-versioned-docs.sh` |
| Wiki clone | `.github/workflows/docs.yml` |

Two details that are easy to get wrong:

- **The assistant is injected, not configured in `book.toml`.** It reaches
  every built book — including archived tag builds whose source predates it —
  the same way the version selector does. Listing it under `additional-js`
  instead would reach `dev/` and nothing else until the next release, because
  the site root is built from the newest *tag*. A plain `mdbook build docs`
  (`make book`) therefore has no **Ask** button; use
  `scripts/build-versioned-docs.sh` to see it.
- **There is one index, built from `main`, shared by every version.** So the
  assistant inside an archived book answers about *current* docs and links to
  the stable book. That is the deliberate trade: one 1.3 MB file rather than
  one per archived tag, and answers that describe the software as it is now.

To build the index locally, including the wiki:

```console
$ git clone --depth 1 https://github.com/jkitchin/pounce.wiki.git /tmp/pounce-wiki
$ POUNCE_WIKI_DIR=/tmp/pounce-wiki scripts/build-versioned-docs.sh ./site
$ mkdir -p _serve/pounce && cp -a site/. _serve/pounce/
$ python3 -m http.server -d _serve 8000    # http://localhost:8000/pounce/
```

Serving under the real base path matters: the assistant resolves its own
paths from the injected `data-p2r` / `data-index` attributes, and a bad
relative path is only visible at a nesting depth greater than one.

The wiki clone is optional. Without `POUNCE_WIKI_DIR` the index is book-only
and the build still succeeds — in CI the clone step is `continue-on-error`
for the same reason, so an unreachable wiki degrades the assistant instead of
taking the docs site down.
