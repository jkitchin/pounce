// Guard for the docs assistant's retrieval half (docs/assets/ask.js).
//
// It loads the SHIPPED ask.js — not a copy of the scoring — through the test
// seam at the bottom of that file, and runs it against an index built from the
// live docs/src. A test of a reimplemented ranker would stay green while the
// ranker readers actually use regressed, which is the whole failure mode here:
// retrieval has no exception to throw. It returns the wrong thing, silently,
// and the only symptom is a reader not finding the page.
//
// Two things are checked, and they fail for different reasons:
//
//   1. The stemmer, against known Porter step-1 outputs and against the
//      specific pairs the corpus needs collapsed. This is a unit test: it
//      does not depend on what the docs say.
//   2. Ranking, against a labelled query set. This is a *corpus* test and it
//      is deliberately loose — see THRESHOLDS below.
//
// Usage:
//   python3 scripts/build-docs-index.py -o /tmp/ask-index.json --quiet
//   node docs/tests/ask_retrieval.mjs /tmp/ask-index.json
//
// `make ask-check` does both steps; ci.yml runs the same two lines.

import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";

const here = dirname(fileURLToPath(import.meta.url));
const require = createRequire(import.meta.url);
const ask = require(resolve(here, "../assets/ask.js"));

const indexPath = process.argv[2];
if (!indexPath) {
  console.error("usage: node docs/tests/ask_retrieval.mjs <ask-index.json>");
  process.exit(2);
}

let failures = 0;
function check(ok, msg) {
  if (!ok) failures++;
  console.log((ok ? "  ok   " : "  FAIL ") + msg);
}

// ---------------------------------------------------------------- stemmer --

console.log("stemmer: pairs the corpus needs collapsed");
for (const [a, b] of [
  ["relaxing", "relax"],
  ["relaxed", "relax"],
  ["started", "start"],
  ["starting", "start"],
  // The +e restoration case: without it "scaling" stems to `scal` and stops
  // matching the page called "Scaling".
  ["scaling", "scale"],
  ["iterations", "iteration"],
  ["solves", "solve"],
  ["tightening", "tighten"],
  ["bounded", "bound"],
  ["duals", "dual"]
]) {
  check(ask.normTok(a) === ask.normTok(b), a + " ≡ " + b + " → " + ask.normTok(a));
}

console.log("stemmer: canonical Porter step-1 outputs");
for (const [w, want] of [
  ["caresses", "caress"],
  ["ponies", "poni"],
  ["cats", "cat"],
  ["feed", "feed"],
  ["agreed", "agree"],
  ["plastered", "plaster"],
  ["motoring", "motor"],
  ["hopping", "hop"],
  ["failing", "fail"],
  ["filing", "file"],
  ["happy", "happi"],
  ["sky", "sky"],
  // Known and accepted: Porter step 1 does not restore the `e` here, because
  // the cvc rule only fires at m == 1 and `converg` has m == 2. Pinned so the
  // asymmetry is a decision on the record rather than a surprise.
  ["converged", "converg"],
  ["converge", "converge"]
]) {
  check(ask.normTok(w) === want, w + " → " + ask.normTok(w));
}

console.log("stemmer: identifiers and short tokens pass through untouched");
for (const w of ["bound_relax_factor", "theta_max", "mu", "tol", "l1", "qp", "ipopt", "nlp"]) {
  check(ask.normTok(w) === w, w + " unchanged");
}

console.log("distinct terms must not collide");
for (const [a, b] of [
  ["scaling", "scalar"],
  ["convex", "converge"],
  ["dual", "duel"],
  ["bound", "bind"]
]) {
  check(ask.normTok(a) !== ask.normTok(b), a + " ≠ " + b);
}

// ---------------------------------------------------------------- ranking --

const doc = JSON.parse(readFileSync(indexPath, "utf8"));
const idx = ask.buildIndex(doc.chunks || []);
console.log(
  "\nindex: " + idx.N + " passages, avg " + idx.avgdl.toFixed(0) + " tokens"
);
check(idx.N > 500, "index is populated");

// The wiki half is the reason this feature exists; an index that quietly
// stopped carrying it would still answer, just worse, and nothing else here
// would notice.
const wikiChunks = (doc.chunks || []).filter((c) => c.k === "wiki").length;
if (doc.counts && doc.counts.wiki > 0) {
  check(wikiChunks > 0, "wiki passages present (" + wikiChunks + ")");
} else {
  console.log("  note  index was built without the wiki; skipping wiki checks");
}

console.log("\nquery-side compound handling");
{
  // An identifier the corpus contains is searched as itself, never split —
  // splitting let a passage about `fix_relax` outrank the documentation of
  // `bound_relax_factor` by matching three of its four parts.
  const known = ask.queryTerms(idx, "bound_relax_factor");
  check(
    known.length === 1 && known[0] === "bound_relax_factor",
    "known identifier stays whole: [" + known.join(", ") + "]"
  );
  const unknown = ask.queryTerms(idx, "no_such_option_here");
  check(unknown.length > 1, "unknown identifier falls back to parts: [" + unknown.join(", ") + "]");
  const stopped = ask.queryTerms(idx, "what does it do");
  check(stopped.length === 0, "an all-stopword question yields no terms");
}

console.log("\nresult shaping");
{
  const hits = ask.search(idx, "scaling", 6);
  const urls = hits.map((h) => h.chunk.u);
  check(new Set(urls).size === urls.length, "no two hits share a URL");
  const pages = urls.map((u) => u.split("#")[0]);
  const worst = Math.max(...pages.map((p) => pages.filter((q) => q === p).length));
  check(worst <= 2, "at most two hits per page (worst: " + worst + ")");
  check(ask.search(idx, "", 6).length === 0, "empty query returns nothing");
  check(ask.search(idx, "zzzqqqxxnotaword", 6).length === 0, "unmatched query returns nothing");
}

// Labelled set. `want` substrings are matched against the citation URL, and a
// query lists every page that genuinely answers it — not one blessed page —
// because several of these questions have more than one honest home in the
// book.
//
// THRESHOLDS ARE DELIBERATELY BELOW THE MEASURED SCORE. This is a corpus
// test: adding or retitling a page can legitimately move a ranking, and a
// guard pinned to the exact current number would fail on honest doc edits and
// get raised until it meant nothing. It is set to catch the shape of a real
// regression — a scoring change that drops several queries at once — not
// single-rank drift.
//
// Measured at the time of writing, in both configurations this runs in:
//   with the wiki (production)  18/20 top-1, 20/20 top-5
//   book only (CI, no clone)    17/20 top-1, 20/20 top-5
// The floors are set under the lower of the two.
const THRESHOLD_TOP1 = 14;
const THRESHOLD_TOP5 = 18;

const EVAL = [
  { q: "why does relaxing the bounds cost iterations",
    want: ["A-hairs-width", "qp-bound-relax", "options.html", "convex-solver"] },
  { q: "bound_relax_factor", want: ["A-hairs-width", "options.html"] },
  { q: "how do I warm start a solve",
    want: ["initialization.html", "active-set-sqp", "warm-start-benchmark", "sessions.html"] },
  { q: "what does MaximumIterationsExceeded mean",
    want: ["troubleshooting", "solution-output", "solve-report-v1", "options.html"] },
  { q: "what does hessian_approximation limited-memory do",
    want: ["options.html", "casadi.html", "algorithm.html"] },
  { q: "how do I install pounce",
    want: ["installation.html", "docker.html", "quick-start", "python.html"] },
  { q: "using pounce from pyomo", want: ["pyomo.html", "quick-start"] },
  { q: "the JSON solve report schema", want: ["json-output", "solve-report-v1"] },
  { q: "which solver options are actually worth setting",
    want: ["Tuning-POUNCE-per-problem", "options.html", "troubleshooting"] },
  { q: "my solve fails because of where it started",
    want: ["Recovering-from-a-bad-start", "initialization.html"] },
  { q: "run pounce in the browser with webassembly", want: ["wasm.html"] },
  { q: "register pounce as a GAMS solver", want: ["gams.html"] },
  { q: "finding multiple minima with deflation",
    want: ["find-minima", "Why-multistart-misses-solutions"] },
  { q: "how does crossover work", want: ["crossover.html"] },
  { q: "sensitivity analysis of the solution to a parameter",
    want: ["sensitivity.html", "path-following", "continuation"] },
  { q: "what is feasibility based bound tightening", want: ["fbbt.html"] },
  { q: "solving a boundary value problem", want: ["bvp.html", "ode.html", "dae.html"] },
  { q: "why is my convex QP slow",
    want: ["convex-solver", "lp-qp-routing", "Tuning-POUNCE-per-problem", "benchmarks.html",
           "qp-bound-relax", "troubleshooting"] },
  { q: "scaling the problem variables", want: ["scaling.html", "options.html"] },
  { q: "debug a solve step by step", want: ["debugger.html", "troubleshooting", "options.html"] }
];

console.log("\nprompt construction (the RAG contract)");
{
  const hits = ask.search(idx, "bound_relax_factor", 5);
  const msgs = ask.buildPrompt("what is bound_relax_factor?", hits);
  check(msgs.length === 2 && msgs[0].role === "system" && msgs[1].role === "user",
    "system + user message");
  // The grounding instructions are the only thing keeping a 1B model from
  // answering about Ipopt from memory; losing them would not fail loudly.
  check(/ONLY from the numbered excerpts/.test(msgs[0].content), "system prompt grounds the model");
  check(/Cite every claim/.test(msgs[0].content), "system prompt demands citations");
  check(msgs[1].content.includes("what is bound_relax_factor?"), "question is in the user message");
  for (let n = 1; n <= hits.length; n++) {
    check(msgs[1].content.includes("[" + n + "] "), "excerpt [" + n + "] is numbered");
  }
  // Every passage must be reachable from the source list the reader sees, or
  // a citation points at nothing.
  check(hits.length <= 6, "no more passages than the panel lists");

  // Truncation: a long passage must be cut, not sent whole, or five of them
  // overflow a small model's context.
  const long = [{ chunk: { h: "H", t: "T", u: "u.html", k: "book",
                           x: "x".repeat(ask.MAX_CTX_CHARS * 3) } }];
  const cut = ask.buildPrompt("q", long)[1].content;
  check(cut.length < ask.MAX_CTX_CHARS * 2, "over-long passage is truncated (" + cut.length + " chars)");
  check(cut.includes("\u2026"), "truncation is marked with an ellipsis");
}

console.log("\nranking over " + EVAL.length + " labelled queries");
let top1 = 0;
let top5 = 0;
for (const c of EVAL) {
  const urls = ask.search(idx, c.q, 6).map((h) => h.chunk.u);
  const hit = (u) => c.want.some((w) => u.includes(w));
  const at1 = urls.length > 0 && hit(urls[0]);
  const at5 = urls.slice(0, 5).some(hit);
  if (at1) top1++;
  if (at5) top5++;
  console.log("  " + (at1 ? "T1" : at5 ? ".5" : "XX") + "  " + c.q);
  if (!at1) console.log("        got: " + (urls[0] || "(nothing)"));
}

console.log("\n  top-1 " + top1 + "/" + EVAL.length + " (floor " + THRESHOLD_TOP1 + ")");
console.log("  top-5 " + top5 + "/" + EVAL.length + " (floor " + THRESHOLD_TOP5 + ")");
check(top1 >= THRESHOLD_TOP1, "top-1 above the floor");
check(top5 >= THRESHOLD_TOP5, "top-5 above the floor");

console.log("\n" + (failures ? failures + " FAILURE(S)" : "ask_retrieval: OK"));
process.exit(failures ? 1 : 0);
