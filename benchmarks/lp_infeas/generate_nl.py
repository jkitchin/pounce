#!/usr/bin/env python3
"""Build the lp_infeas suite: LPs with no optimal solution, with known labels.

Two public sources, every instance labelled with the verdict a correct
solver must reach:

1. **NETLIB lp/infeas** (Chinneck, 1993) — <https://www.netlib.org/lp/infeas/>.
   29 primal-infeasible LPs in netlib's compressed MPS format, expanded with
   the netlib ``emps`` program (fetched and compiled here, as in
   ``benchmarks/lp``). Small to medium; the whole set runs in minutes.

2. **MIPLIB 2017 Infeas** (Lu & Tang, 2026) —
   <https://mit-lu-lab.github.io/MIPLIB2017-Infeas/>. 334 LPs built from
   MIPLIB 2017 with integrality dropped: 229 primal-infeasible (``*_if``,
   from branch-and-bound nodes) and 105 dual-infeasible / unbounded
   (``*_ub``, bounds and rows dropped until the objective is unbounded).
   Every instance has at least ~100k nonzeros and the largest has tens of
   millions, so a size cap applies (``--max-nnz``, default 300 000, which
   keeps 93 + 54 = 147). Files are checked against the published SHA-256.

Each MPS is converted to ``.nl`` by ``benchmarks/lp/mps_to_nl.py`` (HiGHS
parse -> Pyomo -> ASL writer). ``labels.json`` is written beside the
``.nl``: name -> {source, expected, n, m, nnz[, highs]}. ``expected`` is
``primal_infeasible`` or ``dual_infeasible``; ``score.py`` reads it.

``--check-labels`` also solves each MPS with HiGHS (1 thread, 60 s) and
records its verdict under ``highs`` — an independent check on the labels
(NETLIB's ``cplex2`` is documented as "almost feasible") and the reference
column in the score table.

Requires ``highspy`` and ``pyomo`` (``pip install highspy pyomo``), ``curl``
and a C compiler for ``emps``.

Usage:
    python3 generate_nl.py                       # both sources, default cap
    python3 generate_nl.py --netlib-only
    python3 generate_nl.py --miplib-only --max-nnz 1000000
    python3 generate_nl.py --check-labels        # also record HiGHS verdicts
"""
import argparse
import csv
import gzip
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
NL_DIR = os.path.join(HERE, "nl")
MPS_DIR = os.path.join(HERE, "mps")  # downloads + emps binary (gitignored)

NETLIB_BASE = "https://www.netlib.org/lp/infeas"
EMPS_URL = "https://www.netlib.org/lp/data/emps.c"
NETLIB_INFEAS = [
    "bgdbg1", "bgetam", "bgindy", "bgprtr", "box1", "ceria3d", "chemcom",
    "cplex1", "cplex2", "ex72a", "ex73a", "forest6", "galenet", "gosh",
    "gran", "greenbea", "itest2", "itest6", "klein1", "klein2", "klein3",
    "mondou2", "pang", "pilot4i", "qual", "reactor", "refinery", "vol1",
    "woodinfe",
]

MIPLIB_SITE = "https://mit-lu-lab.github.io/MIPLIB2017-Infeas"
MIPLIB_RELEASE = "https://github.com/MIT-Lu-Lab/MIPLIB2017-Infeas/releases/download/v1.0"
MIPLIB_LISTS = [
    ("primal-infeasible.csv", "primal_infeasible"),
    ("dual-infeasible.csv", "dual_infeasible"),
]
DEFAULT_MAX_NNZ = 300_000


def _load_mps_to_nl():
    # Load by path: this file is also called generate_nl.py, so a plain
    # import of benchmarks/lp's modules by name would be ambiguous.
    path = os.path.join(HERE, "..", "lp", "mps_to_nl.py")
    spec = importlib.util.spec_from_file_location("lp_mps_to_nl", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fetch(url, dest):
    subprocess.run(["curl", "-fsSL", "--retry", "3", "--max-time", "600",
                    url, "-o", dest], check=True)


def ensure_emps():
    emps_bin = os.path.join(MPS_DIR, "emps")
    if os.path.exists(emps_bin) and os.access(emps_bin, os.X_OK):
        return emps_bin
    src = os.path.join(MPS_DIR, "emps.c")
    if not os.path.exists(src):
        fetch(EMPS_URL, src)
    subprocess.run([os.environ.get("CC", "cc"), "-w", "-o", emps_bin, src], check=True)
    return emps_bin


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dims(mps_path):
    import highspy
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.readModel(mps_path)
    lp = h.getModel().lp_
    return lp.num_col_, lp.num_row_, len(list(lp.a_matrix_.value_))


def highs_verdict(mps_path, time_limit=60.0):
    import highspy
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("time_limit", time_limit)
    h.setOptionValue("threads", 1)
    h.readModel(mps_path)
    h.run()
    return h.modelStatusToString(h.getModelStatus())


def netlib_instances():
    """Yield (name, mps_path, expected, extra) for the NETLIB set."""
    emps = ensure_emps()
    for prob in NETLIB_INFEAS:
        raw = os.path.join(MPS_DIR, f"netlib-{prob}.emps")
        mps = os.path.join(MPS_DIR, f"netlib-{prob}.mps")
        if not os.path.exists(mps):
            if not os.path.exists(raw):
                fetch(f"{NETLIB_BASE}/{prob}", raw)
            with open(mps, "wb") as out:
                subprocess.run([emps, raw], check=True, stdout=out)
        yield f"netlib-{prob}", mps, "primal_infeasible", {"source": "netlib-infeas"}


def miplib_instances(max_nnz):
    """Yield (name, mps_path, expected, extra) for the capped MIPLIB set."""
    for listing, expected in MIPLIB_LISTS:
        csv_path = os.path.join(MPS_DIR, listing)
        if not os.path.exists(csv_path):
            fetch(f"{MIPLIB_SITE}/{listing}", csv_path)
        with open(csv_path, newline="") as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            if max_nnz and int(r["nnz"]) > max_nnz:
                continue
            gz_name = r["mps_filename"]
            stem = gz_name[: -len(".mps.gz")]
            gz = os.path.join(MPS_DIR, gz_name)
            mps = os.path.join(MPS_DIR, stem + ".mps")
            if not os.path.exists(mps):
                if not os.path.exists(gz) or sha256(gz) != r["sha256"]:
                    fetch(f"{MIPLIB_RELEASE}/{gz_name}", gz)
                got = sha256(gz)
                if got != r["sha256"]:
                    raise RuntimeError(f"{gz_name}: sha256 {got} != published {r['sha256']}")
                with gzip.open(gz, "rb") as src, open(mps, "wb") as dst:
                    shutil.copyfileobj(src, dst)
            yield stem, mps, expected, {"source": "miplib2017-infeas"}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out-dir", default=NL_DIR)
    ap.add_argument("--max-nnz", type=int, default=DEFAULT_MAX_NNZ,
                    help="MIPLIB size cap on nonzeros (0 = no cap)")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--netlib-only", action="store_true")
    g.add_argument("--miplib-only", action="store_true")
    ap.add_argument("--check-labels", action="store_true",
                    help="also solve each MPS with HiGHS and record its verdict")
    args = ap.parse_args()

    os.makedirs(MPS_DIR, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    mps_to_nl = _load_mps_to_nl()

    labels_path = os.path.join(args.out_dir, "labels.json")
    labels = {}
    if os.path.exists(labels_path):
        with open(labels_path) as f:
            labels = json.load(f)

    sources = []
    if not args.miplib_only:
        sources.append(netlib_instances())
    if not args.netlib_only:
        sources.append(miplib_instances(args.max_nnz))

    for gen in sources:
        for name, mps, expected, extra in gen:
            nl = os.path.join(args.out_dir, name + ".nl")
            entry = dict(labels.get(name, {}))
            entry.update(extra)
            entry["expected"] = expected
            if "nnz" not in entry:
                entry["n"], entry["m"], entry["nnz"] = dims(mps)
            if not os.path.exists(nl):
                mps_to_nl.convert(mps, nl)
            if args.check_labels and "highs" not in entry:
                entry["highs"] = highs_verdict(mps)
            labels[name] = entry
            print(f"{name:40s} {expected:18s} n={entry['n']} m={entry['m']} "
                  f"nnz={entry['nnz']}" + (f" highs={entry['highs']}" if "highs" in entry else ""),
                  flush=True)
            # Write as we go so an interrupted run keeps what it converted.
            with open(labels_path, "w") as f:
                json.dump(labels, f, indent=1, sort_keys=True)

    print(f"wrote {len(labels)} labels to {labels_path}")


if __name__ == "__main__":
    sys.exit(main())
