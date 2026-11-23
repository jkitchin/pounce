#!/usr/bin/env python3
"""Generate a small, tractable LP benchmark suite as AMPL ``.nl`` files.

Two sources, both classic, both small/solvable LPs meant for *validation*
(an interior-point solver should reach optimality quickly) -- this is
deliberately not a stress test (the hard, multi-million-row Mittelmann
``lpopt`` instances live in ``benchmarks/lpopt/``):

  1. The **NETLIB LP** test set
     (<https://www.netlib.org/lp/data/>) -- the canonical ~100-instance LP
     collection. Files are stored in netlib's compressed format and must be
     expanded with the netlib ``emps`` program (``emps.c``); this script
     fetches and compiles ``emps.c`` (old K&R C, built with ``cc -w``) and
     pipes each file through it to get standard MPS.

  2. A **size-filtered subset of the Csaba Meszaros LP test set**
     (<https://old.sztaki.hu/~meszaros/public_ftp/lptestset/>, the live
     mirror of the sztaki FTP). Subdirs ``misc/``, ``problematic/``,
     ``stochlp/``, ``New/`` hold gzip'd netlib-format files; we gunzip then
     ``emps``-expand them. The full set runs to millions of rows, so a SIZE
     FILTER keeps only the small ones (see the caps below).

Both formats yield standard MPS, which is converted to ``.nl`` with
``mps_to_nl.convert`` (HiGHS ``highspy`` parse -> Pyomo -> ASL ``.nl`` writer
-- the proven converter copied verbatim from ``benchmarks/lpopt/``). The
converter preserves the objective constant/offset (``-rhs`` of the objective
row, the MPS / HiGHS convention), so the ``.nl`` objective includes it.

Convention (matches MPS and HiGHS):

    minimize   c' x + offset
    subject to row_lower <= A x <= row_upper,  col_lower <= x <= col_upper

For every converted NETLIB instance we cross-check the solved objective
against the published optimum from the netlib readme table (parsed and
embedded below) when ``--validate`` is given; ``generate_nl.py`` prints the
relative error so a regression in the conversion is caught immediately.

The readme table reports ``c' x`` *without* the objective constant. Only
``e226`` has one (offset +7.113: published -18.7519, with the constant
-11.6389), so the check compares against ``published + offset``, with the
offset read off the MPS at conversion time. A conversion that dropped the
constant would therefore still be flagged, rather than e226 being flagged
on every run for a known non-bug.

Usage:
    python3 generate_nl.py                 # fetch + convert all (under cap)
    python3 generate_nl.py --netlib-only   # NETLIB set only
    python3 generate_nl.py --meszaros-only # Meszaros subset only
    python3 generate_nl.py --validate      # also run pounce on NETLIB and
                                           #   compare objective vs published
    python3 generate_nl.py afiro blend     # only the named instances
    python3 generate_nl.py --max-vars 5000 # tighten/loosen the size cap

The downloaded source files (under ``mps/``) and the generated ``.nl`` are
regenerated locally and gitignored; only this driver and ``mps_to_nl.py``
are tracked, so the suite is reproducible.
"""
from __future__ import annotations

import argparse
import gzip
import os
import re
import shutil
import subprocess
import sys

import mps_to_nl

HERE = os.path.dirname(os.path.abspath(__file__))
NL_DIR = os.path.join(HERE, "nl")
MPS_DIR = os.path.join(HERE, "mps")            # downloads + emps binary (gitignored)
DATA_DIR = os.path.join(HERE, "data")          # raw source cache (gitignored)
POUNCE_BIN = os.path.join(HERE, "..", "..", "target", "release", "pounce")

NETLIB_BASE = "https://www.netlib.org/lp/data"
MESZAROS_BASE = "https://old.sztaki.hu/~meszaros/public_ftp/lptestset"
MESZAROS_DIRS = ["misc", "problematic", "stochlp", "New"]  # feasible LPs only

# Size caps -- keep the suite clearly tractable for an IPM. The point is
# small/fast/solvable LPs, not a stress test.
MAX_VARS = 10_000
MAX_CONS = 10_000
MAX_NNZ = 200_000

# NETLIB instances that are NOT plain emps files in lp/data/ and so cannot
# be fetched + expanded by this driver:
#   qap8/qap12/qap15 -- not present in lp/data/ (404; "see NOTES")
#   stocfor3, truss  -- distributed as Fortran-source shar bundles
#   standgub         -- contains GUB 'MARKER' lines (no MINOS optimum)
# These are skipped automatically (download fails or no emps expansion).
NETLIB_SKIP = {"qap8", "qap12", "qap15", "stocfor3", "stocfor3.old",
               "truss", "emps.c", "emps.f", "emps.exe.gz", "minos",
               "mpc.src", "ascii", "changes", "readme", "nams.ps.gz",
               "standgub"}


# --------------------------------------------------------------------------
# Published NETLIB optima (objective, minimization) parsed from the
# lp/data/readme PROBLEM SUMMARY TABLE. Used only for the --validate
# cross-check. Values are strings exactly as published.
# --------------------------------------------------------------------------
NETLIB_OPTIMA = {
    "25fv47": 5.5018458883e+03, "80bau3b": 9.8723216072e+05,
    "adlittle": 2.2549496316e+05, "afiro": -4.6475314286e+02,
    "agg": -3.5991767287e+07, "agg2": -2.0239252356e+07,
    "agg3": 1.0312115935e+07, "bandm": -1.5862801845e+02,
    "beaconfd": 3.3592485807e+04, "blend": -3.0812149846e+01,
    "bnl1": 1.9776292856e+03, "bnl2": 1.8112365404e+03,
    "boeing1": -3.3521356751e+02, "boeing2": -3.1501872802e+02,
    "bore3d": 1.3730803942e+03, "brandy": 1.5185098965e+03,
    "capri": 2.6900129138e+03, "cycle": -5.2263930249e+00,
    "czprob": 2.1851966989e+06, "d2q06c": 1.2278423615e+05,
    "d6cube": 3.1549166667e+02, "degen2": -1.4351780000e+03,
    "degen3": -9.8729400000e+02, "dfl001": 1.12664e+07,
    "e226": -1.8751929066e+01, "etamacro": -7.5571521774e+02,
    "fffff800": 5.5567961165e+05, "finnis": 1.7279096547e+05,
    "fit1d": -9.1463780924e+03, "fit1p": 9.1463780924e+03,
    "fit2d": -6.8464293294e+04, "fit2p": 6.8464293232e+04,
    "forplan": -6.6421873953e+02, "ganges": -1.0958636356e+05,
    "gfrd-pnc": 6.9022359995e+06, "greenbea": -7.2462405908e+07,
    "greenbeb": -4.3021476065e+06, "grow15": -1.0687094129e+08,
    "grow22": -1.6083433648e+08, "grow7": -4.7787811815e+07,
    "israel": -8.9664482186e+05, "kb2": -1.7499001299e+03,
    "lotfi": -2.5264706062e+01, "maros": -5.8063743701e+04,
    "maros-r7": 1.4971851665e+06, "modszk1": 3.2061972906e+02,
    "nesm": 1.4076073035e+07, "perold": -9.3807580773e+03,
    "pilot": -5.5740430007e+02, "pilot.ja": -6.1131344111e+03,
    "pilot.we": -2.7201027439e+06, "pilot4": -2.5811392641e+03,
    "pilot87": 3.0171072827e+02, "pilotnov": -4.4972761882e+03,
    "recipe": -2.6661600000e+02, "sc105": -5.2202061212e+01,
    "sc205": -5.2202061212e+01, "sc50a": -6.4575077059e+01,
    "sc50b": -7.0000000000e+01, "scagr25": -1.4753433061e+07,
    "scagr7": -2.3313892548e+06, "scfxm1": 1.8416759028e+04,
    "scfxm2": 3.6660261565e+04, "scfxm3": 5.4901254550e+04,
    "scorpion": 1.8781248227e+03, "scrs8": 9.0429998619e+02,
    "scsd1": 8.6666666743e+00, "scsd6": 5.0500000078e+01,
    "scsd8": 9.0499999993e+02, "sctap1": 1.4122500000e+03,
    "sctap2": 1.7248071429e+03, "sctap3": 1.4240000000e+03,
    "seba": 1.5711600000e+04, "share1b": -7.6589318579e+04,
    "share2b": -4.1573224074e+02, "shell": 1.2088253460e+09,
    "ship04l": 1.7933245380e+06, "ship04s": 1.7987147004e+06,
    "ship08l": 1.9090552114e+06, "ship08s": 1.9200982105e+06,
    "ship12l": 1.4701879193e+06, "ship12s": 1.4892361344e+06,
    "sierra": 1.5394362184e+07, "stair": -2.5126695119e+02,
    "standata": 1.2576995000e+03, "standmps": 1.4060175000e+03,
    "stocfor1": -4.1131976219e+04, "stocfor2": -3.9024408538e+04,
    "tuff": 2.9214776509e-01, "vtp.base": 1.2983146246e+05,
    "wood1p": 1.4429024116e+00, "woodw": 1.3044763331e+00,
}

# The full NETLIB instance list (lp/data file names). These are the plain
# emps files; the skip-set above filters non-emps entries.
NETLIB_INSTANCES = sorted(NETLIB_OPTIMA.keys())


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def run(cmd, **kw):
    return subprocess.run(cmd, check=True, **kw)


def ensure_emps():
    """Fetch and compile netlib emps.c -> mps/emps (old K&R C; cc -w)."""
    emps_bin = os.path.join(MPS_DIR, "emps")
    if os.path.exists(emps_bin) and os.access(emps_bin, os.X_OK):
        return emps_bin
    src = os.path.join(MPS_DIR, "emps.c")
    if not os.path.exists(src):
        run(["curl", "-fsSL", f"{NETLIB_BASE}/emps.c", "-o", src])
    cc = os.environ.get("CC", "cc")
    run([cc, "-w", "-o", emps_bin, src])
    return emps_bin


def fetch(url, dest):
    run(["curl", "-fsSL", "--max-time", "120", url, "-o", dest])


def expand_emps(emps_bin, compressed_path, mps_path):
    """emps-expand a netlib-format file to standard MPS."""
    with open(mps_path, "wb") as out:
        subprocess.run([emps_bin, compressed_path], check=True, stdout=out)


# Objective constant of each converted instance, read off its MPS by
# ``dims`` (``-rhs`` of the objective row). The NETLIB readme optima exclude
# it, so ``--validate`` adds it back before comparing.
OBJ_OFFSETS: dict[str, float] = {}


def dims(mps_path):
    """Return (n, m, nnz, objective offset) for a standard-MPS file."""
    import highspy
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.readModel(mps_path)
    lp = h.getModel().lp_
    return (lp.num_col_, lp.num_row_, len(list(lp.a_matrix_.value_)),
            float(lp.offset_))


def solve_objective(nl_path, timeout=120):
    """Run pounce on a .nl, return (status, objective) parsed from stdout."""
    try:
        p = subprocess.run([POUNCE_BIN, nl_path], capture_output=True,
                           text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return "Timeout", None
    out = p.stdout + p.stderr
    obj = None
    # pounce's summary line prints two values: "Objective..: <scaled> <unscaled>".
    # The published optimum corresponds to the *unscaled* (second) value, so
    # grab the last float on the Objective summary line.
    mo = re.findall(r"^Objective[. :]+(.+)$", out, re.M)
    if mo:
        nums = re.findall(r"[-+]?\d+\.\d+(?:[eE][-+]?\d+)?", mo[-1])
        if nums:
            try:
                obj = float(nums[-1])
            except ValueError:
                obj = None
    ms = re.findall(r"(?im)^Status:\s*(\w+)", out)
    status = ms[-1] if ms else ("Solve_Succeeded"
                                 if "Optimal Solution Found" in out else "?")
    return status, obj


# --------------------------------------------------------------------------
# converters
# --------------------------------------------------------------------------
def convert_one(name, mps_path, rows):
    """Size-screen and convert a standard-MPS file to nl/<name>.nl."""
    nl_path = os.path.join(NL_DIR, f"{name}.nl")
    n, m, nnz, offset = dims(mps_path)
    OBJ_OFFSETS[name] = offset
    if n > MAX_VARS or m > MAX_CONS or nnz > MAX_NNZ:
        print(f"DEFER {name}: n={n} m={m} nnz={nnz} exceeds cap "
              f"({MAX_VARS}/{MAX_CONS}/{MAX_NNZ})")
        rows.append((name, n, m, nnz, "deferred(too large)"))
        return None
    mps_to_nl.convert(mps_path, nl_path)
    print(f"OK    {name}: n={n} m={m} nnz={nnz} -> {nl_path}")
    rows.append((name, n, m, nnz, "converted"))
    return nl_path


def gen_netlib(emps_bin, names, rows):
    src_dir = os.path.join(DATA_DIR, "netlib")
    os.makedirs(src_dir, exist_ok=True)
    converted = []
    for name in names:
        if name in NETLIB_SKIP:
            continue
        raw = os.path.join(src_dir, name)
        mps_path = os.path.join(MPS_DIR, f"netlib_{name}.mps")
        try:
            if not os.path.exists(raw):
                fetch(f"{NETLIB_BASE}/{name}", raw)
            expand_emps(emps_bin, raw, mps_path)
            nl = convert_one(name, mps_path, rows)
            if nl:
                converted.append(name)
        except Exception as e:  # noqa: BLE001 -- keep the batch going
            print(f"FAIL  netlib/{name}: {e}")
            rows.append((name, "?", "?", "?", f"failed: {e}"))
        finally:
            if os.path.exists(mps_path):
                os.remove(mps_path)
    return converted


def list_meszaros(subdir):
    import urllib.request
    url = f"{MESZAROS_BASE}/{subdir}/"
    html = urllib.request.urlopen(url, timeout=120).read().decode("latin-1")
    return [m[:-3] for m in re.findall(r'href="([^"]+\.gz)"', html)]


def gen_meszaros(emps_bin, only, rows):
    src_dir = os.path.join(DATA_DIR, "meszaros")
    os.makedirs(src_dir, exist_ok=True)
    converted = []
    for subdir in MESZAROS_DIRS:
        try:
            names = list_meszaros(subdir)
        except Exception as e:  # noqa: BLE001
            print(f"WARN  could not list meszaros/{subdir}: {e}")
            continue
        for name in names:
            if only and name not in only:
                continue
            # avoid name clashes with netlib (same stem) by prefixing dir? No:
            # keep the published instance name; if it clashes, last wins.
            gz = os.path.join(src_dir, f"{name}.gz")
            raw = os.path.join(src_dir, name)
            mps_path = os.path.join(MPS_DIR, f"mesz_{name}.mps")
            try:
                if not os.path.exists(gz):
                    fetch(f"{MESZAROS_BASE}/{subdir}/{name}.gz", gz)
                with gzip.open(gz, "rb") as f, open(raw, "wb") as o:
                    shutil.copyfileobj(f, o)
                expand_emps(emps_bin, raw, mps_path)
                nl = convert_one(name, mps_path, rows)
                if nl:
                    converted.append(name)
            except Exception as e:  # noqa: BLE001
                print(f"FAIL  meszaros/{subdir}/{name}: {e}")
                rows.append((name, "?", "?", "?", f"failed: {e}"))
            finally:
                for p in (raw, mps_path):
                    if os.path.exists(p):
                        os.remove(p)
    return converted


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main():
    global MAX_VARS, MAX_CONS, MAX_NNZ, NL_DIR
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="*", help="only these instance names")
    ap.add_argument("--netlib-only", action="store_true")
    ap.add_argument("--meszaros-only", action="store_true")
    ap.add_argument("--validate", action="store_true",
                    help="run pounce on converted NETLIB instances and "
                         "compare objective vs published optimum")
    ap.add_argument("--max-vars", type=int, default=10_000)
    ap.add_argument("--max-cons", type=int, default=10_000)
    ap.add_argument("--max-nnz", type=int, default=200_000)
    ap.add_argument("--out-dir", default=NL_DIR,
                    help="where to write the .nl files "
                         "(default: ./nl next to this script). Set this to a "
                         "POUNCE_BENCH_DATA-rooted dir to keep the heavy .nl "
                         "out of the checkout.")
    args = ap.parse_args()

    MAX_VARS, MAX_CONS, MAX_NNZ = args.max_vars, args.max_cons, args.max_nnz
    NL_DIR = args.out_dir

    os.makedirs(NL_DIR, exist_ok=True)
    os.makedirs(MPS_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    emps_bin = ensure_emps()

    do_netlib = not args.meszaros_only
    do_mesz = not args.netlib_only
    only = set(args.names)

    rows = []  # (name, n, m, nnz, status)
    netlib_done = []
    if do_netlib:
        names = [n for n in NETLIB_INSTANCES if not only or n in only]
        print(f"=== NETLIB ({len(names)} candidates) ===")
        netlib_done = gen_netlib(emps_bin, names, rows)
    if do_mesz:
        print(f"=== Meszaros ({'/'.join(MESZAROS_DIRS)}) ===")
        gen_meszaros(emps_bin, only, rows)

    # manifest
    print("\n=== manifest ===")
    print(f"{'instance':24} {'n':>8} {'m':>8} {'nnz':>10}  status")
    for name, n, m, nnz, st in sorted(rows):
        print(f"{name:24} {str(n):>8} {str(m):>8} {str(nnz):>10}  {st}")
    nconv = sum(1 for r in rows if r[4] == "converted")
    ndefer = sum(1 for r in rows if "deferred" in r[4])
    nfail = sum(1 for r in rows if "failed" in r[4])
    print(f"\nconverted {nconv}, deferred {ndefer}, failed {nfail} "
          f"into {NL_DIR}")

    # optional published-optima cross-check (NETLIB only)
    if args.validate and netlib_done:
        # `expected` = published (which omits the objective constant) plus
        # the constant the MPS declares; see the module docstring.
        print("\n=== NETLIB validation (pounce vs published optimum) ===")
        print(f"{'instance':16} {'published':>18} {'offset':>10} "
              f"{'expected':>18} {'pounce':>18} {'rel.err':>10}  status")
        for name in netlib_done:
            ref = NETLIB_OPTIMA.get(name)
            offset = OBJ_OFFSETS.get(name, 0.0)
            expected = None if ref is None else ref + offset
            nl_path = os.path.join(NL_DIR, f"{name}.nl")
            status, obj = solve_objective(nl_path)
            if expected is None or obj is None:
                rel = float("nan")
            else:
                denom = max(abs(expected), 1.0)
                rel = abs(obj - expected) / denom
            flag = "" if (rel == rel and rel < 1e-4) else "  <-- CHECK"
            print(f"{name:16} {ref!r:>18} {offset:>10.6g} "
                  f"{(f'{expected:.10e}' if expected is not None else 'n/a'):>18} "
                  f"{(f'{obj:.10e}' if obj is not None else 'n/a'):>18} "
                  f"{rel:>10.2e}  {status}{flag}")

    if not os.path.exists(os.path.join(NL_DIR, "afiro.nl")):
        print("\nNOTE: nl/afiro.nl (the Make stamp) was not produced.",
              file=sys.stderr)


if __name__ == "__main__":
    main()
