#!/usr/bin/env python3
"""
Unified benchmark report for POUNCE vs Ipopt.

For each suite it merges the per-release POUNCE run
(benchmarks/<suite>/pounce.json) with the committed Ipopt-MA57 reference
(benchmarks/<suite>/ipopt_ma57.json), both emitted by the shared
benchmarks/scripts/run_nl_bench.sh .nl driver, and produces a single
BENCHMARK_REPORT.md with per-suite and combined statistics.

Usage:
    python benchmark_report.py [--output BENCHMARK_REPORT.md]
    python benchmark_report.py --baseline old_report.json  # regression detection
"""

import json
import math
import os
import sys
from collections import defaultdict
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ---- Helpers ----

_OPTIMAL_STATUSES = {'Optimal', 'Solve_Succeeded'}
_ACCEPTABLE_STATUSES = {'Acceptable', 'Solved_To_Acceptable_Level'}


def normalize_status(status):
    """Map raw POUNCE/Ipopt status strings to the short labels used in the
    report ('Optimal', 'Acceptable', or the raw status for failures).

    Suites may emit either the long Ipopt-style enum names
    (`Solve_Succeeded`, `Solved_To_Acceptable_Level`) or the short
    labels; both are normalized here.
    """
    if status in _OPTIMAL_STATUSES:
        return 'Optimal'
    if status in _ACCEPTABLE_STATUSES:
        return 'Acceptable'
    return status


def is_solved(status):
    """Strict-Optimal only.

    Per the project's "Honesty in Benchmarks" rule (see CLAUDE.md),
    Acceptable is *not* counted as solved in summary metrics — it is
    surfaced in its own "Acceptable (not Optimal)" section. A solver
    that returns Acceptable has not converged to the requested
    tolerance and the result should not inflate the pass rate.
    """
    return status in _OPTIMAL_STATUSES


def obj_diff(ro, co):
    """Relative objective difference with floor of 1.0."""
    if ro is None or co is None:
        return float('nan')
    if not isinstance(ro, (int, float)) or not isinstance(co, (int, float)):
        return float('nan')
    if math.isnan(ro) or math.isnan(co):
        return float('nan')
    denom = max(abs(co), abs(ro), 1.0)
    return abs(ro - co) / denom


def fmt_time(t):
    if t is None or (isinstance(t, float) and math.isnan(t)):
        return "N/A"
    if t >= 1.0:
        return f"{t:.2f}s"
    elif t >= 0.001:
        return f"{t*1000:.1f}ms"
    else:
        return f"{t*1e6:.0f}us"


def geo_mean(values):
    """Geometric mean of positive values."""
    pos = [v for v in values if v > 0]
    if not pos:
        return float('nan')
    return math.exp(sum(math.log(v) for v in pos) / len(pos))


def median(values):
    if not values:
        return float('nan')
    s = sorted(values)
    return s[len(s) // 2]


def compute_stats(diffs):
    if not diffs:
        return float('nan'), float('nan'), float('nan')
    return sum(diffs) / len(diffs), median(diffs), max(diffs)


# ---- Load results ----

def _read_records(path):
    """Read a results JSON array, or [] when the file is absent/empty."""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return []
    with open(path) as f:
        return json.load(f)


def _build_comparisons(records, suite_name, left_key='pounce', right_key='ipopt'):
    """Build the canonical comparison list from a flat list of
    {solver,name,n,m,status,objective,iterations,solve_time} records.

    `left_key`/`right_key` select which `solver` values fill the two slots
    of each comparison (the `pounce_*` and `ipopt_*` dict fields are just
    internal slot names). The default ('pounce','ipopt') is the standard
    pounce-vs-ipopt suite; the head-to-head suites pass ('convex','nlp')."""
    pounce_by_name = {}
    ipopt_by_name = {}
    for r in records:
        if r['solver'] == left_key:
            pounce_by_name[r['name']] = r
        elif r['solver'] == right_key:
            ipopt_by_name[r['name']] = r

    comparisons = []
    for name in sorted(set(pounce_by_name.keys()) | set(ipopt_by_name.keys())):
        rr = pounce_by_name.get(name, {})
        cr = ipopt_by_name.get(name, {})

        r_solved = is_solved(rr.get('status', ''))
        c_solved = is_solved(cr.get('status', ''))
        both = r_solved and c_solved
        od = obj_diff(rr.get('objective'), cr.get('objective')) if both else float('nan')

        comparisons.append({
            'name': name,
            'suite': suite_name,
            'n': rr.get('n', cr.get('n', 0)),
            'm': rr.get('m', cr.get('m', 0)),
            'pounce_status': normalize_status(rr.get('status', 'N/A')),
            'ipopt_status': normalize_status(cr.get('status', 'N/A')),
            'pounce_obj': rr.get('objective', float('nan')),
            'ipopt_obj': cr.get('objective', float('nan')),
            'obj_diff': od,
            'pounce_iters': rr.get('iterations', 0),
            'ipopt_iters': cr.get('iterations', 0),
            'pounce_time': rr.get('solve_time', 0),
            'ipopt_time': cr.get('solve_time', 0),
            'pounce_solved': r_solved,
            'ipopt_solved': c_solved,
            'both_solved': both,
            'passed': both and not math.isnan(od) and od < 1e-4,
            # No record at all on the reference side -- not "the reference
            # failed". See `main`: these rows are kept out of every
            # head-to-head count.
            'ref_missing': name not in ipopt_by_name,
        })

    return comparisons


def load_suite(suite_name, dirname,
               left_file='pounce.json', right_file='ipopt_ma57.json',
               left_key='pounce', right_key='ipopt'):
    """Load one .nl suite by merging its two solver-arm result files.

    By default this merges benchmarks/<dirname>/pounce.json (regenerated
    every release) with benchmarks/<dirname>/ipopt_ma57.json (committed
    reference). The head-to-head suites override the filenames/keys to
    merge convex.json (left) with nlp.json (right).
    Returns (comparisons, has_left, has_right); comparisons is None when
    neither file is present.
    """
    base = os.path.join(SCRIPT_DIR, dirname)
    left = _read_records(os.path.join(base, left_file))
    right = _read_records(os.path.join(base, right_file))
    if not left and not right:
        return None, False, False
    comps = _build_comparisons(left + right, suite_name, left_key, right_key)
    return (comps if comps else None), bool(left), bool(right)


def _make_comparison(name, suite, n, m, p_status, i_status, p_obj, i_obj,
                     p_iters, i_iters, p_time, i_time):
    """Build the canonical comparison dict used by the report tables."""
    p_status = normalize_status(p_status)
    i_status = normalize_status(i_status)
    p_solved = is_solved_norm(p_status)
    i_solved = is_solved_norm(i_status)
    both = p_solved and i_solved
    od = obj_diff(p_obj, i_obj) if both else float('nan')
    return {
        'name': name,
        'suite': suite,
        'n': n,
        'm': m,
        'pounce_status': p_status,
        'ipopt_status': i_status,
        'pounce_obj': p_obj if p_obj is not None else float('nan'),
        'ipopt_obj': i_obj if i_obj is not None else float('nan'),
        'obj_diff': od,
        'pounce_iters': p_iters,
        'ipopt_iters': i_iters,
        'pounce_time': p_time,
        'ipopt_time': i_time,
        'pounce_solved': p_solved,
        'ipopt_solved': i_solved,
        'both_solved': both,
        'passed': both and not math.isnan(od) and od < 1e-4,
    }


def is_solved_norm(status):
    """is_solved that operates on already-normalized status labels."""
    return status == 'Optimal'


# ---- Report generation ----

def suite_summary(name, comps):
    """Generate summary stats for a suite."""
    total = len(comps)
    r_solved = sum(1 for c in comps if c['pounce_solved'])
    i_solved = sum(1 for c in comps if c['ipopt_solved'])
    both = sum(1 for c in comps if c['both_solved'])
    passed = sum(1 for c in comps if c['passed'])

    r_optimal = sum(1 for c in comps if c['pounce_status'] == 'Optimal')
    r_acceptable = sum(1 for c in comps if c['pounce_status'] == 'Acceptable')
    i_optimal = sum(1 for c in comps if c['ipopt_status'] == 'Optimal')
    i_acceptable = sum(1 for c in comps if c['ipopt_status'] == 'Acceptable')

    r_only = sum(1 for c in comps if c['pounce_solved'] and not c['ipopt_solved'])
    i_only = sum(1 for c in comps if c['ipopt_solved'] and not c['pounce_solved'])

    return {
        'name': name, 'total': total,
        'r_solved': r_solved, 'i_solved': i_solved, 'both': both, 'passed': passed,
        'r_optimal': r_optimal, 'r_acceptable': r_acceptable,
        'i_optimal': i_optimal, 'i_acceptable': i_acceptable,
        'r_only': r_only, 'i_only': i_only,
    }


def speed_stats(comps):
    """Compute speed comparison stats for commonly-solved problems."""
    both_data = [c for c in comps if c['both_solved']
                 and c['pounce_time'] > 0 and c['ipopt_time'] > 0]
    if not both_data:
        return None

    speedups = [c['ipopt_time'] / c['pounce_time'] for c in both_data]
    r_times = [c['pounce_time'] for c in both_data]
    i_times = [c['ipopt_time'] for c in both_data]
    r_iters = [c['pounce_iters'] for c in both_data]
    i_iters = [c['ipopt_iters'] for c in both_data]

    return {
        'n_problems': len(both_data),
        'geo_mean_speedup': geo_mean(speedups),
        'median_speedup': median(speedups),
        'r_faster_count': sum(1 for s in speedups if s > 1.0),
        'i_faster_count': sum(1 for s in speedups if s < 1.0),
        'r_10x_faster': sum(1 for s in speedups if s > 10.0),
        'r_total_time': sum(r_times),
        'i_total_time': sum(i_times),
        'r_median_time': median(r_times),
        'i_median_time': median(i_times),
        'r_mean_iters': sum(r_iters) / len(r_iters),
        'i_mean_iters': sum(i_iters) / len(i_iters),
        'r_median_iters': median(r_iters),
        'i_median_iters': median(i_iters),
    }


def failure_analysis(comps):
    """Categorize failures by status."""
    r_failures = defaultdict(int)
    i_failures = defaultdict(int)
    for c in comps:
        if not c['pounce_solved']:
            r_failures[c['pounce_status']] += 1
        if not c['ipopt_solved']:
            i_failures[c['ipopt_status']] += 1
    return dict(r_failures), dict(i_failures)


# Thread-pinning provenance -------------------------------------------
#
# The report's "Threading & timing" note used to be a hardcoded claim that
# the runs were pinned to a single compute thread. It printed whether or
# not that was true. It matters: POUNCE's dense linear algebra (faer/rayon)
# parallelizes across cores, so an unpinned POUNCE run is not comparable to
# the saved single-threaded Ipopt reference, and the report would have said
# it was.
#
# The note is now derived. `run_nl_bench.sh` stamps the thread environment
# it actually ran under into <suite>/pounce.env.json; this reads them back.
# The four variables below are the ones the original claim named — a run
# counts as pinned only when all four were set to 1, because leaving (say)
# RAYON_NUM_THREADS unset lets rayon use every core no matter what OMP says.
PIN_VARS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS', 'RAYON_NUM_THREADS')


def read_env_stamp(dirname):
    """Thread stamp written beside <dirname>/pounce.json, or None.

    None means the suite was produced by a runner that predates the stamp
    (or by hand) — reported as unrecorded, never as pinned."""
    path = os.path.join(SCRIPT_DIR, dirname, 'pounce.env.json')
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            stamp = json.load(f)
    except (OSError, ValueError):
        return None
    return stamp if isinstance(stamp, dict) else None


def _clean_threads(threads):
    """Drop non-values so 'recorded as unset' reads as unset, not as a
    setting named "unset" (the Makefile's sentinel for an absent var)."""
    if not isinstance(threads, dict):
        return {}
    return {k: v for k, v in threads.items()
            if v is not None and str(v) != 'unset'}


def _pin_verdict(threads):
    """('pinned' | 'unpinned', detail) for one recorded thread mapping."""
    threads = _clean_threads(threads)
    missing = [v for v in PIN_VARS if v not in threads]
    other = {v: threads[v] for v in PIN_VARS
             if v in threads and str(threads[v]) != '1'}
    if not missing and not other:
        return 'pinned', ''
    bits = []
    if other:
        bits.append(", ".join(f"`{v}`={w}" for v, w in sorted(other.items())))
    if missing:
        bits.append(", ".join(f"`{v}` unset" for v in missing))
    return 'unpinned', "; ".join(bits)


def _timeout_cutoffs(comps):
    """Per suite, the wall-clock at which POUNCE runs were actually cut off.

    The POUNCE arm gets no time-limit flag — `run_nl_bench.sh` wraps it in
    `timeout $BENCH_TIMELIMIT` and relabels rc=124 as
    `Maximum_CpuTime_Exceeded` — so the limit in force is not recorded
    anywhere in the results. The longest killed run in a suite is therefore
    the only evidence of it, and it is a tight lower bound: a run cannot be
    killed before its limit.
    """
    cutoffs = {}
    for c in comps:
        if 'CpuTime' in c['pounce_status'] or 'Timeout' in c['pounce_status']:
            s = c['suite']
            cutoffs[s] = max(cutoffs.get(s, 0.0), c.get('pounce_time') or 0.0)
    return cutoffs


def _followup_for(prov, name):
    """Any out-of-band result recorded for an instance the limits decide.

    Keyed by instance name under `followups` in the Ipopt provenance file.
    Such a run is deliberately *not* merged into the results — it was made
    at a different limit than the sweep — but leaving no trace of it would
    overstate what is unknown, so the report cites it and says where it
    lives."""
    return (prov.get('ipopt_followups') or {}).get(name)


def _pounce_limits(env_stamps):
    """Per suite, the time limit the POUNCE arm was actually given.

    `run_nl_bench.sh` stamps the `timeout` it wrapped the solver in into
    <suite>/pounce.env.json, so this is a record rather than an inference.
    Keyed by suite display name, matching `read_env_stamp`'s caller.
    """
    limits = {}
    for name, stamp in (env_stamps or {}).items():
        if not stamp:
            continue
        lim = stamp.get('timelimit')
        if isinstance(lim, (int, float)):
            limits[name] = lim
    return limits


def time_limit_note(prov, comps, env_stamps=None):
    """Build the Time limits note, disclosing any per-suite Ipopt override.

    The base provenance stamp carries one `timelimit`, but
    `ipopt_ma57.provenance.json` may override it per suite — mittelmann's
    reference was regenerated at 1800s after a threading bug truncated it at
    300s CPU. Printing only the base number states a limit the Ipopt column
    did not actually run under, and hides that a suite may compare a
    300s POUNCE arm against a 1800s reference. So: print the base, print
    every override, and name any instance the asymmetry decides.

    The POUNCE limit is *recorded*, not guessed: `run_nl_bench.sh` writes it
    into <suite>/pounce.env.json. This used to be inferred from the longest
    run that was killed, which had two failure modes — it read a genuine
    solver-side timeout as evidence of the harness limit, and it went blind
    exactly when the suite stopped timing out, since a run with no kills
    leaves nothing to infer from. Prefer the stamp; fall back to the
    inference only for results predating it (gh#581).
    """
    base = prov.get('ipopt_timelimit')
    overrides = prov.get('ipopt_suite_overrides') or {}
    if base is None and not overrides:
        return []

    cutoffs = _timeout_cutoffs(comps)
    limits = _pounce_limits(env_stamps)

    if limits:
        how = ("The POUNCE arm carries no time-limit flag — it is wrapped in "
               "`timeout $BENCH_TIMELIMIT` and a kill is recorded as "
               "`Maximum_CpuTime_Exceeded` — but the limit in force is stamped "
               "per suite into `<suite>/pounce.env.json` and is read from there "
               "below.")
    else:
        how = ("The POUNCE arm carries no time-limit flag — it is wrapped in "
               "`timeout $BENCH_TIMELIMIT` (default 300s) and a kill is recorded "
               "as `Maximum_CpuTime_Exceeded` — and these results carry no "
               "`pounce.env.json` stamp, so its limit is inferred here from the "
               "longest run that was killed.")

    lines = [f"> **Time limits.** The saved Ipopt reference ran at "
             f"`max_cpu_time` = {base}s unless overridden below. {how}"]

    by_lower = {s.lower(): s for s in set(cutoffs) | set(limits)}

    affected = []
    for suite, ov in sorted(overrides.items()):
        lim = ov.get('timelimit', '?')
        why = ov.get('why', '')
        detail = f"regenerated {ov.get('generated', '?')}"
        if ov.get('threads'):
            detail += f", threads {ov['threads']}"
        lines.append(f"> Override — **{suite}**: Ipopt reference at {lim}s "
                     f"({detail}).")
        if why:
            lines.append(f"> Reason given: {why}")

        display = by_lower.get(suite.lower())
        if display and isinstance(lim, (int, float)):
            # Recorded beats inferred. `cut` is what POUNCE was allowed; it
            # only doubles as "where runs were killed" in the fallback.
            stamped = display in limits
            cut = limits[display] if stamped else cutoffs.get(display)
            if cut is None:
                continue
            if stamped and cut == lim:
                lines.append(f"> POUNCE ran this suite at the same {cut:.0f}s "
                             "(recorded in `pounce.env.json`), so both columns "
                             "are held to the same clock here.")
                # Same clock — nothing for the asymmetry to decide.
                continue
            if stamped:
                lines.append(f"> POUNCE ran this suite at {cut:.0f}s (recorded "
                             f"in `pounce.env.json`) against the reference's "
                             f"{lim}s, so the two columns are **not** held to "
                             "the same clock here.")
            else:
                lines.append(f"> POUNCE runs in this suite were cut off at "
                             f"~{cut:.0f}s, so the two columns are **not** held "
                             "to the same clock here.")
            # An instance is decided by the asymmetry only if POUNCE was
            # killed AND Ipopt needed longer than POUNCE was ever allowed.
            for c in comps:
                if (c['suite'] == display
                        and ('CpuTime' in c['pounce_status']
                             or 'Timeout' in c['pounce_status'])
                        and c['ipopt_status'] == 'Optimal'
                        and (c.get('ipopt_time') or 0.0) > cut):
                    affected.append((c, cut))

    for c, cut in affected:
        line = (f"> Decided by that gap: **{c['name']}** — POUNCE cut off "
                f"at {c['pounce_time']:.0f}s ({c['pounce_iters']} iters), "
                f"Ipopt Optimal at {c['ipopt_time']:.0f}s "
                f"({c['ipopt_iters']} iters), i.e. past POUNCE's cutoff. It is "
                "counted here as an Ipopt-only solve, on a limit POUNCE was "
                "never given.")
        followup = _followup_for(prov, c['name'])
        if followup:
            line += f" {followup}"
        lines.append(line)

    return lines


def threading_note(stamps, ipopt_threads=None):
    """Build the Threading & timing note from what was actually recorded.

    `stamps` maps suite display name -> stamp dict or None. Every branch
    states its own evidence, so a reader can tell a per-suite record from a
    report-time guess from nothing at all."""
    recorded = {n: s for n, s in stamps.items() if s and s.get('threads')}
    unrecorded = sorted(n for n in stamps if n not in recorded)

    lines = []
    if not recorded:
        # Nothing stamped. The only evidence left is this process's own
        # environment, which is the sweep's environment when the report is
        # generated by `make benchmark` (it runs as the sweep's last step)
        # and unrelated to it when generated later by `make
        # benchmark-report`. We cannot tell which, so say so.
        env = {v: os.environ[v] for v in PIN_VARS if v in os.environ}
        verdict, detail = _pin_verdict(env)
        lines.append("> **Threading & timing.** These POUNCE runs carry no "
                     "per-suite thread stamp — they predate it, so the")
        lines.append("> settings they ran under were not recorded.")
        if verdict == 'pinned':
            lines.append("> At report time all of "
                         + ", ".join(f"`{v}`" for v in PIN_VARS)
                         + " = 1. When this report is generated as the last")
            lines.append("> step of `make -C benchmarks benchmark` that is the "
                         "sweep's own environment, but it is not proof: run")
            lines.append("> `benchmark-report` separately and it says nothing "
                         "about the runs.")
        else:
            lines.append(f"> At report time they are not pinned ({detail}), "
                         "which says nothing about the runs themselves.")
        lines.append("> Treat POUNCE-vs-Ipopt time comparisons below as "
                     "unverified on this axis.")
        return lines

    unpinned = {n: _pin_verdict(s['threads'])[1]
                for n, s in recorded.items()
                if _pin_verdict(s['threads'])[0] == 'unpinned'}

    if unpinned:
        lines.append("> **Threading & timing.** POUNCE runs were **not** pinned "
                     "to a single compute thread in every suite, as recorded")
        lines.append("> by the runner: "
                     + "; ".join(f"{n} ({d})" for n, d in sorted(unpinned.items()))
                     + ".")
        lines.append("> POUNCE's dense linear algebra (`faer`/`rayon`) "
                     "parallelizes across cores, so those suites' times are")
        lines.append("> multi-threaded and **not** comparable to the "
                     "single-threaded Ipopt reference.")
    else:
        # Every recorded value is "1" here — that is what 'pinned' means.
        lines.append("> **Threading & timing.** The POUNCE runs were pinned to "
                     "a single compute thread — "
                     + ", ".join(f"`{v}`" for v in PIN_VARS)
                     + " all = 1,")
        lines.append("> recorded by the runner for each suite, not assumed — "
                     "and run sequentially, so POUNCE and Ipopt solve times")
        lines.append("> are directly comparable on one host.")
        lines.append("> POUNCE's dense linear algebra (via `faer`/`rayon`) "
                     "parallelizes across cores, so its *multi-threaded*")
        lines.append("> wall-clock is up to ~2x faster on the larger dense "
                     "problems (e.g. Mittelmann `cont*`/`qcqp*`, QP); the")
        lines.append("> single-threaded times reported here are therefore a "
                     "controlled lower bound, not POUNCE's real-world speed,")
        lines.append("> and should not be compared against multi-threaded runs "
                     "of this report.")

    if unrecorded:
        lines.append("> Not recorded for: " + ", ".join(unrecorded)
                     + " — those suites predate the stamp and are unverified "
                       "on this axis.")

    # The Ipopt column is a saved reference from another machine on another
    # day, so the POUNCE finding never extends to it: report its own record,
    # or say it has none. References generated before `make ipopt-reference`
    # started stamping threads fall in the second case.
    ipopt_clean = _clean_threads(ipopt_threads)
    if ipopt_clean:
        verdict, detail = _pin_verdict(ipopt_clean)
        if verdict == 'pinned':
            lines.append("> The saved Ipopt reference recorded the same "
                         "pinning when it was generated.")
        else:
            lines.append(f"> The saved Ipopt reference was **not** pinned "
                         f"when generated ({detail}), so its times are not")
            lines.append("> comparable to the single-threaded POUNCE column.")
    else:
        lines.append("> The Ipopt reference column carries no thread record of "
                     "its own; its pinning is asserted by the procedure that")
        lines.append("> generated it, not verified here.")
    return lines


def collect_provenance():
    """Gather version + environment metadata for the report header.

    Read-only, never fails: every probe falls back to 'unknown' so the
    report still lands when (e.g.) we're outside a git checkout or the
    Ipopt binary isn't installed yet.
    """
    import subprocess

    def _run(args):
        try:
            return subprocess.run(args, capture_output=True, text=True,
                                  timeout=5, check=False).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return ''

    # POUNCE version from workspace Cargo.toml.
    pounce_version = 'unknown'
    cargo_toml = os.path.join(os.path.dirname(SCRIPT_DIR), 'Cargo.toml')
    try:
        with open(cargo_toml) as f:
            for line in f:
                line = line.strip()
                if line.startswith('version'):
                    pounce_version = line.split('=', 1)[1].strip().strip('"')
                    break
    except OSError:
        pass

    git_sha = _run(['git', '-C', os.path.dirname(SCRIPT_DIR), 'rev-parse', '--short', 'HEAD']) or 'unknown'
    git_branch = _run(['git', '-C', os.path.dirname(SCRIPT_DIR), 'rev-parse', '--abbrev-ref', 'HEAD']) or 'unknown'
    git_dirty = _run(['git', '-C', os.path.dirname(SCRIPT_DIR), 'status', '--porcelain'])
    if git_dirty:
        git_sha = f'{git_sha}-dirty'

    # Ipopt is no longer run during a release — its results come from the
    # committed reference. Read that reference's provenance stamp
    # (benchmarks/ipopt_ma57.provenance.json, written by
    # `make ipopt-reference`) so the report attributes the Ipopt column to
    # the machine/binary that actually produced it, not the current host.
    ipopt_version = 'no saved reference'
    ipopt_linear_solver = 'ma57 (via ref/Ipopt/install-ma57)'
    ipopt_reference = None
    ipopt_threads = None
    ipopt_timelimit = None
    ipopt_suite_overrides = {}
    ipopt_followups = {}
    prov_path = os.path.join(SCRIPT_DIR, 'ipopt_ma57.provenance.json')
    if os.path.exists(prov_path):
        try:
            with open(prov_path) as f:
                ref = json.load(f)
            ipopt_version = ref.get('ipopt_version', 'unknown')
            ipopt_linear_solver = ref.get('linear_solver', ipopt_linear_solver)
            ipopt_threads = ref.get('threads')
            ipopt_timelimit = ref.get('timelimit')
            ipopt_suite_overrides = ref.get('suite_overrides') or {}
            ipopt_followups = ref.get('followups') or {}
            ipopt_reference = (f"generated {ref.get('generated', '?')} on "
                               f"{ref.get('host', '?')} ({ref.get('platform', '?')}), "
                               f"git {ref.get('git_sha', '?')}, "
                               f"timelimit {ref.get('timelimit', '?')}s")
        except (OSError, ValueError):
            pass

    # Pounce default linear solver is FERAL — pounce-ma57 is the
    # MA57-feature build (not the default).
    return {
        'pounce_version': pounce_version,
        'pounce_linear_solver': 'feral (default)',
        'ipopt_version': ipopt_version,
        'ipopt_linear_solver': ipopt_linear_solver,
        'ipopt_reference': ipopt_reference,
        'ipopt_threads': ipopt_threads,
        'ipopt_timelimit': ipopt_timelimit,
        'ipopt_suite_overrides': ipopt_suite_overrides,
        'ipopt_followups': ipopt_followups,
        'git_sha': git_sha,
        'git_branch': git_branch,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S %Z').strip(),
        'platform': _run(['uname', '-srm']),
    }


def load_cute_status():
    """Per-problem reference status for the Vanderbei suite, from
    vanderbei/cute_table_status.json (derived from cute_table.pdf).
    Returns {name -> entry} or None."""
    path = os.path.join(SCRIPT_DIR, 'vanderbei', 'cute_table_status.json')
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f).get('problems', {})


# cute_table status → display order for the cross-check table.
_CUTE_ORDER = ['optimum', 'hard', 'infeasible', 'unbounded', 'untabulated']


def vanderbei_crosscheck_lines(comps):
    """Cross-check the Vanderbei POUNCE results against cute_table.pdf:
    report the expected-solvable denominator (problems with a documented
    finite optimum), break out the known hard / infeasible / unbounded /
    untabulated problems, and flag objectives that disagree with the
    literature reference."""
    status = load_cute_status()
    if not status:
        return []

    buckets = {k: [] for k in _CUTE_ORDER}
    for c in comps:
        s = status.get(c['name'], {}).get('status', 'untabulated')
        buckets.setdefault(s, []).append(c)

    expected = buckets['optimum']
    n_exp = len(expected)
    solved_exp = [c for c in expected if c['pounce_solved']]
    missed_exp = [c for c in expected if not c['pounce_solved']]

    # Objective cross-check: only flag when the three reference solvers
    # agreed among themselves (a single basin) and POUNCE landed elsewhere —
    # otherwise a difference just means multiple local optima.
    mism = []
    for c in solved_exp:
        e = status.get(c['name'], {})
        ref = e.get('ref_obj')
        if ref is None or not e.get('solvers_agree'):
            continue
        po = c['pounce_obj']
        if po is None or (isinstance(po, float) and math.isnan(po)):
            continue
        rel = abs(po - ref) / max(1.0, abs(ref))
        if rel > 1e-3:
            mism.append((c['name'], po, ref, rel))

    lines = []
    lines.append("## Vanderbei Reference Cross-Check")
    lines.append("")
    lines.append("Per-problem status from R. Vanderbei's `cute_table.pdf` "
                 "(`vanderbei/cute_table_status.json`). The meaningful "
                 "denominator is the **expected-solvable** set — problems with a "
                 "documented finite optimum — not all 733: the CUTE collection "
                 "deliberately includes unbounded, infeasible, and no-solver-finishes "
                 "problems.")
    lines.append("")
    lines.append("| cute_table status | problems | POUNCE solved | meaning |")
    lines.append("|---|---|---|---|")
    meanings = {
        'optimum': 'finite reference optimum exists (expected-solvable)',
        'hard': 'in table, but SNOPT+NITRO+LOQO all hit time/iter limits',
        'infeasible': 'a reference solver declared infeasibility',
        'unbounded': 'unbounded below',
        'untabulated': 'not in cute_table — no reference datum',
    }
    for s in _CUTE_ORDER:
        b = buckets.get(s, [])
        if not b:
            continue
        ns = sum(1 for c in b if c['pounce_solved'])
        lines.append(f"| {s} | {len(b)} | {ns} | {meanings[s]} |")
    lines.append("")

    pct = 100.0 * len(solved_exp) / n_exp if n_exp else 0.0
    lines.append(f"**POUNCE solved {len(solved_exp)} / {n_exp} expected-solvable "
                 f"({pct:.1f}%).** The hard / infeasible / unbounded / untabulated "
                 "rows above are excluded from this denominator — a POUNCE failure "
                 "there is shared with the commercial reference solvers and is not "
                 "counted as a miss.")
    lines.append("")

    if missed_exp:
        names = " ".join(sorted(c['name'] for c in missed_exp))
        lines.append(f"**Genuine misses — expected-solvable but POUNCE did not "
                     f"reach Optimal ({len(missed_exp)}):**")
        lines.append("")
        lines.append(f"> {names}")
        lines.append("")

    if mism:
        lines.append(f"**Objective disagreements vs. cute_table reference "
                     f"({len(mism)})** — POUNCE converged but to a different value "
                     "than the agreed reference optimum (possible wrong basin or "
                     "misread problem):")
        lines.append("")
        lines.append("| Problem | POUNCE obj | reference obj | rel. diff |")
        lines.append("|---|---|---|---|")
        for name, po, ref, rel in sorted(mism, key=lambda x: -x[3]):
            lines.append(f"| {name} | {po:.6e} | {ref:.6e} | {rel:.1e} |")
        lines.append("")
    else:
        lines.append("All solved expected-solvable objectives agree with the "
                     "cute_table reference (where the reference solvers themselves "
                     "agreed).")
        lines.append("")

    return lines


def generate_profiles(profile_dirs):
    """Render Dolan–Moré performance + data profiles and return the
    markdown lines that embed them.

    `profile_dirs` is a list of suite directory names that have BOTH a
    fresh pounce.json and an ipopt_ma57.json (a performance profile needs
    two solvers to be meaningful). Figures are written to
    benchmarks/figures/ and referenced with repo-relative paths so they
    render on GitHub and in local Markdown viewers alike.

    Degrades gracefully: if matplotlib is unavailable or nothing is
    plottable, returns a short note instead of figures so the report still
    generates.
    """
    lines = ["## Performance Profiles", ""]
    if not profile_dirs:
        lines += ["_No suite had both a POUNCE run and an Ipopt reference, "
                  "so no performance profile could be drawn._", ""]
        return lines

    fig_dir = os.path.join(SCRIPT_DIR, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    sys.path.insert(0, os.path.join(SCRIPT_DIR, "scripts"))
    try:
        from perf_profile import render_profile
    except Exception as exc:  # matplotlib/numpy missing, etc.
        lines += [f"_Profiles skipped: could not import the plotter "
                  f"({exc}). Install matplotlib + numpy and rerun "
                  "`make -C benchmarks benchmark-report`._", ""]
        return lines

    lines += [
        "[Dolan & Moré (2002)](https://doi.org/10.1007/s101070100263) "
        "performance profiles pooled over every suite with an Ipopt "
        "reference. ρ_s(τ) is the fraction of problems a solver solves "
        "within a factor τ of the fastest solver on each problem: the "
        "**height at τ=1** is how often it was the quickest, and the "
        "**right-hand plateau** is its overall robustness (fraction solved "
        "at all). A problem counts as solved only at strict/acceptable "
        "success; failures and timeouts are charged infinite cost. "
        "Regenerate or slice these with "
        "`python3 scripts/perf_profile.py <suite…> [--metric iters] "
        "[--mode data]`.",
        "",
    ]

    # (filename, kwargs, caption). Time profiles are only fair on one host;
    # the iterations profile is machine-independent, so we always include it.
    figs = [
        ("profile_performance_time.png",
         dict(metric="time", mode="performance"),
         "**Performance profile by wall-clock time.** Valid because POUNCE "
         "and Ipopt-MA57 were run interleaved on this host (see Provenance)."),
        ("profile_performance_iters.png",
         dict(metric="iters", mode="performance"),
         "**Performance profile by iteration count** — machine-independent, "
         "so it stays comparable across hosts and reruns."),
        ("profile_data_time.png",
         dict(metric="time", mode="data"),
         "**Data profile (absolute-time ECDF).** Fraction of problems solved "
         "within a given wall-clock budget, without best-solver "
         "normalization — reads directly as “how many by 1 s? by 10 s?”."),
    ]
    any_ok = False
    for fname, kw, caption in figs:
        out = os.path.join(fig_dir, fname)
        try:
            res = render_profile(profile_dirs, out, **kw)
        except Exception as exc:
            lines += [f"_Could not render {fname}: {exc}._", ""]
            continue
        if res is None:
            continue
        nprob, solvers = res
        any_ok = True
        lines += [f"![{caption}](figures/{fname})", "", caption,
                  f"  \n_{nprob} problems; solvers: "
                  f"{', '.join(solvers)}._", ""]
    if not any_ok:
        lines += ["_No problem was solved by enough solvers to draw a "
                  "profile._", ""]
    return lines


def head_to_head_lines(head_to_head):
    """Render the dedicated-convex-vs-general-NLP head-to-head section.

    `head_to_head` is a list of (suite_name, comps) where each comps was
    built with the 'convex' arm in the left slot (pounce_*) and the 'nlp'
    arm in the right slot (ipopt_*). This is a pounce-vs-pounce comparison
    on identical .nl problems, so it is rendered with its own labels and
    deliberately kept out of the Ipopt-reference machinery (profiles,
    regressions/wins, baseline).
    """
    if not head_to_head:
        return []

    lines = []
    lines.append("## Dedicated Convex Solver vs. General NLP (head-to-head)")
    lines.append("")
    lines.append("The same LP / convex-QP `.nl` problems solved twice by the **same**")
    lines.append("pounce binary: once routed to the dedicated convex interior-point")
    lines.append("solver (`pounce-convex`, via `solver_selection=lp-ipm` / `qp-ipm`) and")
    lines.append("once through the general NLP filter-IPM (`solver_selection=nlp`). This")
    lines.append("quantifies the speedup the dedicated solver buys on its home turf. It")
    lines.append("is a pounce-vs-pounce comparison and is independent of the Ipopt")
    lines.append("reference used by the suites above.")
    lines.append("")
    lines.append("> **The Ipopt-MA57 reference is a status and timing reference, not an")
    lines.append("> objective oracle.** It carries Ipopt's `bound_relax_factor` widening,")
    lines.append("> which changes the answer on constraint-degenerate models by `delta`")
    lines.append("> times the bound's multiplier. On the `LISWET*` / `YAO` families POUNCE")
    lines.append("> disagrees with it by tens of percent and POUNCE is correct: scored on")
    lines.append("> the published Maros-Meszaros optima (DOC 97/6) the convex arm is")
    lines.append("> 138/138 and Ipopt-MA57 misses 8. Score an objective disagreement")
    lines.append("> against DOC 97/6 (`compare_qp_four_way.py`) before calling it a")
    lines.append("> regression -- gh #744 was filed by not doing that.")
    lines.append("")

    for name, comps in head_to_head:
        s = suite_summary(name, comps)
        lines.append(f"### {name}")
        lines.append("")
        lines.append("| Metric | pounce-convex | pounce-nlp |")
        lines.append("|--------|---------------|------------|")
        lines.append(
            f"| Optimal | {s['r_optimal']}/{s['total']} "
            f"({100*s['r_optimal']/max(s['total'],1):.1f}%) "
            f"| {s['i_optimal']}/{s['total']} "
            f"({100*s['i_optimal']/max(s['total'],1):.1f}%) |"
        )
        lines.append(f"| Solved exclusively | {s['r_only']} | {s['i_only']} |")
        lines.append(f"| Both Optimal | {s['both']} | |")
        lines.append(f"| Matching objectives (< 0.01%) | {s['passed']}/{max(s['both'],1)} | |")
        lines.append("")

        sp = speed_stats(comps)
        if sp is None:
            lines.append("_No problem was solved by both arms — no speed comparison._")
            lines.append("")
            continue

        lines.append(f"On {sp['n_problems']} problems solved by both arms:")
        lines.append("")
        lines.append("| Metric | pounce-convex | pounce-nlp |")
        lines.append("|--------|---------------|------------|")
        lines.append(f"| Median time | {fmt_time(sp['r_median_time'])} | {fmt_time(sp['i_median_time'])} |")
        lines.append(f"| Total time | {fmt_time(sp['r_total_time'])} | {fmt_time(sp['i_total_time'])} |")
        lines.append(f"| Mean iterations | {sp['r_mean_iters']:.1f} | {sp['i_mean_iters']:.1f} |")
        lines.append(f"| Median iterations | {sp['r_median_iters']} | {sp['i_median_iters']} |")
        lines.append("")
        lines.append(f"- **Geometric-mean speedup (convex over nlp)**: {sp['geo_mean_speedup']:.1f}x")
        lines.append(f"- **Median speedup**: {sp['median_speedup']:.1f}x")
        lines.append(f"- pounce-convex faster: {sp['r_faster_count']}/{sp['n_problems']} "
                     f"({100*sp['r_faster_count']/sp['n_problems']:.0f}%)")
        lines.append(f"- pounce-convex 10x+ faster: {sp['r_10x_faster']}/{sp['n_problems']}")
        lines.append(f"- pounce-nlp faster: {sp['i_faster_count']}/{sp['n_problems']}")
        lines.append("")

    return lines


def unreferenced_lines(unreferenced):
    """Problems a suite now runs that its saved Ipopt reference predates.

    A suite that has a reference but grew since it was taken has problems with
    no Ipopt record at all. Scoring those as "Ipopt not Optimal" would credit
    POUNCE with wins nobody measured, so they are excluded from every
    head-to-head count above and listed here with POUNCE's result alone.
    """
    out = []
    out.append(f"## Problems without an Ipopt reference — {len(unreferenced)} problems")
    out.append("")
    out.append("These problems were added to a suite after its saved ipopt-ma57 "
               "reference was taken. They are **excluded from every POUNCE-vs-Ipopt "
               "count in this report** (the executive summary, the per-suite table, "
               "the profiles, wins and regressions) rather than scored as Ipopt "
               "failures. Refresh the suite's reference with "
               "`make -C benchmarks ipopt-ref-<suite>` to bring them into the comparison.")
    out.append("")
    out.append("| Problem | Suite | n | m | POUNCE status | POUNCE objective |")
    out.append("|---------|-------|---|---|---------------|------------------|")
    for c in unreferenced:
        ro = c['pounce_obj']
        ro_str = f"{ro:.6e}" if isinstance(ro, (int, float)) and not math.isnan(ro) else "N/A"
        out.append(f"| {c['name']} | {c['suite']} | {c['n']} | {c['m']} | {c['pounce_status']} | {ro_str} |")
    out.append("")
    return out


def generate_report(suites, output_path, baseline=None, profile_dirs=None,
                    head_to_head=None, env_stamps=None, unreferenced=None):
    """Generate the unified benchmark report."""
    prov = collect_provenance()
    lines = []
    lines.append("# POUNCE Benchmark Report")
    lines.append("")
    lines.append(f"Generated: {prov['timestamp']}")
    lines.append("")
    lines.append("## Provenance")
    lines.append("")
    lines.append("| Component | Version / Detail |")
    lines.append("|-----------|------------------|")
    lines.append(f"| POUNCE | v{prov['pounce_version']} ({prov['git_branch']} @ {prov['git_sha']}) |")
    lines.append(f"| POUNCE linear solver | {prov['pounce_linear_solver']} |")
    lines.append(f"| Ipopt | {prov['ipopt_version']} |")
    lines.append(f"| Ipopt linear solver | {prov['ipopt_linear_solver']} |")
    lines.append(f"| Platform | {prov['platform']} |")
    lines.append("")
    lines.append("POUNCE results were produced this run by `make -C benchmarks")
    lines.append("<suite>-run` (pounce only). The Ipopt column is a saved reference")
    lines.append("(`make -C benchmarks ipopt-reference`), rerun only when explicitly")
    if prov.get('ipopt_reference'):
        lines.append(f"regenerated — {prov['ipopt_reference']}. Ipopt solve *times* are")
        lines.append("from that reference machine and only comparable to POUNCE when this")
        lines.append("report is generated on the same host.")
    else:
        lines.append("regenerated. No saved reference is present, so suites without one")
        lines.append("are reported POUNCE-only.")
    lines.append("")
    lines.append("The GAMS solver-link path is exercised separately as a liveness")
    lines.append("smoke check (`make -C benchmarks gams-bench`) and is not aggregated here.")
    lines.append("")
    lines.extend(threading_note(env_stamps or {}, prov.get('ipopt_threads')))

    # Combined summary
    all_comps = []
    for name, comps in suites:
        all_comps.extend(comps)

    tl_note = time_limit_note(prov, all_comps, env_stamps)
    if tl_note:
        lines.append("")
        lines.extend(tl_note)
    lines.append("")

    combined = suite_summary("Combined", all_comps)

    # Count questionable Acceptable solutions
    r_acc_questionable = sum(1 for c in all_comps
                             if c['pounce_status'] == 'Acceptable'
                             and c['ipopt_status'] == 'Optimal'
                             and not math.isnan(c['obj_diff'])
                             and c['obj_diff'] > 0.01)

    lines.append("## Executive Summary")
    lines.append("")
    lines.append(f"| Metric | POUNCE | Ipopt |")
    lines.append(f"|--------|--------|-------|")
    lines.append(f"| Optimal (strict) | **{combined['r_optimal']}/{combined['total']}** ({100*combined['r_optimal']/max(combined['total'],1):.1f}%) | **{combined['i_optimal']}/{combined['total']}** ({100*combined['i_optimal']/max(combined['total'],1):.1f}%) |")
    lines.append(f"| Acceptable (informational, *not* counted as solved) | {combined['r_acceptable']} | {combined['i_acceptable']} |")
    lines.append(f"| Solved exclusively (strict Optimal) | {combined['r_only']} | {combined['i_only']} |")
    lines.append(f"| Both Optimal | {combined['both']} | |")
    lines.append(f"| Matching objectives (< 0.01%) | {combined['passed']}/{max(combined['both'],1)} | |")
    if r_acc_questionable > 0:
        lines.append(f"| Acceptable at worse local min | {r_acc_questionable} | |")
    lines.append("")
    lines.append("> **Note:** All headline counts use strict Optimal status only. `Acceptable`")
    lines.append("> means the iterate met relaxed tolerances but not the requested tolerance —")
    lines.append("> per CLAUDE.md's \"Honesty in Benchmarks\" rule it is reported separately and")
    lines.append("> never folded into the pass rate. See the \"Acceptable (not Optimal)\" and")
    lines.append("> \"Different Local Minima\" sections below.")
    lines.append("")

    # Performance / data profiles (Dolan–Moré) over suites with a reference.
    lines.extend(generate_profiles(profile_dirs or []))

    # Per-suite summary table
    lines.append("## Per-Suite Summary")
    lines.append("")
    lines.append("| Suite | Problems | POUNCE Optimal | Ipopt Optimal | POUNCE only | Ipopt only | Both Optimal | Match |")
    lines.append("|-------|----------|---------------|--------------|-------------|------------|--------------|-------|")
    for name, comps in suites:
        s = suite_summary(name, comps)
        lines.append(
            f"| {name} | {s['total']} "
            f"| {s['r_solved']} ({100*s['r_solved']/max(s['total'],1):.1f}%) "
            f"| {s['i_solved']} ({100*s['i_solved']/max(s['total'],1):.1f}%) "
            f"| {s['r_only']} | {s['i_only']} | {s['both']} "
            f"| {s['passed']}/{max(s['both'],1)} |"
        )
    lines.append("")
    if unreferenced:
        lines.extend(unreferenced_lines(unreferenced))

    # Vanderbei cross-check against the cute_table reference (if present).
    for name, comps in suites:
        if name == 'Vanderbei':
            lines.extend(vanderbei_crosscheck_lines(comps))
            break

    # Per-suite speed and iteration stats
    for name, comps in suites:
        sp = speed_stats(comps)
        if sp is None:
            continue

        lines.append(f"## {name} Suite — Performance")
        lines.append("")
        lines.append(f"On {sp['n_problems']} commonly-solved problems:")
        lines.append("")
        lines.append("| Metric | POUNCE | Ipopt |")
        lines.append("|--------|--------|-------|")
        lines.append(f"| Median time | {fmt_time(sp['r_median_time'])} | {fmt_time(sp['i_median_time'])} |")
        lines.append(f"| Total time | {fmt_time(sp['r_total_time'])} | {fmt_time(sp['i_total_time'])} |")
        lines.append(f"| Mean iterations | {sp['r_mean_iters']:.1f} | {sp['i_mean_iters']:.1f} |")
        lines.append(f"| Median iterations | {sp['r_median_iters']} | {sp['i_median_iters']} |")
        lines.append("")
        lines.append(f"- **Geometric mean speedup**: {sp['geo_mean_speedup']:.1f}x")
        lines.append(f"- **Median speedup**: {sp['median_speedup']:.1f}x")
        lines.append(f"- POUNCE faster: {sp['r_faster_count']}/{sp['n_problems']} ({100*sp['r_faster_count']/sp['n_problems']:.0f}%)")
        lines.append(f"- POUNCE 10x+ faster: {sp['r_10x_faster']}/{sp['n_problems']}")
        lines.append(f"- Ipopt faster: {sp['i_faster_count']}/{sp['n_problems']}")
        lines.append("")

    # Failure analysis per suite
    lines.append("## Failure Analysis")
    lines.append("")
    for name, comps in suites:
        rf, ifail = failure_analysis(comps)
        if not rf and not ifail:
            continue
        lines.append(f"### {name} Suite")
        lines.append("")
        all_statuses = sorted(set(list(rf.keys()) + list(ifail.keys())))
        lines.append("| Failure Mode | POUNCE | Ipopt |")
        lines.append("|-------------|--------|-------|")
        for st in all_statuses:
            lines.append(f"| {st} | {rf.get(st, 0)} | {ifail.get(st, 0)} |")
        lines.append("")

    # Regressions (Ipopt is Optimal, POUNCE is not)
    regressions = [c for c in all_comps if c['ipopt_solved'] and not c['pounce_solved']]
    if regressions:
        lines.append("## Regressions (Ipopt Optimal, POUNCE not Optimal)")
        lines.append("")
        lines.append("| Problem | Suite | n | m | POUNCE status | Ipopt obj |")
        lines.append("|---------|-------|---|---|--------------|-----------|")
        for c in sorted(regressions, key=lambda c: c['name']):
            io = c['ipopt_obj']
            io_str = f"{io:.6e}" if isinstance(io, (int, float)) and not math.isnan(io) else "N/A"
            lines.append(f"| {c['name']} | {c['suite']} | {c['n']} | {c['m']} | {c['pounce_status']} | {io_str} |")
        lines.append("")

    # Wins (POUNCE is Optimal, Ipopt is not)
    wins = [c for c in all_comps if c['pounce_solved'] and not c['ipopt_solved']]
    if wins:
        lines.append(f"## Wins (POUNCE Optimal, Ipopt not Optimal) — {len(wins)} problems")
        lines.append("")
        lines.append("| Problem | Suite | n | m | Ipopt status | POUNCE obj |")
        lines.append("|---------|-------|---|---|-------------|------------|")
        for c in sorted(wins, key=lambda c: c['name']):
            ro = c['pounce_obj']
            ro_str = f"{ro:.6e}" if isinstance(ro, (int, float)) and not math.isnan(ro) else "N/A"
            lines.append(f"| {c['name']} | {c['suite']} | {c['n']} | {c['m']} | {c['ipopt_status']} | {ro_str} |")
        lines.append("")

    # Different local minima: pounce=Acceptable, Ipopt=Optimal, objective >1% different
    # These are cases where pounce found a valid stationary point (KKT conditions
    # satisfied) but at a worse local minimum than Ipopt. This is inherent to
    # nonconvex optimization — different solver trajectories find different basins.
    diff_minima = [c for c in all_comps
                   if c['pounce_status'] == 'Acceptable'
                   and c['ipopt_status'] == 'Optimal'
                   and not math.isnan(c['obj_diff'])
                   and c['obj_diff'] > 0.01]
    if diff_minima:
        lines.append(f"## Different Local Minima — {len(diff_minima)} problems")
        lines.append("")
        lines.append("pounce converged (Acceptable) but to a different — usually worse — local")
        lines.append("minimum than Ipopt found. Both solvers satisfied first-order KKT conditions")
        lines.append("at their respective solutions. For nonconvex problems this is expected;")
        lines.append("for convex problems it indicates the solver trajectory went astray.")
        lines.append("")
        lines.append("| Problem | Suite | n | m | POUNCE obj | Ipopt obj | Rel. error |")
        lines.append("|---------|-------|---|---|------------|-----------|------------|")
        for c in sorted(diff_minima, key=lambda c: -c['obj_diff']):
            ro = c['pounce_obj']
            io = c['ipopt_obj']
            ro_str = f"{ro:.6e}" if isinstance(ro, (int, float)) and not math.isnan(ro) else "N/A"
            io_str = f"{io:.6e}" if isinstance(io, (int, float)) and not math.isnan(io) else "N/A"
            lines.append(f"| {c['name']} | {c['suite']} | {c['n']} | {c['m']} | {ro_str} | {io_str} | {c['obj_diff']:.1%} |")
        lines.append("")

    # Acceptable breakdown (problems where pounce gets Acceptable, not Optimal)
    acceptable = [c for c in all_comps if c['pounce_status'] == 'Acceptable']
    if acceptable:
        lines.append(f"## Acceptable (not Optimal) — {len(acceptable)} problems")
        lines.append("")
        lines.append("These problems converged within relaxed tolerances but not strict tolerances.")
        lines.append("")
        lines.append("| Problem | Suite | n | m | Ipopt status | POUNCE obj | Ipopt obj |")
        lines.append("|---------|-------|---|---|-------------|------------|-----------|")
        for c in sorted(acceptable, key=lambda c: c['name']):
            ro = c['pounce_obj']
            io = c['ipopt_obj']
            ro_str = f"{ro:.6e}" if isinstance(ro, (int, float)) and not math.isnan(ro) else "N/A"
            io_str = f"{io:.6e}" if isinstance(io, (int, float)) and not math.isnan(io) else "N/A"
            lines.append(f"| {c['name']} | {c['suite']} | {c['n']} | {c['m']} | {c['ipopt_status']} | {ro_str} | {io_str} |")
        lines.append("")

    # Baseline regression detection
    if baseline:
        lines.append("## Regression Detection (vs baseline)")
        lines.append("")
        current_by_name = {c['name']: c for c in all_comps}
        new_failures = []
        new_acceptables = []
        for b in baseline:
            name = b['name']
            if name not in current_by_name:
                continue
            cur = current_by_name[name]
            # Was solved, now fails
            if b['pounce_solved'] and not cur['pounce_solved']:
                new_failures.append((name, b['pounce_status'], cur['pounce_status']))
            # Was Optimal, now Acceptable
            if b['pounce_status'] == 'Optimal' and cur['pounce_status'] == 'Acceptable':
                new_acceptables.append(name)

        if new_failures:
            lines.append(f"### New failures ({len(new_failures)})")
            lines.append("")
            lines.append("| Problem | Was | Now |")
            lines.append("|---------|-----|-----|")
            for name, was, now in new_failures:
                lines.append(f"| {name} | {was} | {now} |")
            lines.append("")

        if new_acceptables:
            lines.append(f"### Degraded to Acceptable ({len(new_acceptables)})")
            lines.append("")
            for name in new_acceptables:
                lines.append(f"- {name}")
            lines.append("")

        if not new_failures and not new_acceptables:
            lines.append("No regressions detected vs baseline.")
            lines.append("")

    # Save machine-readable summary for future regression detection
    summary_data = []
    for c in all_comps:
        summary_data.append({
            'name': c['name'],
            'suite': c['suite'],
            'pounce_status': c['pounce_status'],
            'ipopt_status': c['ipopt_status'],
            'pounce_obj': c['pounce_obj'] if isinstance(c['pounce_obj'], (int, float)) and not math.isnan(c['pounce_obj']) else None,
            'ipopt_obj': c['ipopt_obj'] if isinstance(c['ipopt_obj'], (int, float)) and not math.isnan(c['ipopt_obj']) else None,
            'pounce_solved': c['pounce_solved'],
            'ipopt_solved': c['ipopt_solved'],
        })

    # Per-problem detail tables for POUNCE-only suites that aren't run
    # against Ipopt. These don't appear in the cross-solver Performance
    # section above (no `both_solved` rows), so surface their per-problem
    # timing here so users can still see the whole picture.
    pounce_only_suites = [(name, comps) for name, comps in suites
                          if not any(c['ipopt_solved'] for c in comps)
                          and any(c['pounce_time'] > 0 for c in comps)]
    if pounce_only_suites:
        lines.append("## POUNCE-Only Suite Details")
        lines.append("")
        lines.append("These suites currently run POUNCE only — no Ipopt-side comparison "
                     "is captured in their result files. Per-problem timing and iteration "
                     "counts are shown so users can inspect the whole picture.")
        lines.append("")
        for name, comps in pounce_only_suites:
            lines.append(f"### {name}")
            lines.append("")
            lines.append("| Problem | n | m | Status | Objective | Iters | Time |")
            lines.append("|---------|---|---|--------|-----------|-------|------|")
            for c in sorted(comps, key=lambda c: c['name']):
                obj_str = (f"{c['pounce_obj']:.4e}"
                           if isinstance(c['pounce_obj'], (int, float))
                           and not math.isnan(c['pounce_obj']) else "N/A")
                n_str = f"{c['n']:,}" if c['n'] else "—"
                m_str = f"{c['m']:,}" if c['m'] else "—"
                lines.append(
                    f"| {c['name']} | {n_str} | {m_str} "
                    f"| {c['pounce_status']} | {obj_str} "
                    f"| {c['pounce_iters']} | {fmt_time(c['pounce_time'])} |"
                )
            total = sum(c['pounce_time'] for c in comps)
            solved = sum(1 for c in comps if c['pounce_solved'])
            lines.append("")
            lines.append(f"POUNCE: **{solved}/{len(comps)} Optimal** in {fmt_time(total)} total")
            lines.append("")

    # Status-only suites (water, gas) — included for completeness but
    # the .sol files don't carry timing or iteration counts.
    status_only_suites = [(name, comps) for name, comps in suites
                          if not any(c['ipopt_solved'] for c in comps)
                          and not any(c['pounce_time'] > 0 for c in comps)]
    if status_only_suites:
        lines.append("## Status-Only Suites")
        lines.append("")
        lines.append("These AMPL `.nl` suites are solved one-at-a-time and only the "
                     "POUNCE status is recovered from the `.sol` file header — timing "
                     "and iteration counts are not currently captured in machine-readable form.")
        lines.append("")
        for name, comps in status_only_suites:
            lines.append(f"### {name}")
            lines.append("")
            lines.append("| Problem | Status |")
            lines.append("|---------|--------|")
            for c in sorted(comps, key=lambda c: c['name']):
                lines.append(f"| {c['name']} | {c['pounce_status']} |")
            solved = sum(1 for c in comps if c['pounce_solved'])
            lines.append("")
            lines.append(f"POUNCE: **{solved}/{len(comps)} Optimal**")
            lines.append("")

    # Head-to-head: dedicated convex solver vs general NLP. Rendered as a
    # standalone section; intentionally not folded into all_comps / the
    # baseline / the Ipopt profiles (a different comparison axis).
    lines.extend(head_to_head_lines(head_to_head or []))

    lines.append("---")
    lines.append("*Generated by benchmark_report.py*")

    report = '\n'.join(lines)

    with open(output_path, 'w') as f:
        f.write(report)

    # Save baseline JSON for future regression detection
    baseline_path = output_path.replace('.md', '.json')
    with open(baseline_path, 'w') as f:
        json.dump(summary_data, f, indent=2)

    return combined, summary_data


# ---- Main ----

def main():
    output_path = os.path.join(SCRIPT_DIR, 'BENCHMARK_REPORT.md')
    baseline_path = None

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == '--output' and i + 1 < len(args):
            output_path = args[i + 1]
            i += 2
        elif args[i] == '--baseline' and i + 1 < len(args):
            baseline_path = args[i + 1]
            i += 2
        else:
            i += 1

    # Load baseline if provided
    baseline = None
    if baseline_path and os.path.exists(baseline_path):
        with open(baseline_path) as f:
            baseline = json.load(f)
        print(f"Loaded baseline: {baseline_path} ({len(baseline)} problems)")

    # Load all suites
    suites = []

    # Every suite is .nl-driven. load_suite() merges the per-release
    # pounce.json with the committed ipopt_ma57.json reference, both in the
    # canonical {solver,name,n,m,status,objective,iterations,solve_time}
    # shape. Vanderbei (the AMPL transliteration of CUTE) replaces the
    # retired compiled CUTEst suite; large_scale is now generated as .nl by
    # benchmarks/large_scale/generate_nl.py rather than a Rust harness.
    missing_reference = []
    unreferenced = []  # rows a referenced suite has no reference record for
    profile_dirs = []  # dirnames with both pounce + ipopt, for the profiles
    env_stamps = {}    # suite -> thread stamp, for the threading note
    for suite_name, dirname, make_target in (
        ('Vanderbei',   'vanderbei',   'vanderbei-run'),
        ('Electrolyte', 'electrolyte', 'electrolyte-run'),
        ('Grid',        'grid',        'grid-run'),
        ('CHO',         'cho',         'cho-run'),
        ('Water',       'water',       'water-run'),
        ('Gas',         'gas',         'gas-run'),
        ('LargeScale',  'large_scale', 'large-scale'),
        ('Mittelmann',  'mittelmann',  'mittelmann-run'),
        ('QP',          'qp',          'qp-run'),
        ('LP',          'lp',          'lp-run'),
        ('LPopt',       'lpopt',       'lpopt-run'),
    ):
        suite, has_pounce, has_ipopt = load_suite(suite_name, dirname)
        # Render a suite only when the POUNCE arm actually ran. A real run
        # always writes records (even for problems it fails), so an empty
        # pounce arm means the suite was skipped (e.g. lpopt). Without this
        # guard a suite with only the committed ipopt-ma57 reference would
        # render as "pounce 0/N solved" — a spurious total regression.
        if suite and has_pounce and has_ipopt:
            unref = [c for c in suite if c.get('ref_missing')]
            if unref:
                unreferenced.extend(unref)
                suite = [c for c in suite if not c.get('ref_missing')]
                print(f"{suite_name} suite: {len(unref)} problem(s) have no ipopt "
                      f"reference record; excluded from the comparison and listed separately")
        if suite and has_pounce:
            suites.append((suite_name, suite))
            # Only for suites that actually render — a stamp for a skipped
            # suite would describe runs the report does not show.
            env_stamps[suite_name] = read_env_stamp(dirname)
            ref = 'pounce + ipopt-ma57 reference' if has_ipopt else 'POUNCE-only (no ipopt reference)'
            print(f"{suite_name} suite: {len(suite)} records loaded — {ref}")
            if has_pounce and has_ipopt:
                profile_dirs.append(dirname)
            if has_pounce and not has_ipopt:
                missing_reference.append((suite_name, dirname))
        else:
            print(f"{suite_name} suite: skipped or no pounce results "
                  f"(run `make -C benchmarks {make_target}` to include it)")

    if missing_reference:
        print()
        print("NOTE: no saved ipopt-ma57 reference for: "
              + ", ".join(n for n, _ in missing_reference) + ".")
        print("      These suites are reported POUNCE-only. Generate the "
              "reference once with `make -C benchmarks ipopt-reference` "
              "(or per suite, `ipopt-ref-<suite>`) and commit it.")

    # GAMS nlpbench is no longer aggregated as a benchmark suite. Its
    # problem coverage duplicates the .nl suites (princetonlib ≈ vanderbei,
    # GAMS mittelmann ≈ ampl-nlp mittelmann/, powerflow ≈ grid/) and it was
    # compared on the same pounce-vs-ipopt axis as everything else. The
    # GAMS solver-link path is now exercised only as a liveness smoke check
    # via `make -C benchmarks gams-bench` (gams/nlpbench `bench-smoke`),
    # which does not feed this report.

    # Head-to-head suites: the dedicated convex solver (convex.json) vs the
    # general NLP path (nlp.json) on the same .nl problems. Loaded separately
    # so they stay out of the Ipopt-reference machinery (profiles, baseline,
    # regressions/wins, executive summary).
    head_to_head = []
    for suite_name, dirname, files, keys in (
            ('LP — convex vs NLP', 'lp_convex',
             ('convex.json', 'nlp.json'), ('convex', 'nlp')),
            ('QP — convex vs NLP', 'qp_convex',
             ('convex.json', 'nlp.json'), ('convex', 'nlp')),
            # Exact vs limited-memory Hessian on the same problems. The
            # Python frontend and the CasADi plugin pick L-BFGS on their
            # own when no exact Lagrangian Hessian exists, so this arm is
            # a default path for a large share of users, not an opt-in.
            ('Mittelmann — exact vs L-BFGS', 'lbfgs',
             ('exact.json', 'limited.json'), ('exact', 'lbfgs')),
    ):
        comps, has_convex, has_nlp = load_suite(
            suite_name, dirname,
            left_file=files[0], right_file=files[1],
            left_key=keys[0], right_key=keys[1])
        if comps:
            head_to_head.append((suite_name, comps))
            print(f"{suite_name} suite: {len(comps)} records loaded — "
                  f"convex-vs-nlp head-to-head")
        else:
            print(f"{suite_name} suite: no results "
                  f"(run `make -C benchmarks {dirname.replace('_', '-')}-run` first)")

    if not suites:
        print("No benchmark results found. Run `make benchmark` first.")
        sys.exit(1)

    combined, _summary = generate_report(suites, output_path, baseline,
                                          profile_dirs=profile_dirs,
                                          head_to_head=head_to_head,
                                          env_stamps=env_stamps,
                                          unreferenced=unreferenced)

    print(f"\nReport written to {output_path}")
    print(f"Baseline saved to {output_path.replace('.md', '.json')}")
    print(f"\nCombined summary:")
    print(f"  Total: {combined['total']}")
    print(f"  POUNCE solved: {combined['r_solved']}/{combined['total']} "
          f"(Optimal: {combined['r_optimal']}, Acceptable: {combined['r_acceptable']})")
    print(f"  Ipopt solved:  {combined['i_solved']}/{combined['total']} "
          f"(Optimal: {combined['i_optimal']}, Acceptable: {combined['i_acceptable']})")
    print(f"  POUNCE only:   {combined['r_only']}")
    print(f"  Ipopt only:    {combined['i_only']}")


if __name__ == '__main__':
    main()
