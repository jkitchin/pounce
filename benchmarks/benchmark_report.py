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


def _build_comparisons(records, suite_name):
    """Build the canonical comparison list from a flat list of
    {solver,name,n,m,status,objective,iterations,solve_time} records
    (any mix of pounce and ipopt rows)."""
    pounce_by_name = {}
    ipopt_by_name = {}
    for r in records:
        if r['solver'] == 'pounce':
            pounce_by_name[r['name']] = r
        elif r['solver'] == 'ipopt':
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
        })

    return comparisons


def load_suite(suite_name, dirname):
    """Load one .nl suite by merging its per-release pounce run with the
    saved ipopt-ma57 reference.

    Reads benchmarks/<dirname>/pounce.json (regenerated every release) and
    benchmarks/<dirname>/ipopt_ma57.json (committed reference, run rarely).
    Returns (comparisons, has_pounce, has_ipopt); comparisons is None when
    neither file is present.
    """
    base = os.path.join(SCRIPT_DIR, dirname)
    pounce = _read_records(os.path.join(base, 'pounce.json'))
    ipopt = _read_records(os.path.join(base, 'ipopt_ma57.json'))
    if not pounce and not ipopt:
        return None, False, False
    comps = _build_comparisons(pounce + ipopt, suite_name)
    return (comps if comps else None), bool(pounce), bool(ipopt)


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
    prov_path = os.path.join(SCRIPT_DIR, 'ipopt_ma57.provenance.json')
    if os.path.exists(prov_path):
        try:
            with open(prov_path) as f:
                ref = json.load(f)
            ipopt_version = ref.get('ipopt_version', 'unknown')
            ipopt_linear_solver = ref.get('linear_solver', ipopt_linear_solver)
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


def generate_report(suites, output_path, baseline=None):
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

    # Combined summary
    all_comps = []
    for name, comps in suites:
        all_comps.extend(comps)

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
        if suite:
            suites.append((suite_name, suite))
            ref = 'pounce + ipopt-ma57 reference' if has_ipopt else 'POUNCE-only (no ipopt reference)'
            print(f"{suite_name} suite: {len(suite)} records loaded — {ref}")
            if has_pounce and not has_ipopt:
                missing_reference.append((suite_name, dirname))
        else:
            print(f"{suite_name} suite: no results "
                  f"(run `make -C benchmarks {make_target}` first)")

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

    if not suites:
        print("No benchmark results found. Run `make benchmark` first.")
        sys.exit(1)

    combined, _summary = generate_report(suites, output_path, baseline)

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
