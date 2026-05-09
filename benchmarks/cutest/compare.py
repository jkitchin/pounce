#!/usr/bin/env python3
"""
Compare ripopt and Ipopt results on CUTEst test problems.
Reads a single JSON file with solver field per entry.

Usage:
    python compare.py results.json
    python compare.py results.json --output report.md
"""

import json
import math
import os
import sys
from collections import defaultdict


def is_solved(status):
    return status in ('Optimal', 'Acceptable')


def obj_diff(ro, co):
    """Relative objective difference, using max(|r|, |c|, 1) as denominator.

    The 1.0 floor in the denominator prevents near-zero objectives (e.g.,
    ripopt=+1e-12, ipopt=-1e-9) from producing artificially large relative
    differences.  With this floor, two objectives that differ by < 1e-4 in
    absolute terms will always be classified as matching.
    """
    if ro is None or co is None:
        return float('nan')
    if math.isnan(ro) or math.isnan(co):
        return float('nan')
    denom = max(abs(co), abs(ro), 1.0)
    return abs(ro - co) / denom


def fmt_time(t):
    if t >= 1.0:
        return f"{t:.2f}s"
    elif t >= 0.001:
        return f"{t*1000:.1f}ms"
    else:
        return f"{t*1e6:.0f}us"


def classify_problem(n, m):
    if m == 0:
        return "unconstrained"
    else:
        return "constrained"


def main():
    if len(sys.argv) < 2:
        print("Usage: python compare.py results.json [--output report.md]")
        sys.exit(1)

    results_path = sys.argv[1]
    output_path = None
    if '--output' in sys.argv:
        idx = sys.argv.index('--output')
        output_path = sys.argv[idx + 1]
    else:
        output_path = os.path.join(
            os.path.dirname(results_path),
            'CUTEST_REPORT.md'
        )

    with open(results_path) as f:
        data = json.load(f)

    # Split by solver
    ripopt_by_name = {}
    ipopt_by_name = {}
    for r in data:
        if r['solver'] == 'ripopt':
            ripopt_by_name[r['name']] = r
        elif r['solver'] == 'ipopt':
            ipopt_by_name[r['name']] = r

    all_names = sorted(set(ripopt_by_name.keys()) | set(ipopt_by_name.keys()))

    # Comparisons
    comparisons = []
    for name in all_names:
        rr = ripopt_by_name.get(name, {})
        cr = ipopt_by_name.get(name, {})

        r_solved = is_solved(rr.get('status', ''))
        c_solved = is_solved(cr.get('status', ''))
        both_solved = r_solved and c_solved

        od = obj_diff(rr.get('objective'), cr.get('objective')) if both_solved else float('nan')
        passed = both_solved and not math.isnan(od) and od < 1e-4

        n = rr.get('n', cr.get('n', 0))
        m = rr.get('m', cr.get('m', 0))

        comparisons.append({
            'name': name,
            'n': n,
            'm': m,
            'ripopt_status': rr.get('status', 'N/A'),
            'ipopt_status': cr.get('status', 'N/A'),
            'ripopt_obj': rr.get('objective', float('nan')),
            'ipopt_obj': cr.get('objective', float('nan')),
            'obj_diff': od,
            'ripopt_iters': rr.get('iterations', 0),
            'ipopt_iters': cr.get('iterations', 0),
            'ripopt_time': rr.get('solve_time', 0),
            'ipopt_time': cr.get('solve_time', 0),
            'ripopt_cv': rr.get('constraint_violation', 0),
            'ipopt_cv': cr.get('constraint_violation', 0),
            'passed': passed,
            'ripopt_solved': r_solved,
            'ipopt_solved': c_solved,
            'both_solved': both_solved,
            'category': classify_problem(n, m),
        })

    # Statistics
    total = len(comparisons)
    ripopt_solved = sum(1 for c in comparisons if c['ripopt_solved'])
    ipopt_solved = sum(1 for c in comparisons if c['ipopt_solved'])
    both_solved = sum(1 for c in comparisons if c['both_solved'])
    passed = sum(1 for c in comparisons if c['passed'])

    all_diffs = [c['obj_diff'] for c in comparisons
                 if c['both_solved'] and not math.isnan(c['obj_diff'])]
    pass_diffs = [c['obj_diff'] for c in comparisons if c['passed']]
    mismatch_diffs = [c['obj_diff'] for c in comparisons
                      if c['both_solved'] and not c['passed'] and not math.isnan(c['obj_diff'])]

    def compute_stats(diffs):
        if not diffs:
            return float('nan'), float('nan'), float('nan')
        mean = sum(diffs) / len(diffs)
        mx = max(diffs)
        median = sorted(diffs)[len(diffs) // 2]
        return mean, median, mx

    all_mean, all_median, all_max = compute_stats(all_diffs)
    pass_mean, pass_median, pass_max = compute_stats(pass_diffs)

    # Category breakdown
    categories = defaultdict(lambda: {'total': 0, 'ripopt': 0, 'ipopt': 0, 'both': 0, 'passed': 0})
    for c in comparisons:
        cat = c['category']
        categories[cat]['total'] += 1
        if c['ripopt_solved']:
            categories[cat]['ripopt'] += 1
        if c['ipopt_solved']:
            categories[cat]['ipopt'] += 1
        if c['both_solved']:
            categories[cat]['both'] += 1
        if c['passed']:
            categories[cat]['passed'] += 1

    # Generate report
    lines = []
    lines.append("# CUTEst Benchmark Report")
    lines.append("")
    lines.append("Comparison of ripopt vs Ipopt (C++) on the CUTEst test set.")
    lines.append("")
    lines.append("## Executive Summary")
    lines.append("")
    lines.append(f"- **Total problems**: {total}")
    lines.append(f"- **ripopt solved**: {ripopt_solved}/{total} ({100*ripopt_solved/max(total,1):.1f}%)")
    lines.append(f"- **Ipopt solved**: {ipopt_solved}/{total} ({100*ipopt_solved/max(total,1):.1f}%)")
    lines.append(f"- **Both solved**: {both_solved}/{total}")
    lines.append(f"- **Matching solutions** (rel obj diff < 1e-4): {passed}/{max(both_solved,1)}")
    lines.append("")

    lines.append("## Accuracy Statistics (where both solve)")
    lines.append("")
    lines.append(f"Relative difference = |r_obj - i_obj| / max(|r_obj|, |i_obj|, 1.0).  ")
    lines.append(f"The 1.0 floor prevents near-zero objectives from inflating the metric.")
    lines.append("")
    lines.append(f"**Matching solutions** ({len(pass_diffs)} problems, rel diff < 1e-4):")
    lines.append("")
    lines.append("| Metric | Rel Diff |")
    lines.append("|--------|----------|")
    lines.append(f"| Mean   | {pass_mean:.2e} |")
    lines.append(f"| Median | {pass_median:.2e} |")
    lines.append(f"| Max    | {pass_max:.2e} |")
    lines.append("")
    lines.append(f"**All both-solved** ({len(all_diffs)} problems, including {len(mismatch_diffs)} mismatches):")
    lines.append("")
    lines.append("| Metric | Rel Diff |")
    lines.append("|--------|----------|")
    lines.append(f"| Mean   | {all_mean:.2e} |")
    lines.append(f"| Median | {all_median:.2e} |")
    lines.append(f"| Max    | {all_max:.2e} |")
    lines.append("")

    lines.append("## Category Breakdown")
    lines.append("")
    lines.append("| Category | Total | ripopt | Ipopt | Both | Match |")
    lines.append("|----------|-------|--------|-------|------|-------|")
    for cat in sorted(categories.keys()):
        d = categories[cat]
        lines.append(f"| {cat} | {d['total']} | {d['ripopt']} | {d['ipopt']} | {d['both']} | {d['passed']} |")
    lines.append("")

    lines.append("## Detailed Results")
    lines.append("")
    lines.append("| Problem | n | m | ripopt | Ipopt | Obj Diff | r_iter | i_iter | r_time | i_time | Speedup | Status |")
    lines.append("|---------|---|---|--------|-------|----------|--------|--------|--------|--------|---------|--------|")
    for c in comparisons:
        od = f"{c['obj_diff']:.2e}" if not math.isnan(c['obj_diff']) else "N/A"
        rt_str = fmt_time(c['ripopt_time']) if c['ripopt_time'] > 0 else "N/A"
        it_str = fmt_time(c['ipopt_time']) if c['ipopt_time'] > 0 else "N/A"
        if c['ripopt_time'] > 0 and c['ipopt_time'] > 0:
            sp_str = f"{c['ipopt_time']/c['ripopt_time']:.1f}x"
        else:
            sp_str = "N/A"
        if c['passed']:
            status = "PASS"
        elif not c['ripopt_solved'] and not c['ipopt_solved']:
            status = "BOTH_FAIL"
        elif not c['ripopt_solved']:
            status = "ripopt_FAIL"
        elif not c['ipopt_solved']:
            status = "ipopt_FAIL"
        else:
            status = "MISMATCH"
        lines.append(f"| {c['name']} | {c['n']} | {c['m']} | {c['ripopt_status'][:12]} | {c['ipopt_status'][:12]} | {od} | {c['ripopt_iters']} | {c['ipopt_iters']} | {rt_str} | {it_str} | {sp_str} | {status} |")
    lines.append("")

    # Performance comparison
    both_data = [c for c in comparisons if c['both_solved']]
    if both_data:
        lines.append("## Performance Comparison (where both solve)")
        lines.append("")

        r_iters = [c['ripopt_iters'] for c in both_data]
        i_iters = [c['ipopt_iters'] for c in both_data]
        r_times = [c['ripopt_time'] for c in both_data if c['ripopt_time'] > 0]
        i_times = [c['ipopt_time'] for c in both_data if c['ipopt_time'] > 0]

        lines.append("### Iteration Comparison")
        lines.append("")
        lines.append("| Metric | ripopt | Ipopt |")
        lines.append("|--------|--------|-------|")
        lines.append(f"| Mean   | {sum(r_iters)/len(r_iters):.1f} | {sum(i_iters)/len(i_iters):.1f} |")
        lines.append(f"| Median | {sorted(r_iters)[len(r_iters)//2]} | {sorted(i_iters)[len(i_iters)//2]} |")
        lines.append(f"| Total  | {sum(r_iters)} | {sum(i_iters)} |")
        lines.append("")

        r_fewer = sum(1 for r, i in zip(r_iters, i_iters) if r < i)
        i_fewer = sum(1 for r, i in zip(r_iters, i_iters) if i < r)
        tied = sum(1 for r, i in zip(r_iters, i_iters) if r == i)
        lines.append(f"- ripopt fewer iterations: {r_fewer}/{len(r_iters)}")
        lines.append(f"- Ipopt fewer iterations: {i_fewer}/{len(i_iters)}")
        lines.append(f"- Tied: {tied}/{len(r_iters)}")
        lines.append("")

        if r_times and i_times:
            lines.append("### Timing Comparison")
            lines.append("")
            r_total = sum(r_times)
            i_total = sum(i_times)
            lines.append("| Metric | ripopt | Ipopt |")
            lines.append("|--------|--------|-------|")
            lines.append(f"| Mean   | {fmt_time(r_total/len(r_times))} | {fmt_time(i_total/len(i_times))} |")
            lines.append(f"| Median | {fmt_time(sorted(r_times)[len(r_times)//2])} | {fmt_time(sorted(i_times)[len(i_times)//2])} |")
            lines.append(f"| Total  | {fmt_time(r_total)} | {fmt_time(i_total)} |")
            lines.append("")

            speedups = [c['ipopt_time'] / c['ripopt_time']
                        for c in both_data
                        if c['ripopt_time'] > 0 and c['ipopt_time'] > 0]
            if speedups:
                geo_mean = math.exp(sum(math.log(s) for s in speedups) / len(speedups))
                r_faster = sum(1 for s in speedups if s > 1.0)
                i_faster = sum(1 for s in speedups if s < 1.0)
                lines.append(f"- Geometric mean speedup (Ipopt_time/ripopt_time): **{geo_mean:.2f}x**")
                lines.append(f"  - \\>1 means ripopt is faster, <1 means Ipopt is faster")
                lines.append(f"- ripopt faster: {r_faster}/{len(speedups)} problems")
                lines.append(f"- Ipopt faster: {i_faster}/{len(speedups)} problems")
                lines.append(f"- Overall speedup (total time): {i_total/r_total:.2f}x")
                lines.append("")

    # Failure analysis
    ripopt_only_fails = [c for c in comparisons if not c['ripopt_solved'] and c['ipopt_solved']]
    ipopt_only_fails = [c for c in comparisons if c['ripopt_solved'] and not c['ipopt_solved']]
    both_fail = [c for c in comparisons if not c['ripopt_solved'] and not c['ipopt_solved']]

    if ripopt_only_fails or ipopt_only_fails or both_fail:
        lines.append("## Failure Analysis")
        lines.append("")

    if ripopt_only_fails:
        lines.append(f"### Problems where only ripopt fails ({len(ripopt_only_fails)})")
        lines.append("")
        lines.append("| Problem | n | m | ripopt status | Ipopt obj |")
        lines.append("|---------|---|---|---------------|-----------|")
        for c in ripopt_only_fails:
            lines.append(f"| {c['name']} | {c['n']} | {c['m']} | {c['ripopt_status']} | {c['ipopt_obj']:.6e} |")
        lines.append("")

    if ipopt_only_fails:
        lines.append(f"### Problems where only Ipopt fails ({len(ipopt_only_fails)})")
        lines.append("")
        lines.append("| Problem | n | m | Ipopt status | ripopt obj |")
        lines.append("|---------|---|---|--------------|------------|")
        for c in ipopt_only_fails:
            ro = c['ripopt_obj']
            ro_str = f"{ro:.6e}" if not math.isnan(ro) else "N/A"
            lines.append(f"| {c['name']} | {c['n']} | {c['m']} | {c['ipopt_status']} | {ro_str} |")
        lines.append("")

    if both_fail:
        lines.append(f"### Problems where both fail ({len(both_fail)})")
        lines.append("")
        lines.append("| Problem | n | m | ripopt status | Ipopt status |")
        lines.append("|---------|---|---|---------------|--------------|")
        for c in both_fail:
            lines.append(f"| {c['name']} | {c['n']} | {c['m']} | {c['ripopt_status']} | {c['ipopt_status']} |")
        lines.append("")

    # Mismatches — categorize by cause
    mismatches = [c for c in comparisons if c['both_solved'] and not c['passed']]
    if mismatches:
        # Classify mismatches
        diff_local_min = []  # Both Optimal, different basins
        convergence_gap = []  # One Acceptable, didn't fully converge
        for c in mismatches:
            r_opt = c['ripopt_status'] == 'Optimal'
            i_opt = c['ipopt_status'] == 'Optimal'
            if r_opt and i_opt:
                diff_local_min.append(c)
            else:
                convergence_gap.append(c)

        r_better = sum(1 for c in mismatches if c['ripopt_obj'] < c['ipopt_obj'])
        i_better = sum(1 for c in mismatches if c['ipopt_obj'] < c['ripopt_obj'])

        lines.append(f"### Objective mismatches ({len(mismatches)})")
        lines.append("")
        lines.append(f"Both solvers converged but found different objective values (rel diff > 1e-4).")
        lines.append("")
        lines.append(f"- **Different local minimum** (both Optimal): {len(diff_local_min)}")
        lines.append(f"- **Convergence gap** (one Acceptable): {len(convergence_gap)}")
        lines.append(f"- **Better objective found by**: ripopt {r_better}, Ipopt {i_better}")
        lines.append("")
        lines.append("| Problem | ripopt obj | Ipopt obj | Rel Diff | r_status | i_status | Better |")
        lines.append("|---------|-----------|-----------|----------|----------|----------|--------|")
        for c in sorted(mismatches, key=lambda c: -c['obj_diff']):
            better = "ripopt" if c['ripopt_obj'] < c['ipopt_obj'] else "ipopt"
            lines.append(
                f"| {c['name']} | {c['ripopt_obj']:.6e} | {c['ipopt_obj']:.6e} "
                f"| {c['obj_diff']:.2e} | {c['ripopt_status'][:10]} | {c['ipopt_status'][:10]} | {better} |"
            )
        lines.append("")

    lines.append("---")
    lines.append("*Generated by benchmarks/cutest/compare.py*")

    report = '\n'.join(lines)

    with open(output_path, 'w') as f:
        f.write(report)

    print(f"Report written to {output_path}")
    print(f"\nSummary:")
    print(f"  Total: {total}")
    print(f"  ripopt solved: {ripopt_solved}/{total}")
    print(f"  Ipopt solved: {ipopt_solved}/{total}")
    print(f"  Both solved: {both_solved}/{total}")
    print(f"  Matching (rel diff < 1e-4): {passed}/{max(both_solved, 1)}")


if __name__ == '__main__':
    main()
