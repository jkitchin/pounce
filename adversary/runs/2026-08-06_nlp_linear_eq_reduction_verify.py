"""Adversary cross-check: linear-equality variable elimination, end to end.

Family: nlp   Class: equality-constrained, linear-equality structure
Target: `presolve_linear_eq_reduction` (Phase 6, gh#487)
Source: no published instance — a randomized family built around a KNOWN
        feasible point, with `pounce verify` as the independent oracle.

The Rust probe (`adversary-fuzz linelim`) attacks the plan and the derivative
transforms in isolation. This one attacks the whole stack the way a user meets
it: a `.nl` file on disk, the CLI, and the `.sol` file that comes back.

Two oracles, neither of which is "pounce with the option off":

1. **`pounce verify <nl> <sol>`** — an independent feasibility/KKT check of the
   claimed point against the ORIGINAL `.nl`, which knows nothing about the
   reduction. If the reduced solve returns a point that violates a row it
   eliminated, or that is not stationary for the original model, this rejects
   it.
2. **Direct evaluation in this script** — the model's rows are re-evaluated
   here, in Python, from the same data used to write the `.nl`. That catches a
   `.sol` whose primal block is the right length but the wrong permutation,
   which `verify` alone would not distinguish from a genuine solve.

The objective is a sum of quartics, so every column is nonlinear in the
objective and the `.nl` variable ordering is the natural one. Every row is a
linear equality whose right-hand side is computed from a random `x*`, so the
system is consistent by construction and a "proved infeasible" verdict is
always a bug.
"""

import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

POUNCE = Path("/home/user/pounce/target/release/pounce")


def write_nl(path, n, rows, rhs, consts, targets, x_l, x_u, x0):
    """`min Σ (x_j - t_j)^4  s.t.  Σ a_ij x_j + c_i = rhs_i`, all rows linear.

    `c_i` goes in the row's expression segment rather than being folded into
    the bound, because a row constant is exactly what the elimination has to
    read off its probe point — and a generator that always emits `c_i = 0`
    cannot tell whether it did.
    """
    m = len(rows)
    jnnz = sum(len(r) for r in rows)
    L = [
        "g3 1 1 0\t# adversary linear-eq reduction",
        f" {n} {m} 1 0 {m} \t# vars, constraints, objectives, ranges, eqns",
        " 0 1 0 0 0 0\t# nonlinear constrs, objs",
        " 0 0\t# network",
        f" 0 {n} 0 \t# nonlinear vars in constraints, objectives, both",
        " 0 0 0 1\t# linear network vars; functions; arith, flags",
        " 0 0 0 0 0 \t# discrete",
        f" {jnnz} {n} \t# nonzeros in Jacobian, obj gradient",
        " 0 0\t# max name lengths",
        " 0 0 0 0 0\t# common exprs",
    ]
    for r in range(m):
        L += [f"C{r}", f"n{consts[r]!r}"]
    L += ["O0 0", "o54", str(n)]
    for j in range(n):
        L += ["o5", "o0", f"v{j}", f"n{-targets[j]!r}", "n4"]
    L.append(f"x{n}")
    for j in range(n):
        L.append(f"{j} {x0[j]!r}")
    L.append("r")
    for i in range(m):
        L.append(f"4 {rhs[i]!r}")
    L.append("b")
    for j in range(n):
        L.append(f"0 {x_l[j]!r} {x_u[j]!r}")
    # cumulative Jacobian nonzeros per column, columns 0..n-2
    colcount = [0] * n
    for r in rows:
        for j, _ in r:
            colcount[j] += 1
    L.append(f"k{n - 1}")
    cum = 0
    for j in range(n - 1):
        cum += colcount[j]
        L.append(str(cum))
    for i, r in enumerate(rows):
        L.append(f"J{i} {len(r)}")
        for j, a in sorted(r):
            L.append(f"{j} {a!r}")
    L.append(f"G0 {n}")
    for j in range(n):
        L.append(f"{j} 0")
    path.write_text("\n".join(L) + "\n")


def gen(rng):
    n = rng.randint(4, 9)
    m = rng.randint(1, max(1, n - 2))
    x_star = [round(rng.uniform(-4, 4), 6) for _ in range(n)]
    rows, rhs, consts = [], [], []
    for _ in range(m):
        arity = rng.choice([1, 2, 2, 2, 3])
        cols = rng.sample(range(n), min(arity, n))
        row = []
        for c in cols:
            a = round(rng.uniform(0.3, 3.0), 6) * rng.choice([1, -1])
            row.append((c, a))
        rows.append(row)
        # A nonzero constant on most rows; a few zeros to keep both paths live.
        consts.append(0.0 if rng.random() < 0.7 else round(rng.uniform(-3, 3), 6))
        rhs.append(round(sum(a * x_star[c] for c, a in row) + consts[-1], 9))
    targets = [round(rng.uniform(-4, 4), 6) for _ in range(n)]
    x_l = [round(x_star[j] - rng.uniform(1.0, 6.0), 6) for j in range(n)]
    x_u = [round(x_star[j] + rng.uniform(1.0, 6.0), 6) for j in range(n)]
    x0 = [round(x_star[j] + rng.uniform(-0.5, 0.5), 6) for j in range(n)]
    return n, rows, rhs, consts, targets, x_l, x_u, x0, x_star


def run(nl_dir, opts):
    cmd = [str(POUNCE), "m", "-AMPL"] + opts
    p = subprocess.run(cmd, cwd=nl_dir, capture_output=True, text=True, timeout=60)
    return p


def read_sol(path, n, m):
    nums = []
    for line in path.read_text().splitlines():
        if line.startswith("objno"):
            break
        try:
            nums.append(float(line.strip()))
        except ValueError:
            pass
    tail = nums[-(n + m):]
    return tail[:m], tail[m:]


def main():
    trials = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 20260806
    rng = random.Random(seed)

    bad, checked, reduced_any = [], 0, 0
    for t in range(trials):
        n, rows, rhs, consts, targets, x_l, x_u, x0, x_star = gen(rng)
        m = len(rows)
        d = Path(tempfile.mkdtemp(prefix="adv_lineq_"))
        try:
            nl = d / "m.nl"
            write_nl(nl, n, rows, rhs, consts, targets, x_l, x_u, x0)

            off = run(d, [])
            if "Optimal Solution Found" not in off.stdout:
                continue  # the baseline could not solve it; not this pass's business
            sol_off = read_sol(d / "m.sol", n, m)
            shutil.copy(d / "m.sol", d / "off.sol")

            on = run(d, ["presolve=yes", "presolve_linear_eq_reduction=yes"])
            checked += 1
            if "eliminated 0 columns" not in on.stdout:
                reduced_any += 1
            if "Optimal Solution Found" not in on.stdout:
                bad.append((t, "reduced solve did not reach optimality", on.stdout[-400:]))
                continue
            lam_on, x_on = read_sol(d / "m.sol", n, m)

            # --- oracle 1: independent KKT / feasibility check vs the .nl ---
            v = subprocess.run(
                [str(POUNCE), "verify", "m.nl", "m.sol"],
                cwd=d, capture_output=True, text=True, timeout=60,
            )
            if v.returncode != 0:
                bad.append((t, f"pounce verify rejected the reduced solve (rc={v.returncode})",
                            v.stdout[-600:] + v.stderr[-300:]))
                continue

            # --- oracle 2: re-evaluate the rows here, from the source data ---
            if len(x_on) != n or len(lam_on) != m:
                bad.append((t, f".sol shape wrong: {len(x_on)} primals / {len(lam_on)} duals "
                               f"for an {n}x{m} model", ""))
                continue
            for i, row in enumerate(rows):
                body = sum(a * x_on[c] for c, a in row) + consts[i]
                if abs(body - rhs[i]) > 1e-6 * max(1.0, abs(rhs[i])):
                    bad.append((t, f"row {i} violated by {body - rhs[i]:.3e} in the reduced "
                                   f"solve's reported point", ""))
                    break
            else:
                for j in range(n):
                    if not (x_l[j] - 1e-6 <= x_on[j] <= x_u[j] + 1e-6):
                        bad.append((t, f"x[{j}]={x_on[j]} outside its declared box "
                                       f"[{x_l[j]}, {x_u[j]}]", ""))
                        break
                else:
                    f_on = sum((x_on[j] - targets[j]) ** 4 for j in range(n))
                    f_off = sum((sol_off[1][j] - targets[j]) ** 4 for j in range(n))
                    if abs(f_on - f_off) > 1e-4 * max(1.0, abs(f_off)):
                        bad.append((t, f"objective disagrees with the un-reduced solve: "
                                       f"{f_on:.8e} vs {f_off:.8e}", ""))
        finally:
            shutil.rmtree(d, ignore_errors=True)

    print(f"instances solved and checked : {checked}")
    print(f"  where the pass reduced     : {reduced_any}")
    print(f"failures                     : {len(bad)}")
    for t, why, extra in bad[:8]:
        print(f"  [{t}] {why}")
        if extra:
            print("      " + extra.replace("\n", "\n      ")[:500])
    print("VERDICT: PASS" if not bad else f"VERDICT: FAIL ({len(bad)} failures)")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
