# adversary/

Working directory for the `/adversary` agent — automated correctness testing
of pounce's solver families against independent oracles.

**The agent never modifies pounce source.** Everything it produces lands here.

## Layout
- `log.org` — the running problem index + per-family coverage counts.
- `runs/` — one `*.py` cross-check script + one `*.org` report per problem,
  named `YYYY-MM-DD_<family>_<name>.{py,org}`.

## Running it
From the repo root, in Claude Code:

```
/adversary                       # 1 problem, least-tested family
/adversary 5                     # 5 problems, balanced across families
/adversary socp                  # 1 second-order-cone problem
/adversary 3 exp geometric programming
```

Families: `nlp`, `lp`, `qp`, `qp-active-set`, `socp`, `exp`, `power`, `sdp`,
`sos`, `autoroute`, `batch`, `diff`, `sensitivity`.

## Oracles
- `scipy`, `numpy`, `sympy`, `jax`, `torch` — preinstalled in `.venv-qa`.
- `cvxpy` (convex/conic gold standard) and `pyomo` (Ipopt-vs-pounce NLP path)
  are installed on demand into `.venv-qa` by the agent.
- `ipopt` binary at `/opt/homebrew/bin/ipopt`.
- `pounce verify <problem.nl> <claim.sol>` — solver-independent feasibility/KKT
  oracle (does not trust the solver that produced the `.sol`).

The full procedure lives in `.claude/commands/adversary.md`.
