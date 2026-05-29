# v0.3.0 publish-order checklist

Working list for the v0.3.0 release. Drop this file once the tag is up
and the wheels are out — it's a working artefact, not a permanent doc.

## 0. Pre-flight (one-time, before the first `cargo publish`)

- [ ] `cargo check --workspace` clean.
- [ ] `cargo test --workspace` passes (or the known-skipped list is
      explicitly known-skipped).
- [ ] `cd python && python -m pytest tests/ -q` passes (all 36 incl.
      the pounce#73 fix and the pounce#74 warm-start + parallel
      `vmap_solve_parallel` regression tests).
- [ ] `git status` clean apart from the bump + changelog commit.
- [ ] Logged in to crates.io (`cargo login <TOKEN>`).
- [ ] Logged in to PyPI (`~/.pypirc` or `UV_PUBLISH_TOKEN`).
- [ ] Working tree on `main`, fast-forwarded from origin.

### 0a. Headline features for 0.3.0 (spot-check before tagging)

- [ ] **Active-set SQP with working-set warm start** (Phase 5b/5c/5d):
      `add_option("algorithm", "active-set-sqp")` + `working_set=` /
      `info["working_set"]`. Tutorial: `docs/tutorials/active-set-sqp.md`.
- [ ] **`pounce.jax.solve_with_warm`** (pounce#74-2) — dual-warm-start
      surface threaded through the JAX boundary.
- [ ] **`pounce.jax.vmap_solve_parallel`** (pounce#74-1) — parallel
      batched solve over a `ThreadPoolExecutor`, backed by the new
      `py.allow_threads` block around `optimize_tnlp` in
      `crates/pounce-py/src/problem.rs`.
- [ ] **`pounce.jax.solve` backward respects the constraint active
      set** (pounce#73 fix) — slack inequality rows are dropped
      from the implicit-function-theorem KKT block.
- [ ] **Auxiliary-equality preprocessing**, **FBBT**, **problem and
      KKT-system scaling**, **Mehrotra adaptive-μ defaults**,
      **`pounce-solve-report` crate + C API**, **diagnostics
      `--dump` family**, **GAMS Studio tools** — all flagged
      `[0.3.0]` in CHANGELOG.md.

## 1. Tag and push

```sh
git tag -a v0.3.0 -m "POUNCE v0.3.0"
git push origin main
git push origin v0.3.0
```

## 2. crates.io publish order

Cargo requires every transitive path-dep to already be on crates.io
before a dependent crate can publish (the `cargo package` index-lookup
step fails otherwise). The order below is a topological sort of the
internal dep DAG; each level can be published in parallel, but no
level may start until the previous one is on the index.

Excluded (publish=false): `pounce-py`, `pounce-studio-pyo3`. These
ship through PyPI, not crates.io.

| Wave | Crates | Internal deps |
|------|--------|---------------|
| 1 | `pounce-common`, `pounce-studio-core` | (none) |
| 2 | `pounce-linalg` | common |
| 3 | `pounce-linsol` | common, linalg |
| 4 | `pounce-feral`, `pounce-hsl`, `pounce-nlp`, `pounce-solve-report` | through linsol |
| 5 | `pounce-presolve`, `pounce-l1penalty`, `pounce-qp` | through nlp / feral |
| 6 | `pounce-algorithm` | through qp + presolve |
| 7 | `pounce-restoration`, `pounce-sensitivity` | through algorithm |
| 8 | `pounce-cinterface`, `pounce-cli` | through restoration + sensitivity + solve-report |

Per crate:

```sh
cargo publish -p <name>
# wait ~30s for the index to update before the next wave
```

After each wave, smoke-check with `cargo search <name>` or just rerun
`cargo publish --dry-run -p <next-wave-crate>` — the index-lookup
error disappears once deps land.

## 3. Wheels (after the C ABI is on crates.io)

`pounce-solver` and `pyomo-pounce` ship via PyPI:

```sh
# pounce-solver — built via maturin, bundles the Rust CLI
cd python
maturin publish --release --skip-existing

# pyomo-pounce — pure-Python sdist + wheel
cd ../pyomo-pounce
python -m build
twine upload --skip-existing dist/*
```

The GitHub `wheels.yml` workflow runs on the tag push and builds the
multi-platform `pounce-solver` wheels automatically; the manual
`maturin publish` above is for the local macOS wheel only if the CI
matrix doesn't already cover it.

## 4. GitHub release

```sh
gh release create v0.3.0 \
  --title "POUNCE v0.3.0" \
  --notes-file <(awk '/^## \[0.3.0\]/{flag=1; next} /^## \[0.2.0\]/{flag=0} flag' CHANGELOG.md)
```

Zenodo picks up the release automatically via the
`.zenodo.json` + `CITATION.cff` integration and mints a new DOI under
the parent record `10.5281/zenodo.20387011`.

## 5. Post-release

- [ ] Bump workspace to `0.4.0-alpha.0` (or leave at `0.3.0` — pick
      one, but be explicit).
- [ ] Open the next `## Unreleased` section in `CHANGELOG.md`.
- [ ] Announce: README badges should pick the new versions
      automatically (shields.io PyPI / crates.io live queries).
- [ ] Delete this checklist (`git rm dev-notes/release-v0.3.0-checklist.md`).
