# GAMS solver link for POUNCE

This directory builds `libGamsPounce`, a shared library that registers POUNCE
as a GAMS NLP solver. Once installed, a GAMS model can invoke POUNCE with

```
option nlp = pounce;
solve mymodel using nlp minimizing obj;
```

## Files

- `gams_pounce.c` — the solver link. Translates between the GAMS Modeling
  Object (GMO) API and POUNCE's C API (`pounce.h`, a drop-in port of Ipopt
  3.14's `IpStdCInterface.h`). Entry points are `pouCreate`, `pouFree`,
  `pouReadyAPI`, and `pouCallSolver`.
- `Makefile` — builds `libGamsPounce.{dylib,so}` and installs it into a GAMS
  installation.
- `install.sh` — convenience wrapper around `make install` that auto-detects
  the GAMS path on macOS and Linux.
- `test_hs071.gms` — GAMS model of Hock-Schittkowski problem 71. Used by
  `make test` to verify the solver link end-to-end.

## Prerequisites

- A working GAMS installation (the Makefile auto-detects
  `/Library/Frameworks/GAMS.framework/...` on macOS; override with
  `GAMS_PATH=/path/to/gams`).
- `libpounce_cinterface` built in the repo:

  ```
  cargo build --release -p pounce-cinterface
  ```

## Build

```
make -C gams
```

Produces `gams/libGamsPounce.{dylib,so}`, linked against
`../target/release/libpounce_cinterface.{dylib,so}`.

## Install

```
sudo make -C gams install
```

Copies `libGamsPounce` and `libpounce_cinterface` into the GAMS system
directory and registers a `POUNCE` entry in `gmscmpun.txt` so GAMS sees
POUNCE as an available NLP solver. On macOS, rewrites the install-name of
`libpounce_cinterface` inside `libGamsPounce` to
`@loader_path/libpounce_cinterface.dylib` so the loader resolves it from the
GAMS directory.

## Test

```
sudo make -C gams test
```

Runs `test_hs071.gms` through GAMS. The model aborts on an objective
mismatch (`|obj - 17.014| > 1e-2`), an unexpected solve status, or an
unexpected model status — so a clean run is a strong end-to-end check.

## Option files

If a GAMS model sets `mymodel.optfile = 1`, POUNCE reads `pounce.opt`
(`.op2`, `.op3`, ... for higher `optfile` values). Each line is a
`keyword value` pair using POUNCE's option names; lines starting with `*`
or `#` are comments. Integer, real, and string options are auto-detected.

### Active-set SQP with working-set warm start

The Phase 5c SQP driver is opt-in via `pounce.opt`:

```
algorithm active-set-sqp
```

When this is set, the solver link automatically reads the variable
and equation marginals that the previous `solve` statement left on
the GMO and reconstructs a QP working set, which it then forwards
to the next solve via `IpoptSetWarmStartWorkingSet`. No additional
configuration is required — the GAMS-native marginal carry IS the
warm-start channel.

This mechanism (the §7.4(a) **marginal-based reconstruction** in
`docs/research/active-set-sqp-warm-start.md`) is the same idiom
CONOPT, IPOPT, and KNITRO use under GAMS, and shares the same
caveat: at degenerate active sets the marginal signs are
ambiguous, so the reconstructed working set may differ from the
true one by a few rows. The solver degrades gracefully (drops
infeasible rows and re-detects them on the next QP solve); it
never returns the wrong answer.

Mechanism **§7.4(b)** — persistent on-disk state-file carry — is
opt-in via the `sqp_state_file` option in `pounce.opt`:

```
algorithm        active-set-sqp
sqp_state_file   .mymodel.pou-ws
```

When set, the solver link reads the file at the start of each
solve (validating a checksum keyed by the current `n`, `m`,
`x_l`, `x_u`, `g_l`, `g_u`) and writes the post-solve working set
back. Format: 8-byte magic `POUNWS01`, 8-byte FNV-1a checksum,
two int32 dimensions, then `n + m` status bytes. Structural
changes (different bounds, different number of variables) trip
the checksum cleanly and the link silently falls back to the
§7.4(a) marginal reconstruction.

## Examples

`gams/examples/` ships a working parametric-warm-start demo:

- `parametric_sqp_warm_start.gms` — solves a sequence of 20
  perturbed simplex-projection NLPs, each warm-starting from
  the previous solve's marginals via the active-set SQP
  driver.
- `pounce.opt` — companion option file selecting
  `algorithm active-set-sqp` so the SQP warm-start machinery
  activates.

Run after `make install` with:

```
cd gams/examples
gams parametric_sqp_warm_start.gms
```

## Capabilities

Registered model types: `NLP`, `DNLP`, `RMINLP`. Mixed-integer and conic
problem types are not supported by the underlying POUNCE solver.
