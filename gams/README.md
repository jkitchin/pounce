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

## Capabilities

Registered model types: `NLP`, `DNLP`, `RMINLP`. Mixed-integer and conic
problem types are not supported by the underlying POUNCE solver.
