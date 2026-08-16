# CasADi `Nlpsol` plugin for POUNCE

Builds `libcasadi_nlpsol_pounce.so`, which registers POUNCE as a CasADi
NLP solver. Once it is on CasADi's plugin search path:

```python
import casadi as ca
solver = ca.nlpsol("solver", "pounce", nlp, {"pounce": {"tol": 1e-9}})
# …or, with Opti:
opti.solver("pounce", {}, {"tol": 1e-9})
```

User-facing documentation, including the option reference and worked
examples, is in [CasADi integration](../docs/src/casadi.md). This file
covers building and installing the plugin.

## Files

- `casadi_nlpsol_pounce.cpp` — the plugin. A `casadi::Nlpsol` subclass
  that wires CasADi's oracle functions into POUNCE through `pounce.h`,
  the Ipopt-3.14-compatible C API exported by `libpounce_cinterface`.
  Registration entry point: `casadi_register_nlpsol_pounce`.
- `Makefile` — build, install, test, run examples.
- `test_parity.py` — 29 checks cross-referencing POUNCE against CasADi's
  bundled `ipopt` plugin on the same models: primal, both multiplier sets and
  `lam_p`, solution-map derivatives and the bounded-variable gain trap,
  `Opti`, stats, live iteration callbacks, warm starts and the working-set
  carry, L-BFGS masks, exception safety, and a threaded map. Run in CI by the
  `CasADi plugin parity` job.
- `examples/` — six runnable scripts, from hello-world to MPC.

## Build

You need three things: a C++ compiler, POUNCE's C library, and a CasADi
**source tree matching the installed CasADi version** (the pip wheel
ships only public headers — the internal ones a plugin subclasses are
not in it).

```bash
pip install casadi
cargo build --release -p pounce-cinterface   # from the repo root
cd casadi
make fetch-src        # clones casadi at exactly your installed version
make
```

`make fetch-src` clones into `casadi-src/`. If you already have a source
tree, point at it instead: `make CASADI_SRC=/path/to/casadi`.

## Install

CasADi's plugin loader searches its own package directory first, so the
plugin only has to land there — no environment variable, and no `sudo`,
because that is the user's `site-packages`:

```bash
make install       # copies the .so next to the casadi that will load it
```

For a system-wide CasADi you cannot write to, keep the plugin where it
is and point CasADi at it:

```bash
export CASADIPATH=/path/to/pounce/casadi
```

Uninstall with `make uninstall`.

## Verify

```bash
make test          # parity checks against ipopt — all should PASS
make examples      # runs every script in examples/
```

## ABI: what has to match, and why

CasADi's plugin loader does **no** version handshake — it does not
compare the plugin's `CASADI_VERSION` against its own. A plugin built
against the wrong CasADi will either fail to load with an
undefined-symbol error or, worse, load and misbehave. Three things have
to line up:

1. **CasADi version.** Rebuild the plugin for each CasADi minor version.
   The Makefile derives `CASADI_{MAJOR,MINOR,PATCH}_VERSION` from the
   installed package, so a mismatched `CASADI_SRC` usually shows up as a
   compile or link error rather than at runtime.
2. **libstdc++ string ABI.** The pip wheels are built with the
   pre-C++11 ABI, so the plugin is compiled `-D_GLIBCXX_USE_CXX11_ABI=0`
   by default. A CasADi you built yourself probably wants
   `make CXX11_ABI=1`. Getting this wrong looks like:

   ```
   undefined symbol: _ZNK6casadi16FunctionInternal16generate_options...
   ```
3. **The `-D` set.** `-DWITH_DL` is required to compile at all, and the
   rest affect struct layouts. `make abi-flags` prints the flags the
   installed CasADi was actually built with, for comparison with `DEFS`
   in the Makefile.

## Packaging

[`wheel/`](wheel/) packages the plugin as `pounce-casadi`, so that
installing it is all a user does:

```sh
cd wheel && ./build.sh          # builds for the installed casadi -> dist/*.whl
pip install dist/pounce_casadi-*.whl
```

```python
import casadi as ca
import pounce_casadi             # registers the plugin; nothing else needed
ca.nlpsol("solver", "pounce", nlp, {"pounce": {"tol": 1e-9}})
```

Importing it `dlopen`s the shipped plugin and calls its
`casadi_load_nlpsol_pounce` hook — the entry point CasADi's own loader
would call — so nothing is written into CasADi's installation, no
`CASADIPATH` is set, and CasADi's bundled plugins stay loadable
alongside. Nothing is published to PyPI yet; what a release build adds
(the manylinux / macOS / Windows matrix, one entry per CasADi minor
version) is in [`wheel/README.md`](wheel/README.md).
