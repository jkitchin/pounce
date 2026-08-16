# `pounce-casadi` — the plugin as a Python wheel

Packages the CasADi `nlpsol` plugin built in `../` so that installing it is
all a user does:

```sh
pip install pounce-casadi          # not published yet — see below
```

```python
import casadi as ca
import pounce_casadi               # registers the plugin

solver = ca.nlpsol("solver", "pounce", nlp, {"pounce": {"tol": 1e-9}})
```

Importing the package `dlopen`s the shipped plugin and calls its
`casadi_load_nlpsol_pounce` hook — the same entry point CasADi's own loader
calls after finding a plugin on its search path. Nothing is written into
CasADi's installation, no `CASADIPATH` is needed, and CasADi's bundled
plugins (Ipopt included) stay loadable side by side, which is what lets you
cross-check the two solvers in one process.

## Building it here

```sh
./build.sh          # builds the plugin for the installed casadi, writes dist/*.whl
```

Then, in a clean environment:

```sh
pip install casadi==<same minor> dist/pounce_casadi-*.whl
python -c "import casadi, pounce_casadi; print(casadi.nlpsol('S','pounce',
           {'x':casadi.MX.sym('x'),'f':casadi.MX.sym('x')**2}))"
```

## What a published wheel still needs

The plugin is a C++ extension of CasADi, so it is bound to a CasADi **minor**
version and to the platform's C++ ABI. CasADi performs no version handshake,
so this package refuses to guess: it ships one build per supported minor
version under `pounce_casadi/_plugins/<minor>/` and selects on
`casadi.__version__`, raising a clear `ImportError` when there is no match.

To publish, `build.sh` has to run once per (CasADi minor × platform) and the
resulting `_plugins/<minor>/` trees merged into one wheel per platform:

- **Linux** — inside a manylinux image, `-D_GLIBCXX_USE_CXX11_ABI=0` to match
  the CasADi wheels (the Makefile's default), then `auditwheel` **excluding**
  `libcasadi` (it must resolve to the user's installed copy at runtime, not be
  vendored).
- **macOS** — x86_64 and arm64. The Makefile rewrites the plugin's reference
  to `libpounce_cinterface` to `@rpath` and adds `@loader_path`, because Rust
  stamps a cdylib's install name with its absolute *build* path; without that
  the staged plugin loads only on the build machine. Verify with
  `otool -L` before shipping — the line must read
  `@rpath/libpounce_cinterface.dylib`, not an absolute path.
- **Windows** — MSVC, matching CasADi's own toolchain.

CasADi minor releases are infrequent (3.6 in 2023, 3.7 in 2025), so the matrix
is small and mostly static. It is also exactly the maintenance that disappears
if the interface is contributed upstream — see
[`dev-notes/casadi-interface-options.md`](../../dev-notes/casadi-interface-options.md).
