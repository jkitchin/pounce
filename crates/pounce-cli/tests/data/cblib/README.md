# CBLIB test fixtures

These are exponential-cone geometric-program instances from the **Conic
Benchmark Library** (CBLIB, <https://cblib.zib.de>), used as gold-standard
broad validation for the non-symmetric (exp-cone) HSDE solver — see
`dev-notes/hsde.md`, "CBLIB benchmark tier".

| File | Family | Cones |
|---|---|---|
| `demb761.cbf` | Demberg geometric program | exp (over variables) |
| `beck751.cbf` | Beck geometric program | exp (over variables) |
| `fang88.cbf`  | Fang geometric program | exp (over variables) |
| `pow3_synthetic.cbf` | hand-authored (not CBLIB) | power (`POWCONES`) |
| `sdp_synthetic.cbf` | hand-authored (not CBLIB) | semidefinite (`PSDCON`/`DCOORD`) |

The first three are in Conic Benchmark Format (`.cbf`, version 2), the
plain-text format documented at <https://cblib.zib.de/format.html>. They are
small (pure-continuous) and freely distributed by CBLIB for benchmarking;
vendored here so the cross-check tests run offline.

`pow3_synthetic.cbf` and `sdp_synthetic.cbf` are **not** CBLIB instances —
they are tiny hand-authored problems exercising the `POWCONES` (power-cone)
and `PSDCON`/`HCOORD`/`DCOORD` (affine semidefinite-constraint) sections,
each with a known closed-form optimum (`x₂ = 1` and `λ = 2`). The real CBLIB
power-cone instances (`2013_fir*`) are ~120 MB, impractical to vendor.
