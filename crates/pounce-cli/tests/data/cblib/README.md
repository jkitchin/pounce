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

Each is in Conic Benchmark Format (`.cbf`, version 2), the plain-text format
documented at <https://cblib.zib.de/format.html>. They are small
(pure-continuous) and freely distributed by CBLIB for benchmarking; vendored
here so the cross-check tests run offline.
