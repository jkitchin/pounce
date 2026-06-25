# pounce-rs

Single-crate facade for solving nonlinear programs with [POUNCE](https://github.com/jkitchin/pounce)
from Rust. Re-exports the `TNLP` problem trait (`pounce-nlp`), the
`IpoptApplication` driver (`pounce-algorithm`), and the supporting scalar
types (`pounce-common`) in one place, plus a `prelude`:

```rust
use pounce_rs::prelude::*;
```

This is the Rust counterpart to the one-import `import pounce` Python API. It
contains re-exports only; see the crate docs for a complete HS071 example.
