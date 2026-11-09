//! Link this crate's cdylib the way a Python extension module is linked on
//! macOS.
//!
//! The crate is `cdylib` + `rlib`: maturin links the cdylib on its own for
//! `pounce._pounce.studio`, and `pounce-py` links the rlib. But cargo builds
//! the cdylib whenever the crate is built, including as `pounce-py`'s
//! dependency, and maturin passes `-undefined dynamic_lookup` only to the
//! crate it was asked to build. On macOS, where the linker refuses undefined
//! symbols in a shared library by default, the dependency's cdylib then fails
//! on every `_Py*` symbol and `make python-ext` fails with it. Linux allows
//! undefined symbols in a shared library, which is why CI did not see it.
//! The Python symbols are resolved by the interpreter at import time, as for
//! any extension module, so the flag is the correct link mode, not a
//! suppression.

fn main() {
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        println!("cargo:rustc-cdylib-link-arg=-undefined");
        println!("cargo:rustc-cdylib-link-arg=dynamic_lookup");
    }
    println!("cargo:rerun-if-changed=build.rs");
}
