//! `pounce_sens` — backward-compatibility alias for the `pounce`
//! binary.
//!
//! The parametric-sensitivity step (and the reduced-Hessian
//! computation) used to live in a separate `pounce_sens` binary. They
//! are now part of the main `pounce` driver, which auto-detects the
//! sIPOPT-style suffixes (`sens_state_1`, `sens_state_value_1`,
//! `sens_init_constr`) directly from the input `.nl` — see
//! [`pounce_cli::sens`].
//!
//! This alias exists only so existing AMPL / solver scripts that
//! invoke `pounce_sens <in.nl> [<out.sol>]` keep working unchanged; it
//! is the exact same program as `pounce`, sharing `main.rs` verbatim.

#[path = "../main.rs"]
mod pounce;

fn main() -> std::process::ExitCode {
    pounce::main()
}
