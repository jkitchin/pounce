//! Debugger glue for the convex interior-point method.
//!
//! [`ConvexDebugState`] adapts one iteration of the convex IPM /
//! HSDE loops to the shared [`DebugState`] surface, so the CLI's
//! `SolverDebugger` (a [`DebugHook`]) can step, inspect, **mutate**, and
//! break on a convex LP / QP / conic solve as it does on the NLP path.
//!
//! Block names follow the QP standard form: `x` (variables), `s` (cone
//! slacks), `y` (equality multipliers), `z` (inequality / cone
//! multipliers); their search-direction counterparts are addressed by the
//! same names. The HSDE drivers additionally expose the homogenizing
//! scalars `tau` / `kappa` as 1-element blocks.
//!
//! The state borrows the live iterate **mutably**, so `set <block>` edits
//! it in place and `snapshot`/`restore` (the `goto` rewind) round-trip it.
//! `set mu` is rejected: the convex μ is *derived* from `⟨s, z⟩`, not a
//! free knob — edit `s`/`z` instead. There is no backtracking line search
//! or restoration phase, so [`ls_count`](DebugState::ls_count) reports
//! "n/a".

use pounce_common::debug::{Checkpoint, DebugAction, DebugHook, DebugState, IterSnapshot};
use pounce_common::types::Number;
use std::any::Any;

/// A captured convex/HSDE iterate for `goto`/rewind. Stores the primal-dual
/// blocks plus the homogenizing scalars (HSDE) so a restore is exact.
pub(crate) struct ConvexSnapshot {
    iter: i32,
    mu: f64,
    x: Vec<f64>,
    s: Vec<f64>,
    y: Vec<f64>,
    z: Vec<f64>,
    tau: Option<f64>,
    kappa: Option<f64>,
}

impl IterSnapshot for ConvexSnapshot {
    fn iter(&self) -> i32 {
        self.iter
    }
    fn mu(&self) -> Number {
        self.mu
    }
    fn block(&self, name: &str) -> Option<Vec<Number>> {
        match name {
            "x" => Some(self.x.clone()),
            "s" => Some(self.s.clone()),
            "y" => Some(self.y.clone()),
            "z" => Some(self.z.clone()),
            "tau" => self.tau.map(|t| vec![t]),
            "kappa" => self.kappa.map(|k| vec![k]),
            _ => None,
        }
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// A live, mutable view of one convex-IPM / HSDE iteration for the debugger.
///
/// Holds mutable borrows of the live iterate (`x`/`s`/`y`/`z`, and for the
/// HSDE drivers the scalars `τ`/`κ`) plus read-only borrows of the current
/// search direction (`dx`/…). Cheap to build and dropped before the loop
/// touches the iterate again.
pub(crate) struct ConvexDebugState<'a> {
    pub cp: Checkpoint,
    pub iter: i32,
    pub mu: f64,
    /// Max-norm primal infeasibility (max over equality / cone residuals).
    pub pinf: f64,
    /// Max-norm dual (stationarity) infeasibility.
    pub dinf: f64,
    /// `max(pinf, dinf, mu)` — the scalar convergence test.
    pub res: f64,
    pub obj: f64,
    pub alpha: (f64, f64),
    pub x: &'a mut [f64],
    pub s: &'a mut [f64],
    pub y: &'a mut [f64],
    pub z: &'a mut [f64],
    pub dx: &'a [f64],
    pub dy: &'a [f64],
    pub dz: &'a [f64],
    pub ds: &'a [f64],
    /// HSDE homogenizing variable τ (the iterate is the homogeneous
    /// `(x, s, y, z, τ, κ)`; the recovered solution is `x/τ`). `None` for
    /// the direct (non-homogeneous) driver.
    pub tau: Option<&'a mut f64>,
    /// HSDE homogenizing variable κ. `None` for the direct driver.
    pub kappa: Option<&'a mut f64>,
    pub status: Option<&'a str>,
}

impl ConvexDebugState<'_> {
    /// Write `vals` into a named iterate block in place (length-checked).
    fn write_block(&mut self, name: &str, vals: &[Number]) -> Result<(), String> {
        let slot: &mut [f64] = match name {
            "x" => self.x,
            "s" => self.s,
            "y" => self.y,
            "z" => self.z,
            "tau" => {
                return set_scalar(self.tau.as_deref_mut(), "tau", vals);
            }
            "kappa" => {
                return set_scalar(self.kappa.as_deref_mut(), "kappa", vals);
            }
            _ => return Err(format!("unknown block `{name}`")),
        };
        if vals.len() != slot.len() {
            return Err(format!(
                "block `{name}` has dimension {}, got {} value(s)",
                slot.len(),
                vals.len()
            ));
        }
        slot.copy_from_slice(vals);
        Ok(())
    }
}

/// Set a single-element scalar "block" (`tau`/`kappa`) if it exists.
fn set_scalar(slot: Option<&mut f64>, name: &str, vals: &[Number]) -> Result<(), String> {
    let Some(slot) = slot else {
        return Err(format!("this solver has no `{name}`"));
    };
    match vals {
        [v] => {
            *slot = *v;
            Ok(())
        }
        _ => Err(format!(
            "`{name}` is a scalar; expected 1 value, got {}",
            vals.len()
        )),
    }
}

impl DebugState for ConvexDebugState<'_> {
    fn checkpoint(&self) -> Checkpoint {
        self.cp
    }
    fn iter(&self) -> i32 {
        self.iter
    }
    fn mu(&self) -> Number {
        self.mu
    }
    fn objective(&self) -> Number {
        self.obj
    }
    fn inf_pr(&self) -> Number {
        self.pinf
    }
    fn inf_du(&self) -> Number {
        self.dinf
    }
    fn complementarity(&self) -> Number {
        // For a symmetric cone μ = ⟨s, z⟩ / degree is exactly the average
        // complementarity, so it doubles as the central-path gauge.
        self.mu
    }
    fn alpha(&self) -> (Number, Number) {
        self.alpha
    }
    fn block_dims(&self) -> Vec<(&'static str, usize)> {
        let mut v = vec![
            ("x", self.x.len()),
            ("s", self.s.len()),
            ("y", self.y.len()),
            ("z", self.z.len()),
        ];
        // The homogenizing scalars are addressable as 1-element blocks on
        // the HSDE driver (`print tau` / `print kappa`).
        if self.tau.is_some() {
            v.push(("tau", 1));
        }
        if self.kappa.is_some() {
            v.push(("kappa", 1));
        }
        v
    }
    fn block(&self, name: &str) -> Option<Vec<Number>> {
        match name {
            "x" => Some(self.x.to_vec()),
            "s" => Some(self.s.to_vec()),
            "y" => Some(self.y.to_vec()),
            "z" => Some(self.z.to_vec()),
            "tau" => self.tau.as_deref().copied().map(|t| vec![t]),
            "kappa" => self.kappa.as_deref().copied().map(|k| vec![k]),
            _ => None,
        }
    }
    fn delta_block(&self, name: &str) -> Option<Vec<Number>> {
        match name {
            "x" => Some(self.dx.to_vec()),
            "s" => Some(self.ds.to_vec()),
            "y" => Some(self.dy.to_vec()),
            "z" => Some(self.dz.to_vec()),
            _ => None,
        }
    }
    fn status(&self) -> Option<&str> {
        self.status
    }
    /// The convex IPM's scalar convergence error `max(pinf, dinf, μ)`, so
    /// `break if err<…` works the same as on the NLP path.
    fn nlp_error(&self) -> Number {
        self.res
    }

    // ---- mutation -------------------------------------------------------

    /// Rejected: the convex/HSDE μ is derived from `⟨s, z⟩` (and `τκ`), not
    /// a free parameter — editing it would be silently overwritten next
    /// iteration. Edit `s`/`z` to move μ.
    fn set_mu(&mut self, _mu: Number) -> Result<(), String> {
        Err("convex μ is derived from ⟨s,z⟩; edit the `s`/`z` blocks instead".into())
    }

    fn set_block(&mut self, name: &str, vals: &[Number]) -> Result<(), String> {
        self.write_block(name, vals)
    }

    // ---- snapshot / rewind ---------------------------------------------

    fn snapshot(&self) -> Option<Box<dyn IterSnapshot>> {
        Some(Box::new(ConvexSnapshot {
            iter: self.iter,
            mu: self.mu,
            x: self.x.to_vec(),
            s: self.s.to_vec(),
            y: self.y.to_vec(),
            z: self.z.to_vec(),
            tau: self.tau.as_deref().copied(),
            kappa: self.kappa.as_deref().copied(),
        }))
    }

    fn restore(&mut self, snap: &dyn IterSnapshot) -> bool {
        let Some(s) = snap.as_any().downcast_ref::<ConvexSnapshot>() else {
            return false;
        };
        // Dimensions must match the live iterate (a snapshot from a
        // different problem/driver is refused rather than truncated).
        if s.x.len() != self.x.len()
            || s.s.len() != self.s.len()
            || s.y.len() != self.y.len()
            || s.z.len() != self.z.len()
            || s.tau.is_some() != self.tau.is_some()
        {
            return false;
        }
        self.x.copy_from_slice(&s.x);
        self.s.copy_from_slice(&s.s);
        self.y.copy_from_slice(&s.y);
        self.z.copy_from_slice(&s.z);
        if let (Some(dst), Some(v)) = (self.tau.as_deref_mut(), s.tau) {
            *dst = v;
        }
        if let (Some(dst), Some(v)) = (self.kappa.as_deref_mut(), s.kappa) {
            *dst = v;
        }
        true
    }
}

/// Fire a checkpoint at `state` if a hook is attached. A no-op (and
/// always [`DebugAction::Resume`]) when `hook` is `None`, so the
/// hook-free solve path pays nothing.
pub(crate) fn fire(
    hook: &mut Option<&mut dyn DebugHook>,
    state: &mut dyn DebugState,
) -> DebugAction {
    match hook.as_mut() {
        Some(h) => h.at_checkpoint(state),
        None => DebugAction::Resume,
    }
}
