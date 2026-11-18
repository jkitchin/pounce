//! Augmented-system solvers supplied by the embedding build (gh#1175
//! follow-up) — the counterpart of Ipopt's custom `AugSystemSolver` /
//! virtual `AlgorithmBuilder::AugSystemSolverFactory`.
//!
//! `linear_solver=custom` (see [`crate::custom_linsol`]) is the
//! sparse-linear-solver level: a C library is handed the assembled matrix.
//! A [`LinearSolverProvider`] sits one level up. It is handed the
//! augmented-system *pieces* on every solve (`W`, `D_x`, `δ_x`, `J_c`, `J_d`,
//! `D_c`, `D_d`, `δ_c`, through [`AugSysCoeffs`](crate::kkt::aug_system_solver::AugSysCoeffs)),
//! so a structured KKT oracle can own regularization, inertia handling and
//! refinement of the system it builds from them.
//!
//! A provider is registered once per process with
//! [`register_linear_solver_provider`], **before** the first
//! [`IpoptApplication`](crate::application::IpoptApplication) is created, and
//! is then selected with the ordinary option `linear_solver=<name>`. Because
//! the selection is an option, it reaches every frontend linked into the same
//! binary (C API, CasADi plugin, CLI, Python) without any of them knowing the
//! provider exists. Registration also registers the provider's own typed
//! options ([`LinearSolverProvider::register_options`]), so they are
//! validated, documented and type-dispatched exactly like `ma57_*` and
//! `feral_*`.
//!
//! Where the builder would build a `StdAugSystemSolver`, it calls
//! [`LinearSolverProvider::aug_system_solver`] instead, and wraps the result
//! exactly as it wraps the standard solver: in `LowRankAugSystemSolver` (and
//! its bypass solver) under `hessian_approximation=limited-memory`, and in
//! `AugRestoSystemSolver` in the restoration phase (where the context's
//! `prefix` is `"resto."`). The FERAL-specific Schur and block-parallel KKT
//! paths stand down, as they do for `custom`. The active-set SQP's QP
//! subproblem is not an augmented system and keeps the default backend.
//!
//! A provider that returns a solver whose
//! [`owns_refinement`](crate::kkt::aug_system_solver::AugSystemSolver::owns_refinement)
//! is `true` is solved once per step and trusted; see that method.

use crate::custom_linsol::KktMetadataCell;
use crate::kkt::aug_system_solver::AugSystemSolver;
use crate::kkt::std_aug_system_solver::StdAugSystemSolver;
use pounce_common::exception::SolverException;
use pounce_common::options_list::OptionsList;
use pounce_common::reg_options::RegisteredOptions;
use pounce_linsol::SparseSymLinearSolverInterface;
use std::cell::RefCell;
use std::sync::RwLock;

/// An augmented-system solver supplied by the embedding build and selected
/// by `linear_solver = <name>`.
pub trait LinearSolverProvider: Send + Sync {
    /// The `linear_solver` value that selects it. Must not be one of the
    /// built-in values (`feral`, `ma57`, `custom`, or any upstream Ipopt
    /// name) and must be unique among providers.
    fn name(&self) -> &'static str;

    /// One-line description shown in the `linear_solver` option's list of
    /// values.
    fn description(&self) -> &'static str {
        "augmented-system solver registered by the embedding build"
    }

    /// Register the provider's own options, typed and bounded, under a
    /// category of its choosing. Called for every fresh option registry, so
    /// it must be deterministic. Read the resolved values back from
    /// [`AugSolverContext::options`] with [`AugSolverContext::prefix`].
    fn register_options(&self, _reg: &RegisteredOptions) -> Result<(), SolverException> {
        Ok(())
    }

    /// Publish [`KktMetadata`](crate::custom_linsol::KktMetadata) (the KKT
    /// layout and the model's integer var/con metadata in KKT row order) into
    /// [`AugSolverContext::metadata`], on the same terms as
    /// `linear_solver=custom`.
    fn wants_metadata(&self) -> bool {
        true
    }

    /// Build one augmented-system solver. Called wherever the builder would
    /// build a `StdAugSystemSolver`, which is more than once per solve (the
    /// limited-memory bypass solver, the restoration sub-solve), so each
    /// call must return an independent solver.
    fn aug_system_solver(
        &self,
        ctx: &AugSolverContext<'_>,
    ) -> Result<Box<dyn AugSystemSolver>, String>;

    /// Check the resolved options before the solve starts. An `Err` is
    /// reported and the solve refused with `Invalid_Option`, before any
    /// work. A failure in [`Self::aug_system_solver`] itself cannot be
    /// refused that early; it fails the solve at its first factorization.
    fn validate(&self, _options: &OptionsList) -> Result<(), String> {
        Ok(())
    }
}

/// What a provider is given to build a solver with.
pub struct AugSolverContext<'a> {
    /// The application's options, including the provider's own.
    pub options: &'a OptionsList,
    /// `""` for the main solve, `"resto."` for the restoration phase —
    /// pass it to the `OptionsList` getters, as Ipopt components do.
    pub prefix: &'a str,
    /// The KKT layout and integer metadata of the current solve. Populated
    /// before the first factorization (when
    /// [`LinearSolverProvider::wants_metadata`]); read it then, not at
    /// construction — the restoration builder is made before the layout is
    /// known.
    pub metadata: &'a KktMetadataCell,
    pub(crate) default_backend:
        RefCell<&'a mut dyn FnMut() -> Box<dyn SparseSymLinearSolverInterface>>,
    pub(crate) standard: &'a dyn Fn(Box<dyn SparseSymLinearSolverInterface>) -> StdAugSystemSolver,
}

impl AugSolverContext<'_> {
    /// A fresh instance of the general sparse backend the run would
    /// otherwise use (FERAL), for a provider that keeps one under its
    /// structured solver — typically passed straight to [`Self::standard`].
    pub fn default_backend(&self) -> Box<dyn SparseSymLinearSolverInterface> {
        (self.default_backend.borrow_mut())()
    }

    /// POUNCE's `StdAugSystemSolver` on `backend`, configured exactly as the
    /// builder configures its own (`linear_system_scaling`, scaling on
    /// demand, inertia reporting). For a provider that wraps the standard
    /// solver — the counterpart of overriding Ipopt's
    /// `AugSystemSolverFactory` and wrapping `StdAugSystemSolver`.
    pub fn standard(
        &self,
        backend: Box<dyn SparseSymLinearSolverInterface>,
    ) -> Box<dyn AugSystemSolver> {
        Box::new((self.standard)(backend))
    }
}

static PROVIDERS: RwLock<Vec<&'static dyn LinearSolverProvider>> = RwLock::new(Vec::new());

/// The registered providers, in registration order.
pub fn providers() -> Vec<&'static dyn LinearSolverProvider> {
    PROVIDERS.read().map(|g| g.clone()).unwrap_or_default()
}

/// The provider `linear_solver = name` selects, if any (case-insensitive).
pub fn find_provider(name: &str) -> Option<&'static dyn LinearSolverProvider> {
    providers()
        .into_iter()
        .find(|p| p.name().eq_ignore_ascii_case(name))
}

/// Register `provider` for this process. Call it before creating the first
/// `IpoptApplication`: an application registers its options once, when it
/// is created, so one created earlier neither accepts the name nor knows the
/// provider's options.
///
/// Refused, with nothing registered, when the name is empty, is already a
/// `linear_solver` value (built-in or another provider's), or when the
/// provider's options fail to register next to every other option (a
/// duplicate name). Registering the same provider object twice is a no-op.
pub fn register_linear_solver_provider(
    provider: &'static dyn LinearSolverProvider,
) -> Result<(), String> {
    let name = provider.name();
    if name.is_empty() || name.chars().any(|c| c.is_whitespace()) {
        return Err(format!("invalid linear solver provider name '{name}'"));
    }
    if providers()
        .iter()
        .any(|p| std::ptr::addr_eq(*p as *const _, provider as *const _))
    {
        return Ok(());
    }
    // Trial registration into a fresh registry: the built-in options plus
    // every provider already registered, then this one.
    let trial = RegisteredOptions::default();
    crate::upstream_options::register_all_upstream_options(&trial)
        .map_err(|e| format!("option registry: {e}"))?;
    if trial
        .get_option("linear_solver")
        .is_some_and(|ls| ls.is_valid_string(name))
    {
        return Err(format!(
            "linear_solver={name} is already taken; choose another provider name"
        ));
    }
    provider
        .register_options(&trial)
        .map_err(|e| format!("linear solver provider '{name}': {e}"))?;
    let mut g = PROVIDERS
        .write()
        .map_err(|_| "linear solver provider registry is poisoned".to_string())?;
    if g.iter().any(|p| p.name().eq_ignore_ascii_case(name)) {
        return Err(format!(
            "linear_solver={name} is already taken; choose another provider name"
        ));
    }
    g.push(provider);
    Ok(())
}

/// The `linear_solver` values contributed by providers, for the option's
/// registration.
pub(crate) fn provider_values() -> Vec<(&'static str, &'static str)> {
    providers()
        .into_iter()
        .map(|p| (p.name(), p.description()))
        .collect()
}

/// Register every provider's options into `reg`.
pub(crate) fn register_provider_options(reg: &RegisteredOptions) -> Result<(), SolverException> {
    for p in providers() {
        p.register_options(reg)?;
    }
    Ok(())
}

/// A selected provider, as an [`AlgorithmBuilder`](crate::alg_builder::AlgorithmBuilder)
/// carries it: the provider plus what its context needs.
#[derive(Clone)]
pub struct ProviderSelection {
    pub provider: &'static dyn LinearSolverProvider,
    /// The application's options at the time the builder was configured.
    pub options: OptionsList,
    /// `""`, or `"resto."` on the restoration sub-solve's builder.
    pub prefix: String,
    pub metadata: KktMetadataCell,
}

impl std::fmt::Debug for ProviderSelection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ProviderSelection")
            .field("provider", &self.provider.name())
            .field("prefix", &self.prefix)
            .finish_non_exhaustive()
    }
}

/// Stands in for a solver the provider failed to build: every solve is a
/// fatal error, so the run stops instead of quietly using another backend.
pub(crate) struct FailedProviderSolver;

impl AugSystemSolver for FailedProviderSolver {
    fn provides_inertia(&self) -> bool {
        false
    }
    fn number_of_neg_evals(&self) -> pounce_common::types::Index {
        0
    }
    fn increase_quality(&mut self) -> bool {
        false
    }
    fn last_solve_status(&self) -> pounce_linsol::ESymSolverStatus {
        pounce_linsol::ESymSolverStatus::FatalError
    }
    fn solve(
        &mut self,
        _coeffs: &crate::kkt::aug_system_solver::AugSysCoeffs<'_>,
        _rhs: &crate::kkt::aug_system_solver::AugSysRhs<'_>,
        _sol: &mut crate::kkt::aug_system_solver::AugSysSol<'_>,
        _check_neg_evals: bool,
        _num_neg_evals: pounce_common::types::Index,
    ) -> pounce_linsol::ESymSolverStatus {
        pounce_linsol::ESymSolverStatus::FatalError
    }
}
