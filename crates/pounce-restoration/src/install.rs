//! One-call installation of the restoration phase.
//!
//! # Why this exists
//!
//! `IpoptApplication` runs a restoration phase only if a caller installed one:
//! `pounce-algorithm` cannot build the provider itself, because
//! `pounce-restoration` depends on *it* rather than the other way round. So
//! every frontend has to wire restoration up, and until this module existed
//! each one did it by pasting the same ten lines — resolve the FERAL config
//! from the options, wrap it in a backend-factory factory, mint a provider,
//! install it.
//!
//! Four frontends pasted it correctly. Two did not: `pounce-rs` — the facade
//! that bills itself as "the Rust counterpart to the one-import `import
//! pounce` Python API" — and `pounce-wasm` never depended on this crate at
//! all, so a solve that needed restoration returned `Restoration_Failed`
//! where the CLI and the Python extension returned a real verdict. That is not
//! an exotic path: 10 of the 71 `.nl` fixtures in the CLI corpus invoke
//! restoration, and most of them *succeed* through it (`cresc4`, `deb7`,
//! `eigena2`, `eigmaxa`, `pooling_rt2stp`).
//!
//! Duplicated setup that silently degrades a solve when omitted is the wrong
//! shape for an extension point. [`install_default_restoration`] is the whole
//! wiring behind one call, so a new frontend gets it right by default and the
//! omission cannot recur unnoticed.

use pounce_algorithm::application::{
    IpoptApplication, RestorationFactoryProvider, algorithm_builder_from_option_list,
    default_backend_factory, feral_config_from_options,
};
use pounce_common::options_list::OptionsList;
use pounce_feral::FeralConfig;
use std::rc::Rc;

use crate::resto_alg_builder::RestoAlgorithmBuilder;
use crate::resto_inner_solver::{
    InnerBackendFactoryFactory, make_default_restoration_factory_provider,
};

/// Install the restoration phase on `app`, with the linear-algebra
/// configuration resolved from the application's own options.
///
/// This is what a frontend wants unless it has a reason to override the
/// backend configuration; see
/// [`install_default_restoration_configured`] for the case that does.
///
/// Installs both halves of the contract:
///
/// * a **provider** (not the one-shot factory), so passes that run the inner
///   IPM more than once per `optimize_tnlp` — the ℓ₁ outer loop, the
///   ℓ₁-on-restoration-failure retry, the local-infeasibility second-opinion
///   ladder — get a fresh factory each time instead of the "restoration
///   factory invoked more than once" panic; and
/// * a **mint**, so when the second-opinion ladder changes `feral_scaling` the
///   restoration sub-IPM is rebuilt against the rung's options rather than
///   staying on the settings that just failed.
pub fn install_default_restoration(app: &mut IpoptApplication) {
    install_default_restoration_configured(app, |_cfg| {});
}

/// [`install_default_restoration`] with a hook that adjusts the resolved
/// [`FeralConfig`] before it is used.
///
/// `adjust` runs on every rebuild, not just the first, so an override survives
/// a second-opinion rung re-minting the provider. Batched solving is the case
/// this exists for: it drives its own thread pool and must force
/// `parallel = false` on the inner solves, which would otherwise nest
/// parallelism inside an already-parallel batch.
pub fn install_default_restoration_configured(
    app: &mut IpoptApplication,
    adjust: impl Fn(&mut FeralConfig) + Clone + 'static,
) {
    let mint = move |options: &OptionsList| -> RestorationFactoryProvider {
        let mut feral_cfg = feral_config_from_options(options);
        adjust(&mut feral_cfg);
        let bff_mint = move || -> InnerBackendFactoryFactory {
            let feral_cfg = feral_cfg.clone();
            Box::new(move || default_backend_factory(feral_cfg.clone()))
        };
        make_default_restoration_factory_provider(
            RestoAlgorithmBuilder::new(),
            // Mirror the outer options so the inner IPM inherits the user's
            // `mu_strategy` — and, under a second-opinion rung, the rung's.
            // Matches upstream `IpAlgBuilder::BuildRestoIpoptAlgorithm`.
            algorithm_builder_from_option_list(options),
            bff_mint,
        )
    };
    let provider = mint(app.options());
    app.set_restoration_factory_provider(provider);
    app.set_restoration_provider_mint(Rc::new(mint));
}
