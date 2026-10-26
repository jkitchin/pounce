//! `MinimaOptions` equality and hashing (gh#1065): every field takes part,
//! floats compare by bit pattern, so the options can key a cache.

use std::collections::HashSet;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use pounce_minima::options::{Method, MinimaOptions};

fn hash(o: &MinimaOptions) -> u64 {
    let mut h = DefaultHasher::new();
    o.hash(&mut h);
    h.finish()
}

#[test]
fn a_clone_is_equal_and_hashes_the_same() {
    let o = MinimaOptions {
        method: Method::Flooding,
        sigma_per_dim: Some(vec![0.5, 2.0]),
        amp_bump: Some(4.0),
        ..Default::default()
    };
    assert_eq!(o, o.clone());
    assert_eq!(hash(&o), hash(&o.clone()));
}

#[test]
fn nan_options_equal_themselves() {
    let o = MinimaOptions {
        sigma: Some(f64::NAN),
        length_per_dim: Some(vec![f64::NAN]),
        ..Default::default()
    };
    assert_eq!(o, o.clone());
    let mut cache = HashSet::new();
    cache.insert(o.clone());
    assert!(cache.contains(&o));
}

/// One case per kind of field, including the library-only knobs that no CLI
/// flag reaches: a field dropped from the comparison makes its row equal.
#[test]
fn changing_any_one_knob_makes_the_options_unequal() {
    let base = MinimaOptions::default();
    let cases: Vec<(&str, MinimaOptions)> = vec![
        (
            "method",
            MinimaOptions {
                method: Method::Mlsl,
                ..base.clone()
            },
        ),
        (
            "n_minima",
            MinimaOptions {
                n_minima: 3,
                ..base.clone()
            },
        ),
        (
            "dedup",
            MinimaOptions {
                dedup: 1e-3,
                ..base.clone()
            },
        ),
        (
            "kkt_tol",
            MinimaOptions {
                kkt_tol: None,
                ..base.clone()
            },
        ),
        (
            "sobol",
            MinimaOptions {
                sobol: false,
                ..base.clone()
            },
        ),
        (
            "restart_jitter",
            MinimaOptions {
                restart_jitter: Some(0.1),
                ..base.clone()
            },
        ),
        (
            "sigma_per_dim",
            MinimaOptions {
                sigma_per_dim: Some(vec![1.0]),
                ..base.clone()
            },
        ),
        (
            "length_per_dim",
            MinimaOptions {
                length_per_dim: Some(vec![1.0]),
                ..base.clone()
            },
        ),
        (
            "max_samples",
            MinimaOptions {
                max_samples: Some(50),
                ..base.clone()
            },
        ),
        (
            "amp_bump",
            MinimaOptions {
                amp_bump: Some(3.0),
                ..base.clone()
            },
        ),
        (
            "amp_bump_cap",
            MinimaOptions {
                amp_bump_cap: Some(1e3),
                ..base.clone()
            },
        ),
        (
            "amp_fallback",
            MinimaOptions {
                amp_fallback: Some(2.0),
                ..base.clone()
            },
        ),
    ];
    for (field, changed) in cases {
        assert_ne!(base, changed, "{field} does not take part in ==");
    }
    let per_dim = |v: f64| MinimaOptions {
        sigma_per_dim: Some(vec![1.0, v]),
        ..base.clone()
    };
    assert_ne!(
        per_dim(2.0),
        per_dim(3.0),
        "sigma_per_dim compares elementwise"
    );
}
