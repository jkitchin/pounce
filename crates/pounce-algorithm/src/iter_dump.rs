//! Per-iteration binary trace dumper for Track-A bit-equivalence
//! validation against upstream Ipopt.
//!
//! Format spec: `tools/iter-dump/FORMAT.md` (POUNCEIT v1, little-endian,
//! 32-byte fixed header + variable-length name + per-iter records).
//! A reference Python parser lives at `tools/iter-dump/dump_inspect.py`.
//!
//! Activation: gated by the `IPOPT_ITER_DUMP_PATH` environment variable.
//! When unset or empty, [`IterDumper::from_env`] returns `None` and the
//! main loop's hook is a no-op. The optional `IPOPT_ITER_DUMP_NAME`
//! variable supplies the problem-name string written into the header.
//!
//! This module is `pub(crate)` and not exposed in the public API. It is
//! invoked from [`crate::ipopt_alg::IpoptAlgorithm::optimize`] at the
//! same logical points as upstream's writer (after init for iter 0,
//! after every `accept_trial_point`).
//!
//! In v1 the four PD perturbations (`delta_s/c/d`) and the filter
//! contents are advisory and may be left at zero / empty: comparators
//! treat them as such (see FORMAT.md §"`delta_s` / `delta_c` /
//! `delta_d`").

use crate::ipopt_cq::IpoptCqHandle;
use crate::ipopt_data::IpoptDataHandle;
use pounce_common::types::Number;
use pounce_linalg::dense_vector::DenseVector;
use pounce_linalg::Vector;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;

/// Magic bytes identifying a POUNCEIT v1 stream. Matches the upstream
/// patched-Ipopt writer byte-for-byte.
pub const MAGIC: &[u8; 8] = b"POUNCEIT";
/// Format version this writer emits.
pub const FORMAT_VERSION: u32 = 1;

/// Environment variable that enables dumping (set to an absolute file
/// path).
pub const ENV_DUMP_PATH: &str = "IPOPT_ITER_DUMP_PATH";
/// Optional environment variable supplying the problem-name string
/// recorded in the header.
pub const ENV_DUMP_NAME: &str = "IPOPT_ITER_DUMP_NAME";

/// Writer that emits the POUNCEIT v1 binary trace. One instance per
/// `optimize()` call; dropped at the end of the solve, which flushes
/// the underlying buffered file.
pub(crate) struct IterDumper {
    writer: BufWriter<File>,
    /// Whether the header has been emitted. We defer header emission
    /// until the first record, when `(n, m)` are known from the
    /// initialised `curr` iterate.
    header_written: bool,
    name: String,
}

impl IterDumper {
    /// Construct from `IPOPT_ITER_DUMP_PATH`. Returns `None` if the env
    /// var is unset or empty (no-op path). On open-failure, returns
    /// `None` after a stderr note: a broken dump path must never
    /// destabilise the solver.
    pub(crate) fn from_env() -> Option<Self> {
        let path = std::env::var(ENV_DUMP_PATH).ok()?;
        if path.is_empty() {
            return None;
        }
        let pb = PathBuf::from(&path);
        let file = match File::create(&pb) {
            Ok(f) => f,
            Err(e) => {
                tracing::warn!(target: "pounce::diagnostics",
                    "iter_dump: failed to open `{}` for writing: {} — dumping disabled",
                    path, e
                );
                return None;
            }
        };
        let name = std::env::var(ENV_DUMP_NAME).unwrap_or_default();
        Some(Self {
            writer: BufWriter::new(file),
            header_written: false,
            name,
        })
    }

    /// Test-only constructor that opens `path` directly, bypassing the
    /// process environment. Unit tests must NOT round-trip through
    /// `from_env()`/`set_var`: the solver hot paths read `POUNCE_DBG_*`
    /// via `std::env::var` on concurrently-running test threads, and on
    /// glibc a `setenv` racing those `getenv` calls can hand back a
    /// corrupted value — which made `from_env` open the wrong path and the
    /// test read an empty file (the CI flake this replaces).
    #[cfg(test)]
    fn for_test(path: &std::path::Path, name: &str) -> std::io::Result<Self> {
        let file = File::create(path)?;
        Ok(Self {
            writer: BufWriter::new(file),
            header_written: false,
            name: name.to_string(),
        })
    }

    fn write_u32(&mut self, v: u32) -> std::io::Result<()> {
        self.writer.write_all(&v.to_le_bytes())
    }

    fn write_f64(&mut self, v: Number) -> std::io::Result<()> {
        self.writer.write_all(&v.to_le_bytes())
    }

    fn write_vec(&mut self, v: &dyn Vector) -> std::io::Result<()> {
        let len = v.dim() as u32;
        self.write_u32(len)?;
        if len == 0 {
            return Ok(());
        }
        // Try to grab a contiguous f64 slice from a DenseVector. A
        // homogeneous DenseVector materialises to a `len`-long expanded
        // value vector to match upstream's on-disk representation.
        if let Some(dense) = v.as_any().downcast_ref::<DenseVector>() {
            if dense.is_homogeneous() {
                let expanded = dense.expanded_values();
                for x in &expanded {
                    self.writer.write_all(&x.to_le_bytes())?;
                }
                return Ok(());
            }
            // Non-homogeneous DenseVector: write raw little-endian bytes.
            for x in dense.values() {
                self.writer.write_all(&x.to_le_bytes())?;
            }
            return Ok(());
        }
        // Fallback for non-DenseVector backings: this should not occur
        // in v1.0 (POUNCE is dense-only) but we handle it via a copy
        // through `make_new` + `copy`, then probe again.
        let mut tmp = v.make_new();
        tmp.copy(v);
        if let Some(dense) = tmp.as_any().downcast_ref::<DenseVector>() {
            for x in dense.expanded_values().iter() {
                self.writer.write_all(&x.to_le_bytes())?;
            }
            return Ok(());
        }
        // Last resort: write zeros (preserves file structure so a
        // comparator can at least flag the divergence).
        for _ in 0..len {
            self.writer.write_all(&0.0_f64.to_le_bytes())?;
        }
        Ok(())
    }

    /// Flush buffered bytes to the underlying file. `Drop` also flushes,
    /// but that path can only warn on failure; call this when you need to
    /// observe (and surface) a flush error — e.g. before reading the file
    /// back in a test.
    fn flush(&mut self) -> std::io::Result<()> {
        self.writer.flush()
    }

    /// Emit the fixed POUNCEIT header. Called once before the first
    /// record, when `(n, m)` are known.
    fn write_header(&mut self, n: u32, m: u32) -> std::io::Result<()> {
        debug_assert!(!self.header_written);
        self.writer.write_all(MAGIC)?;
        self.write_u32(FORMAT_VERSION)?;
        self.write_u32(n)?;
        self.write_u32(m)?;
        // nnz_jac, nnz_h: written as 0 to match the patched upstream
        // Ipopt's v1 behaviour. Comparators treat these as advisory.
        self.write_u32(0)?;
        self.write_u32(0)?;
        let name_len = self.name.len();
        self.write_u32(name_len as u32)?;
        let name_bytes = self.name.clone();
        self.writer.write_all(name_bytes.as_bytes())?;
        self.header_written = true;
        Ok(())
    }

    /// Emit one iteration record. `data` and `cq` must reference the
    /// post-`accept_trial_point` state (or, for iter 0, the initialised
    /// `curr` iterate).
    pub(crate) fn write_record(&mut self, data: &IpoptDataHandle, cq: &IpoptCqHandle) {
        if let Err(e) = self.write_record_inner(data, cq) {
            tracing::warn!(target: "pounce::diagnostics",
                "iter_dump: failed to write iteration record: {} — dumping aborted",
                e
            );
        }
    }

    fn write_record_inner(
        &mut self,
        data: &IpoptDataHandle,
        cq: &IpoptCqHandle,
    ) -> std::io::Result<()> {
        // Snapshot all data we need before any I/O (avoid holding a
        // borrow across self.writer writes — we don't, but it keeps the
        // structure clear).
        let (iter, mu, tau, alpha_pr, alpha_du, delta_x, delta_s, delta_c, delta_d, curr_opt) = {
            let d = data.borrow();
            (
                d.iter_count as u32,
                d.curr_mu,
                d.curr_tau,
                d.info_alpha_primal,
                d.info_alpha_dual,
                d.info_regu_x,
                d.perturbations.delta_s,
                d.perturbations.delta_c,
                d.perturbations.delta_d,
                d.curr.clone(),
            )
        };
        let Some(curr) = curr_opt else {
            // No `curr` yet (defensive): nothing to write.
            return Ok(());
        };

        // CQ-derived scalars — must be computed *outside* a `data`
        // borrow because CQ accessors take `data.borrow()` themselves.
        let inf_pr = cq.borrow().curr_primal_infeasibility_max();
        let inf_du = cq.borrow().curr_dual_infeasibility_max();
        let constr_viol = cq.borrow().curr_constraint_violation();
        let dual_inf = inf_du; // alias per FORMAT.md
                               // FORMAT.md describes `complementarity` as
                               // `IpCq().curr_complementarity(0.0, NORM_MAX)` — the max-norm
                               // unbarriered complementarity. We compute it directly from the
                               // four `curr_compl_*` blocks (the same pieces curr_nlp_error
                               // already uses).
        let complementarity = {
            let cq_ref = cq.borrow();
            cq_ref
                .curr_compl_x_l()
                .amax()
                .max(cq_ref.curr_compl_x_u().amax())
                .max(cq_ref.curr_compl_s_l().amax())
                .max(cq_ref.curr_compl_s_u().amax())
        };
        let f_val = cq.borrow().curr_f();

        // Header (lazy-write on first record so we know n/m).
        if !self.header_written {
            let n = curr.x.dim() as u32;
            let m = (curr.y_c.dim() + curr.y_d.dim()) as u32;
            self.write_header(n, m)?;
        }

        // Scalar block: u32 iter, u32 status, 14 * f64.
        self.write_u32(iter)?;
        self.write_u32(0)?; // status — always 0 ("in progress") in v1
        self.write_f64(mu)?;
        self.write_f64(tau)?;
        self.write_f64(alpha_pr)?;
        self.write_f64(alpha_du)?;
        self.write_f64(delta_x)?;
        self.write_f64(delta_s)?;
        self.write_f64(delta_c)?;
        self.write_f64(delta_d)?;
        self.write_f64(inf_pr)?;
        self.write_f64(inf_du)?;
        self.write_f64(constr_viol)?;
        self.write_f64(dual_inf)?;
        self.write_f64(complementarity)?;
        self.write_f64(f_val)?;

        // Iterate vector block: x, s, y_c, y_d, z_L, z_U, v_L, v_U.
        self.write_vec(&*curr.x)?;
        self.write_vec(&*curr.s)?;
        self.write_vec(&*curr.y_c)?;
        self.write_vec(&*curr.y_d)?;
        self.write_vec(&*curr.z_l)?;
        self.write_vec(&*curr.z_u)?;
        self.write_vec(&*curr.v_l)?;
        self.write_vec(&*curr.v_u)?;

        // Filter block — advisory in v1, write count=0.
        self.write_u32(0)?;
        Ok(())
    }
}

impl Drop for IterDumper {
    fn drop(&mut self) {
        // BufWriter flushes on drop, but surface any error rather than
        // swallowing it silently.
        if let Err(e) = self.flush() {
            tracing::warn!(target: "pounce::diagnostics", "iter_dump: failed to flush trace file on drop: {}", e);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ipopt_data::IpoptData;
    use crate::iterates_vector::IteratesVector;
    use pounce_linalg::dense_vector::DenseVectorSpace;
    use std::cell::RefCell;
    use std::rc::Rc;

    /// Serializes tests that mutate the process-global `ENV_DUMP_PATH` /
    /// `ENV_DUMP_NAME` env vars. Without this, parallel tests interleave
    /// `set_var`/`remove_var` and one test's `from_env()` observes
    /// another's path. Poison is ignored — a panicking test still
    /// releases the critical section for the rest.
    fn env_guard() -> std::sync::MutexGuard<'static, ()> {
        static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
        ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner())
    }

    /// A process- and call-unique temp path. `std::process::id()` alone is
    /// constant for the whole test binary, so a re-run or any future test
    /// reusing the same prefix would share one file; the atomic counter
    /// makes every path unique within the process, removing that footgun.
    fn unique_temp_path(tag: &str) -> std::path::PathBuf {
        use std::sync::atomic::{AtomicU64, Ordering};
        static SEQ: AtomicU64 = AtomicU64::new(0);
        let n = SEQ.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!(
            "pounce_iter_dump_{tag}_{}_{n}.bin",
            std::process::id()
        ))
    }

    fn dense(n: i32, vals: Option<&[Number]>) -> Rc<dyn Vector> {
        let space = DenseVectorSpace::new(n);
        let mut dv = space.make_new_dense();
        if let Some(v) = vals {
            dv.set_values(v);
        }
        Rc::new(dv) as Rc<dyn Vector>
    }

    #[test]
    fn write_vec_emits_len_then_values_little_endian() {
        // Round-trip a small vector through write_vec → tempfile. Open the
        // file directly (no env) — see `for_test`.
        let path = unique_temp_path("vec");
        let mut dumper = IterDumper::for_test(&path, "").expect("dumper");
        let v = dense(3, Some(&[1.0_f64, 2.0, 3.0]));
        dumper.write_vec(&*v).unwrap();
        // Flush explicitly so a write/flush failure surfaces here as a
        // clear error instead of a mystifying empty-file assert below.
        dumper.flush().expect("flush");
        drop(dumper);
        let bytes = std::fs::read(&path).unwrap();
        // 4 bytes len (=3) + 3 * 8 bytes values
        assert_eq!(bytes.len(), 4 + 3 * 8);
        assert_eq!(&bytes[0..4], &3u32.to_le_bytes());
        assert_eq!(&bytes[4..12], &1.0_f64.to_le_bytes());
        assert_eq!(&bytes[12..20], &2.0_f64.to_le_bytes());
        assert_eq!(&bytes[20..28], &3.0_f64.to_le_bytes());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn from_env_returns_none_when_unset() {
        // The only test that still touches the process environment, and it
        // only reads/clears `ENV_DUMP_PATH` (no test sets it anymore), so
        // there is no setenv/getenv data race with concurrent tests.
        let _env = env_guard();
        std::env::remove_var(ENV_DUMP_PATH);
        assert!(IterDumper::from_env().is_none());
    }

    #[test]
    fn header_writes_magic_and_version() {
        // Open the file directly (no env) — see `for_test`.
        let path = unique_temp_path("hdr");
        let mut dumper = IterDumper::for_test(&path, "hs071").expect("dumper");
        dumper.write_header(4, 2).unwrap();
        dumper.flush().expect("flush");
        drop(dumper);
        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(&bytes[0..8], MAGIC);
        assert_eq!(&bytes[8..12], &1u32.to_le_bytes()); // version
        assert_eq!(&bytes[12..16], &4u32.to_le_bytes()); // n
        assert_eq!(&bytes[16..20], &2u32.to_le_bytes()); // m
        assert_eq!(&bytes[20..24], &0u32.to_le_bytes()); // nnz_jac
        assert_eq!(&bytes[24..28], &0u32.to_le_bytes()); // nnz_h
        assert_eq!(&bytes[28..32], &5u32.to_le_bytes()); // name_len
        assert_eq!(&bytes[32..37], b"hs071");
        assert_eq!(bytes.len(), 37);
        let _ = std::fs::remove_file(&path);
    }

    /// Smoke test: build an IpoptData/Cq pair, write a record, and
    /// verify the byte count matches FORMAT.md's record size formula.
    /// Computing CQ values requires an Nlp; this test stays at the
    /// vector-write layer rather than wiring a full mock NLP.
    #[test]
    fn iv_dim_matches_record_layout_assumption() {
        let iv = IteratesVector::new(
            dense(4, Some(&[1.0, 2.0, 3.0, 4.0])),
            dense(1, Some(&[0.5])),
            dense(1, Some(&[1.0])),
            dense(1, Some(&[1.0])),
            dense(4, Some(&[1.0, 1.0, 1.0, 1.0])),
            dense(4, Some(&[1.0, 1.0, 1.0, 1.0])),
            dense(1, Some(&[1.0])),
            dense(0, None),
        );
        // hs071 layout per FORMAT.md.
        assert_eq!(iv.x.dim(), 4);
        assert_eq!(iv.v_u.dim(), 0);
        let mut data = IpoptData::new();
        data.set_curr(iv);
        let _h: IpoptDataHandle = Rc::new(RefCell::new(data));
    }
}
