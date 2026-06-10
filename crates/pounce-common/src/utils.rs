//! Numerical and timing utilities.
//!
//! Mirrors `Common/IpUtils.{hpp,cpp}`. The PRNG is a portable
//! reimplementation of POSIX `drand48` (48-bit LCG with the
//! glibc-default seed) so output matches a `drand48`-built upstream
//! Ipopt deterministically. Replace with libc's `drand48` later if a
//! particular platform's bit-equivalence target requires it.

use crate::types::Number;
use std::cell::Cell;
use std::sync::OnceLock;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

/// Equivalent to `Ipopt::IsFiniteNumber`. Returns false for NaN/±∞.
#[inline]
pub fn is_finite_number(val: Number) -> bool {
    val.is_finite()
}

/// Equivalent to `Ipopt::Compare_le(lhs, rhs, BasVal)` — relaxed `<=`
/// with a tolerance proportional to `|BasVal|`. Threshold matches
/// upstream's `(10 * machine_epsilon * Max(1, |BasVal|))`.
#[inline]
pub fn compare_le(lhs: Number, rhs: Number, bas_val: Number) -> bool {
    let tol = 10.0 * Number::EPSILON * bas_val.abs().max(1.0);
    lhs - rhs <= tol
}

/// Equivalent to `Ipopt::IpRandom01`. 48-bit LCG matching POSIX
/// `drand48`: X_{n+1} = (0x5DEECE66D · X_n + 0xB) mod 2^48,
/// returning the high-order 48 bits as a `f64` in [0, 1).
pub fn ip_random_01() -> Number {
    LCG_STATE.with(|s| {
        let mut x = s.get();
        x = x.wrapping_mul(LCG_A).wrapping_add(LCG_C) & LCG_MASK;
        s.set(x);
        // Top 32 bits → 53-bit mantissa portion, exactly as drand48
        // converts. Match glibc's `erand48`:
        //   r = ((double)(x >> 16)) * 2^-32 + ((x & 0xffff) << 4) * 2^-48 ... etc.
        // The simplest reformulation that matches: split into 16-bit
        // chunks and accumulate.
        let x0 = (x & 0xFFFF) as f64;
        let x1 = ((x >> 16) & 0xFFFF) as f64;
        let x2 = ((x >> 32) & 0xFFFF) as f64;
        x2 / 65536.0 + x1 / (65536.0 * 65536.0) + x0 / (65536.0 * 65536.0 * 65536.0)
    })
}

/// Reset the PRNG to glibc's default seed. Equivalent to
/// `Ipopt::IpResetRandom01`.
pub fn ip_reset_random_01() {
    LCG_STATE.with(|s| s.set(LCG_DEFAULT_SEED));
}

const LCG_A: u64 = 0x5DEECE66D;
const LCG_C: u64 = 0xB;
const LCG_MASK: u64 = (1 << 48) - 1;
const LCG_DEFAULT_SEED: u64 = 0x1234_ABCD_330E;

thread_local! {
    static LCG_STATE: Cell<u64> = const { Cell::new(LCG_DEFAULT_SEED) };
}

/// Wallclock time in seconds since first call. Equivalent to
/// `Ipopt::WallclockTime`.
pub fn wallclock_time() -> Number {
    static EPOCH: OnceLock<Instant> = OnceLock::new();
    let e = EPOCH.get_or_init(Instant::now);
    e.elapsed().as_secs_f64()
}

/// CPU time in seconds. Equivalent to `Ipopt::CpuTime`.
///
/// On Unix this returns the process's accumulated **user** CPU time via
/// `getrusage(RUSAGE_SELF).ru_utime`, exactly matching upstream Ipopt's
/// `CpuTime()` (`src/Common/IpUtils.cpp`). This is what `max_cpu_time`
/// is meant to bound — a busy solve accrues CPU time at roughly the wall
/// rate, but time spent blocked/sleeping does not count.
///
/// std exposes no portable CPU-time API, so non-Unix targets (Windows)
/// fall back to wallclock. That mirrors upstream too: its Windows path
/// uses `clock()`, which on the MSVC runtime measures elapsed real time
/// rather than CPU time, so the two are already equivalent there.
pub fn cpu_time() -> Number {
    #[cfg(unix)]
    {
        // SAFETY: `getrusage` only writes into the provided `rusage` out-param
        // and reads no global state; a zeroed `rusage` is a valid input.
        let mut usage: libc::rusage = unsafe { core::mem::zeroed() };
        if unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut usage) } == 0 {
            return usage.ru_utime.tv_sec as Number + 1.0e-6 * usage.ru_utime.tv_usec as Number;
        }
        // getrusage failing is exceedingly rare (EFAULT/EINVAL only); degrade
        // to wallclock rather than panicking in a timing helper.
        wallclock_time()
    }
    #[cfg(not(unix))]
    {
        wallclock_time()
    }
}

/// System time in seconds since UNIX epoch. Equivalent to
/// `Ipopt::SysTime`.
pub fn sys_time() -> Number {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

/// Equivalent to `Ipopt::ComputeMemIncrease`. Updates `len` to a new
/// length consistent with `recommended`/`min` while clamping to
/// `T::MAX`. Returns `Err` if even the maximum representable size
/// would not be an increase.
pub fn compute_mem_increase_i64(
    len: &mut i64,
    recommended: f64,
    min: i64,
    context: &str,
) -> Result<(), String> {
    if recommended >= i64::MAX as f64 {
        if *len < i64::MAX {
            *len = i64::MAX;
            Ok(())
        } else {
            Err(format!(
                "Cannot allocate more than {} bytes for {} due to integer-type limit",
                (i64::MAX as f64) * 8.0,
                context
            ))
        }
    } else {
        *len = min.max(recommended as i64);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finite_check() {
        assert!(is_finite_number(0.0));
        assert!(is_finite_number(1e300));
        assert!(!is_finite_number(f64::NAN));
        assert!(!is_finite_number(f64::INFINITY));
    }

    #[test]
    fn compare_le_tolerates_eps() {
        let v = 1.0;
        let near = v + 5.0 * Number::EPSILON;
        assert!(compare_le(near, v, v));
    }

    #[test]
    fn random_01_in_range_and_deterministic() {
        ip_reset_random_01();
        let a: Vec<f64> = (0..16).map(|_| ip_random_01()).collect();
        ip_reset_random_01();
        let b: Vec<f64> = (0..16).map(|_| ip_random_01()).collect();
        assert_eq!(a, b);
        for v in a {
            assert!((0.0..1.0).contains(&v), "{v}");
        }
    }

    #[test]
    fn wallclock_monotonic() {
        let a = wallclock_time();
        let b = wallclock_time();
        assert!(b >= a);
    }

    #[cfg(unix)]
    #[test]
    fn cpu_time_excludes_sleep_but_counts_compute() {
        use std::hint::black_box;
        use std::thread::sleep;
        use std::time::Duration;

        // (1) Sleeping consumes no user CPU time, so `cpu_time()` must lag
        // wallclock across a sleep — the defining property of `max_cpu_time`.
        // Before the L5 fix `cpu_time()` was a `wallclock_time()` alias, so
        // the two advanced in lockstep and the gap below was ~0.
        let cpu0 = cpu_time();
        let wall0 = wallclock_time();
        sleep(Duration::from_millis(300));
        let wall_slept = wallclock_time() - wall0;
        let cpu_slept = cpu_time() - cpu0;
        assert!(
            wall_slept - cpu_slept > 0.1,
            "cpu_time advanced {:.3}s across a {:.3}s sleep; it must measure \
             CPU, not wallclock (wall−cpu gap was only {:.3}s)",
            cpu_slept,
            wall_slept,
            wall_slept - cpu_slept
        );

        // (2) ...but a busy loop *does* accrue CPU time, confirming the clock
        // is live (guards against a degenerate constant implementation).
        let cpu1 = cpu_time();
        let mut acc = 0u64;
        for i in 0..50_000_000u64 {
            acc = acc.wrapping_add(i ^ (i << 1));
        }
        black_box(acc);
        let cpu_busy = cpu_time() - cpu1;
        assert!(
            cpu_busy > 0.0,
            "cpu_time did not advance across a busy loop (got {:.6}s)",
            cpu_busy
        );
    }
}
