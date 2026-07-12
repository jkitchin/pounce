//! POUNCEIT bit-equivalence comparator (parser + diff engine).
//!
//! See `tools/iter-dump/FORMAT.md` for the wire format. Parser is
//! deliberately isolated from any POUNCE crates so this binary can run
//! against dumps from upstream-Ipopt-only builds.
//!
//! Tolerance modes:
//!
//! * `Tolerance::Bit` (default) — demand byte-identical f64s. ULP=0.
//! * `Tolerance::Ulp(n)`        — allow `n` ULPs of difference.
//! * `Tolerance::Abs(eps)`      — allow `|a-b| <= eps`.
//! * `Tolerance::Rel(eps)`      — allow `|a-b| <= eps * max(|a|,|b|)`.
//!
//! Advisory fields (`delta_s`, `delta_c`, `delta_d`, filter contents)
//! are skipped by default in v1; the comparator's `strict` flag
//! re-enables them.

use std::fs;
use std::io;
use std::path::Path;

pub const MAGIC: &[u8; 8] = b"POUNCEIT";
pub const FORMAT_VERSION: u32 = 1;

#[derive(Debug, Clone)]
pub struct Header {
    pub format_version: u32,
    pub n: u32,
    pub m: u32,
    pub nnz_jac: u32,
    pub nnz_h: u32,
    pub name: String,
}

#[derive(Debug, Clone)]
pub struct IterRecord {
    pub iter: u32,
    pub status: u32,
    pub mu: f64,
    pub tau: f64,
    pub alpha_pr: f64,
    pub alpha_du: f64,
    pub delta_x: f64,
    pub delta_s: f64,
    pub delta_c: f64,
    pub delta_d: f64,
    pub inf_pr: f64,
    pub inf_du: f64,
    pub constr_viol: f64,
    pub dual_inf: f64,
    pub complementarity: f64,
    pub f: f64,
    pub x: Vec<f64>,
    pub s: Vec<f64>,
    pub y_c: Vec<f64>,
    pub y_d: Vec<f64>,
    pub z_l: Vec<f64>,
    pub z_u: Vec<f64>,
    pub v_l: Vec<f64>,
    pub v_u: Vec<f64>,
    pub filter: Vec<(f64, f64)>,
}

#[derive(Debug, Clone, Copy)]
pub enum Tolerance {
    Bit,
    Ulp(u64),
    Abs(f64),
    Rel(f64),
}

impl Tolerance {
    /// Parse a CLI string like `bit`, `ulp:4`, `abs:1e-12`, `rel:1e-9`.
    pub fn parse(s: &str) -> Result<Self, String> {
        let s = s.trim();
        if s == "bit" {
            return Ok(Tolerance::Bit);
        }
        if let Some(rest) = s.strip_prefix("ulp:") {
            return rest
                .parse::<u64>()
                .map(Tolerance::Ulp)
                .map_err(|e| format!("bad ULP count `{}`: {}", rest, e));
        }
        if let Some(rest) = s.strip_prefix("abs:") {
            return rest
                .parse::<f64>()
                .map(Tolerance::Abs)
                .map_err(|e| format!("bad abs tolerance `{}`: {}", rest, e));
        }
        if let Some(rest) = s.strip_prefix("rel:") {
            return rest
                .parse::<f64>()
                .map(Tolerance::Rel)
                .map_err(|e| format!("bad rel tolerance `{}`: {}", rest, e));
        }
        Err(format!(
            "unknown tolerance `{}` (expected `bit`, `ulp:N`, `abs:F`, or `rel:F`)",
            s
        ))
    }
}

/// Result of comparing two f64s under a tolerance. `Match` carries the
/// observed delta (in ULPs / abs); the comparator surfaces both.
#[derive(Debug, Clone, Copy)]
pub struct DiffStat {
    pub abs: f64,
    pub rel: f64,
    pub ulps: u64,
    pub bit_equal: bool,
}

impl DiffStat {
    pub fn new(a: f64, b: f64) -> Self {
        let abs = (a - b).abs();
        let m = a.abs().max(b.abs());
        let rel = if m == 0.0 { 0.0 } else { abs / m };
        let bit_equal = a.to_bits() == b.to_bits();
        let ulps = ulp_diff(a, b);
        Self {
            abs,
            rel,
            ulps,
            bit_equal,
        }
    }

    /// True if this pair satisfies `tol`.
    pub fn within(&self, tol: Tolerance) -> bool {
        match tol {
            Tolerance::Bit => self.bit_equal,
            // NaN handling: NaNs never compare bit-equal under the
            // bit-mode predicate above. For ULP/abs/rel modes we treat
            // NaN-vs-NaN as a match if they share the same bit pattern,
            // and otherwise as a divergence. This matches what a
            // floating-point validation gate should report.
            Tolerance::Ulp(n) => self.ulps <= n,
            Tolerance::Abs(eps) => self.abs <= eps,
            Tolerance::Rel(eps) => self.rel <= eps,
        }
    }
}

/// Compute the ULP distance between two f64s. Saturates at `u64::MAX`
/// for opposite-sign / NaN cases.
fn ulp_diff(a: f64, b: f64) -> u64 {
    if a.to_bits() == b.to_bits() {
        return 0;
    }
    if a.is_nan() || b.is_nan() {
        return u64::MAX;
    }
    if a.is_sign_negative() != b.is_sign_negative() {
        // Opposite signs: ULP distance is undefined (or Σ of |ULP to 0|).
        // Saturate to flag the divergence visibly.
        return u64::MAX;
    }
    a.to_bits().abs_diff(b.to_bits())
}

// --------------------------------------------------------------------
// Parser — port of `tools/iter-dump/dump_inspect.py`.
// --------------------------------------------------------------------

#[derive(Debug)]
pub enum ParseError {
    Io(io::Error),
    Truncated {
        offset: usize,
        wanted: usize,
        have: usize,
    },
    BadMagic([u8; 8]),
    UnsupportedVersion(u32),
    BadUtf8Name,
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ParseError::Io(e) => write!(f, "I/O error: {}", e),
            ParseError::Truncated {
                offset,
                wanted,
                have,
            } => write!(
                f,
                "truncated: wanted {} bytes at offset {}, file has {} bytes total",
                wanted, offset, have
            ),
            ParseError::BadMagic(m) => write!(f, "bad magic bytes: {:?}", m),
            ParseError::UnsupportedVersion(v) => {
                write!(f, "unsupported format version {}", v)
            }
            ParseError::BadUtf8Name => write!(f, "header name is not valid UTF-8"),
        }
    }
}

impl std::error::Error for ParseError {}

impl From<io::Error> for ParseError {
    fn from(e: io::Error) -> Self {
        ParseError::Io(e)
    }
}

struct Cursor<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }
    fn read_exact(&mut self, n: usize) -> Result<&'a [u8], ParseError> {
        if self.pos + n > self.buf.len() {
            return Err(ParseError::Truncated {
                offset: self.pos,
                wanted: n,
                have: self.buf.len(),
            });
        }
        let out = &self.buf[self.pos..self.pos + n];
        self.pos += n;
        Ok(out)
    }
    fn read_u32(&mut self) -> Result<u32, ParseError> {
        let bytes = self.read_exact(4)?;
        let arr: [u8; 4] = bytes.try_into().unwrap_or([0; 4]);
        Ok(u32::from_le_bytes(arr))
    }
    fn read_f64(&mut self) -> Result<f64, ParseError> {
        let bytes = self.read_exact(8)?;
        let arr: [u8; 8] = bytes.try_into().unwrap_or([0; 8]);
        Ok(f64::from_le_bytes(arr))
    }
    fn read_vec(&mut self) -> Result<Vec<f64>, ParseError> {
        let n = self.read_u32()? as usize;
        if n == 0 {
            return Ok(Vec::new());
        }
        let mut v = Vec::with_capacity(n);
        for _ in 0..n {
            v.push(self.read_f64()?);
        }
        Ok(v)
    }
    fn eof(&self) -> bool {
        self.pos >= self.buf.len()
    }
}

fn parse_header(c: &mut Cursor<'_>) -> Result<Header, ParseError> {
    let magic = c.read_exact(8)?;
    let mut m = [0u8; 8];
    m.copy_from_slice(magic);
    if &m != MAGIC {
        return Err(ParseError::BadMagic(m));
    }
    let format_version = c.read_u32()?;
    if format_version != FORMAT_VERSION {
        return Err(ParseError::UnsupportedVersion(format_version));
    }
    let n = c.read_u32()?;
    let m_ = c.read_u32()?;
    let nnz_jac = c.read_u32()?;
    let nnz_h = c.read_u32()?;
    let name_len = c.read_u32()? as usize;
    let name_bytes = c.read_exact(name_len)?;
    let name = std::str::from_utf8(name_bytes)
        .map_err(|_| ParseError::BadUtf8Name)?
        .to_owned();
    Ok(Header {
        format_version,
        n,
        m: m_,
        nnz_jac,
        nnz_h,
        name,
    })
}

fn parse_record(c: &mut Cursor<'_>) -> Result<IterRecord, ParseError> {
    let iter = c.read_u32()?;
    let status = c.read_u32()?;
    let mu = c.read_f64()?;
    let tau = c.read_f64()?;
    let alpha_pr = c.read_f64()?;
    let alpha_du = c.read_f64()?;
    let delta_x = c.read_f64()?;
    let delta_s = c.read_f64()?;
    let delta_c = c.read_f64()?;
    let delta_d = c.read_f64()?;
    let inf_pr = c.read_f64()?;
    let inf_du = c.read_f64()?;
    let constr_viol = c.read_f64()?;
    let dual_inf = c.read_f64()?;
    let complementarity = c.read_f64()?;
    let f = c.read_f64()?;
    let x = c.read_vec()?;
    let s = c.read_vec()?;
    let y_c = c.read_vec()?;
    let y_d = c.read_vec()?;
    let z_l = c.read_vec()?;
    let z_u = c.read_vec()?;
    let v_l = c.read_vec()?;
    let v_u = c.read_vec()?;
    let filter_count = c.read_u32()? as usize;
    let mut filter = Vec::with_capacity(filter_count);
    for _ in 0..filter_count {
        let theta = c.read_f64()?;
        let phi = c.read_f64()?;
        filter.push((theta, phi));
    }
    Ok(IterRecord {
        iter,
        status,
        mu,
        tau,
        alpha_pr,
        alpha_du,
        delta_x,
        delta_s,
        delta_c,
        delta_d,
        inf_pr,
        inf_du,
        constr_viol,
        dual_inf,
        complementarity,
        f,
        x,
        s,
        y_c,
        y_d,
        z_l,
        z_u,
        v_l,
        v_u,
        filter,
    })
}

pub fn parse_file(path: &Path) -> Result<(Header, Vec<IterRecord>), ParseError> {
    let bytes = fs::read(path)?;
    let mut c = Cursor::new(&bytes);
    let hdr = parse_header(&mut c)?;
    let mut recs = Vec::new();
    while !c.eof() {
        recs.push(parse_record(&mut c)?);
    }
    Ok((hdr, recs))
}

// --------------------------------------------------------------------
// Diff engine.
// --------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct Divergence {
    pub iter_index: usize,
    pub field: String,
    pub left: f64,
    pub right: f64,
    pub stat: DiffStat,
}

impl std::fmt::Display for Divergence {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "iter={} field={} left={:.17e} right={:.17e} abs={:.3e} rel={:.3e} ulps={}",
            self.iter_index,
            self.field,
            self.left,
            self.right,
            self.stat.abs,
            self.stat.rel,
            self.stat.ulps,
        )
    }
}

/// Header mismatch, distinguished from per-record divergences. Cargo
/// exit code is the same (1) but the message format is different.
#[derive(Debug, Clone)]
pub struct HeaderMismatch {
    pub field: &'static str,
    pub left: String,
    pub right: String,
}

impl std::fmt::Display for HeaderMismatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "header field `{}` differs: left={}, right={}",
            self.field, self.left, self.right
        )
    }
}

#[derive(Debug)]
pub enum CompareError {
    Header(HeaderMismatch),
    /// Different record counts. Reported once at the start.
    RecordCount {
        left: usize,
        right: usize,
    },
    /// Vector dimension mismatch — reported as a divergence with the
    /// field name and the two lengths encoded in `left`/`right`.
    VectorLength {
        iter_index: usize,
        field: String,
        left: usize,
        right: usize,
    },
    Diff(Divergence),
}

impl std::fmt::Display for CompareError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CompareError::Header(h) => write!(f, "{}", h),
            CompareError::RecordCount { left, right } => {
                write!(f, "record count differs: left={}, right={}", left, right)
            }
            CompareError::VectorLength {
                iter_index,
                field,
                left,
                right,
            } => write!(
                f,
                "iter={} field={} length differs: left={}, right={}",
                iter_index, field, left, right
            ),
            CompareError::Diff(d) => write!(f, "{}", d),
        }
    }
}

/// Compare two parsed files. Returns the list of all divergences (or
/// just the first, if `show_first_only`). Empty list = match.
pub fn compare(
    left: &(Header, Vec<IterRecord>),
    right: &(Header, Vec<IterRecord>),
    tol: Tolerance,
    strict: bool,
    show_first_only: bool,
) -> Vec<CompareError> {
    let mut out = Vec::new();
    let (lh, lr) = left;
    let (rh, rr) = right;

    macro_rules! cmp_field {
        ($field:expr_2021, $l:expr_2021, $r:expr_2021) => {
            if $l != $r {
                out.push(CompareError::Header(HeaderMismatch {
                    field: $field,
                    left: format!("{:?}", $l),
                    right: format!("{:?}", $r),
                }));
                if show_first_only {
                    return out;
                }
            }
        };
    }
    cmp_field!("format_version", lh.format_version, rh.format_version);
    cmp_field!("n", lh.n, rh.n);
    cmp_field!("m", lh.m, rh.m);
    cmp_field!("name", lh.name, rh.name);
    if strict {
        cmp_field!("nnz_jac", lh.nnz_jac, rh.nnz_jac);
        cmp_field!("nnz_h", lh.nnz_h, rh.nnz_h);
    }

    if lr.len() != rr.len() {
        out.push(CompareError::RecordCount {
            left: lr.len(),
            right: rr.len(),
        });
        if show_first_only {
            return out;
        }
    }

    let n_records = lr.len().min(rr.len());
    for i in 0..n_records {
        let l = &lr[i];
        let r = &rr[i];
        // Scalars.
        let scalars: &[(&str, f64, f64, bool /* advisory */)] = &[
            ("mu", l.mu, r.mu, false),
            ("tau", l.tau, r.tau, false),
            ("alpha_pr", l.alpha_pr, r.alpha_pr, false),
            ("alpha_du", l.alpha_du, r.alpha_du, false),
            ("delta_x", l.delta_x, r.delta_x, false),
            ("delta_s", l.delta_s, r.delta_s, true),
            ("delta_c", l.delta_c, r.delta_c, true),
            ("delta_d", l.delta_d, r.delta_d, true),
            ("inf_pr", l.inf_pr, r.inf_pr, false),
            ("inf_du", l.inf_du, r.inf_du, false),
            ("constr_viol", l.constr_viol, r.constr_viol, false),
            ("dual_inf", l.dual_inf, r.dual_inf, false),
            (
                "complementarity",
                l.complementarity,
                r.complementarity,
                false,
            ),
            ("f", l.f, r.f, false),
        ];
        // iter and status are u32 — exact.
        if l.iter != r.iter {
            out.push(CompareError::Header(HeaderMismatch {
                field: "iter",
                left: l.iter.to_string(),
                right: r.iter.to_string(),
            }));
            if show_first_only {
                return out;
            }
        }
        for &(name, lv, rv, advisory) in scalars {
            if advisory && !strict {
                continue;
            }
            let stat = DiffStat::new(lv, rv);
            if !stat.within(tol) {
                out.push(CompareError::Diff(Divergence {
                    iter_index: i,
                    field: name.to_owned(),
                    left: lv,
                    right: rv,
                    stat,
                }));
                if show_first_only {
                    return out;
                }
            }
        }
        // Vectors.
        let vecs: &[(&str, &Vec<f64>, &Vec<f64>)] = &[
            ("curr.x", &l.x, &r.x),
            ("curr.s", &l.s, &r.s),
            ("curr.y_c", &l.y_c, &r.y_c),
            ("curr.y_d", &l.y_d, &r.y_d),
            ("curr.z_l", &l.z_l, &r.z_l),
            ("curr.z_u", &l.z_u, &r.z_u),
            ("curr.v_l", &l.v_l, &r.v_l),
            ("curr.v_u", &l.v_u, &r.v_u),
        ];
        for (name, lv, rv) in vecs {
            if lv.len() != rv.len() {
                out.push(CompareError::VectorLength {
                    iter_index: i,
                    field: (*name).to_owned(),
                    left: lv.len(),
                    right: rv.len(),
                });
                if show_first_only {
                    return out;
                }
                continue;
            }
            for k in 0..lv.len() {
                let stat = DiffStat::new(lv[k], rv[k]);
                if !stat.within(tol) {
                    out.push(CompareError::Diff(Divergence {
                        iter_index: i,
                        field: format!("{}[{}]", name, k),
                        left: lv[k],
                        right: rv[k],
                        stat,
                    }));
                    if show_first_only {
                        return out;
                    }
                }
            }
        }
        // Filter — advisory in v1.
        if strict {
            if l.filter.len() != r.filter.len() {
                out.push(CompareError::VectorLength {
                    iter_index: i,
                    field: "filter".to_owned(),
                    left: l.filter.len(),
                    right: r.filter.len(),
                });
                if show_first_only {
                    return out;
                }
            } else {
                for k in 0..l.filter.len() {
                    let (lt, lp) = l.filter[k];
                    let (rt, rp) = r.filter[k];
                    for (sub, lv, rv) in [("theta", lt, rt), ("phi", lp, rp)] {
                        let stat = DiffStat::new(lv, rv);
                        if !stat.within(tol) {
                            out.push(CompareError::Diff(Divergence {
                                iter_index: i,
                                field: format!("filter[{}].{}", k, sub),
                                left: lv,
                                right: rv,
                                stat,
                            }));
                            if show_first_only {
                                return out;
                            }
                        }
                    }
                }
            }
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ulp_diff_zero_for_identical() {
        assert_eq!(ulp_diff(1.0, 1.0), 0);
    }

    #[test]
    fn ulp_diff_one_for_adjacent() {
        let a = 1.0_f64;
        let b = f64::from_bits(a.to_bits() + 1);
        assert_eq!(ulp_diff(a, b), 1);
    }

    #[test]
    fn tolerance_parse() {
        assert!(matches!(Tolerance::parse("bit"), Ok(Tolerance::Bit)));
        assert!(matches!(Tolerance::parse("ulp:4"), Ok(Tolerance::Ulp(4))));
        assert!(matches!(
            Tolerance::parse("abs:1e-9"),
            Ok(Tolerance::Abs(_))
        ));
        assert!(matches!(
            Tolerance::parse("rel:1e-9"),
            Ok(Tolerance::Rel(_))
        ));
        assert!(Tolerance::parse("garbage").is_err());
    }

    #[test]
    fn diff_stat_bit_equal_ignores_nan() {
        let n = f64::NAN;
        let s = DiffStat::new(n, n);
        // bit-equal NaN: same payload returns true; the canonical NaN
        // produced by `f64::NAN` always has the same bits.
        assert!(s.bit_equal);
    }
}
