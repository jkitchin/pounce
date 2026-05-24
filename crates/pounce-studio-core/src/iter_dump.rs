//! POUNCEIT v1 binary iter-dump parser.
//!
//! Spec: `tools/iter-dump/FORMAT.md`. Writer:
//! `crates/pounce-algorithm/src/iter_dump.rs`. Reference Python parser:
//! `tools/iter-dump/dump_inspect.py`.
//!
//! Format outline (all multi-byte values are little-endian; vectors are
//! `u32 len + len*8` bytes of f64):
//!
//! ```text
//! header: "POUNCEIT" | u32 version | u32 n | u32 m | u32 nnz_jac
//!       | u32 nnz_h | u32 name_len | [u8; name_len]
//! per iter:
//!   scalar block (120 bytes): u32 iter, u32 status, 14 * f64
//!   8 vectors:                x, s, y_c, y_d, z_L, z_U, v_L, v_U
//!   filter block:             u32 filter_count, [(f64, f64); count]
//! ```
//!
//! The parser is byte-driven and copy-free for vectors of length 0 (it
//! still allocates a `Vec<f64>` for non-empty vectors so the public API
//! stays simple). For very large traces, see [`IterDumpTrace::lazy_iter`]
//! which yields one record at a time without retaining prior records.

use serde::{Deserialize, Serialize};

use crate::report::Error;

/// ASCII magic bytes identifying a POUNCEIT v1 stream.
pub const MAGIC: &[u8; 8] = b"POUNCEIT";
/// The only format version this parser accepts.
pub const FORMAT_VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IterDumpHeader {
    pub format_version: u32,
    pub n: u32,
    pub m: u32,
    pub nnz_jac: u32,
    pub nnz_h: u32,
    pub name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IterDumpRecord {
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IterDumpTrace {
    pub header: IterDumpHeader,
    pub records: Vec<IterDumpRecord>,
}

impl IterDumpTrace {
    /// Parse a complete POUNCEIT v1 stream from a byte slice.
    ///
    /// Reads the header, then loops reading iteration records until the
    /// stream is exhausted. Any truncation, version mismatch, or bad
    /// magic returns [`Error::IterDump`].
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, Error> {
        let mut cur = Cursor::new(bytes);
        let header = parse_header(&mut cur)?;
        let mut records = Vec::new();
        while cur.remaining() > 0 {
            records.push(parse_record(&mut cur)?);
        }
        Ok(Self { header, records })
    }

    /// Lazy iterator over records. Each call to `next` parses one
    /// record, so memory stays bounded by the largest single record.
    /// Useful when the trace is hundreds of MB.
    pub fn lazy_iter(bytes: &[u8]) -> Result<(IterDumpHeader, LazyRecords<'_>), Error> {
        let mut cur = Cursor::new(bytes);
        let header = parse_header(&mut cur)?;
        Ok((header, LazyRecords { cur }))
    }
}

/// Cursor over a byte slice. We do not use `std::io::Cursor` because
/// the library is `no_fs` (and `no_std`-friendly at the lib level —
/// the bin target is what touches std::io).
struct Cursor<'a> {
    buf: &'a [u8],
    off: usize,
}

impl<'a> Cursor<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, off: 0 }
    }

    fn remaining(&self) -> usize {
        self.buf.len() - self.off
    }

    fn read(&mut self, n: usize) -> Result<&'a [u8], Error> {
        if self.off + n > self.buf.len() {
            return Err(Error::IterDump(format!(
                "truncated: wanted {n} bytes at offset {}, file is {} bytes",
                self.off,
                self.buf.len()
            )));
        }
        let out = &self.buf[self.off..self.off + n];
        self.off += n;
        Ok(out)
    }

    fn read_u32(&mut self) -> Result<u32, Error> {
        let bytes = self.read(4)?;
        Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    fn read_f64(&mut self) -> Result<f64, Error> {
        let bytes = self.read(8)?;
        Ok(f64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]))
    }

    fn read_vec(&mut self) -> Result<Vec<f64>, Error> {
        let len = self.read_u32()? as usize;
        if len == 0 {
            return Ok(Vec::new());
        }
        let bytes = self.read(len * 8)?;
        let mut out = Vec::with_capacity(len);
        for chunk in bytes.chunks_exact(8) {
            out.push(f64::from_le_bytes([
                chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
            ]));
        }
        Ok(out)
    }
}

fn parse_header(cur: &mut Cursor<'_>) -> Result<IterDumpHeader, Error> {
    let magic = cur.read(8)?;
    if magic != MAGIC {
        return Err(Error::IterDump(format!(
            "bad magic: expected {MAGIC:?}, got {magic:?}",
        )));
    }
    let format_version = cur.read_u32()?;
    if format_version != FORMAT_VERSION {
        return Err(Error::IterDump(format!(
            "unsupported format_version {format_version} (only {FORMAT_VERSION} known)",
        )));
    }
    let n = cur.read_u32()?;
    let m = cur.read_u32()?;
    let nnz_jac = cur.read_u32()?;
    let nnz_h = cur.read_u32()?;
    let name_len = cur.read_u32()? as usize;
    let name_bytes = cur.read(name_len)?;
    let name = std::str::from_utf8(name_bytes)
        .map_err(|e| Error::IterDump(format!("name is not UTF-8: {e}")))?
        .to_string();
    Ok(IterDumpHeader {
        format_version,
        n,
        m,
        nnz_jac,
        nnz_h,
        name,
    })
}

fn parse_record(cur: &mut Cursor<'_>) -> Result<IterDumpRecord, Error> {
    let iter = cur.read_u32()?;
    let status = cur.read_u32()?;
    let mu = cur.read_f64()?;
    let tau = cur.read_f64()?;
    let alpha_pr = cur.read_f64()?;
    let alpha_du = cur.read_f64()?;
    let delta_x = cur.read_f64()?;
    let delta_s = cur.read_f64()?;
    let delta_c = cur.read_f64()?;
    let delta_d = cur.read_f64()?;
    let inf_pr = cur.read_f64()?;
    let inf_du = cur.read_f64()?;
    let constr_viol = cur.read_f64()?;
    let dual_inf = cur.read_f64()?;
    let complementarity = cur.read_f64()?;
    let f = cur.read_f64()?;

    let x = cur.read_vec()?;
    let s = cur.read_vec()?;
    let y_c = cur.read_vec()?;
    let y_d = cur.read_vec()?;
    let z_l = cur.read_vec()?;
    let z_u = cur.read_vec()?;
    let v_l = cur.read_vec()?;
    let v_u = cur.read_vec()?;

    let filter_count = cur.read_u32()? as usize;
    let mut filter = Vec::with_capacity(filter_count);
    for _ in 0..filter_count {
        let theta = cur.read_f64()?;
        let phi = cur.read_f64()?;
        filter.push((theta, phi));
    }

    Ok(IterDumpRecord {
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

pub struct LazyRecords<'a> {
    cur: Cursor<'a>,
}

impl Iterator for LazyRecords<'_> {
    type Item = Result<IterDumpRecord, Error>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.cur.remaining() == 0 {
            return None;
        }
        Some(parse_record(&mut self.cur))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal POUNCEIT byte stream programmatically and
    /// confirm the parser reproduces the same scalars and vectors.
    /// We don't depend on pounce-algorithm here.
    fn synth_trace() -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(MAGIC);
        buf.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
        buf.extend_from_slice(&4u32.to_le_bytes()); // n
        buf.extend_from_slice(&2u32.to_le_bytes()); // m
        buf.extend_from_slice(&0u32.to_le_bytes()); // nnz_jac
        buf.extend_from_slice(&0u32.to_le_bytes()); // nnz_h
        buf.extend_from_slice(&5u32.to_le_bytes()); // name_len
        buf.extend_from_slice(b"hs071");

        // One iteration record.
        buf.extend_from_slice(&0u32.to_le_bytes()); // iter = 0
        buf.extend_from_slice(&0u32.to_le_bytes()); // status = 0
        for v in [0.1, 0.99, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.5, 2.0, 0.5, 2.0, 0.25, 17.0_f64] {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        // x (n=4)
        buf.extend_from_slice(&4u32.to_le_bytes());
        for v in [1.0_f64, 5.0, 5.0, 1.0] {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        // s, y_c, y_d, z_L, z_U, v_L, v_U — all length 1 except v_U (0)
        for vals in [
            vec![0.5_f64],
            vec![1.0],
            vec![1.0],
            vec![1.0],
            vec![1.0],
            vec![1.0],
            vec![],
        ] {
            buf.extend_from_slice(&(vals.len() as u32).to_le_bytes());
            for v in vals {
                buf.extend_from_slice(&v.to_le_bytes());
            }
        }
        // filter_count = 0
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf
    }

    #[test]
    fn parses_header_and_one_record() {
        let bytes = synth_trace();
        let trace = IterDumpTrace::from_bytes(&bytes).expect("parse");
        assert_eq!(trace.header.format_version, 1);
        assert_eq!(trace.header.n, 4);
        assert_eq!(trace.header.m, 2);
        assert_eq!(trace.header.name, "hs071");
        assert_eq!(trace.records.len(), 1);
        let rec = &trace.records[0];
        assert_eq!(rec.iter, 0);
        assert_eq!(rec.mu, 0.1);
        assert_eq!(rec.x, vec![1.0, 5.0, 5.0, 1.0]);
        assert_eq!(rec.v_u, Vec::<f64>::new());
    }

    #[test]
    fn rejects_bad_magic() {
        let mut bytes = synth_trace();
        bytes[0] = b'X';
        let err = IterDumpTrace::from_bytes(&bytes).expect_err("should fail");
        assert!(matches!(err, Error::IterDump(_)));
    }

    #[test]
    fn rejects_unsupported_version() {
        let mut bytes = synth_trace();
        bytes[8..12].copy_from_slice(&99u32.to_le_bytes());
        let err = IterDumpTrace::from_bytes(&bytes).expect_err("should fail");
        let msg = format!("{err}");
        assert!(msg.contains("99"), "got: {msg}");
    }

    #[test]
    fn truncated_stream_errors_cleanly() {
        let bytes = synth_trace();
        let err = IterDumpTrace::from_bytes(&bytes[..40]).expect_err("should fail");
        let msg = format!("{err}");
        assert!(msg.contains("truncated"), "got: {msg}");
    }

    #[test]
    fn lazy_iter_returns_same_records() {
        let bytes = synth_trace();
        let (header, mut iter) = IterDumpTrace::lazy_iter(&bytes).expect("hdr");
        assert_eq!(header.name, "hs071");
        let first = iter.next().expect("one rec").expect("ok");
        assert_eq!(first.iter, 0);
        assert!(iter.next().is_none());
    }
}
