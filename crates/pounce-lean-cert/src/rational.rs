//! Exact rationals over ℚ for the Lean certificate.
//!
//! The certificate has **no floats anywhere** (schema rule §2). Every numeric
//! quantity is an exact rational serialized as `{ "num": "<int>", "den": "<int>" }`
//! with `num`/`den` decimal *strings* (JSON numbers cannot hold arbitrary
//! precision), reduced, and `den > 0`.
//!
//! The enabling fact: every finite `f64` is exactly a dyadic rational `m·2^e`,
//! so [`Rat::from_f64`] is **lossless** — it never rounds. Witness arithmetic
//! (LDLᵀ, the KKT solve) is then carried out exactly over [`num_rational::BigRational`].

use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{One, Zero};
use serde::de::{self, Visitor};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::fmt;

/// An exact rational. Wraps [`BigRational`], which keeps the value reduced with a
/// positive denominator, and serializes as `{ "num": "…", "den": "…" }`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rat(pub BigRational);

/// Conversion failures: the certificate cannot represent these, so the caller
/// must error out rather than emit something that will not verify.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RatError {
    /// `±inf` or `NaN` reached rational conversion (a bound sentinel should have
    /// been handled by [`Bound`] *before* this point).
    NotFinite,
}

impl fmt::Display for RatError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RatError::NotFinite => write!(f, "value is not finite (inf/NaN cannot be a rational)"),
        }
    }
}

impl std::error::Error for RatError {}

impl Rat {
    /// Lossless `f64 → ℚ`. Returns [`RatError::NotFinite`] for `±inf`/`NaN`.
    ///
    /// Decomposes the IEEE-754 bits into `sign · mantissa · 2^exp` (the standard
    /// `integer_decode`, bias 1023 + 52 = 1075) and builds the exact dyadic
    /// rational; [`BigRational::new`] reduces it, so the denominator is the
    /// largest power of two needed and the fraction is in lowest terms.
    pub fn from_f64(x: f64) -> Result<Rat, RatError> {
        if !x.is_finite() {
            return Err(RatError::NotFinite);
        }
        if x == 0.0 {
            return Ok(Rat(BigRational::zero()));
        }
        let bits = x.to_bits();
        let negative = (bits >> 63) == 1;
        let exp_field = ((bits >> 52) & 0x7ff) as i64;
        // Subnormals (exp_field == 0) have no implicit leading 1; normals do.
        let mantissa: u64 = if exp_field == 0 {
            (bits & 0x000f_ffff_ffff_ffff) << 1
        } else {
            (bits & 0x000f_ffff_ffff_ffff) | 0x0010_0000_0000_0000
        };
        let exp = exp_field - 1075; // value = ±mantissa · 2^exp
        let mut num = BigInt::from(mantissa);
        if negative {
            num = -num;
        }
        let ratio = if exp >= 0 {
            BigRational::new(num * pow2(exp as u64), BigInt::one())
        } else {
            BigRational::new(num, pow2((-exp) as u64))
        };
        Ok(Rat(ratio))
    }

    /// Construct from an integer numerator/denominator (panics-free; reduces).
    pub fn new(num: i64, den: i64) -> Rat {
        Rat(BigRational::new(BigInt::from(num), BigInt::from(den)))
    }

    /// The exact integer zero, `0/1`.
    pub fn zero() -> Rat {
        Rat(BigRational::zero())
    }

    /// Borrow the underlying [`BigRational`] for exact arithmetic.
    pub fn inner(&self) -> &BigRational {
        &self.0
    }
}

/// `2^k` as a [`BigInt`].
fn pow2(k: u64) -> BigInt {
    BigInt::one() << k
}

impl Serialize for Rat {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct;
        let mut st = serializer.serialize_struct("Rat", 2)?;
        st.serialize_field("num", &self.0.numer().to_string())?;
        st.serialize_field("den", &self.0.denom().to_string())?;
        st.end()
    }
}

impl<'de> Deserialize<'de> for Rat {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        struct Raw {
            num: String,
            den: String,
        }
        let raw = Raw::deserialize(deserializer)?;
        let num: BigInt = raw
            .num
            .parse()
            .map_err(|_| de::Error::custom("`num` is not a decimal integer"))?;
        let den: BigInt = raw
            .den
            .parse()
            .map_err(|_| de::Error::custom("`den` is not a decimal integer"))?;
        if den.is_zero() {
            return Err(de::Error::custom("`den` must be nonzero"));
        }
        Ok(Rat(BigRational::new(num, den)))
    }
}

/// A bound or coefficient slot that may be infinite. Finite values serialize as a
/// rational object; the infinities serialize as the string sentinels `"-inf"` /
/// `"+inf"` (schema rule §2).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Bound {
    NegInf,
    Finite(Rat),
    PosInf,
}

impl Bound {
    /// Map an `f64` bound to a [`Bound`], turning `±inf` into the sentinels and
    /// converting every finite value losslessly.
    pub fn from_f64(x: f64) -> Result<Bound, RatError> {
        if x == f64::NEG_INFINITY {
            Ok(Bound::NegInf)
        } else if x == f64::INFINITY {
            Ok(Bound::PosInf)
        } else if x.is_nan() {
            Err(RatError::NotFinite)
        } else {
            Ok(Bound::Finite(Rat::from_f64(x)?))
        }
    }

    /// The finite rational, if this bound is finite.
    pub fn finite(&self) -> Option<&Rat> {
        match self {
            Bound::Finite(r) => Some(r),
            _ => None,
        }
    }
}

impl Serialize for Bound {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            Bound::NegInf => serializer.serialize_str("-inf"),
            Bound::PosInf => serializer.serialize_str("+inf"),
            Bound::Finite(r) => r.serialize(serializer),
        }
    }
}

impl<'de> Deserialize<'de> for Bound {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct BoundVisitor;
        impl<'de> Visitor<'de> for BoundVisitor {
            type Value = Bound;
            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("a rational object or the string \"-inf\"/\"+inf\"")
            }
            fn visit_str<E: de::Error>(self, v: &str) -> Result<Bound, E> {
                match v {
                    "-inf" => Ok(Bound::NegInf),
                    "+inf" => Ok(Bound::PosInf),
                    other => Err(de::Error::custom(format!(
                        "unexpected bound sentinel {other:?} (want \"-inf\"/\"+inf\")"
                    ))),
                }
            }
            fn visit_map<A: de::MapAccess<'de>>(self, map: A) -> Result<Bound, A::Error> {
                let r = Rat::deserialize(de::value::MapAccessDeserializer::new(map))?;
                Ok(Bound::Finite(r))
            }
        }
        deserializer.deserialize_any(BoundVisitor)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn ser(r: &Rat) -> String {
        serde_json::to_string(r).unwrap()
    }

    #[test]
    fn half_is_lossless_and_reduced() {
        let r = Rat::from_f64(0.5).unwrap();
        assert_eq!(ser(&r), r#"{"num":"1","den":"2"}"#);
    }

    #[test]
    fn two_is_integer() {
        let r = Rat::from_f64(2.0).unwrap();
        assert_eq!(ser(&r), r#"{"num":"2","den":"1"}"#);
    }

    #[test]
    fn zero_normalizes_to_zero_over_one() {
        let r = Rat::from_f64(0.0).unwrap();
        assert_eq!(ser(&r), r#"{"num":"0","den":"1"}"#);
        // negative zero too
        let r = Rat::from_f64(-0.0).unwrap();
        assert_eq!(ser(&r), r#"{"num":"0","den":"1"}"#);
    }

    #[test]
    fn negative_and_thirds_roundtrip() {
        // -7/2 is dyadic and exact.
        let r = Rat::from_f64(-3.5).unwrap();
        assert_eq!(ser(&r), r#"{"num":"-7","den":"2"}"#);
        // 0.1 is NOT dyadic; conversion is still lossless (huge denominator) and
        // must round-trip exactly through f64.
        let r = Rat::from_f64(0.1).unwrap();
        let back: f64 = {
            use num_traits::ToPrimitive;
            r.inner().to_f64().unwrap()
        };
        assert_eq!(back, 0.1);
    }

    #[test]
    fn infinities_rejected_for_rat_but_ok_for_bound() {
        assert_eq!(Rat::from_f64(f64::INFINITY), Err(RatError::NotFinite));
        assert_eq!(Rat::from_f64(f64::NAN), Err(RatError::NotFinite));
        assert_eq!(Bound::from_f64(f64::INFINITY).unwrap(), Bound::PosInf);
        assert_eq!(Bound::from_f64(f64::NEG_INFINITY).unwrap(), Bound::NegInf);
    }

    #[test]
    fn bound_serialization() {
        assert_eq!(serde_json::to_string(&Bound::PosInf).unwrap(), r#""+inf""#);
        assert_eq!(serde_json::to_string(&Bound::NegInf).unwrap(), r#""-inf""#);
        assert_eq!(
            serde_json::to_string(&Bound::Finite(Rat::new(1, 1))).unwrap(),
            r#"{"num":"1","den":"1"}"#
        );
    }

    #[test]
    fn bound_roundtrip() {
        for b in [Bound::NegInf, Bound::PosInf, Bound::Finite(Rat::new(-3, 4))] {
            let s = serde_json::to_string(&b).unwrap();
            let back: Bound = serde_json::from_str(&s).unwrap();
            assert_eq!(b, back);
        }
    }

    #[test]
    fn rat_roundtrip_through_strings() {
        let r = Rat::new(6, 8); // reduces to 3/4
        let s = serde_json::to_string(&r).unwrap();
        assert_eq!(s, r#"{"num":"3","den":"4"}"#);
        let back: Rat = serde_json::from_str(&s).unwrap();
        assert_eq!(r, back);
    }
}
