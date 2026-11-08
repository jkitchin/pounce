//! The `struct_size` contract of the extensible output structs (`pounce.h`,
//! "Extensible output structs"), checked one branch at a time on a getter
//! that has something to report:
//!
//! * (a) `struct_size = 0` (the `= {0}` initializer), and a garbage
//!   negative value, are refused: `FALSE`, every byte untouched;
//! * (b) `struct_size = sizeof(T) - 1` is refused the same way (the floor is
//!   exactly this release's size, not "anything positive");
//! * (c) `struct_size = sizeof(T)` fills the struct, `struct_size` itself
//!   unchanged;
//! * (d) `struct_size = sizeof(T) + 64`, in a buffer that large, succeeds
//!   with the same values and leaves every byte past `sizeof(T)` as it was:
//!   a getter never writes past this release's size, so a caller built
//!   against a newer header keeps its newer fields. A getter that wrote the
//!   whole caller-sized region would pass (c) and fail here.

#![allow(dead_code)]

use pounce_cinterface::{
    Index, PounceComplementaritySolve, PounceComplementarityStats, PounceQpCertificate,
    PounceQpResiduals,
};
use std::fmt::Debug;
use std::mem::size_of;

/// Bytes past `sizeof(T)` in the oversized buffer of case (d).
pub const PAST: usize = 64;
/// A byte pattern no getter writes as a whole field.
const SENTINEL: u8 = 0xA5;

/// The four extensible structs, with their `struct_size` reachable
/// generically.
pub trait Extensible: Copy + Debug {
    fn struct_size(&mut self) -> &mut Index;
}
macro_rules! extensible {
    ($($t:ty),*) => {$(
        impl Extensible for $t {
            fn struct_size(&mut self) -> &mut Index {
                &mut self.struct_size
            }
        }
    )*};
}
extensible!(
    PounceComplementarityStats,
    PounceComplementaritySolve,
    PounceQpResiduals,
    PounceQpCertificate
);

/// `sizeof(T)` as the `struct_size` a caller passes.
pub fn size_of_t<T>() -> Index {
    Index::try_from(size_of::<T>()).unwrap()
}

/// Call `get` on an 8-aligned buffer of `sizeof(T) + PAST` sentinel bytes
/// whose `struct_size` (offset 0) is `claimed`; the getter's return and
/// every byte of the buffer afterwards.
fn call<T>(claimed: Index, get: &mut impl FnMut(*mut T) -> u8) -> (u8, Vec<u8>, Vec<u8>) {
    let len = size_of::<T>() + PAST;
    let mut buf = vec![u64::from_ne_bytes([SENTINEL; 8]); len.div_ceil(8)];
    let p = buf.as_mut_ptr().cast::<u8>();
    unsafe { std::ptr::write(p.cast::<Index>(), claimed) };
    let before = unsafe { std::slice::from_raw_parts(p, len) }.to_vec();
    let ok = get(p.cast::<T>());
    let after = unsafe { std::slice::from_raw_parts(p, len) }.to_vec();
    (ok, before, after)
}

/// Read the `T` at the front of a buffer `call` returned.
fn read<T: Copy>(bytes: &[u8]) -> T {
    assert!(bytes.len() >= size_of::<T>());
    unsafe { std::ptr::read_unaligned(bytes.as_ptr().cast::<T>()) }
}

fn claimed_of<T: Extensible>(mut v: T) -> Index {
    *v.struct_size()
}

/// The value with `struct_size` zeroed, as text (`NaN` reads `NaN`), so that
/// two fills can be compared field for field whatever their padding holds.
fn fields<T: Extensible>(mut v: T) -> String {
    *v.struct_size() = 0;
    format!("{v:?}")
}

/// Run cases (a)–(d) on `get`, which must be a call that succeeds for a
/// correctly sized struct; returns case (c)'s value.
pub fn check_struct_size_contract<T: Extensible>(
    name: &str,
    mut get: impl FnMut(*mut T) -> u8,
) -> T {
    let size = size_of::<T>();
    // (a) and (b): refused, untouched — 0, a garbage negative, one short.
    for claimed in [
        0,
        -1,
        Index::from_ne_bytes([SENTINEL; 4]),
        size_of_t::<T>() - 1,
    ] {
        let (ok, before, after) = call::<T>(claimed, &mut get);
        assert_eq!(ok, 0, "{name}: struct_size = {claimed} was accepted");
        assert_eq!(
            after, before,
            "{name}: struct_size = {claimed} wrote the buffer"
        );
    }
    // (c): exactly this release's size.
    let (ok, _, after) = call::<T>(size_of_t::<T>(), &mut get);
    assert_eq!(ok, 1, "{name}: struct_size = sizeof was refused");
    let exact: T = read(&after);
    assert_eq!(
        claimed_of(exact),
        size_of_t::<T>(),
        "{name}: the getter wrote struct_size"
    );
    assert!(
        after[size..].iter().all(|&b| b == SENTINEL),
        "{name}: wrote past sizeof"
    );
    // (d): a newer caller's larger struct.
    let claimed = size_of_t::<T>() + PAST as Index;
    let (ok, _, after) = call::<T>(claimed, &mut get);
    assert_eq!(ok, 1, "{name}: struct_size = sizeof + {PAST} was refused");
    let big: T = read(&after);
    assert_eq!(
        claimed_of(big),
        claimed,
        "{name}: the getter wrote struct_size"
    );
    assert_eq!(
        fields(big),
        fields(exact),
        "{name}: a larger struct_size changed the values"
    );
    let past: Vec<usize> = (size..size + PAST)
        .filter(|&i| after[i] != SENTINEL)
        .collect();
    assert!(
        past.is_empty(),
        "{name}: struct_size = sizeof + {PAST} wrote bytes past sizeof at offsets {past:?}"
    );
    exact
}
