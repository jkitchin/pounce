//! AMPL imported (external) function support via the `funcadd_ASL` ABI.
//!
//! This module implements enough of AMPL's `funcadd.h` ABI to:
//!
//! 1. `dlopen` a user-supplied shared library;
//! 2. resolve the `funcadd_ASL` symbol and call it;
//! 3. receive registration callbacks of the form `Addfunc(name, rfunc, type,
//!    nargs, funcinfo, ae)` and record them;
//! 4. later call back into the registered `rfunc` with an `arglist` to obtain
//!    function values, gradients, and Hessians.
//!
//! The `AmplExports` and `Arglist` struct layouts are taken from
//! AMPL-MP/ASL `funcadd.h`; cross-checked against the ctypes mapping in
//! `pyomo.core.base.external`. Fields we don't populate are left null —
//! Pyomo does the same and it is sufficient for IDAES's Helmholtz library
//! (see issue #15).
//!
//! All unsafe FFI is contained in this module. Public surface is safe.

use std::collections::HashMap;
use std::ffi::{c_char, c_int, c_long, c_void, CStr, CString};
use std::path::Path;
use std::ptr;
use std::sync::{Arc, Mutex, OnceLock};

use libloading::{Library, Symbol};

use crate::nl_reader::{Expr, FuncallArg, ImportedFunc};

/// Resolved AMPL imported function: shared library + registered name.
/// `NlProblem` carries one of these per `ImportedFunc` id when external
/// functions are wired up at problem-build time. The same `Arc<ExternalLibrary>`
/// may be shared across many funcall ids (one library typically registers
/// several functions).
#[derive(Default, Clone)]
pub struct ExternalResolver {
    /// `Funcall { id }` -> (library, registered function name).
    pub funcs_by_id: HashMap<usize, (Arc<ExternalLibrary>, String)>,
}

impl ExternalResolver {
    pub fn is_empty(&self) -> bool {
        self.funcs_by_id.is_empty()
    }

    /// Build a resolver for every `ImportedFunc` declared in the `.nl` file
    /// that is *actually referenced* somewhere in the problem's expressions.
    ///
    /// Library paths are resolved through the `AMPLFUNC` environment variable
    /// (a `\n`-separated list of shared-library paths, matching AMPL/IPOPT
    /// conventions). Each path is loaded once and queried for every name we
    /// need. Returns an error if a referenced name cannot be found in any
    /// listed library, or if `AMPLFUNC` is missing.
    pub fn build_for_problem(
        imported_funcs: &[ImportedFunc],
        referenced_ids: &std::collections::BTreeSet<usize>,
    ) -> Result<Self, String> {
        if referenced_ids.is_empty() {
            return Ok(Self::default());
        }
        let amplfunc = std::env::var("AMPLFUNC").map_err(|_| {
            "problem uses external functions but AMPLFUNC is not set; \
             set AMPLFUNC to a newline-separated list of AMPL shared-library paths"
                .to_string()
        })?;
        let mut libs: Vec<Arc<ExternalLibrary>> = Vec::new();
        for path_str in amplfunc
            .split('\n')
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
        {
            let path = std::path::Path::new(path_str);
            let lib = ExternalLibrary::load(path).map_err(|e| format!("AMPLFUNC: {e}"))?;
            libs.push(Arc::new(lib));
        }

        let mut funcs_by_id: HashMap<usize, (Arc<ExternalLibrary>, String)> = HashMap::new();
        for id in referenced_ids {
            let decl = imported_funcs
                .iter()
                .find(|f| f.id == *id)
                .ok_or_else(|| format!("funcall id {id} has no F<{id}> declaration"))?;
            let found = libs
                .iter()
                .find(|lib| lib.get(&decl.name).is_some())
                .ok_or_else(|| {
                    format!(
                        "external function '{}' (id {}) not found in any library on AMPLFUNC",
                        decl.name, decl.id
                    )
                })?;
            funcs_by_id.insert(*id, (found.clone(), decl.name.clone()));
        }
        Ok(Self { funcs_by_id })
    }
}

/// Walk an `Expr` and collect every funcall id it references (including
/// through CSEs). Used to build an `ExternalResolver` covering exactly the
/// functions a problem actually uses.
pub fn collect_funcall_ids(e: &Expr, out: &mut std::collections::BTreeSet<usize>) {
    match e {
        Expr::Const(_) | Expr::Var(_) => {}
        Expr::Binary(_, a, b) => {
            collect_funcall_ids(a, out);
            collect_funcall_ids(b, out);
        }
        Expr::Unary(_, a) => collect_funcall_ids(a, out),
        Expr::Sum(args) | Expr::MinList(args) | Expr::MaxList(args) => {
            for a in args {
                collect_funcall_ids(a, out);
            }
        }
        Expr::Compare(_, a, b) | Expr::And(a, b) | Expr::Or(a, b) => {
            collect_funcall_ids(a, out);
            collect_funcall_ids(b, out);
        }
        Expr::Not(a) => collect_funcall_ids(a, out),
        Expr::Cond { cond, then_, else_ } => {
            collect_funcall_ids(cond, out);
            collect_funcall_ids(then_, out);
            collect_funcall_ids(else_, out);
        }
        Expr::Cse(body) => collect_funcall_ids(body, out),
        Expr::Funcall { id, args } => {
            out.insert(*id);
            for arg in args {
                if let FuncallArg::Real(e) = arg {
                    collect_funcall_ids(e, out);
                }
            }
        }
    }
}

/// Process-wide lock serialising every call that crosses the AMPL external
/// ABI. Real AMPL libraries (e.g. IDAES general_helmholtz) keep mutable
/// global state (cached parameters, tabulated lookups) and are not safe for
/// concurrent entry. Python's `pyomo.core.base.external` relies on the GIL
/// for the same guarantee.
fn ampl_lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}

/// FUNCADD_TYPE bits (mirrors `funcadd.h`).
pub const FUNCADD_REAL_VALUED: i32 = 0;
/// Set if the function consumes string arguments. Value is still real.
pub const FUNCADD_STRING_ARGS: i32 = 1;
/// Set if the function is allowed to have a variable number of args.
pub const FUNCADD_OUTPUT_ARGS: i32 = 2;
pub const FUNCADD_RANDOM_VALUED: i32 = 4;

/// The `arglist` struct from AMPL's `funcadd.h`. Layout must match exactly.
#[repr(C)]
pub struct Arglist {
    pub n: c_int,               // number of args
    pub nr: c_int,              // number of real input args
    pub at: *mut c_int,         // argument types
    pub ra: *mut f64,           // pure real args (IN/OUT/INOUT)
    pub sa: *mut *const c_char, // symbolic IN args
    pub derivs: *mut f64,       // partial derivatives (if non-null)
    pub hes: *mut f64,          // second partials (if non-null)
    pub dig: *mut c_char,       // skip-derivatives flags
    pub funcinfo: *mut c_void,  // per-function cookie (set by Addfunc)
    pub ae: *mut AmplExports,   // points back at our AmplExports
    pub f: *mut c_void,         // AMPL-internal
    pub tva: *mut c_void,       // AMPL-internal
    pub errmsg: *mut c_char,    // error description set by the function
    pub tmi: *mut c_void,       // Tempmem cookie
    pub private: *mut c_char,
    pub nin: c_int,
    pub nout: c_int,
    pub nsin: c_int,
    pub nsout: c_int,
}

/// Pointer to a user-defined real-valued function, matching
/// `typedef real (*rfunc)(arglist*)`.
pub type Rfunc = unsafe extern "C" fn(*mut Arglist) -> f64;

/// Pointer to the `Addfunc` callback provided by the caller.
pub type AddfuncFn = unsafe extern "C" fn(
    name: *const c_char,
    f: Rfunc,
    ty: c_int,
    nargs: c_int,
    funcinfo: *mut c_void,
    ae: *mut AmplExports,
);

/// Pointer to the `RandSeedSetter` callback.
pub type RandSeedSetter = unsafe extern "C" fn(*mut c_void, std::os::raw::c_ulong);

/// Pointer to the `Addrandinit` callback.
pub type AddrandinitFn =
    unsafe extern "C" fn(ae: *mut AmplExports, setter: RandSeedSetter, v: *mut c_void);

/// Pointer to the `AtReset` callback.
pub type AtResetFn = unsafe extern "C" fn(ae: *mut AmplExports, f: *mut c_void, v: *mut c_void);

/// The `AmplExports` struct from AMPL's `funcadd.h`. Layout must match
/// exactly. Function pointers we don't implement are held as `*mut c_void`
/// (null) — AMPL's ABI does not require a caller to populate them unless the
/// loaded library actually invokes them.
#[repr(C)]
pub struct AmplExports {
    pub std_err: *mut c_void,
    pub addfunc: Option<AddfuncFn>,
    pub asl_date: c_long,
    pub fprintf: *mut c_void,
    pub printf: *mut c_void,
    pub sprintf: *mut c_void,
    pub vfprintf: *mut c_void,
    pub vsprintf: *mut c_void,
    pub strtod: *mut c_void,
    pub crypto: *mut c_void,
    pub asl: *mut c_char,
    pub at_exit: *mut c_void,
    pub at_reset: Option<AtResetFn>,
    pub tempmem: *mut c_void,
    pub add_table_handler: *mut c_void,
    pub private_ae: *mut c_char,
    pub qsortv: *mut c_void,

    pub std_in: *mut c_void,
    pub std_out: *mut c_void,
    pub clearerr: *mut c_void,
    pub fclose: *mut c_void,
    pub fdopen: *mut c_void,
    pub feof: *mut c_void,
    pub ferror: *mut c_void,
    pub fflush: *mut c_void,
    pub fgetc: *mut c_void,
    pub fgets: *mut c_void,
    pub fileno: *mut c_void,
    pub fopen: *mut c_void,
    pub fputc: *mut c_void,
    pub fputs: *mut c_void,
    pub fread: *mut c_void,
    pub freopen: *mut c_void,
    pub fscanf: *mut c_void,
    pub fseek: *mut c_void,
    pub ftell: *mut c_void,
    pub fwrite: *mut c_void,
    pub pclose: *mut c_void,
    pub perror: *mut c_void,
    pub popen: *mut c_void,
    pub puts: *mut c_void,
    pub rewind: *mut c_void,
    pub scanf: *mut c_void,
    pub setbuf: *mut c_void,
    pub setvbuf: *mut c_void,
    pub sscanf: *mut c_void,
    pub tempnam: *mut c_void,
    pub tmpfile: *mut c_void,
    pub tmpnam: *mut c_void,
    pub ungetc: *mut c_void,
    pub ai: *mut c_void,
    pub getenv: *mut c_void,
    pub breakfunc: *mut c_void,
    pub breakarg: *mut c_char,

    // Items available with ASLdate >= 20020501.
    pub snprintf: *mut c_void,
    pub vsnprintf: *mut c_void,

    pub addrand: *mut c_void,
    pub addrandinit: Option<AddrandinitFn>,
}

// SAFETY: AmplExports itself contains only raw pointers and integers. The
// library never reads/writes it from another thread concurrently with us
// (AMPL's model is single-threaded per problem), and we never share it
// across threads. The Send/Sync bounds only matter because we box the
// registry inside Arcs.
unsafe impl Send for AmplExports {}
unsafe impl Sync for AmplExports {}

/// A function registered by a library via `Addfunc`. Mirrors the ASL
/// `FUNCADD_TYPE` bits in `funcadd.h`.
#[derive(Debug, Clone)]
pub struct RegisteredFunc {
    pub name: String,
    pub rfunc: Rfunc,
    /// OR of FUNCADD_TYPE bits.
    pub ty: i32,
    /// Declared arg count. >=0 means exactly that many, <=-1 means "at least
    /// -(nargs+1) args".
    pub nargs: i32,
    /// Cookie set by the library; must be passed through to arglist.funcinfo.
    pub funcinfo: *mut c_void,
}

// SAFETY: funcinfo is an opaque cookie owned by the library. We never
// dereference it; we only pass it back to the library's functions, which
// expect it. No thread-safety contract is violated by sending the struct.
unsafe impl Send for RegisteredFunc {}
unsafe impl Sync for RegisteredFunc {}

impl std::fmt::Debug for ExternalLibrary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExternalLibrary")
            .field("funcs", &self.funcs.keys().collect::<Vec<_>>())
            .finish()
    }
}

/// A loaded external-function library plus its registered functions.
pub struct ExternalLibrary {
    /// Keep the library alive — it owns the code pages the function pointers
    /// reference. Arc so `LoadedExternals` can share it.
    _lib: Arc<Library>,
    /// The AmplExports we handed to `funcadd_ASL`. Must be kept alive (pinned
    /// in a Box) because some libraries may capture its address for later
    /// use (e.g. for `AtReset` bookkeeping).
    _ae: Box<AmplExports>,
    /// Registrations collected during `funcadd_ASL`.
    funcs: HashMap<String, RegisteredFunc>,
}

impl ExternalLibrary {
    /// Open a shared library at `path` and invoke its `funcadd_ASL` entry
    /// point, collecting all functions it registers.
    pub fn load(path: &Path) -> Result<Self, String> {
        // Serialise all ABI crossings: library init code and registration
        // may touch global state that isn't safe under concurrent entry.
        let _guard = ampl_lock().lock().unwrap_or_else(|e| e.into_inner());
        // SAFETY: libloading::Library::new is unsafe because it can run
        // arbitrary initialisers from the shared object. We trust the user's
        // AMPLFUNC path the same way AMPL/IPOPT do.
        let lib = unsafe { Library::new(path) }
            .map_err(|e| format!("failed to open '{}': {}", path.display(), e))?;

        // Resolve `funcadd_ASL`. AMPL's macro `#define funcadd funcadd_ASL`
        // means every conforming library exports this symbol.
        type FuncaddFn = unsafe extern "C" fn(*mut AmplExports);
        let funcadd: Symbol<FuncaddFn> = unsafe { lib.get(b"funcadd_ASL\0") }
            .map_err(|e| format!("no funcadd_ASL in '{}': {}", path.display(), e))?;

        // Build an AmplExports. Most fields null — the library doesn't call
        // them (same assumption Pyomo makes). Only the three hooks we can
        // realistically service are set.
        let mut ae = Box::new(AmplExports {
            std_err: ptr::null_mut(),
            addfunc: Some(trampoline_addfunc),
            // ASLdate >= 20020501 unlocks the SnprintF/VsnprintF slots.
            // Pyomo uses 20160307; mirror that.
            asl_date: 20160307,
            fprintf: ptr::null_mut(),
            printf: ptr::null_mut(),
            sprintf: ptr::null_mut(),
            vfprintf: ptr::null_mut(),
            vsprintf: ptr::null_mut(),
            strtod: ptr::null_mut(),
            crypto: ptr::null_mut(),
            asl: ptr::null_mut(),
            at_exit: ptr::null_mut(),
            at_reset: Some(trampoline_atreset),
            tempmem: ptr::null_mut(),
            add_table_handler: ptr::null_mut(),
            private_ae: ptr::null_mut(),
            qsortv: ptr::null_mut(),
            std_in: ptr::null_mut(),
            std_out: ptr::null_mut(),
            clearerr: ptr::null_mut(),
            fclose: ptr::null_mut(),
            fdopen: ptr::null_mut(),
            feof: ptr::null_mut(),
            ferror: ptr::null_mut(),
            fflush: ptr::null_mut(),
            fgetc: ptr::null_mut(),
            fgets: ptr::null_mut(),
            fileno: ptr::null_mut(),
            fopen: ptr::null_mut(),
            fputc: ptr::null_mut(),
            fputs: ptr::null_mut(),
            fread: ptr::null_mut(),
            freopen: ptr::null_mut(),
            fscanf: ptr::null_mut(),
            fseek: ptr::null_mut(),
            ftell: ptr::null_mut(),
            fwrite: ptr::null_mut(),
            pclose: ptr::null_mut(),
            perror: ptr::null_mut(),
            popen: ptr::null_mut(),
            puts: ptr::null_mut(),
            rewind: ptr::null_mut(),
            scanf: ptr::null_mut(),
            setbuf: ptr::null_mut(),
            setvbuf: ptr::null_mut(),
            sscanf: ptr::null_mut(),
            tempnam: ptr::null_mut(),
            tmpfile: ptr::null_mut(),
            tmpnam: ptr::null_mut(),
            ungetc: ptr::null_mut(),
            ai: ptr::null_mut(),
            getenv: ptr::null_mut(),
            breakfunc: ptr::null_mut(),
            breakarg: ptr::null_mut(),
            snprintf: ptr::null_mut(),
            vsnprintf: ptr::null_mut(),
            addrand: ptr::null_mut(),
            addrandinit: Some(trampoline_addrandinit),
        });

        // Drive registrations into a thread-local sink so the C trampoline
        // has somewhere to deposit them without capturing Rust state.
        REGISTRY_SINK.with(|sink| {
            let mut guard = sink.borrow_mut();
            assert!(
                guard.is_none(),
                "nested ExternalLibrary::load is not supported"
            );
            *guard = Some(HashMap::new());
        });

        // SAFETY: funcadd is a valid C function from the loaded library; we
        // pass it a correctly-shaped AmplExports.
        unsafe { funcadd(ae.as_mut()) };

        let funcs = REGISTRY_SINK
            .with(|sink| sink.borrow_mut().take())
            .unwrap_or_default();

        Ok(ExternalLibrary {
            _lib: Arc::new(lib),
            _ae: ae,
            funcs,
        })
    }

    /// Names of all functions registered by this library.
    pub fn function_names(&self) -> impl Iterator<Item = &str> {
        self.funcs.keys().map(|s| s.as_str())
    }

    /// Look up a registered function by name.
    pub fn get(&self, name: &str) -> Option<&RegisteredFunc> {
        self.funcs.get(name)
    }

    /// Evaluate a registered function with the given positional arguments.
    ///
    /// Arguments are encoded per the AMPL `arglist` ABI: real args are stored
    /// in `ra[]`, string args in `sa[]`, and `at[i]` maps argument position
    /// `i` to either a real-slot index (`at[i] >= 0`) or a string-slot index
    /// (`at[i] < 0`, decoded as `-(at[i]+1)`).
    ///
    /// If `want_derivs` is set, a length-`nr` derivative buffer is allocated
    /// and returned on success. If `want_hes` is set, a length-`nr*(nr+1)/2`
    /// Hessian buffer is also allocated and returned. The library is told to
    /// fill both by the non-null `arglist.derivs` / `arglist.hes` pointers.
    pub fn eval(
        &self,
        name: &str,
        args: &[ExternalArg<'_>],
        want_derivs: bool,
        want_hes: bool,
    ) -> Result<EvalResult, String> {
        let rf = self
            .funcs
            .get(name)
            .ok_or_else(|| format!("no such external function '{name}'"))?;

        // Validate arity against the registered signature.
        let n = args.len() as i32;
        if rf.nargs >= 0 {
            if rf.nargs != n {
                return Err(format!(
                    "external '{name}' expects {} args, got {}",
                    rf.nargs, n
                ));
            }
        } else {
            // Negative: minimum -(nargs+1) args.
            let min_args = -(rf.nargs + 1);
            if n < min_args {
                return Err(format!(
                    "external '{name}' expects at least {min_args} args, got {n}"
                ));
            }
        }

        // Bucket args: build at[], ra[], sa[] in lockstep with their indices.
        let mut at_vec: Vec<c_int> = Vec::with_capacity(args.len());
        let mut ra_vec: Vec<f64> = Vec::new();
        let mut sa_owned: Vec<CString> = Vec::new();
        for a in args {
            match a {
                ExternalArg::Real(x) => {
                    at_vec.push(ra_vec.len() as c_int);
                    ra_vec.push(*x);
                }
                ExternalArg::Str(s) => {
                    let cs = CString::new(*s)
                        .map_err(|_| format!("external '{name}' string arg contains NUL"))?;
                    at_vec.push(-(sa_owned.len() as c_int + 1));
                    sa_owned.push(cs);
                }
            }
        }
        let nr = ra_vec.len() as c_int;
        let sa_ptrs: Vec<*const c_char> = sa_owned.iter().map(|s| s.as_ptr()).collect();

        // If the library declared FUNCADD_STRING_ARGS we let it see sa; if it
        // did not, the library shouldn't be called with strings. Surface that.
        let has_strings = !sa_owned.is_empty();
        if has_strings && (rf.ty & FUNCADD_STRING_ARGS) == 0 {
            return Err(format!(
                "external '{name}' is not declared FUNCADD_STRING_ARGS but was \
                 called with string arguments"
            ));
        }

        // Optional output buffers.
        let mut derivs_buf: Vec<f64> = if want_derivs {
            vec![0.0; nr as usize]
        } else {
            Vec::new()
        };
        let hes_len = if want_hes {
            (nr as usize) * ((nr as usize) + 1) / 2
        } else {
            0
        };
        let mut hes_buf: Vec<f64> = if want_hes {
            vec![0.0; hes_len]
        } else {
            Vec::new()
        };

        // Space for a library-set error message. The ABI lets a library
        // signal an error two ways (see `decode_external_errmsg`): by writing
        // into this buffer, OR — the canonical conforming path — by
        // *reassigning* `arglist.errmsg` to its own string. We seed the field
        // with this buffer's address and remember it so the reassignment is
        // detectable afterwards.
        let mut errmsg_buf: Vec<c_char> = vec![0; 1024];
        let errmsg_orig_ptr = errmsg_buf.as_ptr();

        // Build the arglist. Pointers into Rust-owned buffers are valid for
        // the duration of the call since we hold those Vecs in this stack
        // frame and the callee runs synchronously.
        let mut al = Arglist {
            n,
            nr,
            at: if at_vec.is_empty() {
                ptr::null_mut()
            } else {
                at_vec.as_mut_ptr()
            },
            ra: if ra_vec.is_empty() {
                ptr::null_mut()
            } else {
                ra_vec.as_mut_ptr()
            },
            sa: if sa_ptrs.is_empty() {
                ptr::null_mut()
            } else {
                sa_ptrs.as_ptr() as *mut *const c_char
            },
            derivs: if want_derivs {
                derivs_buf.as_mut_ptr()
            } else {
                ptr::null_mut()
            },
            hes: if want_hes {
                hes_buf.as_mut_ptr()
            } else {
                ptr::null_mut()
            },
            dig: ptr::null_mut(),
            funcinfo: rf.funcinfo,
            // Some libraries read arglist.ae (e.g. to call fprintf); point at
            // the same AmplExports we handed to funcadd_ASL.
            ae: self._ae_ptr(),
            f: ptr::null_mut(),
            tva: ptr::null_mut(),
            errmsg: errmsg_buf.as_mut_ptr(),
            tmi: ptr::null_mut(),
            private: ptr::null_mut(),
            nin: 0,
            nout: 0,
            nsin: 0,
            nsout: 0,
        };

        // SAFETY: rfunc is a valid extern "C" function pointer provided by
        // the loaded library; arglist layout matches funcadd.h exactly.
        // The AMPL lock serialises concurrent entry into the library.
        let _guard = ampl_lock().lock().unwrap_or_else(|e| e.into_inner());
        let value = unsafe { (rf.rfunc)(&mut al as *mut Arglist) };
        drop(_guard);

        // Surface a library-reported error from *either* ABI channel: the
        // reassigned `arglist.errmsg` pointer (the conforming path) or our
        // pre-pointed buffer. Checking only the buffer would miss every
        // library that does `al->Errmsg = "...";`, silently consuming garbage.
        // SAFETY: `al.errmsg` is either our zeroed NUL-terminated buffer or a
        // C string the library assigned; both are valid to read as a CStr.
        if let Some(msg) =
            unsafe { decode_external_errmsg(al.errmsg, errmsg_orig_ptr, errmsg_buf[0]) }
        {
            return Err(format!("external '{name}' reported: {msg}"));
        }

        Ok(EvalResult {
            value,
            derivs: if want_derivs { Some(derivs_buf) } else { None },
            hessian: if want_hes { Some(hes_buf) } else { None },
        })
    }

    // Raw mutable pointer to the owned AmplExports. Used when building an
    // arglist so the library can call back through the same table it was
    // registered with. The Box is pinned for the lifetime of self.
    fn _ae_ptr(&self) -> *mut AmplExports {
        // Cast away the const; we never mutate the AmplExports ourselves.
        (&*self._ae as *const AmplExports) as *mut AmplExports
    }
}

/// One positional argument to an external function.
#[derive(Debug, Clone, Copy)]
pub enum ExternalArg<'a> {
    Real(f64),
    Str(&'a str),
}

/// Return value from [`ExternalLibrary::eval`].
#[derive(Debug, Clone)]
pub struct EvalResult {
    /// Function value.
    pub value: f64,
    /// `df/dx_i` for each real argument, in `ra[]` order, if `want_derivs`.
    pub derivs: Option<Vec<f64>>,
    /// Packed upper-triangular Hessian in AMPL's convention,
    /// `hes[i + j*(j+1)/2]` for `0 <= i <= j < nr`, if `want_hes`.
    pub hessian: Option<Vec<f64>>,
}

/// Decode an external function's error signal after its `rfunc` returns.
///
/// The AMPL `funcadd` ABI lets a library report an error two ways:
///
/// 1. **Reassign** `arglist.errmsg` to its own (usually static) C string —
///    `al->Errmsg = "T out of range";`. This is the conforming path used by
///    real libraries (e.g. IDAES Helmholtz on out-of-domain evals). The
///    caller's pre-pointed buffer is left untouched.
/// 2. Write a string into the buffer the caller pointed `errmsg` at before the
///    call.
///
/// We seed `arglist.errmsg` with our buffer's address (`orig_buf_ptr`). After
/// the call: if the field no longer equals that address (and is non-null) the
/// library reassigned it → read from the new pointer; otherwise fall back to
/// the buffer when its first byte is non-zero. Returns `None` when neither
/// channel carries a message. Checking only the buffer (the prior behavior)
/// silently dropped every channel-1 error and let the IPM consume NaN/garbage
/// f/∇f/∇²f.
///
/// # Safety
/// `errmsg_field` (when reassigned) and `orig_buf_ptr` must each point at a
/// readable NUL-terminated C string for the duration of the read.
unsafe fn decode_external_errmsg(
    errmsg_field: *const c_char,
    orig_buf_ptr: *const c_char,
    buf_first: c_char,
) -> Option<String> {
    if !errmsg_field.is_null() && errmsg_field != orig_buf_ptr {
        // Channel 1: the library reassigned the pointer to its own string.
        // SAFETY: caller guarantees `errmsg_field` is a NUL-terminated string.
        return Some(
            unsafe { CStr::from_ptr(errmsg_field) }
                .to_string_lossy()
                .into_owned(),
        );
    }
    if buf_first != 0 {
        // Channel 2: the library wrote into the caller-provided buffer.
        // SAFETY: caller guarantees `orig_buf_ptr` is a NUL-terminated string.
        return Some(
            unsafe { CStr::from_ptr(orig_buf_ptr) }
                .to_string_lossy()
                .into_owned(),
        );
    }
    None
}

// ---------------------------------------------------------------------------
// Registration trampoline.
//
// `funcadd_ASL` can call Addfunc multiple times (once per registered name).
// Rust closures can't be converted to `extern "C"` function pointers, so we
// route each call through a free function that deposits into a thread-local
// sink populated by `ExternalLibrary::load`.
// ---------------------------------------------------------------------------

thread_local! {
    static REGISTRY_SINK: std::cell::RefCell<Option<HashMap<String, RegisteredFunc>>> =
        std::cell::RefCell::new(None);
}

/// C-callable trampoline that receives Addfunc calls from the shared library.
unsafe extern "C" fn trampoline_addfunc(
    name: *const c_char,
    f: Rfunc,
    ty: c_int,
    nargs: c_int,
    funcinfo: *mut c_void,
    _ae: *mut AmplExports,
) {
    if name.is_null() {
        return;
    }
    // SAFETY: AMPL guarantees name is a NUL-terminated C string.
    let cname = unsafe { CStr::from_ptr(name) };
    let name_str = match cname.to_str() {
        Ok(s) => s.to_owned(),
        Err(_) => return, // non-UTF8 name — skip; real libs use ASCII.
    };
    REGISTRY_SINK.with(|sink| {
        if let Some(map) = sink.borrow_mut().as_mut() {
            map.insert(
                name_str.clone(),
                RegisteredFunc {
                    name: name_str,
                    rfunc: f,
                    ty: ty as i32,
                    nargs: nargs as i32,
                    funcinfo,
                },
            );
        }
    });
}

/// Stub — some libraries ask us to register an AtReset callback. Pyomo logs a
/// warning and does nothing. We do the same.
unsafe extern "C" fn trampoline_atreset(_ae: *mut AmplExports, _f: *mut c_void, _v: *mut c_void) {
    tracing::debug!("external library registered an AtReset callback; ignoring");
}

/// Stub — invoked by libraries that use random-valued externals. We just
/// seed with 1 (matches Pyomo's default; no randomness in KKT paths).
unsafe extern "C" fn trampoline_addrandinit(
    _ae: *mut AmplExports,
    setter: RandSeedSetter,
    v: *mut c_void,
) {
    unsafe { setter(v, 1) };
}

#[cfg(test)]
mod tests {
    use super::*;

    fn idaes_dylib() -> Option<std::path::PathBuf> {
        let home = std::env::var_os("HOME")?;
        let p = std::path::PathBuf::from(home).join(".idaes/bin/general_helmholtz_external.dylib");
        if p.exists() {
            Some(p)
        } else {
            None
        }
    }

    fn idaes_params_dir() -> Option<String> {
        let home = std::env::var_os("HOME")?;
        let p = std::path::PathBuf::from(home).join(
            "Dropbox/uv/.venv/lib/python3.12/site-packages/idaes/\
             models/properties/general_helmholtz/components/parameters/",
        );
        if p.exists() {
            p.to_str().map(|s| s.to_owned())
        } else {
            None
        }
    }

    /// Opening the IDAES Helmholtz dylib (when present locally) should
    /// surface the three functions used by the issue #15 fixture.
    #[test]
    fn load_idaes_helmholtz_dylib_registers_known_functions() {
        let Some(path) = idaes_dylib() else {
            eprintln!("skipping: IDAES dylib not present");
            return;
        };

        let lib = ExternalLibrary::load(&path).expect("load should succeed");
        let names: Vec<String> = lib.function_names().map(|s| s.to_owned()).collect();

        for required in &["vf_hp", "h_liq_hp", "h_vap_hp"] {
            assert!(
                names.iter().any(|n| n == required),
                "expected {required} in registered names: {names:?}"
            );
        }
    }

    /// Evaluate vf_hp at the NL fixture's initial guess. We don't assert the
    /// exact numeric value (that's an IDAES invariant, not a ripopt one), but
    /// the return value must be finite and the call must not set errmsg.
    #[test]
    fn eval_vf_hp_at_fixture_initial_point() {
        let Some(path) = idaes_dylib() else {
            eprintln!("skipping: IDAES dylib not present");
            return;
        };
        let Some(params_dir) = idaes_params_dir() else {
            eprintln!("skipping: IDAES parameters directory not present");
            return;
        };

        let lib = ExternalLibrary::load(&path).expect("load");
        // Fixture initial guess: h = 1878.71 kJ/kg-scaled, p = 101.325 kPa
        // (the scaled values actually passed through the v3/v4 slots are
        // 1878.71 * 0.0555... and 101325 * 0.001 respectively; using raw
        // values here, the function should still return a finite number).
        let args = [
            ExternalArg::Str("h2o"),
            ExternalArg::Real(1878.71 * 0.055508472036052976),
            ExternalArg::Real(101325.0 * 0.001),
            ExternalArg::Str(&params_dir),
        ];
        let res = lib.eval("vf_hp", &args, false, false).expect("eval");
        assert!(
            res.value.is_finite(),
            "vf_hp returned non-finite value {}",
            res.value
        );
    }

    /// Same call path, but asking for first derivatives. derivs must be a
    /// length-2 buffer (nr=2) of finite values.
    #[test]
    fn eval_vf_hp_with_derivatives() {
        let Some(path) = idaes_dylib() else {
            eprintln!("skipping: IDAES dylib not present");
            return;
        };
        let Some(params_dir) = idaes_params_dir() else {
            eprintln!("skipping: IDAES parameters directory not present");
            return;
        };

        let lib = ExternalLibrary::load(&path).expect("load");
        let args = [
            ExternalArg::Str("h2o"),
            ExternalArg::Real(1878.71 * 0.055508472036052976),
            ExternalArg::Real(101325.0 * 0.001),
            ExternalArg::Str(&params_dir),
        ];
        let res = lib.eval("vf_hp", &args, true, false).expect("eval");
        let derivs = res.derivs.expect("derivs requested");
        assert_eq!(derivs.len(), 2, "nr=2 reals -> 2 derivatives");
        for (i, d) in derivs.iter().enumerate() {
            assert!(d.is_finite(), "derivs[{i}] = {d} not finite");
        }
    }

    /// Also request the packed Hessian. For nr=2 reals, that's 3 entries
    /// (H00, H01, H11) in AMPL's packed upper-triangular layout.
    #[test]
    fn eval_vf_hp_with_hessian() {
        let Some(path) = idaes_dylib() else {
            eprintln!("skipping: IDAES dylib not present");
            return;
        };
        let Some(params_dir) = idaes_params_dir() else {
            eprintln!("skipping: IDAES parameters directory not present");
            return;
        };

        let lib = ExternalLibrary::load(&path).expect("load");
        let args = [
            ExternalArg::Str("h2o"),
            ExternalArg::Real(1878.71 * 0.055508472036052976),
            ExternalArg::Real(101325.0 * 0.001),
            ExternalArg::Str(&params_dir),
        ];
        let res = lib.eval("vf_hp", &args, true, true).expect("eval");
        let hes = res.hessian.expect("hessian requested");
        assert_eq!(hes.len(), 3, "nr=2 -> packed Hessian of length 3");
        for (i, h) in hes.iter().enumerate() {
            assert!(h.is_finite(), "hes[{i}] = {h} not finite");
        }
    }

    // --- H5: errmsg detection across both funcadd ABI channels ---

    /// A conforming `rfunc` that signals an error the canonical AMPL way: by
    /// **reassigning** `al->Errmsg` to its own static C string (leaving any
    /// caller-provided buffer untouched), and returning NaN like an
    /// out-of-domain evaluation.
    unsafe extern "C" fn rfunc_reassigns_errmsg(al: *mut Arglist) -> f64 {
        static MSG: &[u8] = b"T out of range\0";
        // SAFETY: `al` is a valid, exclusively-borrowed Arglist for the call.
        unsafe {
            (*al).errmsg = MSG.as_ptr() as *mut c_char;
        }
        f64::NAN
    }

    /// Build an `Arglist` with every pointer null except `errmsg`. Sufficient
    /// for a `rfunc` that only manipulates the error channel.
    fn null_arglist(errmsg: *mut c_char) -> Arglist {
        Arglist {
            n: 1,
            nr: 1,
            at: ptr::null_mut(),
            ra: ptr::null_mut(),
            sa: ptr::null_mut(),
            derivs: ptr::null_mut(),
            hes: ptr::null_mut(),
            dig: ptr::null_mut(),
            funcinfo: ptr::null_mut(),
            ae: ptr::null_mut(),
            f: ptr::null_mut(),
            tva: ptr::null_mut(),
            errmsg,
            tmi: ptr::null_mut(),
            private: ptr::null_mut(),
            nin: 0,
            nout: 0,
            nsin: 0,
            nsout: 0,
        }
    }

    /// End-to-end over the real `Arglist` + a real `extern "C"` call: a library
    /// that reports an error by reassigning `al->Errmsg` (channel 1) must be
    /// detected. Pre-fix, `eval` only inspected the caller buffer — which a
    /// reassigning library never touches — so the error was invisible and the
    /// IPM consumed the NaN return as a valid value.
    #[test]
    fn reassigned_errmsg_pointer_is_detected_end_to_end() {
        let mut errmsg_buf: Vec<c_char> = vec![0; 1024];
        let orig_ptr = errmsg_buf.as_ptr();
        let mut al = null_arglist(errmsg_buf.as_mut_ptr());

        // SAFETY: the rfunc matches the ABI and only writes `al.errmsg`.
        let v = unsafe { rfunc_reassigns_errmsg(&mut al) };
        assert!(v.is_nan(), "the failing eval returned NaN");

        // A reassigning library leaves the caller buffer zeroed, so the old
        // `errmsg_buf[0] != 0` check (the bug) saw nothing.
        assert_eq!(
            errmsg_buf[0], 0,
            "a reassigning library must not touch the caller buffer"
        );

        // The fixed decode reads the reassigned pointer and surfaces the error.
        let decoded = unsafe { decode_external_errmsg(al.errmsg, orig_ptr, errmsg_buf[0]) };
        assert_eq!(
            decoded.as_deref(),
            Some("T out of range"),
            "the reassigned errmsg pointer must be surfaced as an error"
        );
    }

    /// The buffer channel (a library that writes into the caller buffer) and
    /// the no-error cases still behave correctly.
    #[test]
    fn decode_external_errmsg_buffer_and_none_channels() {
        // Channel 2: library wrote a string into the caller buffer.
        let mut buf: Vec<c_char> = vec![0; 16];
        for (i, b) in b"bad input".iter().enumerate() {
            buf[i] = *b as c_char;
        }
        let orig = buf.as_ptr();
        let decoded = unsafe { decode_external_errmsg(orig, orig, buf[0]) };
        assert_eq!(decoded.as_deref(), Some("bad input"));

        // No error: field still points at the (zeroed) buffer.
        let zero: Vec<c_char> = vec![0; 16];
        let z = zero.as_ptr();
        assert_eq!(unsafe { decode_external_errmsg(z, z, zero[0]) }, None);

        // No error via an explicitly NULL field (some libraries zero it).
        assert_eq!(unsafe { decode_external_errmsg(ptr::null(), z, 0) }, None);
    }
}
