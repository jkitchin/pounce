// POUNCE as a CasADi `Nlpsol` plugin.
//
// Registers `casadi_register_nlpsol_pounce`, so once
// `libcasadi_nlpsol_pounce.so` is on CasADi's plugin search path a model
// solves with `nlpsol('S', 'pounce', nlp, opts)` or
// `opti.solver('pounce')`, exactly like the bundled `ipopt` plugin.
//
// The plugin is a thin shim: CasADi's oracle functions are wired into
// POUNCE through `pounce.h`, the Ipopt-3.14-compatible C API that
// `libpounce_cinterface` exports. Everything CasADi layers on top of a
// plugin — solution-map derivatives, `Opti`, bound consistency — comes
// from the `Nlpsol` base class and needs nothing from us.
//
// Build: see the Makefile in this directory. Two constraints are not
// optional and are easy to get wrong:
//   * the internal headers must come from the *matching* CasADi source
//     tree (the pip wheel ships only public headers);
//   * the libstdc++ ABI and the `-D` set must match the CasADi build —
//     for the pip wheels that means `-D_GLIBCXX_USE_CXX11_ABI=0`.

#include "casadi/core/nlpsol_impl.hpp"
#include "casadi/core/convexify.hpp"
#include "convexify_compat.hpp"
#include "pounce_runtime.hpp"
#include <cmath>
#include <cstring>
#include <ostream>
#include <string>
#include <vector>

// Ipopt's C API takes non-const `char*` for option keywords and values;
// POUNCE keeps bit-for-bit parity with it, so the casts live here.
#define CC(s) const_cast<char*>(s)

extern "C" {
#include "pounce.h"
}

namespace casadi {

  class PounceInterface;

  struct PounceMemory : public NlpsolMemory {
    const PounceInterface* self = nullptr;
    IpoptProblem prob = nullptr;
    std::vector<double> xk, gk, lam_g, z_L, z_U, xl, xu, gl, gu;
    double obj = 0;
    int return_status = 0;
    int iter = 0;
    double t_solve = 0;
    // per-iteration trace, mirroring casadi's ipopt `iterations` stats
    std::vector<double> inf_pr, inf_du, mu_trace, d_norm, obj_trace,
                        alpha_pr, alpha_du, regularization_size;
    std::vector<casadi_int> ls_trials, alg_mod;
    /// Final KKT errors, read off the problem before it is freed —
    /// `FreeIpoptProblem` takes the accessors with it.
    double final_inf_pr = 0, final_inf_du = 0, final_compl_inf = 0;
    /// Whether the last solve's report reached disk (gh#644). Reported
    /// through `stats()` so a script can check it without scraping the
    /// warning, and absent when no report was asked for.
    bool report_written = false;
    /// Linear-solver post-mortem, harvested at the same moment.
    PounceLinearSolverStats linsol{};
    bool linsol_valid = false;
    /// Restoration-phase activity, likewise.
    ipindex resto_calls = 0, resto_inner = 0, resto_outer = 0;
    double resto_secs = 0;
    /// Finite-difference Hessian census. `fd_pattern` stays -1 on every
    /// solve that was not `hessian_approximation=finite-difference`,
    /// which is what keeps `stats()["fd_hessian"]` absent rather than
    /// present-and-zero for the modes it does not describe.
    ipindex fd_pattern = -1, fd_nnz = 0, fd_n = 0, fd_groups = 0,
            fd_rho_max = 0, fd_fell_back = 0, fd_clique_widened = 0;
    /// Working set carried from this memory object's previous solve
    /// (`warm_start_from_previous`). Statuses are ints, in the caller's own
    /// variable / row numbering. Empty until a solve produces one.
    std::vector<IpoptBoundStatus> ws_bounds;
    std::vector<IpoptConsStatus> ws_cons;
    bool ws_valid = false;      // a set is stored and worth trying
    bool ws_used = false;       // the last solve actually started from it
    /// Set when a callback caught a Ctrl-C. The solve is asked to stop and
    /// the interrupt is re-thrown once control is back on the C++ side.
    bool interrupted = false;
    /// How many evaluations failed by throwing, for the warning and stats.
    int eval_errors = 0;
  };

  class PounceInterface : public Nlpsol {
  public:
    explicit PounceInterface(const std::string& name, const Function& nlp)
      : Nlpsol(name, nlp) {}
    ~PounceInterface() override { clear_mem(); }

    /// Code generation of the solve itself (`solver.generate()`), so a
    /// deployed target runs the model and the solver as compiled C with no
    /// CasADi and no Python — it links `libpounce_cinterface`, the way
    /// CasADi's generated Ipopt code links libipopt. See `pounce_runtime.hpp`.
    void codegen_declarations(CodeGenerator& g) const override;
    void codegen_body(CodeGenerator& g) const override;
    void codegen_init_mem(CodeGenerator& g) const override;
    void codegen_free_mem(CodeGenerator& g) const override;
    std::string codegen_mem_type() const override { return "struct casadi_pounce_data"; }
    /// CasADi 3.8 asks a second question before it emits per-instance memory
    /// in generated code (gh#782). Through 3.7 the question was implicit:
    /// `CodeGenerator` treated a non-empty `codegen_mem_type()` as the
    /// request, so the override above was the whole declaration. 3.8 split
    /// the request out into this virtual, which defaults to `false` — so the
    /// `<name>_mem` array stops being declared while `codegen_mem()` keeps
    /// expanding to `<name>_mem[mem]` in the bodies below, and the generated
    /// C fails to compile on `use of undeclared identifier`.
    ///
    /// Deliberately NOT marked `override`: 3.7 and earlier declare no such
    /// virtual, and `override` there is a hard error. Without it this is an
    /// override where the virtual exists and an unused member where it does
    /// not, so one spelling serves both — the same reasoning as
    /// `convexify_compat.hpp`, which detects the fact rather than the
    /// version. `tests/codegen_mem_compat/` is what keeps the unused branch
    /// from rotting: it compiles this very declaration, extracted from this
    /// file, against a mock base in all three shapes — virtual absent,
    /// virtual present, and a virtual whose signature has drifted — and
    /// asserts through a base pointer which of them actually bind.
    ///
    /// Clang notices the missing `override` on a class that uses it
    /// elsewhere. That warning is the binding working, not a defect, and
    /// there is nowhere to put the keyword that is correct on both CasADis,
    /// so it is silenced here and checked by the test above instead.
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Winconsistent-missing-override"
#endif
    bool codegen_needs_mem() const { return true; }
#if defined(__clang__)
#pragma clang diagnostic pop
#endif
    /// Emit the `p.…` assignments that describe the problem to the runtime.
    void set_pounce_prob(CodeGenerator& g) const;
    /// Refuse, at generation time, the options the generated code cannot
    /// honour — better a message naming the option than C that silently
    /// solves a different problem than the interpreted call did.
    void assert_codegen_supported() const;

    /// Reconstruct from a serialized stream (`Function::load`).
    explicit PounceInterface(DeserializingStream& s);
    void serialize_body(SerializingStream& s) const override;
    static ProtoFunction* deserialize(DeserializingStream& s) {
      return new PounceInterface(s);
    }

    const char* plugin_name() const override { return "pounce"; }
    std::string class_name() const override { return "PounceInterface"; }

    static Nlpsol* creator(const std::string& name, const Function& nlp) {
      return new PounceInterface(name, nlp);
    }

    static const Options options_;
    const Options& get_options() const override { return options_; }

    void init(const Dict& opts) override;

    void* alloc_mem() const override { return new PounceMemory(); }
    int init_mem(void* mem) const override;
    void free_mem(void* mem) const override { delete static_cast<PounceMemory*>(mem); }

    int solve(void* mem) const override;
    Dict get_stats(void* mem) const override;

    // NLP function sparsities
    Sparsity jacg_sp_, hesslag_sp_;
    // Two independent capabilities, deliberately not one flag.
    //
    // `exact_hessian_` is whether POUNCE may ask `cb_h` for Hessian
    // *values*; `hessian_structure_` is whether the plugin can declare a
    // sparsity *pattern* at all. They coincide for `exact` (both true)
    // and for `limited-memory` (both false), and they differ for
    // `finite-difference`, which is the mode that needs the pattern and
    // must never be asked for a value.
    //
    // Collapsing them into one flag is what made `finite-difference`
    // unusable on the model class it exists for: the flag stayed true, so
    // `init` built `nlp_hess_l`, and on a model whose first derivatives
    // are the last ones CasADi can produce that construction throws —
    // the FD path failed with the same error as `exact` (gh#823 review).
    bool exact_hessian_ = true;
    bool hessian_structure_ = true;
    Dict opts_;                        // forwarded to POUNCE
    bool pass_nonlinear_variables_ = false;
    std::vector<bool> nl_ex_;          // which x enter nonlinearly
    bool clip_inactive_lam_ = true;
    bool warm_start_from_previous_ = false;
    std::string inactive_lam_strategy_ = "reltol";
    double inactive_lam_value_ = 10;

    /// Where to write POUNCE's structured solve report, and how much of
    /// it (gh#644). Empty path means off, which is the default: the
    /// report is a file write per solve, and `stats()` already carries
    /// what most callers want.
    std::string solve_report_;
    std::string solve_report_detail_ = "summary";
    /// `detail = "full"` embeds the per-iteration trajectory, which
    /// POUNCE only retains when asked before the solve. Derived rather
    /// than exposed as a third option: capturing the history has no use
    /// here except to reach the report, so making the caller ask twice
    /// only creates a way to ask wrong — `detail="full"` with an empty
    /// trajectory and nothing saying why.
    bool report_needs_iter_history() const { return solve_report_detail_ == "full"; }

    /// Hand one side's metadata dicts to POUNCE (gh#1175).
    static void forward_metadata(IpoptProblem prob, int target, const char* side,
                                 const Dict& strings,
                                 const Dict& integers, const Dict& numerics) {
      auto refused = [&](const char* kind, const std::string& key, int rc) {
        casadi_warning(std::string("POUNCE refused ") + side + "_" + kind +
                       "_md key '" + key + "': " + PounceErrorString(rc) +
                       "; solving without it.");
      };
      for (auto&& kv : integers) {
        std::vector<casadi_int> v = kv.second.to_int_vector();
        std::vector<ipindex> w(v.begin(), v.end());
        int rc = SetPounceIntegerMetaData(
            prob, target, kv.first.c_str(), static_cast<ipindex>(w.size()),
            w.empty() ? nullptr : w.data(), 0);
        if (rc != POUNCE_OK) refused("integer", kv.first, rc);
      }
      for (auto&& kv : numerics) {
        std::vector<double> v = kv.second.to_double_vector();
        int rc = SetPounceNumericMetaData(
            prob, target, kv.first.c_str(), static_cast<ipindex>(v.size()),
            v.empty() ? nullptr : v.data(), 0);
        if (rc != POUNCE_OK) refused("numeric", kv.first, rc);
      }
      for (auto&& kv : strings) {
        std::vector<std::string> v = kv.second.to_string_vector();
        std::vector<const char*> p;
        for (auto&& e : v) p.push_back(e.c_str());
        int rc = SetPounceStringMetaData(
            prob, target, kv.first.c_str(), static_cast<ipindex>(p.size()),
            p.empty() ? nullptr : p.data(), 0);
        if (rc != POUNCE_OK) refused("string", kv.first, rc);
      }
    }

    /// Convexification of the Lagrangian Hessian before it reaches the
    /// solver, using CasADi's own `Convexify` (the same code path its ipopt
    /// plugin uses), so `convexify_strategy` means exactly what it means
    /// there. Changes `hesslag_sp_`, hence the ordering in `init`.
    bool convexify_ = false;
    ConvexifyData convexify_data_;

    /// Per-variable / per-constraint metadata, as CasADi's ipopt plugin
    /// takes it (key -> one value per variable / constraint). Forwarded
    /// through `SetPounce*MetaData` (gh#1175) — so the integer keys reach a
    /// `linear_solver=custom` solver in KKT order — and echoed back
    /// through `stats()`.
    Dict var_string_md_, var_integer_md_, var_numeric_md_;
    Dict con_string_md_, con_integer_md_, con_numeric_md_;

    static const std::string meta_doc;

    /// Drain CasADi's output streams before handing control back to POUNCE
    /// (gh#667).
    ///
    /// Two writers share file descriptor 1 and neither knows about the
    /// other. POUNCE journals from Rust, where `io::stdout()` is a
    /// `LineWriter` that goes out on every newline — unconditionally, not
    /// only on a tty. CasADi writes through `uout()`, whose streambuf holds
    /// nothing and passes each chunk straight to `Logger::writeFun`, leaving
    /// the buffering to whatever sits behind it. Behind a pipe that buffer is
    /// a *fully* buffered one, so an embedder printing a line long enough to
    /// straddle it leaves the tail sitting there — and a POUNCE iteration row
    /// written in the meantime lands in the middle of the embedder's line.
    /// CasADi's own ipopt plugin has no equivalent problem: Ipopt is C++ and
    /// its journal shares the stream it is competing with.
    ///
    /// The only instant at which the plugin *knows* POUNCE is not writing is
    /// while POUNCE is blocked inside one of these callbacks, so the last
    /// thing to do before returning is empty the buffer. Doing it in
    /// `cb_iter` alone is not enough: model code logging from inside the
    /// oracle callbacks tears just as readily, and `guarded` is the one choke
    /// point all seven call sites pass through.
    ///
    /// `uerr()` as well as `uout()` because `casadi_warning` below writes
    /// there, and `OpenIpoptOutputFile(prob, "stderr", ...)` is a supported
    /// way to put POUNCE's journal on that same descriptor.
    ///
    /// Caveats worth knowing before trusting this too far:
    ///   * It assumes a single-threaded host. Two `nlpsol` instances solving
    ///     on two threads can still interleave, because solver A is free to
    ///     write while solver B sits in a callback. Ipopt is no better off.
    ///   * It does nothing for a *Python* embedder. CasADi's Python binding
    ///     points `Logger::writeFun` at `PySys_WriteStdout` but leaves
    ///     `Logger::flush` at `flushDefault`, so the bytes go into Python's
    ///     `sys.stdout` and the flush drains `std::cout` instead. There is no
    ///     portable way to reach Python's buffer from here; the fix for that
    ///     case is to route POUNCE's journal through `uout()` rather than to
    ///     flush harder. See `docs/src/casadi.md` for the interim workaround.
    ///
    /// Nothing in here may throw. A destructor is implicitly `noexcept`, and
    /// `Logger::flush` is a host-supplied callback, so a throwing sink would
    /// `std::terminate` inside the very function whose job is to keep
    /// exceptions away from Rust frames.
    struct FlushCasadiOutputOnExit {
      ~FlushCasadiOutputOnExit() {
        try {
          uout() << std::flush;
          uerr() << std::flush;
        } catch (...) {}                        // see above: never throw
      }
    };

    /// Run an oracle evaluation, converting *any* escaping exception into the
    /// C API's "this point could not be evaluated" answer.
    ///
    /// This is not defensive style, it is a hard requirement of the boundary:
    /// POUNCE is Rust, and an exception unwinding out of a callback into Rust
    /// frames aborts the process outright —
    ///
    ///     fatal runtime error: Rust cannot catch foreign exceptions, aborting
    ///
    /// — which is what a model containing a `casadi.Callback` that raises, or a
    /// Ctrl-C during a long solve, used to do. Returning `false` instead is the
    /// contract Ipopt's own callbacks use, and the solver responds by cutting
    /// the step, so a transient bad point is recoverable rather than fatal.
    ///
    /// A KeyboardInterrupt is remembered rather than swallowed: the iteration
    /// callback then stops the solve and `solve()` re-throws it, so Ctrl-C is
    /// responsive without ever crossing the language boundary.
    ///
    /// It is also where CasADi's output buffer gets drained; see
    /// `FlushCasadiOutputOnExit` below for why here and nowhere else.
    template <typename F>
    static bool guarded(PounceMemory* m, const char* what, F&& body) {
      // Function scope, above the `try`: the destructor then runs on the
      // ordinary return path *after* the handlers below have caught, never
      // while an exception is still unwinding.
      FlushCasadiOutputOnExit flush_on_exit;
      if (m->interrupted) return false;         // fail fast once stopping
      try {
        return body();
      } catch (KeyboardInterruptException&) {
        m->interrupted = true;
        return false;
      } catch (std::exception& e) {
        m->eval_errors++;
        if (m->self->show_eval_warnings_) {
          casadi_warning(std::string("POUNCE: ") + what + " failed: " + e.what());
        }
        return false;
      } catch (...) {
        m->eval_errors++;
        if (m->self->show_eval_warnings_) {
          casadi_warning(std::string("POUNCE: ") + what + " failed: unknown exception");
        }
        return false;
      }
    }

    /// `constr_viol_tol` as POUNCE will see it: the user's value from the
    /// `pounce` dict when given, else upstream's registered default.
    double constr_viol_tol() const {
      auto it = opts_.find("constr_viol_tol");
      return it == opts_.end() ? 1e-4 : static_cast<double>(it->second);
    }

    // callbacks
    static bool cb_f(ipindex n, ipnumber* x, bool new_x, ipnumber* obj, UserDataPtr ud);
    static bool cb_grad_f(ipindex n, ipnumber* x, bool new_x, ipnumber* gf, UserDataPtr ud);
    static bool cb_g(ipindex n, ipnumber* x, bool new_x, ipindex m, ipnumber* g, UserDataPtr ud);
    static bool cb_jac_g(ipindex n, ipnumber* x, bool new_x, ipindex m, ipindex nele,
                         ipindex* iRow, ipindex* jCol, ipnumber* values, UserDataPtr ud);
    static bool cb_h(ipindex n, ipnumber* x, bool new_x, ipnumber obj_factor, ipindex m,
                     ipnumber* lambda, bool new_lambda, ipindex nele,
                     ipindex* iRow, ipindex* jCol, ipnumber* values, UserDataPtr ud);
    /// `alg_mod` is 0 for an outer-loop iteration and 1 for one of the
    /// feasibility-restoration subproblem (gh#645 made the second kind
    /// fire at all; before that it was constant 0 and deliberately not
    /// recorded). It is published in `stats()['iterations']['alg_mod']`
    /// because without it the other columns of a restoration row are
    /// unreadable — they describe the min-||c||_1 subproblem, not this
    /// NLP. `stats()['restoration']` still carries the solve-level
    /// totals; see gh#634.
    static bool cb_iter(ipindex alg_mod, ipindex iter_count, ipnumber obj_value,
                        ipnumber inf_pr, ipnumber inf_du, ipnumber mu, ipnumber d_norm,
                        ipnumber regularization_size, ipnumber alpha_du, ipnumber alpha_pr,
                        ipindex ls_trials, UserDataPtr ud);
  };

  const std::string PounceInterface::meta_doc =
    "Interface to POUNCE, a primal-dual interior-point / active-set-SQP NLP "
    "solver. Options are Ipopt-compatible and are passed through the `pounce` "
    "dict.";

  const Options PounceInterface::options_
  = {{&Nlpsol::options_},
     {{"pounce",
       {OT_DICT, "Options to be passed to POUNCE (Ipopt-compatible option names)"}},
      {"pass_nonlinear_variables",
       {OT_BOOL, "Pass the list of variables entering nonlinearly to POUNCE"}},
      {"nonlinear_variables",
       {OT_BOOLVECTOR, "Manually specify which variables enter nonlinearly"}},
      {"clip_inactive_lam",
       {OT_BOOL,
        "Set multipliers of demonstrably inactive bounds to exactly zero "
        "(default true). An interior-point solve leaves a residual ~1e-12 "
        "multiplier on bounds it never touched, and CasADi's solution-map "
        "derivative reads any nonzero multiplier as an active constraint — "
        "which silently zeroes the sensitivity rows of every bounded "
        "variable. Set false for bit-identical parity with CasADi's ipopt "
        "plugin, which defaults this off."}},
      {"inactive_lam_strategy",
       {OT_STRING, "How to size the inactivity margin: 'reltol' (margin = "
                   "inactive_lam_value * constr_viol_tol) or 'abstol' "
                   "(margin = inactive_lam_value)"}},
      {"inactive_lam_value",
       {OT_DOUBLE, "Value used by inactive_lam_strategy (default 10)"}},
      {"warm_start_from_previous",
       {OT_BOOL,
        "Carry the active-set-SQP working set from one call of this solver to "
        "the next (default false). Only the active-set path produces one, so "
        "this is inert under the interior-point default. It makes the "
        "function stateful — call k+1 starts from what call k found — which "
        "is why it is opt-in; see the docs before switching it on."}},
      {"hess_lag",
       {OT_FUNCTION,
        "Function for calculating the Hessian of the Lagrangian "
        "(autogenerated by default). Signature (x, p, lam_f, lam_g) -> "
        "(triu(hess)), as CasADi's ipopt plugin expects."}},
      {"jac_g",
       {OT_FUNCTION,
        "Function for calculating the Jacobian of the constraints "
        "(autogenerated by default). Signature (x, p) -> (g, jac_g)."}},
      {"grad_f",
       {OT_FUNCTION,
        "Function for calculating the gradient of the objective "
        "(autogenerated by default). Signature (x, p) -> (f, grad_f)."}},
      {"convexify_strategy",
       {OT_STRING,
        "none|regularize|eigen-reflect|eigen-clip. Strategy to convexify the "
        "Lagrangian Hessian before it reaches the solver. POUNCE already "
        "regularizes an indefinite KKT matrix internally, so this is for "
        "shaping the Hessian itself; it applies only on the exact-Hessian "
        "path."}},
      {"convexify_margin",
       {OT_DOUBLE,
        "When using a convexification strategy, make sure that the smallest "
        "eigenvalue is at least this (default: 1e-7)."}},
      {"max_iter_eig",
       {OT_DOUBLE,
        "Maximum number of iterations to compute an eigenvalue decomposition "
        "(default: 200)."}},
      {"solve_report",
       {OT_STRING,
        "Path to write POUNCE's structured solve report (pounce.solve-report/v1 "
        "JSON) after each solve. Empty (the default) writes nothing. The file "
        "is rewritten per solve, so a solver called in a loop leaves only the "
        "last one — give each call its own path if you need to keep them."}},
      {"solve_report_detail",
       {OT_STRING,
        "'summary' (default) or 'full'. 'full' embeds the per-iteration "
        "trajectory, which costs a retained iterate per iteration and is "
        "enabled automatically when you ask for it."}},
      {"var_string_md",
       {OT_DICT, "String metadata about variables: key -> one string per "
                 "variable. Forwarded to POUNCE (SetPounce*MetaData); integer "
                 "keys reach a linear_solver=custom solver in KKT row order. "
                 "Echoed back through stats()."}},
      {"var_integer_md",
       {OT_DICT, "Integer metadata about variables (see var_string_md)"}},
      {"var_numeric_md",
       {OT_DICT, "Numeric metadata about variables (see var_string_md)"}},
      {"con_string_md",
       {OT_DICT, "String metadata about constraints (see var_string_md)"}},
      {"con_integer_md",
       {OT_DICT, "Integer metadata about constraints (see var_string_md)"}},
      {"con_numeric_md",
       {OT_DICT, "Numeric metadata about constraints (see var_string_md)"}}
     }
  };

  void PounceInterface::init(const Dict& opts) {
    Nlpsol::init(opts);

    std::string convexify_strategy = "none";
    double convexify_margin = 1e-7;
    casadi_int max_iter_eig = 200;

    for (auto&& op : opts) {
      if (op.first == "pounce") {
        opts_ = op.second;
      } else if (op.first == "pass_nonlinear_variables") {
        pass_nonlinear_variables_ = op.second;
      } else if (op.first == "nonlinear_variables") {
        nl_ex_ = op.second;
      } else if (op.first == "clip_inactive_lam") {
        clip_inactive_lam_ = op.second;
      } else if (op.first == "inactive_lam_strategy") {
        inactive_lam_strategy_ = op.second.to_string();
      } else if (op.first == "inactive_lam_value") {
        inactive_lam_value_ = op.second;
      } else if (op.first == "warm_start_from_previous") {
        warm_start_from_previous_ = op.second;
      } else if (op.first == "hess_lag") {
        Function f = op.second;
        casadi_assert(f.n_in() == 4 && f.n_out() == 1,
                      "hess_lag must take 4 inputs (x, p, lam_f, lam_g) and "
                      "return 1 output, got " + str(f.n_in()) + " and " +
                      str(f.n_out()) + ".");
        set_function(f, "nlp_hess_l");
      } else if (op.first == "jac_g") {
        Function f = op.second;
        casadi_assert(f.n_in() == 2 && f.n_out() == 2,
                      "jac_g must take 2 inputs (x, p) and return 2 outputs "
                      "(g, jac_g), got " + str(f.n_in()) + " and " +
                      str(f.n_out()) + ".");
        set_function(f, "nlp_jac_g");
      } else if (op.first == "grad_f") {
        Function f = op.second;
        casadi_assert(f.n_in() == 2 && f.n_out() == 2,
                      "grad_f must take 2 inputs (x, p) and return 2 outputs "
                      "(f, grad_f), got " + str(f.n_in()) + " and " +
                      str(f.n_out()) + ".");
        set_function(f, "nlp_grad_f");
      } else if (op.first == "convexify_strategy") {
        convexify_strategy = op.second.to_string();
      } else if (op.first == "convexify_margin") {
        convexify_margin = op.second;
      } else if (op.first == "max_iter_eig") {
        max_iter_eig = op.second;
      } else if (op.first == "solve_report") {
        solve_report_ = op.second.to_string();
      } else if (op.first == "solve_report_detail") {
        solve_report_detail_ = op.second.to_string();
      } else if (op.first == "var_string_md") {
        var_string_md_ = op.second;
      } else if (op.first == "var_integer_md") {
        var_integer_md_ = op.second;
      } else if (op.first == "var_numeric_md") {
        var_numeric_md_ = op.second;
      } else if (op.first == "con_string_md") {
        con_string_md_ = op.second;
      } else if (op.first == "con_integer_md") {
        con_integer_md_ = op.second;
      } else if (op.first == "con_numeric_md") {
        con_numeric_md_ = op.second;
      }
    }

    // Reject a bad `solve_report_detail` here rather than at write time.
    // The C API validates it too, but only when the report is written —
    // i.e. after a solve that has already run. A typo should cost the
    // construction of the solver, not a solve.
    casadi_assert(solve_report_detail_ == "summary" || solve_report_detail_ == "full",
                  "solve_report_detail must be 'summary' or 'full', got '"
                  + solve_report_detail_ + "'.");

    // Which Hessian capabilities does the chosen mode need?
    //
    //   exact             values + structure — `nlp_hess_l` is required,
    //                     and a model that cannot build it must say so.
    //   limited-memory    neither; the quasi-Newton matrix is POUNCE's.
    //   finite-difference structure only. POUNCE recovers the values by
    //                     probing the analytic Jacobian and never calls
    //                     `cb_h` for them, so the pattern is the whole
    //                     contribution — and it is worth a lot: on
    //                     `laptime` the declared pattern is 17 probe
    //                     groups against the Jacobian-derived pattern's
    //                     341.
    std::string hess_mode = "exact";
    auto hess_it = opts_.find("hessian_approximation");
    if (hess_it != opts_.end()) hess_mode = hess_it->second.to_string();

    exact_hessian_ = (hess_mode != "limited-memory" && hess_mode != "finite-difference");
    hessian_structure_ = (hess_mode != "limited-memory");

    // `fd_hessian_pattern=jacobian` says the pattern is to come from the
    // Jacobian, so building CasADi's symbolic Hessian just to throw the
    // pattern away is pure cost. Honour it here rather than paying it.
    if (hess_mode == "finite-difference") {
      auto pat_it = opts_.find("fd_hessian_pattern");
      if (pat_it != opts_.end() && pat_it->second.to_string() == "jacobian") {
        hessian_structure_ = false;
      }
    }

    create_function("nlp_f", {"x", "p"}, {"f"});
    create_function("nlp_g", {"x", "p"}, {"g"});
    if (!has_function("nlp_grad_f")) {
      create_function("nlp_grad_f", {"x", "p"}, {"f", "grad:f:x"});
    }
    if (!has_function("nlp_jac_g")) {
      create_function("nlp_jac_g", {"x", "p"}, {"g", "jac:g:x"});
    }
    jacg_sp_ = get_function("nlp_jac_g").sparsity_out(1);
    casadi_assert(jacg_sp_.size1() == ng_, "nlp_jac_g must have " + str(ng_) +
                  " rows, but has " + str(jacg_sp_.size1()) + " instead.");
    casadi_assert(jacg_sp_.size2() == nx_, "nlp_jac_g must have " + str(nx_) +
                  " columns, but has " + str(jacg_sp_.size2()) + " instead.");

    convexify_ = false;
    if (hessian_structure_) {
      if (!has_function("nlp_hess_l")) {
        try {
          create_function("nlp_hess_l", {"x", "p", "lam:f", "lam:g"},
                          {"triu:hess:gamma:x:x"},
                          {{"gamma", {"f", "g"}}});
        } catch (std::exception& e) {
          // For `exact` this is fatal and should read exactly as it
          // always did. For `finite-difference` the Hessian is a bonus,
          // not a requirement — the pattern sharpens the probe colouring
          // and the values are never read — so a model that cannot be
          // differentiated twice degrades to the Jacobian-derived
          // pattern rather than failing.
          if (exact_hessian_) throw;
          hessian_structure_ = false;
          if (verbose_) {
            casadi_message(std::string("POUNCE: no symbolic Lagrangian Hessian for this "
                                       "model, so hessian_approximation='finite-difference' "
                                       "will derive its pattern from the Jacobian. CasADi "
                                       "said: ") + e.what());
          }
        }
      }
    }
    // Re-tested, not an `else`: the block above turns it off when CasADi
    // could not build the Hessian after all.
    if (hessian_structure_) {
      hesslag_sp_ = get_function("nlp_hess_l").sparsity_out(0);
      casadi_assert(hesslag_sp_.is_triu(),
                    "nlp_hess_l must be upper triangular.");
      casadi_assert(hesslag_sp_.size1() == nx_ && hesslag_sp_.size2() == nx_,
                    "nlp_hess_l must be " + str(nx_) + "-by-" + str(nx_) +
                    ", but is " + str(hesslag_sp_.size1()) + "-by-" +
                    str(hesslag_sp_.size2()) + " instead.");
      // Convexification rewrites Hessian *values*, so it belongs to the
      // one mode that reads them. Under `finite-difference` the pattern
      // is all that crosses, and running `Convexify::setup` there would
      // widen it — a superset is still safe, but it would buy nothing and
      // cost probe groups.
      if (convexify_strategy != "none" && exact_hessian_) {
        convexify_ = true;
        Dict cvx_opts;
        cvx_opts["strategy"] = convexify_strategy;
        cvx_opts["margin"] = convexify_margin;
        cvx_opts["max_iter_eig"] = max_iter_eig;
        cvx_opts["verbose"] = verbose_;
        // Convexification can *widen* the pattern (it works block-wise), so
        // this is the sparsity the solver is told about and the size of the
        // values buffer the callback writes into.
        hesslag_sp_ = Convexify::setup(convexify_data_, hesslag_sp_, cvx_opts);
      }
    }
    if (convexify_strategy != "none" && !convexify_) {
      casadi_warning("convexify_strategy is ignored under "
                     "hessian_approximation='" + hess_mode + "': there is no "
                     "exact Hessian to convexify.");
    }

    if (pass_nonlinear_variables_ && nl_ex_.empty()) {
      nl_ex_ = oracle_.which_depends("x", {"f", "g"}, 2, false);
    }

    if (convexify_) {
      alloc_iw(convexify_data_.sz_iw);
      alloc_w(convexify_data_.sz_w);
    }

    // Scratch for the bound multipliers split into z_L / z_U. The interpreted
    // path keeps those in the memory object, so this looks unused — but the
    // work-vector sizes the *generated* entry point reports are taken from
    // this accounting, and its runtime carves z_L / z_U out of `w`. Without
    // the reservation the generated code writes past the caller's buffer,
    // which is a heap corruption rather than an error.
    alloc_w(2 * nx_, true);
  }

  int PounceInterface::init_mem(void* mem) const {
    if (Nlpsol::init_mem(mem)) return 1;
    auto m = static_cast<PounceMemory*>(mem);
    m->self = this;
    if (convexify_) m->add_stat("convexify");
    return 0;
  }

  bool PounceInterface::cb_f(ipindex, ipnumber* x, bool, ipnumber* obj, UserDataPtr ud) {
    auto m = static_cast<PounceMemory*>(ud);
    return guarded(m, "objective evaluation", [&] {
      m->arg[0] = x;
      m->arg[1] = m->d_nlp.p;
      m->res[0] = obj;
      return m->self->calc_function(m, "nlp_f") == 0;
    });
  }

  bool PounceInterface::cb_grad_f(ipindex, ipnumber* x, bool, ipnumber* gf, UserDataPtr ud) {
    auto m = static_cast<PounceMemory*>(ud);
    return guarded(m, "objective gradient", [&] {
      m->arg[0] = x;
      m->arg[1] = m->d_nlp.p;
      m->res[0] = nullptr;
      m->res[1] = gf;
      return m->self->calc_function(m, "nlp_grad_f") == 0;
    });
  }

  bool PounceInterface::cb_g(ipindex, ipnumber* x, bool, ipindex, ipnumber* g, UserDataPtr ud) {
    auto m = static_cast<PounceMemory*>(ud);
    return guarded(m, "constraint evaluation", [&] {
      m->arg[0] = x;
      m->arg[1] = m->d_nlp.p;
      m->res[0] = g;
      return m->self->calc_function(m, "nlp_g") == 0;
    });
  }

  bool PounceInterface::cb_jac_g(ipindex, ipnumber* x, bool, ipindex, ipindex nele,
                                 ipindex* iRow, ipindex* jCol, ipnumber* values,
                                 UserDataPtr ud) {
    auto m = static_cast<PounceMemory*>(ud);
    const PounceInterface* self = m->self;
    if (values) {
      return guarded(m, "constraint Jacobian", [&] {
        m->arg[0] = x;
        m->arg[1] = m->d_nlp.p;
        m->res[0] = nullptr;
        m->res[1] = values;
        return self->calc_function(m, "nlp_jac_g") == 0;
      });
    }
    // sparsity, CCS -> triplet
    casadi_int ncol = self->jacg_sp_.size2();
    const casadi_int* colind = self->jacg_sp_.colind();
    const casadi_int* row = self->jacg_sp_.row();
    if (nele != colind[ncol]) return false;
    for (casadi_int cc = 0; cc < ncol; ++cc) {
      for (casadi_int el = colind[cc]; el < colind[cc + 1]; ++el) {
        *iRow++ = static_cast<ipindex>(row[el]);
        *jCol++ = static_cast<ipindex>(cc);
      }
    }
    return true;
  }

  bool PounceInterface::cb_h(ipindex, ipnumber* x, bool, ipnumber obj_factor, ipindex,
                             ipnumber* lambda, bool, ipindex nele,
                             ipindex* iRow, ipindex* jCol, ipnumber* values, UserDataPtr ud) {
    auto m = static_cast<PounceMemory*>(ud);
    const PounceInterface* self = m->self;
    if (values && !self->exact_hessian_) {
      // Structure-only mode (`finite-difference`). POUNCE recovers the
      // values by probing, so this branch is unreachable by design —
      // failing here rather than quietly answering keeps that a fact
      // about the code instead of a claim about it. If this ever fires,
      // the FD updater is not the one supplying the Hessian and the
      // measurement that says FD was exercised is wrong.
      return false;
    }
    if (values) {
      bool ok = guarded(m, "Lagrangian Hessian", [&] {
        m->arg[0] = x;
        m->arg[1] = m->d_nlp.p;
        m->arg[2] = &obj_factor;
        m->arg[3] = lambda;
        m->res[0] = values;
        return self->calc_function(m, "nlp_hess_l") == 0;
      });
      if (!ok || !self->convexify_) return ok;
      // In place, over the widened pattern `Convexify::setup` returned —
      // which is the pattern the solver was given, so `values` is big enough.
      return guarded(m, "Hessian convexification", [&] {
        ScopedTiming tic(m->fstats.at("convexify"));
        // `convexify_eval_compat`, not `convexify_eval`: CasADi renamed the
        // helper after 3.7.2 and the shim picks whichever name is declared
        // (gh#668, convexify_compat.hpp).
        return convexify_eval_compat(0, &self->convexify_data_.config, values,
                                     values, m->iw, m->w) == 0;
      });
    }
    // upper-triangular CCS == lower-triangular triplet (row/col swap)
    casadi_int ncol = self->hesslag_sp_.size2();
    const casadi_int* colind = self->hesslag_sp_.colind();
    const casadi_int* row = self->hesslag_sp_.row();
    if (nele != colind[ncol]) return false;
    for (casadi_int cc = 0; cc < ncol; ++cc) {
      for (casadi_int el = colind[cc]; el < colind[cc + 1]; ++el) {
        *iRow++ = static_cast<ipindex>(cc);
        *jCol++ = static_cast<ipindex>(row[el]);
      }
    }
    return true;
  }

  bool PounceInterface::cb_iter(ipindex alg_mod, ipindex iter_count, ipnumber obj_value,
                                ipnumber inf_pr, ipnumber inf_du, ipnumber mu,
                                ipnumber d_norm, ipnumber regularization_size,
                                ipnumber alpha_du, ipnumber alpha_pr, ipindex ls_trials,
                                UserDataPtr ud) {
    auto m = static_cast<PounceMemory*>(ud);
    m->alg_mod.push_back(alg_mod);
    m->inf_pr.push_back(inf_pr);
    m->inf_du.push_back(inf_du);
    m->mu_trace.push_back(mu);
    m->d_norm.push_back(d_norm);
    m->regularization_size.push_back(regularization_size);
    m->obj_trace.push_back(obj_value);
    m->alpha_pr.push_back(alpha_pr);
    m->alpha_du.push_back(alpha_du);
    m->ls_trials.push_back(ls_trials);
    // On a restoration fire `iter_count` counts the *subproblem's* inner
    // iterations and restarts from 0 on every entry, so letting it
    // through here would make `stats()['iter_count']` report whatever
    // the last restoration episode happened to reach. Outer fires only.
    if (alg_mod == 0) m->iter = iter_count;

    // A Ctrl-C caught in an oracle callback stops the solve here: returning
    // false is `User_Requested_Stop`, the one channel POUNCE offers for
    // "stop now" that does not involve unwinding through it.
    if (m->interrupted) return false;

    // Restoration fires stop here. CasADi fixes the callback signature
    // at `(x, f, g, lam_x, lam_g)` and a restoration iterate supplies
    // none of them: it is a point of the min-||c||_1 subproblem, in the
    // subproblem's own variable space, and POUNCE's `GetIpoptCurrent*`
    // inspectors report no data for its duration by design. Handing the
    // user's callback a stale `x` from the previous outer iteration
    // beside a fresh restoration `f` would be worse than not calling it.
    // The trace above still records the iteration, tagged `alg_mod = 1`,
    // so the episode is visible in `stats()` without being fed to user
    // code as if it were a solution estimate.
    if (alg_mod != 0) return true;

    // Full callback: pull the current iterate out of POUNCE and drive
    // casadi's `iteration_callback` with it.
    //
    // `iteration_callback_step` throttles the *user's* callback only. CasADi's
    // ipopt plugin returns before recording anything, so a step of 10 also
    // punches holes in `stats()['iterations']`; here the trace above is always
    // complete, because throttling an expensive callback and losing the
    // convergence history are unrelated wishes.
    const PounceInterface* self = m->self;
    if (!self->fcallback_.is_null() && iter_count % self->callback_step_ == 0) {
      const int n = static_cast<int>(self->nx_);
      const int ng = static_cast<int>(self->ng_);
      std::vector<double> x(n), zl(n), zu(n), g(ng), lam(ng);
      bool ok = GetIpoptCurrentIterate(m->prob, false, n, x.data(), zl.data(), zu.data(),
                                       ng, ng ? g.data() : nullptr, ng ? lam.data() : nullptr);
      if (!ok) {
        if (iter_count == 0) uerr() << "POUNCE: iterate not available for callback\n";
        return true;
      }
      auto d_nlp = &m->d_nlp;
      casadi_copy(x.data(), n, d_nlp->z);
      for (int i = 0; i < n; ++i) d_nlp->lam[i] = zu[i] - zl[i];
      casadi_copy(lam.data(), ng, d_nlp->lam + n);
      std::fill_n(m->arg, self->fcallback_.n_in(), nullptr);
      m->arg[NLPSOL_X] = x.data();
      m->arg[NLPSOL_F] = &obj_value;
      m->arg[NLPSOL_G] = g.data();
      m->arg[NLPSOL_LAM_X] = d_nlp->lam;
      m->arg[NLPSOL_LAM_G] = d_nlp->lam + n;
      std::fill_n(m->res, self->fcallback_.n_out(), nullptr);
      double ret_double = 0;
      m->res[0] = &ret_double;
      // The user's callback is user code: the same boundary rule applies.
      // `iteration_callback_ignore_errors` is CasADi's switch for whether a
      // throwing callback should stop the solve or be shrugged off.
      bool cb_ok = guarded(m, "iteration callback", [&] {
        ScopedTiming tic(m->fstats.at("callback_fun"));
        self->fcallback_(m->arg, m->res, m->iw, m->w, 0);
        return true;
      });
      if (!cb_ok) return self->iteration_callback_ignore_errors_ && !m->interrupted;
      return static_cast<casadi_int>(ret_double) == 0;
    }
    return true;
  }

  int PounceInterface::solve(void* mem) const {
    auto m = static_cast<PounceMemory*>(mem);
    auto d_nlp = &m->d_nlp;

    const int n = static_cast<int>(nx_);
    const int ng = static_cast<int>(ng_);

    // Reset the per-iteration trace. A memory object is reused across
    // calls — every receding-horizon loop calls the same solver
    // repeatedly — and without this the traces concatenate, so
    // `stats()['iterations']` describes every solve so far while
    // `iter_count` beside it describes only the last (gh#634). CasADi's
    // ipopt plugin clears the same vectors at the same point.
    m->inf_pr.clear();
    m->inf_du.clear();
    m->mu_trace.clear();
    m->d_norm.clear();
    m->regularization_size.clear();
    m->obj_trace.clear();
    m->alpha_pr.clear();
    m->alpha_du.clear();
    m->ls_trials.clear();
    m->alg_mod.clear();
    m->final_inf_pr = m->final_inf_du = m->final_compl_inf = 0;
    m->report_written = false;
    m->linsol_valid = false;
    m->resto_calls = m->resto_inner = m->resto_outer = 0;
    m->fd_pattern = -1;
    m->fd_nnz = m->fd_n = m->fd_groups = m->fd_rho_max = m->fd_fell_back = 0;
    m->fd_clique_widened = 0;
    m->resto_secs = 0;

    m->xl.assign(d_nlp->lbz, d_nlp->lbz + n);
    m->xu.assign(d_nlp->ubz, d_nlp->ubz + n);
    m->gl.assign(d_nlp->lbz + n, d_nlp->lbz + n + ng);
    m->gu.assign(d_nlp->ubz + n, d_nlp->ubz + n + ng);
    m->xk.assign(d_nlp->z, d_nlp->z + n);
    m->gk.assign(ng, 0.0);
    m->lam_g.assign(d_nlp->lam + n, d_nlp->lam + n + ng);
    m->z_L.resize(n);
    m->z_U.resize(n);
    for (int i = 0; i < n; ++i) {
      m->z_L[i] = std::max(0.0, -d_nlp->lam[i]);
      m->z_U[i] = std::max(0.0, d_nlp->lam[i]);
    }

    const int nnz_jac = ng == 0 ? 0 : static_cast<int>(jacg_sp_.nnz());
    // The pattern crosses whenever we have one — under `finite-difference`
    // that is the declared sparsity POUNCE colours its probes from, and
    // `cb_h` serves the structure request while refusing a values one.
    const int nnz_h = hessian_structure_ ? static_cast<int>(hesslag_sp_.nnz()) : 0;

    IpoptProblem prob = CreateIpoptProblem(
      n, m->xl.data(), m->xu.data(), ng, m->gl.data(), m->gu.data(),
      nnz_jac, nnz_h, 0 /* C index style */,
      &PounceInterface::cb_f, &PounceInterface::cb_g,
      &PounceInterface::cb_grad_f, &PounceInterface::cb_jac_g,
      hessian_structure_ ? &PounceInterface::cb_h : nullptr);
    casadi_assert(prob != nullptr, "POUNCE: CreateIpoptProblem failed");

    // Has to precede the solve: POUNCE keeps the per-iteration trajectory
    // only when asked beforehand, and there is no way to reconstruct it
    // afterwards.
    if (!solve_report_.empty() && report_needs_iter_history()) {
      IpoptEnableIterHistory(prob);
    }
    m->prob = prob;

    // Deliberately keyed on the mode, NOT on `exact_hessian_`. Both said
    // the same thing while `limited-memory` was the only inexact mode;
    // once `finite-difference` joined it, `!exact_hessian_` would have
    // forced limited-memory over the mode the user actually asked for.
    // It survives only because the forwarding loop below re-sends the
    // user's own `hessian_approximation` afterwards — a defect masked by
    // statement order is still a defect, so state the condition instead.
    if (!hessian_structure_ && !exact_hessian_) {
      auto it = opts_.find("hessian_approximation");
      if (it == opts_.end() || it->second.to_string() == "limited-memory") {
        AddIpoptStrOption(prob, CC("hessian_approximation"), CC("limited-memory"));
      }
    }
    // Forward user options.
    //
    // The type to send is POUNCE's, not the one the value happens to
    // carry. A `Dict` value's type comes from how the user typed the
    // literal, and the two disagree for the commonest case there is:
    // `{'tol': 1}` is an int in Python and a number to POUNCE, so
    // dispatching on the value alone sent it to AddIpoptIntOption,
    // which refuses it — the option silently kept its default. So ask
    // the registry first (gh#634) and fall back to the value's own type
    // only for keywords this build does not register, where POUNCE will
    // report the unknown option itself.
    for (auto&& op : opts_) {
      const std::string& key = op.first;
      const GenericType& val = op.second;
      switch (GetPounceOptionType(prob, key.c_str())) {
        case POUNCE_OPTION_NUMBER:
          AddIpoptNumOption(prob, CC(key.c_str()), val.to_double());
          continue;
        case POUNCE_OPTION_INTEGER:
          AddIpoptIntOption(prob, CC(key.c_str()), static_cast<int>(val.to_int()));
          continue;
        case POUNCE_OPTION_STRING: {
          // A bool is how a Python caller most naturally writes POUNCE's
          // yes/no string options.
          std::string v = val.is_bool() ? (static_cast<bool>(val) ? "yes" : "no")
                                        : val.to_string();
          AddIpoptStrOption(prob, CC(key.c_str()), CC(v.c_str()));
          continue;
        }
        default:
          break;
      }
      if (val.is_double() && !val.is_int()) {
        AddIpoptNumOption(prob, CC(key.c_str()), val.to_double());
      } else if (val.is_int() || val.is_bool()) {
        if (val.is_bool()) {
          AddIpoptStrOption(prob, CC(key.c_str()),
                            CC(static_cast<bool>(val) ? "yes" : "no"));
        } else {
          AddIpoptIntOption(prob, CC(key.c_str()), static_cast<int>(val.to_int()));
        }
      } else {
        { std::string v = val.to_string();
          AddIpoptStrOption(prob, CC(key.c_str()), CC(v.c_str())); }
      }
    }
    // gh#624 — hand POUNCE the variables that enter nonlinearly, so the
    // limited-memory Hessian is approximated over that subspace only.
    // CasADi derives the set with `which_depends` (or takes it verbatim
    // from `nonlinear_variables`); POUNCE ignores it on the exact-Hessian
    // path, matching Ipopt.
    if (!nl_ex_.empty()) {
      std::vector<casadi_int> pos;
      for (casadi_int i = 0; i < static_cast<casadi_int>(nl_ex_.size()); ++i) {
        if (nl_ex_[i]) pos.push_back(i);
      }
      std::vector<ipindex> idx(pos.begin(), pos.end());
      if (!IpoptSetNonlinearVariables(prob, static_cast<ipindex>(idx.size()),
                                      idx.empty() ? nullptr : idx.data())) {
        casadi_warning("POUNCE refused the nonlinear-variable list; "
                       "approximating over all variables.");
      }
    }

    // gh#1175 — variable / constraint metadata. A key POUNCE refuses (wrong
    // length, NaN) is reported and skipped; the solve goes on without it.
    forward_metadata(prob, POUNCE_META_VARIABLES, "var", var_string_md_,
                     var_integer_md_, var_numeric_md_);
    forward_metadata(prob, POUNCE_META_CONSTRAINTS, "con", con_string_md_,
                     con_integer_md_, con_numeric_md_);

    // Start this solve from the active set the previous one ended on.
    //
    // The working set is the SQP's guess at which bounds and constraints are
    // active; identifying it is most of the work, and in a receding-horizon
    // loop the answer barely moves between steps. There is nowhere in
    // `nlpsol`'s fixed input signature to pass one, so it is carried here, in
    // this memory object, rather than by the caller.
    //
    // A stale set is a guess, not a claim: bounds arrive as per-call inputs
    // and may have moved under it, in which case POUNCE validates and refuses
    // it, and this solve simply cold-starts its working set.
    m->ws_used = false;
    if (warm_start_from_previous_ && m->ws_valid) {
      m->ws_used = IpoptSetWarmStartWorkingSet(
          prob, m->ws_bounds.data(), ng ? m->ws_cons.data() : nullptr) != 0;
      if (!m->ws_used) {
        m->ws_valid = false;      // do not keep re-offering a rejected set
        if (verbose_) {
          casadi_message("POUNCE: previous working set refused; cold-starting it.");
        }
      }
    }

    SetIntermediateCallback(prob, &PounceInterface::cb_iter);

    // The one window `guarded` cannot close (gh#667): whatever the host and
    // CasADi buffered before this call is still pending when POUNCE prints
    // its banner, and no callback has fired yet to drain it. Once per solve.
    uout() << std::flush;
    uerr() << std::flush;

    enum ApplicationReturnStatus st = IpoptSolve(
      prob, m->xk.data(), ng ? m->gk.data() : nullptr, &m->obj,
      ng ? m->lam_g.data() : nullptr, m->z_L.data(), m->z_U.data(),
      static_cast<UserDataPtr>(m));

    // Harvest the working set for the next call. `IpoptGetWorkingSet`
    // returns false when there is nothing to carry — the interior-point path
    // produces no working set, and neither does an SQP solve that converged
    // before its first QP — so the option is inert rather than wrong there.
    if (warm_start_from_previous_) {
      m->ws_bounds.resize(n);
      m->ws_cons.resize(ng);
      m->ws_valid = IpoptGetWorkingSet(prob, m->ws_bounds.data(),
                                       ng ? m->ws_cons.data() : nullptr) != 0;
    }

    m->return_status = static_cast<int>(st);
    m->iter = GetIpoptIterCount(prob);
    m->t_solve = GetIpoptSolveTime(prob);
    // Everything below reads through `prob`, so it has to happen before
    // the free — and `m->prob` has to stop pointing at freed memory
    // afterwards, because `get_stats` uses it to tell a solve in flight
    // (where the live-iterate accessors work) from one that has ended.
    m->final_inf_pr = GetIpoptPrimalInf(prob);
    m->final_inf_du = GetIpoptDualInf(prob);
    m->final_compl_inf = GetIpoptComplInf(prob);
    m->linsol_valid = GetPounceLinearSolverStats(prob, &m->linsol);
    GetPounceRestorationStats(prob, &m->resto_calls, &m->resto_inner,
                              &m->resto_outer, &m->resto_secs);
    GetPounceFdHessianStats(prob, &m->fd_pattern, &m->fd_nnz, &m->fd_n,
                            &m->fd_groups, &m->fd_rho_max, &m->fd_fell_back,
                            &m->fd_clique_widened);
    // Same window as the harvest above, and for the same reason: the
    // report is built from the solve retained on `prob`, which the free
    // below takes with it.
    //
    // A failed write is a warning, not an error. The solve succeeded and
    // its answer is already in `m`; refusing to return it because a
    // diagnostic file could not be written would be the wrong trade, and
    // an unwritable path is a caller mistake that a warning names
    // exactly. `m->report_written` records what happened so `stats()`
    // can be asked rather than the log read.
    if (!solve_report_.empty()) {
      m->report_written = IpoptWriteSolveReport(prob, solve_report_.c_str(),
                                                solve_report_detail_.c_str()) != 0;
      if (!m->report_written) {
        casadi_warning("POUNCE: could not write solve report to '" + solve_report_
                       + "' (unwritable path, or no solve to report).");
      }
    }
    FreeIpoptProblem(prob);
    m->prob = nullptr;

    // Back on the C++ side, with POUNCE's frames unwound and its handle
    // freed: now a Ctrl-C caught during a callback can be re-thrown safely.
    if (m->interrupted) throw KeyboardInterruptException();

    // Write back to casadi's nlpsol data layout
    casadi_copy(m->xk.data(), n, d_nlp->z);
    casadi_copy(m->gk.data(), ng, d_nlp->z + n);
    d_nlp->objective = m->obj;
    for (int i = 0; i < n; ++i) d_nlp->lam[i] = m->z_U[i] - m->z_L[i];
    casadi_copy(m->lam_g.data(), ng, d_nlp->lam + n);

    // Zero the multipliers of bounds the iterate is demonstrably far from.
    //
    // An interior-point method leaves a residual multiplier — order 1e-12
    // here — on every bound it never came near, because those multipliers
    // approach zero from above rather than reaching it. CasADi's
    // solution-map derivative treats any nonzero bound multiplier as an
    // *active* constraint and fixes that variable, so a single stray
    // 1e-12 turns the whole sensitivity row into zeros. On an NMPC model
    // whose controls are bounded, that means `jacobian(u0, x0)` — the
    // feedback gain — silently reads 0 where a re-solve says -9.11.
    //
    // The test is primal distance, not multiplier magnitude: a variable
    // more than `margin` away from a bound is not sitting on it, whatever
    // the arithmetic left behind. Same rule, option names and margin as
    // CasADi's ipopt plugin (`clip_inactive_lam`), except that this
    // defaults **on** — the Ipopt plugin defaults it off, which is where
    // the trap comes from.
    if (clip_inactive_lam_) {
      double margin;
      if (inactive_lam_strategy_ == "abstol") {
        margin = inactive_lam_value_;
      } else if (inactive_lam_strategy_ == "reltol") {
        margin = inactive_lam_value_ * constr_viol_tol();
      } else {
        casadi_error("inactive_lam_strategy '" + inactive_lam_strategy_ +
                     "' unknown. Use 'abstol' or 'reltol'.");
      }
      for (casadi_int i = 0; i < nx_ + ng_; ++i) {
        if (d_nlp->lam[i] > 0 && d_nlp->ubz[i] - d_nlp->z[i] > margin) d_nlp->lam[i] = 0;
        if (d_nlp->lam[i] < 0 && d_nlp->z[i] - d_nlp->lbz[i] > margin) d_nlp->lam[i] = 0;
      }
    }

    m->n_iter = m->iter;
    m->success = (st == Solve_Succeeded || st == Solved_To_Acceptable_Level);
    if (m->success) {
      m->unified_return_status = SOLVER_RET_SUCCESS;
    } else if (st == Maximum_Iterations_Exceeded) {
      m->unified_return_status = SOLVER_RET_LIMITED;
    } else if (st == Infeasible_Problem_Detected) {
      m->unified_return_status = SOLVER_RET_INFEASIBLE;
    } else {
      m->unified_return_status = SOLVER_RET_UNKNOWN;
    }
    return 0;
  }

  static const char* pounce_status_name(int st) {
    switch (st) {
      case Solve_Succeeded: return "Solve_Succeeded";
      case Solved_To_Acceptable_Level: return "Solved_To_Acceptable_Level";
      case Infeasible_Problem_Detected: return "Infeasible_Problem_Detected";
      case Search_Direction_Becomes_Too_Small: return "Search_Direction_Becomes_Too_Small";
      case Diverging_Iterates: return "Diverging_Iterates";
      case User_Requested_Stop: return "User_Requested_Stop";
      case Feasible_Point_Found: return "Feasible_Point_Found";
      case Maximum_Iterations_Exceeded: return "Maximum_Iterations_Exceeded";
      case Restoration_Failed: return "Restoration_Failed";
      case Error_In_Step_Computation: return "Error_In_Step_Computation";
      case Maximum_CpuTime_Exceeded: return "Maximum_CpuTime_Exceeded";
      case Invalid_Option: return "Invalid_Option";
      case Invalid_Problem_Definition: return "Invalid_Problem_Definition";
      case Invalid_Number_Detected: return "Invalid_Number_Detected";
      case Unrecoverable_Exception: return "Unrecoverable_Exception";
      case Insufficient_Memory: return "Insufficient_Memory";
      default: return "Internal_Error";
    }
  }

  Dict PounceInterface::get_stats(void* mem) const {
    Dict stats = Nlpsol::get_stats(mem);
    auto m = static_cast<PounceMemory*>(mem);
    stats["return_status"] = pounce_status_name(m->return_status);
    stats["iter_count"] = m->iter;
    stats["t_solve_pounce"] = m->t_solve;
    // Whether this call started from the previous call's active set, and
    // whether it left one behind for the next.
    stats["warm_started_working_set"] = m->ws_used;
    stats["working_set_available"] = m->ws_valid;
    stats["n_eval_errors"] = m->eval_errors;
    // Only when one was asked for: an absent key means "no report
    // requested", which is a different thing from "the write failed".
    if (!solve_report_.empty()) {
      stats["solve_report"] = solve_report_;
      stats["solve_report_written"] = m->report_written;
    }
    // Metadata POUNCE has no channel for: given back rather than dropped, so
    // a caller that set it can at least see it survived the round trip.
    if (!var_string_md_.empty()) stats["var_string_md"] = var_string_md_;
    if (!var_integer_md_.empty()) stats["var_integer_md"] = var_integer_md_;
    if (!var_numeric_md_.empty()) stats["var_numeric_md"] = var_numeric_md_;
    if (!con_string_md_.empty()) stats["con_string_md"] = con_string_md_;
    if (!con_integer_md_.empty()) stats["con_integer_md"] = con_integer_md_;
    if (!con_numeric_md_.empty()) stats["con_numeric_md"] = con_numeric_md_;
    Dict iterations;
    iterations["inf_pr"] = m->inf_pr;
    iterations["inf_du"] = m->inf_du;
    iterations["mu"] = m->mu_trace;
    iterations["d_norm"] = m->d_norm;
    iterations["regularization_size"] = m->regularization_size;
    iterations["obj"] = m->obj_trace;
    iterations["alpha_pr"] = m->alpha_pr;
    iterations["alpha_du"] = m->alpha_du;
    iterations["ls_trials"] = m->ls_trials;
    // 0 = outer iteration, 1 = restoration subproblem iteration. Every
    // other vector here is only interpretable against it.
    iterations["alg_mod"] = m->alg_mod;
    stats["iterations"] = iterations;

    // `stats()` is callable from inside `iteration_callback`, where the
    // trace above already carries the current iteration as its last
    // element — the plugin records it before handing control to the
    // callback. So the live diagnostics a caller wants mid-solve
    // (mode, mu, step norm, regularization, line-search trials) need no
    // second channel, and none is invented here: CasADi's `Nlpsol`
    // callback signature is fixed at (x, f, g, lam_x, lam_g), and this
    // keeps everything else reachable without parsing the solver log.
    //
    // What is *not* in the trace is the current violation vectors, so
    // those are fetched here, and only here: they cost an evaluation
    // and are wanted by the rare caller, not by every `stats()` call
    // after a solve. `m->prob` is non-NULL only while a solve is in
    // flight; POUNCE itself then reports whether an intermediate
    // callback is actually on the stack.
    if (m->prob) {
      const int n = static_cast<int>(nx_);
      const int ng = static_cast<int>(ng_);
      std::vector<double> xlv(n), xuv(n), cxl(n), cxu(n), glag(n), gv(ng), cg(ng);
      if (GetIpoptCurrentViolations(m->prob, false, n, xlv.data(), xuv.data(),
                                    cxl.data(), cxu.data(), glag.data(), ng,
                                    ng ? gv.data() : nullptr,
                                    ng ? cg.data() : nullptr)) {
        Dict v;
        v["x_L_violation"] = xlv;
        v["x_U_violation"] = xuv;
        v["compl_x_L"] = cxl;
        v["compl_x_U"] = cxu;
        v["grad_lag_x"] = glag;
        v["nlp_constraint_violation"] = gv;
        v["compl_g"] = cg;
        stats["current_violations"] = v;
      }
    } else {
      // Final KKT errors, harvested in `solve` before the problem was
      // freed. Only meaningful once a solve has ended, which is exactly
      // when `m->prob` is NULL.
      stats["final_inf_pr"] = m->final_inf_pr;
      stats["final_inf_du"] = m->final_inf_du;
      stats["final_compl_inf"] = m->final_compl_inf;

      // Solve-level restoration totals. Per-iteration labelling now
      // exists too — see `iterations['alg_mod']` (gh#645) — but these
      // are the only source for the inner iteration count and the wall
      // time, and they answer "how much restoration?" in one read.
      Dict resto;
      resto["calls"] = static_cast<casadi_int>(m->resto_calls);
      resto["inner_iters"] = static_cast<casadi_int>(m->resto_inner);
      resto["outer_iters"] = static_cast<casadi_int>(m->resto_outer);
      resto["wall_secs"] = m->resto_secs;
      stats["restoration"] = resto;

      // Only present when the finite-difference Hessian actually built a
      // pattern. Absent rather than zero on every other mode: a zero
      // probe count would read as "free", not as "not this mode".
      //
      // `pattern` is what the solve ENDED UP with. Asking for
      // `declared` on a model that declares no Hessian structure
      // silently yields the Jacobian derivation, and that is the whole
      // reason to read this -- on `laptime` the two differ by 17 probe
      // groups against 341.
      if (m->fd_pattern >= 0) {
        Dict fd;
        fd["pattern"] = std::string(m->fd_pattern == 0 ? "declared" : "jacobian");
        fd["nnz"] = static_cast<casadi_int>(m->fd_nnz);
        // `groups / n` is the compression: the fraction of a dense
        // finite-difference scheme's probes this pattern costs.
        fd["n"] = static_cast<casadi_int>(m->fd_n);
        fd["groups"] = static_cast<casadi_int>(m->fd_groups);
        fd["rho_max"] = static_cast<casadi_int>(m->fd_rho_max);
        fd["coloring_fell_back"] = m->fd_fell_back != 0;
        // Why `groups` is what it is, when it surprises: with no stated
        // objective linearity the clique widens to every nonlinear
        // variable, or to every variable, and the probe count follows.
        fd["objective_clique_widened"] = m->fd_clique_widened != 0;
        stats["fd_hessian"] = fd;
      }
    }

    // What the KKT linear solver did. `solver_name` is the backend that
    // actually ran — the answer to "did my `linear_solver` option take
    // effect?", which no other stat reports. Absent fields are absent
    // rather than zero: POUNCE does not instrument phase timings, and a
    // zero there would read as "instantaneous" instead of "unmeasured".
    if (m->linsol_valid) {
      Dict ls;
      ls["solver_name"] = std::string(m->linsol.solver_name);
      ls["n_factors"] = static_cast<casadi_int>(m->linsol.n_factors);
      ls["n_pattern_reuse"] = static_cast<casadi_int>(m->linsol.n_pattern_reuse);
      ls["n_pattern_changes"] = static_cast<casadi_int>(m->linsol.n_pattern_changes);
      if (!std::isnan(m->linsol.max_fill_ratio)) ls["max_fill_ratio"] = m->linsol.max_fill_ratio;
      if (!std::isnan(m->linsol.min_abs_pivot)) ls["min_abs_pivot"] = m->linsol.min_abs_pivot;
      if (!std::isnan(m->linsol.max_abs_pivot)) ls["max_abs_pivot"] = m->linsol.max_abs_pivot;
      if (m->linsol.last_inertia_positive >= 0) {
        ls["last_inertia"] = std::vector<casadi_int>{
          static_cast<casadi_int>(m->linsol.last_inertia_positive),
          static_cast<casadi_int>(m->linsol.last_inertia_negative),
          static_cast<casadi_int>(m->linsol.last_inertia_zero)};
      }
      if (m->linsol.last_nnz_a >= 0) ls["last_nnz_a"] = static_cast<casadi_int>(m->linsol.last_nnz_a);
      if (m->linsol.last_nnz_l >= 0) ls["last_nnz_l"] = static_cast<casadi_int>(m->linsol.last_nnz_l);
      stats["linear_solver"] = ls;
    }
    return stats;
  }

  // ---------------------------------------------------------------------
  // Code generation
  //
  // `solver.generate('solver.c')` emits the model *and* the solve as C. What
  // the generated file needs at build time is `pounce.h` and
  // `libpounce_cinterface`; what it does not need is CasADi, Python, or this
  // plugin. That is the same bargain CasADi's own Ipopt plugin strikes (its
  // generated code includes `<coin-or/IpStdCInterface.h>` and links libipopt),
  // and it works here for the same reason: `pounce.h` is that API.
  //
  // The generated solve must agree with the interpreted one, which is why
  // `clip_inactive_lam` is reproduced in the runtime rather than skipped, and
  // why anything that cannot be reproduced is refused below instead of
  // silently dropped.
  // ---------------------------------------------------------------------

  void PounceInterface::assert_codegen_supported() const {
    casadi_assert(!fcallback_.is_null() == false,
                  "iteration_callback cannot be code generated: the callback is "
                  "a CasADi Function living in this process, and generated code "
                  "runs without CasADi. Drop it, or keep this solver "
                  "interpreted.");
    casadi_assert(!convexify_,
                  "convexify_strategy cannot be code generated by this plugin "
                  "yet. Drop it, or keep this solver interpreted.");
    casadi_assert(!warm_start_from_previous_,
                  "warm_start_from_previous cannot be code generated: it "
                  "carries an active-set working set between calls of one "
                  "solver object, which the generated entry point has no "
                  "channel for. Pass x0/lam_g0/lam_x0 instead.");
    casadi_assert(solve_report_.empty(),
                  "solve_report cannot be code generated by this plugin yet. "
                  "The generated code links the same C API and could call "
                  "IpoptWriteSolveReport, but the emitted runtime does not, "
                  "and silently dropping the option would leave you waiting "
                  "for a file that is never written. Drop it, or keep this "
                  "solver interpreted.");
    casadi_assert(jacg_sp_.size1() == 0 || jacg_sp_.nnz() > 0,
                  "A constraint Jacobian with no nonzeros is not supported by "
                  "the C API this generates against.");
  }

  void PounceInterface::codegen_init_mem(CodeGenerator& g) const {
    g << "pounce_init_mem(&" + codegen_mem(g) + ");\n";
    g << "return 0;\n";
  }

  void PounceInterface::codegen_free_mem(CodeGenerator& g) const {
    g << "pounce_free_mem(&" + codegen_mem(g) + ");\n";
  }

  void PounceInterface::codegen_declarations(CodeGenerator& g) const {
    assert_codegen_supported();
    Nlpsol::codegen_declarations(g);
    g.add_auxiliary(CodeGenerator::AUX_NLP);
    g.add_auxiliary(CodeGenerator::AUX_COPY);
    g.add_auxiliary(CodeGenerator::AUX_FMAX);
    g.add_dependency(get_function("nlp_f"));
    g.add_dependency(get_function("nlp_grad_f"));
    g.add_dependency(get_function("nlp_g"));
    g.add_dependency(get_function("nlp_jac_g"));
    if (exact_hessian_) g.add_dependency(get_function("nlp_hess_l"));
    g.add_include("pounce.h");

    // The five oracle callbacks, in the C API's signatures. Each is the
    // generated-code twin of the `cb_*` methods above; the exception guard
    // those carry has nothing to guard here — generated C does not throw.
    std::string name = "nlp_f";
    std::string f = g.shorthand(g.wrapper(get_function(name), name));
    g << "bool " << f
      << "(ipindex n, ipnumber *x, bool new_x, ipnumber *obj_value, UserDataPtr user_data) {\n";
    g.flush(g.body);
    g.scope_enter();
    g << "struct casadi_pounce_data* d = (struct casadi_pounce_data*) user_data;\n";
    g << "d->arg[0] = x;\n";
    g << "d->arg[1] = d->nlp->p;\n";
    g << "d->res[0] = obj_value;\n";
    std::string flag = g(get_function(name), "d->arg", "d->res", "d->iw", "d->w", "false");
    g << "if (" + flag + ") return false;\n";
    g << "return true;\n";
    g.scope_exit();
    g << "}\n";

    name = "nlp_g";
    f = g.shorthand(g.wrapper(get_function(name), name));
    g << "bool " << f
      << "(ipindex n, ipnumber *x, bool new_x, ipindex m, ipnumber *g, "
      << "UserDataPtr user_data) {\n";
    g.flush(g.body);
    g.scope_enter();
    g << "struct casadi_pounce_data* d = (struct casadi_pounce_data*) user_data;\n";
    g << "d->arg[0] = x;\n";
    g << "d->arg[1] = d->nlp->p;\n";
    g << "d->res[0] = g;\n";
    flag = g(get_function(name), "d->arg", "d->res", "d->iw", "d->w", "false");
    g << "if (" + flag + ") return false;\n";
    g << "return true;\n";
    g.scope_exit();
    g << "}\n";

    name = "nlp_grad_f";
    f = g.shorthand(g.wrapper(get_function(name), name));
    g << "bool " << f
      << "(ipindex n, ipnumber *x, bool new_x, ipnumber *grad_f, UserDataPtr user_data) {\n";
    g.flush(g.body);
    g.scope_enter();
    g << "struct casadi_pounce_data* d = (struct casadi_pounce_data*) user_data;\n";
    g << "d->arg[0] = x;\n";
    g << "d->arg[1] = d->nlp->p;\n";
    g << "d->res[0] = 0;\n";
    g << "d->res[1] = grad_f;\n";
    flag = g(get_function(name), "d->arg", "d->res", "d->iw", "d->w", "false");
    g << "if (" + flag + ") return false;\n";
    g << "return true;\n";
    g.scope_exit();
    g << "}\n";

    name = "nlp_jac_g";
    f = g.shorthand(g.wrapper(get_function(name), name));
    g << "bool " << f
      << "(ipindex n, ipnumber *x, bool new_x, ipindex m, ipindex nele_jac, "
      << "ipindex *iRow, ipindex *jCol, ipnumber *values, UserDataPtr user_data) {\n";
    g.flush(g.body);
    g.scope_enter();
    g << "struct casadi_pounce_data* d = (struct casadi_pounce_data*) user_data;\n";
    g << "if (values) {\n";
    g << "d->arg[0] = x;\n";
    g << "d->arg[1] = d->nlp->p;\n";
    g << "d->res[0] = 0;\n";
    g << "d->res[1] = values;\n";
    flag = g(get_function(name), "d->arg", "d->res", "d->iw", "d->w", "false");
    g << "if (" + flag + ") return false;\n";
    g << "} else {\n";
    g << "casadi_pounce_sparsity(d->prob->sp_a, iRow, jCol);\n";
    g << "}\n";
    g << "return true;\n";
    g.scope_exit();
    g << "}\n";

    if (exact_hessian_) {
      name = "nlp_hess_l";
      f = g.shorthand(g.wrapper(get_function(name), name));
      g << "bool " << f << "(ipindex n, ipnumber *x, bool new_x, ipnumber obj_factor, "
        << "ipindex m, ipnumber *lambda, bool new_lambda, ipindex nele_hess, "
        << "ipindex *iRow, ipindex *jCol, ipnumber *values, UserDataPtr user_data) {\n";
      g.flush(g.body);
      g.scope_enter();
      g << "struct casadi_pounce_data* d = (struct casadi_pounce_data*) user_data;\n";
      g << "if (values) {\n";
      g << "d->arg[0] = x;\n";
      g << "d->arg[1] = d->nlp->p;\n";
      g << "d->arg[2] = &obj_factor;\n";
      g << "d->arg[3] = lambda;\n";
      g << "d->res[0] = values;\n";
      flag = g(get_function(name), "d->arg", "d->res", "d->iw", "d->w", "false");
      g << "if (" + flag + ") return false;\n";
      g << "} else {\n";
      g << "casadi_pounce_sparsity_h(d->prob->sp_h, iRow, jCol);\n";
      g << "}\n";
      g << "return true;\n";
      g.scope_exit();
      g << "}\n";
    } else if (hessian_structure_) {
      // Structure-only, the generated twin of `cb_h`'s refusal above.
      // `finite-difference` reads the pattern and recovers the values by
      // probing, so this serves `p.sp_h` — baked in as a literal, which
      // is why the block needs no `nlp_hess_l` dependency — and refuses
      // a values request rather than answering one nothing should ask.
      f = g.shorthand("pounce_hess_l_struct");
      g << "bool " << f << "(ipindex n, ipnumber *x, bool new_x, ipnumber obj_factor, "
        << "ipindex m, ipnumber *lambda, bool new_lambda, ipindex nele_hess, "
        << "ipindex *iRow, ipindex *jCol, ipnumber *values, UserDataPtr user_data) {\n";
      g.flush(g.body);
      g.scope_enter();
      g << "struct casadi_pounce_data* d = (struct casadi_pounce_data*) user_data;\n";
      g << "(void)n; (void)x; (void)new_x; (void)obj_factor; (void)m;\n";
      g << "(void)lambda; (void)new_lambda; (void)nele_hess;\n";
      g << "if (values) return false;\n";
      g << "casadi_pounce_sparsity_h(d->prob->sp_h, iRow, jCol);\n";
      g << "return true;\n";
      g.scope_exit();
      g << "}\n";
    }
  }

  void PounceInterface::set_pounce_prob(CodeGenerator& g) const {
    g << "d->nlp = &d_nlp;\n";
    g << "d->prob = &p;\n";
    g << "p.nlp = &p_nlp;\n";
    g << "p.sp_a = " << g.sparsity(jacg_sp_) << ";\n";
    // Whenever a pattern exists, including structure-only under
    // `finite-difference` — `casadi_pounce_setup` reads `nnz_h` off it.
    if (hessian_structure_) {
      g << "p.sp_h = " << g.sparsity(hesslag_sp_) << ";\n";
    } else {
      g << "p.sp_h = 0;\n";
    }
    g << "casadi_pounce_setup(&p);\n";

    // The nonlinear-variable subset, as an `ipindex` array. `g.constant`
    // would give a `casadi_int` one, and `ipindex` is `int` — a different
    // width, so the array is emitted rather than cast.
    std::vector<casadi_int> pos;
    for (casadi_int i = 0; i < static_cast<casadi_int>(nl_ex_.size()); ++i) {
      if (nl_ex_[i]) pos.push_back(i);
    }
    if (!pos.empty() && pos.size() < nl_ex_.size()) {
      std::string arr = g.shorthand(name_ + "_nl_vars");
      g.auxiliaries << "static const ipindex " << arr << "[] = {";
      for (size_t i = 0; i < pos.size(); ++i) {
        g.auxiliaries << (i ? ", " : "") << pos[i];
      }
      g.auxiliaries << "};\n";
      g << "p.nonlin_vars = " << arr << ";\n";
      g << "p.n_nonlin_vars = " << pos.size() << ";\n";
    } else {
      g << "p.nonlin_vars = 0;\n";
      g << "p.n_nonlin_vars = 0;\n";
    }

    // A negative margin is the runtime's "leave the multipliers alone".
    if (clip_inactive_lam_) {
      double margin = inactive_lam_strategy_ == "abstol"
                    ? inactive_lam_value_
                    : inactive_lam_value_ * constr_viol_tol();
      casadi_assert(inactive_lam_strategy_ == "abstol"
                    || inactive_lam_strategy_ == "reltol",
                    "inactive_lam_strategy '" + inactive_lam_strategy_ +
                    "' unknown. Use 'abstol' or 'reltol'.");
      g << "p.inactive_lam_margin = " << margin << ";\n";
    } else {
      g << "p.inactive_lam_margin = -1;\n";
    }

    g << "p.eval_f = " << g.shorthand(g.wrapper(get_function("nlp_f"), "nlp_f")) << ";\n";
    g << "p.eval_g = " << g.shorthand(g.wrapper(get_function("nlp_g"), "nlp_g")) << ";\n";
    g << "p.eval_grad_f = "
      << g.shorthand(g.wrapper(get_function("nlp_grad_f"), "nlp_grad_f")) << ";\n";
    g << "p.eval_jac_g = "
      << g.shorthand(g.wrapper(get_function("nlp_jac_g"), "nlp_jac_g")) << ";\n";
    if (exact_hessian_) {
      g << "p.eval_h = "
        << g.shorthand(g.wrapper(get_function("nlp_hess_l"), "nlp_hess_l")) << ";\n";
    } else if (hessian_structure_) {
      g << "p.eval_h = " << g.shorthand("pounce_hess_l_struct") << ";\n";
    } else {
      g << "p.eval_h = casadi_pounce_hess_l_empty;\n";
    }
  }

  void PounceInterface::codegen_body(CodeGenerator& g) const {
    assert_codegen_supported();
    codegen_body_enter(g);
    g.auxiliaries << pounce_runtime_str;

    g.local("d", "struct casadi_pounce_data*");
    g.init_local("d", "&" + codegen_mem(g));
    g.local("p", "struct casadi_pounce_prob");
    set_pounce_prob(g);

    g << "casadi_pounce_init(d, &arg, &res, &iw, &w);\n";
    g << "casadi_pounce_presolve(d);\n";

    // Mode-keyed, matching the interpreted path: `!exact_hessian_` is
    // true for `finite-difference` too, and forcing limited-memory there
    // would override the mode the user asked for.
    if (!hessian_structure_ && !exact_hessian_) {
      auto it = opts_.find("hessian_approximation");
      if (it == opts_.end() || it->second.to_string() == "limited-memory") {
        g << "AddIpoptStrOption(d->pounce, \"hessian_approximation\", \"limited-memory\");\n";
      }
    }
    // The user's options, typed the same way the interpreted path types
    // them: POUNCE's registry decides, and the value's own `GenericType`
    // is only the fallback for a keyword the registry does not know.
    // `GetPounceOptionType` takes a NULL problem handle for exactly this
    // caller — there is no problem at generation time — and asking it
    // here is what keeps generated and interpreted solves from
    // disagreeing about an option's type (gh#634).
    //
    // Unlike CasADi's ipopt codegen, nothing is emitted for an option
    // the user did not set: that one asks Ipopt's registry for every
    // option's type and writes a `linear_solver=mumps` default into the
    // emitted code, which POUNCE would refuse.
    for (auto&& op : opts_) {
      const std::string& key = op.first;
      const GenericType& val = op.second;
      switch (GetPounceOptionType(nullptr, key.c_str())) {
        case POUNCE_OPTION_NUMBER:
          g << "AddIpoptNumOption(d->pounce, \"" << key << "\", "
            << val.to_double() << ");\n";
          continue;
        case POUNCE_OPTION_INTEGER:
          g << "AddIpoptIntOption(d->pounce, \"" << key << "\", "
            << val.to_int() << ");\n";
          continue;
        case POUNCE_OPTION_STRING:
          g << "AddIpoptStrOption(d->pounce, \"" << key << "\", \""
            << (val.is_bool() ? (static_cast<bool>(val) ? "yes" : "no")
                              : val.to_string())
            << "\");\n";
          continue;
        default:
          break;
      }
      if (val.is_double() && !val.is_int()) {
        g << "AddIpoptNumOption(d->pounce, \"" << key << "\", "
          << val.to_double() << ");\n";
      } else if (val.is_bool()) {
        g << "AddIpoptStrOption(d->pounce, \"" << key << "\", \""
          << (static_cast<bool>(val) ? "yes" : "no") << "\");\n";
      } else if (val.is_int()) {
        g << "AddIpoptIntOption(d->pounce, \"" << key << "\", "
          << val.to_int() << ");\n";
      } else {
        g << "AddIpoptStrOption(d->pounce, \"" << key << "\", \""
          << val.to_string() << "\");\n";
      }
    }

    g << "casadi_pounce_solve(d);\n";

    codegen_body_exit(g);

    if (error_on_fail_) {
      g << "return d->unified_return_status;\n";
    } else {
      g << "return 0;\n";
    }
  }

  // ---------------------------------------------------------------------
  // Serialization
  //
  // `S.save('s.casadi')` / `Function.load` round-trips the solver, the same
  // as CasADi's own plugins. Everything below is configuration — no solver
  // handle and no working set crosses, so a loaded function is a cold
  // solver with the options it was built with. (The C API's `IpoptProblem`
  // is created per solve anyway, and a carried working set belongs to the
  // memory object, which is never serialized.)
  //
  // Reading a saved function needs this plugin loadable in the reading
  // process; that is CasADi's rule for every out-of-tree plugin, and the
  // failure is a clean "Plugin 'pounce' is not found" rather than garbage.
  // ---------------------------------------------------------------------

  void PounceInterface::serialize_body(SerializingStream& s) const {
    Nlpsol::serialize_body(s);
    s.version("PounceInterface", 2);
    s.pack("PounceInterface::jacg_sp", jacg_sp_);
    s.pack("PounceInterface::hesslag_sp", hesslag_sp_);
    s.pack("PounceInterface::exact_hessian", exact_hessian_);
    s.pack("PounceInterface::hessian_structure", hessian_structure_);
    s.pack("PounceInterface::opts", opts_);
    s.pack("PounceInterface::pass_nonlinear_variables", pass_nonlinear_variables_);
    s.pack("PounceInterface::nl_ex", nl_ex_);
    s.pack("PounceInterface::clip_inactive_lam", clip_inactive_lam_);
    s.pack("PounceInterface::warm_start_from_previous", warm_start_from_previous_);
    s.pack("PounceInterface::inactive_lam_strategy", inactive_lam_strategy_);
    s.pack("PounceInterface::inactive_lam_value", inactive_lam_value_);
    s.pack("PounceInterface::convexify", convexify_);
    if (convexify_) Convexify::serialize(s, "PounceInterface::", convexify_data_);
    s.pack("PounceInterface::var_string_md", var_string_md_);
    s.pack("PounceInterface::var_integer_md", var_integer_md_);
    s.pack("PounceInterface::var_numeric_md", var_numeric_md_);
    s.pack("PounceInterface::con_string_md", con_string_md_);
    s.pack("PounceInterface::con_integer_md", con_integer_md_);
    s.pack("PounceInterface::con_numeric_md", con_numeric_md_);
  }

  PounceInterface::PounceInterface(DeserializingStream& s) : Nlpsol(s) {
    // v1 predates the values/structure split, and in it the two were the
    // same flag — so `exact_hessian_` is exactly the right value for
    // `hessian_structure_` when reading one back.
    int ver = s.version("PounceInterface", 1, 2);
    s.unpack("PounceInterface::jacg_sp", jacg_sp_);
    s.unpack("PounceInterface::hesslag_sp", hesslag_sp_);
    s.unpack("PounceInterface::exact_hessian", exact_hessian_);
    if (ver >= 2) {
      s.unpack("PounceInterface::hessian_structure", hessian_structure_);
    } else {
      hessian_structure_ = exact_hessian_;
    }
    s.unpack("PounceInterface::opts", opts_);
    s.unpack("PounceInterface::pass_nonlinear_variables", pass_nonlinear_variables_);
    s.unpack("PounceInterface::nl_ex", nl_ex_);
    s.unpack("PounceInterface::clip_inactive_lam", clip_inactive_lam_);
    s.unpack("PounceInterface::warm_start_from_previous", warm_start_from_previous_);
    s.unpack("PounceInterface::inactive_lam_strategy", inactive_lam_strategy_);
    s.unpack("PounceInterface::inactive_lam_value", inactive_lam_value_);
    s.unpack("PounceInterface::convexify", convexify_);
    if (convexify_) Convexify::deserialize(s, "PounceInterface::", convexify_data_);
    s.unpack("PounceInterface::var_string_md", var_string_md_);
    s.unpack("PounceInterface::var_integer_md", var_integer_md_);
    s.unpack("PounceInterface::var_numeric_md", var_numeric_md_);
    s.unpack("PounceInterface::con_string_md", con_string_md_);
    s.unpack("PounceInterface::con_integer_md", con_integer_md_);
    s.unpack("PounceInterface::con_numeric_md", con_numeric_md_);
  }

  extern "C"
  int casadi_register_nlpsol_pounce(Nlpsol::Plugin* plugin) {
    plugin->creator = PounceInterface::creator;
    plugin->name = "pounce";
    plugin->doc = PounceInterface::meta_doc.c_str();
    plugin->version = CASADI_VERSION;
    plugin->options = &PounceInterface::options_;
    plugin->deserialize = &PounceInterface::deserialize;
    return 0;
  }

  extern "C"
  void casadi_load_nlpsol_pounce() {
    Nlpsol::registerPlugin(casadi_register_nlpsol_pounce);
  }

} // namespace casadi
