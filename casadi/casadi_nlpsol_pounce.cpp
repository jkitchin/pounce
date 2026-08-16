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
#include "pounce_runtime.hpp"
#include <cstring>
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
    std::vector<casadi_int> ls_trials;
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
    bool exact_hessian_ = true;
    Dict opts_;                        // forwarded to POUNCE
    bool pass_nonlinear_variables_ = false;
    std::vector<bool> nl_ex_;          // which x enter nonlinearly
    bool clip_inactive_lam_ = true;
    bool warm_start_from_previous_ = false;
    std::string inactive_lam_strategy_ = "reltol";
    double inactive_lam_value_ = 10;

    /// Convexification of the Lagrangian Hessian before it reaches the
    /// solver, using CasADi's own `Convexify` (the same code path its ipopt
    /// plugin uses), so `convexify_strategy` means exactly what it means
    /// there. Changes `hesslag_sp_`, hence the ordering in `init`.
    bool convexify_ = false;
    ConvexifyData convexify_data_;

    /// Per-variable / per-constraint metadata. CasADi's ipopt plugin
    /// forwards these to Ipopt's `get_var_con_metadata`; POUNCE's C API has
    /// no counterpart, so they are accepted (a script that sets them keeps
    /// working when `ipopt` is swapped for `pounce`), stored, and echoed
    /// back through `stats()` rather than silently dropped.
    Dict var_string_md_, var_integer_md_, var_numeric_md_;
    Dict con_string_md_, con_integer_md_, con_numeric_md_;

    static const std::string meta_doc;

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
    template <typename F>
    static bool guarded(PounceMemory* m, const char* what, F&& body) {
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
      {"var_string_md",
       {OT_DICT, "String metadata about variables. Accepted for ipopt-plugin "
                 "compatibility; not forwarded (POUNCE has no metadata "
                 "channel), echoed back through stats()."}},
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

    // Do we have an exact Hessian?
    exact_hessian_ = true;
    auto hess_it = opts_.find("hessian_approximation");
    if (hess_it != opts_.end() && hess_it->second.to_string() == "limited-memory") {
      exact_hessian_ = false;
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
    if (exact_hessian_) {
      if (!has_function("nlp_hess_l")) {
        create_function("nlp_hess_l", {"x", "p", "lam:f", "lam:g"},
                        {"triu:hess:gamma:x:x"},
                        {{"gamma", {"f", "g"}}});
      }
      hesslag_sp_ = get_function("nlp_hess_l").sparsity_out(0);
      casadi_assert(hesslag_sp_.is_triu(),
                    "nlp_hess_l must be upper triangular.");
      casadi_assert(hesslag_sp_.size1() == nx_ && hesslag_sp_.size2() == nx_,
                    "nlp_hess_l must be " + str(nx_) + "-by-" + str(nx_) +
                    ", but is " + str(hesslag_sp_.size1()) + "-by-" +
                    str(hesslag_sp_.size2()) + " instead.");
      if (convexify_strategy != "none") {
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
    } else if (convexify_strategy != "none") {
      casadi_warning("convexify_strategy is ignored under "
                     "hessian_approximation='limited-memory': there is no "
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
        return convexify_eval(&self->convexify_data_.config, values, values,
                              m->iw, m->w) == 0;
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

  bool PounceInterface::cb_iter(ipindex, ipindex iter_count, ipnumber obj_value,
                                ipnumber inf_pr, ipnumber inf_du, ipnumber mu,
                                ipnumber d_norm, ipnumber regularization_size,
                                ipnumber alpha_du, ipnumber alpha_pr, ipindex ls_trials,
                                UserDataPtr ud) {
    auto m = static_cast<PounceMemory*>(ud);
    m->inf_pr.push_back(inf_pr);
    m->inf_du.push_back(inf_du);
    m->mu_trace.push_back(mu);
    m->d_norm.push_back(d_norm);
    m->regularization_size.push_back(regularization_size);
    m->obj_trace.push_back(obj_value);
    m->alpha_pr.push_back(alpha_pr);
    m->alpha_du.push_back(alpha_du);
    m->ls_trials.push_back(ls_trials);
    m->iter = iter_count;

    // A Ctrl-C caught in an oracle callback stops the solve here: returning
    // false is `User_Requested_Stop`, the one channel POUNCE offers for
    // "stop now" that does not involve unwinding through it.
    if (m->interrupted) return false;

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
    const int nnz_h = exact_hessian_ ? static_cast<int>(hesslag_sp_.nnz()) : 0;

    IpoptProblem prob = CreateIpoptProblem(
      n, m->xl.data(), m->xu.data(), ng, m->gl.data(), m->gu.data(),
      nnz_jac, nnz_h, 0 /* C index style */,
      &PounceInterface::cb_f, &PounceInterface::cb_g,
      &PounceInterface::cb_grad_f, &PounceInterface::cb_jac_g,
      exact_hessian_ ? &PounceInterface::cb_h : nullptr);
    casadi_assert(prob != nullptr, "POUNCE: CreateIpoptProblem failed");
    m->prob = prob;

    if (!exact_hessian_) {
      AddIpoptStrOption(prob, CC("hessian_approximation"), CC("limited-memory"));
    }
    // Forward user options (typed dispatch by GenericType)
    for (auto&& op : opts_) {
      if (op.second.is_double() && !op.second.is_int()) {
        AddIpoptNumOption(prob, CC(op.first.c_str()), op.second.to_double());
      } else if (op.second.is_int() || op.second.is_bool()) {
        if (op.second.is_bool()) {
          AddIpoptStrOption(prob, CC(op.first.c_str()),
                            CC(static_cast<bool>(op.second) ? "yes" : "no"));
        } else {
          AddIpoptIntOption(prob, CC(op.first.c_str()), static_cast<int>(op.second.to_int()));
        }
      } else {
        { std::string v = op.second.to_string();
          AddIpoptStrOption(prob, CC(op.first.c_str()), CC(v.c_str())); }
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
    FreeIpoptProblem(prob);

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
    stats["iterations"] = iterations;
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
    }
  }

  void PounceInterface::set_pounce_prob(CodeGenerator& g) const {
    g << "d->nlp = &d_nlp;\n";
    g << "d->prob = &p;\n";
    g << "p.nlp = &p_nlp;\n";
    g << "p.sp_a = " << g.sparsity(jacg_sp_) << ";\n";
    if (exact_hessian_) {
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

    if (!exact_hessian_) {
      g << "AddIpoptStrOption(d->pounce, \"hessian_approximation\", \"limited-memory\");\n";
    }
    // The user's options, typed the same way the interpreted path types them
    // — off the `GenericType`, not off a registry. CasADi's ipopt codegen asks
    // Ipopt's registry for each option's type and, finding none set, writes a
    // `linear_solver=mumps` default into the emitted code. Neither is right
    // here: POUNCE's registry lives in Rust, and it refuses `mumps`.
    for (auto&& op : opts_) {
      if (op.second.is_double() && !op.second.is_int()) {
        g << "AddIpoptNumOption(d->pounce, \"" << op.first << "\", "
          << op.second.to_double() << ");\n";
      } else if (op.second.is_bool()) {
        g << "AddIpoptStrOption(d->pounce, \"" << op.first << "\", \""
          << (static_cast<bool>(op.second) ? "yes" : "no") << "\");\n";
      } else if (op.second.is_int()) {
        g << "AddIpoptIntOption(d->pounce, \"" << op.first << "\", "
          << op.second.to_int() << ");\n";
      } else {
        g << "AddIpoptStrOption(d->pounce, \"" << op.first << "\", \""
          << op.second.to_string() << "\");\n";
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
    s.version("PounceInterface", 1);
    s.pack("PounceInterface::jacg_sp", jacg_sp_);
    s.pack("PounceInterface::hesslag_sp", hesslag_sp_);
    s.pack("PounceInterface::exact_hessian", exact_hessian_);
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
    s.version("PounceInterface", 1);
    s.unpack("PounceInterface::jacg_sp", jacg_sp_);
    s.unpack("PounceInterface::hesslag_sp", hesslag_sp_);
    s.unpack("PounceInterface::exact_hessian", exact_hessian_);
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
