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
  };

  class PounceInterface : public Nlpsol {
  public:
    explicit PounceInterface(const std::string& name, const Function& nlp)
      : Nlpsol(name, nlp) {}
    ~PounceInterface() override { clear_mem(); }

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

    static const std::string meta_doc;

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
       {OT_BOOLVECTOR, "Manually specify which variables enter nonlinearly"}}
     }
  };

  void PounceInterface::init(const Dict& opts) {
    Nlpsol::init(opts);

    for (auto&& op : opts) {
      if (op.first == "pounce") {
        opts_ = op.second;
      } else if (op.first == "pass_nonlinear_variables") {
        pass_nonlinear_variables_ = op.second;
      } else if (op.first == "nonlinear_variables") {
        nl_ex_ = op.second;
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
    create_function("nlp_grad_f", {"x", "p"}, {"f", "grad:f:x"});
    Function jac_g_fcn = create_function("nlp_jac_g", {"x", "p"}, {"g", "jac:g:x"});
    jacg_sp_ = jac_g_fcn.sparsity_out(1);

    if (exact_hessian_) {
      Function hess_l_fcn = create_function("nlp_hess_l", {"x", "p", "lam:f", "lam:g"},
                                            {"triu:hess:gamma:x:x"},
                                            {{"gamma", {"f", "g"}}});
      hesslag_sp_ = hess_l_fcn.sparsity_out(0);
    }

    if (pass_nonlinear_variables_ && nl_ex_.empty()) {
      nl_ex_ = oracle_.which_depends("x", {"f", "g"}, 2, false);
    }
  }

  int PounceInterface::init_mem(void* mem) const {
    if (Nlpsol::init_mem(mem)) return 1;
    auto m = static_cast<PounceMemory*>(mem);
    m->self = this;
    return 0;
  }

  bool PounceInterface::cb_f(ipindex, ipnumber* x, bool, ipnumber* obj, UserDataPtr ud) {
    auto m = static_cast<PounceMemory*>(ud);
    m->arg[0] = x;
    m->arg[1] = m->d_nlp.p;
    m->res[0] = obj;
    return m->self->calc_function(m, "nlp_f") == 0;
  }

  bool PounceInterface::cb_grad_f(ipindex, ipnumber* x, bool, ipnumber* gf, UserDataPtr ud) {
    auto m = static_cast<PounceMemory*>(ud);
    m->arg[0] = x;
    m->arg[1] = m->d_nlp.p;
    m->res[0] = nullptr;
    m->res[1] = gf;
    return m->self->calc_function(m, "nlp_grad_f") == 0;
  }

  bool PounceInterface::cb_g(ipindex, ipnumber* x, bool, ipindex, ipnumber* g, UserDataPtr ud) {
    auto m = static_cast<PounceMemory*>(ud);
    m->arg[0] = x;
    m->arg[1] = m->d_nlp.p;
    m->res[0] = g;
    return m->self->calc_function(m, "nlp_g") == 0;
  }

  bool PounceInterface::cb_jac_g(ipindex, ipnumber* x, bool, ipindex, ipindex nele,
                                 ipindex* iRow, ipindex* jCol, ipnumber* values,
                                 UserDataPtr ud) {
    auto m = static_cast<PounceMemory*>(ud);
    const PounceInterface* self = m->self;
    if (values) {
      m->arg[0] = x;
      m->arg[1] = m->d_nlp.p;
      m->res[0] = nullptr;
      m->res[1] = values;
      return self->calc_function(m, "nlp_jac_g") == 0;
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
      m->arg[0] = x;
      m->arg[1] = m->d_nlp.p;
      m->arg[2] = &obj_factor;
      m->arg[3] = lambda;
      m->res[0] = values;
      return self->calc_function(m, "nlp_hess_l") == 0;
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

    // Full callback: pull the current iterate out of POUNCE and drive
    // casadi's `iteration_callback` with it.
    const PounceInterface* self = m->self;
    if (!self->fcallback_.is_null()) {
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
      self->fcallback_(m->arg, m->res, m->iw, m->w, 0);
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

    SetIntermediateCallback(prob, &PounceInterface::cb_iter);

    enum ApplicationReturnStatus st = IpoptSolve(
      prob, m->xk.data(), ng ? m->gk.data() : nullptr, &m->obj,
      ng ? m->lam_g.data() : nullptr, m->z_L.data(), m->z_U.data(),
      static_cast<UserDataPtr>(m));

    m->return_status = static_cast<int>(st);
    m->iter = GetIpoptIterCount(prob);
    m->t_solve = GetIpoptSolveTime(prob);
    FreeIpoptProblem(prob);

    // Write back to casadi's nlpsol data layout
    casadi_copy(m->xk.data(), n, d_nlp->z);
    casadi_copy(m->gk.data(), ng, d_nlp->z + n);
    d_nlp->objective = m->obj;
    for (int i = 0; i < n; ++i) d_nlp->lam[i] = m->z_U[i] - m->z_L[i];
    casadi_copy(m->lam_g.data(), ng, d_nlp->lam + n);

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

  extern "C"
  int casadi_register_nlpsol_pounce(Nlpsol::Plugin* plugin) {
    plugin->creator = PounceInterface::creator;
    plugin->name = "pounce";
    plugin->doc = PounceInterface::meta_doc.c_str();
    plugin->version = CASADI_VERSION;
    plugin->options = &PounceInterface::options_;
    plugin->deserialize = nullptr;
    return 0;
  }

  extern "C"
  void casadi_load_nlpsol_pounce() {
    Nlpsol::registerPlugin(casadi_register_nlpsol_pounce);
  }

} // namespace casadi
