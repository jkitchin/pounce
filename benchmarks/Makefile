# Benchmark suite Makefile.
#
# This Makefile drives the per-suite benchmarks under benchmarks/.
# The root Makefile delegates to this one for all benchmark targets so there's
# one source of truth. Cargo commands are invoked with --manifest-path pointing
# at the repo root so this Makefile works regardless of the user's CWD.
#
# See benchmarks/README.md for suite descriptions and per-suite READMEs for
# details on each benchmark.

SHELL = /bin/bash
REPO_ROOT := $(shell cd .. && pwd)
CARGO_MANIFEST := $(REPO_ROOT)/Cargo.toml
CARGO := cargo

# ---- benchmark .nl data root --------------------------------------------
# The .nl problem inputs are large (~2 GB) and gitignored. By default they
# live in the in-repo benchmarks/ tree, but POUNCE_BENCH_DATA can point them
# at a stable, syncable location outside the checkout (e.g. a Dropbox folder)
# so the same dataset is shared across machines and runs work from any CWD:
#
#     export POUNCE_BENCH_DATA="$HOME/Dropbox/pounce-bench-data"
#
# The directory mirrors the in-repo layout (<root>/vanderbei/nl/*.nl,
# <root>/gas/*.nl, <root>/cho/nl_export_results/*.nl, ...). Only the .nl
# inputs (and the .sol scratch the solver writes beside them) live here;
# pounce.json results, logs/, the committed ipopt_ma57.json references, and
# the download/source caches (mps/, data/, source/) stay in the repo.
POUNCE_BENCH_DATA ?= $(REPO_ROOT)/benchmarks

# ---- ipopt-ma57 for the .nl-driven suites --------------------------------
# Every suite is now .nl-driven and compares against the locally-built
# MA57 Ipopt AMPL executable at ref/Ipopt/install-ma57/bin/ipopt (built
# locally; no build recipe is tracked, see `build-ipopt-ma57`). This
# matches POUNCE-MA57 against the same linear-solver family. No pkg-config / libipopt FFI is
# needed any more — the compiled CUTEst and large_scale Rust harnesses
# that linked libipopt have been retired.
IPOPT_MA57_INSTALL ?= $(REPO_ROOT)/ref/Ipopt/install-ma57
IPOPT_MA57_BIN     ?= $(IPOPT_MA57_INSTALL)/bin/ipopt
# Ipopt's compiled default linear solver is ma27 even in this HSL build,
# so the driver passes `linear_solver=$(IPOPT_LINEAR_SOLVER)` explicitly —
# otherwise the "ma57" reference would silently run ma27. Exported so the
# run_nl_bench.sh sub-shells pick it up.
IPOPT_LINEAR_SOLVER ?= ma57
export IPOPT_LINEAR_SOLVER

.PHONY: help benchmark benchmark-rerun benchmark-report build-ipopt-ma57 \
	vanderbei-run vanderbei-rerun \
	electrolyte-run electrolyte-rerun grid-run grid-rerun cho-run cho-rerun \
	grid-ecf-run grid-ecf-rerun grid-polar-run grid-polar-rerun \
	grid-ecf-generate \
	large-scale large-scale-generate large-scale-rerun \
	mittelmann-run mittelmann-rerun mittelmann-check \
	gas-run gas-rerun water-run water-rerun \
	qp-run qp-rerun qp-generate \
	lp-run lp-rerun lp-generate \
	lp-convex-run lp-convex-rerun qp-convex-run qp-convex-rerun \
	lbfgs-run lbfgs-rerun \
	lpopt-run lpopt-rerun lpopt-generate \
	ipopt-reference ipopt-reference-provenance \
	$(addprefix ipopt-ref-,$(REF_SUITES)) \
	gams-bench gams-rerun \
	warmstart-run warmstart-quick warmstart-selftest warmstart-report \
	warmstart-ipopt warmstart-transfer warmstart-composite warmstart-all \
	mpcc-selftest mpcc-smoke mpcc-run mpcc-report mpcc-manifest \
	flash-selftest flash-smoke flash-run \
	clean-bench clean-bench-large-scale clean-bench-nl clean-bench-report \
	pounce-bin

# Show available targets
help:
	@echo "Benchmark suite targets (run from benchmarks/ or via root Makefile shims)."
	@echo ""
	@echo "  benchmark         Release sweep: rerun pounce on each suite (incremental), then composite report"
	@echo "  benchmark-rerun   Wipe all pounce results then full rebuild (force everything)"
	@echo "  benchmark-report  Regenerate benchmarks/BENCHMARK_REPORT.md only (no suites rerun)"
	@echo ""
	@echo "The ipopt-ma57 reference is decoupled from releases — run it once and commit it:"
	@echo "  ipopt-reference        Run ipopt-ma57 on every suite -> committed <suite>/ipopt_ma57.json + provenance"
	@echo "  ipopt-ref-<suite>      Refresh the ipopt reference for one suite only"
	@echo "  (releases run pounce only; the report compares against the saved reference)"
	@echo ""
	@echo "Per-suite pounce targets (each *-run is incremental; *-rerun wipes its pounce.json then runs):"
	@echo ""
	@echo "  vanderbei-run   / -rerun   Vanderbei CUTE-in-AMPL .nl suite (733 problems)"
	@echo "  electrolyte-run / -rerun   Electrolyte thermodynamics benchmark"
	@echo "  grid-run        / -rerun   AC optimal power flow (electrical grid) benchmark"
	@echo "  grid-ecf-run    / -rerun   AC-OPF, equivalent-circuit formulation"
	@echo "  grid-polar-run  / -rerun   AC-OPF, matched polar control"
	@echo "  grid-ecf-generate          regenerate BOTH .nl sets above"
	@echo "  cho-run         / -rerun   CHO parameter estimation benchmark"
	@echo "  large-scale     / large-scale-rerun  Synthetic large-scale NLPs (.nl from generate_nl.py)"
	@echo "  large-scale-generate       (Re)generate the large_scale .nl files only (needs pyomo)"
	@echo "  mittelmann-run  / -rerun   Mittelmann ampl-nlp .nl suite (shared driver; .nl from POUNCE_BENCH_DATA)"
	@echo "  gas-run         / -rerun   Gas pipeline NLPs (AMPL .nl files)"
	@echo "  water-run       / -rerun   Water distribution network NLPs (AMPL .nl files)"
	@echo "  qp-run          / -rerun   Maros-Meszaros convex QP suite (.nl from generate_nl.py)"
	@echo "  qp-generate                (Re)generate the qp .nl files only (needs pyomo/scipy; downloads .mat)"
	@echo "  lp-run          / -rerun   Netlib + small Meszaros LP validation set (.nl from generate_nl.py)"
	@echo "  lp-generate                (Re)generate the lp .nl files only (needs highspy/pyomo; downloads MPS)"
	@echo "  lp-convex-run   / -rerun   LP head-to-head: pounce-convex (lp-ipm) vs pounce-nlp (reuses lp .nl)"
	@echo "  qp-convex-run   / -rerun   QP head-to-head: pounce-convex (qp-ipm) vs pounce-nlp (reuses qp .nl)"
	@echo "  lbfgs-run       / -rerun   Hessian head-to-head: exact vs limited-memory (reuses mittelmann .nl)"
	@echo "  warmstart-run              Warm-start suite: parametric NLP sequences, cold vs warm (needs the built Python ext)"
	@echo "  warmstart-quick            Warm-start suite, short sweep (3 families, one scale)"
	@echo "  warmstart-selftest         Finite-difference check of the warm-start families (no solver needed)"
	@echo "  warmstart-report           Re-render warmstart/results.md from the existing results.json"
	@echo "  warmstart-ipopt            Warm-start suite under the external Ipopt adapter (needs cyipopt)"
	@echo "  warmstart-transfer         Changed-structure arms: horizon shift and mesh prolongation"
	@echo "  warmstart-composite        Render the pounce#611 composite report into dev-notes/"
	@echo "  warmstart-all              Every warm-start run above, then the composite report"
	@echo "  mpcc-selftest              MPCC suite: derivatives, oracle, classifier, manifest (no solver needed)"
	@echo "  mpcc-smoke                 MPCC suite: deterministic asserted subset (~10 s, needs the built Python ext)"
	@echo "  mpcc-run                   MPCC suite: the full gh#794 configuration, writes results-full.{json,md}"
	@echo "  mpcc-report                Re-render mpcc/results-full.md from the existing results-full.json"
	@echo "  mpcc-manifest              Regenerate mpcc/manifest.json after a corpus edit"
	@echo "  flash-selftest             Gate 1 flash: derivatives, oracle cross-check, regime coverage (no solver)"
	@echo "  flash-smoke                Gate 1 flash: asserted subset (~15 s, needs the built Python ext)"
	@echo "  flash-run                  Gate 1 flash: full gh#776 configuration, writes flash/results-full.{json,md}"
	@echo "  lpopt-run       / -rerun   HARD Mittelmann lpopt LP subset (stress tier; use BENCH_TIMELIMIT=1800)"
	@echo "  lpopt-generate             (Re)generate the lpopt .nl files only (downloads MPS from plato lptestset)"
	@echo "  gams-bench      / gams-rerun  GAMS solver-link smoke check (10 problems; not in the report)"
	@echo ""
	@echo "  clean-bench       Wipe pounce results/logs across every suite (keeps the committed ipopt reference)"
	@echo "  clean-bench-large-scale Wipe large_scale pounce.json"
	@echo "  clean-bench-nl         Wipe per-problem .sol outputs + pounce.json (.nl suites)"
	@echo "  clean-bench-report     Remove benchmarks/BENCHMARK_REPORT.md"
	@echo ""
	@echo "All suites feed into benchmarks/BENCHMARK_REPORT.md via benchmark_report.py."

# Convenience shim — the actual recipe lives in benchmarks/mittelmann/Makefile.
build-ipopt-ma57:
	@echo "No build recipe for the Ipopt-MA57 reference binary is tracked." >&2
	@echo "Build Ipopt with CoinHSL and the AMPL (ASL) interface, then run" >&2
	@echo "  make -C benchmarks IPOPT_MA57_BIN=<path>/bin/ipopt ipopt-ref-<suite>" >&2
	@exit 1

# ---- Domain benchmarks ----
# Every suite is .nl-driven through the shared driver
# (benchmarks/scripts/run_nl_bench.sh). The ipopt-ma57 reference and the
# per-release pounce run are DECOUPLED:
#
#   * <suite>-run                 runs ONLY pounce  -> benchmarks/<suite>/pounce.json
#                                 (regenerated every release; gitignored)
#   * ipopt-reference / ipopt-ref-<suite>
#                                 runs ONLY ipopt-ma57 -> benchmarks/<suite>/ipopt_ma57.json
#                                 (committed; run rarely — see below)
#
# benchmark_report.py merges the fresh pounce.json with the saved
# ipopt_ma57.json per suite. ipopt-ma57 is slow and its results don't
# change between pounce iterations, so the reference is generated once
# (per machine / ipopt build) and committed; releases rerun only pounce.
NL_DRIVER       := bash $(REPO_ROOT)/benchmarks/scripts/run_nl_bench.sh
POUNCE_BIN      := $(REPO_ROOT)/target/release/pounce
BENCH_TIMELIMIT ?= 300
LS_DIR          := $(REPO_ROOT)/benchmarks/large_scale
MITT_DIR        := $(REPO_ROOT)/benchmarks/mittelmann
QP_DIR          := $(REPO_ROOT)/benchmarks/qp
LP_DIR          := $(REPO_ROOT)/benchmarks/lp
LPOPT_DIR       := $(REPO_ROOT)/benchmarks/lpopt
LP_CONVEX_DIR   := $(REPO_ROOT)/benchmarks/lp_convex
QP_CONVEX_DIR   := $(REPO_ROOT)/benchmarks/qp_convex
LBFGS_DIR       := $(REPO_ROOT)/benchmarks/lbfgs

# Ensure POUNCE_BIN exists; helps when a target is run before any cargo
# build. Cheap when already built. Existence-only on purpose, so a
# single-suite target run against a hand-built binary still works -- a full
# sweep gets the `pounce-bin` step below instead, which is the one that
# guarantees the binary matches the tree.
$(POUNCE_BIN):
	$(CARGO) build --release --manifest-path $(CARGO_MANIFEST) --bin pounce

# Suite -> .nl directory map, shared by the pounce file-targets and the
# ipopt-ref-% reference targets.
nldir_vanderbei   := $(POUNCE_BENCH_DATA)/vanderbei/nl
nldir_electrolyte := $(POUNCE_BENCH_DATA)/electrolyte/nl
nldir_grid        := $(POUNCE_BENCH_DATA)/grid/nl
# The ECF formulation comparison: two .nl sets from one generator, differing
# only in formulation. See benchmarks/grid_ecf/README.md.
nldir_grid_ecf    := $(POUNCE_BENCH_DATA)/grid_ecf/nl
nldir_grid_polar  := $(POUNCE_BENCH_DATA)/grid_polar/nl
nldir_cho         := $(POUNCE_BENCH_DATA)/cho/nl_export_results
nldir_large_scale := $(POUNCE_BENCH_DATA)/large_scale/nl
nldir_gas         := $(POUNCE_BENCH_DATA)/gas
nldir_water       := $(POUNCE_BENCH_DATA)/water
nldir_mittelmann  := $(POUNCE_BENCH_DATA)/mittelmann/nl
nldir_qp          := $(POUNCE_BENCH_DATA)/qp/nl
nldir_lp          := $(POUNCE_BENCH_DATA)/lp/nl
nldir_lpopt       := $(POUNCE_BENCH_DATA)/lpopt/nl

# --- pounce release runs (incremental, pounce-only) ---
# Each <suite>/pounce.json rebuilds when the pounce binary or the suite's
# .nl change; <suite>-rerun forces by wiping first.

# Vanderbei CUTE-in-AMPL suite (733 .nl problems) — the canonical NLP
# collection, replacing the retired compiled CUTEst suite. See
# benchmarks/vanderbei/README.md for how the .nl were produced.
$(REPO_ROOT)/benchmarks/vanderbei/pounce.json: $(POUNCE_BIN) \
	$(wildcard $(nldir_vanderbei)/*.nl)
	$(NL_DRIVER) vanderbei $(nldir_vanderbei) $@ $(POUNCE_BIN) - $(BENCH_TIMELIMIT) pounce

vanderbei-run: $(REPO_ROOT)/benchmarks/vanderbei/pounce.json

vanderbei-rerun:
	rm -f $(REPO_ROOT)/benchmarks/vanderbei/pounce.json
	$(MAKE) vanderbei-run

$(REPO_ROOT)/benchmarks/electrolyte/pounce.json: $(POUNCE_BIN) \
	$(wildcard $(nldir_electrolyte)/*.nl)
	$(NL_DRIVER) electrolyte $(nldir_electrolyte) $@ $(POUNCE_BIN) - $(BENCH_TIMELIMIT) pounce

electrolyte-run: $(REPO_ROOT)/benchmarks/electrolyte/pounce.json

electrolyte-rerun:
	rm -f $(REPO_ROOT)/benchmarks/electrolyte/pounce.json
	$(MAKE) electrolyte-run

$(REPO_ROOT)/benchmarks/grid/pounce.json: $(POUNCE_BIN) \
	$(wildcard $(nldir_grid)/*.nl)
	$(NL_DRIVER) grid $(nldir_grid) $@ $(POUNCE_BIN) - $(BENCH_TIMELIMIT) pounce

grid-run: $(REPO_ROOT)/benchmarks/grid/pounce.json

grid-rerun:
	rm -f $(REPO_ROOT)/benchmarks/grid/pounce.json
	$(MAKE) grid-run

# --- ECF vs polar formulation comparison ---------------------------------
# Both .nl sets come from benchmarks/grid_ecf/ecf_nl_export.py. Regenerate
# them together: a comparison across two different generator invocations
# (different --thermal, say) is not a comparison. Each nl dir carries a
# manifest.json recording the conventions it was built under.
GRID_ECF_GEN := $(REPO_ROOT)/benchmarks/grid_ecf/ecf_nl_export.py

grid-ecf-generate:
	$(PYTHON) $(GRID_ECF_GEN) --form ecf
	$(PYTHON) $(GRID_ECF_GEN) --form polar

$(REPO_ROOT)/benchmarks/grid_ecf/pounce.json: $(POUNCE_BIN) \
	$(wildcard $(nldir_grid_ecf)/*.nl)
	$(NL_DRIVER) grid_ecf $(nldir_grid_ecf) $@ $(POUNCE_BIN) - $(BENCH_TIMELIMIT) pounce

grid-ecf-run: $(REPO_ROOT)/benchmarks/grid_ecf/pounce.json

grid-ecf-rerun:
	rm -f $(REPO_ROOT)/benchmarks/grid_ecf/pounce.json
	$(MAKE) grid-ecf-run

$(REPO_ROOT)/benchmarks/grid_polar/pounce.json: $(POUNCE_BIN) \
	$(wildcard $(nldir_grid_polar)/*.nl)
	$(NL_DRIVER) grid_polar $(nldir_grid_polar) $@ $(POUNCE_BIN) - $(BENCH_TIMELIMIT) pounce

grid-polar-run: $(REPO_ROOT)/benchmarks/grid_polar/pounce.json

grid-polar-rerun:
	rm -f $(REPO_ROOT)/benchmarks/grid_polar/pounce.json
	$(MAKE) grid-polar-run

$(REPO_ROOT)/benchmarks/cho/pounce.json: $(POUNCE_BIN) \
	$(wildcard $(nldir_cho)/*.nl)
	$(NL_DRIVER) cho $(nldir_cho) $@ $(POUNCE_BIN) - $(BENCH_TIMELIMIT) pounce

cho-run: $(REPO_ROOT)/benchmarks/cho/pounce.json

cho-rerun:
	rm -f $(REPO_ROOT)/benchmarks/cho/pounce.json
	$(MAKE) cho-run

# Large scale: five large, sparse synthetic NLPs (chained Rosenbrock,
# Bratu, optimal control, Poisson control, sparse QP) emitted as .nl by
# generate_nl.py (Pyomo). The sparseqp stub stands in as the generation
# stamp. Resize/regenerate via `make large-scale-generate`.
$(nldir_large_scale)/sparseqp.nl: $(LS_DIR)/generate_nl.py
	python3 $(LS_DIR)/generate_nl.py --out-dir $(nldir_large_scale)

large-scale-generate: $(nldir_large_scale)/sparseqp.nl

$(LS_DIR)/pounce.json: $(POUNCE_BIN) $(nldir_large_scale)/sparseqp.nl
	$(NL_DRIVER) large_scale $(nldir_large_scale) $@ $(POUNCE_BIN) - $(BENCH_TIMELIMIT) pounce

large-scale: $(LS_DIR)/pounce.json

large-scale-rerun:
	rm -f $(LS_DIR)/pounce.json
	$(MAKE) large-scale

$(REPO_ROOT)/benchmarks/gas/pounce.json: $(POUNCE_BIN) \
	$(wildcard $(nldir_gas)/*.nl)
	$(NL_DRIVER) gas $(nldir_gas) $@ $(POUNCE_BIN) - $(BENCH_TIMELIMIT) pounce

gas-run: $(REPO_ROOT)/benchmarks/gas/pounce.json

gas-rerun:
	rm -f $(REPO_ROOT)/benchmarks/gas/pounce.json
	$(MAKE) gas-run

$(REPO_ROOT)/benchmarks/water/pounce.json: $(POUNCE_BIN) \
	$(wildcard $(nldir_water)/*.nl)
	$(NL_DRIVER) water $(nldir_water) $@ $(POUNCE_BIN) - $(BENCH_TIMELIMIT) pounce

water-run: $(REPO_ROOT)/benchmarks/water/pounce.json

water-rerun:
	rm -f $(REPO_ROOT)/benchmarks/water/pounce.json
	$(MAKE) water-run

# Maros-Meszaros convex QP suite (~138 problems) emitted as .nl by
# generate_nl.py, which downloads the qpsolvers .mat mirror of the
# Maros-Meszaros set and converts each QP (1/2 x'Px + q'x + r s.t. linear
# rows and bounds) into a Pyomo model written as .nl. QPTEST.nl stands in
# as the generation stamp (like large_scale's sparseqp.nl). Regenerate via
# `make qp-generate`.
$(nldir_qp)/QPTEST.nl: $(QP_DIR)/generate_nl.py
	python3 $(QP_DIR)/generate_nl.py --out-dir $(nldir_qp) --data-dir $(QP_DIR)/data

qp-generate: $(nldir_qp)/QPTEST.nl

$(QP_DIR)/pounce.json: $(POUNCE_BIN) $(nldir_qp)/QPTEST.nl
	$(NL_DRIVER) qp $(nldir_qp) $@ $(POUNCE_BIN) - $(BENCH_TIMELIMIT) pounce

qp-run: $(QP_DIR)/pounce.json

qp-rerun:
	rm -f $(QP_DIR)/pounce.json
	$(MAKE) qp-run

# LP validation suite — the classic Netlib LP set plus a size-filtered
# subset of Hans Mészáros's LP test set: small LPs with published optima
# that pounce should solve to optimality quickly. generate_nl.py fetches
# the sources, size-screens, and converts to .nl (HiGHS parse -> Pyomo ->
# .nl, see mps_to_nl.py). afiro.nl (the canonical smallest LP) stands in
# as the generation stamp. Regenerate via `make lp-generate`.
$(nldir_lp)/afiro.nl: $(LP_DIR)/generate_nl.py $(LP_DIR)/mps_to_nl.py
	python3 $(LP_DIR)/generate_nl.py --out-dir $(nldir_lp)

lp-generate: $(nldir_lp)/afiro.nl

$(LP_DIR)/pounce.json: $(POUNCE_BIN) $(nldir_lp)/afiro.nl
	$(NL_DRIVER) lp $(nldir_lp) $@ $(POUNCE_BIN) - $(BENCH_TIMELIMIT) pounce

lp-run: $(LP_DIR)/pounce.json

lp-rerun:
	rm -f $(LP_DIR)/pounce.json
	$(MAKE) lp-run

# Head-to-head suites: the dedicated convex interior-point solver
# (pounce-convex, via solver_selection=lp-ipm / qp-ipm) vs the general NLP
# filter-IPM (solver_selection=nlp) on the SAME .nl problems as the lp/qp
# suites above. Two pounce arms (same binary), two result files: convex.json
# and nlp.json. No ipopt reference — this is pounce-vs-pounce. They reuse the
# lp/qp .nl data (nldir_lp / nldir_qp); a missing .nl regenerates via the
# existing lp-generate / qp-generate stamps (afiro.nl / QPTEST.nl).

# --- LP: pounce-convex (lp-ipm) vs pounce-nlp ---
$(LP_CONVEX_DIR)/convex.json: $(POUNCE_BIN) $(nldir_lp)/afiro.nl
	$(NL_DRIVER) lp_convex $(nldir_lp) $@ $(POUNCE_BIN) - $(BENCH_TIMELIMIT) \
		pounce solver_selection=lp-ipm convex

$(LP_CONVEX_DIR)/nlp.json: $(POUNCE_BIN) $(nldir_lp)/afiro.nl
	$(NL_DRIVER) lp_convex $(nldir_lp) $@ $(POUNCE_BIN) - $(BENCH_TIMELIMIT) \
		pounce solver_selection=nlp nlp

lp-convex-run: $(LP_CONVEX_DIR)/convex.json $(LP_CONVEX_DIR)/nlp.json

lp-convex-rerun:
	rm -f $(LP_CONVEX_DIR)/convex.json $(LP_CONVEX_DIR)/nlp.json
	$(MAKE) lp-convex-run

# --- QP: pounce-convex (qp-ipm) vs pounce-nlp ---
$(QP_CONVEX_DIR)/convex.json: $(POUNCE_BIN) $(nldir_qp)/QPTEST.nl
	$(NL_DRIVER) qp_convex $(nldir_qp) $@ $(POUNCE_BIN) - $(BENCH_TIMELIMIT) \
		pounce solver_selection=qp-ipm convex

$(QP_CONVEX_DIR)/nlp.json: $(POUNCE_BIN) $(nldir_qp)/QPTEST.nl
	$(NL_DRIVER) qp_convex $(nldir_qp) $@ $(POUNCE_BIN) - $(BENCH_TIMELIMIT) \
		pounce solver_selection=nlp nlp

qp-convex-run: $(QP_CONVEX_DIR)/convex.json $(QP_CONVEX_DIR)/nlp.json

qp-convex-rerun:
	rm -f $(QP_CONVEX_DIR)/convex.json $(QP_CONVEX_DIR)/nlp.json
	rm -f $(LBFGS_DIR)/exact.json $(LBFGS_DIR)/limited.json
	$(MAKE) qp-convex-run

# ---- Hessian head-to-head: exact vs limited-memory -----------------------
# L-BFGS is not a rare opt-in: the Python frontend and the CasADi plugin
# both select `hessian_approximation=limited-memory` automatically when no
# exact Lagrangian Hessian is available, so a corpus measured only with an
# exact Hessian leaves that path unbenchmarked. That gap shipped gh#677.
# Two pounce arms (same binary) over the SAME mittelmann .nl problems, two
# result files: exact.json and limited.json. No ipopt reference — this is
# pounce-vs-pounce, and the interesting number is the cost of giving up the
# exact Hessian, not a cross-solver comparison.
$(LBFGS_DIR)/exact.json: $(POUNCE_BIN) $(wildcard $(nldir_mittelmann)/*.nl)
	$(NL_DRIVER) lbfgs $(nldir_mittelmann) $@ $(POUNCE_BIN) - $(BENCH_TIMELIMIT) \
		pounce hessian_approximation=exact exact

$(LBFGS_DIR)/limited.json: $(POUNCE_BIN) $(wildcard $(nldir_mittelmann)/*.nl)
	$(NL_DRIVER) lbfgs $(nldir_mittelmann) $@ $(POUNCE_BIN) - $(BENCH_TIMELIMIT) \
		pounce hessian_approximation=limited-memory lbfgs

lbfgs-run: $(LBFGS_DIR)/exact.json $(LBFGS_DIR)/limited.json

lbfgs-rerun:
	rm -f $(LBFGS_DIR)/exact.json $(LBFGS_DIR)/limited.json
	$(MAKE) lbfgs-run

# ---- warm-start suite ----------------------------------------------------
# The only suite that is not .nl-driven: warm starting is a property of a
# *sequence* of related solves, and carrying a working set between solves
# needs an in-process handle, which the .nl CLI path does not offer. The
# harness therefore drives the Python extension directly — build it first
# with `cd python && maturin develop --release`. PYTHON can be pointed at
# whichever interpreter has that extension installed.
#
# Outputs (warmstart/results.json, warmstart/results.md) are regenerated
# per run and gitignored, like every other suite's results.
PYTHON ?= python3

warmstart-selftest:
	cd $(CURDIR) && $(PYTHON) -m warmstart.selftest

warmstart-run: warmstart-selftest
	cd $(CURDIR) && $(PYTHON) -m warmstart.run -v

warmstart-quick:
	cd $(CURDIR) && $(PYTHON) -m warmstart.run --quick -v

warmstart-report:
	cd $(CURDIR) && $(PYTHON) -m warmstart.report \
		warmstart/results.json -o warmstart/results.md

# --- MPCC benchmark (gh#794) ------------------------------------------
#
# Gate 0 of gh#776: the smallest corpus that can separate a source
# formulation deficiency from a lowering deficiency from a POUNCE
# deficiency on complementarity-constrained problems. Like warmstart it
# drives the Python extension in process (build it with `make
# python-ext` from the repo root) and needs SciPy for its independent
# oracle and its stationarity classifier.
#
# `mpcc-selftest` needs no solver at all and is the gate on the others:
# it finite-difference-checks every declared derivative, re-derives every
# expected optimum by branch enumeration, and checks that the
# stationarity classifier still discriminates S from M from C.
#
# Outputs (mpcc/results-*.json, mpcc/results-*.md) are regenerated per run
# and gitignored; mpcc/manifest.json and mpcc/schema.json are tracked.

mpcc-selftest:
	cd $(CURDIR) && $(PYTHON) -m mpcc.selftest

mpcc-smoke: mpcc-selftest
	cd $(CURDIR) && $(PYTHON) -m mpcc.run --smoke

mpcc-run: mpcc-selftest
	cd $(CURDIR) && $(PYTHON) -m mpcc.run --full -v

mpcc-report:
	cd $(CURDIR) && $(PYTHON) -m mpcc.report \
		mpcc/results-full.json -o mpcc/results-full.md

mpcc-manifest:
	cd $(CURDIR) && $(PYTHON) -m mpcc.run --write-manifest

# ---- Gate 1 flash fixture (gh#776) ---------------------------------------
# The phase-changing model Gate 0 gated. `flash-selftest` needs no solver
# and is the gate on the others: it finite-difference-checks the declared
# derivatives, asserts the path still crosses all three phase regimes with
# both switch points interior, and -- the load-bearing one -- substitutes
# the independent oracle's answer into the model's own rows and requires
# it to satisfy them.
#
# The smoke subset also runs in CI as python/tests/test_flash_mpcc.py.
# Outputs (flash/results-*.json, flash/results-*.md) are regenerated per
# run and gitignored; flash/schema.json is tracked.

flash-selftest:
	cd $(CURDIR) && $(PYTHON) -m flash.selftest

flash-smoke: flash-selftest
	cd $(CURDIR) && $(PYTHON) -m flash.run --smoke

flash-run: flash-selftest
	cd $(CURDIR) && $(PYTHON) -m flash.run --full -v

# pounce#611. The external arm needs cyipopt, which has no PyPI wheel:
#   apt-get install -y coinor-libipopt-dev liblapack-dev libblas-dev
#   pip install cython && pip install --no-build-isolation cyipopt
# `warmstart.run --solver ipopt` exits with those instructions if it is
# missing, so this target fails loudly rather than skipping quietly.
warmstart-ipopt:
	cd $(CURDIR) && $(PYTHON) -m warmstart.run --solver ipopt \
		--scales all --arms all --out warmstart/results-ipopt.json --no-report

warmstart-transfer:
	cd $(CURDIR) && $(PYTHON) -m warmstart.transfer --experiment all \
		--out warmstart/transfer.json

# Regenerates dev-notes/warm-start-611-composite.{md,json} from whichever
# of the three raw result files exist. A missing input produces a section
# that says so, not a silently shorter report.
warmstart-composite:
	cd $(CURDIR) && $(PYTHON) -m warmstart.composite \
		--results warmstart/results.json \
		--ipopt warmstart/results-ipopt.json \
		--transfer warmstart/transfer.json \
		-o ../dev-notes/warm-start-611-composite.md \
		--json-out ../dev-notes/warm-start-611-composite.json

warmstart-all: warmstart-run warmstart-ipopt warmstart-transfer warmstart-composite

# Mittelmann lpopt LP suite — the curated HARD subset of Hans Mittelmann's
# lpopt benchmark (https://plato.asu.edu/ftp/lpopt.html). These are large,
# degenerate LPs (a stress tier; even HiGHS/ipopt time out at short
# limits). build_subset.py fetches the smaller disclosed instances as
# bzip2'd MPS from the plato lptestset mirror and converts each (HiGHS ->
# Pyomo -> .nl), size-gating anything too large. qap15.nl is the
# generation stamp. Run with a long time limit (e.g. BENCH_TIMELIMIT=1800).
$(nldir_lpopt)/qap15.nl: $(LPOPT_DIR)/build_subset.py $(LPOPT_DIR)/mps_to_nl.py
	python3 $(LPOPT_DIR)/build_subset.py --out-dir $(nldir_lpopt)

lpopt-generate: $(nldir_lpopt)/qap15.nl

$(LPOPT_DIR)/pounce.json: $(POUNCE_BIN) $(nldir_lpopt)/qap15.nl
	$(NL_DRIVER) lpopt $(nldir_lpopt) $@ $(POUNCE_BIN) - $(BENCH_TIMELIMIT) pounce

lpopt-run: $(LPOPT_DIR)/pounce.json

lpopt-rerun:
	rm -f $(LPOPT_DIR)/pounce.json
	$(MAKE) lpopt-run

# Mittelmann ampl-nlp suite. The .nl files are not generated here: the
# translated set lives in the benchmark data root ($(POUNCE_BENCH_DATA)).
# This only checks it is there, so a missing data root fails with a message
# instead of an empty run. (It used to call a benchmarks/mittelmann/Makefile
# that was never tracked, so `mittelmann-run` failed from any fresh checkout.)
mittelmann-check:
	@n=$$(ls $(nldir_mittelmann)/*.nl 2>/dev/null | wc -l | tr -d ' '); \
	if [ "$$n" -eq 0 ]; then \
	  echo "mittelmann: no .nl files in $(nldir_mittelmann); set POUNCE_BENCH_DATA to the benchmark data root" >&2; \
	  exit 1; \
	fi; \
	echo "mittelmann: $$n .nl files in $(nldir_mittelmann)"

$(MITT_DIR)/pounce.json: $(POUNCE_BIN) $(wildcard $(nldir_mittelmann)/*.nl)
	$(NL_DRIVER) mittelmann $(nldir_mittelmann) $@ $(POUNCE_BIN) - $(BENCH_TIMELIMIT) pounce

mittelmann-run: mittelmann-check $(MITT_DIR)/pounce.json

mittelmann-rerun:
	rm -f $(MITT_DIR)/pounce.json
	$(MAKE) mittelmann-run

# --- ipopt-ma57 reference (run rarely; committed) ---
# `make ipopt-reference` runs ipopt-ma57 across every suite and writes the
# committed benchmarks/<suite>/ipopt_ma57.json plus a provenance stamp.
# Run it once per machine / ipopt build (and after adding problems), then
# commit the results — every release thereafter reruns only pounce and the
# report compares against this saved reference. `make ipopt-ref-<suite>`
# refreshes a single suite. Override the per-instance cap with
# `make ipopt-reference REF_TIMELIMIT=120`.
REF_SUITES    := vanderbei electrolyte grid cho large_scale gas water mittelmann qp lp lpopt
REF_TIMELIMIT ?= $(BENCH_TIMELIMIT)
IPOPT_PROVENANCE := $(REPO_ROOT)/benchmarks/ipopt_ma57.provenance.json

ipopt-ref-%:
	$(NL_DRIVER) $* $(nldir_$*) \
		$(REPO_ROOT)/benchmarks/$*/ipopt_ma57.json \
		- $(IPOPT_MA57_BIN) $(REF_TIMELIMIT) ipopt

# Suites whose .nl must be produced before ipopt can run on them.
ipopt-ref-large_scale: $(nldir_large_scale)/sparseqp.nl
ipopt-ref-mittelmann:  mittelmann-check
ipopt-ref-qp:          $(nldir_qp)/QPTEST.nl
ipopt-ref-lp:          $(nldir_lp)/afiro.nl
ipopt-ref-lpopt:       $(nldir_lpopt)/qap15.nl

ipopt-reference: $(addprefix ipopt-ref-,$(REF_SUITES)) ipopt-reference-provenance
	@echo "ipopt-ma57 reference written."
	@echo "Commit benchmarks/*/ipopt_ma57.json + benchmarks/ipopt_ma57.provenance.json."

# Record which ipopt produced the saved reference, on what machine and
# when, so the report can attribute its Ipopt column honestly.
ipopt-reference-provenance:
	@{ \
	  echo '{'; \
	  printf '  "generated": "%s",\n' "$$(date '+%Y-%m-%d %H:%M:%S %Z')"; \
	  printf '  "host": "%s",\n'      "$$(uname -n)"; \
	  printf '  "platform": "%s",\n'  "$$(uname -srm)"; \
	  printf '  "ipopt_bin": "%s",\n' "$(IPOPT_MA57_BIN)"; \
	  printf '  "ipopt_version": "%s",\n' "$$($(IPOPT_MA57_BIN) --version 2>/dev/null | head -1)"; \
	  printf '  "linear_solver": "%s (via ref/Ipopt/install-ma57)",\n' "$(IPOPT_LINEAR_SOLVER)"; \
	  printf '  "timelimit": %s,\n'   "$(REF_TIMELIMIT)"; \
	  printf '  "threads": {"OMP_NUM_THREADS": "%s", "OPENBLAS_NUM_THREADS": "%s", "VECLIB_MAXIMUM_THREADS": "%s", "RAYON_NUM_THREADS": "%s"},\n' \
	    "$${OMP_NUM_THREADS-unset}" "$${OPENBLAS_NUM_THREADS-unset}" \
	    "$${VECLIB_MAXIMUM_THREADS-unset}" "$${RAYON_NUM_THREADS-unset}"; \
	  printf '  "git_sha": "%s"\n'    "$$(git -C $(REPO_ROOT) rev-parse --short HEAD 2>/dev/null)"; \
	  echo '}'; \
	} > $(IPOPT_PROVENANCE)
	@echo "wrote benchmarks/ipopt_ma57.provenance.json"

# ---- GAMS solver-link smoke check ----
# The GAMS nlpbench suite is no longer aggregated into the composite
# report: its problem coverage duplicates the .nl suites (princetonlib ≈
# vanderbei, GAMS mittelmann ≈ ampl-nlp mittelmann/, powerflow ≈ grid/)
# and it was compared on the same pounce-vs-ipopt axis as everything
# else. What it *uniquely* exercises is the GAMS solver-link path
# (pounce-cinterface + libGamsPounce via the GMO/GEV protocol), so this
# target keeps only a 10-problem liveness check (bench-smoke, seconds)
# proving that path still works. It does not feed benchmark_report.py.
#
# For a fuller GAMS comparison run on demand:
#   make -C gams/nlpbench bench-release   # globallib + powerflow, ~2h
#   make -C gams/nlpbench bench-full      # all eight testsets, ~14h
#
# The target FORCES a re-run (`clean-bench-smoke` first) and then validates
# the traces it produced. Both halves are load-bearing (gh#747):
#
#   * without the clean, the vendored `runsolver` Makefile keys `rungams`
#     off the per-instance trace CSVs rather than off the pounce build, so
#     a rebuilt solver leaves them satisfied -- `Nothing to be done for
#     'rungams'` -- and the report is regenerated from months-old traces
#     under the current commit's name;
#
#   * without the check, a link that runs but reports nothing usable still
#     looks green. `nlpbench_report.py` drops an instance from the
#     head-to-head when either side has no SolverTime, so a POUNCE link
#     that never set `resUsed` produced a report claiming `10/10 solved`
#     for both solvers and `both solved: 0` in the same breath.
#
# `nlpbench_report.py` and the `runsolver` Makefile are in `gams/nlpbench/`,
# a private GAMS-licensed clone this repository does not track, so the
# guarantee has to live on this side of the boundary.
#
# Only the pounce trace is required to be fresh: `clean-bench-smoke`
# defaults to `CLEAN_SOLVERS=pounce`, deliberately leaving the ipopt
# reference cached (its binary does not change between pounce iterations).
gams-bench:
	@if [ ! -d $(REPO_ROOT)/gams/nlpbench ]; then \
	  echo "gams-bench: skipped, gams/nlpbench is not present (a private GAMS-licensed clone; see .gitignore)"; \
	  exit 0; \
	fi; \
	stamp=$$(mktemp); \
	$(MAKE) -C $(REPO_ROOT)/gams/nlpbench clean-bench-smoke bench-smoke; \
	rc=$$?; \
	if [ $$rc -ne 0 ]; then rm -f $$stamp; exit $$rc; fi; \
	python3 $(REPO_ROOT)/benchmarks/scripts/check_gams_smoke.py \
	    --trace pounce=$(REPO_ROOT)/gams/nlpbench/runsolver/smoke.gms_pounce.csv \
	    --trace ipopt=$(REPO_ROOT)/gams/nlpbench/runsolver/smoke.gms_ipopt.csv \
	    --fresh pounce --newer-than $$stamp --min-common 10; \
	rc=$$?; rm -f $$stamp; exit $$rc

# Retained as the historical spelling of "force a re-run"; `gams-bench`
# now always does that, so the two are the same thing.
gams-rerun: gams-bench

# ---- Aggregation ----
# Incremental sweep — each *-run target is now a file-target wrapper
# that no-ops when its results.json is up-to-date. Use `benchmark-rerun`
# to force a full rebuild. Each sub-target runs via `-$(MAKE)` so a
# single failure (missing toolchain, broken instance) doesn't abort the
# rest — the report at the end always lands.
# Rebuild the sweep binary, letting cargo decide whether anything moved.
#
# The `$(POUNCE_BIN)` rule above has no prerequisites, so before gh#840 a
# sweep reused whatever target/release/pounce happened to be lying around:
# 0.11.0's first release run put all 14 suites through a binary stamped
# `main @ 5492ab46-dirty` -- 27 commits behind the tag, built from an unclean
# tree -- and recorded that in exactly one line of the report's provenance
# table. Three and a half hours of measurement described a binary nobody
# could rebuild.
#
# Phony, so cargo is always consulted. cargo leaves the binary's mtime alone
# when it has nothing to do, so the suite file-targets that depend on
# $(POUNCE_BIN) stay incremental: only a genuinely rebuilt binary invalidates
# them. A prerequisite rather than a `-$(MAKE)` line in the recipe, so a
# broken build stops the sweep instead of silently measuring the old binary.
pounce-bin:
	$(CARGO) build --release --manifest-path $(CARGO_MANIFEST) --bin pounce

benchmark: pounce-bin
	-$(MAKE) vanderbei-run
	-$(MAKE) electrolyte-run
	-$(MAKE) grid-run
	-$(MAKE) cho-run
	-$(MAKE) large-scale
	-$(MAKE) mittelmann-run
	-$(MAKE) gas-run
	-$(MAKE) water-run
	-$(MAKE) qp-run
	-$(MAKE) lp-run
	-$(MAKE) lp-convex-run
	-$(MAKE) qp-convex-run
	-$(MAKE) lbfgs-run
	-$(MAKE) lpopt-run
	-$(MAKE) gams-bench
	-$(MAKE) benchmark-report
	@echo "Benchmark sweep complete. Report: benchmarks/BENCHMARK_REPORT.md"

# Force a full rebuild — wipes every suite's outputs first, then runs
# the incremental sweep (which now finds nothing cached).
benchmark-rerun: clean-bench benchmark

# Generate composite report from existing results JSONs.
benchmark-report:
	python $(REPO_ROOT)/benchmarks/benchmark_report.py
	@echo "Report: benchmarks/BENCHMARK_REPORT.md"

# ---- Cleanup ----
# `clean-bench` wipes the per-release pounce outputs and the composite
# report so the next `make benchmark` starts from a clean slate. It does
# NOT touch the committed ipopt-ma57 reference (benchmarks/*/ipopt_ma57.json
# + provenance) — that is regenerated only by `make ipopt-reference`.
# Generated large_scale .nl are also preserved (use `make
# large-scale-generate` to refresh). Sub-makefiles for mittelmann and
# gams own their own clean targets and are delegated to with `-$(MAKE)`
# so a missing toolchain there does not abort the rest of the cleanup.
clean-bench: clean-bench-large-scale clean-bench-nl clean-bench-report
	-$(MAKE) -C $(REPO_ROOT)/benchmarks/mittelmann clean
	-$(MAKE) -C $(REPO_ROOT)/gams/nlpbench clean-bench
	@echo "clean-bench complete (committed ipopt-ma57 reference preserved)."

# .sol scratch is written by the solver beside each .nl, so it now lives
# under $(POUNCE_BENCH_DATA) (via nldir_*); pounce.json stays in the repo.
clean-bench-large-scale:
	rm -f $(nldir_large_scale)/*.sol \
	      $(REPO_ROOT)/benchmarks/large_scale/pounce.json

clean-bench-nl:
	rm -f $(nldir_vanderbei)/*.sol \
	      $(REPO_ROOT)/benchmarks/vanderbei/pounce.json
	rm -f $(nldir_electrolyte)/*.sol \
	      $(REPO_ROOT)/benchmarks/electrolyte/pounce.json
	rm -f $(nldir_grid)/*.sol \
	      $(REPO_ROOT)/benchmarks/grid/pounce.json
	rm -f $(nldir_grid_ecf)/*.sol \
	      $(REPO_ROOT)/benchmarks/grid_ecf/pounce.json
	rm -f $(nldir_grid_polar)/*.sol \
	      $(REPO_ROOT)/benchmarks/grid_polar/pounce.json
	rm -f $(nldir_gas)/*.sol \
	      $(REPO_ROOT)/benchmarks/gas/pounce.json
	rm -f $(nldir_water)/*.sol \
	      $(REPO_ROOT)/benchmarks/water/pounce.json
	rm -f $(nldir_cho)/*.sol \
	      $(REPO_ROOT)/benchmarks/cho/pounce.json
	rm -f $(REPO_ROOT)/benchmarks/cho/cho_stderr.txt
	rm -f $(nldir_mittelmann)/*.sol \
	      $(REPO_ROOT)/benchmarks/mittelmann/pounce.json
	rm -f $(nldir_qp)/*.sol \
	      $(REPO_ROOT)/benchmarks/qp/pounce.json
	rm -f $(nldir_lp)/*.sol \
	      $(REPO_ROOT)/benchmarks/lp/pounce.json
	rm -f $(nldir_lpopt)/*.sol \
	      $(REPO_ROOT)/benchmarks/lpopt/pounce.json
	rm -f $(LP_CONVEX_DIR)/convex.json $(LP_CONVEX_DIR)/nlp.json
	rm -f $(QP_CONVEX_DIR)/convex.json $(QP_CONVEX_DIR)/nlp.json
	rm -f $(LBFGS_DIR)/exact.json $(LBFGS_DIR)/limited.json
	rm -rf $(REPO_ROOT)/benchmarks/logs

clean-bench-report:
	rm -f $(REPO_ROOT)/benchmarks/BENCHMARK_REPORT.md
