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
# by `make -C mittelmann build-ipopt-ma57`). This matches POUNCE-MA57
# against the same linear-solver family. No pkg-config / libipopt FFI is
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
	large-scale large-scale-generate large-scale-rerun \
	mittelmann-run mittelmann-rerun mittelmann-translate \
	gas-run gas-rerun water-run water-rerun \
	qp-run qp-rerun qp-generate \
	lp-run lp-rerun lp-generate \
	lpopt-run lpopt-rerun lpopt-generate \
	globallib-run globallib-rerun globallib-translate \
	globallib-micro globallib-fast .PHONY-build-simplex-obbt \
	ipopt-reference ipopt-reference-provenance \
	$(addprefix ipopt-ref-,$(REF_SUITES)) \
	gams-bench gams-rerun \
	clean-bench clean-bench-large-scale clean-bench-nl clean-bench-report

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
	@echo "  cho-run         / -rerun   CHO parameter estimation benchmark"
	@echo "  large-scale     / large-scale-rerun  Synthetic large-scale NLPs (.nl from generate_nl.py)"
	@echo "  large-scale-generate       (Re)generate the large_scale .nl files only (needs pyomo)"
	@echo "  mittelmann-run  / -rerun   Mittelmann ampl-nlp .nl suite (shared driver; .nl via mittelmann-translate)"
	@echo "  gas-run         / -rerun   Gas pipeline NLPs (AMPL .nl files)"
	@echo "  water-run       / -rerun   Water distribution network NLPs (AMPL .nl files)"
	@echo "  qp-run          / -rerun   Maros-Meszaros convex QP suite (.nl from generate_nl.py)"
	@echo "  qp-generate                (Re)generate the qp .nl files only (needs pyomo/scipy; downloads .mat)"
	@echo "  lp-run          / -rerun   Netlib + small Meszaros LP validation set (.nl from generate_nl.py)"
	@echo "  lp-generate                (Re)generate the lp .nl files only (needs highspy/pyomo; downloads MPS)"
	@echo "  lpopt-run       / -rerun   HARD Mittelmann lpopt LP subset (stress tier; use BENCH_TIMELIMIT=1800)"
	@echo "  lpopt-generate             (Re)generate the lpopt .nl files only (downloads MPS from plato lptestset)"
	@echo "  gams-bench      / gams-rerun  GAMS solver-link smoke check (10 problems; not in the report)"
	@echo "  globallib-run   / -rerun   GLOBALLib proven-optimum global suite (solver_selection=global; vs known optima)"
	@echo "  globallib-micro            fast dev loop (~3s): IPM-vs-simplex OBBT cross-check on tiers/micro.txt"
	@echo "  globallib-fast             broader fast cross-check (~30s) on tiers/fast.txt"
	@echo "  globallib-translate        (Re)generate the globallib .nl from AMPL .mod (needs ampl; see benchmarks/globallib)"
	@echo ""
	@echo "  clean-bench       Wipe pounce results/logs across every suite (keeps the committed ipopt reference)"
	@echo "  clean-bench-large-scale Wipe large_scale pounce.json"
	@echo "  clean-bench-nl         Wipe per-problem .sol outputs + pounce.json (.nl suites)"
	@echo "  clean-bench-report     Remove benchmarks/BENCHMARK_REPORT.md"
	@echo ""
	@echo "All suites feed into benchmarks/BENCHMARK_REPORT.md via benchmark_report.py."

# Convenience shim — the actual recipe lives in benchmarks/mittelmann/Makefile.
build-ipopt-ma57:
	$(MAKE) -C $(REPO_ROOT)/benchmarks/mittelmann build-ipopt-ma57

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

# Ensure POUNCE_BIN exists; helps when a target is run before any cargo
# build. Cheap when already built.
$(POUNCE_BIN):
	$(CARGO) build --release --manifest-path $(CARGO_MANIFEST) --bin pounce

# Suite -> .nl directory map, shared by the pounce file-targets and the
# ipopt-ref-% reference targets.
nldir_vanderbei   := $(POUNCE_BENCH_DATA)/vanderbei/nl
nldir_electrolyte := $(POUNCE_BENCH_DATA)/electrolyte/nl
nldir_grid        := $(POUNCE_BENCH_DATA)/grid/nl
nldir_cho         := $(POUNCE_BENCH_DATA)/cho/nl_export_results
nldir_large_scale := $(POUNCE_BENCH_DATA)/large_scale/nl
nldir_gas         := $(POUNCE_BENCH_DATA)/gas
nldir_water       := $(POUNCE_BENCH_DATA)/water
nldir_mittelmann  := $(POUNCE_BENCH_DATA)/mittelmann/nl
nldir_qp          := $(POUNCE_BENCH_DATA)/qp/nl
nldir_lp          := $(POUNCE_BENCH_DATA)/lp/nl
nldir_lpopt       := $(POUNCE_BENCH_DATA)/lpopt/nl
nldir_globallib   := $(POUNCE_BENCH_DATA)/globallib/nl

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

# Mittelmann ampl-nlp suite. The .nl are translated from .mod by the
# mittelmann sub-Makefile (`fetch translate`, needs AMPL; incremental).
# The sub-Makefile's broader multi-solver comparison harness still lives
# in benchmarks/mittelmann/Makefile for on-demand use.
mittelmann-translate:
	$(MAKE) -C $(MITT_DIR) fetch translate NLDIR=$(nldir_mittelmann)

$(MITT_DIR)/pounce.json: $(POUNCE_BIN) $(wildcard $(nldir_mittelmann)/*.nl)
	$(NL_DRIVER) mittelmann $(nldir_mittelmann) $@ $(POUNCE_BIN) - $(BENCH_TIMELIMIT) pounce

mittelmann-run: mittelmann-translate $(MITT_DIR)/pounce.json

mittelmann-rerun:
	rm -f $(MITT_DIR)/pounce.json
	$(MAKE) mittelmann-run

# --- GLOBALLib global-optimization benchmark (pounce-global) ---
# The only tier that drives `solver_selection=global`: the GLOBALLib subset
# with a *proven* global optimum (MINLPLib `=opt=`). Translation clones the
# AMPL .mod sources and runs AMPL `write` into the bench-data tree; the run
# checks each certified objective against the known optimum in optima.txt.
# Needs `ampl` on PATH (set $AMPL to override). GLOBALLIB_TIMEOUT caps each
# problem (default 30s). This is a hard external (subprocess) kill; the global
# solver also accepts `global_max_cpu_time` (and `global_max_nodes` / gap / cut
# tunables — see `register_global_options` in pounce-cli) for a graceful
# node-boundary stop that still reports the best incumbent, but the external
# timeout stays as the hard backstop (it catches a single pathological slow node).
GLOBALLIB_DIR     := $(REPO_ROOT)/benchmarks/globallib
GLOBALLIB_TIMEOUT ?= 30

globallib-translate:
	$(GLOBALLIB_DIR)/translate.sh $(nldir_globallib)

$(GLOBALLIB_DIR)/pounce.json: $(POUNCE_BIN) $(wildcard $(nldir_globallib)/*.nl)
	python3 $(GLOBALLIB_DIR)/run_globallib.py --bin $(POUNCE_BIN) \
		--nl-dir $(nldir_globallib) --timeout $(GLOBALLIB_TIMEOUT) --out $@

globallib-run: $(GLOBALLIB_DIR)/pounce.json

globallib-rerun:
	rm -f $(GLOBALLIB_DIR)/pounce.json
	$(MAKE) globallib-run

# --- fast dev tiers: small/fast subsets for wiring the per-node pieces (OBBT,
# simplex/IPM warm-starts, branching) with a seconds-long edit->run loop.
# Both run the IPM and simplex OBBT engines and assert they certify identical
# optima (the soundness gate), so the simplex binary needs the feature on.
GLOBALLIB_DEV_BIN := $(REPO_ROOT)/target/release/pounce
.PHONY-build-simplex-obbt:
	$(CARGO) build --release --manifest-path $(CARGO_MANIFEST) \
		-p pounce-cli --features simplex-obbt

# Inner loop (~2-3s): `make globallib-micro`
globallib-micro: .PHONY-build-simplex-obbt
	python3 $(GLOBALLIB_DIR)/compare_obbt_engines.py \
		--bin $(GLOBALLIB_DEV_BIN) --nl-dir $(nldir_globallib) \
		--stems-file $(GLOBALLIB_DIR)/tiers/micro.txt --timeout 10

# Broader fast regression (~25-30s both engines): `make globallib-fast`
globallib-fast: .PHONY-build-simplex-obbt
	python3 $(GLOBALLIB_DIR)/compare_obbt_engines.py \
		--bin $(GLOBALLIB_DEV_BIN) --nl-dir $(nldir_globallib) \
		--stems-file $(GLOBALLIB_DIR)/tiers/fast.txt --timeout 10

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
ipopt-ref-mittelmann:  mittelmann-translate
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
gams-bench:
	$(MAKE) -C $(REPO_ROOT)/gams/nlpbench bench-smoke

gams-rerun:
	$(MAKE) -C $(REPO_ROOT)/gams/nlpbench clean-bench-smoke bench-smoke

# ---- Aggregation ----
# Incremental sweep — each *-run target is now a file-target wrapper
# that no-ops when its results.json is up-to-date. Use `benchmark-rerun`
# to force a full rebuild. Each sub-target runs via `-$(MAKE)` so a
# single failure (missing toolchain, broken instance) doesn't abort the
# rest — the report at the end always lands.
benchmark:
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
	rm -rf $(REPO_ROOT)/benchmarks/logs

clean-bench-report:
	rm -f $(REPO_ROOT)/benchmarks/BENCHMARK_REPORT.md
