# Mittelmann ampl-nlp suite: fetch the .mod sources and translate them to .nl.
#
# Called by benchmarks/Makefile as
#
#     $(MAKE) -C benchmarks/mittelmann fetch translate NLDIR=<dir>
#
# before every Mittelmann run (`mittelmann-run`, `ipopt-ref-mittelmann`).
# Running the solvers and building the report are NOT done here: every suite
# goes through benchmarks/scripts/run_nl_bench.sh and benchmarks/benchmark_report.py.
#
# Both steps are per problem and skip any problem whose .nl is already in
# NLDIR, so a machine that already has the translated set (e.g. through
# POUNCE_BENCH_DATA) never needs AMPL or the network. Only a missing .nl
# triggers a download from plato.asu.edu and a translation with AMPL.
#
# The .mod sources and the .nl translations are not redistributed here;
# only this harness and the problem list are tracked.
#
# Before this file was tracked, `mittelmann-run` failed
# from any fresh checkout with "No rule to make target `fetch'": the
# `/benchmarks/*/*` ignore rule had kept the file out of git.

REPO_ROOT  := $(abspath ../..)
NLDIR      ?= $(REPO_ROOT)/benchmarks/mittelmann/nl
SOURCEDIR  ?= $(REPO_ROOT)/benchmarks/mittelmann/source
LOGDIR     ?= $(REPO_ROOT)/benchmarks/mittelmann/logs/translate
SOURCE_URL ?= https://plato.asu.edu/ftp/ampl-nlp-source
# AMPL Community Edition, installed with amplpy (see README.md).
AMPL       ?= $(REPO_ROOT)/.venv-ampl/lib/python3.12/site-packages/ampl_module_base/bin/ampl

PROBLEMS := $(shell cat $(REPO_ROOT)/benchmarks/mittelmann/problems.txt)

.PHONY: help fetch translate build-ipopt-ma57

help:
	@echo "Mittelmann ampl-nlp suite ($(words $(PROBLEMS)) problems):"
	@echo "  fetch      download the .mod of every problem whose .nl is missing"
	@echo "  translate  .mod -> .nl with AMPL, for every problem whose .nl is missing"
	@echo "Variables: NLDIR (default nl/), AMPL, SOURCE_URL"

fetch:
	@mkdir -p $(SOURCEDIR); have=0; got=0; \
	for p in $(PROBLEMS); do \
	  if [ -f $(NLDIR)/$$p.nl ]; then have=$$((have+1)); continue; fi; \
	  if [ -f $(SOURCEDIR)/$$p.mod ]; then got=$$((got+1)); continue; fi; \
	  echo "  fetching $$p.mod"; \
	  curl -sf -o $(SOURCEDIR)/$$p.mod $(SOURCE_URL)/$$p.mod \
	    || { curl -sf -o $(SOURCEDIR)/$$p.mod.gz $(SOURCE_URL)/$$p.mod.gz \
	         && gunzip -f $(SOURCEDIR)/$$p.mod.gz; } \
	    || { echo "    FAILED: $$p"; rm -f $(SOURCEDIR)/$$p.mod $(SOURCEDIR)/$$p.mod.gz; exit 1; }; \
	  got=$$((got+1)); \
	done; \
	echo "mittelmann fetch: $$have already translated, $$got .mod present or fetched"

translate:
	@mkdir -p $(NLDIR) $(LOGDIR); have=0; made=0; \
	for p in $(PROBLEMS); do \
	  if [ -f $(NLDIR)/$$p.nl ]; then have=$$((have+1)); continue; fi; \
	  if [ ! -x "$(AMPL)" ]; then \
	    echo "mittelmann translate: $$p.nl is missing and no AMPL at $(AMPL)" >&2; \
	    echo "  install AMPL CE (README.md) or pass AMPL=<path>, or point NLDIR at a translated set" >&2; \
	    exit 1; \
	  fi; \
	  echo "  translating $$p"; \
	  (cd $(NLDIR) && echo "model $(SOURCEDIR)/$$p.mod; option auxfiles cr; write g$$p;" \
	    | "$(AMPL)" > $(LOGDIR)/$$p.log 2>&1) \
	    || { echo "    FAILED: see $(LOGDIR)/$$p.log"; rm -f $(NLDIR)/$$p.nl; exit 1; }; \
	  made=$$((made+1)); \
	done; \
	echo "mittelmann translate: $$have already present, $$made translated"

# benchmarks/Makefile's `build-ipopt-ma57` shim lands here. The recipe that
# produced ref/Ipopt/install-ma57 was never committed, so there is nothing to
# run; say so instead of failing with "No rule to make target".
build-ipopt-ma57:
	@echo "No build recipe is tracked for the Ipopt-MA57 reference binary." >&2
	@echo "Build Ipopt with CoinHSL and the ASL (AMPL) interface, then point the" >&2
	@echo "harness at it: make -C benchmarks IPOPT_MA57_BIN=<path>/bin/ipopt ipopt-ref-<suite>" >&2
	@exit 1
