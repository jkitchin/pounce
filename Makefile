# POUNCE — Makefile wrapper around cargo for build, test, and install.
#
# Usage:
#   make                  # release build of the workspace
#   make build            # release build (alias)
#   make debug            # debug build
#   make test             # run all tests
#   make check            # cargo check
#   make clippy           # lint with clippy (treats warnings as errors)
#   make fmt              # rustfmt the workspace
#   make doc              # build rustdoc
#   make install          # install pounce CLI + cinterface cdylib under $(PREFIX)
#   make uninstall        # remove installed artifacts
#   make clean            # cargo clean
#
# Benchmark targets (drive `pounce` on .nl files under benchmarks/):
#   make bench            # run cho + gas + water + mittelmann (whatever is on disk)
#   make bench-cho        # CHO parameter-estimation .nl (large, ~480 iters)
#   make bench-gas        # GasLib pipelines (4 .nl files)
#   make bench-water      # Water network (~7 .nl files)
#   make bench-mittelmann # Mittelmann ampl-nlp suite (whatever's been translated)
#   make bench-cutest     # CUTEst comparison harness (POUNCE vs native Ipopt)
#   make bench-cutest-prepare  # compile SIF problem dylibs (one-time, ~7GB)
#
# Tunables — pass on the command line:
#   make bench-cho LINEAR_SOLVER=ma57 MAX_ITER=2000 PRINT_LEVEL=5
#   make bench-gas BENCH_OPTIONS="tol=1e-10 mu_strategy=adaptive"
# Defaults: LINEAR_SOLVER=ma57, MAX_ITER=3000, PRINT_LEVEL=5.
#
# Default install prefix is $(HOME)/.local — a user-owned directory
# that needs no sudo. Make sure $(HOME)/.local/bin is on your PATH
# (and $(HOME)/.local/lib on DYLD_LIBRARY_PATH / LD_LIBRARY_PATH if
# you intend to link against libpounce_cinterface from outside cargo).
#
# Override for a system-wide install (requires sudo):
#   sudo make install PREFIX=/usr/local
#
# Or pick any other user-owned directory:
#   make install PREFIX=$$HOME/opt/pounce
#
# Pass extra flags through to cargo:
#   make build CARGO_FLAGS="--features feral"

CARGO       ?= cargo
PREFIX      ?= $(HOME)/.local
BINDIR      ?= $(PREFIX)/bin
LIBDIR      ?= $(PREFIX)/lib
INCLUDEDIR  ?= $(PREFIX)/include
PROFILE     ?= release
CARGO_FLAGS ?=

TARGET_DIR    := target/$(PROFILE)
CLI_BIN       := $(TARGET_DIR)/pounce
CDYLIB_NAME   := libpounce_cinterface
UNAME_S       := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
  CDYLIB_EXT := dylib
else ifeq ($(UNAME_S),Linux)
  CDYLIB_EXT := so
else
  CDYLIB_EXT := dll
endif
CDYLIB        := $(TARGET_DIR)/$(CDYLIB_NAME).$(CDYLIB_EXT)

ifeq ($(PROFILE),release)
  CARGO_PROFILE_FLAG := --release
else
  CARGO_PROFILE_FLAG :=
endif

BENCH_DIR      := benchmarks
BENCH_LOG_DIR  := $(BENCH_DIR)/logs
LINEAR_SOLVER  ?= ma57
MAX_ITER       ?= 3000
PRINT_LEVEL    ?= 5
BENCH_OPTIONS  ?=
BENCH_ARGS     := linear_solver=$(LINEAR_SOLVER) max_iter=$(MAX_ITER) print_level=$(PRINT_LEVEL) $(BENCH_OPTIONS)

.PHONY: all build debug test check clippy fmt fmt-check doc install uninstall clean help \
        bench bench-cho bench-gas bench-water bench-mittelmann bench-cutest \
        bench-cutest-prepare bench-cutest-smoke bench-cutest-report bench-clean

all: build

build:
	$(CARGO) build --workspace $(CARGO_PROFILE_FLAG) $(CARGO_FLAGS)

debug:
	$(MAKE) build PROFILE=debug

test:
	$(CARGO) test --workspace $(CARGO_PROFILE_FLAG) $(CARGO_FLAGS)

check:
	$(CARGO) check --workspace $(CARGO_FLAGS)

clippy:
	$(CARGO) clippy --workspace --all-targets $(CARGO_FLAGS) -- -D warnings

fmt:
	$(CARGO) fmt --all

fmt-check:
	$(CARGO) fmt --all -- --check

doc:
	$(CARGO) doc --workspace --no-deps $(CARGO_PROFILE_FLAG)

install: build
	@echo "Installing pounce into $(PREFIX)"
	install -d "$(DESTDIR)$(BINDIR)" "$(DESTDIR)$(LIBDIR)"
	install -m 0755 "$(CLI_BIN)" "$(DESTDIR)$(BINDIR)/pounce"
	install -m 0644 "$(CDYLIB)" "$(DESTDIR)$(LIBDIR)/$(CDYLIB_NAME).$(CDYLIB_EXT)"

uninstall:
	@echo "Removing pounce from $(PREFIX)"
	rm -f "$(DESTDIR)$(BINDIR)/pounce"
	rm -f "$(DESTDIR)$(LIBDIR)/$(CDYLIB_NAME).$(CDYLIB_EXT)"

clean:
	$(CARGO) clean

help:
	@sed -n 's/^# \{0,1\}//p' Makefile | sed -n '1,45p'

# ---- Benchmarks ----------------------------------------------------------
# Each suite runs `pounce <nl-file> $(BENCH_ARGS)` for every .nl on disk and
# tees solver output to $(BENCH_LOG_DIR)/<suite>/<problem>.log. The final
# status line of each run (e.g. "EXIT: Optimal Solution Found.") is printed
# as a summary so the operator can scan results without trawling logs.

# Path to the cho_parmest .nl (override if you've moved the export).
CHO_NL ?= $(BENCH_DIR)/cho/nl_export_results/cho_parmest.nl

# Glob patterns for the per-suite .nl files. Suites silently no-op if their
# directory isn't populated (benchmarks/ is gitignored — these are local).
GAS_NL_FILES        := $(wildcard $(BENCH_DIR)/gas/*.nl)
WATER_NL_FILES      := $(wildcard $(BENCH_DIR)/water/*.nl)
MITTELMANN_NL_FILES := $(wildcard $(BENCH_DIR)/mittelmann/nl/*.nl)

# $(call run-bench,suite,nl-file) — solve one .nl and tee to a log.
# The trailing `|| true` keeps make going across failing problems so the
# whole suite reports together.
define run-bench
	@mkdir -p "$(BENCH_LOG_DIR)/$(1)"
	@nl="$(2)"; name="$$(basename $$nl .nl)"; \
	  log="$(BENCH_LOG_DIR)/$(1)/$$name.log"; \
	  echo "[$(1)] $$name"; \
	  ./$(CLI_BIN) "$$nl" $(BENCH_ARGS) > "$$log" 2>&1 || true; \
	  status="$$(grep -E '^EXIT:' "$$log" | tail -1)"; \
	  iters="$$(awk '/^Number of Iterations/ {print $$NF}' "$$log" | tail -1)"; \
	  printf "  iters=%-6s %s\n" "$${iters:-?}" "$${status:-no EXIT line — check $$log}"

endef

bench-cho: $(CLI_BIN)
	@if [ ! -f "$(CHO_NL)" ]; then \
	  echo "bench-cho: $(CHO_NL) not found (set CHO_NL=<path> or regenerate)"; exit 0; \
	fi
	$(call run-bench,cho,$(CHO_NL))
	@echo "Logs in $(BENCH_LOG_DIR)/cho/"

bench-gas: $(CLI_BIN)
	@if [ -z "$(GAS_NL_FILES)" ]; then \
	  echo "bench-gas: no .nl files under $(BENCH_DIR)/gas/"; exit 0; \
	fi
	$(foreach nl,$(GAS_NL_FILES),$(call run-bench,gas,$(nl)))
	@echo "Logs in $(BENCH_LOG_DIR)/gas/"

bench-water: $(CLI_BIN)
	@if [ -z "$(WATER_NL_FILES)" ]; then \
	  echo "bench-water: no .nl files under $(BENCH_DIR)/water/"; exit 0; \
	fi
	$(foreach nl,$(WATER_NL_FILES),$(call run-bench,water,$(nl)))
	@echo "Logs in $(BENCH_LOG_DIR)/water/"

bench-mittelmann: $(CLI_BIN)
	@if [ -z "$(MITTELMANN_NL_FILES)" ]; then \
	  echo "bench-mittelmann: no .nl files under $(BENCH_DIR)/mittelmann/nl/"; \
	  echo "  (run \`make -C $(BENCH_DIR)/mittelmann translate\` to populate)"; exit 0; \
	fi
	$(foreach nl,$(MITTELMANN_NL_FILES),$(call run-bench,mittelmann,$(nl)))
	@echo "Logs in $(BENCH_LOG_DIR)/mittelmann/"

bench: bench-cho bench-gas bench-water bench-mittelmann
	@echo "All benchmarks complete. Logs under $(BENCH_LOG_DIR)/."

bench-clean:
	rm -rf "$(BENCH_LOG_DIR)"

# ---- CUTEst comparison harness ------------------------------------------
# These recipes use bash process-substitution (`2> >(tee …)`) to copy stderr
# to a log without buffering, so they need bash explicitly. (Other recipes
# in this Makefile are POSIX-sh-compatible; SHELL is overridden only for
# the cutest recipes via per-target assignment below.)
bench-cutest bench-cutest-smoke bench-cutest-prepare: SHELL := /bin/bash

# Drives the pounce-cutest workspace member, which runs each SIF problem
# in a subprocess against POUNCE and native Ipopt and emits JSON.
# Requirements: CUTEst installed under ~/.local/cutest/, libipopt
# discoverable via pkg-config, gfortran on PATH.

CUTEST_RESULTS ?= $(BENCH_DIR)/cutest/results.json

bench-cutest-prepare:
	@if ! command -v cutest2sif >/dev/null 2>&1 && [ ! -f "$$HOME/.local/cutest/env.sh" ]; then \
	  echo "CUTEst not found at ~/.local/cutest/. See benchmarks/cutest/README.md."; exit 1; \
	fi
	source $$HOME/.local/cutest/env.sh && bash $(BENCH_DIR)/cutest/prepare.sh

bench-cutest:
	RESULTS_FILE=$(CUTEST_RESULTS) \
	  $(CARGO) run --bin cutest_suite $(CARGO_PROFILE_FLAG) $(CARGO_FLAGS) \
	  2> >(tee $(BENCH_DIR)/cutest/benchmark_stderr.txt >&2)
	@echo "CUTEst run complete. Results: $(CUTEST_RESULTS)"

bench-cutest-smoke:
	RESULTS_FILE=$(BENCH_DIR)/cutest/smoke_results.json \
	  $(CARGO) run --bin cutest_suite $(CARGO_PROFILE_FLAG) $(CARGO_FLAGS) -- \
	    ROSENBR BEALE CUBE DENSCHNB BROWNBS \
	    HS6 HS10 HS35 HS71 HS106 \
	    MARATOS BT1 HS13 HS57 \
	  2> >(tee $(BENCH_DIR)/cutest/smoke_stderr.txt >&2)

bench-cutest-report:
	@python $(BENCH_DIR)/cutest/compare.py $(CUTEST_RESULTS)

# Build CLI before any bench target if it's missing.
$(CLI_BIN):
	$(CARGO) build --workspace $(CARGO_PROFILE_FLAG) $(CARGO_FLAGS)
