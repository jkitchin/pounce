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
#   make book             # build the mdbook documentation (docs/)
#   make install          # install pounce CLI + cinterface cdylib under $(PREFIX)
#   make uninstall        # remove installed artifacts
#   make install-mcp      # build studio/mcp + register with Claude Code
#   make uninstall-mcp    # unregister + remove the studio/mcp venv
#   make install-skill    # build pounce + pounce-studio, drop SKILL.md into ~/.claude/skills/
#   make uninstall-skill  # remove the installed skill directory
#   make clean            # cargo clean
#
# Benchmark targets — single source of truth in benchmarks/Makefile.
# Top-level shims delegate so commands are runnable from the repo root:
#   make benchmark            # full sweep: cutest + all .nl suites + gams + report
#   make benchmark-report     # regenerate benchmarks/BENCHMARK_REPORT.md
#   make benchmark-<suite>    # one suite (cutest, water, gas, electrolyte,
#                             #   grid, cho, large-scale, mittelmann, gams)
#
# See `make -C benchmarks help` for the full target list.
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

.PHONY: all build debug test check clippy fmt fmt-check doc book install uninstall clean help \
        install-mcp uninstall-mcp install-skill uninstall-skill \
        python-ext python-test \
        benchmark benchmark-rerun benchmark-report benchmark-gams

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

book:
	mdbook build docs

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

# ---- Python extension + tests -------------------------------------------
# Rebuild the native extension in place, then run the Python test suite.
# This is the safe way to run pytest: a stale in-place `_pounce*.so` (left
# by an earlier `maturin develop`) silently shadows the current binding and
# makes the suite fail with confusing errors. `python-ext` rebuilds it, and
# `python/tests/conftest.py` additionally guards against running pytest
# against a stale artifact. Requires `maturin` and the test extras in the
# active environment (`pip install -e 'python[dev]'`).
python-ext:
	cd python && maturin develop

python-test: python-ext
	cd python && python -m pytest tests -q

# ---- Benchmarks ----------------------------------------------------------
# Single source of truth: benchmarks/Makefile. These shims forward
# everything so users can drive runs from the repo root.
#
# All `*-run` targets are incremental (skip when results.json is fresh).
# `*-rerun` variants wipe the results.json then run, forcing a rebuild.
benchmark:
	$(MAKE) -C benchmarks benchmark

benchmark-rerun:
	$(MAKE) -C benchmarks benchmark-rerun

benchmark-report:
	$(MAKE) -C benchmarks benchmark-report

# Pattern-rule shims for per-suite targets. Examples:
#   make benchmark-water         -> make -C benchmarks water-run
#   make benchmark-water-rerun   -> make -C benchmarks water-rerun
#   make benchmark-cutest        -> make -C benchmarks cutest-run
#   make benchmark-gams          -> make -C benchmarks gams-bench
benchmark-gams:
	$(MAKE) -C benchmarks gams-bench

benchmark-%-rerun:
	$(MAKE) -C benchmarks $*-rerun

benchmark-%:
	$(MAKE) -C benchmarks $*-run

# ---- MCP server (studio/mcp) --------------------------------------------
# Builds the pounce-studio-mcp server into a private venv under
# studio/mcp/.venv (PyO3 extension compiled in release mode) and
# registers it with Claude Code via `claude mcp add`. Idempotent — rerun
# after pulling new studio changes to rebuild the extension.
#
#   make install-mcp                   # user scope (visible to all sessions)
#   make install-mcp MCP_SCOPE=local   # this project only
#   make uninstall-mcp                 # unregister + delete the venv

MCP_DIR   := studio/mcp
MCP_VENV  := $(MCP_DIR)/.venv
MCP_PY    := $(MCP_VENV)/bin/python
MCP_BIN   := $(MCP_VENV)/bin/pounce-studio-mcp
MCP_SCOPE ?= user

install-mcp:
	@command -v claude >/dev/null 2>&1 || { \
	  echo "install-mcp: 'claude' CLI not on PATH (install Claude Code first)"; exit 1; }
	@if [ ! -d "$(MCP_VENV)" ]; then \
	  echo "Creating venv at $(MCP_VENV)"; \
	  python3 -m venv "$(MCP_VENV)"; \
	fi
	@$(MCP_PY) -m pip install --quiet --upgrade pip maturin
	@echo "Building native extension (maturin develop --release)"
	@cd $(MCP_DIR) && . .venv/bin/activate && maturin develop --release
	@echo "Registering with Claude Code (scope=$(MCP_SCOPE))"
	@claude mcp remove pounce-studio --scope $(MCP_SCOPE) >/dev/null 2>&1 || true
	@claude mcp add pounce-studio --scope $(MCP_SCOPE) -- "$(abspath $(MCP_BIN))"
	@echo
	@echo "Done. Restart Claude Code to pick up the new server."
	@echo "Verify with: claude mcp list"

uninstall-mcp:
	-@command -v claude >/dev/null 2>&1 && \
	  claude mcp remove pounce-studio --scope $(MCP_SCOPE) >/dev/null 2>&1 || true
	rm -rf "$(MCP_VENV)"
	@echo "Removed $(MCP_VENV) and unregistered pounce-studio (scope=$(MCP_SCOPE))"

# ---- Claude skill (studio/skill) ---------------------------------------
# Build the pounce + pounce-studio binaries, install them under $(PREFIX),
# and drop the skill directory at ~/.claude/skills/pounce/ so any Claude
# Code session picks it up. Override SKILL_DIR for a non-default location.
#
#   make install-skill                          # ~/.claude/skills/pounce
#   make install-skill SKILL_DIR=$$HOME/elsewhere/pounce
#   make uninstall-skill

SKILL_DIR ?= $(HOME)/.claude/skills/pounce
STUDIO_BIN := $(TARGET_DIR)/pounce-studio

install-skill: build
	@echo "Installing pounce + pounce-studio into $(PREFIX) and skill into $(SKILL_DIR)"
	install -d "$(DESTDIR)$(BINDIR)"
	install -m 0755 "$(CLI_BIN)" "$(DESTDIR)$(BINDIR)/pounce"
	install -m 0755 "$(STUDIO_BIN)" "$(DESTDIR)$(BINDIR)/pounce-studio"
	install -d "$(SKILL_DIR)"
	install -m 0644 studio/skill/SKILL.md "$(SKILL_DIR)/SKILL.md"
	install -m 0644 studio/skill/README.md "$(SKILL_DIR)/README.md"
	@echo
	@echo "Done. Verify with:"
	@echo "  $(BINDIR)/pounce-studio --version"
	@echo "  ls $(SKILL_DIR)"
	@echo
	@echo "In a fresh Claude Code session, ask:"
	@echo '  "diagnose studio/mcp/fixtures/rosenbrock-stalled.json"'

uninstall-skill:
	rm -rf "$(SKILL_DIR)"
	rm -f "$(DESTDIR)$(BINDIR)/pounce-studio"
	@echo "Removed $(SKILL_DIR) and $(BINDIR)/pounce-studio"
	@echo "Note: $(BINDIR)/pounce was not removed (shared with \`make install\`)."
