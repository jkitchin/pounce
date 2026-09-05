# POUNCE — Makefile wrapper around cargo for build, test, and install.
#
# Usage:
#   make                  # release build of the workspace
#   make build            # release build (alias)
#   make debug            # debug build
#   make test             # run all tests
#   make coverage         # combined Rust + Python coverage report
#   make coverage-quick   # same, skipping the slow pytest suite
#   make check            # cargo check
#   make clippy           # lint with clippy (treats warnings as errors)
#   make fmt              # rustfmt the workspace
#   make doc              # build rustdoc
#   make book             # build the mdbook documentation (docs/)
#   make book-site        # build the full multi-version site + docs assistant
#   make ask-check        # test the docs assistant's retrieval (needs node)
#   make wasm             # build the browser (WebAssembly) solver + demo page
#   make wasm-serve       # ...and serve it on http://localhost:8000
#   make install          # install pounce CLI + cinterface cdylib under $(PREFIX)
#   make uninstall        # remove installed artifacts
#   make docker           # container image compiled from the current tree
#   make docker-release   # container image from the published PyPI wheels
#   make dev              # build the CLI + extension for a source checkout
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

.PHONY: all build debug test check clippy fmt fmt-check doc book screencast install uninstall clean help \
        install-mcp uninstall-mcp install-skill uninstall-skill pounce-ma57 \
        test-ma57 \
        dev python-ext python-cli-bin python-test coverage coverage-quick \
        benchmark benchmark-rerun benchmark-report benchmark-gams wasm wasm-serve \
        docker docker-release book-site ask-check

all: build

build:
	$(CARGO) build --workspace $(CARGO_PROFILE_FLAG) $(CARGO_FLAGS)

debug:
	$(MAKE) build PROFILE=debug

test:
	$(CARGO) test --workspace $(CARGO_PROFILE_FLAG) $(CARGO_FLAGS)

# ---- Coverage ------------------------------------------------------------
# `cargo llvm-cov --workspace` instruments only the Rust test suite, so every
# path reached solely through the Python extension or the pytest/pyomo-driven
# CLI reads as 0% — which turns the report into a source of invented gaps
# rather than a usable "what is under-tested?" signal. `coverage` drives
# llvm-profdata / llvm-cov directly and attributes every instrumented artifact
# (Rust test binaries, the CLI, and the installed extension module), so the
# number reflects what the whole project actually exercises.
#
# Note: the run leaves `python/pounce/_pounce*.so` built WITH instrumentation,
# which is slower and can upset timing-sensitive tests. Restore it with
# `make python-ext` (or `cd python && maturin develop --release`).
coverage:
	scripts/coverage-combined.sh

coverage-quick:
	scripts/coverage-combined.sh --quick

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

# ---- WebAssembly ---------------------------------------------------------
# Build the solver for the browser and stage it into the demo page
# (crates/pounce-wasm/web). Needs the target once:
#   rustup target add wasm32-wasip1
# See docs/src/wasm.md.
wasm:
	crates/pounce-wasm/build.sh

wasm-serve:
	crates/pounce-wasm/build.sh --serve

book:
	mdbook build docs

# The full deployed site: stable at the root, dev/ and every archived v* tag,
# plus the docs assistant (docs/src/ask.md) and its retrieval index. `make
# book` deliberately does NOT include the assistant — it is injected per page
# by this script, not configured in book.toml — so this is the target to use
# when changing docs/assets/ask.{js,css}. Needs the release tags present
# (`git fetch --tags`). Set POUNCE_WIKI_DIR to a pounce.wiki clone to index
# the wiki as CI does; without it the index is book-only.
book-site:
	scripts/build-versioned-docs.sh ./site
	@echo
	@echo "Serve it under the real base path (the assistant resolves relative"
	@echo "paths, and a bad one only shows up below the top level):"
	@echo "  mkdir -p _serve/pounce && cp -a site/. _serve/pounce/"
	@echo "  python3 -m http.server -d _serve 8000   # http://localhost:8000/pounce/"

# Retrieval guard for the docs assistant: builds the index from docs/src and
# runs the shipped ask.js against a labelled query set. Same two lines CI runs.
ask-check:
	python3 scripts/build-docs-index.py -o /tmp/pounce-ask-index.json --quiet
	node docs/tests/ask_retrieval.mjs /tmp/pounce-ask-index.json

# Record asciinema screencasts of `pounce --debug` (one per scenario in
# scripts/demo/scenarios/) into docs/demo/*.{cast,gif}. Requires asciinema,
# python pexpect, and pounce on PATH; agg is optional (cast -> gif).
screencast:
	scripts/demo/record.sh

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

# ---- Container images ----------------------------------------------------
# Both Dockerfiles must be built from the repository root, because the source
# build's context has to contain Cargo.toml, crates/, python/ and
# pyomo-pounce/. These targets exist mostly to supply the two build args that
# are easy to forget by hand; see docker/README.md.
#
# The images run their own smoke test as the final build step (CLI solve,
# import pounce, Pyomo plugin lookup), so a build that succeeds is an image
# that works — there is no separate `docker-test`.
DOCKER        ?= docker
DOCKER_IMAGE  ?= pounce
# Read straight out of Cargo.toml so the tag cannot drift from the release.
# scripts/check-release-consistency.sh already guarantees the two PyPI
# manifests agree with this value.
POUNCE_VERSION := $(shell grep -m1 -E '^version[[:space:]]*=' Cargo.toml \
                    | sed -E 's/^version[[:space:]]*=[[:space:]]*"([^"]+)".*/\1/')

# Compile the current working tree. POUNCE_BUILD_GIT is passed in because
# .git is deliberately kept out of the build context (see .dockerignore) —
# without it the image cannot report which commit it came from. Falls back to
# "unknown" outside a git checkout rather than failing the build.
docker:
	$(DOCKER) build -f docker/Dockerfile -t "$(DOCKER_IMAGE):dev" \
	  --build-arg POUNCE_BUILD_GIT="$$(git rev-parse --short=8 HEAD 2>/dev/null || echo unknown)" \
	  --build-arg POUNCE_VERSION="$(POUNCE_VERSION)" \
	  .

# Install the published wheels. No Rust toolchain involved; builds in
# seconds. Requires $(POUNCE_VERSION) to actually be on PyPI, which it is
# only after the python-v$(POUNCE_VERSION) tag has been released.
docker-release:
	$(DOCKER) build -f docker/Dockerfile.release \
	  -t "$(DOCKER_IMAGE):$(POUNCE_VERSION)" \
	  --build-arg POUNCE_VERSION="$(POUNCE_VERSION)" \
	  .

# Build pounce-cli with the HSL MA57 backend enabled and emit the
# binary as `pounce-ma57` so it sits alongside the default `pounce`.
# Requires libcoinhsl discoverable to the linker.
pounce-ma57:
	$(CARGO) build -p pounce-cli $(CARGO_PROFILE_FLAG) --features ma57 $(CARGO_FLAGS)
	cp "$(CLI_BIN)" "$(TARGET_DIR)/pounce-ma57"
	@echo
	@echo "Built $(TARGET_DIR)/pounce-ma57"
	@echo "Run it like the default pounce, e.g.:"
	@echo "    $(TARGET_DIR)/pounce-ma57 problem.nl linear_solver=ma57"
	@echo "Add $(abspath $(TARGET_DIR)) to PATH or copy the binary into ~/.local/bin"
	@echo "to invoke it as just 'pounce-ma57'."

# Run the whole workspace with the MA57 backend actually LINKED.
#
# CI cannot do this: it has no CoinHSL, so `ci.yml` passes
# `--exclude pounce-hsl` everywhere and compile-checks that crate with
# `cargo check`, which neither links nor runs anything. The entire MA57
# link path is therefore exercised only here, by someone with an install
# — which is how gh#811's defect survived in four more crates than the
# one it was fixed in.
#
# Every crate that links coinhsl needs its own `ma57` feature, because
# the rpath travels as `links` metadata to *direct* dependents only; the
# feature list below is the set, and a new such crate has to join it.
#
# pounce-wasm is excluded rather than added to it. Enabling `ma57`
# anywhere in the workspace unifies the feature onto pounce-algorithm, so
# pounce-wasm's *host* test binary links coinhsl and needs the rpath like
# any other — but its real target is wasm32, where libcoinhsl does not
# exist and MA57 can never run. Giving it an `ma57` feature to satisfy a
# host-only build artifact would advertise a backend it cannot have.
#
# Needs COINHSL_DIR pointing at an install whose lib/ holds libcoinhsl.
MA57_FEATURES := pounce-cli/ma57,pounce-cinterface/ma57,pounce-algorithm/ma57,\
pounce-py/ma57,pounce-restoration/ma57,pounce-rs/ma57,pounce-sensitivity/ma57

test-ma57:
	@test -n "$(COINHSL_DIR)" || { \
	  echo "COINHSL_DIR is not set. Point it at a CoinHSL install whose"; \
	  echo "lib/ holds libcoinhsl.dylib (or .so), e.g."; \
	  echo "    make test-ma57 COINHSL_DIR=/path/to/CoinHSL"; \
	  exit 1; }
	$(CARGO) test --workspace --exclude pounce-wasm $(CARGO_PROFILE_FLAG) \
	  --features "$(MA57_FEATURES)" $(CARGO_FLAGS)

help:
	@sed -n 's/^# \{0,1\}//p' Makefile | sed -n '1,48p'

# ---- Python extension + tests -------------------------------------------
# Rebuild the native extension in place, then run the Python test suite.
# This is the safe way to run pytest: a stale in-place `_pounce*.so` (left
# by an earlier `maturin develop`) silently shadows the current binding and
# makes the suite fail with confusing errors. `python-ext` rebuilds it, and
# `python/tests/conftest.py` additionally guards against running pytest
# against a stale artifact. Requires `maturin` and the test extras in the
# active environment (`pip install -e 'python[dev]'`).
#
# `python-ext` also stages the CLI binary into `python/pounce/bin/`, because
# `maturin develop` does not. The wheel ships the Rust `pounce` executable
# there and installs a console-script shim that execs it (see
# `python/pounce/_cli.py`); `maturin develop` installs that same shim but
# builds only the extension module, so in a dev install the shim always points
# at a file nothing ever created. The visible symptom is not an error: the
# broken shim sits later on `PATH` than `~/.local/bin`, so `pounce` silently
# resolves to whatever older binary was installed there — and anything driving
# the CLI (Pyomo via NL/SOL, the benchmark harness) quietly runs that instead.
# A stale binary that still works is far worse than one that fails.
#
# `python/pounce/bin/` is already gitignored for exactly this artifact.
#
# The `rm -f` before `install` is load-bearing. If the staged path is a symlink
# back to $(CLI_BIN) — an easy thing to set up by hand, since it makes the stage
# track rebuilds automatically — then `install` sees one file, refuses with
# "are the same file", and exits 64. That takes down `python-ext` and
# `python-test` with it, including the restore step `make coverage` tells you to
# run. Removing the destination first makes the target idempotent whatever is
# there (real file, symlink, or nothing) and keeps the stage a real copy, which
# is what the wheel ships. `rm -f` on a symlink unlinks the symlink, not
# $(CLI_BIN).
python-cli-bin:
	$(CARGO) build -p pounce-cli $(CARGO_PROFILE_FLAG) $(CARGO_FLAGS)
	install -d python/pounce/bin
	rm -f python/pounce/bin/pounce
	install -m 0755 "$(CLI_BIN)" python/pounce/bin/pounce

python-ext: python-cli-bin
	cd python && maturin develop --release

# The one command that makes a source checkout behave like an installed
# wheel. `maturin develop` on its own builds the extension module and
# nothing else, which leaves the `pounce` console script pointing at a
# bundled binary that was never built — every CLI invocation fails,
# `--version` included, and Pyomo reports the solver unavailable while an
# in-process solve works fine (gh #816). `dev` is `python-ext` under the
# name the error messages tell people to run.
dev: python-ext
	@echo
	@echo "Source checkout ready:"
	@echo "  CLI staged at python/pounce/bin/pounce (what the wheel ships)"
	@echo "  extension module built in place"
	@echo "Verify with: pounce --version"

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
