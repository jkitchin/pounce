# Build the POUNCE CasADi `Nlpsol` plugin.
#
#   make                 build libcasadi_nlpsol_pounce.so
#   make test            build, then run the parity checks against ipopt
#   make examples        run every script in examples/
#   make clean
#
# Two inputs are needed and both are auto-detected where possible:
#
#   CASADI_LIB   directory holding libcasadi.so — taken from the installed
#                casadi Python package.
#   CASADI_SRC   a CasADi *source* tree at the SAME version, for the
#                internal headers (nlpsol_impl.hpp, plugin_interface.hpp,
#                …). The pip wheel ships only public headers, so this has
#                to be fetched:
#
#                    make fetch-src        # clones the matching tag
#                    make CASADI_SRC=/path/to/casadi
#
# The ABI flags below are not stylistic. CasADi's plugin loader performs
# no version handshake — a plugin built against a different CasADi, a
# different libstdc++ ABI, or a different -D set loads and then
# misbehaves. `abi-flags` reads them back from the installed CasADi so a
# mismatch shows up as a link error, not as a mystery.

PYTHON      ?= python3
CASADI_LIB  ?= $(shell $(PYTHON) -c "import casadi; print(casadi.GlobalOptions.getCasadiPath())" 2>/dev/null)
CASADI_INC  ?= $(shell $(PYTHON) -c "import casadi; print(casadi.GlobalOptions.getCasadiIncludePath())" 2>/dev/null)
CASADI_VER  ?= $(shell $(PYTHON) -c "import casadi; print(casadi.__version__)" 2>/dev/null)
CASADI_SRC  ?= $(CURDIR)/casadi-src

# Repo-relative default for POUNCE's C library and header.
POUNCE_ROOT ?= $(CURDIR)/..
POUNCE_INC  ?= $(POUNCE_ROOT)/crates/pounce-cinterface/include
POUNCE_LIB  ?= $(POUNCE_ROOT)/target/release

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
  SO_EXT := dylib
  RPATH   = -Wl,-rpath,$(CASADI_LIB) -Wl,-rpath,$(POUNCE_LIB)
else
  SO_EXT := so
  RPATH   = -Wl,-rpath,$(CASADI_LIB) -Wl,-rpath,$(POUNCE_LIB)
endif

PLUGIN := libcasadi_nlpsol_pounce.$(SO_EXT)

# Version macros CasADi's headers expect, derived from the installed
# CasADi rather than hard-coded.
VER_MAJOR := $(word 1,$(subst ., ,$(CASADI_VER)))
VER_MINOR := $(word 2,$(subst ., ,$(CASADI_VER)))
VER_PATCH := $(word 3,$(subst ., ,$(CASADI_VER)))

DEFS := -DWITH_DL -DWITH_DEEPBIND -DUSE_CXX11 -DHAVE_MKSTEMPS \
        -DCASADI_WITH_THREAD -DCASADI_WITH_THREADSAFE_SYMBOLICS \
        -DWITH_DEPRECATED_FEATURES \
        -DCASADI_MAJOR_VERSION=$(VER_MAJOR) \
        -DCASADI_MINOR_VERSION=$(VER_MINOR) \
        -DCASADI_PATCH_VERSION=$(VER_PATCH) \
        -DCASADI_VERSION=$(VER_MAJOR)$(VER_MINOR) \
        -DCASADI_IS_RELEASE=1

# The pip wheels are built against the pre-C++11 libstdc++ string ABI.
# Override with CXX11_ABI=1 for a CasADi you built yourself with the
# modern ABI.
CXX11_ABI ?= 0
DEFS      += -D_GLIBCXX_USE_CXX11_ABI=$(CXX11_ABI)

CXXFLAGS ?= -O2 -std=c++17 -fPIC -Wall
INCLUDES  = -I$(CASADI_SRC) -I$(CASADI_INC) -I$(POUNCE_INC)
LDFLAGS   = -L$(CASADI_LIB) -lcasadi -L$(POUNCE_LIB) -lpounce_cinterface $(RPATH)

.PHONY: all check-env fetch-src abi-flags install uninstall test examples clean

all: check-env $(PLUGIN)

check-env:
ifeq ($(strip $(CASADI_LIB)),)
	$(error CasADi not found. `pip install casadi`, or set CASADI_LIB / CASADI_INC)
endif
	@test -f "$(CASADI_SRC)/casadi/core/nlpsol_impl.hpp" || { \
	  echo "CasADi internal headers not found under CASADI_SRC=$(CASADI_SRC)."; \
	  echo "The pip wheel does not ship them. Run: make fetch-src"; \
	  echo "(or point CASADI_SRC at a casadi $(CASADI_VER) source tree)"; \
	  exit 1; }
	@test -f "$(POUNCE_LIB)/libpounce_cinterface.$(SO_EXT)" || { \
	  echo "libpounce_cinterface not found in $(POUNCE_LIB)."; \
	  echo "Run: cargo build --release -p pounce-cinterface"; \
	  exit 1; }

## Clone the CasADi source tree at exactly the installed version.
fetch-src:
	@test -n "$(CASADI_VER)" || { echo "CasADi not installed"; exit 1; }
	git clone --depth 1 --branch $(CASADI_VER) \
	  https://github.com/casadi/casadi $(CASADI_SRC)

## Print the flags the installed CasADi was built with, for comparison
## with DEFS above when a plugin fails to load.
abi-flags:
	@echo "installed casadi $(CASADI_VER) at $(CASADI_LIB)"
	@$(PYTHON) -c "import casadi; print(casadi.CasadiMeta.compiler_flags())"

$(PLUGIN): casadi_nlpsol_pounce.cpp
	$(CXX) -shared $(CXXFLAGS) $(DEFS) $(INCLUDES) $< -o $@ $(LDFLAGS)

## Install the plugin next to the CasADi that will load it. CasADi's
## plugin loader searches its own package directory first, so this makes
## `nlpsol(..., 'pounce', ...)` work with no environment variable — and
## it is the user's site-packages, so no sudo.
install: all
	@test -w "$(CASADI_LIB)" || { \
	  echo "$(CASADI_LIB) is not writable — install CasADi in a venv,"; \
	  echo "or skip this and set CASADIPATH=$(CURDIR) instead."; exit 1; }
	cp $(PLUGIN) $(CASADI_LIB)/
	@echo "installed $(PLUGIN) -> $(CASADI_LIB)"

uninstall:
	rm -f $(CASADI_LIB)/$(PLUGIN)

test: all
	CASADIPATH=$(CURDIR) $(PYTHON) test_parity.py

examples: all
	@for f in examples/*.py; do \
	  echo "=== $$f"; \
	  CASADIPATH=$(CURDIR) $(PYTHON) $$f || exit 1; \
	done

clean:
	rm -f $(PLUGIN)
