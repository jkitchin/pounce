# Makefile for the GAMS POUNCE solver link
#
# Usage:
#   make                       — build libGamsPounce.dylib (macOS) or .so (Linux)
#   make GAMS_PATH=/path/gams  — override GAMS installation path
#   make install               — build + install into the GAMS system directory
#   make test                  — install + run test_hs071.gms
#   make clean                 — remove build artifacts (the shared library)
#
# Prerequisites:
#   - GAMS installation (for C API headers and stubs)
#   - libpounce_cinterface built:
#       cargo build --release -p pounce-cinterface   (in repo root)

# --- Paths ---

# GAMS system directory (contains apifiles/)
GAMS_PATH ?= /Library/Frameworks/GAMS.framework/Versions/Current/Resources

# pounce repo root
POUNCE_ROOT ?= ..

# GAMS C API source directory
GAMS_API = $(GAMS_PATH)/apifiles/C/api

# pounce C API header directory
POUNCE_INCLUDE = $(POUNCE_ROOT)/crates/pounce-cinterface/include

# --- Platform detection ---

UNAME := $(shell uname -s)
ifeq ($(UNAME),Darwin)
  SHLIB_EXT   = dylib
  SHLIB_FLAGS = -dynamiclib -install_name @rpath/libGamsPounce.dylib
  RPATH_FLAG  = -Wl,-rpath,$(POUNCE_ROOT)/target/release
else
  SHLIB_EXT   = so
  SHLIB_FLAGS = -shared
  RPATH_FLAG  = -Wl,-rpath,'$$ORIGIN'
endif

# --- Compiler ---

CC       = cc
CFLAGS   = -O2 -Wall -Wextra -Wno-unused-parameter \
           -I$(POUNCE_INCLUDE) -I$(GAMS_API)
LDFLAGS  = -L$(POUNCE_ROOT)/target/release -lpounce_cinterface $(RPATH_FLAG)

# --- Targets ---

TARGET = libGamsPounce.$(SHLIB_EXT)
POUNCE_LIB = libpounce_cinterface.$(SHLIB_EXT)

GAMS_SRCS = $(GAMS_API)/gmomcc.c $(GAMS_API)/gevmcc.c

.PHONY: all clean install test help

all: $(TARGET)

help:
	@echo "Targets:"
	@echo "  all                  Build libGamsPounce.dylib/.so (default)"
	@echo "  install              Build and install library + register POUNCE solver in GAMS"
	@echo "  test                 install + run test_hs071.gms"
	@echo "  clean                Remove build artifacts (shared library)"
	@echo ""
	@echo "Variables:"
	@echo "  GAMS_PATH=<path>     Override GAMS installation directory"

$(TARGET): gams_pounce.c $(GAMS_SRCS)
	$(CC) $(CFLAGS) $(SHLIB_FLAGS) -o $@ $^ $(LDFLAGS)

install: $(TARGET)
	cp $(TARGET) $(GAMS_PATH)/
	cp $(POUNCE_ROOT)/target/release/$(POUNCE_LIB) $(GAMS_PATH)/
ifeq ($(UNAME),Darwin)
	install_name_tool -change \
		"$$(otool -L $(GAMS_PATH)/$(TARGET) | grep $(POUNCE_LIB) | awk '{print $$1}')" \
		@loader_path/$(POUNCE_LIB) \
		$(GAMS_PATH)/$(TARGET)
	install_name_tool -id @loader_path/$(TARGET) $(GAMS_PATH)/$(TARGET)
	# install_name_tool invalidates the linker-signed ad-hoc signature, so
	# macOS will SIGKILL on dlopen ("Code Signature Invalid"). Re-sign ad-hoc.
	codesign --force --sign - $(GAMS_PATH)/$(TARGET)
	codesign --force --sign - $(GAMS_PATH)/$(POUNCE_LIB)
endif
	@grep -q '^POUNCE ' $(GAMS_PATH)/gmscmpun.txt 2>/dev/null || \
		(cp $(GAMS_PATH)/gmscmpun.txt $(GAMS_PATH)/gmscmpun.txt.bak && \
		awk '/^DEFAULTS$$/{print "POUNCE 11 5 00010203040506070809 1 0 2 NLP DNLP RMINLP"; print "gmsgenus.run"; print "gmsgenux.out"; print "libGamsPounce.$(SHLIB_EXT) pou 1 0"}1' \
		$(GAMS_PATH)/gmscmpun.txt.bak > $(GAMS_PATH)/gmscmpun.txt)
	@echo "Installed to $(GAMS_PATH)"

test: install
	$(GAMS_PATH)/gams test_hs071.gms

clean:
	rm -f $(TARGET)
