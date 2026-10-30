#!/usr/bin/env bash
# Build the POUNCE wasm module and stage it for the demo page.
#
#   crates/pounce-wasm/build.sh          # build + copy into web/
#   crates/pounce-wasm/build.sh --serve  # ...and serve web/ on :8000
#   crates/pounce-wasm/build.sh --studio [--fast]  # the studio modules (below)
#
# Requires the wasm target once:  rustup target add wasm32-wasip1
#
# This builds the standalone `wasm32-wasip1` module — the `.nl` page, and the
# Python page's `import pounce_browser` route. It does NOT build the
# `pounce-solver` wheel the Python page's `import pounce` route installs:
# that is a `wasm32-unknown-emscripten` build with its own pinned toolchain,
# and it lives in `crates/pounce-wasm/build-wheel.sh`.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd "$here/../.." && pwd)"
target=wasm32-wasip1

# `--studio [--fast]`: the two modules the pounce studio front end loads,
# staged in studio/ with the host it drives them through (studio/README.md):
#
#   pounce-studio.wasm        the `studio` feature build (adds the debugger
#                             engine and `pounce_debug_solve`), wasm-opt -Oz;
#                             the Solve route.
#   pounce-studio-debug.wasm  the same module after Binaryen's Asyncify pass,
#                             which lets the debugger's blocking stdin read
#                             suspend into JavaScript; the Debug route.
#
# Two modules because Asyncify instruments every function that can reach the
# read, and the debugger hook is an indirect call inside the iteration loop,
# so that is nearly all of them: measured on deb7 (5 interleaved solves
# each), 1000 ms (sd 96) without the pass against 1617 ms (sd 253) with it.
# A plain solve should not pay that. `--fast` skips the optimizer after
# Asyncify (about 40 s instead of several minutes, at roughly twice the
# size), which is what CI's protocol test uses.
if [[ "${1:-}" == "--studio" ]]; then
  if ! command -v wasm-opt >/dev/null 2>&1; then
    echo "build.sh --studio needs wasm-opt (binaryen) for the Asyncify pass" >&2
    exit 1
  fi
  cargo build --manifest-path "$root/Cargo.toml" \
    -p pounce-wasm --features studio --release --target "$target" \
    --config 'profile.release.debug=false' \
    --config 'profile.release.strip="symbols"'
  out="$root/target/$target/release/pounce_wasm.wasm"
  mkdir -p "$here/studio"
  cp "$here/web/wasi.js" "$here/studio/wasi.js"
  asyncify=(--asyncify --pass-arg=asyncify-imports@wasi_snapshot_preview1.fd_read)
  if [[ "${2:-}" == "--fast" ]]; then
    cp "$out" "$here/studio/pounce-studio.wasm"
    wasm-opt -all "${asyncify[@]}" "$out" -o "$here/studio/pounce-studio-debug.wasm"
  else
    wasm-opt -Oz -all "$out" -o "$here/studio/pounce-studio.wasm"
    wasm-opt -O2 -all "${asyncify[@]}" "$out" -o "$here/studio/pounce-studio-debug.wasm"
  fi
  for f in pounce-studio.wasm pounce-studio-debug.wasm; do
    size=$(wc -c < "$here/studio/$f")
    printf 'studio/%-25s %s bytes (%.1f MB)\n' "$f" "$size" "$(echo "$size" | awk '{print $1/1048576}')"
  done
  exit 0
fi

# The workspace release profile carries debug info (`debug = 1`), which is
# ~10x the code size in a wasm module and useless to the page. Override it
# for this build only rather than changing the profile everyone else uses.
cargo build --manifest-path "$root/Cargo.toml" \
  -p pounce-wasm --release --target "$target" \
  --config 'profile.release.debug=false' \
  --config 'profile.release.strip="symbols"'

out="$root/target/$target/release/pounce_wasm.wasm"

# wasm-opt (binaryen) is optional; it typically takes another ~15% off.
if command -v wasm-opt >/dev/null 2>&1; then
  # rustc's wasm32-wasip1 output uses post-MVP features — bulk memory,
  # non-trapping float-to-int, sign extension, SIMD — and wasm-opt rejects
  # the whole module ("error validating input") unless each one is enabled
  # for validation. Enumerating them is a losing game: the exact set moves
  # with the Rust release, and distro binaryen lags badly enough to disagree
  # about the defaults (Ubuntu noble ships 108, from 2022; a current Homebrew
  # is 131). `-all` enables everything the local binaryen knows about. It
  # only widens what wasm-opt will *accept* — the emitted feature set is
  # still whatever rustc produced, so this does not change what the page
  # requires of a browser.
  wasm-opt -Oz -all "$out" -o "$here/web/pounce.wasm"
else
  cp "$out" "$here/web/pounce.wasm"
fi

# The Pyodide app runs the same module and the same WASI shim; each app
# directory stays self-contained so either one can be copied to a host on
# its own.
cp "$here/web/pounce.wasm" "$here/web-python/pounce.wasm"
cp "$here/web/wasi.js" "$here/web-python/wasi.js"

size=$(wc -c < "$here/web/pounce.wasm")
printf 'web/pounce.wasm  %s bytes (%.1f MB)\n' "$size" "$(echo "$size" | awk '{print $1/1048576}')"

if [[ "${1:-}" == "--serve" ]]; then
  echo "serving $here/web on http://localhost:8000 (Pyodide app: --serve-python)"
  # Any static server works; the page needs no headers beyond correct MIME
  # types for .wasm and .js.
  cd "$here/web" && python3 -m http.server 8000
fi

if [[ "${1:-}" == "--serve-python" ]]; then
  echo "serving $here/web-python on http://localhost:8000"
  cd "$here/web-python" && python3 -m http.server 8000
fi
