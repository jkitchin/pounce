#!/usr/bin/env bash
# Build the POUNCE wasm module and stage it for the demo page.
#
#   crates/pounce-wasm/build.sh          # build + copy into web/
#   crates/pounce-wasm/build.sh --serve  # ...and serve web/ on :8000
#
# Requires the wasm target once:  rustup target add wasm32-wasip1
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd "$here/../.." && pwd)"
target=wasm32-wasip1

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
  # Rust's wasm32-wasip1 code uses non-trapping float-to-int conversions.
  # Keep the feature enabled when Binaryen validates the module.
  wasm-opt -Oz --enable-bulk-memory --enable-nontrapping-float-to-int --enable-simd \
    "$out" -o "$here/web/pounce.wasm"
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
