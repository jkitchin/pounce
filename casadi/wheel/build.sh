#!/usr/bin/env bash
# Stage the plugin for the *installed* casadi and build a wheel.
#
#   ./build.sh                 # wheel for the casadi in this environment
#
# A release build runs this once per (casadi minor x platform) inside the
# appropriate manylinux / macOS / Windows image and merges the staged
# _plugins/<minor>/ trees into a single wheel per platform.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$here/../.."

ver=$(python3 -c "import casadi; print(casadi.__version__)")
minor=$(python3 -c "import casadi; print('.'.join(casadi.__version__.split('.')[:2]))")
echo "building pounce-casadi for casadi $ver"

cargo build --release -p pounce-cinterface --manifest-path "$root/Cargo.toml"
make -C "$here/.." "$@"

dest="$here/pounce_casadi/_plugins/$minor"
mkdir -p "$dest"
cp "$here/../libcasadi_nlpsol_pounce."* "$dest/"
cp "$root/target/release/libpounce_cinterface."* "$dest/" 2>/dev/null || true

python3 -m pip wheel --no-deps -w "$here/dist" "$here"
echo "wheel written to $here/dist"
