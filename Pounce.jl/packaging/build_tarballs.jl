# Local release preparation only. Publication/registration needs separate approval.
using BinaryBuilder

for arg in ARGS
    if startswith(arg, "--register") ||
       (startswith(arg, "--deploy") && arg != "--deploy=local")
        error(
            "This preparation recipe allows only local output (--deploy=local); publication is a separate release step.",
        )
    end
end

name = "Pounce"
version = v"0.15.0"
sources = [
    GitSource(
        "https://github.com/jkitchin/pounce.git",
        "b4e5ffb641990738fd69cb53c02ae5966fd68edc",
    ),
]

script = raw"""
cd ${WORKSPACE}/srcdir/pounce
cargo build --release --locked -p pounce-cinterface
install -Dvm 0755 target/${rust_target}/release/*pounce_cinterface.${dlext} ${libdir}/libpounce_cinterface.${dlext}
install -Dvm 0644 crates/pounce-cinterface/include/pounce.h ${includedir}/pounce.h
install_license LICENSE
"""

# Release candidates, not a claim that all of these have passed validation.
# Windows is deliberately not advertised until its binary path and suite pass.
platforms =
    [Platform(arch, os) for arch in ("x86_64", "aarch64") for os in ("linux", "macos")]
products = [LibraryProduct("libpounce_cinterface", :libpounce_cinterface)]
# Rust's Linux unwinder links libgcc_s. Declare it so the audit resolves the
# dependency and JLL initialization loads the matching runtime explicitly.
dependencies =
    [Dependency("CompilerSupportLibraries_jll"; platforms = filter(Sys.islinux, platforms))]

build_tarballs(
    ARGS,
    name,
    version,
    sources,
    script,
    platforms,
    products,
    dependencies;
    julia_compat = "1.10",
    compilers = [:c, :rust],
    preferred_rust_version = v"1.91.0",
)
