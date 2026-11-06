# Create an isolated environment; keep the resulting Manifest.toml with results.
using Pkg
length(ARGS) == 1 || error("Usage: julia setup.jl /path/to/benchmark-environment")
Pkg.activate(only(ARGS))
Pkg.develop(path = normpath(joinpath(@__DIR__, "..", "..", "Pounce.jl")))
Pkg.add([
    PackageSpec(
        url = "https://github.com/frederikgeth/BMOPFTools.jl",
        rev = "542a48ba7f4c95e50b8acdc790a5323a4ef24930",
    ),
    PackageSpec(name = "JuMP", version = "1.32.1"),
    PackageSpec(name = "Ipopt", version = "1.16.0"),
    PackageSpec(name = "JSON3", version = "1.14.3"),
])
