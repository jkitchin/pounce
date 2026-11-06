# Reproduce the measured dependency graph in a separate environment.
using Pkg, TOML
length(ARGS) == 1 || error("Usage: julia setup.jl /path/to/empty-benchmark-environment")
VERSION >= v"1.12" ||
    error("The benchmark lockfile requires Julia 1.12; the wrapper itself supports 1.10.")
destination = abspath(only(ARGS))
any(isfile(joinpath(destination, f)) for f in ("Project.toml", "Manifest.toml")) &&
    error("Choose a new environment directory to preserve its existing package files.")
mkpath(destination)
cp(joinpath(@__DIR__, "Project.toml"), joinpath(destination, "Project.toml"))
manifest = TOML.parsefile(joinpath(@__DIR__, "Manifest.toml"))
only(manifest["deps"]["Pounce"])["path"] =
    normpath(joinpath(@__DIR__, "..", "..", "Pounce.jl"))
open(joinpath(destination, "Manifest.toml"), "w") do io
    TOML.print(io, manifest; sorted = true)
end
Pkg.activate(destination)
Pkg.instantiate()
