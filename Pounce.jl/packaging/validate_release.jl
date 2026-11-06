# Run in a fresh Julia process on the candidate artifact's target platform.
using Pkg, TOML
length(ARGS) in (1, 2) ||
    error("Usage: validate_release.jl /candidate-environment [--no-rust]")
if length(ARGS) == 2
    ARGS[2] == "--no-rust" || error("Unknown validation flag")
    Sys.which("cargo") === nothing || error("Clean-install check requires no cargo on PATH")
    Sys.which("rustc") === nothing || error("Clean-install check requires no rustc on PATH")
end
haskey(ENV, "POUNCE_LIBRARY_PATH") &&
    error("Unset POUNCE_LIBRARY_PATH to test default artifact selection")
Pkg.activate(abspath(ARGS[1]))
Pkg.instantiate()
using Pounce, Pounce_jll
@assert realpath(Pounce.library_path()) == realpath(Pounce_jll.libpounce_cinterface_path)
@assert !isdefined(Pounce, :Optimizer)
@assert Base.get_extension(Pounce, :PounceMathOptInterfaceExt) === nothing
@assert all(id.name != "MathOptInterface" for id in keys(Base.loaded_modules))
@assert any(
    startswith(
        realpath(Pounce.library_path()),
        realpath(joinpath(depot, "artifacts")) * Base.Filesystem.path_separator,
    ) for depot in DEPOT_PATH if isdir(joinpath(depot, "artifacts"))
)
println("Candidate: ", TOML.parsefile(joinpath(ARGS[1], "Candidate.toml")))
println("Julia: ", VERSION, "; machine: ", Sys.MACHINE)
println("Artifact library: ", Pounce.library_path())
println("Native version: ", Pounce.GetIpoptVersion())
flush(stdout)
Pkg.test("Pounce")
