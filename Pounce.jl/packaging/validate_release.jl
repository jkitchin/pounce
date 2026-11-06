# Run in a fresh Julia process on the candidate artifact's target platform.
using Pkg, TOML
!isempty(ARGS) || error(
    "Usage: validate_release.jl /candidate-environment [--no-rust] [--custom-library]",
)
flags = ARGS[2:end]
all(flag in ("--no-rust", "--custom-library") for flag in flags) ||
    error("Unknown validation flag")
if "--no-rust" in flags
    Sys.which("cargo") === nothing || error("Clean-install check requires no cargo on PATH")
    Sys.which("rustc") === nothing || error("Clean-install check requires no rustc on PATH")
end
custom_library = "--custom-library" in flags
if custom_library
    haskey(ENV, "POUNCE_LIBRARY_PATH") ||
        error("Set POUNCE_LIBRARY_PATH to the library to validate")
    isabspath(ENV["POUNCE_LIBRARY_PATH"]) || error("Use an absolute custom library path")
else
    haskey(ENV, "POUNCE_LIBRARY_PATH") &&
        error("Unset POUNCE_LIBRARY_PATH to test default artifact selection")
end
Pkg.activate(abspath(ARGS[1]))
Pkg.instantiate()
using Pounce, Pounce_jll
if custom_library
    @assert realpath(Pounce.library_path()) == realpath(ENV["POUNCE_LIBRARY_PATH"])
    @assert realpath(Pounce.library_path()) !=
            realpath(Pounce_jll.libpounce_cinterface_path)
else
    @assert realpath(Pounce.library_path()) ==
            realpath(Pounce_jll.libpounce_cinterface_path)
    @assert any(
        startswith(
            realpath(Pounce.library_path()),
            realpath(joinpath(depot, "artifacts")) * Base.Filesystem.path_separator,
        ) for depot in DEPOT_PATH if isdir(joinpath(depot, "artifacts"))
    )
end
@assert !isdefined(Pounce, :Optimizer)
@assert Base.get_extension(Pounce, :PounceMathOptInterfaceExt) === nothing
@assert all(id.name != "MathOptInterface" for id in keys(Base.loaded_modules))
println("Candidate: ", TOML.parsefile(joinpath(ARGS[1], "Candidate.toml")))
println("Julia: ", VERSION, "; machine: ", Sys.MACHINE)
println(custom_library ? "Custom library: " : "Artifact library: ", Pounce.library_path())
println("Native version: ", Pounce.GetIpoptVersion())
flush(stdout)
Pkg.test("Pounce")
