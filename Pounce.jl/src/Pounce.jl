module Pounce

import Libdl
# Populated by the optional MOI extension when JuMP/MathOptInterface is loaded.
global Optimizer, CallbackFunction, NativeSolveTimeSec, MOI

# Set at runtime, never serialize an absolute library path or a dlopen handle
# into Julia's precompile cache. Calls name this library explicitly, so Ipopt.jl
# can be loaded alongside Pounce.
libpounce = ""
const _library_handle = Ref{Ptr{Cvoid}}(C_NULL)

# POUNCE is pre-1.0: audit the C ABI before accepting another minor release.
# Required symbols also catch incomplete/custom builds within this version line.
function _check_native_version(version::VersionNumber)
    v"0.15.0" <= version < v"0.16.0-0" || error(
        "Unsupported POUNCE native version $version. Pounce.jl 0.1 supports " *
        "POUNCE 0.15.x; use a compatible library or update the Julia wrapper.",
    )
    return
end

const _required_native_symbols = (
    :GetPounceOptionType,
    :GetIpoptVersion,
    :CreateIpoptProblem,
    :FreeIpoptProblem,
    :AddIpoptStrOption,
    :AddIpoptNumOption,
    :AddIpoptIntOption,
    :OpenIpoptOutputFile,
    :SetIpoptProblemScaling,
    :SetIntermediateCallback,
    :GetIpoptIterCount,
    :GetIpoptSolveTime,
    :IpoptSolve,
    :GetIpoptCurrentIterate,
    :GetIpoptCurrentViolations,
)

function __init__()
    filename = (Sys.iswindows() ? "" : "lib") * "pounce_cinterface." * Libdl.dlext
    path =
        get(ENV, "POUNCE_LIBRARY_PATH", joinpath(@__DIR__, "..", "..", "target", "release"))
    path = abspath(isdir(path) ? joinpath(path, filename) : path)
    isfile(path) || error(
        "POUNCE library not found at $path. Run cargo build --release -p pounce-cinterface " *
        "in the POUNCE repository, or set POUNCE_LIBRARY_PATH to the library file or its directory before using Pounce.",
    )
    handle = Libdl.dlopen(path, Libdl.RTLD_LOCAL | Libdl.RTLD_NOW)
    try
        # POUNCE-specific symbol prevents accidentally loading libipopt, whose
        # similarly named entry points do not establish POUNCE ABI compatibility.
        for symbol in _required_native_symbols
            Libdl.dlsym_e(handle, symbol) != C_NULL || error(
                "The library at $path does not provide the required POUNCE C API symbol $symbol.",
            )
        end
        major, minor, patch = Ref{Cint}(), Ref{Cint}(), Ref{Cint}()
        ccall(
            Libdl.dlsym(handle, :GetIpoptVersion),
            Cvoid,
            (Ptr{Cint}, Ptr{Cint}, Ptr{Cint}),
            major,
            minor,
            patch,
        )
        _check_native_version(VersionNumber(major[], minor[], patch[]))
    catch
        Libdl.dlclose(handle)
        rethrow()
    end
    _library_handle[] = handle
    global libpounce = path
    return
end

"""Return the absolute path of the POUNCE shared library in use."""
library_path() = libpounce

include("C_wrapper.jl")
function column end

end # module Pounce
