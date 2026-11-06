module Pounce

import Libdl
# Populated by the optional MOI extension when JuMP/MathOptInterface is loaded.
global Optimizer, CallbackFunction, NativeSolveTimeSec, MOI

# Set at runtime, never serialize an absolute library path or a dlopen handle
# into Julia's precompile cache. Calls name this library explicitly, so Ipopt.jl
# can be loaded alongside Pounce.
libpounce = ""
const _library_handle = Ref{Ptr{Cvoid}}(C_NULL)

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
        Libdl.dlsym(handle, :GetPounceOptionType)
        Libdl.dlsym(handle, :IpoptSolve)
    catch
        Libdl.dlclose(handle)
        error("The library at $path does not provide the POUNCE C API.")
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
