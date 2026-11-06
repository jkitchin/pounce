module PounceMathOptInterfaceExt
import Pounce
import MathOptInterface as MOI
import PrecompileTools

include("MOI_wrapper.jl")
include("precompile.jl")

function __init__()
    setglobal!(Pounce, :Optimizer, Optimizer)
    setglobal!(Pounce, :CallbackFunction, CallbackFunction)
    setglobal!(Pounce, :NativeSolveTimeSec, NativeSolveTimeSec)
    setglobal!(Pounce, :MOI, MOI)
    return
end
end
