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
    setglobal!(Pounce, :NativeResiduals, NativeResiduals)
    setglobal!(Pounce, :RestorationStatistics, RestorationStatistics)
    setglobal!(
        Pounce,
        :FiniteDifferenceHessianStatistics,
        FiniteDifferenceHessianStatistics,
    )
    setglobal!(Pounce, :LinearSolverStatistics, LinearSolverStatistics)
    setglobal!(Pounce, :SolveReport, SolveReport)
    setglobal!(Pounce, :RecordIterationHistory, RecordIterationHistory)
    setglobal!(Pounce, :ComplementarityStatistics, ComplementarityStatistics)
    setglobal!(Pounce, :ComplementaritySolves, ComplementaritySolves)
    setglobal!(Pounce, :ComplementarityText, ComplementarityText)
    setglobal!(Pounce, :MOI, MOI)
    return
end
end
