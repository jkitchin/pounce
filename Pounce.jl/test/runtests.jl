using Test
import Pounce

println("Julia ", VERSION, "; POUNCE ", Pounce.GetIpoptVersion())
println("Native library: ", Pounce.library_path())
flush(stdout)

@testset "Pounce.jl" begin
    include("library.jl")
    include("cinterface.jl")
    include("jump.jl")
    include("MOI_wrapper.jl")
end
