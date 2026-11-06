using Test
import Pounce

println("Julia ", VERSION, "; POUNCE ", Pounce.GetIpoptVersion())
println("Native library: ", Pounce.library_path())
flush(stdout)

@testset "Pounce.jl" begin
    include("library.jl")
    include("extensions.jl")
    include("cinterface.jl")
    @testset "Threaded C interface subprocess" begin
        script = joinpath(@__DIR__, "threading.jl")
        cmd = `$(Base.julia_cmd()) --startup-file=no --threads=2 --project=$(Base.active_project()) $script`
        @test success(run(cmd))
    end
    include("diagnostics.jl")
    include("jump.jl")
    include("updates.jl")
    include("callbacks.jl")
    include("lifecycle.jl")
    include("mixed_nlp.jl")
    include("MOI_wrapper.jl")
end
