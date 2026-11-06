@testset "Library selection after precompilation" begin
    # Child processes use the test environment and the already precompiled
    # package, but read POUNCE_LIBRARY_PATH afresh.
    function child(path; with_moi = false)
        code = if with_moi
            """
            import Pounce
            import MathOptInterface as MOI
            model = Pounce.Optimizer()
            MOI.set(model, MOI.Silent(), true)
            MOI.add_variable(model)
            MOI.optimize!(model)
            @assert MOI.get(model, MOI.TerminationStatus()) == MOI.LOCALLY_SOLVED
            print(Pounce.library_path())
            """
        else
            "using Pounce; print(Pounce.library_path())"
        end
        cmd = `$(Base.julia_cmd()) --startup-file=no --project=$(Base.active_project()) -e $code`
        return addenv(cmd, "POUNCE_LIBRARY_PATH" => path)
    end
    file = Pounce.library_path()
    @test read(child(file), String) == file
    @test read(child(dirname(file)), String) == file
    @test read(child(file; with_moi = true), String) == file
    @test read(child(dirname(file); with_moi = true), String) == file
    mktempdir() do dir
        missing = joinpath(dir, "missing-library")
        output = IOBuffer()
        process =
            run(pipeline(ignorestatus(child(missing)); stdout = output, stderr = output))
        @test !success(process)
        @test occursin("POUNCE library not found", String(take!(output)))
    end
end
