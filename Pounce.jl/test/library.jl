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
    output = IOBuffer()
    unrelated = Pounce.Libdl.dlpath("libjulia")
    process =
        run(pipeline(ignorestatus(child(unrelated)); stdout = output, stderr = output))
    @test !success(process)
    @test occursin("required POUNCE C API symbol", String(take!(output)))
    mktempdir() do dir
        missing = joinpath(dir, "missing-library")
        output = IOBuffer()
        process =
            run(pipeline(ignorestatus(child(missing)); stdout = output, stderr = output))
        @test !success(process)
        @test occursin("POUNCE library not found", String(take!(output)))
    end
end

@testset "Native version compatibility" begin
    @test Pounce._check_native_version(v"0.15.0") === nothing
    @test Pounce._check_native_version(v"0.15.1") === nothing
    @test Pounce._check_native_version(v"0.16.0-rc1") === nothing
    @test Pounce._check_native_version(v"0.16.0") === nothing
    @test Pounce._check_native_version(v"0.16.99") === nothing
    for version in (v"0.14.9", v"0.15.0-rc1", v"0.17.0-rc1", v"0.17.0", v"1.0.0")
        @test_throws ErrorException Pounce._check_native_version(version)
    end
end
