@testset "Library selection after precompilation" begin
    # Child processes use the test environment and the already precompiled
    # package, but read POUNCE_LIBRARY_PATH afresh.
    function child(path)
        code = "using Pounce; print(Pounce.library_path())"
        cmd = `$(Base.julia_cmd()) --startup-file=no --project=$(Base.active_project()) -e $code`
        return addenv(cmd, "POUNCE_LIBRARY_PATH" => path)
    end
    file = Pounce.library_path()
    @test read(child(file), String) == file
    @test read(child(dirname(file)), String) == file
    mktempdir() do dir
        missing = joinpath(dir, "missing-library")
        output = IOBuffer()
        process =
            run(pipeline(ignorestatus(child(missing)); stdout = output, stderr = output))
        @test !success(process)
        @test occursin("POUNCE library not found", String(take!(output)))
    end
end
