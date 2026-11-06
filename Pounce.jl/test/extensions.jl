@testset "Optional MOI extension" begin
    function extension_child(code)
        cmd = `$(Base.julia_cmd()) --startup-file=no --project=$(Base.active_project()) -e $code`
        return read(cmd, String)
    end
    @test extension_child("""
        import Pounce
        @assert Base.get_extension(Pounce, :PounceMathOptInterfaceExt) === nothing
        @assert !isdefined(Pounce, :Optimizer)
        @assert all(id.name != "MathOptInterface" for id in keys(Base.loaded_modules))
        print("core-only")
    """) == "core-only"
    for imports in
        ("import Pounce; import MathOptInterface", "import MathOptInterface; import Pounce")
        @test extension_child(imports * """
            ; ext = Base.get_extension(Pounce, :PounceMathOptInterfaceExt)
            @assert Pounce.Optimizer === ext.Optimizer
            @assert Pounce.CallbackFunction === ext.CallbackFunction
            @assert Pounce.NativeSolveTimeSec === ext.NativeSolveTimeSec
            @assert Pounce.MOI === MathOptInterface
            @assert Pounce.Optimizer() isa MathOptInterface.AbstractOptimizer
            print("extension")
        """) == "extension"
    end
    # A fresh environment containing only Pounce must not resolve/install MOI.
    mktempdir() do dir
        path = pkgdir(Pounce)
        # A locally generated, unpublished JLL is developed alongside the release
        # candidate. The normal source-checkout package has no JLL dependency.
        paths = [path]
        if isdefined(Pounce, :Pounce_jll)
            push!(paths, pkgdir(Pounce.Pounce_jll))
        end
        code = """
            import Pkg
            Pkg.develop([Pkg.PackageSpec(path = p) for p in $(repr(paths))])
            @assert all(info.name != "MathOptInterface" for info in values(Pkg.dependencies()))
            import Pounce
            @assert Base.get_extension(Pounce, :PounceMathOptInterfaceExt) === nothing
            @assert Pounce.GetIpoptVersion() isa VersionNumber
            print("installed-core-only")
        """
        cmd = `$(Base.julia_cmd()) --startup-file=no --project=$dir -e $code`
        # Pkg.test restricts JULIA_LOAD_PATH. The fresh project must be able to
        # load the Pkg stdlib, while still excluding the user's global project.
        cmd = addenv(
            cmd,
            "JULIA_LOAD_PATH" => join(["@", "@stdlib"], Sys.iswindows() ? ';' : ':'),
        )
        @test read(cmd, String) == "installed-core-only"
    end
end
