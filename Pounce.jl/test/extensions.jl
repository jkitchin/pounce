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
        code = """
            import Pkg
            Pkg.develop(path = $(repr(path)))
            @assert all(info.name != "MathOptInterface" for info in values(Pkg.dependencies()))
            import Pounce
            @assert Base.get_extension(Pounce, :PounceMathOptInterfaceExt) === nothing
            @assert Pounce.GetIpoptVersion() isa VersionNumber
            print("installed-core-only")
        """
        cmd = `$(Base.julia_cmd()) --startup-file=no --project=$dir -e $code`
        @test read(cmd, String) == "installed-core-only"
    end
end
