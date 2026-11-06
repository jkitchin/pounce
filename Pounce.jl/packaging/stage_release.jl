# Produce a reviewable, local binary-backed candidate without modifying source.
using Pkg, TOML

function replace_once(text, before, after)
    length(findall(before, text)) == 1 ||
        error("Release staging template has changed: $before")
    return replace(text, before => after; count = 1)
end

function stage_release(jll, destination)
    jll = abspath(jll)
    destination = abspath(destination)
    ispath(destination) && error("Choose a new candidate directory: $destination")
    jll_project = TOML.parsefile(joinpath(jll, "Project.toml"))
    jll_project["name"] == "Pounce_jll" || error("Expected a generated Pounce_jll package")
    native_version = VersionNumber(jll_project["version"])
    v"0.15.0" <= native_version < v"0.16.0-0" || error("Expected Pounce_jll 0.15.x")
    isfile(joinpath(jll, "Artifacts.toml")) ||
        error("JLL artifacts have not been generated")
    source = dirname(@__DIR__)
    package = joinpath(destination, "Pounce.jl")
    mkpath(package)
    for file in (
        "src",
        "ext",
        "test",
        "examples",
        "Project.toml",
        "README.md",
        "LICENSE.md",
        "NOTICE.md",
    )
        cp(joinpath(source, file), joinpath(package, file))
    end
    project = TOML.parsefile(joinpath(package, "Project.toml"))
    project["deps"]["Pounce_jll"] = jll_project["uuid"]
    project["compat"]["Pounce_jll"] = "0.15"
    open(joinpath(package, "Project.toml"), "w") do io
        TOML.print(io, project; sorted = true)
    end
    entry = joinpath(package, "src", "Pounce.jl")
    text =
        replace_once(read(entry, String), "import Libdl", "import Libdl\nimport Pounce_jll")
    text = replace_once(
        text,
        "joinpath(@__DIR__, \"..\", \"..\", \"target\", \"release\")",
        "Pounce_jll.libpounce_cinterface_path",
    )
    write(entry, text)
    readme = joinpath(package, "README.md")
    text = read(readme, String)
    first_api_section = findfirst("## JuMP example", text)
    first_api_section === nothing && error("README structure changed")
    end_api_section = findfirst("## Tests and development", text)
    end_api_section === nothing && error("README structure changed")
    write(
        readme,
        """
# Pounce.jl — unpublished binary-backed release candidate

This candidate uses the locally generated Pounce_jll $(jll_project["version"]).
Julia 1.10 or later is required; users do not need Rust. Load `JuMP` and
`Pounce`, then use `Model(Pounce.Optimizer)`. MOI remains an optional extension.
Set `POUNCE_LIBRARY_PATH` before importing Pounce to select a custom library.
`Pounce.library_path()` identifies the runtime library, which must provide
the supported POUNCE 0.15.x C ABI.

This directory is local release preparation, not a published package.
Instantiate the environment one directory above this package and run
`Pkg.test("Pounce")`. The native library is supplied by JLL artifacts.

        """ *
        text[first(first_api_section):prevind(text, first(end_api_section))] *
        "## Attribution\n\nSee [NOTICE.md](NOTICE.md) for upstream attribution and " *
        "[LICENSE.md](LICENSE.md) for the Julia wrapper's MIT license. " *
        "The separately packaged native POUNCE library is EPL-2.0.\n",
    )
    Pkg.activate(destination)
    Pkg.develop([PackageSpec(path = jll), PackageSpec(path = package)])
    metadata = Dict(
        "wrapper_commit" => readchomp(`git -C $source rev-parse HEAD`),
        "working_tree" => readchomp(`git -C $source status --porcelain`),
        "jll_version" => jll_project["version"],
        "jll_uuid" => jll_project["uuid"],
        "jll_path" => jll,
    )
    open(joinpath(destination, "Candidate.toml"), "w") do io
        TOML.print(io, metadata; sorted = true)
    end
    println("Staged binary-backed package: ", package)
    return package
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) == 2 ||
        error("Usage: stage_release.jl /path/to/Pounce_jll /new/candidate-directory")
    stage_release(ARGS...)
end
