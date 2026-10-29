# CALIPSO.jl baseline: run one of the paper's examples with the authors' own
# setup code and record the result as one JSON line.
#
#   julia --project=<env with CALIPSO, RoboDojo, JSON> calipso_baseline.jl \
#       <CALIPSO.jl checkout> <example> <seed[,seed...]> <out.jsonl>
#
# <example> is a path under examples/, e.g. contact_implicit/bunnyhop.jl.
#
# The example's source is executed verbatim up to its first top-level
# `solve!(solver)` line (dropping `Pkg.*` and `using Pkg` lines, so the
# caller's environment is used), which keeps the problem definition, the
# initial guess and the solver options exactly the authors'. Everything after
# that line is visualization and is not run. The solve is then timed, its
# printed iteration table is captured to read the iteration count and final
# residuals, and the primal solution is written out so the Python port can be
# evaluated at CALIPSO's own answer.

using JSON
using Random
using CALIPSO
using RoboDojo
using LinearAlgebra

root, example, seeds, out = ARGS[1], ARGS[2], parse.(Int, split(ARGS[3], ',')), ARGS[4]
path = joinpath(root, "examples", example)

src = read(path, String)
lines = split(src, '\n')
cut = findfirst(l -> occursin(r"^\s*solve!\(solver\)", l), lines)
cut === nothing && error("no top-level solve!(solver) in $example")
# `Pkg.*` would switch environments; MeshCatMechanisms is visualization only
# (quadruped_gait.jl) and not a dependency of anything that runs before the cut.
head = filter(l -> !occursin(r"^\s*(Pkg\.|using Pkg|using MeshCatMechanisms)", l), lines[1:cut-1])
setup = join(head, '\n')

# Methods defined by include_string are newer than this loop's world age, so
# everything that reaches them goes through invokelatest.
build!(seed) = (Random.seed!(seed); include_string(Main, setup, path); getfield(Main, :solver))

for seed in seeds
    # The first solve! in a session includes JIT compilation, which would
    # dominate a small problem's wall time: solve once to compile, then
    # rebuild from the same seed and time the second solve.
    first_wall = NaN
    if get(ENV, "CALIPSO_NO_WARMUP", "") == ""
        s0 = build!(seed)
        first_wall = @elapsed redirect_stdout(devnull) do
            Base.invokelatest(solve!, s0)
        end
    end

    solver = build!(seed)
    logfile = tempname()
    converged = false
    wall = @elapsed open(logfile, "w") do io
        redirect_stdout(io) do
            converged = Base.invokelatest(solve!, solver)
        end
    end

    # last iteration-table row: total outer inner |residual| |equality| |comp| |slack| ...
    rows = filter(l -> occursin(r"^\s*\d+\s+\d+\s+\d+\s+[-\d.eE+]+", l), readlines(logfile))
    last_row = isempty(rows) ? String[] : split(strip(rows[end]))
    num(i) = length(last_row) >= i ? parse(Float64, last_row[i]) : NaN

    rec = Dict{String,Any}(
        "solver" => "calipso",
        "example" => example,
        "seed" => seed,
        "converged" => converged,
        "iters" => length(last_row) >= 1 ? parse(Int, last_row[1]) : 0,
        "solve_wall" => wall,
        "first_solve_wall" => first_wall,
        "residual" => num(4), "equality" => num(5), "comp" => num(6), "slack" => num(7),
    )
    if solver.problem.custom isa CALIPSO.TrajectoryOptimizationProblem
        xs, us = Base.invokelatest(CALIPSO.get_trajectory, solver)
        rec["states"] = [collect(x) for x in xs]
        rec["actions"] = [collect(u) for u in us]
    end
    rec["variables"] = collect(solver.solution.variables)

    open(out, "a") do io
        println(io, JSON.json(rec))
    end
    println("done ", example, " seed=", seed, " converged=", converged,
            " iters=", rec["iters"], " wall=", round(wall, digits=2))
end
