# Run in an environment containing Pounce and JuMP. Warm package caches first.
# Each invocation measures one fresh process; repeat for several samples.
using TOML
repo = normpath(joinpath(@__DIR__, ".."))
core_load = @elapsed @eval import Pounce
jump_load = @elapsed @eval using JuMP
function build_probe()
    m = Model(Pounce.Optimizer)
    set_silent(m)
    @variable(m, 1 <= x[1:4] <= 5)
    set_start_value.(x, [1.0, 5.0, 5.0, 1.0])
    @objective(m, Min, x[1] * x[4] * (x[1] + x[2] + x[3]) + x[3])
    @constraint(m, prod(x) >= 25)
    @constraint(m, sum(x .^ 2) == 40)
    return m
end
first_build = @elapsed m = build_probe()
first_solve = @elapsed optimize!(m)
@assert termination_status(m) == MOI.LOCALLY_SOLVED
first_native = MOI.get(m, Pounce.NativeSolveTimeSec())
second_build = @elapsed m2 = build_probe()
second_solve = @elapsed optimize!(m2)
@assert termination_status(m2) == MOI.LOCALLY_SOLVED
TOML.print(
    Dict(
        "wrapper_commit" => readchomp(`git -C $repo rev-parse HEAD`),
        "native_commit" => get(ENV, "POUNCE_NATIVE_REVISION", "unknown"),
        "library_path" => Pounce.library_path(),
        "julia" => string(VERSION),
        "moi" => string(pkgversion(MOI)),
        "jump" => string(pkgversion(JuMP)),
        "pounce" => string(Pounce.GetIpoptVersion()),
        "core_load_seconds" => core_load,
        "jump_load_seconds" => jump_load,
        "first_build_seconds" => first_build,
        "first_solve_seconds" => first_solve,
        "first_native_seconds" => first_native,
        "second_build_seconds" => second_build,
        "second_solve_seconds" => second_solve,
        "second_native_seconds" => MOI.get(m2, Pounce.NativeSolveTimeSec()),
    ),
)
