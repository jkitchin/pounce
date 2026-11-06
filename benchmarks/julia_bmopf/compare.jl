using JSON3
length(ARGS) == 2 || error("Usage: compare.jl pounce.json ipopt.json")
a, b = [JSON3.read(read(path, String)) for path in ARGS]
@assert a.solver == "pounce" && b.solver == "ipopt"
@assert a.model_commit == b.model_commit && a.fixture_sha256 == b.fixture_sha256
@assert a.julia == b.julia && a.dependencies.JuMP == b.dependencies.JuMP
@assert length(a.rows) == length(b.rows) == 10
for (x, y) in zip(a.rows, b.rows)
    @assert (x.label, x.factor, x.variables, x.constraints) ==
            (y.label, y.factor, y.variables, y.constraints)
    @assert x.physical_errors == y.physical_errors == 0
    @assert max(x.max_model_violation, y.max_model_violation) <= 1e-6
    for row in (x, y)
        @assert row.independent_physics.max_kcl_ampere <= 1e-5
        @assert row.independent_physics.max_branch_current_error_ampere <= 1e-5
    end
    @assert isapprox(x.objective, y.objective; rtol = 1e-6, atol = 1e-6)
    println(
        x.label,
        " factor=",
        x.factor,
        " objective=",
        x.objective,
        " / ",
        y.objective,
        " residual=",
        x.max_model_violation,
        " / ",
        y.max_model_violation,
    )
end
println("All 10 cross-solver comparisons passed.")
