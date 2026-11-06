using JuMP, Pounce

model = Model(Pounce.Optimizer)
set_silent(model)
@variable(model, 1 <= x[1:4] <= 5)
set_start_value.(x, [1.0, 5.0, 5.0, 1.0])
@objective(model, Min, x[1] * x[4] * (x[1] + x[2] + x[3]) + x[3])
@constraint(model, prod(x) >= 25)
@constraint(model, sum(x .^ 2) == 40)
optimize!(model)

@show termination_status(model)
@show objective_value(model)
@show value.(x)
