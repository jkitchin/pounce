# Compile the shared MOI construction, sparse derivative, and C-callback paths.
# No model, native pointer, or library path is stored in a module-global cache.
PrecompileTools.@setup_workload begin
    PrecompileTools.@compile_workload begin
        model = MOI.Utilities.CachingOptimizer(
            MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}()),
            MOI.instantiate(Optimizer; with_bridge_type = Float64),
        )
        try
            MOI.set(model, MOI.Silent(), true)
            x = MOI.add_variables(model, 2)
            for xi in x
                MOI.add_constraint(model, xi, MOI.Interval(0.5, 3.0))
                MOI.set(model, MOI.VariablePrimalStart(), xi, 1.0)
            end
            f = MOI.ScalarQuadraticFunction(
                [MOI.ScalarQuadraticTerm(2.0, xi, xi) for xi in x],
                MOI.ScalarAffineTerm{Float64}[],
                0.0,
            )
            MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
            MOI.add_constraint(model, f, MOI.LessThan(10.0))
            objective = MOI.ScalarNonlinearFunction(
                :+,
                Any[MOI.ScalarNonlinearFunction(:^, Any[xi, 2]) for xi in x],
            )
            MOI.set(model, MOI.ObjectiveFunction{typeof(objective)}(), objective)
            c = MOI.ScalarNonlinearFunction(:*, Any[x[1], x[2]])
            MOI.add_constraint(model, c, MOI.GreaterThan(1.0))
            MOI.optimize!(model)
            MOI.get(model, MOI.TerminationStatus())
            MOI.get(model, MOI.VariablePrimal(), x)
            MOI.get(model, MOI.ObjectiveValue())
        finally
            MOI.empty!(model)
        end
    end
end
