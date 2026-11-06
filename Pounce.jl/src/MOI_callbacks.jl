# Copyright (c) 2013: Iain Dunning, Miles Lubin, and contributors
# Adapted for POUNCE; see ../NOTICE.md and ../LICENSE.md.
#
# POUNCE-specific MOI intermediate callbacks and live primal inspection.

"""
    CallbackFunction()

A solver-dependent callback for POUNCE's intermediate callback.
Pass `nothing` to remove the callback between solves. During restoration
(alg_mod == 1), metrics describe the restoration subproblem and primal
inspection is unavailable.

The callback should be a function like the following:
```julia
function my_intermediate_callback(
    alg_mod::Cint,
    iter_count::Cint,
    obj_value::Float64,
    inf_pr::Float64,
    inf_du::Float64,
    mu::Float64,
    d_norm::Float64,
    regularization_size::Float64,
    alpha_du::Float64,
    alpha_pr::Float64,
    ls_trials::Cint,
)
    # ... user code ...
    return true # or `return false` to terminate the solve.
end
```

The arguments are defined in the Ipopt documentation:
https://coin-or.github.io/Ipopt/OUTPUT.html

The MOI adapter manages callback registration at each solve. Use this
attribute for MOI models; use `SetIntermediateCallback` for low-level problems.
"""
struct CallbackFunction <: MOI.AbstractCallback end

MOI.supports(::Optimizer, ::CallbackFunction) = true

function MOI.set(model::Optimizer, ::CallbackFunction, f::Union{Function,Nothing})
    if model.native.problem !== nothing && model.native.problem.solving
        error("Cannot change the intermediate callback during a solve")
    end
    model.callback = f
    return
end

function _dispatch_callback(model::Optimizer, args...)
    state = model.callback_state
    state.task = current_task()
    state.mode = args[1]
    state.has_primal = false
    try
        return model.callback(args...)
    finally
        state.task = nothing
        state.has_primal = false
    end
end

function _setup_callback!(model::Optimizer, inner::Pounce.IpoptProblem)
    if model.callback === nothing
        # The default path never registers a callback, so Rust need not build
        # live iterate snapshots merely to supply an iteration count.
        if inner.intermediate !== nothing
            Pounce.SetIntermediateCallback(inner, nothing)
        end
    else
        Pounce.SetIntermediateCallback(
            inner,
            (args...) -> _dispatch_callback(model, args...),
        )
    end
    return
end

function MOI.get(model::Optimizer, attr::MOI.CallbackVariablePrimal, x::MOI.VariableIndex)
    state = model.callback_state
    if state.task !== current_task()
        throw(
            MOI.GetAttributeNotAllowed(
                attr,
                "Primal inspection requires the active callback task",
            ),
        )
    elseif state.mode != 0
        throw(
            MOI.GetAttributeNotAllowed(
                attr,
                "Primal inspection is unavailable during restoration",
            ),
        )
    end
    MOI.throw_if_not_valid(model, x)
    if _is_parameter(x)
        return model.data.nlp_model[model.data.parameters[x]]
    end
    if !state.has_primal
        inner = model.native.problem::Pounce.IpoptProblem
        resize!(state.primal, inner.n)
        Pounce.GetIpoptCurrentIterate(
            inner,
            false,
            inner.n,
            state.primal,
            C_NULL,
            C_NULL,
            inner.m,
            C_NULL,
            C_NULL,
        )
        state.has_primal = true
    end
    return state.primal[Pounce.column(x)]
end
