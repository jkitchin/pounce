# Experiment 2, CALIPSO side: the same implicit contact step as
# exp_contact_sens.py, differentiated by CALIPSO's own implicit-function
# solve (solver.data.solution_sensitivity), at the same base points.
#
#   julia --project=<env with CALIPSO, JSON> exp_contact_sens.jl > results_contact_sens_calipso.jsonl
#   TIGHT=1 julia ... >> results_contact_sens_calipso.jsonl     (1e-8 tolerances)
#
# bound: x = [z],     cone x >= 0,  objective 1/2 (z - zhat(v))^2
# comp:  x = [z, g],  cone x >= 0,  equality [z - zhat(v) - g; z * g]  (CALIPSO's
#        own way of writing contact), objective 0
# The parameter is v (theta = [v]).

using CALIPSO
using JSON
using LinearAlgebra

const Z0, H, G = 1.0, 0.1, 9.81
const V_STAR = (H * H * G - Z0) / H
zhat(v) = Z0 + H * v - H * H * G
exact_z(v) = max(0.0, zhat(v))

# CALIPSO's defaults stop at 1e-4; TIGHT=1 asks for 1e-8 on every criterion,
# so its derivative is judged at the accuracy POUNCE's is.
const TIGHT = get(ENV, "TIGHT", "") == "1"
const TOLS = TIGHT ? (residual_tolerance=1e-8, optimality_tolerance=1e-8, equality_tolerance=1e-8,
                      complementarity_tolerance=1e-8, slack_tolerance=1e-8) : NamedTuple()

function solve_form(form, v)
    if form == "bound"
        obj(x, θ) = 0.5 * (x[1] - (Z0 + H * θ[1] - H * H * G))^2
        eq(x, θ) = zeros(0)
        cone(x, θ) = x
        n = 1
        x0 = [max(exact_z(v), 0.1)]
    else
        obj2(x, θ) = 0.0 * x[1]
        eq2(x, θ) = [x[1] - (Z0 + H * θ[1] - H * H * G) - x[2]; x[1] * x[2]]
        cone2(x, θ) = x
        obj, eq, cone = obj2, eq2, cone2
        n = 2
        x0 = [max(exact_z(v), 0.1), 0.1]
    end
    solver = Solver(obj, eq, cone, n; parameters=[v],
        options=Options(; verbose=false, differentiate=true, TOLS...))
    initialize!(solver, x0)
    converged = solve!(solver)
    z = solver.solution.variables[1]
    dzdv = solver.data.solution_sensitivity[1, 1]
    return converged, z, dzdv
end

for form in ("bound", "comp")
    for off in (0.0, 1e-6, -1e-6, 1e-3, -1e-3)
        v = V_STAR + off
        converged, z, dzdv = solve_form(form, v)
        for d in (1e-2, -1e-2, 1e-3, -1e-3)
            zh = zhat(v)
            ex = (zh > 1e-12 || (abs(zh) <= 1e-12 && d > 0)) ? H : 0.0
            z_est = z + dzdv * d
            rec = Dict("solver" => TIGHT ? "calipso-tol1e-8" : "calipso", "form" => form, "base_offset" => off,
                "converged" => converged, "z" => z, "z_exact" => exact_z(v),
                "dzdv" => dzdv, "method" => "calipso_ift", "d" => d, "exact_slope" => ex,
                "z_est" => z_est, "slope" => dzdv, "err" => abs(dzdv - ex),
                "val_err" => abs(z_est - exact_z(v + d)) / abs(d))
            println(JSON.json(rec))
        end
    end
end
