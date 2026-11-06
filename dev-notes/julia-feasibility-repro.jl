import Pounce
# Direct-C reproduction of the model in julia-wrapper-validation.md.
# Build pounce-cinterface from this checkout before interpreting the result.
repo = normpath(joinpath(@__DIR__, ".."))
println("Checkout: ", readchomp(`git -C $repo rev-parse HEAD`))
println("POUNCE ", Pounce.GetIpoptVersion(), "; library: ", Pounce.library_path())
function jac(x, r, c, v)
    if v === nothing
        r .= 1:3
        c .= 1
    else
        v .= [1, 1, 1/x[1]]
    end
end
function hess(x, r, c, s, l, v)
    if v === nothing
        r .= 1
        c .= 1
    else
        v[1] = -l[3]/x[1]^2
    end
end
for limited in (true, false), start in (0.0, 0.5)
    p=Pounce.CreateIpoptProblem(
        1,
        [0.0],
        [1.0],
        3,
        [0.1, 0.25, -1.0],
        [1.1, 0.75, 1.0],
        3,
        1,
        x->0.0,
        (x, g)->(g .= [x[1], x[1], log(x[1])]),
        (x, g)->(g .= 0),
        jac,
        hess,
    )
    p.x .= start
    Pounce.AddIpoptIntOption(p, "print_level", 0)
    Pounce.AddIpoptStrOption(
        p,
        "hessian_approximation",
        limited ? "limited-memory" : "exact",
    )
    @show limited start Pounce.IpoptSolve(p) p.x p.g
    Pounce.FreeIpoptProblem(p)
end
