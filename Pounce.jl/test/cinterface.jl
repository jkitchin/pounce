# Copyright (c) 2013: Iain Dunning, Miles Lubin, and contributors
# HS071 fixture adapted from Ipopt.jl; see ../NOTICE.md and ../LICENSE.md.

function hs071_problem()
    # hs071
    # min x1 * x4 * (x1 + x2 + x3) + x3
    # st  x1 * x2 * x3 * x4 >= 25
    #     x1^2 + x2^2 + x3^2 + x4^2 = 40
    #     1 <= x1, x2, x3, x4 <= 5
    # Start at (1,5,5,1)
    # End at (1.000..., 4.743..., 3.821..., 1.379...)

    function eval_f(x::Vector{Float64})
        return x[1] * x[4] * (x[1] + x[2] + x[3]) + x[3]
    end

    function eval_g(x::Vector{Float64}, g::Vector{Float64})
        # Bad: g    = zeros(2)  # Allocates new array
        # OK:  g[:] = zeros(2)  # Modifies 'in place'
        g[1] = x[1] * x[2] * x[3] * x[4]
        return g[2] = x[1]^2 + x[2]^2 + x[3]^2 + x[4]^2
    end

    function eval_grad_f(x::Vector{Float64}, grad_f::Vector{Float64})
        # Bad: grad_f    = zeros(4)  # Allocates new array
        # OK:  grad_f[:] = zeros(4)  # Modifies 'in place'
        grad_f[1] = x[1] * x[4] + x[4] * (x[1] + x[2] + x[3])
        grad_f[2] = x[1] * x[4]
        grad_f[3] = x[1] * x[4] + 1
        return grad_f[4] = x[1] * (x[1] + x[2] + x[3])
    end

    function eval_jac_g(
        x::Vector{Float64},
        rows::Vector{Int32},
        cols::Vector{Int32},
        values::Union{Nothing,Vector{Float64}},
    )
        if values === nothing
            # Constraint (row) 1
            rows[1] = 1
            cols[1] = 1
            rows[2] = 1
            cols[2] = 2
            rows[3] = 1
            cols[3] = 3
            rows[4] = 1
            cols[4] = 4
            # Constraint (row) 2
            rows[5] = 2
            cols[5] = 1
            rows[6] = 2
            cols[6] = 2
            rows[7] = 2
            cols[7] = 3
            rows[8] = 2
            cols[8] = 4
        else
            # Constraint (row) 1
            values[1] = x[2] * x[3] * x[4]  # 1,1
            values[2] = x[1] * x[3] * x[4]  # 1,2
            values[3] = x[1] * x[2] * x[4]  # 1,3
            values[4] = x[1] * x[2] * x[3]  # 1,4
            # Constraint (row) 2
            values[5] = 2 * x[1]  # 2,1
            values[6] = 2 * x[2]  # 2,2
            values[7] = 2 * x[3]  # 2,3
            values[8] = 2 * x[4]  # 2,4
        end
        return
    end

    function eval_h(
        x::Vector{Float64},
        rows::Vector{Int32},
        cols::Vector{Int32},
        obj_factor::Float64,
        lambda::Vector{Float64},
        values::Union{Nothing,Vector{Float64}},
    )
        if values === nothing
            # Symmetric matrix, fill the lower left triangle only
            idx = 1
            for row = 1:4
                for col = 1:row
                    rows[idx] = row
                    cols[idx] = col
                    idx += 1
                end
            end
        else
            # Again, only lower left triangle
            # Objective
            values[1] = obj_factor * (2 * x[4])  # 1,1
            values[2] = obj_factor * (x[4])  # 2,1
            values[3] = 0                      # 2,2
            values[4] = obj_factor * (x[4])  # 3,1
            values[5] = 0                      # 3,2
            values[6] = 0                      # 3,3
            values[7] = obj_factor * (2 * x[1] + x[2] + x[3])  # 4,1
            values[8] = obj_factor * (x[1])  # 4,2
            values[9] = obj_factor * (x[1])  # 4,3
            values[10] = 0                     # 4,4

            # First constraint
            values[2] += lambda[1] * (x[3] * x[4])  # 2,1
            values[4] += lambda[1] * (x[2] * x[4])  # 3,1
            values[5] += lambda[1] * (x[1] * x[4])  # 3,2
            values[7] += lambda[1] * (x[2] * x[3])  # 4,1
            values[8] += lambda[1] * (x[1] * x[3])  # 4,2
            values[9] += lambda[1] * (x[1] * x[2])  # 4,3

            # Second constraint
            values[1] += lambda[2] * 2  # 1,1
            values[3] += lambda[2] * 2  # 2,2
            values[6] += lambda[2] * 2  # 3,3
            values[10] += lambda[2] * 2  # 4,4
        end
        return
    end

    n = 4
    x_L = [1.0, 1.0, 1.0, 1.0]
    x_U = [5.0, 5.0, 5.0, 5.0]

    m = 2
    g_L = [25.0, 40.0]
    g_U = [2.0e19, 40.0]

    prob = Pounce.CreateIpoptProblem(
        n,
        x_L,
        x_U,
        m,
        g_L,
        g_U,
        8,
        10,
        eval_f,
        eval_g,
        eval_grad_f,
        eval_jac_g,
        eval_h,
    )

    prob.x .= [1.0, 5.0, 5.0, 1.0]
    Pounce.AddIpoptIntOption(prob, "print_level", 0)
    return prob
end

@testset "Native statistics and callback removal" begin
    prob = hs071_problem()
    @test Pounce.GetIpoptIterCount(prob) == 0
    @test Pounce.GetIpoptSolveTime(prob) == 0.0
    calls = Ref(0)
    Pounce.SetIntermediateCallback(
        prob,
        function (args...)
            calls[] += 1
            @test_throws ErrorException Pounce.GetIpoptIterCount(prob)
            @test_throws ErrorException Pounce.GetIpoptSolveTime(prob)
            @test_throws ErrorException Pounce.SetIntermediateCallback(prob, nothing)
            return false
        end,
    )
    @test Pounce.IpoptSolve(prob) == 5
    @test calls[] == 1
    @test Pounce.GetIpoptIterCount(prob) == 0
    @test Pounce.GetIpoptSolveTime(prob) >= 0.0
    Pounce.SetIntermediateCallback(prob, nothing)
    prob.x .= [1.0, 5.0, 5.0, 1.0]
    @test Pounce.IpoptSolve(prob) == 0
    @test calls[] == 1
    @test Pounce.GetIpoptIterCount(prob) > 0
    @test isfinite(Pounce.GetIpoptSolveTime(prob))
    Pounce.AddIpoptIntOption(prob, "max_iter", 0)
    prob.x .= [1.0, 5.0, 5.0, 1.0]
    @test Pounce.IpoptSolve(prob) == -1
    @test Pounce.GetIpoptIterCount(prob) == 0
    Pounce.FreeIpoptProblem(prob)
    @test_throws ErrorException Pounce.GetIpoptIterCount(prob)
    @test_throws ErrorException Pounce.GetIpoptSolveTime(prob)
    @test_throws ErrorException Pounce.SetIntermediateCallback(prob, nothing)
end

@testset "C ABI, lifetime, and failure handling" begin
    prob = hs071_problem()
    @test Pounce.IpoptSolve(prob) == 0
    @test prob.obj_val ≈ 17.014017145179164 atol = 1e-6
    @test prod(prob.x) >= 25.0 - 1e-6
    @test sum(abs2, prob.x) ≈ 40.0 atol = 1e-6

    # Force collection while Rust holds the callback's opaque Julia pointer.
    original_f = prob.eval_f
    prob.eval_f = x -> (GC.gc(); original_f(x))
    prob.x .= [1.0, 5.0, 5.0, 1.0]
    @test Pounce.IpoptSolve(prob) == 0

    calls = Ref(0)
    Pounce.SetIntermediateCallback(prob, (args...) -> (calls[] += 1; false))
    @test Pounce.IpoptSolve(prob) == 5
    @test calls[] == 1
    Pounce.SetIntermediateCallback(prob, (args...) -> true)

    for field in (:eval_f, :eval_g, :eval_grad_f, :eval_jac_g, :eval_h, :intermediate)
        original = getfield(prob, field)
        setfield!(prob, field, (args...) -> error("callback sentinel"))
        @test_throws ErrorException("callback sentinel") Pounce.IpoptSolve(prob)
        @test !prob.solving
        @test !prob.solved
        setfield!(prob, field, original)
    end
    prob.eval_f = x -> throw(DomainError(x[1]))
    @test Pounce.IpoptSolve(prob) < 0
    @test prob.callback_error === nothing
    prob.eval_f = original_f
    @test Pounce.IpoptSolve(prob) == 0

    @test_throws DimensionMismatch Pounce.SetIpoptProblemScaling(prob, 1.0, [1.0], C_NULL)
    @test_throws ErrorException Pounce.AddIpoptStrOption(
        prob,
        "hessian_approximation",
        "not-an-option",
    )
    @test_throws ErrorException Pounce.AddIpoptIntOption(prob, "not_an_option", 1)
    Pounce.FreeIpoptProblem(prob)
    @test Pounce.FreeIpoptProblem(prob) === nothing
    @test_throws ErrorException Pounce.IpoptSolve(prob)
end

@testset "C strings reject embedded NULs" begin
    prob = hs071_problem()
    try
        @test_throws ArgumentError Pounce.AddIpoptStrOption(
            prob,
            "hessian_approximation\0ignored",
            "limited-memory",
        )
        @test_throws ArgumentError Pounce.AddIpoptStrOption(
            prob,
            "hessian_approximation",
            "limited-memory\0ignored",
        )
        @test_throws ArgumentError Pounce.AddIpoptNumOption(prob, "tol\0ignored", 1e-8)
        @test_throws ArgumentError Pounce.AddIpoptIntOption(prob, "max_iter\0ignored", 0)
        mktempdir() do dir
            filename = joinpath(dir, "solver.log")
            @test_throws ArgumentError Pounce.OpenIpoptOutputFile(
                prob,
                filename * "\0ignored",
                0,
            )
            @test !isfile(filename)
        end
        # Rejected option calls must not change the native problem.
        @test Pounce.IpoptSolve(prob) == 0
        @test prob.obj_val ≈ 17.014017145179164 atol = 1e-6
    finally
        Pounce.FreeIpoptProblem(prob)
    end
end

@testset "Missing and structurally zero Hessians" begin
    prob = Pounce.CreateIpoptProblem(
        1,
        [-5.0],
        [5.0],
        0,
        Float64[],
        Float64[],
        0,
        0,
        x -> (x[1] - 1)^2,
        (x, g) -> nothing,
        (x, grad) -> (grad[1] = 2 * (x[1] - 1)),
        (args...) -> nothing,
        nothing,
    )
    try
        Pounce.AddIpoptIntOption(prob, "print_level", 0)
        @test_throws ArgumentError Pounce.IpoptSolve(prob) # default exact mode
        for mode in ("limited-memory", "finite-difference", "LIMITED-MEMORY")
            Pounce.AddIpoptStrOption(prob, "hessian_approximation", mode)
            prob.x .= 0.0
            @test Pounce.IpoptSolve(prob) == 0
            @test prob.x[1] ≈ 1.0 atol = 1e-6
        end
        Pounce.AddIpoptStrOption(prob, "HESSIAN_APPROXIMATION", "EXACT")
        @test_throws ArgumentError Pounce.IpoptSolve(prob)
        @test !prob.solving
        @test !prob.solved
    finally
        Pounce.FreeIpoptProblem(prob)
    end
    # A linear objective has a genuinely zero Hessian and is valid in exact mode.
    linear = Pounce.CreateIpoptProblem(
        1,
        [0.0],
        [5.0],
        0,
        Float64[],
        Float64[],
        0,
        0,
        x -> x[1],
        (x, g) -> nothing,
        (x, grad) -> (grad[1] = 1.0),
        (args...) -> nothing,
        (args...) -> nothing,
    )
    try
        Pounce.AddIpoptIntOption(linear, "print_level", 0)
        @test Pounce.IpoptSolve(linear) == 0
        @test linear.x[1] ≈ 0.0 atol = 1e-6
    finally
        Pounce.FreeIpoptProblem(linear)
    end
end
