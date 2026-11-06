# Run in a fresh process with two default worker threads (see runtests.jl).
using Test
import Pounce

function threaded_problem()
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
    Pounce.AddIpoptIntOption(prob, "print_level", 0)
    Pounce.AddIpoptStrOption(prob, "hessian_approximation", "limited-memory")
    return prob
end

function current_x(prob)
    x = zeros(1)
    Pounce.GetIpoptCurrentIterate(prob, false, 1, x, C_NULL, C_NULL, 0, C_NULL, C_NULL)
    return x
end

@testset "Thread-local native state" begin
    @test Threads.nthreads(:default) >= 2
    fetch(
        Threads.@spawn begin
            @test !current_task().sticky
            prob = threaded_problem()
            other = threaded_problem()
            try
                tid = Threads.threadid()
                Pounce.SetIntermediateCallback(
                    prob,
                    function (args...)
                        @test current_task().sticky
                        @test Threads.threadid() == tid
                        before = current_x(prob)
                        for _ = 1:10
                            yield()
                            sleep(0.001)
                            @test Threads.threadid() == tid
                        end
                        @test current_x(prob) == before
                        # A suspended callback must not allow another task to overwrite
                        # the same thread's Rust TLS. This task is deliberately sticky.
                        task = @async try
                            Pounce.IpoptSolve(other)
                        catch err
                            err
                        end
                        err = fetch(task)
                        @test err isa ErrorException
                        @test occursin(
                            "Another task has an active POUNCE solve",
                            sprint(showerror, err),
                        )
                        @test !other.solving
                        @test current_x(prob) == before
                        return false
                    end,
                )
                @test Pounce.IpoptSolve(prob) == 5
                @test !current_task().sticky
                Pounce.SetIntermediateCallback(prob, function (args...)
                    yield()
                    error("thread callback sentinel")
                end)
                @test_throws ErrorException("thread callback sentinel") Pounce.IpoptSolve(
                    prob,
                )
                @test !current_task().sticky
                @test !prob.solving
                # The failed solve must release thread ownership.
                Pounce.SetIntermediateCallback(prob, nothing)
                @test Pounce.IpoptSolve(prob) == 0
                @test !current_task().sticky
            finally
                Pounce.FreeIpoptProblem(prob)
                Pounce.FreeIpoptProblem(other)
            end
        end
    )

    # Independent solves can overlap on separate threads, including callbacks
    # that wait for each other and query distinct native iterate snapshots.
    arrived = [Channel{Nothing}(1), Channel{Nothing}(1)]
    Threads.@threads :static for i = 1:2
        prob = threaded_problem()
        try
            prob.x[1] = i == 1 ? -2.0 : 2.0
            Pounce.SetIntermediateCallback(
                prob,
                function (args...)
                    before = current_x(prob)
                    put!(arrived[i], nothing)
                    @test timedwait(() -> isready(arrived[3-i]), 10) == :ok
                    @test current_x(prob) == before
                    @test before[1] ≈ (i == 1 ? -2.0 : 2.0)
                    return false
                end,
            )
            @test Pounce.IpoptSolve(prob) == 5
        finally
            Pounce.FreeIpoptProblem(prob)
        end
    end
end

@testset "Finalization on a different thread" begin
    for solved in (false, true)
        holder = Ref{Pounce.IpoptProblem}()
        creator = Ref(0)
        Threads.@threads :static for i = 1:2
            if i == 1
                prob = threaded_problem()
                if solved
                    @test Pounce.IpoptSolve(prob) == 0
                end
                holder[] = prob
                creator[] = Threads.threadid()
            end
        end
        Threads.@threads :static for i = 1:2
            if i == 2
                @test Threads.threadid() != creator[]
                prob = holder[]
                # Invoke the installed finalizer deterministically on this
                # worker, without depending on GC scheduling to choose it.
                finalize(prob)
                @test prob.ipopt_problem == C_NULL
                @test !prob.solved
                @test Pounce.FreeIpoptProblem(prob) === nothing
            end
        end
    end
end
