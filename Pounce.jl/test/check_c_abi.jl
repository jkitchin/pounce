# Run separately from Pkg.test(): this check needs a C compiler and the headers
# belonging to the selected native build. CI supplies both for every native leg.
using Test
import Pounce

include_dir = abspath(only(ARGS))
stats = Pounce._CLinearSolverStats
expressions = [
    "sizeof(ipindex)",
    "sizeof(ipnumber)",
    "sizeof(Bool)",
    "sizeof(PounceLinearSolverStats)",
    ["offsetof(PounceLinearSolverStats, $field)" for field in fieldnames(stats)]...,
]
expected = [
    sizeof(Cint),
    sizeof(Cdouble),
    sizeof(Bool),
    sizeof(stats),
    [fieldoffset(stats, i) for i in 1:fieldcount(stats)]...,
]
mktempdir() do dir
    source = joinpath(dir, "layout.c")
    executable = joinpath(dir, "layout")
    open(source, "w") do io
        println(io, "#include <stddef.h>\n#include <stdio.h>\n#include \"pounce.h\"")
        println(io, "int main(void) {")
        for expression in expressions
            println(io, "    printf(\"%zu\\n\", $expression);")
        end
        println(io, "    return 0;\n}")
    end
    compiler = Cmd(Base.shell_split(get(ENV, "CC", "cc")))
    run(`$compiler -std=c99 -Wall -Wextra -Werror -pedantic -I$include_dir $source -o $executable`)
    actual = parse.(Int, split(read(`$executable`, String)))
    @testset "C ABI layout (native $(Pounce.GetIpoptVersion()))" begin
        @test length(actual) == length(expected)
        for (expression, c_value, julia_value) in zip(expressions, actual, expected)
            @testset "$expression" begin
                @test c_value == julia_value
            end
        end
    end
end
