# pounce-minima

The multistart / find-minima global search of
[POUNCE](https://github.com/jkitchin/pounce). `find_minima` drives a local
NLP solver from many starts — six strategies: multistart, MLSL,
basin-hopping, flooding, deflation, tunneling — and returns the distinct
local minima it found, deduplicated and ranked, each with the base problem's
constraint duals. The local solves go through the `LocalSolve` trait;
`IpoptLocalSolve` runs them on a reused `IpoptApplication`.

It is the search behind `pounce model.nl --minima <method>`, and the
algorithm of Python's `pounce.find_minima`. Most Rust users reach it through
the facade, `pounce-rs` with `features = ["minima"]`, as
`pounce_rs::minima`. See the
[find-minima documentation](https://jkitchin.github.io/pounce/find-minima-cli.html).

Licensed under EPL-2.0.
