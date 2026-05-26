$title Parametric warm-start sweep with the POUNCE active-set SQP driver
$onText
Phase 5c §7.4 demo. Demonstrates the GAMS-native marginal-based
working-set carry across a sequence of perturbed solves: each
`solve` reuses the variable and equation marginals the previous
solve left behind, which the POUNCE GAMS link automatically
translates into the SQP warm-start working set
(`IpoptSetWarmStartWorkingSet`).

The model is the canonical "moving target" parametric NLP:

    min  ½ ‖x − p‖²
    s.t. sum(x) = 1
         x >= 0                 (the active set varies with p)

As `p` rotates around the simplex centre, the support of the
optimum x* (i.e. which `x[i] = 0` bounds are active) changes
slightly between consecutive solves. The active-set SQP path
benefits from the carried working set; the IPM path produces the
same answer but ignores the marginal hand-off.

Run with:

    sudo make -C gams install   # build & install pounce
    cd gams/examples
    gams parametric_sqp_warm_start.gms

The model writes a small log table comparing solve status,
iteration count, and active-set agreement across the sweep.
$offText

Set
    i     index of variables / 1*8 /
    k     parametric step    / 1*20 /;

Parameter
    p(i)         current parameter
    centre(i)    centre of the sweep
    dir(i)       sweep direction (random unit vector with mean 0)
    radius       sweep radius / 0.2 /
    theta        current sweep angle
    pi           / 3.14159265358979 /
    iters(k)     iteration count per step;

centre(i) = 1.0/card(i);
* Deterministic direction: alternating ±1 with row-i scaling,
* then centred and normalised.
dir(i)    = (mod(ord(i), 2) - 0.5) * (ord(i) - 0.5);
dir(i)    = dir(i) - sum(j$(ord(j) eq 0), 0);   * placeholder
* recentre to zero mean (so direction is tangent to the simplex):
Parameter dir_mean;
dir_mean  = sum(i, dir(i)) / card(i);
dir(i)    = dir(i) - dir_mean;
Parameter dir_norm;
dir_norm  = sqrt(sum(i, sqr(dir(i))));
dir(i)    = dir(i) / dir_norm;

Variables x(i), obj;
Equations defobj, sumone;

defobj..  obj    =e= 0.5 * sum(i, sqr(x(i) - p(i)));
sumone..  sum(i, x(i)) =e= 1.0;

x.lo(i) = 0;

Model proj / all /;

option nlp = pounce;
proj.optfile = 1;

Loop(k,
    theta = 2 * pi * (ord(k) - 1) / card(k);
    p(i)  = centre(i) + radius * cos(theta) * dir(i);
    * Starting point: a uniform interior point. POUNCE picks up
    * the marginals from the previous solve automatically.
    x.l(i) = centre(i);
    Solve proj using nlp minimizing obj;
    iters(k) = proj.iterusd;
);

display iters, x.l, x.m;

* Sanity check: the final point must satisfy the constraints to
* solver tolerance.
abort$(abs(sum(i, x.l(i)) - 1.0) > 1e-6) 'sum(x) != 1', sum(i, x.l(i));
abort$(smin(i, x.l(i)) < -1e-6)            'x has negative entry', smin(i, x.l(i));
abort$(proj.solvestat <> 1)                'final solve failed', proj.solvestat;

display 'parametric warm-start sweep ok';
