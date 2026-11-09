---
title: 'POUNCE: a pure-Rust interior-point optimizer for linear, quadratic, nonlinear and complementarity-constrained problems'
tags:
  - Rust
  - Python
  - nonlinear programming
  - interior-point methods
  - mathematical programs with complementarity constraints
  - AI-assisted software engineering
authors:
  # TODO(co-authors): confirm consent, name spelling, ORCID and affiliation.
  # Germán Heim first, John Kitchin last; the middle is alphabetical by surname.
  - name: Germán Heim
    orcid: 0000-0000-0000-0000
    affiliation: 1
  - name: David E. Bernal Neira
    orcid: 0000-0002-8308-5016
    affiliation: 2
  - name: Johannes P. Dürholt
    orcid: 0000-0002-3494-1621
    affiliation: 3
  - name: Frederik Geth
    orcid: 0000-0002-9345-2959
    affiliation: 4
  - name: Devin Griffith
    orcid: 0009-0008-8619-0971
    affiliation: 5
  - name: John R. Kitchin
    orcid: 0000-0003-2625-9232
    corresponding: true
    affiliation: 6
affiliations:
  - name: Universidad Nacional del Sur, Bahía Blanca, Argentina
    index: 1
  - name: Davidson School of Chemical Engineering, Purdue University, West Lafayette, IN, United States
    index: 2
  - name: Evonik Operations GmbH, Germany
    index: 3
  - name: School of Electrical Engineering and Computer Science, The University of Queensland, Brisbane, Australia
    index: 4
  - name: Griffith Decision Science, Richmond, TX, United States
    index: 5
  - name: Department of Chemical Engineering, Carnegie Mellon University, Pittsburgh, PA, United States
    index: 6
date: 7 November 2026
bibliography: paper.bib
---

# Summary

Many problems in engineering and science are optimization problems: choose the operating conditions of a chemical plant, fit a model to data, or set the power flows on a grid, subject to physical laws written as equations and limits written as constraints. POUNCE solves these problems. Its core is a port of Ipopt [@Wachter2006], the interior-point solver most widely used for large, sparse, smooth nonlinear programs. Around that core it adds engines for linear, convex quadratic and conic programs, active-set QP and SQP, parametric sensitivity [@Pirnay2012], multistart searches for several local minima, and problems with complementarity constraints (MPCCs), in which one of two quantities must be zero at every pair.

POUNCE is written entirely in Rust, including its default sparse symmetric-indefinite linear solver, FERAL. It has no Fortran, BLAS or external linear algebra dependency, so it builds from `cargo`, installs from `pip` on Linux, macOS and Windows, and runs in a web browser through WebAssembly and Pyodide. It reads AMPL `.nl` files and is available from Python (a `scipy.optimize.minimize`-compatible API and a `Problem` API), Pyomo [@Bynum2021], CasADi [@Andersson2019], Julia through JuMP and MathOptInterface [@Lubin2023; @Legat2022], GAMS, a C API, and a container image. The source is at <https://github.com/jkitchin/pounce> under the Eclipse Public License 2.0, and the documentation is at <https://jkitchin.github.io/pounce/>.

POUNCE is also designed to be operated by AI agents, not only by people. Its interior-point loop has an interactive debugger with a self-describing JSON protocol. An MCP server and an agent skill diagnose solves, and an independent checker verifies a claimed solution against the model.

POUNCE was built AI-first. We set the scope, the architecture and the standards of evidence. Most of the code, tests and documentation were written by an AI coding agent working under those standards and our guidance. We describe that process, and how its output was checked, in **AI usage disclosure**.

# Statement of need

Ipopt is a standard local NLP solver in most algebraic modeling systems. Using it well has a practical cost. Good performance needs a licensed HSL linear solver [@Duff2004], and the Fortran and C++ toolchain makes it hard to ship in a Python wheel or a browser and is a challenge for new users. Teaching, web demonstrations, and research codes that people install with a single `pip install` all run into this cost. POUNCE removes all those issues. It keeps Ipopt's algorithm, option names and console output, so existing models, option files and experience carry over practically unchanged.

The second need is for solver features that are usually separate programs. POUNCE routes each model to the most specialized engine it supports. A convex QP goes to a homogeneous self-dual interior point [@Ye1994], a general NLP to the Ipopt port, and a model with declared complementarity pairs to a driver that runs Scholtes regularization [@Scholtes2001] and an exact product formulation, then certifies the answer as strongly or weakly stationary. Sensitivity of the solution to parameters, a multistart search for distinct local minima, and a machine-readable JSON solve report are built in, not add-ons.

The third need is new. Users increasingly ask an AI agent to formulate, solve and debug their models. A conventional solver gives that agent a log to read and a status to trust. An agent needs an interface it can drive, a structured record of what the solver did, and a way to check an answer without trusting the solver that produced it. POUNCE provides these.

The intended users are researchers and students in process systems engineering, power systems and optimal control. They write models in Pyomo, CasADi, JuMP or AMPL and need a solver that installs anywhere and explains its answers.

# State of the field

Ipopt [@Wachter2006] is the reference implementation that POUNCE follows. MadNLP [@Shin2024] is a Julia interior-point solver that targets GPUs and condensed-space methods. Uno [@Vanaret2026] is a composable C++ framework that can reproduce several existing solver designs. Clarabel [@Goulart2024] is a Rust conic solver for convex problems only. None of these is a dependency-free, drop-in replacement for Ipopt that can also be shipped to WebAssembly.

The choice between building and contributing was made in favor of a port. The goal was Ipopt's behavior without its toolchain. That cannot be added to Ipopt itself, and it is a different goal from MadNLP's or Uno's. The port follows upstream closely enough to compare iterate by iterate. Where POUNCE departs from upstream, it does so behind an option whose documented value restores upstream behavior where possible. 

# Software design

POUNCE is a Cargo workspace of 25 crates. They separate the parts a reviewer would want to check on their own: the NLP algorithm (`pounce-algorithm`, `pounce-nlp`, `pounce-restoration`), linear algebra and linear solvers (`pounce-linalg`, `pounce-linsol`, `pounce-feral`, optional `pounce-hsl`), the convex and active-set engines (`pounce-convex`, `pounce-qp`), presolve, sensitivity, multistart, the `.nl` reader, and the front ends (CLI, Python, C, WebAssembly).

Four design decisions follow from the research setting.

1. **Upstream parity is the default.** We generally keep Ipopt's data flow and
   option semantics. A fix that changes the trajectory is measured against
   upstream and against the previous release before it merges.
2. **One entry point, routing to many engines.** At the default
   `solver_selection=auto`, POUNCE classifies each model and sends it to the
   most specialized engine available. Every report records which engine
   answered, so a routing change shows up in the record even when the status
   and objective do not move.
3. **Answers carry their own evidence.** Each solve can emit a versioned JSON
   report. It contains residuals, the iterations spent in every re-solve and
   fallback, and, for complementarity models, the stationarity certificate
   and how complementary the returned point is. A companion tool,
   `pounce-studio`, diagnoses a solve from that report.
4. **Agents are first-class users.** Every capability a person reaches
   through the console is also exposed in a form an agent can discover
   and drive without reading the source.

The fourth decision shapes several components. The **interactive debugger** is a debugger for the interior-point loop, in the spirit of `pdb`. It pauses a solve at named checkpoints inside an iteration. Breakpoints can be set by iteration, on a numeric condition such as `inf_pr<1e-6`, or on a solver event such as entering restoration. At a pause it can inspect and *modify* the iterate, the multipliers and the barrier parameter. It can report KKT inertia and the numerical rank of the equality Jacobian, naming the dependent constraints. It can rewind to an earlier iterate and re-solve from a saved point with new options. The same command engine serves two front ends: a REPL for people (`--debug`, or `--debug-on-error` to stop only when a solve fails) and a newline-delimited JSON protocol (`--debug-json`). The protocol's first message is a handshake that lists every command, checkpoint, event, metric and capability. An agent can therefore feature-detect and drive a live solve with no documentation. An `ask` command packages the paused state into a prompt for an LLM, or returns the prompt to an agent that is itself the model. Attaching the debugger does not change the solve: the debugged and plain runs follow the same trajectory, and the checkpoints cost nothing when it is detached.

Around the debugger, POUNCE ships the tools an agent needs to use a solver well:

- the `pounce-studio-mcp` server, a Model Context Protocol server. It
  offers post-mortem tools over a finished solve report (diagnose, find
  stalls, compare runs) and a stateful proxy for live debug sessions.
- a matching agent skill that installs with `make install-skill` and
  drives the `pounce-studio` command-line tool. It carries recipes for
  convergence diagnosis, starting-point pre-flight (`pounce check-x0`),
  warm starts and complementarity models.
- `pounce verify`, which re-derives feasibility of a claimed solution
  from the `.nl` model itself, without rerunning the solver or trusting
  the `.sol` status line. Checking a solution is far cheaper than
  producing it. That asymmetry lets a small deterministic checker serve
  as the trust anchor when an agent proposes an answer.

# Research impact statement

POUNCE has been public since May 2026 and has had 15 releases. `pounce-solver` has been downloaded about 42,600 times from PyPI since its first release on 25 May 2026, excluding mirrors (about 55,000 including them). `pyomo-pounce` has been downloaded about 14,500 times. Hans Mittelmann's public benchmark of AMPL NLP solvers includes POUNCE [@MittelmannNLP].

POUNCE is in research use outside its authors' group:

- Opyrability, a process-operability package [@Alves2024], uses POUNCE as its
  default solver.
- SiNDAE, which trains neural differential-algebraic equations by the
  simultaneous approach [@Lueg2026], runs its whole pipeline on POUNCE.
- oximo, a Rust algebraic modeling library, has a POUNCE backend, and
  tellegen runs live interactive power-flow analysis on it.
- The Julia interface was contributed by power-systems researchers and is
  validated against an optimal power flow adapter.

Eleven users besides the lead developer have filed 70 issues, six of them from outside the author list, and five contributors have opened 58 pull requests.

On the release benchmark (1,345 problems from the CUTE/Vanderbei AMPL collection, Mittelmann's NLP set, Maros–Mészáros QPs [@Maros1999], LP collections, and application suites), POUNCE solves 1,309 to strict optimality, against 1,260 for Ipopt 3.14 with MA57 under the same 300 s limit. On MacMPEC, it reaches the best known objective on 158 of 178 problems and leaves none unsolved. Performance profiles [@Dolan2002] and the full per-problem tables are in the repository.

# AI usage disclosure

Development used Anthropic’s Claude through Claude Code (models Opus 4.7, 4.8, 5 and 5.5, Fable 5 and 5.1, Sonnet 5 and 5.5; 2,712 of 3,711 commits), with smaller contributions from OpenAI Codex and GitHub Copilot. AI-written work includes the Rust port, tests, benchmark harnesses, documentation, notebooks and a first draft of this paper. We decided what to build, chose the architecture (a faithful port first, extensions behind options, engine routing, the evidence-carrying report) and set how correctness is shown, and when the work was ready to be merged. 

AI-written code is checked against evidence independent of the tests written alongside it. The strongest check is a fixture sweep. Any change that can alter the solver's path is run on two Hessian modes and two CPU architectures, and every line that moves must be explained before merge. Every release runs the benchmark suites described above and is compared against upstream Ipopt and published best-known objectives. An adversarial test harness poses problems with known solutions to every engine and reports wrong answers.

The authors take full responsibility for the software and this paper.

# Acknowledgements

POUNCE's NLP core is a port of Ipopt, by Andreas Wächter, Lorenz T. Biegler and the COIN-OR community. 

 <!-- TODO: funding sources for each author. -->

# References
