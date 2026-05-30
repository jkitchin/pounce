# POUNCE on embedded devices — market and competitive landscape

*Investigation snapshot, 2026-05-30. Point-in-time research, not authoritative
documentation. Captures a sourced assessment of whether "embedded / on-device
control" is a viable direction for pounce, and whether "native Rust /
memory-safe / no-FFI" is a real procurement driver.*

## Question

Is there a market for running pounce (a faithful pure-Rust port of Ipopt — a
general nonlinear-constrained interior-point NLP solver) on embedded devices /
microcontrollers for control (e.g. MPC/NMPC)? Is "native Rust / memory-safe /
no-FFI" a feature industry actually buys on, or a nice-to-have?

## Bottom line

- **Memory-safety / native-Rust is *not* a primary procurement driver for an
  embedded numerical solver.** Buyers select on worst-case execution time
  (WCET) / determinism, code-generation to fixed-size allocation-free C,
  memory footprint, warm-starting, and safety-standard packaging (MISRA-C) —
  not language.
- **Architecturally, an Ipopt port is the *offline* tool, not the in-the-loop
  one.** The real dividing line in embedded MPC is *structure-exploiting +
  warm-startable + code-generated*, not "IPM vs. not." Being in Rust does not
  move pounce across that line; the line is algorithmic.
- **The defensible pounce story is integration, not control:** "the first
  general nonlinear-constrained NLP solver native to Rust — no FFI, fits your
  Cargo workspace," for the Rust-first segment, deployed on the **edge /
  embedded-Linux / Cortex-A / RISC-V application-core tier** (FPU + allocator),
  *not* the bare-metal MCU tier that OSQP/TinyMPC own.

## 1. What industry buys an embedded solver on

Consistent across sources — language is not on the list:

- **WCET / determinism over average speed.** For real-time MPC the worst-case
  execution time, not the average, must be provably below the sampling period,
  ideally from analysis rather than empirical timing.
  (arxiv.org/abs/2304.11576, arxiv.org/pdf/2306.15079)
- **Code-gen to fixed-size, allocation-free, library-free C.** The headline
  feature for OSQP, acados, FORCESPRO, TinyMPC. (osqp.org,
  arxiv.org/abs/2310.16985)
- **Memory footprint, warm-starting, field/deployment history.**
- **Safety/coding-standard evidence (MISRA-C, ISO 26262).** MISRA-C is the
  concrete differentiator (FORCESPRO advertises a MISRA-compliant codegen
  option, limited to its interior-point algorithms). No public, independently
  verifiable ISO 26262 / ASIL certification of any of these *solvers
  themselves* was found — thin evidence. (forces.embotech.com/Documentation)

## 2. The competitive landscape

| Solver | Class | Niche | License |
|---|---|---|---|
| **OSQP** | convex QP (ADMM) | default embeddable QP, code-gen | Apache-2.0 |
| **acados** | NLP/NMPC (SQP-RTI) | the standard for embedded NMPC | BSD-2 |
| **HPIPM + BLASFEO** | structured IPM QP | OCP-structured, often *fastest* | BSD-2 |
| **qpOASES** | active-set QP | classic embedded QP, warm-start | LGPL-2.1 |
| **FORCESPRO** (Embotech) | QP + NLP (PDIP/SQP) | commercial code-gen, automotive | commercial |
| **TinyMPC** | convex MPC (ADMM) | microcontroller-class, KB footprint | open (MIT-style) |
| **Clarabel** | convex conic IPM | modern, **Rust**, CVXPY default | Apache-2.0 |
| **Ipopt** | general nonconvex NLP | the offline NLP workhorse | EPL |

### Important correction to the "IPM is dead for embedded" narrative

That framing is **too strong, and the nuance cuts against pounce specifically:**

- **HPIPM is an interior-point method and is frequently the *fastest* embedded
  MPC solver** because it exploits OCP block structure; FORCESPRO's flagship
  embedded algorithm is *also* interior-point.
  (publications.syscop.de/Frison2020a.pdf, arxiv.org/pdf/2502.01329)
- The real dividing line is **structure-exploiting + warm-startable +
  code-generated**, not "IPM vs. not."
- What is genuinely unsuitable in-the-loop is the **general-purpose sparse NLP
  IPM with dynamic factorization — i.e. Ipopt-style.** Ipopt is explicitly an
  offline/desktop tool. (coin-or.github.io/Ipopt)

pounce, as a faithful Ipopt port, sits architecturally in the *offline*
category. Classic IPMs also largely destroy warm-start benefit (central-path
following + fresh KKT factorization per iteration), which is precisely what
embedded MPC needs and what active-set/ADMM/SQP-RTI methods provide.

## 3. Does memory safety move procurement for *numerical* code?

The crux, and the evidence is clear: **memory-safety value concentrates on
code handling untrusted input (parsers, network, OS) — not trusted-input
numerical kernels.**

- Kelly Shortridge's "SUX rule" (from Chromium's Rule of Two): the danger is
  sandbox-free + unsafe-language + e**X**ogenous (untrusted) input, with the
  explicit corollary that code *not* processing untrusted input may be
  acceptable in C. A solver eating trusted `f64` arrays is the textbook
  weakest case. (kellyshortridge.com/blog/posts/the-sux-rule-for-safer-code/)
- The US government push (ONCD "Back to the Building Blocks", CISA memory-safe
  roadmaps) is **recommendation, not mandate**, aimed at internet-facing /
  systems code, and **explicitly hedges that memory-safe languages are not yet
  proven for embedded / real-time / space**. It does not move embedded-controls
  procurement. (bidenwhitehouse.archives.gov ONCD report,
  cisa.gov/resources-tools/resources/case-memory-safe-roadmaps)
- The *one* non-security argument for numerical code is **UB-elimination and
  certification / defect-rate** (Volvo's self-reported ~0.25 bugs/KLOC is a
  defect-rate claim, not security) — a reliability argument that incumbents
  partly neutralize with field history.

## 4. Where the Rust angle *is* real

- **The native-Rust NLP gap genuinely exists.** Clarabel is Rust but
  convex-conic only (now the CVXPY default — real shipped adoption); argmin is
  unconstrained; OpEn (`optimization_engine`) is restricted-form embedded
  nonconvex (PANOC + projectable sets, not arbitrary nonlinear inequalities);
  **every Ipopt-class capability in Rust today is an FFI binding**
  (`ipopt-sys`, etc.). No mature native-Rust general constrained NLP solver
  exists — pounce would be first. (github.com/cvxpy/cvxpy/discussions/2178,
  github.com/elrnv/ipopt-rs, github.com/alphaville/optimization-engine)
- **"No-FFI / one Cargo build" is a genuine, *stated* value** — but in the
  robotics tooling ecosystem (r2r markets exactly this: "`cargo build` is all
  you need," no bindgen/colcon). It is an **integration-ergonomics** argument,
  not a safety one. (github.com/sequenceplanner/r2r)

### Where it is *not* real (yet)

- **Ferrocene qualifies the *compiler*, not your application** — ISO 26262
  ASIL-D, IEC 61508 SIL-3, IEC 62304 Class C, but **not DO-178C** (aerospace);
  named adopters are niche pilots (Sonair, mining), not OEM fleets.
  (ferrous-systems.com/blog/officially-qualified-ferrocene/)
- **Shipped production Rust control code is rare:** Volvo/Polestar 3 is the one
  credible example, and it is a *low-criticality* ECU, not a braking/steering
  controller; Renault/Ampere is announced-for-2026; Bosch/Continental show no
  shipped evidence. (tweedegolf.nl/en/blog/137/rust-is-rolling-off-the-volvo-assembly-line)
- **No company has publicly stated "native Rust" as a *solver* procurement
  reason** — that inference is thin.

## 5. Market size

There is **no clean TAM for "embedded MPC solvers"** — it is a niche inside
Advanced Process Control (~$2.7B in 2025, ~10% CAGR, analyst-grade /
directional). Commercial value concentrates in **codegen + safety packaging +
support** (Embotech FORCESPRO; ODYS claims 3M+ vehicles — vendor-asserted,
NDA-gated, unverifiable). The *application* activity is large and growing; the
*solver-licensing* market is small and contested.
(grandviewresearch.com/industry-analysis/advanced-process-control-apc-market,
embotech.com/forcespro, odys.it/embedded-mpc)

## 6. Technical readiness (from a read of the workspace)

Snapshot of how far the code is from an on-device build, separate from the
market question:

- **TNLP trait is fully slice-based** (`&[Number]` / `&mut [Number]`, caller
  owns all buffers) — exactly the right shape for an embedded interface; no
  allocation in the interface itself.
- **feral** (the LDLᵀ numerical core) has zero runtime dependencies and uses
  only `core`/`alloc`-compatible namespaces plus `BTreeMap`/`BTreeSet`. No
  `io`/`fs`/`thread`/`Instant`. Effectively `no_std + alloc`-clean already.
- Heavy `std` surface (`fs`, `io`, `print!`, `Instant`, `env`) is concentrated
  in `pounce-algorithm`'s logging / diagnostics / iter-dump paths and in the
  CLI — feature-gateable or out of scope for a device build.
- Real obstacles are not portability but: **alloc-heavy** (`Vec`/`Box`/`Rc`/
  `RefCell` throughout — the ported Ipopt `SmartPtr` pattern; needs a heap and,
  for deterministic control, a preallocated-workspace path); **f64
  everywhere** (fine on M7/M55/M85 or RISC-V w/ D-ext, painful soft-float on
  M0–M4); **libm** (hundreds of `.sqrt()/.exp()/...` sites need routing in
  `no_std`); and the IPM's **variable iteration count + dynamic sparse
  factorization → no natural WCET guarantee**.
- Available cross targets in the toolchain include `thumbv7em-none-eabihf`,
  `riscv32imac-unknown-none-elf`, `thumbv8m.*` — a `no_std` feasibility spike
  is buildable if desired.

## Strategic takeaway

What industry would say to "it's memory-safe native Rust": *"That's nice — is
it deterministic, does it give me bounded WCET, does it code-gen alloc-free,
and does it exploit my problem's structure?"* On those, an Ipopt port is behind
by architecture, and Rust does not change that.

- The honest pitch is **"the general NLP solver native to the Rust ecosystem"**
  — competing on *integration / no-FFI*, for the Rust-first segment (small,
  fastest-growing, longest tailwind).
- Natural deployment tier: **edge / embedded-Linux / Cortex-A / RISC-V
  application cores** with FPU + allocator — where general NLP and
  offline/advanced-step NMPC live — not bare-metal MCUs.
- Memory-safety-as-certification is a **long-dated option**, not a present-day
  selling point.
- If pounce wants in-the-loop embedded relevance, the lever is **algorithmic**
  — structure-exploitation, warm-starting, fixed-size/alloc-free path. This is
  why the already-ported sIPOPT / advanced-step sensitivity machinery
  (`pounce-sensitivity`) is the most strategically interesting asset: "solve
  once offline, cheap online sensitivity updates on-device" is the established
  way to make an IPM viable for fast NMPC, and most embedded solvers do not
  ship sensitivity.

## Confidence / caveats

- Load-bearing conclusions (memory-safety concentrates on untrusted input;
  structure-exploitation is the real embedded line, not IPM-vs-not; the
  native-Rust NLP gap is real) are each corroborated across multiple
  independent sources.
- The Ferrocene / Volvo / US-government specifics rest on search-summary
  sources (full-text fetch was blocked during research) — slightly lower
  confidence than the solver-landscape and ecosystem findings.
- All vendor production-volume claims (Embotech, ODYS) are vendor-asserted and
  NDA-gated — treat as unverified.
- Market-size figures are analyst-grade / directional, not precise; no
  dedicated "embedded MPC solver" market report exists.

## Key sources

- WCET/determinism: arxiv.org/abs/2304.11576, arxiv.org/pdf/2306.15079
- Solvers: osqp.org, github.com/acados/acados,
  publications.syscop.de/Frison2020a.pdf (HPIPM), arxiv.org/abs/2310.16985
  (TinyMPC), github.com/oxfordcontrol/Clarabel.rs, coin-or.github.io/Ipopt
- Rust NLP gap: github.com/cvxpy/cvxpy/discussions/2178,
  github.com/elrnv/ipopt-rs, github.com/alphaville/optimization-engine,
  github.com/sequenceplanner/r2r
- Memory safety: kellyshortridge.com/blog/posts/the-sux-rule-for-safer-code/,
  cisa.gov/resources-tools/resources/case-memory-safe-roadmaps,
  bidenwhitehouse.archives.gov ONCD "Back to the Building Blocks" (Feb 2024)
- Rust in automotive: ferrous-systems.com/blog/officially-qualified-ferrocene/,
  tweedegolf.nl/en/blog/137/rust-is-rolling-off-the-volvo-assembly-line
- Market size: grandviewresearch.com/industry-analysis/advanced-process-control-apc-market,
  embotech.com/forcespro, odys.it/embedded-mpc
