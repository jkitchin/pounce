# LinkedIn post — pounce 0.4.0

> Draft. Edit freely. The `---` rules mark the start/end of the post body
> you'd paste into LinkedIn; everything outside them is notes.

---

🚀 pounce 0.4.0 is out — a pure-Rust interior-point NLP solver, now with a debugger for your optimization problems.

When a nonlinear solver stalls, most tools give you a wall of iteration logs and a shrug. pounce 0.4.0 ships something different: an **interactive solver debugger**.

Start a run with `--debug` and you can:

- Break into a live solve — Ctrl-C pauses at the next iteration instead of killing the run
- Inspect the iterate — primals, duals, KKT residuals, the barrier parameter, inertia
- Probe the problem — `sweep` a variable, `multistart` from jittered points, `load` a saved iterate and step forward
- Drive it from an LLM — the same diagnostics are exposed over MCP, so you can ask Claude *why* a model isn't converging instead of decoding it yourself

Plus: signed solve receipts (`pounce verify`), sparse colored AD for the JAX front-ends, and `curve_fit` in Python.

Pure Rust. No Fortran or C.

```
pip install pounce-solver        # core solver + Python API
pip install pyomo-pounce         # Pyomo plugin
```

📦 Docs: https://kitchingroup.cheme.cmu.edu/pounce/
🐙 Source: https://github.com/jkitchin/pounce

#Rust #Optimization #NonlinearProgramming 


