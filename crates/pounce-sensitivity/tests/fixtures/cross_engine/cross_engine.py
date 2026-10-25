"""Write cross_engine.nl and cross_engine_scaled.nl, the models
tests/cross_engine_ipopt.rs hands to both pounce and Ipopt.

    python cross_engine.py

needs Pyomo. The solution has one variable at an active bound (x1 at its upper
bound), one active inequality row (cap), one inequality row bounded on both
sides that it sits strictly inside (band), and a parameter p held by an
equality row (pin) that a perturbation moves. Pyomo writes the nonlinear rows
first, so the row order in the .nl is band, pin, cap.

cross_engine_scaled.nl is the same model with the objective multiplied by
1000. Its gradient at the starting point is near 2950, above Ipopt's default
nlp_scaling_max_gradient of 100, so Ipopt scales the objective by about 0.034
and its bound multipliers with it.

The .sol and the excerpt of Ipopt's output beside each .nl come from

    ipopt <name>.nl -AMPL linear_solver=ma57 bound_relax_factor=0 print_level=8
"""
import pyomo.environ as pyo


def write(name, weight):
    m = model(weight)
    m.write(name, format="nl", io_options={"symbolic_solver_labels": True})


def model(weight):
    m = pyo.ConcreteModel()
    m.x1 = pyo.Var(bounds=(0.0, 1.5), initialize=0.5)
    m.x2 = pyo.Var(bounds=(0.0, 10.0), initialize=0.5)
    m.x3 = pyo.Var(bounds=(-5.0, 5.0), initialize=0.0)
    m.p = pyo.Var(initialize=1.0)
    m.pin = pyo.Constraint(expr=m.p == 1.0)
    m.cap = pyo.Constraint(expr=m.x1 + m.x2 <= 2.4 + 0.5 * m.p)
    m.band = pyo.Constraint(expr=pyo.inequality(-1.0, m.x1 * m.x2 - m.x3, 4.0))
    f = (
        (m.x1 - 2.0 * m.p) ** 2 + (m.x2 - 1.5) ** 2 + 0.1 * m.x1 * m.x2
        + (m.x3 - 0.3 * m.p) ** 2 + 0.05 * pyo.exp(0.5 * m.x3)
    )
    m.obj = pyo.Objective(expr=f if weight == 1.0 else weight * f)
    return m


write("cross_engine.nl", 1.0)
write("cross_engine_scaled.nl", 1000.0)
