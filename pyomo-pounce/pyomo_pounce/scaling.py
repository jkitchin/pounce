"""User NLP scaling from Pyomo's `scaling_factor` Suffix (gh #483).

The Pyomo/Ipopt workflow for hand-supplied scaling is::

    model.scaling_factor = Suffix(direction=Suffix.EXPORT)
    model.scaling_factor[model.obj] = 1e-3
    model.scaling_factor[model.mass_balance] = 1e2
    solver.options['nlp_scaling_method'] = 'user-scaling'

Through ASL the writer emits those entries as `.nl` `S4`/`S5`/`S6`
suffix segments and the solver reads them back. pyomo-pounce contained
no scaling code at all, so the option was accepted and meant "none": a
badly conditioned model kept converging badly with nothing to say the
suffix had been ignored.

Three things happen here now:

* **Objective and constraint factors are honored.** The subprocess
  (ASL) path needs nothing from this module — the writer emits the
  suffix segments and `NlTnlp::get_scaling_parameters` now reads them.
  The in-process sensitivity path builds no `.nl` for the solver, so
  :func:`problem_scaling` translates the Suffix into the vectors
  `pounce.Problem.set_problem_scaling` wants.
* **Variable factors reach the solver** (gh #486 stage 2). The core
  applies them as a change of variables one level below the algorithm
  and returns the solution in the model's own units, so no clone and
  no ``propagate_solution`` step is involved.
* **The sensitivity path honors them too** (gh #486 stage 3).
  :func:`problem_scaling` translates the variable entries alongside the
  row ones, and the sensitivity accessors carry the factors through
  their natural-units translation, so a variable-scaled solve answers
  every query in the model's own units. Stages 1 and 2 refused those
  queries rather than answering them in scaled coordinates; nothing is
  refused now.

Semantics, matching the AMPL/Ipopt reading of the suffix:

* Only an **export-enabled** Suffix named ``scaling_factor`` counts.
* Components the Suffix does not list are unscaled (factor 1.0), as are
  components listed with a factor of 0 — AMPL's suffix default is 0 and
  0 is not a usable scale.
* Entries on **inactive** constraints/objectives and on **fixed**
  variables are skipped: none of them is a row or column of the problem
  the solver is handed.
* An entry on a container (``scaling_factor[m.mass_balance]`` for an
  IndexedConstraint) applies to every member, which is how the NL
  writer expands it.
"""

import warnings

import pyomo.environ as pyo

#: The Suffix name AMPL, Ipopt, and Pyomo's own `core.scale_model` all use.
SUFFIX_NAME = "scaling_factor"

#: `nlp_scaling_method` value that turns the Suffix into actual scaling.
USER_SCALING = "user-scaling"


def user_scaling_requested(options):
    """True when `options` selects `nlp_scaling_method=user-scaling`.

    `options` is any mapping of solver options (Pyomo's `solver.options`,
    a per-call `options={...}`, or the merged form the in-process path
    receives). Anything else — including the default `gradient-based` —
    means the Suffix is not consulted, exactly as with Ipopt.
    """
    value = (options or {}).get("nlp_scaling_method")
    return value is not None and str(value).strip().lower() == USER_SCALING


def _iter_data(comp):
    """Every ComponentData under `comp` (itself, when scalar)."""
    if comp.is_indexed():
        for idx in comp:
            yield comp[idx]
    else:
        yield comp


def _suffixes(model):
    """Every export-enabled Suffix named `scaling_factor` on the model.

    Scanned across sub-blocks, since that is where the NL writer looks
    too; a block-local `scaling_factor` is as real as a top-level one.
    """
    found = []
    for sfx in model.component_objects(pyo.Suffix, active=True,
                                       descend_into=True):
        if sfx.local_name == SUFFIX_NAME and sfx.export_enabled():
            found.append(sfx)
    return found


def read_scaling(model):
    """Parse the model's `scaling_factor` Suffix.

    Returns ``None`` when the model declares no export-enabled
    `scaling_factor` Suffix — "the user supplied nothing", which leaves
    the solver on its own default (no scaling under `user-scaling`).

    Otherwise returns ``(obj_factor, constraints, variables)``:

    * ``obj_factor`` — float factor for the active objective (1.0 when
      untagged),
    * ``constraints`` — ``{ConstraintData: factor}`` for active
      constraints tagged with a usable (non-zero) factor,
    * ``variables`` — ``[(VarData, factor)]`` for non-fixed variables
      tagged with a factor other than 1.0, i.e. the entries pounce
      cannot honor. Empty when the model asks for nothing on that axis.
    """
    suffixes = _suffixes(model)
    if not suffixes:
        return None

    obj_factor = 1.0
    constraints = {}
    variables = []
    for sfx in suffixes:
        for key, value in sfx.items():
            for data in _iter_data(key):
                factor = float(value)
                ctype = data.ctype
                if ctype is pyo.Objective:
                    # Only the objective actually being minimized has a
                    # scale; a deactivated one is not in the problem.
                    if data.active and factor != 0.0:
                        obj_factor = factor
                elif ctype is pyo.Constraint:
                    if data.active and factor != 0.0:
                        constraints[data] = factor
                elif ctype is pyo.Var:
                    # A fixed variable is a constant folded into the
                    # rows, not a column the solver could rescale.
                    if not data.fixed and factor != 1.0:
                        variables.append((data, factor))
    return obj_factor, constraints, variables


def warn_if_no_suffix(model):
    """Warn when `user-scaling` is on but the model tags nothing.

    Without a Suffix the option is a no-op — the exact silence gh #483
    is about, just with the model rather than the solver at fault (a
    misspelled Suffix name, or one built with the default LOCAL
    direction so it never reaches the solver)."""
    if read_scaling(model) is None:
        warnings.warn(
            "pounce solve: nlp_scaling_method=user-scaling was requested, but "
            f"the model declares no export-enabled `{SUFFIX_NAME}` Suffix, so "
            "no scaling will be applied. Declare it with "
            f"`model.{SUFFIX_NAME} = Suffix(direction=Suffix.EXPORT)` and tag "
            "the objective / constraints you want scaled.",
            UserWarning,
            stacklevel=3)


def problem_scaling(model, con_names, con_alias, var_names):
    """Suffix -> `(obj_factor, g_scaling, x_scaling, g_scaling_given)`
    for `Problem.set_problem_scaling`.

    Used by the in-process sensitivity path, which hands pounce
    evaluator callbacks rather than an `.nl` file and so has no suffix
    segments for the solver to read.

    `con_names` is the solve's constraint rows in `.nl` order,
    `var_names` its variable columns in the same order, and `con_alias`
    maps an original constraint's name to the clone name the
    declared-parameter surgery gave it, exactly as in
    `_warm_start_from_suffixes`. Rows and columns the Suffix does not
    mention stay at 1.0, and `g_scaling_given` is ``False`` for those
    rows, which is how the `.nl` path reports a row with no suffix entry
    to `nlp_scaling_untagged_constr`. Returns ``None`` when the model
    declares no Suffix.

    Variables are translated here rather than refused (gh #486 stage
    3): the core applies them as a change of variables and every
    sensitivity accessor carries the factors back out, so the
    in-process path has the same reach as the ASL one, where the
    writer emits the entries as `.nl` suffix segments. `x_scaling` is
    ``None`` when the Suffix asks for nothing on that axis, which
    keeps the wrapper — and its cost — off an unscaled solve.

    Variables the Suffix names but the written model has no column for
    are skipped, like the constraints: the surgery clone is what was
    written, and a component missing from it is not a column pounce
    could rescale.
    """
    parsed = read_scaling(model)
    if parsed is None:
        return None
    obj_factor, constraints, variables = parsed
    con_row = {name: i for i, name in enumerate(con_names)}
    g_scaling = [1.0] * len(con_names)
    g_given = [False] * len(con_names)
    for cd, factor in constraints.items():
        # A constraint the writer dropped (trivially satisfied, or
        # absorbed into a bound) has no row to scale; skipping it
        # matches what the ASL path does with the same suffix.
        row = con_row.get(con_alias.get(cd.name, cd.name))
        if row is not None:
            g_scaling[row] = factor
            g_given[row] = True

    var_col = {name: i for i, name in enumerate(var_names)}
    x_scaling = None
    for vd, factor in variables:
        col = var_col.get(vd.name)
        if col is None:
            continue
        if x_scaling is None:
            x_scaling = [1.0] * len(var_names)
        x_scaling[col] = factor
    return obj_factor, g_scaling, x_scaling, g_given
