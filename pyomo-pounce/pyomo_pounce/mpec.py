"""Write Pyomo ``Complementarity`` pairs to ``.nl`` without losing them.

`SolverFactory("pounce")` reaches POUNCE through Pyomo's ASL interface, which
lowers every active ``Complementarity`` with the ``mpec.nl`` transformation
(`pyomo.mpec.plugins.mpec4`) and then writes the result with the NL writer.
Measured on Pyomo 6.10.0, that route loses information for two families of
pairs, and the solve that follows reports ``optimal`` at a point that does not
satisfy the complementarity the model states:

* **Two-sided ("mixed") pairs** -- ``complements(inequality(lo, r, hi), f)`` in
  either argument order. When ``mpec.nl`` puts the ranged side on the variable
  it writes the row as a free constraint ``(None, bv, None)`` with flag 3, and
  the NL writer skips every constraint that has neither bound
  (``nl_writer.py``: ``if lb is None and ub is None: continue``), so the pair
  disappears (header ``n_cc = 0``) and only the definition ``bv == f`` reaches
  the solver. When it puts the ranged side on the row instead (the free side is
  a bare variable, or its swap heuristic fires) it keeps one of the two bounds,
  drops the other, and writes flag 0 -- under AMPL semantics a free variable
  complementing a row, i.e. ``r == lo`` as an equation.
* **One-sided pairs on a bare variable that has its own, different bound on
  the same side** -- e.g. ``complements(x <= 1, g >= 0)`` with ``x`` declared
  ``ub=0.2``. ``mpec.nl`` merges the pair's bound into the variable's: it keeps
  the larger lower bound (so a tighter declared ``lb`` silently becomes the
  complementarity's reference point) and, through a ``>`` where ``<`` was
  meant, also keeps the larger *upper* bound (so a tighter declared ``ub`` is
  discarded, and a looser one replaces the pair's).

Neither is a POUNCE defect: POUNCE solves the file it receives correctly. The
information is gone before it sees the file.

:func:`lower_complementarity` therefore lowers those pairs itself, before the
ASL layer runs, into the form POUNCE's ``.nl`` reader documents: an auxiliary
variable ``w`` carrying the pair's bound(s) and defined by ``w == r``, and a
complementarity row whose body is an auxiliary ``bv == f`` (``5 3 w`` for a
mixed pair, ``5 1 w`` with ``w >= 0`` for a one-sided one). The row is given a
finite placeholder bound so the writer keeps it; the bound itself never reaches
the file, because the writer replaces a complementarity row's ``r`` line with
``5 <flag> <var>`` wholesale. The body is a bare ``bv`` (not ``f``) for the
same reason ``mpec.nl`` does it: the writer drops a complementarity row's
constant term. The user's ``Complementarity`` data are deactivated for the
solve so ``mpec.nl`` skips them, and everything is undone afterwards.

Pairs that ``mpec.nl`` writes correctly are left to it, so their ``.nl`` is
byte-for-byte what it was. The bounds ``mpec.nl`` writes onto the user's own
variables for those pairs are restored after the solve.

As a fail-closed backstop, :func:`check_written_nl` counts the complementarity
rows in the file actually written and refuses the solve if any pair is
missing or any row carries flag 0. A Pyomo release that changes either code
path then produces an error naming the cause, not a wrong answer.
"""

from pyomo.core.base.block import Block
from pyomo.core.base.component import ComponentData
from pyomo.core.base.constraint import Constraint
from pyomo.core.base.var import Var
from pyomo.core.expr.numvalue import value as _value
from pyomo.common.modeling import unique_component_name

# The row bound that keeps the NL writer from skipping a mixed pair's
# (otherwise free) complementarity row. It is never written: the writer
# replaces the row's whole `r` line with `5 3 <var>`.
_KEEP_ROW_LB = -1.0e20


class ComplementarityLoweringError(RuntimeError):
    """The ``.nl`` file written for the solve does not carry every
    complementarity pair of the model, so solving it could return a point
    that violates complementarity. Raised instead of solving."""


def _canonical(cdata):
    try:
        return (cdata._canonical_expression(cdata._args[0]),
                cdata._canonical_expression(cdata._args[1]))
    except Exception:
        return None


def _bare_var(expr):
    return (expr is not None and getattr(expr, "is_variable_type", lambda: False)()
            and isinstance(expr, ComponentData))


def _num(x):
    return None if x is None else _value(x)


def _side_conflicts(side):
    """True when ``side`` is ``(lo, v, None)`` / ``(None, v, hi)`` on a bare
    variable whose declared bound on that side exists and differs from the
    pair's -- the case ``mpec.nl`` merges incorrectly."""
    lo, body, hi = side
    if not _bare_var(body):
        return False
    if lo is not None and hi is None:
        return body.lb is not None and body.lb != _num(lo)
    if hi is not None and lo is None:
        return body.ub is not None and body.ub != _num(hi)
    return False


def _classify(cdata):
    """``("skip", None)`` for a pair ``mpec.nl`` turns into a plain equality
    or rejects itself; ``("mpec4", sides)`` for one it writes correctly;
    ``("mixed", (rng, free))`` / ``("onesided", sides)`` for one this module
    must lower."""
    sides = _canonical(cdata)
    if sides is None:
        return "skip", None
    e1, e2 = sides
    if len(e1) == 2 or len(e2) == 2:
        return "skip", None  # an equality: written as a plain constraint
    finite = [(s[0] is not None, s[2] is not None) for s in (e1, e2)]
    if sum(a + b for a, b in finite) != 2:
        return "skip", None  # mpec.nl raises its own error for this
    if finite[0] == (True, True) and finite[1] == (False, False):
        return "mixed", (e1, e2)
    if finite[1] == (True, True) and finite[0] == (False, False):
        return "mixed", (e2, e1)
    if _side_conflicts(e1) or _side_conflicts(e2):
        return "onesided", (e1, e2)
    return "mpec4", (e1, e2)


def _iter_pairs(model):
    from pyomo.mpec import Complementarity

    for cdata in model.component_data_objects(
            Complementarity, active=True, descend_into=True):
        yield cdata


def _init(var, expr):
    try:
        v = _value(expr, exception=False)
    except Exception:
        v = None
    if v is not None:
        var.set_value(v, skip_validation=True)


def _nonneg_form(side):
    """``body - lo`` for ``(lo, body, None)``, ``hi - body`` for
    ``(None, body, hi)``: a quantity that the pair requires to be ``>= 0``."""
    lo, body, hi = side
    return body - lo if lo is not None else hi - body


class Lowering:
    """What :func:`lower_complementarity` changed, and how to undo it."""

    def __init__(self, model):
        self.model = model
        self.block = None
        self.deactivated = []
        self.saved_bounds = []
        self.expected_pairs = 0
        self.lowered = []  # (pair name, kind)

    def restore(self):
        for cdata in self.deactivated:
            cdata.activate()
        self.deactivated = []
        if self.block is not None:
            # Imported suffix values (duals, bound multipliers) for the
            # auxiliary rows and columns would otherwise outlive them.
            from pyomo.core.base.suffix import Suffix

            aux = list(self.block.component_data_objects(
                (Var, Constraint), descend_into=True))
            for suffix in self.model.component_data_objects(
                    Suffix, descend_into=True):
                for comp in aux:
                    if comp in suffix:
                        del suffix[comp]
            self.model.del_component(self.block)
            self.block = None
        for var, lower, upper in self.saved_bounds:
            var.lower = lower
            var.upper = upper
        self.saved_bounds = []


def lower_complementarity(model):
    """Lower the complementarity pairs Pyomo's ``mpec.nl`` route would lose.

    Returns a :class:`Lowering` (call ``.restore()`` once the solve and its
    solution load are done), or ``None`` when ``model`` has no active
    ``Complementarity``.
    """
    pairs = list(_iter_pairs(model))
    if not pairs:
        return None
    low = Lowering(model)
    todo = []
    seen_vars = set()
    for cdata in pairs:
        kind, sides = _classify(cdata)
        if kind == "skip":
            continue
        low.expected_pairs += 1
        if kind == "mpec4":
            # mpec.nl may write the pair's bound onto a bare variable of
            # either side; put the user's declared bounds back afterwards.
            for s in sides:
                v = s[1]
                if _bare_var(v) and id(v) not in seen_vars:
                    seen_vars.add(id(v))
                    low.saved_bounds.append((v, v.lower, v.upper))
            continue
        todo.append((cdata, kind, sides))
    if not todo:
        return low

    name = unique_component_name(model, "_pounce_complementarity")
    blk = Block(concrete=True)
    model.add_component(name, blk)
    low.block = blk
    try:
        for i, (cdata, kind, sides) in enumerate(todo):
            b = Block(concrete=True)
            blk.add_component(f"pair{i}", b)
            if kind == "mixed":
                rng, free = sides
                lo, hi = _num(rng[0]), _num(rng[2])
                if lo > hi:
                    raise ValueError(
                        f"pounce solve: complementarity {cdata.name} has an "
                        f"empty range [{lo}, {hi}].")
                b.w = Var(bounds=(lo, hi))
                b.w_def = Constraint(expr=b.w == rng[1])
                row_expr, flag, row_lb = free[1], 3, _KEEP_ROW_LB
                _init(b.w, rng[1])
            else:
                e1, e2 = sides
                b.w = Var(bounds=(0, None))
                b.w_def = Constraint(expr=b.w == _nonneg_form(e1))
                row_expr, flag, row_lb = _nonneg_form(e2), 1, 0
                _init(b.w, _nonneg_form(e1))
            b.bv = Var()
            b.bv_def = Constraint(expr=b.bv == row_expr)
            _init(b.bv, row_expr)
            b.c = Constraint(expr=b.bv >= row_lb)
            # The NL writer's complementarity hook: `r` line `5 <flag> <w>`.
            b.c._complementarity = flag
            b.c._vid = id(b.w)
            cdata.deactivate()
            low.deactivated.append(cdata)
            low.lowered.append((cdata.name, kind))
    except Exception:
        low.restore()
        raise
    return low


def count_written_pairs(nl_path):
    """``(n_rows, n_flag0)``: complementarity rows in the ``r`` segment of a
    text ``.nl`` file, and how many of them carry flag 0. ``None`` when the
    file is not a text ``.nl`` this can read."""
    with open(nl_path) as fh:
        first = fh.readline()
        if not first.startswith("g"):
            return None
        counts = fh.readline().split("#")[0].split()
        try:
            n_cons = int(counts[1])
        except (IndexError, ValueError):
            return None
        for line in fh:
            if line.startswith("r"):
                break
        else:
            return (0, 0) if n_cons == 0 else None
        rows = flag0 = 0
        for _ in range(n_cons):
            line = fh.readline()
            if not line:
                break
            tok = line.split("#")[0].split()
            if tok and tok[0] == "5":
                rows += 1
                if len(tok) > 1 and tok[1] == "0":
                    flag0 += 1
        return rows, flag0


def check_written_nl(nl_path, lowering):
    """Refuse the solve when the written ``.nl`` lost a complementarity pair.

    Every active, non-equality pair must appear as exactly one complementarity
    row, and none may carry flag 0 (a free variable complementing a row, which
    is ``mpec.nl``'s rendering of a two-sided pair it mis-oriented, and reads
    as an equation)."""
    if lowering is None:
        return
    got = count_written_pairs(nl_path)
    if got is None:
        return
    rows, flag0 = got
    if rows == lowering.expected_pairs and flag0 == 0:
        return
    detail = []
    if rows != lowering.expected_pairs:
        detail.append(
            f"the model has {lowering.expected_pairs} complementarity pair(s) "
            f"but the .nl file Pyomo wrote carries {rows}")
    if flag0:
        detail.append(
            f"{flag0} row(s) were written with flag 0, which states an "
            f"equation, not a complementarity")
    raise ComplementarityLoweringError(
        "pounce solve: refusing to solve -- "
        + "; ".join(detail)
        + ". Solving that file would return a point that need not satisfy "
        "the complementarity your model states. This is a loss in Pyomo's "
        "mpec.nl transformation / NL writer, which pyomo-pounce did not "
        "manage to work around for this form. Restate the pair one-sided "
        "(e.g. split `complements(inequality(lo, x, hi), f)` with f = fp - fm, "
        "fp, fm >= 0 into `complements(x - lo >= 0, fp >= 0)` and "
        "`complements(hi - x >= 0, fm >= 0)`), or solve through the POUNCE "
        "Python API, `pounce.Problem(..., complementarity=...)`, which takes "
        "mixed pairs directly. Please report the model at "
        "https://github.com/jkitchin/pounce/issues.")
