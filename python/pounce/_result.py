"""Shared dict/attribute access for SciPy-style result objects.

SciPy's ``solve_ivp`` / ``solve_bvp`` return a ``Bunch`` whose fields are
reachable both as attributes (``res.y``) and as dict items (``res["y"]``).
:class:`ResultMixin` gives pounce's dataclass results the same dual access
so they are drop-in for code written against SciPy.
"""

from __future__ import annotations

import dataclasses


class ResultMixin:
    """Dict-style access for dataclass results, mirroring SciPy's ``Bunch``.

    Adds ``res["field"]`` indexing, ``"field" in res`` membership,
    ``res.keys()`` / ``res.get(...)`` and iteration over field names on top
    of normal attribute access, so a pounce result is interchangeable with a
    SciPy result in downstream code.

    The dict view is restricted to the **public** fields (those that appear
    in ``repr`` — i.e. fields declared ``repr=False``, such as internal
    solver state, are excluded), and ``__getitem__`` honours that same set,
    so ``"k" in res`` and ``res["k"]`` agree and private state is never
    exposed through the SciPy-Bunch surface.
    """

    def keys(self):
        return [f.name for f in dataclasses.fields(self) if f.repr]

    def __getitem__(self, key):
        if key not in self.keys():
            raise KeyError(key)
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __contains__(self, key):
        return key in self.keys()

    def __iter__(self):
        return iter(self.keys())

    def get(self, key, default=None):
        return getattr(self, key, default) if key in self.keys() else default
