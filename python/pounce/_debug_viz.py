"""Interactive viewer for the POUNCE debugger's `viz` / `save` artifacts.

The interactive debugger (`pounce --debug` / `--debug-json`) writes a JSON
artifact and launches the command in ``$POUNCE_DBG_VIEWER``. Point that at
this viewer to get **interactive Plotly** figures instead of raw JSON::

    export POUNCE_DBG_VIEWER='pounce-dbg-viz {}'

then, at the ``pounce-dbg>`` prompt::

    stop-at kkt
    continue
    viz kkt      # → spy/heatmap of the augmented (KKT) matrix + inertia
    viz L        # → spy/heatmap of the LDLᵀ factor
    viz x        # → bar chart of the primal block

It also renders `save` artifacts (the full iterate) and any vector block.
Requires plotly (``pip install 'pounce-solver[viz]'``).
"""

from __future__ import annotations

import json
import sys
from typing import Any


def _require_plotly():
    try:
        import plotly.graph_objects as go  # noqa: F401

        return go
    except ImportError:
        sys.stderr.write(
            "pounce-dbg-viz: plotly is required.\n"
            "    pip install 'pounce-solver[viz]'   (or: pip install plotly)\n"
        )
        raise SystemExit(2)


def _spy(go, rows, cols, vals, n, title, *, symmetric=False, unit_diag=False):
    """A spy/heatmap scatter: one marker per nonzero, colored by value,
    hover shows (row, col, value). 1-based row/col inputs."""
    r = list(rows)
    c = list(cols)
    v = list(vals) if vals is not None else [1.0] * len(r)
    if symmetric:
        # Mirror the strict lower triangle to the upper triangle.
        for i in range(len(rows)):
            if rows[i] != cols[i]:
                r.append(cols[i])
                c.append(rows[i])
                v.append(v[i])
    if unit_diag:
        for i in range(1, n + 1):
            r.append(i)
            c.append(i)
            v.append(1.0)
    has_vals = any(x not in (0.0, 1.0) for x in v) or vals is not None
    fig = go.Figure(
        go.Scatter(
            x=c,
            y=r,
            mode="markers",
            marker=dict(
                size=max(4, min(18, int(420 / max(n, 1)))),
                symbol="square",
                color=v if has_vals else "#3b6",
                colorscale="RdBu" if has_vals else None,
                cmid=0 if has_vals else None,
                showscale=has_vals,
                line=dict(width=0),
            ),
            text=[f"({rr},{cc}) = {vv:.6g}" for rr, cc, vv in zip(r, c, v)],
            hoverinfo="text",
        )
    )
    fig.update_yaxes(autorange="reversed", scaleanchor="x", constrain="domain")
    fig.update_xaxes(constrain="domain")
    fig.update_layout(
        title=title,
        xaxis_title="column",
        yaxis_title="row",
        width=720,
        height=720,
        template="plotly_white",
    )
    return fig


def _bar(go, values, title):
    fig = go.Figure(go.Bar(x=list(range(len(values))), y=values))
    fig.update_layout(
        title=title,
        xaxis_title="index",
        yaxis_title="value",
        template="plotly_white",
        height=420,
    )
    return fig


def figure_for(go, art: dict[str, Any]):
    """Build a Plotly figure for one debugger artifact."""
    label = art.get("label", "")
    it = art.get("iter")

    # KKT matrix (symmetric augmented system).
    if label == "kkt" and isinstance(art.get("matrix"), dict):
        m = art["matrix"]
        n = m["dim"]
        inertia = (
            f"n+={art.get('n_pos')} n-={art.get('n_neg')} "
            f"(expected n-={art.get('expected_neg')}, "
            f"{'correct' if art.get('inertia_correct') else 'WRONG'})"
        )
        title = (
            f"KKT matrix — iter {it}, dim {n}<br>"
            f"<sub>{inertia} | δ_w={art.get('delta_w'):.2e} "
            f"δ_c={art.get('delta_c'):.2e} | {art.get('status')}</sub>"
        )
        return _spy(go, m["irn"], m["jcn"], m["vals"], n, title, symmetric=True)

    # LDLᵀ factor (strict lower, unit diagonal implicit).
    if label == "L" and "l_irn" in art:
        n = art["n"]
        title = f"LDLᵀ factor L — iter {it}, n {n} (unit diagonal implicit)"
        return _spy(
            go, art["l_irn"], art["l_jcn"], art.get("l_vals"), n, title, unit_diag=True
        )

    # A single vector block (`viz x`, `viz dx`, …).
    if "values" in art:
        return _bar(go, art["values"], f"{label} — iter {it}")

    # A `save` artifact: the full iterate. Plot the primal x by default.
    if isinstance(art.get("iterate"), dict):
        iterate = art["iterate"]
        block = "x" if "x" in iterate else next(iter(iterate), None)
        if block is not None:
            return _bar(go, iterate[block], f"iterate[{block}] — iter {it}")

    return None


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        sys.stderr.write("usage: pounce-dbg-viz <artifact.json>\n")
        return 2
    go = _require_plotly()
    path = argv[0]
    try:
        with open(path) as fh:
            art = json.load(fh)
    except (OSError, json.JSONDecodeError) as e:
        sys.stderr.write(f"pounce-dbg-viz: cannot read {path}: {e}\n")
        return 1
    fig = figure_for(go, art)
    if fig is None:
        sys.stderr.write(
            f"pounce-dbg-viz: don't know how to visualize this artifact "
            f"(label={art.get('label')!r}); keys: {sorted(art)}\n"
        )
        return 1
    fig.show()  # opens an interactive figure in the browser
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
