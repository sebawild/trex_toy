"""The compression ledger: every representation step, priced in bits.

This is a standalone Python re-implementation of the "explore the compression"
mode of the companion website for

    Ismaili Alaoui, Nakajima, Namrata, Wild.
    Rooting Out Entropy: Optimal Tree Extraction for Ultra-Succinct Graphs.
    arXiv:2603.14649

It produces, for one graph, the same rows as the website

    (0)  adjacency lists                     m*ceil(lg n) + n*ceil(lg m)
    (1)  one string + separator bitvector    m*ceil(lg n) + lg C(n+m, n)
    (1u) each edge stored once (undirected)  m'*ceil(lg n) + lg C(n+m', n)
    (2)  entropy-compressed adjacency string H(G) + lg C(n+m, n)
    (3b) TREX, best MST-based tree           H(G-T) + lg C(n+m', n) + 3n (2n undirected)
    (3w) TREX, worst MST-based tree          same, with the maximum spanning tree
    (4)  twin removal, then TREX             H(G0-T0) + lg C(n0+m0', n0) + 3n0 + lg C(N, n0)

and draws them as horizontal bars, split into their components and -- in the
default "absolute" mode -- all on one bits-per-pixel scale, so that row (0)
fills the width and every later row is literally as short as it is small.

Two columns are kept apart throughout, exactly as in Theorem 1.8: the bits that
information theory forces (entropies, the combinatorial term for the separator
bitvector, and the exact linear terms), and the o(.) redundancy that succinct
rank/select structures add on top.  The redundancy is *not* included in any
total; on small graphs it would swamp everything.

The graph-level quantities (entropies, orientations, MST weights, the residual
graph G-T) are computed exactly as in ``evaluation/evaluator.py`` and
``src/structure/Builder.py`` so that the numbers here agree with the pipeline's
``array total bits`` / ``bitvector total bits`` / ``total bits trex``.

Validated against ``merged.csv``: rows (0), (2) and (3b) reproduce the pipeline
to floating-point on every graph in ``Datasets/`` except ``Amazon0302``, where
tie-breaking among equal MST weights moves row (3b) by 0.011%.

Note one deliberate deviation from the website: for undirected graphs the page
weights MST edges by ``min(deg u, deg v)``, whereas ``Builder.build_undirected``
uses ``max(deg u, deg v)``.  Since the orientation points at the busier endpoint,
``max`` is the weight the greedy argument of Corollary 1.3 calls for, and it is
what this module uses.

Usage
-----
    python -m evaluation.ledger Datasets/fig2-example.txt
    python -m evaluation.ledger Datasets/Undirected/power-US-Grid.txt --undirected
    python -m evaluation.ledger --directory Datasets/Directed --outdir figures
    python -m evaluation.ledger Datasets/fig2-example.txt --relative --no-show

Every run writes ``<name>-ledger.csv`` (one row per step) and
``<name>-ledger.pdf``/``.png`` into ``--outdir``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass, field
from typing import Callable, Iterable

import networkx as nx
import pandas as pd

# --------------------------------------------------------------------------
# themes
# --------------------------------------------------------------------------
#
# ``web``   the colours of the companion website, so page and figures match
# ``paper`` neutral, white background, colourblind-safe (Okabe-Ito), serif
# ``mono``  greyscale ramp for black-and-white printing
#
# Switch with ``use_theme("paper")`` before plotting, or ``--theme paper`` on
# the command line.  ``PALETTE`` and ``COMPONENT_COLOURS`` are mutated in place,
# so ``from evaluation.ledger import PALETTE`` keeps working after a switch.

THEMES = {
    "web": {
        "palette": {
            "bg": "#F1EED9", "panel": "#f8f6ec", "border": "#cdc9bb",
            "ink": "#46433A", "ink_soft": "#6b675c", "ink_faint": "#8b8779",
            "accent": "#902015", "inert": "#d8d4c2",
        },
        "components": ["#902015", "#b07d18", "#2a6a63", "#6f6a5c", "#c08777"],
        "rc": {"font.family": "sans-serif"},
    },
    "paper": {
        "palette": {
            "bg": "#ffffff", "panel": "#ffffff", "border": "#b0b0b0",
            "ink": "#000000", "ink_soft": "#404040", "ink_faint": "#707070",
            "accent": "#000000", "inert": "#d9d9d9",
        },
        # Okabe-Ito: distinguishable under the common colour-vision deficiencies
        "components": ["#D55E00", "#E69F00", "#0072B2", "#666666", "#CC79A7"],
        "rc": {"font.family": "serif", "axes.titleweight": "bold"},
    },
    "mono": {
        "palette": {
            "bg": "#ffffff", "panel": "#ffffff", "border": "#999999",
            "ink": "#000000", "ink_soft": "#333333", "ink_faint": "#666666",
            "accent": "#000000", "inert": "#dddddd",
        },
        "components": ["#1a1a1a", "#666666", "#a6a6a6", "#d9d9d9", "#f0f0f0"],
        "rc": {"font.family": "serif", "axes.titleweight": "bold"},
    },
}

DEFAULT_THEME = "web"

PALETTE: dict = dict(THEMES[DEFAULT_THEME]["palette"])
COMPONENT_COLOURS: list = list(THEMES[DEFAULT_THEME]["components"])
_ACTIVE = [DEFAULT_THEME]


def active_theme() -> str:
    return _ACTIVE[0]


def theme_rc(name: str | None = None) -> dict:
    """The matplotlib rcParams for a theme, so notebooks can style their own
    figures to match the ledger plots."""
    t = THEMES[name or _ACTIVE[0]]
    p = t["palette"]
    rc = {
        "figure.facecolor": p["bg"], "savefig.facecolor": p["bg"],
        "axes.facecolor": p["panel"], "axes.edgecolor": p["border"],
        "axes.labelcolor": p["ink"], "text.color": p["ink"],
        "xtick.color": p["ink_faint"], "ytick.color": p["ink_faint"],
        "axes.titlecolor": p["accent"], "grid.color": p["border"],
        "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
    }
    rc.update(t.get("rc", {}))
    return rc


def use_theme(name: str = "paper", apply_rc: bool = True) -> dict:
    """Switch the palette used by :func:`plot_ledger` (and, with ``apply_rc``,
    by every other matplotlib figure in the session).

    Returns the active palette, so ``P = use_theme("paper")`` is a convenient
    one-liner in a notebook.
    """
    if name not in THEMES:
        raise ValueError(f"unknown theme {name!r}; choose from {sorted(THEMES)}")
    t = THEMES[name]
    PALETTE.clear()
    PALETTE.update(t["palette"])
    COMPONENT_COLOURS[:] = list(t["components"])
    _ACTIVE[0] = name
    if apply_rc:
        import matplotlib.pyplot as plt

        plt.rcParams.update(theme_rc(name))
    return PALETTE


def series_colours() -> tuple[str, str, str, str]:
    """The four colours the notebook figures use, in the current theme:
    baseline (0), entropy-compressed (2), TREX (3b), and a contrasting fourth."""
    return (PALETTE["inert"], COMPONENT_COLOURS[1], COMPONENT_COLOURS[0],
            COMPONENT_COLOURS[2])


def _readable_on(hex_colour: str) -> str:
    """Black or white, whichever is legible on the given fill."""
    h = hex_colour.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "#000000" if lum > 0.55 else "#ffffff"


# --------------------------------------------------------------------------
# information-theoretic primitives
# --------------------------------------------------------------------------


def lg(x: float) -> float:
    return math.log2(x)


def lg_binom(a: float, b: float) -> float:
    """lg C(a, b) in bits, via lgamma so that it survives large n."""
    if b < 0 or b > a or a <= 0:
        return 0.0
    return (math.lgamma(a + 1) - math.lgamma(b + 1) - math.lgamma(a - b + 1)) / math.log(2)


def entropy(counts: Iterable[int], total: int) -> float:
    """H(d_1,...,d_n) = sum_v d_v lg(total / d_v), i.e. m * H^in_deg."""
    if total <= 0:
        return 0.0
    return sum(k * lg(total / k) for k in counts if k > 0)


# --------------------------------------------------------------------------
# the rows
# --------------------------------------------------------------------------


@dataclass
class Row:
    key: str
    name: str
    formula: str
    bits: float | None                      # information-theoretic bits; None = not applicable
    redundancy: list[str] = field(default_factory=list)
    components: list[tuple[str, float, str]] = field(default_factory=list)
    m: int = 0                              # arcs the row actually stores
    H: float = 0.0                          # entropy part of the row
    tree_edges: int = 0
    vs_prev: float | None = None            # fraction saved vs the previous priced row
    vs_base: float | None = None            # fraction saved vs row (0)
    vs_entropy: float | None = None         # only for (3w): vs row (2)


@dataclass
class Ledger:
    name: str
    directed: bool
    n: int
    arcs: int                               # arcs stored the textbook way (2|E| if undirected)
    edges: int                              # |E|
    rows: list[Row]
    H0: float                               # entropy of the row-(2) string
    m_ent: int                              # arcs that entropy is taken over in row (2)
    indegrees: list[int]

    def row(self, key: str) -> Row | None:
        return next((r for r in self.rows if r.key == key), None)

    def frame(self) -> pd.DataFrame:
        """One row per step.

        The frame is a complete description of the ledger, not just a report of
        it: the per-component split is kept as JSON in ``components``, and the
        graph-level fields are repeated on every row.  :func:`ledger_from_frame`
        turns it back into a :class:`Ledger`, so a figure can be redrawn -- in a
        different theme, say -- without pricing the graph again.
        """
        def pct(x):
            return None if x is None else 100.0 * x

        return pd.DataFrame(
            [
                {
                    "graph": self.name,
                    "n": self.n,
                    "arcs": self.arcs,
                    "edges": self.edges,
                    "directed": self.directed,
                    "step": r.key,
                    "name": r.name,
                    "formula": r.formula,
                    "bits": r.bits,
                    "bits per edge": None if r.bits is None else r.bits / max(self.edges, 1),
                    "redundancy": " + ".join(r.redundancy) if r.redundancy else "",
                    "saved vs previous (%)": pct(r.vs_prev),
                    "saved vs (0) (%)": pct(r.vs_base),
                    "saved vs (2) (%)": pct(r.vs_entropy),
                    "H part": r.H if r.bits is not None else None,
                    "stored arcs": r.m if r.bits is not None else None,
                    "tree edges": r.tree_edges if r.bits is not None else None,
                    "components": json.dumps(r.components),
                }
                for r in self.rows
            ]
        )


def _infer_missing_metadata(df: pd.DataFrame) -> dict:
    """Recover n, the arc counts and the per-component split from an old-format
    ledger CSV -- one written before ``components``/``n``/``edges`` were stored.

    Everything here is arithmetic on numbers already in the file; the graph is
    never touched.  ``n`` comes from inverting row (0),
    ``bits = m*ceil(lg n) + n*ceil(lg m)``, by trying each possible id width.
    """
    by_step = {str(d["step"]): d for _, d in df.iterrows()}
    if "0" not in by_step:
        raise ValueError("cannot rebuild: the CSV has no row (0) to invert")

    r0 = by_step["0"]
    directed = "1u" not in by_step
    edges = int(round(float(r0["bits"]) / float(r0["bits per edge"])))
    arcs = edges if directed else 2 * edges
    ptr_bits = max(1, math.ceil(lg(max(arcs, 2))))

    n = None
    for id_bits in range(1, 65):
        rest = float(r0["bits"]) - arcs * id_bits
        cand = rest / ptr_bits
        if cand < 1 or abs(cand - round(cand)) > 1e-6:
            continue
        cand = int(round(cand))
        if max(1, math.ceil(lg(max(cand, 2)))) == id_bits:
            n, id_bits_final = cand, id_bits
            break
    if n is None:
        raise ValueError("cannot rebuild: row (0) does not invert to an integer n")

    sep = lambda a: lg_binom(n + a, n)          # noqa: E731
    linear = (3 if directed else 2) * n
    lin_label = "3n - LOUDS, plus n for D" if directed else "2n - LOUDS"

    comps = {}
    for key, d in by_step.items():
        bits = d.get("bits")
        if bits is None or (isinstance(bits, float) and math.isnan(bits)):
            comps[key] = []
            continue
        bits = float(bits)
        m = int(d["stored arcs"]) if not pd.isna(d.get("stored arcs")) else arcs
        H = 0.0 if pd.isna(d.get("H part")) else float(d["H part"])
        if key == "0":
            comps[key] = [("m*ceil(lg n) - one id per stored arc", arcs * id_bits_final,
                           f"{arcs} x {id_bits_final}"),
                          ("n*ceil(lg m) - one offset per vertex", n * ptr_bits,
                           f"{n} x {ptr_bits}")]
        elif key in ("1", "1u"):
            paid = m * id_bits_final
            comps[key] = [("m*ceil(lg n) - the adjacency string, uncompressed", paid,
                           f"{m} x {id_bits_final}"),
                          ("lg C(n+m, n) - the separator bitvector", bits - paid, "")]
        elif key == "2":
            comps[key] = [("H(G) - in-degree entropy of the string", H,
                           f"sum d lg(m/d) over {m} arcs"),
                          ("lg C(n+m, n) - the separator bitvector", bits - H, "")]
        elif key in ("3b", "3w"):
            comps[key] = [("H(G-T) - entropy of what is left", H, f"{m} residual arcs"),
                          ("lg C(n+m', n) - separator bitvector of A'",
                           bits - H - linear, ""),
                          (lin_label, linear, f"{3 if directed else 2} x {n}")]
        else:
            # row (4) needs n0, which the old format did not store: show it whole
            comps[key] = [("total (component split not stored in this CSV)", bits, "")]
    return {"n": n, "arcs": arcs, "edges": edges, "directed": directed,
            "components": comps, "sep": sep}


def ledger_from_frame(df: pd.DataFrame, name: str | None = None) -> Ledger:
    """Rebuild a :class:`Ledger` from the CSV written by :meth:`Ledger.frame`.

    Only the fields the plot needs are restored; ``H0``, ``m_ent`` and
    ``indegrees`` are not stored per row, so :func:`compression_bounds` still
    needs the graph itself.
    """
    def maybe(v):
        return None if v is None or (isinstance(v, float) and math.isnan(v)) else float(v)

    # old-format CSVs lack the graph-level fields and the component split;
    # both are recoverable from the numbers already in the file
    legacy = None
    if "components" not in df.columns or "n" not in df.columns:
        legacy = _infer_missing_metadata(df)

    rows = []
    for _, d in df.iterrows():
        bits = maybe(d.get("bits"))
        red = str(d.get("redundancy") or "")
        rows.append(Row(
            key=str(d["step"]),
            name=str(d["name"]),
            formula=str(d["formula"]),
            bits=bits,
            redundancy=([] if red.lower() in ("", "nan", "none", "-")
                        else [t.strip() for t in red.split("+") if t.strip()]),
            components=(legacy["components"].get(str(d["step"]), []) if legacy
                        else [tuple(c) for c in json.loads(d.get("components") or "[]")]),
            m=int(d["stored arcs"]) if maybe(d.get("stored arcs")) is not None else 0,
            H=maybe(d.get("H part")) or 0.0,
            tree_edges=(int(d["tree edges"])
                        if "tree edges" in df.columns and maybe(d.get("tree edges")) is not None
                        else 0),
            vs_prev=(lambda v: None if v is None else v / 100)(maybe(d.get("saved vs previous (%)"))),
            vs_base=(lambda v: None if v is None else v / 100)(maybe(d.get("saved vs (0) (%)"))),
            vs_entropy=(lambda v: None if v is None else v / 100)(maybe(d.get("saved vs (2) (%)"))),
        ))
    first = df.iloc[0]
    meta = legacy or {"n": int(first["n"]), "arcs": int(first["arcs"]),
                      "edges": int(first["edges"]), "directed": bool(first["directed"])}
    return Ledger(
        name=name or str(first["graph"]),
        directed=meta["directed"],
        n=meta["n"],
        arcs=meta["arcs"],
        edges=meta["edges"],
        rows=rows,
        H0=float("nan"),
        m_ent=0,
        indegrees=[],
    )


# --------------------------------------------------------------------------
# graph preparation -- mirrors Builder.build_* and Evaluator.*_metrics
# --------------------------------------------------------------------------


def _directed_orientation(G: nx.DiGraph) -> tuple[dict[int, int], nx.Graph, int]:
    """In-degree counts of G, and the undirected skeleton carrying MST weights.

    The weight of an edge is the in-degree of its target; for a 2-cycle it is
    the smaller of the two in-degrees, since the tree may take either arc.
    (Builder.build_directed, extraction_strategy="greedy".)
    """
    indeg = {v: G.in_degree(v) for v in G.nodes()}
    skeleton = nx.Graph()
    skeleton.add_nodes_from(G.nodes())
    for u, v in G.edges():
        w = indeg[v]
        if G.has_edge(v, u):
            w = min(indeg[u], w)
        if skeleton.has_edge(u, v):
            skeleton[u][v]["weight"] = min(skeleton[u][v]["weight"], w)
        else:
            skeleton.add_edge(u, v, weight=w)
    return indeg, skeleton, G.number_of_edges()


def _undirected_orientation(G: nx.Graph) -> tuple[nx.DiGraph, nx.Graph]:
    """Orient every edge towards its busier endpoint; weight = that degree.

    This is the U-MINETREX orientation used in Builder.build_undirected and in
    Evaluator.undirected_metrics ("bitvector greedy total bits").
    """
    oriented = nx.DiGraph()
    oriented.add_nodes_from(G.nodes())
    skeleton = nx.Graph()
    skeleton.add_nodes_from(G.nodes())
    deg = dict(G.degree())
    for u, v in G.edges():
        skeleton.add_edge(u, v, weight=max(deg[u], deg[v]))
        if deg[v] > deg[u]:
            oriented.add_edge(u, v)
        else:
            oriented.add_edge(v, u)
    return oriented, skeleton


def _spanning_forest(skeleton: nx.Graph, worst: bool) -> nx.Graph:
    """The two ends of the MST-based approximation: min- and max-weight forest."""
    if worst:
        return nx.maximum_spanning_tree(skeleton, weight="weight")
    return nx.minimum_spanning_tree(skeleton, weight="weight")


def _residual_directed(G: nx.DiGraph, forest: nx.Graph, indeg: dict, worst: bool):
    """Remove one arc per tree edge; return residual in-degree counts and m'."""
    counts = dict(indeg)
    removed = 0
    for u, v in forest.edges():
        if G.has_edge(u, v) and G.has_edge(v, u):
            # greedy takes the arc into the cheaper (lower in-degree) endpoint
            take_v = indeg[v] <= indeg[u] if not worst else indeg[v] > indeg[u]
            counts[v if take_v else u] -= 1
        elif G.has_edge(u, v):
            counts[v] -= 1
        else:
            counts[u] -= 1
        removed += 1
    return counts, G.number_of_edges() - removed, removed


def _residual_undirected(oriented: nx.DiGraph, forest: nx.Graph):
    counts = {v: oriented.in_degree(v) for v in oriented.nodes()}
    removed = 0
    for u, v in forest.edges():
        counts[v if oriented.has_edge(u, v) else u] -= 1
        removed += 1
    return counts, oriented.number_of_edges() - removed, removed


def _twin_classes(G) -> dict:
    """False twins: identical open neighbourhoods (in *and* out for digraphs)."""
    directed = isinstance(G, nx.DiGraph)
    classes: dict[tuple, list] = {}
    for v in G.nodes():
        if directed:
            key = (
                tuple(sorted(x for x in G.successors(v) if x != v)),
                tuple(sorted(x for x in G.predecessors(v) if x != v)),
            )
        else:
            key = tuple(sorted(x for x in G.neighbors(v) if x != v))
        classes.setdefault(key, []).append(v)
    return classes


def _contract_twins(G, classes: dict):
    rep_of = {}
    members = list(classes.values())
    for cls in members:
        for v in cls:
            rep_of[v] = cls[0]
    if isinstance(G, nx.DiGraph):
        H = nx.DiGraph()
    else:
        H = nx.Graph()
    H.add_nodes_from(cls[0] for cls in members)
    for u, v in G.edges():
        ru, rv = rep_of[u], rep_of[v]
        # false twins are pairwise non-adjacent, so ru == rv only for self-loops,
        # which the contracted graph does not store
        if ru != rv:
            H.add_edge(ru, rv)
    return H


# --------------------------------------------------------------------------
# the ledger itself
# --------------------------------------------------------------------------


def _trex_rows(G, n, worst: bool, key: str, name: str, id_bits_unused=None):
    """Shared body of rows (3b)/(3w) and of the reduced graph inside row (4)."""
    directed = isinstance(G, nx.DiGraph)
    if directed:
        indeg, skeleton, _ = _directed_orientation(G)
        forest = _spanning_forest(skeleton, worst)
        counts, m_res, tree_edges = _residual_directed(G, forest, indeg, worst)
    else:
        oriented, skeleton = _undirected_orientation(G)
        forest = _spanning_forest(skeleton, worst)
        counts, m_res, tree_edges = _residual_undirected(oriented, forest)
    H = entropy(counts.values(), m_res)
    linear = (3 if directed else 2) * n
    return H, m_res, tree_edges, linear


def compression_ledger(G, name: str = "graph", with_twins: bool = True) -> Ledger:
    """Price every representation step for one graph."""
    directed = isinstance(G, nx.DiGraph)
    n = G.number_of_nodes()
    E = G.number_of_edges()
    M = E if directed else 2 * E            # arcs a textbook adjacency list stores

    id_bits = max(1, math.ceil(lg(max(n, 2))))
    ptr_bits = max(1, math.ceil(lg(max(M, 2))))
    sep = lambda a: lg_binom(n + a, n)      # noqa: E731

    rows: list[Row] = []

    # ---- (0) adjacency lists ------------------------------------------------
    rows.append(
        Row(
            key="0",
            name="Adjacency lists",
            formula="m*ceil(lg n) + n*ceil(lg m)",
            bits=M * id_bits + n * ptr_bits,
            redundancy=[],
            components=[
                ("m*ceil(lg n) - one id per stored arc", M * id_bits, f"{M} x {id_bits}"),
                ("n*ceil(lg m) - one offset per vertex", n * ptr_bits, f"{n} x {ptr_bits}"),
            ],
            m=M,
        )
    )

    # ---- (1) one string + separator bitvector -------------------------------
    rows.append(
        Row(
            key="1",
            name="One string + separator bitvector",
            formula="m*ceil(lg n) + lg C(n+m, n)",
            bits=M * id_bits + sep(M),
            redundancy=["o(m)"],
            components=[
                ("m*ceil(lg n) - the adjacency string, uncompressed", M * id_bits, f"{M} x {id_bits}"),
                ("lg C(n+m, n) - the separator bitvector", sep(M), f"C({n + M}, {n})"),
            ],
            m=M,
        )
    )

    # ---- (1u) undirected: each edge stored once -----------------------------
    if not directed:
        rows.append(
            Row(
                key="1u",
                name="Wavelet tree: each edge stored once",
                formula="m'*ceil(lg n) + lg C(n+m', n),  m' = |E|",
                bits=E * id_bits + sep(E),
                redundancy=["o(m)"],
                components=[
                    ("m'*ceil(lg n) - each edge stored once", E * id_bits, f"{E} x {id_bits}"),
                    ("lg C(n+m', n)", sep(E), f"C({n + E}, {n})"),
                ],
                m=E,
            )
        )

    # ---- (2) entropy-compressed adjacency string ----------------------------
    if directed:
        counts = {v: G.in_degree(v) for v in G.nodes()}
        m_ent = M
    else:
        oriented, _ = _undirected_orientation(G)
        counts = {v: oriented.in_degree(v) for v in oriented.nodes()}
        m_ent = E
    H0 = entropy(counts.values(), m_ent)
    rows.append(
        Row(
            key="2",
            name="Entropy-compressed adjacency string",
            formula="H(G) + lg C(n+m, n)",
            bits=H0 + sep(m_ent),
            redundancy=["o(m)"],
            components=[
                ("H(G) - in-degree entropy of the string", H0, f"sum d lg(m/d) over {m_ent} arcs"),
                ("lg C(n+m, n) - the separator bitvector", sep(m_ent), ""),
            ],
            m=m_ent,
            H=H0,
        )
    )

    # ---- (3b)/(3w) TREX at both ends of the MST-based range -----------------
    for worst, key in ((False, "3b"), (True, "3w")):
        H, m_res, tree_edges, linear = _trex_rows(G, n, worst, key, "")
        lin_label = "3n - LOUDS, plus n for D" if directed else "2n - LOUDS"
        rows.append(
            Row(
                key=key,
                name="TREX - " + ("worst" if worst else "best") + " MST-based tree",
                formula=f"H(G-T) + lg C(n+m', n) + {'3n' if directed else '2n'}",
                bits=H + sep(m_res) + linear,
                redundancy=["o(m)", "o(n)"],
                components=[
                    ("H(G-T) - entropy of what is left", H, f"{m_res} residual arcs"),
                    ("lg C(n+m', n) - separator bitvector of A'", sep(m_res), ""),
                    (lin_label, linear, f"{3 if directed else 2} x {n}"),
                ],
                m=m_res,
                H=H,
                tree_edges=tree_edges,
            )
        )

    # ---- (4) twin removal, then TREX ---------------------------------------
    if with_twins:
        classes = _twin_classes(G)
        n0 = len(classes)
        if n0 < n:
            G0 = _contract_twins(G, classes)
            H, m_res, tree_edges, linear = _trex_rows(G0, n0, False, "4", "")
            class_bits = lg_binom(n, n0)
            rows.append(
                Row(
                    key="4",
                    name="Twin removal, then TREX",
                    formula=(
                        f"H(G0-T0) + lg C(n0+m0', n0) + {'3n0' if directed else '2n0'} "
                        "+ lg C(N, n0)"
                    ),
                    bits=H + lg_binom(n0 + m_res, n0) + linear + class_bits,
                    redundancy=["o(m)", "o(n)", "o(N)"],
                    components=[
                        ("H(G0-T0) - entropy after contraction", H, f"{m_res} residual arcs, n0 = {n0}"),
                        ("lg C(n0+m0', n0)", lg_binom(n0 + m_res, n0), ""),
                        (
                            ("3n0 - LOUDS and D" if directed else "2n0 - LOUDS"),
                            linear,
                            f"{3 if directed else 2} x {n0}",
                        ),
                        ("lg C(N, n0) - the class sizes in B", class_bits, f"{n - n0} twins removed"),
                    ],
                    m=m_res,
                    H=H,
                    tree_edges=tree_edges,
                )
            )
        else:
            rows.append(
                Row(key="4", name="Twin removal, then TREX", formula="no twins in this graph", bits=None)
            )

    # ---- savings, relative to the previous priced row and to row (0) --------
    base = rows[0].bits
    ent_row = next(r for r in rows if r.key == "2")
    prev = base
    for r in rows:
        if r.bits is None:
            continue
        r.vs_prev = None if r.key == "0" else 1 - r.bits / prev
        r.vs_base = 1 - r.bits / base
        if r.key == "3w":
            # the worst tree is a side branch: compare it with the best tree AND
            # with the entropy-compressed string it would have replaced
            r.vs_entropy = 1 - r.bits / ent_row.bits
        else:
            prev = r.bits

    return Ledger(
        name=name,
        directed=directed,
        n=n,
        arcs=M,
        edges=E,
        rows=rows,
        H0=H0,
        m_ent=m_ent,
        indegrees=sorted(counts.values(), reverse=True),
    )


# --------------------------------------------------------------------------
# the bounds of Theorems 1.6 and 1.7
# --------------------------------------------------------------------------


def compression_bounds(led: Ledger, G, exact_density: bool = False) -> dict:
    """What the theorems guarantee, next to what the greedy tree achieved.

    Theorem 1.6 needs alpha = max subgraph density / density, which we get from
    the repo's Greedy++ peeling (never an over-estimate of the optimum, so the
    resulting bound stays a valid statement about a *feasible* alpha only
    approximately -- flagged as such).  Theorem 1.7 needs only the number of
    vertices with non-zero in-degree and is exact.
    """
    per_edge = led.H0 / led.m_ent if led.m_ent else 0.0
    n = led.n
    n_pos = sum(1 for d in led.indegrees if d > 0)
    best, worst = led.row("3b"), led.row("3w")

    def bound(alpha):
        if alpha is None or alpha == 0:
            return None
        return max(0.0, (n / (2 * alpha)) * per_edge - 2 * n / math.log(2))

    alpha_7 = n / n_pos if n_pos else None
    alpha_6 = None
    if exact_density:
        try:
            from src.functions.density import density_greedy

            if led.directed:
                MG = nx.MultiGraph()
                MG.add_nodes_from(G.nodes())
                MG.add_edges_from(G.edges())
                dens_max = density_greedy(MG, 1)[0]
            else:
                dens_max = density_greedy(G, 1)[0]
            alpha_6 = dens_max / (led.edges / n) if led.edges else None
        except Exception:                   # pragma: no cover - optional extra
            alpha_6 = None

    return {
        "H^in_deg(G) (bits/edge)": per_edge,
        "H(G)": led.H0,
        "vertices with non-zero in-degree": n_pos,
        "alpha (Thm 1.7)": alpha_7,
        "guaranteed saving (Thm 1.7)": bound(alpha_7),
        "alpha (Thm 1.6, peeling)": alpha_6,
        "guaranteed saving (Thm 1.6)": bound(alpha_6),
        "achieved saving, best tree": led.H0 - (best.H if best else 0.0),
        "achieved saving, worst tree": led.H0 - (worst.H if worst else 0.0),
        "lg(n!)": math.lgamma(n + 1) / math.log(2),
        "n lg n": n * lg(max(n, 2)),
    }


# --------------------------------------------------------------------------
# the figure
# --------------------------------------------------------------------------


ROLE_LABELS = [
    "adjacency payload - vertex ids, or H(.) once entropy-compressed",
    "vertex boundaries - offsets, resp. the separator bitvector",
    "tree topology - LOUDS (plus D for digraphs)",
    "twin class sizes - lg C(N, n0)",
]


def _fmt_bits(x: float) -> str:
    """Exact-ish, for the number printed next to a bar."""
    return f"{x:,.1f}" if x < 1e4 else f"{x:,.0f}"


def _fmt_axis(x: float) -> str:
    """Compact, for tick labels."""
    if x == 0:
        return "0"
    if x >= 1e9:
        return f"{x / 1e9:.2f}G"
    if x >= 1e6:
        return f"{x / 1e6:.2f}M"
    if x >= 1e3:
        return f"{x / 1e3:.0f}k"
    return f"{x:.0f}"


def plot_ledger(led: Ledger, absolute: bool = True, ax=None, title: str | None = None,
                show_redundancy: bool = True, theme: str | None = None):
    """One horizontal stacked bar per step, on a shared bits-per-pixel scale.

    ``absolute=True`` reproduces the website's absolute mode: row (0) fills the
    width, so a row that only shrinks one component visibly shows it.
    ``absolute=False`` normalises every bar to full width, which reads the
    *split* rather than the size.

    Colours are by role, consistently across rows: the payload string, the
    vertex boundaries, the extracted tree, the twin classes.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    if theme is not None and theme != active_theme():
        use_theme(theme)

    priced = [r for r in led.rows if r.bits is not None]
    skipped = [r for r in led.rows if r.bits is None]
    base = priced[0].bits

    if ax is None:
        height = 0.80 * len(led.rows) + 2.6
        fig, ax = plt.subplots(figsize=(14.0, height), layout="constrained")
        fig.patch.set_facecolor(PALETTE["bg"])
    else:
        fig = ax.figure
    ax.set_facecolor(PALETTE["panel"])

    ypos = list(range(len(led.rows)))[::-1]
    bar_h = 0.44
    widest = max((r.bits / base if absolute else 1.0) for r in priced)
    x0 = max(1.0, widest) + 0.05
    col_bits = x0 + 0.30      # right-aligned
    col_red = x0 + 0.36       # left-aligned
    col_prev = x0 + 0.94      # right-aligned
    col_base = x0 + 1.16      # right-aligned

    for y, r in zip(ypos, led.rows):
        if r.bits is None:
            ax.barh(y, 1.0, height=bar_h, color="none",
                    edgecolor=PALETTE["border"], linewidth=0.8)
            ax.text(0.006, y, r.formula, va="center", ha="left", fontsize=8.5,
                    color=PALETTE["ink_faint"], style="italic")
            continue
        total = r.bits
        width = total / base if absolute else 1.0
        left = 0.0
        for i, (_label, bits, _note) in enumerate(r.components):
            seg = width * bits / total
            ax.barh(y, seg, left=left, height=bar_h,
                    color=COMPONENT_COLOURS[i % len(COMPONENT_COLOURS)],
                    edgecolor=PALETTE["panel"], linewidth=0.6)
            if seg > 0.06:
                fill = COMPONENT_COLOURS[i % len(COMPONENT_COLOURS)]
                ax.text(left + seg / 2, y, f"{100 * bits / total:.0f}%",
                        va="center", ha="center", fontsize=8,
                        color=_readable_on(fill), fontweight="bold")
            left += seg
        ax.text(col_bits, y, _fmt_bits(total), va="center", ha="right",
                fontsize=9.5, color=PALETTE["ink"], family="monospace")
        if show_redundancy:
            ax.text(col_red, y, " + ".join(r.redundancy) if r.redundancy else "-",
                    va="center", ha="left", fontsize=8, color=PALETTE["ink_faint"],
                    family="monospace")
        if r.vs_prev is not None:
            ax.text(col_prev, y, f"{-100 * r.vs_prev:+.1f}%", va="center", ha="right",
                    fontsize=8.5, color=PALETTE["ink"], family="monospace")
        if r.vs_entropy is not None:
            ax.text(col_prev, y - 0.30, f"{-100 * r.vs_entropy:+.1f}% vs (2)",
                    va="center", ha="right", fontsize=7.5,
                    color=PALETTE["ink_faint"], family="monospace")
        if r.vs_base is not None:
            ax.text(col_base, y, f"{-100 * r.vs_base:+.1f}%", va="center", ha="right",
                    fontsize=8.5, color=PALETTE["ink"], family="monospace")

    ax.set_yticks(ypos)
    ax.set_yticklabels([f"({r.key})  {r.name}" for r in led.rows],
                       fontsize=10, color=PALETTE["ink"])
    for y, r in zip(ypos, led.rows):
        if r.bits is not None:
            ax.text(-0.004, y - 0.44, r.formula, va="top", ha="right",
                    fontsize=7.5, color=PALETTE["ink_faint"], family="monospace",
                    transform=ax.get_yaxis_transform())

    ax.set_xlim(0, col_base + 0.05)
    ax.set_ylim(-0.8, len(led.rows) - 0.2)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(PALETTE["border"])
    ax.tick_params(axis="x", colors=PALETTE["ink_faint"], labelsize=8)
    ax.tick_params(axis="y", length=0)

    ticks = [0, 0.25, 0.5, 0.75, 1.0]
    ax.set_xticks(ticks)
    if absolute:
        ax.set_xticklabels([_fmt_axis(t * base) for t in ticks])
        ax.set_xlabel(f"bits - one scale for every row; the full width is row (0) = "
                      f"{_fmt_bits(base)} bits",
                      fontsize=9, color=PALETTE["ink_soft"])
        for t in ticks[1:]:
            ax.axvline(t, color=PALETTE["border"], lw=0.6, ls=":", zorder=0)
    else:
        ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
        ax.set_xlabel("share of the row's own total - reads the split, not the size",
                      fontsize=9, color=PALETTE["ink_soft"])

    head = title or (
        f"{led.name.replace('_', ' ')}    n = {led.n:,},  "
        f"{led.edges:,} {'edges (directed)' if led.directed else 'edges (undirected)'}"
    )
    ax.set_title(head, fontsize=12, color=PALETTE["accent"], loc="left", pad=14)
    top = len(led.rows) - 0.42
    hdr = dict(va="center", fontsize=8, color=PALETTE["ink_faint"], family="monospace")
    ax.text(col_bits, top, "info-theoretic bits", ha="right", **hdr)
    if show_redundancy:
        ax.text(col_red, top, "redundancy", ha="left", **hdr)
    ax.text(col_prev, top, "vs previous", ha="right", **hdr)
    ax.text(col_base, top, "vs (0)", ha="right", **hdr)

    n_roles = max(len(r.components) for r in priced)
    handles = [Patch(facecolor=COMPONENT_COLOURS[i], label=ROLE_LABELS[i])
               for i in range(min(n_roles, len(ROLE_LABELS)))]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, -0.14),
              ncol=2, frameon=False, fontsize=8.5, labelcolor=PALETTE["ink"])

    caption = ("the o(.) redundancy of the rank/select structures is quoted but left out of "
               "every total; on small graphs it would swamp them")
    if skipped:
        caption = "; ".join(f"({r.key}) {r.formula}" for r in skipped) + " - " + caption
    fig.text(0.012, 0.008, caption, fontsize=7.5, color=PALETTE["ink_faint"])
    # fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig, ax


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------


def read_graph(path: str, undirected: bool):
    creator = nx.Graph() if undirected else nx.DiGraph()
    return nx.read_edgelist(path, create_using=creator, comments="#")


def run(path: str, undirected: bool, outdir: str, absolute: bool,
        with_twins: bool, bounds: bool, show: bool, theme: str = DEFAULT_THEME):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    use_theme(theme)

    name = os.path.splitext(os.path.basename(path))[0]
    G = read_graph(path, undirected)
    led = compression_ledger(G, name=name, with_twins=with_twins)

    df = led.frame()
    os.makedirs(outdir, exist_ok=True)
    csv_path = os.path.join(outdir, f"{name}-ledger.csv")
    df.to_csv(csv_path, index=False)

    with pd.option_context("display.width", 200, "display.max_columns", 20,
                           "display.float_format", lambda v: f"{v:,.1f}"):
        print(df[["step", "name", "bits", "bits per edge", "redundancy",
                  "saved vs previous (%)", "saved vs (0) (%)"]].to_string(index=False))

    if bounds:
        print()
        for k, v in compression_bounds(led, G, exact_density=True).items():
            print(f"  {k:<34} {v if v is None else f'{v:,.2f}'}")

    fig, _ = plot_ledger(led, absolute=absolute, theme=theme)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(outdir, f"{name}-ledger.{ext}"), dpi=200,
                    facecolor=fig.get_facecolor())
    print(f"\nwritten: {csv_path}, {outdir}/{name}-ledger.pdf")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return led, df


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("graph", nargs="?", help="edgelist file")
    p.add_argument("--directory", help="run over every edgelist in a directory")
    p.add_argument("--undirected", action="store_true")
    p.add_argument("--outdir", default="figures")
    p.add_argument("--relative", action="store_true",
                   help="normalise every bar to full width instead of a shared bits scale")
    p.add_argument("--theme", default=DEFAULT_THEME, choices=sorted(THEMES),
                   help="web = companion-website colours, paper = neutral and "
                        "colourblind-safe, mono = greyscale for print")
    p.add_argument("--no-twins", action="store_true", help="skip the twin-removal row")
    p.add_argument("--bounds", action="store_true",
                   help="also report the Theorem 1.6/1.7 guarantees (needs the densest-subgraph peeling)")
    p.add_argument("--no-show", action="store_true")
    a = p.parse_args()

    if bool(a.graph) == bool(a.directory):
        p.error("give exactly one of a graph file or --directory")

    frames = []
    paths = ([a.graph] if a.graph else
             sorted(os.path.join(a.directory, f) for f in os.listdir(a.directory)
                    if not f.startswith(".")))
    for path in paths:
        print(f"\n=== {path}")
        _led, df = run(path, a.undirected, a.outdir, not a.relative,
                       not a.no_twins, a.bounds, not a.no_show, a.theme)
        frames.append(df)

    if len(frames) > 1:
        merged = os.path.join(a.outdir, "ledger-all.csv")
        pd.concat(frames, ignore_index=True).to_csv(merged, index=False)
        print(f"\nwritten: {merged}")


if __name__ == "__main__":
    main()
