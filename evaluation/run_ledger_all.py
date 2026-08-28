"""Run the compression ledger over every graph in Datasets/ and merge the result.

This is the ledger counterpart of ``run-all.sh`` + ``merge-results.sh``: it walks
the same graph list, prices every representation step for each one, and writes a
single ``ledger-all.csv`` that the last cell of ``evaluation.ipynb`` picks up.

By default the graph list is read from ``Datasets/all-graphs`` -- the same
manifest ``run-all.sh`` uses -- so the ledger covers exactly the 21 graphs that
``merged.csv`` covers, and the two can be joined on the graph name.  Pass
``--walk`` to price every ``.txt`` under ``Datasets/`` instead.

Directedness follows the convention of ``run-all.sh``: a path containing
"Undirected" is read as an undirected graph, everything else as a digraph.

The run is resumable.  Graphs already present in ``ledger-all.csv`` are skipped
unless ``--force`` is given, and a graph that fails is reported and skipped
rather than taking the whole batch down with it.

Usage
-----
    python -m evaluation.run_ledger_all
    python -m evaluation.run_ledger_all --outdir figures --max-edges 2000000
    python -m evaluation.run_ledger_all --walk --no-figures
    python -m evaluation.run_ledger_all --only roadNet-PA --force

Expect a few minutes for the whole set; the spanning forest and the twin classes
dominate, and Amazon0302 alone takes about a minute.  ``--no-figures`` skips the
21 per-graph plots when only the merged table is wanted.
"""

from __future__ import annotations

import argparse
import collections
import os
import sys
import time
import traceback

import pandas as pd

from evaluation.ledger import (THEMES, compression_ledger, ledger_from_frame,
                               plot_ledger, read_graph, use_theme)

DEFAULT_MANIFEST = os.path.join("Datasets", "all-graphs")
DEFAULT_ROOT = "Datasets"
MERGED = "ledger-all.csv"

# directories that hold archive leftovers rather than graphs
SKIP_DIRS = {"__MACOSX", "all-graphs"}


DEFAULT_TYPES = "merged.csv"


class UnknownType(Exception):
    """Raised when a graph's directedness cannot be established from a record."""


def graphs_from_manifest(manifest: str, root: str) -> list[tuple]:
    """Read the graph list.

    Each line is a path relative to ``root``, optionally followed by a type:

        Directed/Wiki-Vote.txt       directed    real       benchmark
        Synthetic/G_erdos_renyi.txt  undirected  synthetic  benchmark

    Fields after the type (origin, role) follow thesis.pdf section 4.2 and are
    carried through to the output so results can be grouped by them.  A bare
    path (the original format) leaves everything unset, to be resolved from the
    types table.  Returns ``(path, declared_type, origin, role)`` tuples.
    """
    out = []
    with open(manifest) as f:
        for ln in f:
            ln = ln.split("#", 1)[0].strip()
            if not ln:
                continue
            parts = ln.replace(",", " ").split()
            name = parts[0]
            declared = parts[1].lower() if len(parts) > 1 else None
            if declared not in (None, "directed", "undirected"):
                raise ValueError(f"{manifest}: bad type {parts[1]!r} for {name}")
            origin = parts[2].lower() if len(parts) > 2 else None
            role = parts[3].lower() if len(parts) > 3 else None
            out.append((os.path.join(root, name), declared, origin, role))
    return out


def graphs_from_walk(root: str) -> list[tuple[str, None, None, None]]:
    found = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fn in sorted(filenames):
            if fn.endswith(".txt") and fn != "all-graphs":
                found.append((os.path.join(dirpath, fn), None, None, None))
    return sorted(found)


def load_types(path: str) -> dict[str, str]:
    """The repo's record of how each graph was evaluated: the ``type`` column of
    ``merged.csv``, keyed by dataset filename."""
    if not path or not os.path.exists(path):
        return {}
    try:
        df = pd.read_csv(path)
    except Exception:                       # empty or malformed: fall through
        return {}
    if not {"Dataset", "type"} <= set(df.columns):
        return {}
    return {str(d): str(t).strip().lower()
            for d, t in zip(df["Dataset"], df["type"]) if pd.notna(t)}


def resolve_type(path: str, declared: str | None, types: dict[str, str],
                 override: str | None) -> tuple[bool, str]:
    """Decide whether ``path`` is undirected, and say where the decision came from.

    Precedence, most explicit first:

    1. ``--directed`` / ``--undirected`` on the command line
    2. the second field of the manifest line
    3. the ``type`` column of the types table (``merged.csv`` by default)

    There is deliberately no fallback to guessing from the directory name.  That
    convention -- ``run-all.sh`` keying on the string "Undirected" in the path --
    is what caused ``Datasets/Synthetic/`` to be priced as digraphs even though
    ``generator.py`` builds those three graphs undirected.  A graph with no
    recorded type is an error, not a guess.
    """
    if override:
        return override == "undirected", "command line"
    if declared:
        return declared == "undirected", "manifest"
    recorded = types.get(os.path.basename(path))
    if recorded in ("directed", "undirected"):
        return recorded == "undirected", "merged.csv"
    raise UnknownType(
        f"no recorded type for {os.path.basename(path)}: add a second field in the "
        f"manifest, list it in the types table, or pass --directed/--undirected"
    )


def reciprocity_warning(G, undirected: bool, name: str) -> str | None:
    """Flag a digraph in which no arc is reciprocated.

    That is the fingerprint of an undirected edgelist read as a digraph -- how the
    Synthetic/ graphs went unnoticed.  It is only a hint: a DAG, a citation graph
    or a tree legitimately has no reciprocal arcs, so this warns and never
    overrides the recorded type.
    """
    if undirected or G.number_of_edges() == 0:
        return None
    recip = sum(1 for u, v in G.edges() if G.has_edge(v, u))
    if recip == 0:
        return (f"{name}: read as directed, but none of its {G.number_of_edges():,} "
                f"arcs is reciprocated -- check it is not an undirected edgelist")
    return None


def already_done(merged_path: str) -> set[str]:
    if not os.path.exists(merged_path):
        return set()
    try:
        return set(pd.read_csv(merged_path)["graph"].unique())
    except Exception:
        return set()


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--root", default=DEFAULT_ROOT, help="dataset root (default: Datasets)")
    p.add_argument("--manifest", default=DEFAULT_MANIFEST,
                   help="file listing the graphs to price, one relative path per line")
    p.add_argument("--walk", action="store_true",
                   help="price every .txt under --root instead of reading the manifest")
    p.add_argument("--types", default=DEFAULT_TYPES,
                   help="CSV with Dataset/type columns recording how each graph was "
                        "evaluated (default: merged.csv)")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--undirected", dest="override", action="store_const",
                   const="undirected", help="treat every graph as undirected")
    g.add_argument("--directed", dest="override", action="store_const",
                   const="directed", help="treat every graph as directed")
    p.set_defaults(override=None)
    p.add_argument("--outdir", default="figures",
                   help="where per-graph CSVs, figures and ledger-all.csv go")
    p.add_argument("--only", action="append", default=None,
                   help="restrict to graphs whose name contains this (repeatable)")
    p.add_argument("--max-edges", type=int, default=None,
                   help="skip graphs with more edges than this")
    p.add_argument("--no-figures", action="store_true", help="write only the tables")
    p.add_argument("--no-twins", action="store_true", help="skip the twin-removal row")
    p.add_argument("--theme", default="web", choices=sorted(THEMES),
                   help="web = companion-website colours, paper = neutral and "
                        "colourblind-safe, mono = greyscale for print")
    p.add_argument("--relative", action="store_true",
                   help="normalise every bar to full width instead of a shared bits scale")
    p.add_argument("--figures-only", action="store_true",
                   help="redraw the per-graph PDFs from the CSVs already in --outdir, "
                        "without pricing any graph again (use this to re-theme)")
    p.add_argument("--force", action="store_true",
                   help="re-price graphs already present in ledger-all.csv")
    a = p.parse_args()

    import matplotlib
    matplotlib.use("pgf")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "pgf.texsystem": "pdflatex",
        "pgf.rcfonts": False,              # stop the DejaVu preamble being emitted
        "pgf.preamble": r"\usepackage{amsmath}\usepackage{lmodern}",   # match your document
        "font.family": "serif",
        "font.size": 9,                    # your \normalsize, so sizes agree
    })


    use_theme(a.theme)

    if a.walk:
        entries = graphs_from_walk(a.root)
    else:
        if not os.path.exists(a.manifest):
            sys.exit(f"{a.manifest} not found -- pass --walk, or --manifest <file>")
        entries = graphs_from_manifest(a.manifest, a.root)

    missing = [e[0] for e in entries if not os.path.exists(e[0])]
    entries = [e for e in entries if os.path.exists(e[0])]
    if a.only:
        entries = [e for e in entries if any(k in e[0] for k in a.only)]

    types = load_types(a.types)
    resolved, unknown = [], []
    for q, declared, origin, role in entries:
        try:
            undirected, source = resolve_type(q, declared, types, a.override)
            resolved.append((q, undirected, source, origin, role))
        except UnknownType as exc:
            unknown.append(str(exc))
    if unknown:
        for msg in unknown:
            print(f"  ! {msg}", file=sys.stderr)
        sys.exit(f"{len(unknown)} graph(s) with no recorded type -- refusing to guess")
    paths = resolved

    os.makedirs(a.outdir, exist_ok=True)
    merged_path = os.path.join(a.outdir, MERGED)
    done = set() if a.force else already_done(merged_path)

    def draw(led, path_pdf):
        fig, _ = plot_ledger(led, absolute=not a.relative)
        fig.savefig(path_pdf, facecolor=fig.get_facecolor())
        # Save as pgf for LaTeX inclusion, but only if the backend is pgf (it may be Agg in batch mode)
        if matplotlib.get_backend() == "pgf":
            path_pgf = os.path.splitext(path_pdf)[0] + ".pgf"
            fig.savefig(path_pgf, facecolor=fig.get_facecolor())
        plt.close(fig)

    if a.figures_only:
        csvs = sorted(f for f in os.listdir(a.outdir) if f.endswith("-ledger.csv"))
        if not csvs:
            sys.exit(f"no *-ledger.csv in {a.outdir} -- run without --figures-only first")
        for j, fn in enumerate(csvs, 1):
            nm = fn[:-len("-ledger.csv")]
            led = ledger_from_frame(pd.read_csv(os.path.join(a.outdir, fn)))
            draw(led, os.path.join(a.outdir, f"{nm}-ledger.pdf"))
            print(f"[{j:2}/{len(csvs)}] {nm:32} redrawn from {fn}")
        print(f"\n{len(csvs)} figure(s) redrawn in theme {a.theme!r}; nothing recomputed")
        return

    frames, failed, skipped = [], [], []
    if done and not a.force:
        # keep what is already there, so the merge stays additive
        frames.append(pd.read_csv(merged_path))

    by_source = collections.Counter(e[2] for e in paths)
    n_undir = sum(1 for e in paths if e[1])
    print(f"{len(paths)} graph(s) to price -> {merged_path}")
    print(f"  {len(paths) - n_undir} directed, {n_undir} undirected; type from "
          + ", ".join(f"{v} x {k}" for k, v in by_source.most_common()))
    kinds = collections.Counter(f"{e[3] or '?'}/{e[4] or '?'}" for e in paths)
    print("  " + ", ".join(f"{v} x {k}" for k, v in sorted(kinds.items())))
    if missing:
        print(f"  ({len(missing)} listed but not on disk -- unpack Datasets/*.zip: "
              f"{', '.join(os.path.basename(q) for q in missing[:4])}"
              f"{' ...' if len(missing) > 4 else ''})")
    print()

    warnings = []
    for i, (path, undirected, source, origin, role) in enumerate(paths, 1):
        name = os.path.splitext(os.path.basename(path))[0]
        tag = "undirected" if undirected else "directed"

        pdf_path = os.path.join(a.outdir, f"{name}-ledger.pdf")
        csv_path = os.path.join(a.outdir, f"{name}-ledger.csv")

        if name in done:
            need_pdf = not a.no_figures and not os.path.exists(pdf_path)
            if need_pdf and os.path.exists(csv_path):
                # the metrics are already on disk -- draw from them, do not re-price
                draw(ledger_from_frame(pd.read_csv(csv_path)), pdf_path)
                print(f"[{i:2}/{len(paths)}] {name:32} figure redrawn from CSV "
                      f"(not recomputed)")
                skipped.append(name)
                continue
            if not need_pdf:
                print(f"[{i:2}/{len(paths)}] {name:32} already in {MERGED}, skipping")
                skipped.append(name)
                continue

        t0 = time.time()
        try:
            G = read_graph(path, undirected)
            warn = reciprocity_warning(G, undirected, name)
            if warn:
                warnings.append(f"{warn}  [type from {source}]")
            if a.max_edges is not None and G.number_of_edges() > a.max_edges:
                print(f"[{i:2}/{len(paths)}] {name:32} {G.number_of_edges():,} edges "
                      f"> --max-edges, skipping")
                skipped.append(name)
                continue

            led = compression_ledger(G, name=name, with_twins=not a.no_twins)
            frame = led.frame()
            frame.insert(1, "type", tag)
            frame.insert(2, "origin", origin or "")
            frame.insert(3, "role", role or "")
            frame.insert(4, "path", path)
            frame.to_csv(csv_path, index=False)
            frames.append(frame)

            if not a.no_figures:
                draw(led, pdf_path)

            best = led.row("3b").bits
            base = led.row("0").bits
            print(f"[{i:2}/{len(paths)}] {name:32} {tag:10} "
                  f"n={led.n:>9,} m={led.edges:>9,} "
                  f"(3b)={best:>14,.0f} bits  {-100 * (1 - best / base):+6.1f}% vs (0)  "
                  f"[{time.time() - t0:5.1f}s]")
        except Exception as exc:                      # keep the batch going
            failed.append((name, repr(exc)))
            print(f"[{i:2}/{len(paths)}] {name:32} FAILED: {exc}", file=sys.stderr)
            traceback.print_exc(limit=2, file=sys.stderr)

    if not frames:
        sys.exit("nothing priced -- no output written")

    merged = pd.concat(frames, ignore_index=True)
    merged = merged.drop_duplicates(subset=["graph", "step"], keep="last")
    merged.to_csv(merged_path, index=False)

    n_graphs = merged["graph"].nunique()
    print(f"\nwritten: {merged_path}  ({n_graphs} graphs, {len(merged)} rows)")
    if skipped:
        print(f"skipped: {len(skipped)} ({', '.join(skipped[:6])}"
              f"{' ...' if len(skipped) > 6 else ''})")
    if failed:
        print(f"failed:  {len(failed)}")
        for nm, err in failed:
            print(f"  {nm}: {err}")
    if warnings:
        print(f"\n{len(warnings)} directedness warning(s):")
        for w in warnings:
            print(f"  ! {w}")

    # a compact view of what merged.csv cannot show: the cost of the tree choice
    wide = merged.pivot_table(index="graph", columns="step", values="bits")
    if {"2", "3b", "3w"} <= set(wide.columns):
        gap = pd.DataFrame({
            "tree choice worth [% of (2)]": 100 * (wide["3w"] - wide["3b"]) / wide["2"],
        })
        if "4" in wide.columns:
            gap["twin removal worth [% of (3b)]"] = (
                100 * (wide["3b"] - wide["4"]) / wide["3b"])
        print()
        print(gap.round(3).sort_values(gap.columns[0], ascending=False).to_string())


if __name__ == "__main__":
    main()
