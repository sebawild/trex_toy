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
import os
import sys
import time
import traceback

import pandas as pd

from evaluation.ledger import compression_ledger, plot_ledger, read_graph

DEFAULT_MANIFEST = os.path.join("Datasets", "all-graphs")
DEFAULT_ROOT = "Datasets"
MERGED = "ledger-all.csv"

# directories that hold archive leftovers rather than graphs
SKIP_DIRS = {"__MACOSX", "all-graphs"}


def is_undirected(path: str) -> bool:
    """The run-all.sh convention: 'Undirected' anywhere in the path."""
    return "undirected" in path.replace(os.sep, "/").lower()


def graphs_from_manifest(manifest: str, root: str) -> list[str]:
    with open(manifest) as f:
        names = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
    return [os.path.join(root, n) for n in names]


def graphs_from_walk(root: str) -> list[str]:
    found = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fn in sorted(filenames):
            if fn.endswith(".txt") and fn != "all-graphs":
                found.append(os.path.join(dirpath, fn))
    return sorted(found)


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
    p.add_argument("--outdir", default="figures",
                   help="where per-graph CSVs, figures and ledger-all.csv go")
    p.add_argument("--only", action="append", default=None,
                   help="restrict to graphs whose name contains this (repeatable)")
    p.add_argument("--max-edges", type=int, default=None,
                   help="skip graphs with more edges than this")
    p.add_argument("--no-figures", action="store_true", help="write only the tables")
    p.add_argument("--no-twins", action="store_true", help="skip the twin-removal row")
    p.add_argument("--relative", action="store_true",
                   help="normalise every bar to full width instead of a shared bits scale")
    p.add_argument("--force", action="store_true",
                   help="re-price graphs already present in ledger-all.csv")
    a = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if a.walk:
        paths = graphs_from_walk(a.root)
    else:
        if not os.path.exists(a.manifest):
            sys.exit(f"{a.manifest} not found -- pass --walk, or --manifest <file>")
        paths = graphs_from_manifest(a.manifest, a.root)

    missing = [q for q in paths if not os.path.exists(q)]
    paths = [q for q in paths if os.path.exists(q)]
    if a.only:
        paths = [q for q in paths if any(k in q for k in a.only)]

    os.makedirs(a.outdir, exist_ok=True)
    merged_path = os.path.join(a.outdir, MERGED)
    done = set() if a.force else already_done(merged_path)

    frames, failed, skipped = [], [], []
    if done and not a.force:
        # keep what is already there, so the merge stays additive
        frames.append(pd.read_csv(merged_path))

    print(f"{len(paths)} graph(s) to price -> {merged_path}")
    if missing:
        print(f"  ({len(missing)} listed but not on disk -- unpack Datasets/*.zip: "
              f"{', '.join(os.path.basename(q) for q in missing[:4])}"
              f"{' ...' if len(missing) > 4 else ''})")
    print()

    for i, path in enumerate(paths, 1):
        name = os.path.splitext(os.path.basename(path))[0]
        undirected = is_undirected(path)
        tag = "undirected" if undirected else "directed"

        if name in done:
            print(f"[{i:2}/{len(paths)}] {name:32} already in {MERGED}, skipping")
            skipped.append(name)
            continue

        t0 = time.time()
        try:
            G = read_graph(path, undirected)
            if a.max_edges is not None and G.number_of_edges() > a.max_edges:
                print(f"[{i:2}/{len(paths)}] {name:32} {G.number_of_edges():,} edges "
                      f"> --max-edges, skipping")
                skipped.append(name)
                continue

            led = compression_ledger(G, name=name, with_twins=not a.no_twins)
            frame = led.frame()
            frame.insert(1, "type", tag)
            frame.insert(2, "path", path)
            frame.to_csv(os.path.join(a.outdir, f"{name}-ledger.csv"), index=False)
            frames.append(frame)

            if not a.no_figures:
                fig, _ = plot_ledger(led, absolute=not a.relative)
                fig.savefig(os.path.join(a.outdir, f"{name}-ledger.pdf"),
                            facecolor=fig.get_facecolor())
                plt.close(fig)

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
