"""
Arguments:
path: "/home/seb/Uni/projects/2024-compressed-graphs/trex_toy/scripts/algs4_to_edgelist.py"
file_text: "Convert an algs4 ``Graph`` input file to this project's edge list format."

The algs4 format (see edu.princeton.cs.algs4.Graph) is a whitespace
separated stream of integers:

    V
    E
    v1 w1
    v2 w2
    ...
    vE wE

Any tokens after the E-th edge (e.g. vertex coordinates used for drawing)
are ignored. Tokens are read irrespective of line breaks, matching the
behaviour of Java's ``In``/``Scanner`` based reader used by algs4.

The output is a plain edge list, one "u v" pair per line, compatible with
``networkx.read_edgelist(path, comments='#')`` as used throughout this
project (see evaluation/pipeline.py).
"""

import argparse
import re


def parse_algs4_graph(path: str):
    """Parse an algs4 Graph file and return (V, E, edges)."""
    with open(path, "r") as f:
        tokens = re.findall(r"\S+", f.read())

    if len(tokens) < 2:
        raise ValueError(f"{path} does not contain a valid algs4 graph header")

    num_vertices = int(tokens[0])
    num_edges = int(tokens[1])

    required_tokens = 2 + 2 * num_edges
    if len(tokens) < required_tokens:
        raise ValueError(
            f"{path} declares {num_edges} edges but only has "
            f"{(len(tokens) - 2) // 2} edge token pairs"
        )

    edges = []
    for i in range(num_edges):
        v = int(tokens[2 + 2 * i])
        w = int(tokens[2 + 2 * i + 1])
        edges.append((v, w))

    return num_vertices, num_edges, edges


def convert(input_path: str, output_path: str):
    num_vertices, num_edges, edges = parse_algs4_graph(input_path)

    with open(output_path, "w") as f:
        f.write("# numeric_id_1 numeric_id_2\n")
        for v, w in edges:
            f.write(f"{v} {w}\n")

    return num_vertices, num_edges


def main():
    parser = argparse.ArgumentParser(
        description="Convert an algs4 Graph input file to an edge list file."
    )
    parser.add_argument("input", help="path to the algs4 formatted input file")
    parser.add_argument("output", help="path to write the edge list file to")
    args = parser.parse_args()

    num_vertices, num_edges = convert(args.input, args.output)
    print(f"Converted {args.input} -> {args.output} ({num_vertices} vertices, {num_edges} edges)")


if __name__ == "__main__":
    main()