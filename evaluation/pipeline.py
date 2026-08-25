import pandas as pd
import networkx as nx
from src.structure.Builder import Builder
from evaluation.evaluator import Evaluator
import os
import time
import argparse
import numpy as np

# Names of every metric this pipeline can produce, grouped by where they come
# from. Used to validate --metrics selections and to decide which (possibly
# expensive) computations can be skipped entirely.
BASE_METRICS = ["type", "n", "m", "avg cc"]

GRAPH_METRICS = [
    "array total bits",
    "indegree entropy",
    "indegree entropy greedy",
    "bitvector total bits",
    "bitvector lexicographical total bits",
    "bitvector worst case total bits",
    "bitvector greedy total bits",
    "bitvector random total bits",
    "total bits trex",
    "trex entropy",
    "alpha 1.6",
    "DUB",
    "non zero indegree nodes",
    "NIB",
    "indegree coefficient of variation",
    "number of classes",
    "maximum class degree",
]

PLANAR_METRICS = ["total bits planar", "planar edges"]

TIMING_METRICS = ["trex time", "evaluation time"]

DERIVED_METRICS = [
    "trex vs bitvector (greedy) (%)",
    "trex vs array (%)",
    "planar vs trex (%)",
    "trex bpe",
    "planar bpe",
    "planar edges vs maximum",
    "DUB vs trex",
    "normalized DUB difference",
    "NIB vs trex",
    "normalized NIB difference",
    "alpha 1.7",
    "density",
]

ALL_METRICS = BASE_METRICS + GRAPH_METRICS + PLANAR_METRICS + TIMING_METRICS + DERIVED_METRICS

# Metrics that require the (possibly expensive) greedy density computation.
SUBGRAPH_DENSITY_DEPENDENT_METRICS = {
    "alpha 1.6", "DUB", "NIB",
    "DUB vs trex", "normalized DUB difference",
    "NIB vs trex", "normalized NIB difference",
}

# Metrics that require the planar extraction step.
PLANAR_DEPENDENT_METRICS = {
    "total bits planar", "planar edges",
    "planar vs trex (%)", "planar bpe", "planar edges vs maximum",
}


def resolve_metrics(metrics):
    """Validate and normalize a requested metrics selection.

    `metrics` may be None (meaning "all metrics") or a list of metric names.
    Returns the resolved list of metric names to keep in the final output.
    """
    if not metrics:
        return list(ALL_METRICS)

    unknown = [m for m in metrics if m not in ALL_METRICS]
    if unknown:
        raise ValueError(
            "Unknown metric(s): " + ", ".join(unknown) +
            ". Available metrics: " + ", ".join(ALL_METRICS)
        )
    return list(metrics)


def evaluate_file(path: str, filename: str, builder: Builder, undirected: bool,
                  extraction_strategy: str, planar: bool, skip_subgraph_density: bool):
    """Build and evaluate trex on a single edgelist file, returning its metrics dict."""
    print("\n Evaluating " + filename)

    if undirected:
        G = nx.read_edgelist(path, create_using=nx.Graph(), comments='#')
        start = time.time()
        G_built, G_minus_T, G_greedy = builder.build(G, extraction_strategy=extraction_strategy)
        trex_time = time.time() - start

        start = time.time()
        metrics = Evaluator.evaluate(G, G_minus_T, G_built, G_greedy, planar=planar, skip_greedy=skip_subgraph_density)
        evaluation_time = time.time() - start
        metrics["Dataset"] = filename
        if "bitvector greedy total bits" in metrics and metrics["bitvector greedy total bits"] not in (0, -1):
            metrics["trex vs bitvector (greedy) (%)"] = (1 - metrics["total bits trex"] / metrics["bitvector greedy total bits"]) * 100
        metrics["trex time"] = trex_time
        metrics["evaluation time"] = evaluation_time
    else:
        G = nx.read_edgelist(path, create_using=nx.DiGraph(), comments='#')
        start = time.time()
        G_built, G_minus_T = builder.build(G, extraction_strategy=extraction_strategy)
        trex_time = time.time() - start

        start = time.time()
        metrics = Evaluator.evaluate(G, G_minus_T, G_built, planar=planar, skip_greedy=skip_subgraph_density)
        evaluation_time = time.time() - start
        metrics["Dataset"] = filename
        if "bitvector total bits" in metrics and metrics["bitvector total bits"] not in (0, -1):
            metrics["trex vs bitvector (greedy) (%)"] = (1 - metrics["total bits trex"] / metrics["bitvector total bits"]) * 100
        metrics["trex time"] = trex_time
        metrics["evaluation time"] = evaluation_time

    print(metrics)
    return metrics


def compute_derived_metrics(df: pd.DataFrame, requested_metrics):
    """Add the derived (dataframe-level) metrics that were requested and whose
    source columns are available."""
    if "trex vs array (%)" in requested_metrics:
        df["trex vs array (%)"] = (1 - df["total bits trex"] / df["array total bits"]) * 100
    if "planar vs trex (%)" in requested_metrics and "total bits planar" in df.columns:
        df["planar vs trex (%)"] = (1 - df["total bits planar"] / df["total bits trex"]) * 100
    if "trex bpe" in requested_metrics:
        df["trex bpe"] = df["total bits trex"] / df["m"]
    if "planar bpe" in requested_metrics and "total bits planar" in df.columns:
        df["planar bpe"] = df["total bits planar"] / df["m"]
    if "planar edges vs maximum" in requested_metrics and "planar edges" in df.columns:
        df["planar edges vs maximum"] = 100 * df["planar edges"] / (np.floor(df["n"] * 1.5 - 1.5))

    if "DUB vs trex" in requested_metrics and "DUB" in df.columns:
        df["DUB vs trex"] = df["DUB"] - df["trex entropy"]
    if "normalized DUB difference" in requested_metrics and "DUB vs trex" in df.columns:
        df["normalized DUB difference"] = df["DUB vs trex"] / (df["n"] * np.log2(df["n"]))

    if "NIB vs trex" in requested_metrics and "NIB" in df.columns:
        df["NIB vs trex"] = df["NIB"] - df["trex entropy"]
    if "normalized NIB difference" in requested_metrics and "NIB vs trex" in df.columns:
        df["normalized NIB difference"] = df["NIB vs trex"] / (df["n"] * np.log2(df["n"]))

    if "alpha 1.7" in requested_metrics:
        df["alpha 1.7"] = df["n"] / df["non zero indegree nodes"]
    if "density" in requested_metrics:
        df["density"] = df["m"] / df["n"]

    return df


def finalize(results_as_dict, output_path, requested_metrics):
    df = pd.DataFrame(results_as_dict)
    df = compute_derived_metrics(df, requested_metrics)

    # Always keep "Dataset" for identification, then the requested metrics
    # that actually ended up in the dataframe.
    columns = [c for c in ["Dataset"] + requested_metrics if c in df.columns]
    df = df[columns]

    df.to_csv(output_path)
    return df


def trex_on_directory(directory: str, output_path="trex_results.csv", undirected=False,
                       extraction_strategy="greedy", metrics=None):
    requested_metrics = resolve_metrics(metrics)
    planar = bool(PLANAR_DEPENDENT_METRICS.intersection(requested_metrics))
    skip_subgraph_density = not SUBGRAPH_DENSITY_DEPENDENT_METRICS.intersection(requested_metrics)

    results_as_dict = []
    builder = Builder()

    for filename in os.listdir(directory):
        # in case of .DS_store or other
        if filename.startswith('.'):
            continue
        path = directory + "/" + filename
        metrics_dict = evaluate_file(path, filename, builder, undirected, extraction_strategy, planar, skip_subgraph_density)
        results_as_dict.append(metrics_dict)

    return finalize(results_as_dict, output_path, requested_metrics)


def trex_on_single_graph(graph_path: str, output_path="trex_results.csv", undirected=False,
                          extraction_strategy="greedy", metrics=None):
    requested_metrics = resolve_metrics(metrics)
    planar = bool(PLANAR_DEPENDENT_METRICS.intersection(requested_metrics))
    skip_subgraph_density = not SUBGRAPH_DENSITY_DEPENDENT_METRICS.intersection(requested_metrics)

    builder = Builder()
    filename = os.path.basename(graph_path)
    metrics_dict = evaluate_file(graph_path, filename, builder, undirected, extraction_strategy, planar, skip_subgraph_density)

    return finalize([metrics_dict], output_path, requested_metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", nargs="?", default=None,
                         help="Directory of edgelist files to evaluate. Mutually exclusive with --single-graph.")
    parser.add_argument("--single-graph", default=None,
                         help="Path to a single edgelist file to evaluate instead of a whole directory.")
    parser.add_argument("--undirected", action="store_true")
    parser.add_argument("--output", default="trex_results.csv")
    parser.add_argument("--antiGreedy", action="store_true")
    parser.add_argument("--random", action="store_true")
    parser.add_argument("--metrics", default=None,
                         help="Comma-separated list of metrics to compute. Defaults to all. "
                              "Available: " + ", ".join(ALL_METRICS).replace("%", "%%"))

    args = parser.parse_args()

    if bool(args.directory) == bool(args.single_graph):
        parser.error("Provide exactly one of 'directory' or --single-graph.")

    extraction_strategy = "greedy"
    if args.antiGreedy:
        extraction_strategy = "anti-greedy"
    if args.random:
        extraction_strategy = "random"

    requested_metrics = args.metrics.split(",") if args.metrics else None
    if requested_metrics is not None:
        requested_metrics = [m.strip() for m in requested_metrics]

    if args.single_graph:
        df = trex_on_single_graph(args.single_graph, undirected=args.undirected,
                                   extraction_strategy=extraction_strategy, output_path=args.output,
                                   metrics=requested_metrics)
    else:
        df = trex_on_directory(args.directory, undirected=args.undirected,
                                extraction_strategy=extraction_strategy, output_path=args.output,
                                metrics=requested_metrics)
