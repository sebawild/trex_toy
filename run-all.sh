#! /bin/bash
# Use venv
source .venv/bin/activate

# Iterate over the graphs stored in textfile Datasets/all-graphs
for graph in $(cat Datasets/all-graphs); do
    echo
    echo
    echo
    echo "Running for graph: $graph"
    # If the name contains "Undirected", we need to add the --undirected flag
    #python -m evaluation.pipeline --single-graph Datasets/fig2-example.txt  --metrics "n","m","type","array total bits","bitvector total bits","total bits trex","trex entropy","bitvector total bits","total bits planar","planar edges"
    OPTIONS=()
    if [[ $graph == *"Undirected"* ]]; then
        OPTIONS+=(--undirected)
    fi
    # Also undirected for the following graphs, even though they don't contain "Undirected" in their name
    if [[ $graph == *"G_erdos_renyi.txt"* ]]; then
        OPTIONS+=(--undirected)
    fi
    if [[ $graph == *"G_bipartite.txt"* ]]; then
        OPTIONS+=(--undirected)
    fi
    if [[ $graph == *"G_barabasi_albert.txt"* ]]; then
        OPTIONS+=(--undirected)
    fi
    OPTIONS+=(--metrics "n,m,type,indegree entropy,indegree entropy greedy,array total bits,bitvector total bits,bitvector greedy total bits,total bits trex,trex entropy,total bits planar,planar edges")
    # replace directory / with - in the output filename
    OPTIONS+=(--output "results-$(echo "$graph" | tr '/' '-').csv")
    echo "Running with options:" "${OPTIONS[@]}"
    python -m evaluation.pipeline --single-graph "Datasets/$graph" "${OPTIONS[@]}"
done
