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
    OPTIONS+=(--metrics "n,m,type,array total bits,bitvector total bits,total bits trex,trex entropy,bitvector total bits,total bits planar,planar edges")
    OPTIONS+=(--output "results-$graph.csv")
    python -m evaluation.pipeline --single-graph "Datasets/$graph" "${OPTIONS[@]}"
done
