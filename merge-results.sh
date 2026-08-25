#! /bin/bash
# Concatenate all results*.csv files into a single merged.csv
# The first file contributes its header; subsequent files skip their header row.

OUTPUT="merged.csv"

first=1
for f in results*.csv; do
    if [[ $first -eq 1 ]]; then
        cat "$f" > "$OUTPUT"
        first=0
    else
        tail -n +2 "$f" >> "$OUTPUT"
    fi
done

echo "Merged $(ls results*.csv | wc -l) files into $OUTPUT"
