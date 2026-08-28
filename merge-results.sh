#! /bin/bash
# Concatenate all results*.csv files into a single merged.csv
# The first file contributes its header; subsequent files skip their header row.

OUTPUT="merged.csv"
FILES=()
for f in results-*.csv; do
    FILES+=("$f")
done

first=1
for f in "${FILES[@]}"; do
    if [[ $first -eq 1 ]]; then
        cat "$f" > "$OUTPUT"
        first=0
    else
        tail -n +2 "$f" >> "$OUTPUT"
    fi
done

echo "Merged ${#FILES[@]} files into $OUTPUT"
