#!/bin/bash

# create output directory
mkdir -p by_dimension

# loop over all CEC functions
for func in {1..29}; do
    for suffix in "" "_agg"; do
        files=()
        # collect files for this function and suffix, sorted by dimension
        for dim in 10 30 50 100; do
            filename="CEC${func}_${dim}${suffix}.png"
            if [[ -f "$filename" ]]; then
                files+=("$filename")
            fi
        done

        # if we found any files, stack them vertically
        if [[ ${#files[@]} -gt 0 ]]; then
            output="by_dimension/CEC${func}${suffix}.png"
            magick convert "${files[@]}" -append "$output"
            echo "Created $output"
        fi
    done
done
