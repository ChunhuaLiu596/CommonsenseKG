#!/usr/bin/env bash

gammas=(1.0 0.9) 
betas=(1.0 0.9 0.8 0.7 0.6 0.5 0.4) 
maximum_count=5
count=0

for gamma in ${gammas[@]};do
    for beta in ${betas[@]}; do
        let count++
        echo "gamma:${gamma}, beta:${beta}, count:${count}"
        #if [$count -eq 5]; then
        if [[ "$count" -eq "$maximum_count" ]]; then
            wait
            count=0
        fi
    done
done 