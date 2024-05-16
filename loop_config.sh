#!/bin/bash

alignment_sampler=( "lower" "midpoint" "bracket" )
seeds=( 1 2 3 4 )
layers=( 10 15 20 25 30 )

for seed in ${seeds[*]};
do
        for layer in ${layers[*]};
        do
                for al in ${alignment_sampler[*]};
                do
                        sbatch /home/qinjerem/scratch/IFT6168/sbatch_run.sh '{"model": "mistral", "seed": '"$seed"', "layer": '"$layer"', "alignment_sampler": "'$al'"}'
                done
        done
done 