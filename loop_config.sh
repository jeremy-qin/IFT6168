#!/bin/bash

alignment_sampler=( "lower" "midpoint" "bracket" )
seeds=( 1 2 3 4 )
batch_sizes=( 16 32 64 128 )
layers=( 0 5 10 15 20 25 30 )

for seed in ${seeds[*]};
do
        for layer in ${layers[*]};
        do
                for b in ${batch_sizes[*]};
                do
                        for al in ${alignment_sampler[*]};
                        do
                                sbatch /home/qinjerem/scratch/IFT6168/sbatch_run.sh '{"seed": '"$seed"', "layer": '"$layer"', "batch_size": '"$b"', "alignment_sampler": '"$al"'}'
                        done
                done
        done
done 