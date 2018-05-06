#!/usr/bin/env zsh

for ((i=0; i<=7; ++i)); do
    replicate=$((2 ** $i))
    for ((iter=1; iter<=50; ++iter)); do
        cp true_data/N_5/p_8.0_1.0/dt_0.01/spinup_73.0/T_1.0/seed_0.tsv true_data/N_5/p_8.0_1.0/dt_0.01/spinup_73.0/T_1.0/seed_0_forreplicate_${replicate}_foriter_${iter}.tsv
    done
done
