#!/usr/bin/env zsh
#$ -cwd
#$ -e /home/konta/bitbucket/androsace/adjoint_julia/test/Lorenz96/log/experiment/err
#$ -o /home/konta/bitbucket/androsace/adjoint_julia/test/Lorenz96/log/experiment/out
#$ -S /usr/local/bin/zsh
##$ -m es -M konta@hgc.jp
#$ -V
#$ -l s_vmem=2G,mem_req=2G
#$ -r no

remainder=$(($SGE_TASK_ID - 1))

iter_array=(${(@s/ /)iter})
iter_index=$((1 + remainder % ${#iter_array}))
remainder=$((remainder / ${#iter_array}))

replicates_array=(${(@s/ /)replicates})
replicates_index=$((1 + remainder % ${#replicates_array}))
remainder=$((remainder / ${#replicates_array}))

trials_array=(${(@s/ /)trials})
trials_index=$((1 + remainder % ${#trials_array}))
remainder=$((remainder / ${#trials_array}))

generation_seed_array=(${(@s/ /)generation_seed})
generation_seed_index=$((1 + remainder % ${#generation_seed_array}))
remainder=$((remainder / ${#generation_seed_array}))

duration_array=(${(@s/ /)duration})
duration_index=$((1 + remainder % ${#duration_array}))
remainder=$((remainder / ${#duration_array}))

spinup_array=(${(@s/ /)spinup})
spinup_index=$((1 + remainder % ${#spinup_array}))
remainder=$((remainder / ${#spinup_array}))

dt_array=(${(@s/ /)dt})
dt_index=$((1 + remainder % ${#dt_array}))
remainder=$((remainder / ${#dt_array}))

obs_iteration_array=(${(@s/ /)obs_iteration})
obs_iteration_index=$((1 + remainder % ${#obs_iteration_array}))
remainder=$((remainder / ${#obs_iteration_array}))

obs_variance_array=(${(@s/ /)obs_variance})
obs_variance_index=$((1 + remainder % ${#obs_variance_array}))

# outdir=$1
# outfile=${outdir}/RMSE
# errfile=${outdir}/log

# mkdir -p $outdir
echo "./builddir/experiment_arguments --dir $dir --true-params ${=true_params[@]} --initial-lower-bounds ${=initial_lower_bounds[@]} --initial-upper-bounds ${=initial_upper_bounds[@]} --obs-variance ${obs_variance_array[$obs_variance_index]} --obs-iteration ${obs_iteration_array[$obs_iteration_index]} --dt ${dt_array[$dt_index]} --spinup ${spinup_array[$spinup_index]} --duration ${duration_array[$duration_index]} --generation-seed ${generation_seed_array[$generation_seed_index]} --trials ${trials_array[$trials_index]} --replicates ${replicates_array[$replicates_index]} --iter ${iter_array[$iter_index]}"
./builddir/experiment_arguments --dir $dir --true-params ${=true_params[@]} --initial-lower-bounds ${=initial_lower_bounds[@]} --initial-upper-bounds ${=initial_upper_bounds[@]} --obs-variance ${obs_variance_array[$obs_variance_index]} --obs-iteration ${obs_iteration_array[$obs_iteration_index]} --dt ${dt_array[$dt_index]} --spinup ${spinup_array[$spinup_index]} --duration ${duration_array[$duration_index]} --generation-seed ${generation_seed_array[$generation_seed_index]} --trials ${trials_array[$trials_index]} --replicates ${replicates_array[$replicates_index]} --iter ${iter_array[$iter_index]}
