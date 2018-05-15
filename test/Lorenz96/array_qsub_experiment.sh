#$ -cwd
#$ -e /home/konta/bitbucket/androsace/adjoint_julia/test/Lorenz96/log/experiment/err
#$ -o /home/konta/bitbucket/androsace/adjoint_julia/test/Lorenz96/log/experiment/out
#$ -S /usr/local/bin/zsh
##$ -m es -M konta@hgc.jp
#$ -V
#$ -l s_vmem=2G,mem_req=2G
#$ -r no

iter=$SGE_TASK_ID
outdir=$1/iter_$iter
outfile=${outdir}/RMSE
errfile=${outdir}/log

true_params=(${true_params//_/ })
initial_lower_bounds=(${initial_lower_bounds//_/ })
initial_upper_bounds=(${initial_upper_bounds//_/ })

# A subscript of the form ‘[*]’ or ‘[@]’ evaluates to all elements of an array; there is no difference between the two except when they appear within double quotes. ‘"$foo[*]"’ evaluates to ‘"$foo[1] $foo[2] ..."’, whereas ‘"$foo[@]"’ evaluates to ‘"$foo[1]" "$foo[2]" ...’.
# ${=var} means compulsory elementwise expansion

# true_params=`eval echo $true_params | awk '{split($0,a,"_"); for (i=1; i<length(a); i++) printf "%s " ,a[i]; printf "%s", a[length(a)]}'`
# true_params="${true_params%\"}"
# true_params="${true_params#\"}"
# initial_lower_bounds=`eval echo $initial_lower_bounds | awk '{split($0,a,"_"); for (i=1; i<length(a); i++) printf "%s " ,a[i]; printf "%s", a[length(a)]}'`
# initial_lower_bounds="${initial_lower_bounds%\"}"
# initial_lower_bounds="${initial_lower_bounds#\"}"
# initial_upper_bounds=`eval echo ${initial_upper_bounds} | awk '{split($0,a,"_"); for (i=1; i<length(a); i++) printf "%s " ,a[i]; printf "%s", a[length(a)]}'`
# initial_upper_bounds="${initial_upper_bounds%\"}"
# initial_upper_bounds="${initial_upper_bounds#\"}"

mkdir -p $outdir
echo "./builddir/experiment_arguments --dir $dir --true-params ${=true_params[@]} --initial-lower-bounds ${=initial_lower_bounds[@]} --initial-upper-bounds ${=initial_upper_bounds[@]} --obs-variance $obs_variance --obs-iteration $obs_iteration --dt $dt --spinup $spinup --duration $duration --generation-seed $generation_seed --trials $trials --replicates $replicates --iter $iter 1> $outfile 2> $errfile"
./builddir/experiment_arguments --dir $dir --true-params ${=true_params[@]} --initial-lower-bounds ${=initial_lower_bounds[@]} --initial-upper-bounds ${=initial_upper_bounds[@]} --obs-variance $obs_variance --obs-iteration $obs_iteration --dt $dt --spinup $spinup --duration $duration --generation-seed $generation_seed --trials $trials --replicates $replicates --iter $iter 1> $outfile 2> $errfile
