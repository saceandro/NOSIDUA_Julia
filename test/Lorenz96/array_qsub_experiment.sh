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

mkdir -p outdir
./builddir/experiment_arguments --dir $dir --true-params $true_params --initial-lower-bounds $initial_lower_bounds --initial-upper-bounds $initial_upper_bounds --obs-variance $obs_variance --obs-iteration $obs_iteration --dt $dt --spinup $spinup --duration $duration --generation-seed $generation_seed --trials $trials --replicates $replicates --iter $iter 1> $outfile 2> $errfile
