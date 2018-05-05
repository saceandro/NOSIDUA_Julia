#$ -cwd
#$ -e /home/konta/bitbucket/androsace/adjoint_julia/test/Lorenz96/log/replicate/err
#$ -o /home/konta/bitbucket/androsace/adjoint_julia/test/Lorenz96/log/replicate/out
#$ -S /usr/local/bin/zsh
##$ -m es -M konta@hgc.jp
#$ -V
#$ -l s_vmem=2G,mem_req=2G
#$ -r no

taskid_1=$(($SGE_TASK_ID - 1))
replicate=$((2 ** ($taskid_1 / 50)))
iter=$(($taskid_1 % 50 + 1))
outdir=result/replicate_${replicate}/iter_${iter}
outfile=${outdir}/RMSE
errdir=log/replicate/replicate_${replicate}
errfile=${errdir}/iter_${iter}

mkdir -p outdir
mkdir -p errdir
echo "./builddir/replicate_ccompile_each $replicate $iter 1> $outfile 2> $errfile"
./builddir/replicate_ccompile_each $replicate $iter 1> $outfile 2> $errfile
