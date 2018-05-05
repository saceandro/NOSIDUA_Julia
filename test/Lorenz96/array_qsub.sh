#$ -cwd
#$ -e /home/konta/.ugeerr
#$ -o /home/konta/.ugeout
#$ -S /usr/local/bin/zsh
##$ -m es -M konta@hgc.jp
#$ -V
#$ -l s_vmem=2G,mem_req=2G
#$ -r no

taskid_1=$(($SGE_TASK_ID - 1))
replicate=$(($taskid_1 / 50 + 1))
iter=$(($taskid_1 % 50 + 1))
outfile=result/replicate_${replicate}/iter_${iter}/RMSE
errfile=log/replicate/replicate_${replicate}/iter_${iter}

./builddir/replicate_ccompile_each $replicate $iter 1> $outfile 2> $errfile
