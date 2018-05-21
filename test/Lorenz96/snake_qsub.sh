#usage: qsub snake_qsub.sh <target>
#$ -cwd
#$ -e /home/konta/.ugeerr
#$ -o /home/konta/.ugeout
#$ -S /usr/local/bin/zsh
##$ -m es -M konta@hgc.jp
#$ -V
#$ -l s_vmem=1G,mem_req=1G
#$ -r no
snakemake --verbose --cluster "qsub -l s_vmem={params.mem},mem_req={params.mem} -cwd -V -o $HOME/.snakeout -e $HOME/.snakeerr -r no" --jobs 730 $1
