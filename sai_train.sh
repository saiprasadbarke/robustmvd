#!/bin/bash
#PBS -N sai_test_mvsnet
#PBS -S /bin/bash
#PBS -l nodes=1:ppn=1:gpus=1:nvidiaP100:exclusive_process,mem=50gb,nice=1,walltime=23:59:59
#PBS -m a
#PBS -M <user>@informatik.uni-freiburg.de
#PBS -j oe
sleep 10
echo "Hello World"
exit 0


qsub -l nodes=1:ppn=1:gpus=1:nvidiaP100:exclusive_process,mem=50gb,nice=1,walltime=23:59:59 -q student -I
