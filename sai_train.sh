qsub -l nodes=X:ppn=Y:gpus=Z:FEATURE1:FEATURE2,mem=A,walltime=T,other=R -q QUEUE SCRIPTNAME


#!/bin/bash
#PBS -N ClusterHelloWorld
#PBS -S /bin/bash
#PBS -l nodes=1:ppn=1,mem=100mb,nice=10,walltime=00:01:00
#PBS -m a
#PBS -M <user>@informatik.uni-freiburg.de
#PBS -j oe
sleep 10
echo "Hello World"
exit 0