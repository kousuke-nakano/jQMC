#!/bin/sh
#PBS -N _JOBNAME_
#PBS -q _QUEUE_
#PBS -l nodes=_NODES_
#PBS -l walltime=_MAX_TIME_

export OMP_NUM_THREADS=_OMP_NUM_THREADS_

INPUT=_INPUT_
OUTPUT=_OUTPUT_

jqmc ${INPUT} > ${OUTPUT} 2>&1
