#!/bin/sh
#PBS -N _JOBNAME_
#PBS -q _QUEUE_
#PBS -l nodes=_NODES_:ppn=_MPI_PER_NODE_
#PBS -l walltime=_MAX_TIME_

export OMP_NUM_THREADS=_OMP_NUM_THREADS_

INPUT=_INPUT_
OUTPUT=_OUTPUT_

mpirun -np _NUM_CORES_ jqmc ${INPUT} > ${OUTPUT} 2>&1
