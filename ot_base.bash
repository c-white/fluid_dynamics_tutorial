#! /bin/bash

# Script for running base Orszag-Tang vortex with Athena++.

#SBATCH --partition temp
#SBATCH --nodes 1
#SBATCH --ntasks 4
#SBATCH --ntasks-per-node 4
#SBATCH --cpus-per-task 1
#SBATCH --time 00:05:00
#SBATCH --output tutorial_ot/ot_base.out

# Parameters
bin_name=tutorial_ot/athena
input_file=tutorial_ot/ot_base.athinput
data_dir=tutorial_ot

# Load modules
module load modules/2.2-alpha4 openmpi/4.0.7 hdf5/mpi-1.14.1-2
export LD_PRELOAD=/mnt/sw/fi/cephtweaks/lib/libcephtweaks.so
export CEPHTWEAKS_LAZYIO=1

# Run code
srun --cpus-per-task=$SLURM_CPUS_PER_TASK --cpu-bind=cores \
  $bin_name  -i $input_file -d $data_dir
