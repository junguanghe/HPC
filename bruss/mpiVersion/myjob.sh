#!/bin/bash
#SBATCH -A e30514
#SBATCH -p short
#SBATCH --job-name="bruss"
#SBATCH -n 8
#SBATCH -t 00:01:00
#SBATCH --mem=1G
#SBATCH --mail-user=junguanghe2023@u.northwestern.edu
#SBATCH --mail-type=END,FAIL

cd $SLURM_SUBMIT_DIR
module load mpi
module load blas-lapack
mpirun -np 8 bruss 128 5.0e-5 5.0e-6 1 3 10000 12345