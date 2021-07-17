#!/bin/bash
#SBATCH -A e30514
#SBATCH -p short
#SBATCH --job-name="cgl"
#SBATCH -n 8
#SBATCH -t 00:05:00
#SBATCH --mem=1G
#SBATCH --mail-user=junguanghe2023@u.northwestern.edu
#SBATCH --mail-type=END,FAIL

cd $SLURM_SUBMIT_DIR
module load mpi/openmpi-1.10.5-gcc-4.8.3
module load fftw/3.3.3-gcc
mpirun -np 8 cgl 128 1.5 0.25 100000 12345