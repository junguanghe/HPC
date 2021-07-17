#!/bin/bash
#PBS -N stokes
#PBS -m e
#PBS -M jhk1997@u.northwestern.edu
#PBS -l walltime=00:05:00
#PBS -q batch
cd $PBS_O_WORKDIR
./stokes 128 1 1 0.4 1e-9 100000