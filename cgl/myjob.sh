#!/bin/bash
#PBS -N cgl
#PBS -m e
#PBS -M jhk1997@u.northwestern.edu
#PBS -l walltime=00:05:00
#PBS -q batch
cd $PBS_O_WORKDIR
./cgl 128 1.5 0.25 100000 12345