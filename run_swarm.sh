#!/bin/bash
# sbatch -J sweep_eps -o sweep_eps.out -e sweep_eps.err --nodes 6 --cpus-per-task 12 -p longq --mem 32000 run_swarm.sh

python compare_unified.py 
