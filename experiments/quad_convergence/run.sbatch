#!/bin/bash -l

#SBATCH --ntasks=1
#SBATCH --mem=1GB
#SBATCH --cpus-per-task=25
#SBATCH --time=72:00:00
#SBATCH -A plgcmaes
#SBATCH -p plgrid
#SBATCH --output="logs/%A_%a.out"
#SBATCH --error="logs/%A_%a.err"
#SBATCH -J quad_convergence
#SBATCH --chdir .

# Read parameters for this array task
read DIMENSIONS N_RUNS EVERY_N_CALLS < <(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" experiments/quad_convergence/config.txt)
export DIMENSIONS N_RUNS CMAES_COLLECTION_INTERVAL

echo "Running with dimensions: $DIMENSIONS, runs: $N_RUNS. Collecting every $CMAES_ACOLLECTION_INTERVAL iters"

module load trytonp/python3/3.13.0

source /home/plgrid/plgmbienkowski/thesis/.venv/bin/activate  
which python3
python3 --version

/home/plgrid/plgmbienkowski/thesis/.venv/bin/python3 -m experiments.quad_convergence.quad_convergence
