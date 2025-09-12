#!/bin/bash -l

#SBATCH --ntasks=1
#SBATCH --mem=1GB
#SBATCH --cpus-per-task=25
#SBATCH --time=72:00:00
#SBATCH -A plgcmaes
#SBATCH -p plgrid
#SBATCH --output="logs/%A_%a.out"
#SBATCH --error="logs/%A_%a.err"
#SBATCH -J switch_grid_search
#SBATCH --chdir .

# Read parameters for this array task
read OBJECTIVE DIMENSIONS N_RUNS SWITCH_AFTER < <(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" experiments/switch_grid_search/config.txt)
export OBJECTIVE DIMENSIONS N_RUNS SWITCH_AFTER

echo "Running on $OBJECTIVE with dimensions: $DIMENSIONS, runs: $N_RUNS, switch_after: $SWITCH_AFTER"

# Load Python
module load trytonp/python3/3.13.0

source /home/plgrid/plgmbienkowski/thesis/.venv/bin/activate  
which python3
python3 --version

# Run experiment
/home/plgrid/plgmbienkowski/thesis/.venv/bin/python3 -m experiments.switch_grid_search.switch_grid_search
