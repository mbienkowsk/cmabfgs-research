#!/bin/bash -l

#SBATCH --ntasks=1
#SBATCH --mem=1GB
#SBATCH --cpus-per-task=25
#SBATCH --time=72:00:00
#SBATCH -A plgcmaes
#SBATCH -p plgrid
#SBATCH --output="logs/%A_%a.out"
#SBATCH --error="logs/%A_%a.err"
#SBATCH -J experimentum_crucis
#SBATCH --chdir .

# Read parameters for this array task
read DIMENSIONS N_RUNS EVERY_N_CALLS < <(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" experiments/experimentum_crucis/config.txt)
export DIMENSIONS N_RUNS EVERY_N_CALLS

echo "Running with dimensions: $DIMENSIONS, runs: $N_RUNS. Collecting $EVERY_N_CALLS"

module load trytonp/python3/3.13.0

source /home/plgrid/plgmbienkowski/thesis/.venv/bin/activate  
which python3
python3 --version

/home/plgrid/plgmbienkowski/thesis/.venv/bin/python3 -m experiments.experimentum_crucis.experimentum_crucis
