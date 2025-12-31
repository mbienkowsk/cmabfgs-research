#!/bin/bash -l

#SBATCH --ntasks=1
#SBATCH --mem=1GB
#SBATCH --cpus-per-task=25
#SBATCH --time=72:00:00
#SBATCH -A plgcmaes
#SBATCH -p plgrid
#SBATCH --output="logs/%A_%a.out"
#SBATCH --error="logs/%A_%a.err"
#SBATCH -J find_switch_interval_cmaes
#SBATCH --chdir .

# Read parameters for this array task
read DIMENSIONS N_RUNS OBJECTIVE_CHOICE OPTIMUM_POSITION < <(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" experiments/find_switch_interval/cmaes/config.txt)
export DIMENSIONS N_RUNS OBJECTIVE_CHOICE OPTIMUM_POSITION

echo "Running with dimensions: $DIMENSIONS, runs: $N_RUNS, objective: $OBJECTIVE_CHOICE, optimum: $OPTIMUM_POSITION."

module load trytonp/python3/3.13.0

source /home/plgrid/plgmbienkowski/thesis/.venv/bin/activate  
which python3
python3 --version

/home/plgrid/plgmbienkowski/thesis/.venv/bin/python3 -m experiments.find_switch_interval.cmaes.experiment
