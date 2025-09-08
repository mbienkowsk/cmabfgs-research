#!/bin/bash -l

#SBATCH --array=0-$(($(wc -l < config.txt)-1))%10
#SBATCH --ntasks=1
#SBATCH --mem=1GB
#SBATCH --cpus-per-task=25
#SBATCH --time=72:00:00
#SBATCH -A plgcmaes
#SBATCH -p plgrid
#SBATCH --output="logs/%A_%a.out"
#SBATCH --error="logs/%A_%a.err"
#SBATCH -J bfgs_hybrids

# Read parameters for this array task
read OBJECTIVE DIMENSIONS N_RUNS SWITCH_AFTER < <(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" experiments/find_switch_condition/config.txt)

echo "Running on $OBJECTIVE with dimensions: $DIMENSIONS, runs: $N_RUNS, switch_after: $SWITCH_AFTER"

cd $SLURM_SUBMIT_DIR

# Load Python
module load ~/modules/3.13.2

# Create venv
python3 -m venv .venv
source .venv/bin/activate  

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Run experiment
python3 -m experiments.find_switch_condition.main
