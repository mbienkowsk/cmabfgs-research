#!/bin/bash -l

# Name of the job
#SBATCH -J placeholder

# One node, one task, one task per node
#SBATCH --ntasks=1

# Memory limit
#SBATCH --mem=1GB

# num of cpus per task, time limit HH:MM:SS
#SBATCH --cpus-per-task=25
#SBATCH --time=72:00:00
# Grant name
#SBATCH -A plgcmaes

# Partition (plgrid for 72 hours)
#SBATCH -p plgrid

# Output files
#SBATCH --output="out.log"
#SBATCH --error="err.log"

DEFAULT_DIMENSIONS=50
DEFAULT_N_RUNS=50

# Set job name based on arguments
: "${DIMENSIONS:=$DEFAULT_DIMENSIONS}"
: "${N_RUNS:=$DEFAULT_N_RUNS}"

echo "Running with dimensions: $DIMENSIONS and number of runs: $N_RUNS"

JOB_NAME="bfgs_hybrids_dim_${DIMENSIONS}_nruns_${N_RUNS}"
scontrol update JobName=$JOB_NAME JobId=$SLURM_JOB_ID

cd $SLURM_SUBMIT_DIR

# Load Python and activate your environment
module load ~/modules/3.13.2

echo "Creating venv"
python3 -m venv .venv
source .venv/bin/activate  

# Install dependencies
echo "Installing dependencies"
pip install --upgrade pip
pip install -r requirements.txt

# Run the Python script
echo "Running the bfgs_hybrids"
python3 -m experiments.all_hybrids --dimensions $DIMENSIONS --n_runs $N_RUNS
