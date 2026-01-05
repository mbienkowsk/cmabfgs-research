#!/bin/bash -l

#SBATCH --ntasks=1
#SBATCH --mem=1GB
#SBATCH --cpus-per-task=25
#SBATCH --time=72:00:00
#SBATCH -A plgcmaes
#SBATCH -p plgrid
#SBATCH --output="logs/%A_%a.out"
#SBATCH --error="logs/%A_%a.err"
#SBATCH -J find_switch_interval_postprocessing
#SBATCH --chdir .

module load trytonp/python3/3.13.0

source /home/plgrid/plgmbienkowski/thesis/.venv/bin/activate  
which python3
python3 --version

DEBUG="" /home/plgrid/plgmbienkowski/thesis/.venv/bin/python3 -m experiments.find_switch_interval.cmabfgs.postprocessing
