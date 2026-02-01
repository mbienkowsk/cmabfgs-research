#!/bin/bash

CONFIG_FILE=/home/plgrid/plgmbienkowski/thesis/experiments/find_switch_interval/cmabfgs/slurm/postprocessing_config.txt

MAX_CONCURRENT=10

NUM_LINES=$(wc -l < "$CONFIG_FILE")
# NOTE: -2 is due to a global flag at the top of the config
ARRAY_END=$((NUM_LINES - 2))

sbatch --array=0-$ARRAY_END%$MAX_CONCURRENT /home/plgrid/plgmbienkowski/thesis/experiments/find_switch_interval/cmabfgs/slurm/postprocessing_single.sbatch
