#!/bin/bash

CONFIG_FILE=/home/plgrid/plgmbienkowski/thesis/experiments/find_switch_interval/cmabfgs/config.txt

MAX_CONCURRENT=10

NUM_LINES=$(wc -l < "$CONFIG_FILE")
ARRAY_END=$((NUM_LINES - 1))

sbatch --array=0-$ARRAY_END%$MAX_CONCURRENT /home/plgrid/plgmbienkowski/thesis/experiments/find_switch_interval/cmabfgs/run.sbatch
