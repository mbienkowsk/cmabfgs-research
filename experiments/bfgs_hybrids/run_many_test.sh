#!/bin/bash

CONFIGURATIONS=(
  "10 10 1"
  "10 10 2"
  "10 10 5"
)

submit_job() {
  DIM=$1
  RUNS=$2
  SWITCH=$3

  export DIMENSIONS=$DIM
  export N_RUNS=$RUNS
  export SWITCH_AFTER=$SWITCH

  sbatch job_template.sh
}

# Main loop
JOB_COUNT=0
for CONFIG in "${CONFIGURATIONS[@]}"; do
  read -r DIM RUNS SWITCH <<< "$CONFIG"
  submit_job "$DIM" "$RUNS" "$SWITCH" &
done

# Wait for any remaining jobs
wait
