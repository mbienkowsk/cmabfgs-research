#!/bin/bash

CONFIGURATIONS=(
  "10 50 1"
  "10 50 2"
  "10 50 5"
  "10 50 10"
  "10 50 20"
  "20 50 2"
  "20 50 4"
  "20 50 10"
  "20 50 20"
  "20 50 40"
  "20 50 100"
  "10 50 50"
  "50 50 5"
  "50 50 10"
  "50 50 50"
  "50 50 100"
  "50 50 200"
  "100 50 25"
  "100 50 50"
  "100 50 100"
  "100 50 200"
  "100 50 500"
)

submit_job() {
  DIM=$1
  RUNS=$2
  SWITCH=$3

  export DIMENSIONS=$DIM
  export N_RUNS=$RUNS
  export SWITCH_AFTER=$SWITCH

  sbatch experiments/bfgs_hybrids/run.sbatch
}

# Main loop
JOB_COUNT=0
for CONFIG in "${CONFIGURATIONS[@]}"; do
  read -r DIM RUNS SWITCH <<< "$CONFIG"
  submit_job "$DIM" "$RUNS" "$SWITCH" &
done

# Wait for any remaining jobs
wait
