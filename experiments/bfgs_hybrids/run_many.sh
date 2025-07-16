#!/bin/bash

CONFIGURATIONS=(
  "10 50 1"
  "10 50 2"
  "10 50 5"
  "10 50 10"
  "10 50 20"
  "100 50 25"
  "100 50 50"
  "100 50 100"
  "100 50 200"
  "100 50 500"
)

OBJECTIVES=(
  "Elliptic"
  "CEC1"
  "CEC2"
  "CEC3"
  "CEC4"
  "CEC5"
  "CEC6"
  "CEC7"
)

echo "Submitting job: DIMENSIONS=$DIM, N_RUNS=$RUNS, SWITCH_AFTER=$SWITCH, OBJECTIVE=$OBJ"
sbatch experiments/bfgs_hybrids/run.sbatch
export DIMENSIONS=$DIM
MAX_JOBS=10
job_count=0

for CONFIG in "${CONFIGURATIONS[@]}"; do
  read -r DIM RUNS SWITCH <<< "$CONFIG"
  for OBJ in "${OBJECTIVES[@]}"; do
    submit_job "$DIM" "$RUNS" "$SWITCH" "$OBJ" &
    ((job_count++))
    echo "Launched job $job_count: $DIM $RUNS $SWITCH $OBJ"
    if (( job_count >= MAX_JOBS )); then
      echo "Max jobs reached ($MAX_JOBS). Waiting for a job to finish..."
      wait -n
      ((job_count--))
      echo "A job finished. Jobs running: $job_count"
    fi
wait
