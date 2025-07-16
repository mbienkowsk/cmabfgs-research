#!/bin/bash

CONFIGURATIONS=(
  "10 50 1"
  "10 50 2"
)

OBJECTIVES=(
  "Elliptic"
  "CEC1"
  "CEC2"
)

submit_job() {
  DIM=$1
  RUNS=$2
  SWITCH=$3
  OBJ=$4

  export DIMENSIONS=$DIM
  export N_RUNS=$RUNS
  export SWITCH_AFTER=$SWITCH
  export OBJECTIVE=$OBJ

  echo "Submitting job: DIMENSIONS=$DIM, N_RUNS=$RUNS, SWITCH_AFTER=$SWITCH, OBJECTIVE=$OBJ"
  sbatch experiments/bfgs_hybrids/run.sbatch
}

MAX_JOBS=3
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
  done
done

wait
