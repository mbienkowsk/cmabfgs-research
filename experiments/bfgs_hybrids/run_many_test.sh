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
  MAX_JOBS=3
  job_count=0

  for CONFIG in "${CONFIGURATIONS[@]}"; do
    read -r DIM RUNS SWITCH <<< "$CONFIG"
    for OBJ in "${OBJECTIVES[@]}"; do
      submit_job "$DIM" "$RUNS" "$SWITCH" "$OBJ" &
      ((job_count++))
      if (( job_count >= MAX_JOBS )); then
        wait -n
        ((job_count--))
      fi
    done
  done

  wait
}
