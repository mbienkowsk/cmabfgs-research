
#!/bin/bash

CONFIGURATIONS=(
  "10 50"
  "100 50"
)

OBJECTIVES=(
  "CEC1"
  "CEC2"
  "CEC3"
  "CEC4"
  "CEC5"
  "CEC6"
  "CEC7"
  "CEC8"
  "CEC9"
  "CEC10"
  "CEC11"
  "CEC12"
  "CEC13"
  "CEC14"
  "CEC15"
  "CEC16"
  "CEC17"
  "CEC18"
  "CEC19"
  "CEC20"
  "CEC21"
  "CEC22"
  "CEC23"
  "CEC24"
  "CEC25"
  "CEC26"
  "CEC27"
  "CEC28"
  "CEC29"
)


submit_job() {
  DIM=$1
  RUNS=$2
  OBJ=$3

  export DIMENSIONS=$DIM
  export N_RUNS=$RUNS
  export OBJECTIVE=$OBJ

  echo "Submitting job: DIMENSIONS=$DIM, N_RUNS=$RUNS, OBJECTIVE=$OBJ"
  sbatch experiments/cmaes_matrix_convergence/run.sbatch
}

MAX_JOBS=10
job_count=0

for CONFIG in "${CONFIGURATIONS[@]}"; do
  read -r DIM RUNS <<< "$CONFIG"
  for OBJ in "${OBJECTIVES[@]}"; do
    submit_job "$DIM" "$RUNS" "$OBJ" &
    ((job_count++))
    echo "Launched job $job_count: $DIM $RUNS $OBJ"
    if (( job_count >= MAX_JOBS )); then
      echo "Max jobs reached ($MAX_JOBS). Waiting for a job to finish..."
      wait -n
      ((job_count--))
      echo "A job finished. Jobs running: $job_count"
    fi
  done
done

wait
