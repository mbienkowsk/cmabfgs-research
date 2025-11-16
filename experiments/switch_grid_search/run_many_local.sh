#!/bin/bash -l

BASE_DIR=$(pwd)
CONFIG_FILE="$BASE_DIR/experiments/switch_grid_search/config.txt"
VENV_ACTIVATE="$BASE_DIR/.venv/bin/activate"
PYTHON_MODULE="experiments.switch_grid_search.switch_grid_search"

# Check base directory contents
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found at $CONFIG_FILE"
    echo "Please run this script from your project's root directory."
    exit 1
fi

if [ ! -f "$VENV_ACTIVATE" ]; then
    echo "Error: Virtual environment not found at $VENV_ACTIVATE"
    echo "Please ensure your .venv is in the root directory."
    exit 1
fi

echo "Activating virtual environment: $VENV_ACTIVATE"
source "$VENV_ACTIVATE"

# Runs the experiments sequentially
TOTAL_LINES=$(wc -l < "$CONFIG_FILE")
CURRENT_LINE=0

while IFS= read -r line; do
    CURRENT_LINE=$((CURRENT_LINE + 1))
 
    # Skip empty lines or lines starting with #
    if [ -z "$line" ] || [[ "$line" == \#* ]]; then
        continue
    fi

    read OBJECTIVE DIMENSIONS N_RUNS SWITCH_AFTER <<< "$line"

    export OBJECTIVE DIMENSIONS N_RUNS SWITCH_AFTER

    echo "=================================================="
    echo " $(date +%T) Starting experiment $CURRENT_LINE / $TOTAL_LINES"
    echo "  OBJECTIVE:    $OBJECTIVE"
    echo "  DIMENSIONS:   $DIMENSIONS"
    echo "  N_RUNS:       $N_RUNS"
    echo "  SWITCH_AFTER: $SWITCH_AFTER"
    echo "=================================================="

    python3 -m "$PYTHON_MODULE"

    echo "--------------------------------------------------"
    echo "Finished experiment $CURRENT_LINE / $TOTAL_LINES: $OBJECTIVE"
    echo "--------------------------------------------------"
    echo ""

done < "$CONFIG_FILE"

echo "All experiments completed."
