#!/bin/zsh

BASE_DIR="experiments/experimentum_crucis/results/plots"

files=(
  "convergence_curve.png"
  "consecutive_ratios_stats.png"
  "corresp_eig_ratios_stats.png"
  "eig_max_min_ratio.png"
)

for file in "${files[@]}"; do
  matches=()
  matches+=($BASE_DIR/*/$file)
  feh -FZ "${matches[@]}"
done

