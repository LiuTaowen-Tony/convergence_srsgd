#!/bin/bash

# Define the grid search parameters
man_widths=(0 1)
roundings=("stochastic" "nearest")
lrs=(0.4 0.45 0.5 0.55 0.6)
round_mode="stochastic"
experiment_name="cifar10_unbias"

# Job count for controlling GPU assignment
job_count=0

# Outer loop for batch_size
for lr in "${lrs[@]}"; do
  for man_width in "${man_widths[@]}"; do
  for rounding in "${roundings[@]}"; do
    # Loop for running each configuration 8 times
    cmd="OMP_NUM_THREADS=1 \
    python experiment.py \
    --experiment_name=$experiment_name \
    --weight_man_width=$man_width \
    --act_man_width=$man_width \
    --back_man_width=$man_width \
    --act_rounding=$rounding \
    --weight_rounding=$rounding \
    --back_rounding=$roudnding \
    --lr=$lr \
    "
    for (( i=1; i<= 4; i++ )); do
      gpu_id=$((job_count % 4))
      CUDA_VISIBLE_DEVICES=$gpu_id eval $cmd &
      job_count=$((job_count + 1))
    done
    wait
    for (( i=1; i<= 4; i++ )); do
      gpu_id=$((job_count % 4))
      CUDA_VISIBLE_DEVICES=$gpu_id eval $cmd &
      job_count=$((job_count + 1))
    done
    wait
  done
done
done