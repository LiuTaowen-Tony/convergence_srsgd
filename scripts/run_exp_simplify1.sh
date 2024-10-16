#!/bin/bash

git submodule update --recursive

# Define the grid search parameters
act_man_widths=(2 3 4 8)
weight_man_widths=(8)
batch_sizes=(1000 2000) 
lrs=(0.015)
round_mode="stochastic"
experiment_name="back_only_act_only5"
model="linear"
dataset="linear"

# Job count for controlling GPU assignment
job_count=0

# Outer loop for batch_size
for lr in "${lrs[@]}"; do
for batch_size in "${batch_sizes[@]}"; do
  # Inner loop for man_width
  for man_width in "${act_man_widths[@]}"; do
  for weight_man_width in "${weight_man_widths[@]}"; do
    # Loop for running each configuration 8 times
    for (( i=1; i<= 4; i++ )); do
      # Calculate GPU ID to use
      gpu_id=$((job_count % 4))

      cmd="OMP_NUM_THREADS=1 \
      CUDA_VISIBLE_DEVICES=$gpu_id \
      time python exp_simplify1.py \
      --experiment_name=$experiment_name
      --batch_size=$batch_size \
      --weight_man_width=$weight_man_width \
      --act_man_width=$man_width \
      --lr=$lr \
      "
      
      # Run the command in the background
      eval $cmd &
      # Increment job count
      job_count=$((job_count + 1))
    done
    wait
  done
done
done
done