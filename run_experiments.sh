#!/bin/bash

git submodule update --recursive

# Define the grid search parameters
man_widths=(0 1 2 4 8 16 23)
batch_sizes=(1 2 4 8 16 32)
lrs=(0.03)
round_mode="stochastic"
experiment_name="precision_scheduling_small_logistic"
model="logistic"
dataset="small_dataset"
precision_scheduling=0.8

# Job count for controlling GPU assignment
job_count=0

# Outer loop for batch_size
for lr in "${lrs[@]}"; do
for batch_size in "${batch_sizes[@]}"; do
  # Inner loop for man_width
  for man_width in "${man_widths[@]}"; do
    # Loop for running each configuration 8 times
    for (( i=1; i<= 8; i++ )); do
      # Calculate GPU ID to use
      gpu_id=$((job_count % 2 + 1))

      cmd="OMP_NUM_THREADS=8 \
      CUDA_VISIBLE_DEVICES=$gpu_id \
      python experiment.py \
      --experiment_name=$experiment_name
      --model=${model} \
      --dataset=${small_dataset} \
      --steps=10_000 \
      --batch_size=$batch_size \
      --weight_man_width=$man_width \
      --act_man_width=$man_width \
      --precision_scheduling=$precision_scheduling \
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