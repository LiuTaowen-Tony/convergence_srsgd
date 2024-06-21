#!/bin/bash

git submodule update --recursive

# Define the grid search parameters
man_widths=(0 23)
batch_sizes=(1024)
lrs=(0.03)
round_mode="stochastic"
experiment_name="linear_regression_act_only"
model="linear"
dataset="linear"
precision_scheduling=(inf)
weight_man_width=23
back_man_width=0

# Job count for controlling GPU assignment
job_count=0

# Outer loop for batch_size
for pr in "${precision_scheduling[@]}"; do
for lr in "${lrs[@]}"; do
for batch_size in "${batch_sizes[@]}"; do
  # Inner loop for man_width
  for man_width in "${man_widths[@]}"; do
    # Loop for running each configuration 8 times
    for (( i=1; i<= 8; i++ )); do
      # Calculate GPU ID to use
      gpu_id=$((job_count % 2 + 2))

      cmd="OMP_NUM_THREADS=1 \
      CUDA_VISIBLE_DEVICES=$gpu_id \
      python experiment.py \
      --experiment_name=$experiment_name
      --model=${model} \
      --dataset=${dataset} \
      --steps=10_000 \
      --batch_size=$batch_size \
      --weight_man_width=$weight_man_width \
      --act_man_width=$man_width \
      --back_man_width=$back_man_width \
      --precision_scheduling=$pr \
      --act_rounding=$round_mode \
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