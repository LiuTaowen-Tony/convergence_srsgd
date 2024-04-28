#!/bin/bash

# Define the grid search parameters
batch_sizes=(32 40 48)
man_widths=(2 3)
lr=0.01
round_mode="stochastic"
scheduler="fix"
steps=50000

# Job count for controlling GPU assignment
job_count=0

# Outer loop for batch_size
for batch_size in "${batch_sizes[@]}"; do
  # Inner loop for man_width
  for man_width in "${man_widths[@]}"; do
    group_name="b${batch_size}m${man_width}lr${lr}${round_mode}${scheduler}"
    
    # Loop for running each configuration 8 times
    for (( i=1; i<=8; i++ )); do
      # Calculate GPU ID to use
      gpu_id=$((job_count % 4))

      # Formulate the command with dynamic GPU assignment
      cmd="OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=$gpu_id python convergence_sr.py --round_mode=$round_mode --batch_size=$batch_size --man_width=$man_width --group=$group_name --id=${group_name}_$i --lr=$lr --steps=$steps --scheduler=$scheduler"
      
      # Run the command in the background
      eval $cmd &
      # Increment job count
      job_count=$((job_count + 1))
    done
    wait
  done
done
