#!/bin/bash

# Define the grid search parameters
batch_sizes=(32 40 48)
man_widths=(2 3)
lr=0.01
round_mode="stochastic"
scheduler="fix"
steps=50000

group_name="b${batch_size}m${man_width}lr${lr}${round_mode}${scheduler}"
    
# Formulate the command with dynamic GPU assignment
cmd="OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=$gpu_id python convergence_sr.py --round_mode=$round_mode --batch_size=$batch_size --man_width=$man_width --group=$group_name --id=${group_name}_$i --lr=$lr --steps=$steps --scheduler=$scheduler"

# Run the command in the background
eval $cmd
