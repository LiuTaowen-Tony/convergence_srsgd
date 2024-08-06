import json
import itertools
import subprocess
import time
from ml_utils.job_queue import JobQueue

# Define the parameter space
params_space = {
    "bact.wl": [4],
    "bweight.wl": [4],
    "goact.wl": [7],
    "goweight.wl": [4],
    "rounding": ["nearest", "stochastic"],
}

# Function to create and run the command
def run_experiment(device, params):
    # Generate the command to run the experiment
    cmd = [f"CUDA_VISIBLE_DEVICES={device}", "python", "cifar10.py", ]
    json_dict = {}
    for key, value in params.items():
        if "wl" in key:
            name = key.split(".")[0]
            json_dict[name] = {"number_type": "scaled_int", "round_mode": params["rounding"],"wl": value}
    
    cmd.append(f"--quant_scheme '{json.dumps(json_dict)}'")
    # Run the command and return the process
    print(" ".join(cmd))
    subprocess.run(" ".join(cmd), shell=True)

queue = JobQueue(4, 1)

# Generate all possible combinations of parameters
callables = []
for params in itertools.product(*params_space.values()):
    # Create a dictionary of parameters
    params = {key: value for key, value in zip(params_space.keys(), params)}
    # Add the job to the queue
    callables.append(lambda device: run_experiment(device, params))

# Run the jobs
queue.map(callables)