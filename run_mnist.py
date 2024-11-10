import json
import itertools
import subprocess
import time
import copy

from ml_utils import job_queue 
from threading import Thread 

JobQueue = job_queue.JobQueue

# Define the parameter space
params_space = {
    "bact.wl": [0.01, 0.0],
    "bweight.wl": [0.0],
    "goact.wl": [0.0],
    "goweight.wl": [0.0],
    "batch_size": [64, 128, 256, 512],
    "lr": [0.003],
    "rounding": ["stochastic"],
}


# Function to create and run the command
def get_cmd(params):
    # Generate the command to run the experiment
    cmd = ["python", "experiment2.py", ]
    json_dict = {}
    for key, value in params.items():
        if "wl" in key:
            name = key.split(".")[0]
            json_dict[name] = {"number_type": "noise","std": 0.01}
    
    cmd.append(f"--quant_scheme '{json.dumps(json_dict)}'")
    cmd.append(f"--batch_size {params['batch_size']}")
    cmd.append(f"--lr {params['lr']}")
    cmd.append(f"--experiment_name mnist1 > /dev/null")
    # Run the command and return the process
    return " ".join(cmd)

queue = JobQueue(4, 1)

# Generate all possible combinations of parameters
cmds = []
for values in itertools.product(*params_space.values()):
    # Create a dictionary of parameters
    params = {key: value for key, value in zip(params_space.keys(), values)}
    # Add the job to the queue
    cmds.append(get_cmd(params))

# Run the jobs
# print(cmds)
for cmd in cmds:
    threads = []
    print(cmd)
    for device in range(4):
        cmd_ = f"CUDA_VISIBLE_DEVICES={device} {cmd}"
        thread = Thread(target= lambda : subprocess.run(cmd_, shell=True))
        thread.start()
        threads.append(thread)
    for thread in threads:
        thread.join()
    

        