import json
import itertools
import subprocess
import time

# Define the parameter space
params_space = {
    "bact.wl": [3,4],
    "bweight.wl": [2,3,4],
    "goact.wl": [5,6],
    "goweight.wl": [3,4],
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
            json_dict[name] = {"number_type": "fixed", "round_mode": params["rounding"], "fl": value - 1, "wl": value}
    
    cmd.append(f"--quant_scheme_json '{json.dumps(json_dict)}'")
    # Run the command and return the process
    print(" ".join(cmd))
    return subprocess.Popen(" ".join(cmd), shell=True)

# Number of devices
num_devices = 4
# Experiment limit per device
experiment_limit = 1
# Track the number of experiments run on each device
device_counters = [0] * num_devices
# Track the currently running processes
running_processes = [None] * num_devices

# Iterate over all combinations of parameters
for pvals in itertools.product(*params_space.values()):
    while True:
        # Check for available devices
        for device in range(num_devices):
            # Check if the process on this device has finished
            if running_processes[device] is not None and running_processes[device].poll() is not None:
                running_processes[device] = None  # Reset if the process has finished
                device_counters[device] -= 1      # Decrement the counter as the process has finished

            # If the device is available and under the experiment limit
            if device_counters[device] < experiment_limit:
                # Increment the counter for the selected device
                device_counters[device] += 1
                break
        else:
            # If no devices are available, wait and retry
            time.sleep(1)
            continue
        break

    # Create a dictionary of parameters for the current combination
    params = dict(zip(params_space.keys(), pvals))

    # Run the experiment with the current set of parameters
    running_processes[device] = run_experiment(device, params)
