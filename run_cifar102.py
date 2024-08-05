import json
import subprocess

number_format = {
    "bact": {
        "number_type":"fixed", 
        "round_mode":"stochastic",
        "fl":2,
        "wl":3
    },
    "bweight": {
        "number_type":"fixed", 
        "round_mode":"nearest",
        "fl":2,
        "wl":3
    },
    "goact": {
        "number_type":"fixed", 
        "round_mode":"nearest",
        "fl":5,
        "wl":6
    },
    "goweight": {
        "number_type":"fixed", 
        "round_mode":"stochastic",
        "fl":2,
        "wl":3
    },
}


subprocess.run(
    [f"python cifar10.py --number_json_string '{json.dumps(number_format)}'"], shell=True
)