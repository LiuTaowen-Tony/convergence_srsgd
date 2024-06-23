import subprocess, os

configs = []
for lr in [0.9, 1.0, 1.1, 1.2, 1.3]:
    for man_width in [0, 1]:
        configs.append({
            "man_width":man_width,
            "rounding": "stochastic",
            "same_input": False,
            "same_weight": False,
            "lr": lr,
        })
        configs.append({
            "man_width": man_width,
            "rounding": "nearest",
            "same_input": True,
            "same_weight": True,
            "lr": lr,
        })
        # for same_input in [True, False]:
        #     for same_weight in [True, False]:
        #         configs.append({
        #             "man_width": man_width,
        #             "rounding": "stochastic",
        #             "same_input": same_input,
        #             "same_weight": same_weight,
        #             "lr": lr,
        #         })

    configs.append({
        "man_width": 23,
        "rounding": "stochastic",
        "same_input": False,
        "same_weight": False,
        "lr": lr
    })

def config_to_list(config):
    l = ["python", "cifar10.py"]
    for k, v in config.items():
        if k == "man_width":
            l.append(f"--act_man_width={v}")
            l.append(f"--weight_man_width={v}")
            l.append(f"--back_man_width={v}")
            continue
        if k == "rounding":
            l.append(f"--act_rounding={v}")
            l.append(f"--weight_rounding={v}")
            l.append(f"--back_rounding={v}")
            continue
        l.append(f"--{k}={v}")
    return l
        

for config in configs:
    config["experiment_name"] = "cifar10_unbias"
    handles: list[subprocess.Popen] = []
    for i in range(4):
        cmd = config_to_list(config)
        my_env = os.environ.copy()
        my_env["CUDA_VISIBLE_DEVICES"] = str(i)
        process = subprocess.Popen(cmd, env=my_env)
        handles.append(process)
    for handle in handles:
        handle.wait()
    print("Done with config", config)
    
