import torch
from torch import nn

def grad_on_dataset(network, data, target):
    total_norm = 0
    network.zero_grad()
    idx = torch.randperm(len(data))[:2048]
    data = data[idx]
    target = target[idx]
    output = network(data)
    loss = torch.nn.CrossEntropyLoss()(output, target)
    loss.backward()
    total_norm = nn.utils.clip_grad_norm_(network.parameters(), float('inf'))
    ps = list(network.parameters())
    # first param that is large in dimension
    initial_layer_param = ps[0]
    last_layer_param = ps[-1]
    initial_layer_zero_ratio = torch.sum(initial_layer_param == 0).item() / initial_layer_param.numel()
    last_layer_zero_ratio = torch.sum(last_layer_param == 0).item() / last_layer_param.numel()
    network.zero_grad()
    return {"grad_norm_entire": total_norm.item(),
            "initial_layer_zero_ratio": initial_layer_zero_ratio,
            "last_layer_zero_ratio": last_layer_zero_ratio}

def probe_activation(network, data, target):
    # add hook to the network to gather statistics of activations
    