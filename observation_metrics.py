import torch
from torch import nn

def grad_on_dataset(network, data, target):
    network.train()
    total_norm = 0
    network.zero_grad()
    idx = torch.randperm(len(data))[:2048]
    data = data[idx]
    target = target[idx]
    output = network(data)
    loss = torch.nn.CrossEntropyLoss()(output, target)
    loss.backward()
    total_norm = nn.utils.clip_grad_norm_(network.parameters(), float('inf'))
    network.zero_grad()
    network.eval()
    return {"grad_norm_entire": total_norm.item()}

def zero_grad_ratio(network, data, target):
    network.train()
    network.zero_grad()
    idx = torch.randperm(len(data))[:2048]
    data = data[idx]
    target = target[idx]
    output = network(data)
    loss = torch.nn.CrossEntropyLoss()(output, target)
    loss.backward()
    network.zero_grad()
    network.eval()
    return {"zero_grad_ratio": sum([1 for p in network.parameters() if p.grad is None]) / len(list(network.parameters()))}

def probe_activation(network, data, target):
    # add hook to the network to gather statistics of activations
    