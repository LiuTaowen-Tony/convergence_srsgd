# %%
from copy import deepcopy
import torch
from torch import nn
import qtorch
from qtorch.quant import Quantizer
import qtorch
import tqdm

from torchvision import datasets, transforms
import torch

import argparse

parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--man_width', type=int, default=5)
parser.add_argument('--group', type=str)
parser.add_argument('--id', type=str)
parser.add_argument("--round_mode", type=str, default="stochastic")
parser.add_argument("--steps", type=int, default=10000)
parser.add_argument("--scheduler", type=str, default="fix")

# import namespace
from argparse import Namespace

args = parser.parse_args()
# args = Namespace()
# args.batch_size = 64
# args.lr = 0.01
# args.man_width = 5
# args.group = "convergence_srsgd"
# args.id = "1"
# args.round_mode = "stochastic"
# %%
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])


class MyDataLoader():
    def __init__(self, data, targets, batch_size):
        self.data = data
        self.targets = targets
        self.batch_size = batch_size
        self.idx = list(range(len(data)))
        random.shuffle(self.idx)


    def __iter__(self):
        for i in range(0, len(data), self.batch_size):
            idx = self.idx[i:i + self.batch_size]
            if len(idx) == 0: continue
            yield self.data[idx], self.targets[idx]

    def __len__(self):
        return len(self.data) // self.batch_size

fnumber = qtorch.FloatingPoint(8, args.man_width)
bnumber = qtorch.FloatingPoint(8, args.man_width)

def get_network():
    network = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(784, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 10),
        torch.nn.Softmax(dim=1)
    )
    return network

class ActivationQuantizedLinear(nn.Linear):
    def __init__(self, in_features, out_features, fnumber, bnumber, round_mode):
        super(ActivationQuantizedLinear, self).__init__(in_features, out_features)
        self.quant = Quantizer(forward_number=fnumber, forward_rounding=round_mode)
        self.quant_b = Quantizer(backward_number=bnumber, backward_rounding=round_mode)

    def forward(self, x):
        x = self.quant(x)
        x = super().forward(x)
        x = self.quant_b(x)
        return x

def replace_linear_with_quantized(network, fnumber, bnumber, round_mode):
    for i, m in enumerate(network.children()):
        if isinstance(m, nn.Linear):
            layer = ActivationQuantizedLinear(m.in_features, m.out_features, fnumber, bnumber, round_mode)
            layer.weight.data = m.weight.data
            setattr(network, f"{i}", layer)
    return network

network = get_network()
network_q = replace_linear_with_quantized(deepcopy(network), fnumber, bnumber, args.round_mode)

# %%

class MasterWeightOptimizerWrapper():
    def __init__(
            self,
            master_weight,
            model_weight,
            optimizer,
            weight_quant=None,
            grad_scaling=1.0,
    ):
        self.master_weight = master_weight
        self.model_weight = model_weight
        self.optimizer = optimizer
        self.weight_quant = weight_quant
        self.grad_scaling = grad_scaling
        self.grad_clip = 100.0
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=0.2)

    # --- for mix precision training ---
    def model_grads_to_master_grads(self):
        for model, master in zip(self.model_weight.parameters(), self.master_weight.parameters()):
            if master.grad is None:
                master.grad = master.data.new(*master.data.size())
            master.grad.data.copy_(model.grad.data)

    def master_grad_apply(self, fn):
        for master in (self.master_weight.parameters()):
            if master.grad is None:
                master.grad = master.data.new(*master.data.size())
            master.grad.data = fn(master.grad.data)

    def master_params_to_model_params(self):
        for model, master in zip(self.model_weight.parameters(), self.master_weight.parameters()):
            model.data.copy_(self.weight_quant(master.data))

    def train_on_batch(self, data, target):
        if args.man_width == 23:
            self.master_weight.zero_grad()
            output = self.master_weight(data)
            loss = self.loss_fn(output, target)
            loss.backward()
            self.optimizer.step()
            return loss
        self.model_weight.zero_grad()
        output = self.model_weight(data)
        loss = self.loss_fn(output, target)
        loss = loss * self.grad_scaling
        loss.backward()
        self.model_grads_to_master_grads()
        self.master_grad_apply(lambda x: x / self.grad_scaling)
        nn.utils.clip_grad_norm_(self.master_weight.parameters(), self.grad_clip)
        self.optimizer.step()
        self.master_params_to_model_params()
        return loss


# %%
import torch
import torch.nn as nn
import wandb
device = "cuda"


def grad_on_dataset(network, dataloader):
    network.train()
    total_norm = 0
    network.zero_grad()
    data = dataloader.data
    target = dataloader.targets
    output = network(data)
    loss = torch.nn.CrossEntropyLoss()(output, target)
    loss.backward()
    total_norm = nn.utils.clip_grad_norm_(network.parameters(), float('inf'))
    network.zero_grad()
    network.eval()
    return {"grad_norm": total_norm.item()}

def test(network, dataset):
    network.eval()
    correct = 0
    total_loss = 0
    criterion = torch.nn.CrossEntropyLoss()
    i = 0
    with torch.no_grad():
        for data, target in dataset:
            i += 1
            output = network(data)
            total_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    accuracy = correct / len(dataset.data) 
    avg_loss = total_loss / i
    network.train()
    return {"acc": accuracy, "test_loss": avg_loss}

class MovingAvg():
    def __init__(self, beta=0.9):
        self.beta = beta
        self.average = None

    def update(self, value):
        if self.average is None:
            self.average = value
        else:
            self.average = self.beta * self.average + (1 - self.beta) * value

    def get(self):
        return self.average

class MovingAvgStat():
    def __init__(self, beta):
        self.beta = beta
        self.stats = {}

    def add_value(self, stats):
        for key in stats:
            if key not in self.stats:
                self.stats[key] = MovingAvg(self.beta)
            self.stats[key].update(stats[key])

    def get(self):
        return {f"{key}_mov_avg": self.stats[key].get() for key in self.stats}

def report_stats(network):
    stats = {}
    i = 0
    for _, m in enumerate(network.modules()):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            stats[f"{i}_w_norm"] = m.weight.data.norm().item()
            stats[f"{i}_w_mean"] = m.weight.data.mean().item()
            stats[f"{i}_w_std"] = m.weight.data.std().item()
            stats[f"{i}_w_max"] = m.weight.data.max().item()
            stats[f"{i}_w_min"] = m.weight.data.min().item()
            stats[f"{i}_g_norm"] = torch.sqrt((m.weight.grad.data ** 2).sum()).item() if m.weight.grad is not None else 0
            stats[f"{i}_g_mean"] = m.weight.grad.data.mean().item() if m.weight.grad is not None else 0
            stats[f"{i}_g_std"] = m.weight.grad.data.std().item() if m.weight.grad is not None else 0
            i += 1
    return stats

def train(network, dataset, test_dataset, steps, lr=args.lr):
    wandb.init(project="convergence_srsgd", group=args.group, id=args.id)
    wandb.watch(network, log='all', log_freq=30)

    if args.scheduler == "inverse":
        lr = 100.0
    optimizer = torch.optim.SGD(network.parameters(), lr=lr)
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)
    elif args.scheduler == "inverse":
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda x: 1 / (x + 1))
    elif args.scheduler == "fix":
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda x: 1)
    iteration = 0
    loss_sum = 0
    running_weight = replace_linear_with_quantized(deepcopy(network), fnumber, bnumber, args.round_mode)
    running_weight.to(device)
    wrapper = MasterWeightOptimizerWrapper(network, model_weight=running_weight, optimizer=optimizer, weight_quant=qtorch.quant.quantizer(fnumber, forward_rounding=args.round_mode), grad_scaling=1.0)

    bar = tqdm.tqdm(total=steps)
    result_log = {}
    while True:
        for data, target in (dataset):
            if iteration >= steps:
                return network
            bar.update(1)
            iteration += 1
            loss = wrapper.train_on_batch(data, target)
            scheduler.step()
            loss_sum += loss.item()
            result_log["loss"] = loss.item()

            if iteration % 30 == 0:
                result_log.update(grad_on_dataset(network, dataset))
                test_metrics = test(network, test_dataset)
                result_log.update(test_metrics)
            result_log.update({"lr": scheduler.get_last_lr()[0]})
                
            bar.set_postfix(result_log)
            wandb.log(result_log)


# %%
import random

network = get_network()

MnistDataset = datasets.MNIST("./data", download=True, train=False, transform=transform)
device = "cuda"
data = MnistDataset.data.to(device).float()
targets = MnistDataset.targets.to(device)
train_set_end = 9000
test_set_end = 10000
train_loader = MyDataLoader(data[:train_set_end], targets[:train_set_end], args.batch_size)
test_loader = MyDataLoader(data[train_set_end:test_set_end], targets[train_set_end:test_set_end], test_set_end - train_set_end)
network = network.to(device)
train(network,train_loader, test_loader, steps=args.steps)
# %%
