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

# %%
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

MnistDataset = datasets.MNIST("./data", download=True, train=False, transform=transform)

MnistDatasetTrain =torch.utils.data.Subset(MnistDataset, range(7500)) 
MnistDatasetTest =torch.utils.data.Subset(MnistDataset, range(7500, 10000)) 
fnumber = qtorch.FloatingPoint(5, 3)
bnumber = qtorch.FloatingPoint(5, 3)

def get_network():
    network = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(784, 256),
        torch.nn.Dropout(0.1),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 10),
        torch.nn.Softmax(dim=1)
    )
    return network

def get_low_precision_network():
    low_precision_network = torch.nn.Sequential(
        torch.nn.Flatten(),
        Quantizer(forward_number=fnumber),
        torch.nn.Linear(784, 256),
        Quantizer(backward_number=bnumber),
        torch.nn.ReLU(),
        Quantizer(forward_number=fnumber),
        torch.nn.Linear(256, 10),
        Quantizer(backward_number=bnumber),
        torch.nn.Softmax(dim=1)
    )
    return low_precision_network



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
        self.grad_clip = 5.0

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

    def _apply_model_weights(self, model, quant_func):
        for m in model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                m.weight.data = quant_func(m.weight.data)
                if m.bias is not None:
                    m.bias.data = quant_func(m.bias.data)

    def train_on_loss(self, loss):
        loss = loss * self.grad_scaling
        opt = self.optimizer
        self.model_weight.zero_grad()
        loss.backward()
        self.model_grads_to_master_grads()
        self.master_grad_apply(lambda x: x / self.grad_scaling)
        nn.utils.clip_grad_norm_(self.master_weight.parameters(), self.grad_clip)
        opt.step()
        self.master_params_to_model_params()
        self._apply_model_weights(self.model_weight, self.weight_quant)
        return loss.item()

# %%
import torch
import torch.nn as nn
import wandb
device = "cuda"

MnistDatasetTrain.data = MnistDataset.data.to(device)
MnistDatasetTrain.targets = MnistDataset.targets.to(device)
MnistDatasetTest.data = MnistDataset.data.to(device)
MnistDatasetTest.targets = MnistDataset.targets.to(device)


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

    accuracy = correct / len(dataset.dataset) 
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

def train(network, dataset, test_dataset, steps, lr=0.01):
    wandb.init(project="convergence_srsgd")
    wandb.watch(network, log='all', log_freq=10)

    optimizer = torch.optim.SGD(network.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    iteration = 0
    loss_sum = 0
    running_weight = deepcopy(network)
    # wrapper = MasterWeightOptimizerWrapper(network, model_weight=running_weight, optimizer=optimizer, weight_quant=qtorch.quant.quantizer(fnumber))

    bar = tqdm.tqdm(total=steps)
    while True:
        for data, target in (dataset):
            if iteration >= steps:
                return network
            bar.update(1)
            iteration += 1
            optimizer.zero_grad()
            output = running_weight(data)
            loss = criterion(output, target)
            # wrapper.train_on_loss(loss)
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()

            if iteration % 10 == 0:
                avg_loss = loss_sum / 10
                wandb.log({"iteration_loss": avg_loss})
                loss_sum = 0
                stats = report_stats(network)
                # stats_moving_average.add_value(stats)
                wandb.log({"0_g_norm": stats["0_g_norm"]})
                wandb.log({"1_g_norm": stats["1_g_norm"]})
                # wandb.log({"0_g_norm_mov_avg": stats_moving_average.get()["0_g_norm_mov_avg"]})
                # wandb.log({"1_g_norm_mov_avg": stats_moving_average.get()["1_g_norm_mov_avg"]})

            if iteration % 100 == 0:
                test_metrics = test(network, test_dataset)
                wandb.log(test_metrics)


# %%
import random
class MyDataLoader():
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.idx = list(range(len(dataset)))
        random.shuffle(self.idx)


    def __iter__(self):
        data = self.dataset.data
        targets = self.dataset.targets
        for i in range(0, len(data), self.batch_size):
            idx = self.idx[i:i + self.batch_size]
            yield data[idx].float(), targets[idx]

    def __len__(self):
        return len(self.dataset) // self.batch_size

network = get_low_precision_network()


for i in range(30):
    train_loader = MyDataLoader(MnistDatasetTrain, 64)
    test_loader = MyDataLoader(MnistDatasetTest, 512)
    network = network.to(device)
    train(network,train_loader, test_loader, steps=100000)