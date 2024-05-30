import wandb
import pandas as pd
import qtorch.quant
import tqdm
import torch, torchvision, sys, time, numpy
from torch import nn
import copy
import qtorch

import torch

from low_precision_utils.utils import *
from low_precision_utils.metrics import *
from torchsummary import summary

device = torch.device("cuda")
dtype = torch.float32
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--experiment_name", type=str)
parser.add_argument("--model", type=str )
parser.add_argument("--group")
parser.add_argument("--dataset", type=str, default="small_dataset")
parser.add_argument("--lr", type=float, default=0.03)
parser.add_argument("--steps", type=int, default=10000)
parser.add_argument("--act_man_width", type=int, default=23)
parser.add_argument("--weight_man_width", type=int, default=23)
parser.add_argument("--weight_rounding", type=str, default="stochastic")
parser.add_argument("--act_rounding", type=str, default="stochastic")
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--precision_scheduling", type=float, default=float("inf"))
args = parser.parse_args()

wandb.init(project=args.experiment_name, config=args, group=args.group)

wandbconfig = wandb.config

DESIRED_STEPS = wandbconfig.steps
if args.dataset == "small_dataset":
    DATASET_SIZE = 700
elif args.dataset == "mnist":
    DATASET_SIZE = 50000
else:
    raise ValueError("Unknown dataset")
BATCH_SIZE = wandbconfig.batch_size
STEPS_PER_EPOCH = DATASET_SIZE // BATCH_SIZE
EPOCHS = DESIRED_STEPS // STEPS_PER_EPOCH + 1 
PRECISION_SCHEDULING_STEP = wandbconfig.precision_scheduling * DESIRED_STEPS

MOMENTUM = 0.0
WEIGHT_DECAY = 0
LR = wandbconfig.lr
GRAD_ACC_STEPS = 1

WEIGHT_MAN_WIDTH = wandbconfig.weight_man_width
ACT_MAN_WIDTH = wandbconfig.act_man_width


def loadMNISTData(device):
    train = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=torchvision.transforms.ToTensor())
    data = train.train_data.float().to(device)
    data = torch.nn.functional.interpolate(data.view(-1, 1, 28, 28), size=(14, 14)).view(-1, 14 * 14)
    train_data = data[:59000]
    test_data = data[59000:]
    train_labels = train.train_labels[:59000].to(device)
    test_labels = train.train_labels[59000:].to(device)
    return train_data, train_labels, test_data, test_labels


def loadSmallTableData(device):
    train_data = pd.read_csv('train_data.csv')
    X_train = train_data.drop('purchased', axis=1).values
    y_train = train_data['purchased'].values

    # Load the test data
    test_data = pd.read_csv('test_data.csv')
    X_test = test_data.drop('purchased', axis=1).values
    y_test = test_data['purchased'].values

    # Convert the data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    return X_train_tensor.to(device), y_train_tensor.to(device), X_test_tensor.to(device), y_test_tensor.to(device)

def getBatches(X, y):
    length = len(X)
    idx = torch.randperm(length)
    X = X[idx]
    y = y[idx]
    for i in range(0, length, BATCH_SIZE):
        yield X[i:i+BATCH_SIZE], y[i:i+BATCH_SIZE]


class LogisticRegression(nn.Module):
    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(5, 1)
        self.input_size = (5,)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

    def loss_acc(self, x, y):
        output = self(x)
        loss = nn.BCELoss()(output, y)
        pred = output.round()
        acc = pred.eq(y.view_as(pred)).sum().item() / len(y)
        return {"loss": loss, "acc": acc}


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.input_size = (5,)
        self.linear1 = nn.Linear(5, 5)
        self.linear2 = nn.Linear(5, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.linear1(x))
        return torch.sigmoid(self.linear2(x))

    def loss_acc(self, x, y):
        output = self(x)
        loss = nn.BCELoss()(output, y)
        pred = output.round()
        acc = pred.eq(y.view_as(pred)).sum().item() / len(y)
        return {"loss": loss, "acc": acc}

class MNIST_MLP(nn.Module):
    def __init__(self):
        super(MNIST_MLP, self).__init__()
        self.input_size = (14 * 14,)
        self.linear1 = nn.Linear(14 * 14, 100)
        self.linear2 = nn.Linear(100, 10)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.relu(self.linear1(x))
        return self.linear2(x)

    def loss_acc(self, x, y):
        output = self(x)
        output = self.softmax(output)
        loss = self.criterion(output, y)
        pred = output.argmax(dim=1, keepdim=True)
        acc = pred.eq(y.view_as(pred)).sum().item() / len(y)
        return {"loss": loss, "acc": acc}
        
if args.model == "logistic":
    Model = LogisticRegression
elif args.model == "mlp":
    Model = MLP
elif args.model == "mnist_mlp":
    Model = MNIST_MLP
else:
    raise ValueError("Unknown model")

if args.dataset == "mnist":
    X_train, y_train, X_test, y_test = loadMNISTData(device)
else:
    X_train, y_train, X_test, y_test = loadSmallTableData(device)

master_weight = Model()
master_weight = master_weight.to(dtype).to(device)
summary(master_weight, master_weight.input_size)
full_precision_number = qtorch.FloatingPoint(8, 23)
act_number = qtorch.FloatingPoint(8, ACT_MAN_WIDTH)
weight_number = qtorch.FloatingPoint(8, WEIGHT_MAN_WIDTH)
master_weight = replace_linear_with_quantized(master_weight, full_precision_number, full_precision_number, "stochastic")
model_weight = copy.deepcopy(master_weight)
model_weight = apply_number_format(model_weight, act_number, act_number, args.act_rounding, args.act_rounding)

opt = torch.optim.SGD(master_weight.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ConstantLR(opt)

training_time = stepi = 0
bar = tqdm.tqdm(range(EPOCHS * len(X_train) // BATCH_SIZE))
wrapper = MasterWeightOptimizerWrapper(master_weight, model_weight, 
                                       opt, scheduler, 
                                       weight_quant=qtorch.quant.quantizer(weight_number, forward_rounding=args.weight_rounding)
                                       )

@torch.no_grad()
def test(network, dataset):
    network.eval()
    correct = 0
    total_loss = 0
    i = 0
    n = 0
    with torch.no_grad():
        for data, target in dataset:
            i += 1
            n += len(data)
            loss_acc = network.loss_acc(data, target)
            total_loss += loss_acc["loss"].item()
            correct += loss_acc["acc"] * len(data)
    accuracy = correct / n
    avg_loss = total_loss / i
    network.train()
    return {"test_acc": accuracy, "test_loss": avg_loss}
                                       

ema_metrics = EMAMetrics()
test_result_m = {}
result_log = {}
def ema_grad_on_dataset(network, data, target):
    grad = grad_on_dataset(network, data, target)
    ema_grad = ema_metrics.update(grad)
    return grad | ema_grad
for epoch in range(EPOCHS):
    start_time = time.perf_counter()
    train_losses, train_accs = [], []
    master_weight.train()
    model_weight.train()

    for X,y in getBatches(X_train,y_train):
        stepi += 1
        result_log.update(wrapper.train_on_batch(X, y))
        bar.update(1)
        bar.set_postfix(result_log)
        result_log["lr"] = opt.param_groups[0]["lr"]
        if stepi % 50 == 0:
            wandb.log(result_log | test_result_m)
        if stepi >= PRECISION_SCHEDULING_STEP:
            model_weight = apply_number_format(model_weight, full_precision_number, full_precision_number, "stochastic", "stochastic")
            wrapper.weight_quant = lambda x: x
            PRECISION_SCHEDULING_STEP = float("inf")
        if stepi >= DESIRED_STEPS:
            test_result_m = test(master_weight, ((X_test, y_test),))
            grad_entire = ema_grad_on_dataset(master_weight, X_train, y_train)
            result_log.update(grad_entire)
            wandb.log(result_log | test_result_m)
            break

    # wrapper.master_params_to_model_params(quantize=False)
    result_log.update(ema_grad_on_dataset(master_weight, X_train, y_train))
    training_time += time.perf_counter()-start_time
    test_result_m = test(master_weight, ((X_test, y_test),))
    print(f'epoch % 2d  test loss m %.3f   test acc m %.3f  training time %.2f'%(epoch+1, test_result_m["test_loss"], test_result_m["test_acc"], training_time))
  