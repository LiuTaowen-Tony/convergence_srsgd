import wandb
import pandas as pd
import qtorch.quant
import tqdm
import torch, torchvision, sys, time, numpy
from torch import nn
import copy
import qtorch

import torch

from low_precision_utils import utils
from low_precision_utils.metrics import *
from torchsummary import summary

device = torch.device("cuda")
dtype = torch.float32
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--experiment_name", type=str)
parser.add_argument("--model", type=str, default="mnist_cnn")
parser.add_argument("--dataset", type=str, default="mnist")
parser.add_argument("--lr", type=float, default=0.03)
parser.add_argument("--momentum", type=float, default=0.0)
parser.add_argument("--steps", type=int, default=10000)
parser.add_argument("--act_man_width", type=int, default=23)
parser.add_argument("--weight_man_width", type=int, default=23)
parser.add_argument("--back_man_width", type=int, default=23)
parser.add_argument("--weight_rounding", type=str, default="stochastic")
parser.add_argument("--act_rounding", type=str, default="stochastic")
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--precision_scheduling", type=float, default=float("inf"))
parser.add_argument("--log_interval", type=int, default=50)
parser.add_argument("--watch_interval", type=int, default=100)
parser.add_argument("--opt", type=str, default="sgd")
parser.add_argument("--test_nan", action="store_true")
parser.add_argument("--grad_acc_steps", type=int, default=1)
parser.add_argument("--same_input", action="store_true")
args = parser.parse_args()

wandb.init(project=args.experiment_name, config=args)

wandbconfig = wandb.config

if wandbconfig.test_nan:
    utils.TEST_NAN = True

DESIRED_STEPS = wandbconfig.steps
if args.dataset == "small_dataset":
    DATASET_SIZE = 700
elif args.dataset == "mnist":
    DATASET_SIZE = 50000
elif args.dataset == "linear":
    DATASET_SIZE = 405
else:
    raise ValueError("Unknown dataset")
BATCH_SIZE = wandbconfig.batch_size
STEPS_PER_EPOCH = DATASET_SIZE // BATCH_SIZE
if BATCH_SIZE <= DATASET_SIZE:
    EPOCHS = DESIRED_STEPS // STEPS_PER_EPOCH + 1 
else:
    EPOCHS = DESIRED_STEPS
PRECISION_SCHEDULING_STEP = wandbconfig.precision_scheduling * DESIRED_STEPS

MOMENTUM = wandbconfig.momentum
WEIGHT_DECAY = 0
LR = wandbconfig.lr
GRAD_ACC_STEPS = wandbconfig.grad_acc_steps

WEIGHT_MAN_WIDTH = wandbconfig.weight_man_width
ACT_MAN_WIDTH = wandbconfig.act_man_width


def loadMNISTData(device):
    train = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=torchvision.transforms.ToTensor())
    data = train.train_data.float().to(device)
    # data = torch.nn.functional.interpolate(data.view(-1, 1, 28, 28), size=(14, 14)).view(-1, 14 * 14)
    data = data.view(-1, 28 * 28)
    train_data = data[:59000]
    test_data = data[59000:]
    train_labels = train.train_labels[:59000].to(device)
    test_labels = train.train_labels[59000:].to(device)
    return train_data, train_labels, test_data, test_labels

def loadCIFAR10(device):
  train = torchvision.datasets.CIFAR10(root="./cifar10", train=True, download=True)
  test  = torchvision.datasets.CIFAR10(root="./cifar10", train=False, download=True)
  ret = [torch.tensor(i, device=device) for i in (train.data, train.targets, test.data, test.targets)]
  std, mean = torch.std_mean(ret[0].float(),dim=(0,1,2),unbiased=True,keepdim=True)
  for i in [0,2]: ret[i] = ((ret[i]-mean)/std).to(dtype).permute(0,3,1,2)
  return ret

def loadLinearData(device):
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import fetch_openml

    # Load the Boston Housing Dataset
    boston = fetch_openml(name='boston', version=1)

    # Creating a DataFrame
    boston_df = pd.DataFrame(boston.data, columns=boston.feature_names)

    # Adding the target variable to the DataFrame
    boston_df['PRICE'] = boston.target

    # Display the first few rows of the dataset
    print(boston_df.head())

    # Selecting multiple features
    features = ['CRIM', 'RM', 'AGE', 'DIS', 'LSTAT']
    X = boston_df[features].values
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    y = boston_df['PRICE'].values
    y = (y - y.mean()) / y.std() 

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return (torch.tensor(X_train, dtype=torch.float32).to(device), 
            torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device), 
            torch.tensor(X_test, dtype=torch.float32).to(device), 
            torch.tensor(y_test, dtype=torch.float32).view(-1, 1).to(device))



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

def getBatches(X, y, batch_size=-1):
    length = len(X)
    if batch_size == -1:
        batch_size = BATCH_SIZE
    if batch_size > length:
        # pad X to next multiple of batch_size
        repeat = (batch_size + length - 1) // length
        X = X.repeat(repeat, 1)
        y = y.repeat(repeat, 1)
        length = len(X)
    idx = torch.randperm(length)
    X = X[idx]
    y = y[idx]
    for i in range(0, length, batch_size):
        if i + batch_size > length:
            break
        yield X[i:i+batch_size], y[i:i+BATCH_SIZE]


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

class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(5, 1)
        self.input_size = (5,)

    def forward(self, x):
        return self.linear(x)

    def loss_acc(self, x, y):
        output = self(x)
        loss = ((output - y)**2).mean()
        return {"loss": loss, "acc": 0}

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
        self.input_size = (28 * 28,)
        self.linear1 = nn.Linear(28 * 28, 100)
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

class MNIST_MLP_500(nn.Module):
    def __init__(self):
        super(MNIST_MLP_500, self).__init__()
        self.input_size = (28 * 28,)
        self.linear1 = nn.Linear(28 * 28, 500)
        self.linear2 = nn.Linear(500, 10)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.relu(self.linear1(x))
        return self.linear2(x)

    def loss_acc(self, x, y):
        output = self(x)
        output = self.softmax(output)
        assert not torch.any(torch.isnan(output))
        loss = self.criterion(output, y)
        pred = output.argmax(dim=1, keepdim=True)
        acc = pred.eq(y.view_as(pred)).sum().item() / len(y)
        return {"loss": loss, "acc": acc}

class MNIST_CNN(nn.Module):
    def __init__(self):
        super(MNIST_CNN, self).__init__()
        self.input_size = (1, 28, 28)
        self.conv1 = nn.Conv2d(1, 64, 3, 3)
        self.conv2 = nn.Conv2d(64, 64, 3, 3)
        self.conv3 = nn.Conv2d(64, 64, 3, 3)
        self.adaptive_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.head = nn.Linear(64, 10)
        self.softmax = nn.Softmax(dim=1)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = self.adaptive_pool(x)
        x = x.view(-1, 64)
        return self.head(x)
    
    def loss_acc(self, x, y):
        output = self(x)
        output = self.softmax(output)
        loss = self.criterion(output, y)
        pred = output.argmax(dim=1, keepdim=True)
        acc = pred.eq(y.view_as(pred)).sum().item() / len(y)
        return {"loss": loss, "acc": acc}

class CNN(nn.Module):
  def __init__(self):
    super().__init__()

    dims = [3,64,128,128,128,256,512,512,512]
    seq = []
    for i in range(len(dims)-1):
      c_in,c_out = dims[i],dims[i+1]
      seq.append( nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False) )
      if c_out == c_in * 2:
        seq.append( nn.MaxPool2d(2) )
      seq.append( nn.BatchNorm2d(c_out) )
      seq.append( nn.CELU(alpha=0.075) )
    self.seq = nn.Sequential(*seq, nn.MaxPool2d(4), nn.Flatten(), nn.Linear(dims[-1], 10, bias=False))

  def forward(self, x):
    x = self.seq(x) / 8
    return x
        
if args.model == "logistic":
    Model = LogisticRegression
elif args.model == "mlp":
    Model = MLP
elif args.model == "mnist_mlp":
    Model = MNIST_MLP
elif args.model == "mnist_mlp_500":
    Model = MNIST_MLP_500
elif args.model == "mnist_cnn":
    Model = MNIST_CNN
elif args.model == "linear":
    Model = LinearRegression
else:
    raise ValueError("Unknown model")

if args.dataset == "mnist":
    X_train, y_train, X_test, y_test = loadMNISTData(device)
elif args.dataset == "cifar10":
    X_train, y_train, X_test, y_test = loadCIFAR10(device)
elif args.dataset == "linear":
    X_train, y_train, X_test, y_test = loadLinearData(device)
else:
    X_train, y_train, X_test, y_test = loadSmallTableData(device)

master_weight = Model()
master_weight = master_weight.to(dtype).to(device)
summary(master_weight, master_weight.input_size)
full_precision_number = qtorch.FloatingPoint(8, 23)
act_number = qtorch.FloatingPoint(8, ACT_MAN_WIDTH)
weight_number = qtorch.FloatingPoint(8, WEIGHT_MAN_WIDTH)
back_number = qtorch.FloatingPoint(8, args.back_man_width)
master_weight = utils.replace_linear_with_quantized(master_weight, full_precision_number, full_precision_number, "stochastic")
model_weight = copy.deepcopy(master_weight)
model_weight = utils.apply_number_format(model_weight, act_number, back_number, args.act_rounding, args.act_rounding, args.same_input)

if args.opt == "sgd":
    opt = torch.optim.SGD(master_weight.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
elif args.opt == "lamb":
    opt = Lamb(master_weight.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ConstantLR(opt)

training_time = stepi = 0
bar = tqdm.tqdm(range(DESIRED_STEPS))
wrapper = utils.MasterWeightOptimizerWrapper(master_weight, model_weight, 
                                       opt, scheduler, 
                                       weight_quant=qtorch.quant.quantizer(weight_number, forward_rounding=args.weight_rounding),
                                       grad_acc_steps=GRAD_ACC_STEPS)

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
wandb.watch(master_weight, log="all", log_freq=wandbconfig.watch_interval)
epoch = 0
while True:
    epoch += 1
    start_time = time.perf_counter()
    train_losses, train_accs = [], []
    master_weight.train()
    model_weight.train()

    for X,y in getBatches(X_train,y_train):
        stepi += 1
        # result_log.update(wrapper.train_compare_true_gradient(X, y, X_train, y_train))
        result_log.update(wrapper.train_on_batch(X, y))
        bar.update(1)
        bar.set_postfix(result_log)
        result_log["lr"] = opt.param_groups[0]["lr"]
        if stepi % (args.log_interval * 4) == 0:
            test_result_m = test(master_weight, ((X_test, y_test),))
            grad_entire = ema_grad_on_dataset(master_weight, X_train[:2048], y_train[:2048])
            result_log.update(grad_entire)
        if stepi % args.log_interval == 0:
            wandb.log(result_log | test_result_m)
        if stepi >= PRECISION_SCHEDULING_STEP:
            model_weight = utils.apply_number_format(model_weight, full_precision_number, full_precision_number, "stochastic", "stochastic")
            wrapper.weight_quant = lambda x: x
            PRECISION_SCHEDULING_STEP = float("inf")
        if stepi >= DESIRED_STEPS:
            test_result_m = test(master_weight, ((X_test, y_test),))
            grad_entire = ema_grad_on_dataset(master_weight, X_train[:2048], y_train[:2048])
            result_log.update(grad_entire)
            wandb.log(result_log | test_result_m)
            exit()

    # wrapper.master_params_to_model_params(quantize=False)
    training_time += time.perf_counter()-start_time
    test_result_m = test(master_weight, ((X_test, y_test),))
    print(f'epoch % 2d  test loss m %.3f   test acc m %.3f  training time %.2f'%(epoch+1, test_result_m["test_loss"], test_result_m["test_acc"], training_time))
  