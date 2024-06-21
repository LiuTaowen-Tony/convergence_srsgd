import wandb
import pandas as pd
import qtorch.quant
import tqdm
import torch
import time
import qtorch

from low_precision_utils import utils
from low_precision_utils.metrics import *

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
parser.add_argument("--log_interval", type=int, default=100)
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
DATASET_SIZE = 404
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

def loadLinearData(device):
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import fetch_openml

    # Load the Boston Housing Dataset
    boston = fetch_openml(name='boston', version=1)

    # Creating a DataFrame
    boston_df = pd.DataFrame(boston.data, columns=boston.feature_names)

    # Adding the target variable to the DataFrame
    boston_df['PRICE'] = boston.target

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

X_train, y_train, X_test, y_test = loadLinearData(device)

def lp_grad(w, b, x, y):
    y = y.view(-1, 1)
    x_q = qtorch.quant.float_quantize(x, 8, args.act_man_width, "stochastic")
    x_q2 = qtorch.quant.float_quantize(x, 8, args.act_man_width, "stochastic")

    y_pred = x_q @ w + b
    loss = ((y_pred - y) ** 2).mean()
    y_pred_grad = 2 * (y_pred - y) / len(y)
    y_pred_grad_q = qtorch.quant.float_quantize(y_pred_grad, 8, args.back_man_width, "stochastic")

    if args.same_input:
        w_grad = x_q.t() @ y_pred_grad_q
    else:
        w_grad = x_q2.t() @ y_pred_grad_q
    b_grad = y_pred_grad_q.sum()
    return w_grad, b_grad, loss

def full_precision_grad(w, b, x, y):
    y_pred = x @ w + b
    loss = ((y_pred - y) ** 2).mean()
    y_pred_grad = 2 * (y_pred - y) / len(y)
    w_grad = x.T @ y_pred_grad 
    b_grad = y_pred_grad.sum()
    return w_grad, b_grad, loss


training_time = stepi = 0
bar = tqdm.tqdm(range(DESIRED_STEPS))

ema_metrics = EMAMetrics()
result_log = {}

epoch = 0
w = torch.randn(5, 1, requires_grad=False, device=device)
b = torch.randn(1, requires_grad=False, device=device)

while True:
    epoch += 1
    for X,y in getBatches(X_train,y_train):
        stepi += 1
        grad_w, grad_b, loss = lp_grad(w, b, X, y)
        w -= LR * grad_w
        b -= LR * grad_b
        result_log["loss"] = loss.item()

        if stepi % wandbconfig.log_interval == 0:
            lp_grad_w, lp_grad_b, lp_loss = lp_grad(w, b, X_train, y_train)
            # lp_grad on training set
            full_grad_w, full_grad_b, full_loss = full_precision_grad(w, b, X_train, y_train)
            # full_grad on training set
            lp_grad_norm = torch.sqrt((lp_grad_w ** 2).sum() + (lp_grad_b ** 2).sum())
            full_grad_norm = torch.sqrt((full_grad_w ** 2).sum() + (full_grad_b ** 2).sum())

            result_log["lp_grad_loss"] = lp_loss.item()
            result_log["full_grad_loss"] = full_loss.item()
            result_log["lp_grad_norm"] = lp_grad_norm.item()
            result_log["full_grad_norm"] = full_grad_norm.item()
            wandb.log(result_log)
            # print(
            #     f"""lp loss: {loss.item()}, f loss {full_loss.item()}, lp grad norm: {lp_grad_norm.item()}, f grad norm: {full_grad_norm.item()}"""
            # )
        bar.update(1)
        bar.set_postfix(result_log)

        if stepi >= DESIRED_STEPS:
            exit()


        

