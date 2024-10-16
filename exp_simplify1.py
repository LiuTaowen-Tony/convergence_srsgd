import wandb
import pandas as pd
from torch import nn
import qtorch.quant
import tqdm
import torch
import time
import qtorch

from low_precision_utils import quant
from low_precision_utils import metrics

device = torch.device("cuda")
dtype = torch.float32
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--experiment_name", type=str)
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--act_man_width", type=int, default=8)
parser.add_argument("--weight_man_width", type=int, default=8)
parser.add_argument("--back_man_width", type=int, default=8)
parser.add_argument("--batch_size", type=int, default=31)
args = parser.parse_args()

wandb.init(project=args.experiment_name, config=args)


DATASET_SIZE = 404
DESIRED_STEPS = 10_000

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
        batch_size = batch_size
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
        yield X[i:i+batch_size], y[i:i+batch_size]

X_train, y_train, X_test, y_test = loadLinearData(device)


def grad(network: quant.QuantWrapper, x, y, require_grad_norm=True):
    network.train()
    total_norm = 0
    network.zero_grad()
    y_pred = network(x)
    loss = ((y - y_pred) ** 2).mean()
    loss.backward()
    result = {"loss": loss.item()}
    if require_grad_norm:
        total_norm = nn.utils.clip_grad_norm_(network.parameters(), float('inf'))
        result["grad_norm_entire"] = total_norm.item()
    return result


training_time = stepi = 0
bar = tqdm.tqdm(range(DESIRED_STEPS))
low_scheme = quant.QuantScheme(
    quant.FP32,
    quant.FP32,
    quant.NPoint(args.act_man_width),
    quant.NPoint(args.weight_man_width),
    quant.NPoint(args.act_man_width),
    quant.NPoint(args.act_man_width)
    # quant.FPQuant(8, args.act_man_width),
    # quant.FPQuant(8, args.weight_man_width),
    # quant.FPQuant(8, args.act_man_width),
    # quant.FPQuant(8, args.act_man_width)
)

result_log = {}

epoch = 0

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(5, 50)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(50, 50)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(50, 1)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x) 
        x = self.fc2  (x) 
        x = self.relu2(x) 
        x = self.fc3  (x) 
        return x

network = Net()
network.to(device)
network = quant.QuantWrapper(network)
sgd = torch.optim.SGD(network.parameters(), lr=args.lr)
logger = metrics

while True:
    epoch += 1
    for X,y in getBatches(X_train,y_train, args.batch_size):
        stepi += 1
        if stepi % 1 == 0:
            lp_result = grad(network, X, y, True)
            sgd.step()
            network.apply_quant_scheme(quant.FP32_SCHEME)
            full_result = grad(network, X, y, True)
            network.apply_quant_scheme(low_scheme)
            result_log["lp_grad_loss"] = lp_result["loss"]
            result_log["full_grad_loss"] = full_result["loss"]
            result_log["full_grad_norm"] = full_result["grad_norm_entire"]
            result_log["lp_grad_norm"] = lp_result["grad_norm_entire"]
            wandb.log(result_log)
        else:
            result = grad(network, X, y, False)
            result_log["loss"] = result["loss"]
            # print(
            #     f"""lp loss: {loss.item()}, f loss {full_loss.item()}, lp grad norm: {lp_grad_norm.item()}, f grad norm: {full_grad_norm.item()}"""
            # )
        bar.update(1)
        bar.set_postfix(result_log)

        if stepi >= DESIRED_STEPS:
            exit()


        

