import argparse
from ml_utils import log_util
import low_precision_utils as lputils
import low_precision_utils.quant as quant
import torch
import torch.nn as nn
from utils import load_data


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(10, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


parser = argparse.ArgumentParser()
parser = quant.add_argparse(parser)
parser = log_util.add_log_args(parser, "backward_no_bias")
parser.add_argument("--bs", type=int, default=32)
parser.add_argument("--epochs", type=int, default=100)

args = parser.parse_args()
log_util.Logger.from_args(args)

model = Model()

quant_scheme = quant.QuantScheme.from_args(args)
model = quant.QuantWrapper(model, quant_scheme=quant_scheme)
X_train, y_train, X_test, y_test = load_data.get_data_loader("small_table")("cuda")
opt = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(args.epochs):
    for X, y in load_data.getBatches(X_train, y_train, args.bs):
        model.zero_grad()
        loss = model.loss_acc(X, y)["loss"]
        loss.backward()
        opt.step()

    






