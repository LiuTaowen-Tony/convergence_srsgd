import wandb
import tqdm
import torch
import time

from argparse import ArgumentParser

import qtorch
import qtorch.quant
from torchsummary import summary

from low_precision_utils import quant
from low_precision_utils.metrics import *

import models
import load_data

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



if args.test_nan:
    quant.TEST_NAN = True

PRECISION_SCHEDULING_STEP = args.precision_scheduling * args.steps
args.weight_decay = 0

dtype = torch.float32
device = torch.device("cuda")

full_precision_number = qtorch.FloatingPoint(8, 23)
act_number = qtorch.FloatingPoint(8, args.act_man_width)
weight_number = qtorch.FloatingPoint(8, args.weight_man_width)
back_number = qtorch.FloatingPoint(8, args.back_man_width)
quant = quant.QuantScheme(act_number, back_number, weight_number, 
                                 args.act_rounding, args.back_rounding, args.weight_rounding,
                                 args.same_input, args.same_weight)

data_loader = load_data.get_data_loader(args.dataset)
X_train, y_train, X_test, y_test = data_loader(device)

model = models.get_model(args.model)  
model = model.to(dtype).to(device)
summary(model, model.input_size)
model = quant.replace_with_quantized(model, quant)
model = quant.apply_quant_scheme(model, quant)

if args.opt == "sgd":
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
elif args.opt == "lamb":
    opt = torch.optim.Lamb(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.ConstantLR(opt)

wandb.init(project=args.experiment_name, config=args)

@torch.no_grad()
def test(network: nn.Module, dataset):
    network.eval()
    correct = 0
    total_loss = 0
    i = 0
    n = 0
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

bar = tqdm.tqdm(range(args.steps))
stepi = 0
test_result = {}
result_log = {}
grad_log = {}
result_log = {}

while True:
    for X,y in load_data.getBatches(X_train,y_train,args.batch_size):
        model.train()
        stepi += 1
        model.zero_grad()
        loss, acc = model.loss_acc(X, y).values()
        loss.backward()
        opt.step()
        scheduler.step()
        result_log.update({
            "loss": loss.item(),
            "acc": acc,
            "lr": opt.param_groups[0]["lr"]
        })
        if stepi % (args.log_interval * 10) == 0:
            test_result = test(model, ((X_test, y_test),))
        if stepi % args.log_interval == 0:
            wandb.log(result_log | test_result)
        if stepi >= args.steps:
            test_result = test(model, ((X_test, y_test),))
            wandb.log(result_log | test_result)
            exit()
