import tqdm
import torch
from torch import nn

from argparse import ArgumentParser

from low_precision_utils import quant
from low_precision_utils import metrics
from ml_utils import log_util

from utils import models
from utils import load_data

parser = ArgumentParser()

parser.add_argument("--model", type=str, default="mnist_mlp")
parser.add_argument("--dataset", type=str, default="mnist")
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--weight_decay", type=float, default=0.0)
parser.add_argument("--momentum", type=float, default=0.0)
parser.add_argument("--steps", type=int, default=20000)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--grad_acc_steps", type=int, default=1)
log_util.add_log_args(parser)
quant.add_argparse(parser)
args = parser.parse_args()

quant_scheme = quant.QuantScheme.from_args(args)
logger = log_util.Logger.from_args(args)

dtype = torch.float32
device = torch.device("cuda")


data_loader = load_data.get_data_loader(args.dataset)
X_train, y_train, X_test, y_test = data_loader(device)

model = models.get_model(args.model)
model = model.to(dtype).to(device)
model = quant.replace_with_quantized(model, quant_scheme)
model = quant.apply_quant_scheme(model, quant_scheme)
print(model)

opt = torch.optim.SGD(model.parameters(), lr=args.lr / args.batch_size, momentum=args.momentum, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.ConstantLR(opt)

@torch.no_grad()
def test(network: nn.Module):
    network.eval()
    correct = 0
    total_loss = 0
    i = 0
    for data, target in ((X_test, y_test),):
        i += 1
        loss_acc = network.loss_acc(data, target)
        total_loss += loss_acc["loss"].item()
        correct += loss_acc["acc"] * len(data)
    accuracy = correct / len(X_test)
    avg_loss = total_loss / i
    network.train()
    return {"test_acc": accuracy, "test_loss": avg_loss}

def grad_norm(network: models.MNIST_MLP):
    network.zero_grad()
    network.loss_acc(X_train[:2048], y_train[:2048])["loss"].backward()
    grad_monitor = network.linear1.weight.grad
    network.zero_grad()
    return {"grad_norm": grad_monitor.norm().item()}

bar = tqdm.tqdm(range(args.steps))
stepi = 0

while True:
    for X,y in load_data.getBatches(X_train,y_train,args.batch_size):
        model.train()
        stepi += 1
        model.zero_grad()
        loss, acc = model.loss_acc(X, y).values()
        loss = loss * args.batch_size
        loss.backward()
        opt.step()
        scheduler.step()
        logger.log({
            "loss": loss.item() / args.batch_size,
            "acc": acc,
            "lr": opt.param_groups[0]["lr"]
        })
        bar.update(1)
        bar.set_postfix(loss=loss.item() / args.batch_size, acc=acc, lr=opt.param_groups[0]["lr"])
        if stepi % (args.wandb_interval) == 0 or stepi >= args.steps:
            quant.apply_quant_scheme(model, quant.FP32_SCHEME)
            test_result = test(model)
            norm = grad_norm(model)
            print(norm)
            quant.apply_quant_scheme(model, quant_scheme)
            logger.log_same_iter(test_result)
            logger.log_same_iter({"grad_norm": norm})
        if stepi >= args.steps:
            exit()
