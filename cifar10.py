import json
import numpy as np
import torch, torchvision, sys, time, numpy
from argparse import ArgumentParser
import models 
import load_data
import wandb
from low_precision_utils import utils
from low_precision_utils import log_util

device = torch.device("cuda")
dtype = torch.float32

parser = ArgumentParser()
parser.add_argument("--lr", type=float, default=0.6)
parser.add_argument("--epochs",type=int, default=24)
parser = utils.add_argparse(parser)
parser = log_util.add_log_args(parser, "cifar10", default_log_interval=1, default_wandb_interval=1, default_wandb_watch_interval=1)
args = parser.parse_args()


quant_scheme = utils.QuantScheme.from_args(args)
args.quant_scheme = json.loads(args.quant_scheme_json)
logger = log_util.Logger.from_args(args)

EPOCHS = args.epochs
BATCH_SIZE = 512
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4

model = models.CNN().to(device)
model = utils.QuantWrapper(model, quant_scheme)
X_train, y_train, X_test, y_test = load_data.loadCIFAR10(device)
opt = torch.optim.SGD(model.parameters(), lr=0.0, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY, nesterov=True)
scheduler = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=0.6, total_steps=EPOCHS*len(X_train)//BATCH_SIZE, pct_start=0.2)

# logger.wandb_watch(model)
print(model)

training_time = stepi = 0
for epoch in range(EPOCHS):
  start_time = time.perf_counter()
  train_losses, train_accs = [], []
  model.train()
  X_aug, y_aug = load_data.augment_data(X_train, y_train)
  for X,y in load_data.getBatches(X_aug,y_aug, BATCH_SIZE):
    stepi += 1
    loss,acc = model.loss_acc(X,y).values()
    model.zero_grad()
    loss.backward()
    opt.step()
    scheduler.step()
    
    train_losses.append(loss.item())
    train_accs.append(acc)
  training_time += time.perf_counter()-start_time

  with torch.no_grad():
    test_accs = []
    test_losses = []
    for X,y in load_data.getBatches(X_test,y_test, BATCH_SIZE):
      loss,acc = model.loss_acc(X, y).values()
      test_losses.append(loss.item())
      test_accs.append(acc)

  print(f'epoch % 2d  train loss %.3f  train acc %.2f  training time %.2f'%(epoch+1, np.mean(train_losses), np.mean(train_accs),  training_time))
  print(f'epoch % 2d  test loss %.3f  test acc %.2f '%(epoch+1, np.mean(test_losses), np.mean(test_accs)))
  logger.log({
    "train_loss": np.mean(train_losses),
    "train_acc": np.mean(train_accs),
    "test_loss": np.mean(test_losses),
    "test_acc": np.mean(test_accs)
  })
  # logger.log_hists(model, X, y)
