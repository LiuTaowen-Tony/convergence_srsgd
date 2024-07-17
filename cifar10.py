import numpy as np
import torch, torchvision, sys, time, numpy
from argparse import ArgumentParser
import models 
import load_data
from low_precision_utils import utils

device = torch.device("cuda")
dtype = torch.float32

parser = ArgumentParser()
parser.add_argument("--experiment_name", type=str)
parser.add_argument("--lr", type=float, default=0.6)
parser.add_argument("--act_man_width", type=int, default=23)
parser.add_argument("--weight_man_width", type=int, default=23)
parser.add_argument("--back_man_width", type=int, default=23)
#parser.add_argument
parser.add_argument("--weight_rounding", type=str, default="stochastic")
parser.add_argument("--act_rounding", type=str, default="stochastic")
parser.add_argument("--back_rounding", type=str, default="stochastic")
parser.add_argument("--same_input", type=lambda x: (str(x).lower() == 'true'), default=False)
parser.add_argument("--same_weight", type=lambda x: (str(x).lower() == 'true'), default=False)
parser.add_argument("--epochs",type=int, default=24)

args = parser.parse_args()

EPOCHS = args.epochs
BATCH_SIZE = 512
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4

model = models.CNN().to(device)
X_train, y_train, X_test, y_test = load_data.loadCIFAR10(device)
opt = torch.optim.SGD(model.parameters(), lr=0.0, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY, nesterov=True)
scheduler = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=0.6, total_steps=EPOCHS*len(X_train)//BATCH_SIZE, pct_start=0.2)


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
