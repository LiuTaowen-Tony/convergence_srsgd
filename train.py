import wandb
import qtorch.quant
import tqdm
import torch, torchvision, sys, time, numpy
from torch import nn
import torch.nn.functional as F
import copy
import qtorch
from qtorch.quant import Quantizer
import math

device = torch.device("cuda")
dtype = torch.float32

wandb.init(project="low_precision_cifar10")

sweep_config = {
   "method": "bayes"
}
metric = {"name": "test_loss", "goal": "minimize"}
parameter_dict = {
   "batch_size": {"distribution": "q_log_uniform_values", "q": 8, "low": 256, "high": 1024},
   "lr_factor": {"distribution": "uniform", "min": 5e-5, "max": 5e-3},
   "pct_start": {"distribution": "uniform", "min": 0.05, "max": 0.5},
   "grad_acc_steps": {"distribution": "q_log_uniform_values", "q": 8, "low": 1, "high": 64},

}
wandbconfig = wandb.config

STEPS = 4882
DATASET_SIZE = 50000
BATCH_SIZE = wandbconfig.batch_size
EPOCHS = math.ceil(STEPS / (DATASET_SIZE / BATCH_SIZE))
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
LR = wandbconfig.lr_factor * BATCH_SIZE
GRAD_ACC_STEPS = wandbconfig.grad_acc_steps
W,H = 32,32
CUTSIZE = 8
CROPSIZE = 4
MAN_WIDTH = 0
PCT_START = wandbconfig.pct_start


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

def loadCIFAR10(device):
  train = torchvision.datasets.CIFAR10(root="./cifar10", train=True, download=True)
  test  = torchvision.datasets.CIFAR10(root="./cifar10", train=False, download=True)
  ret = [torch.tensor(i, device=device) for i in (train.data, train.targets, test.data, test.targets)]
  std, mean = torch.std_mean(ret[0].float(),dim=(0,1,2),unbiased=True,keepdim=True)
  for i in [0,2]: ret[i] = ((ret[i]-mean)/std).to(dtype).permute(0,3,1,2)
  return ret

def getBatches(X,y, istrain):
  if istrain:
    perm = torch.randperm(len(X), device=device)
    X,y = X[perm],y[perm]

    Crop = ([(y0,x0) for x0 in range(CROPSIZE+1) for y0 in range(CROPSIZE+1)], 
        lambda img, y0, x0 : nn.ReflectionPad2d(CROPSIZE)(img)[..., y0:y0+H, x0:x0+W])
    FlipLR = ([(True,),(False,)], 
        lambda img, choice : torch.flip(img,[-1]) if choice else img)
    def cutout(img,y0,x0):
      img[..., y0:y0+CUTSIZE, x0:x0+CUTSIZE] = 0
      return img
    Cutout = ([(y0,x0) for x0 in range(W+1-CUTSIZE) for y0 in range(H+1-CUTSIZE)], cutout)

    for options, transform in (Crop, FlipLR, Cutout):
      optioni = torch.randint(len(options),(len(X),), device=device)
      for i in range(len(options)):
        X[optioni==i] = transform(X[optioni==i], *options[i])

  return ((X[i:i+BATCH_SIZE], y[i:i+BATCH_SIZE]) for i in range(0,len(X) - istrain*(len(X)%BATCH_SIZE),BATCH_SIZE))

def grad_on_dataset(network, data, target):
    network.train()
    total_norm = 0
    network.zero_grad()
    idx = torch.randperm(len(data))[:2048]
    data = data[idx]
    target = target[idx]
    output = network(data)
    loss = torch.nn.CrossEntropyLoss()(output, target)
    loss.backward()
    total_norm = nn.utils.clip_grad_norm_(network.parameters(), float('inf'))
    network.zero_grad()
    network.eval()
    return {"grad_norm_entire": total_norm.item()}

class QuantisedWrapper(nn.Module):
    def __init__(self, network, fnumber, bnumber, round_mode):
        super(QuantisedWrapper, self).__init__()
        self.network = network
        self.quant = Quantizer(forward_number=fnumber, forward_rounding=round_mode)
        self.quant_b = Quantizer(backward_number=bnumber, backward_rounding=round_mode)

    def forward(self, x):
        assert torch.all(x.isnan() == False)
        before = self.quant(x)
        assert torch.all(before.isnan() == False)
        a = self.network(before)
        assert torch.all(a.isnan() == False)
        after = self.quant_b(a)
        return after

def replace_linear_with_quantized(network, fnumber, bnumber, round_mode):
    # Create a temporary list to store the modules to replace
    to_replace = []
    
    # Iterate to collect modules that need to be replaced
    for name, module in network.named_children():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            layer = QuantisedWrapper(module, fnumber, bnumber, round_mode)
            to_replace.append((name, layer))
        else:
            replace_linear_with_quantized(module, fnumber, bnumber, round_mode)
    
    for name, new_module in to_replace:
        setattr(network, name, new_module)
    
    return network

class MasterWeightOptimizerWrapper():
    def __init__(
            self,
            master_weight,
            model_weight,
            optimizer,
            scheduler,
            weight_quant,
            grad_acc_steps=1,
            grad_clip=float("inf"),
            grad_scaling=1.0,
    ):
        self.master_weight = master_weight
        self.model_weight = model_weight
        self.optimizer = optimizer
        self.grad_scaling = grad_scaling
        self.grad_clip = grad_clip
        self.scheduler = scheduler
        self.weight_quant = weight_quant
        self.grad_acc_steps = grad_acc_steps
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=0.2)

    # --- for mix precision training ---
    def model_grads_to_master_grads(self):
        for model, master in zip(self.model_weight.parameters(), self.master_weight.parameters()):
            if master.grad is None:
                master.grad = master.data.new(*master.data.size())
                master.grad.data.copy_(model.grad.data)
            else:
                master.grad.data.add_(model.grad.data)

    def master_grad_apply(self, fn):
        for master in (self.master_weight.parameters()):
            if master.grad is None:
                master.grad = fn(master.data.new(*master.data.size()))
            master.grad.data = fn(master.grad.data)

    def master_params_to_model_params(self, quantize=True):
        for model, master in zip(self.model_weight.parameters(), self.master_weight.parameters()):
            if quantize:
                model.data.copy_(self.weight_quant(master.data))
            else:
                model.data.copy_(master.data)

    def train_on_batch(self, data, target):
        self.master_weight.zero_grad()
        self.model_weight.zero_grad()
        output = self.model_weight(data)
        loss = self.loss_fn(output, target)
        loss = loss * self.grad_scaling
        loss.backward()
        acc = (output.argmax(dim=1) == target).float().mean()
        loss /= self.grad_acc_steps
        self.model_grads_to_master_grads()
        self.master_grad_apply(lambda x: x / self.grad_scaling)
        grad_norm = nn.utils.clip_grad_norm_(self.master_weight.parameters(), self.grad_clip)
        self.optimizer.step()
        self.scheduler.step()
        self.master_params_to_model_params()
        return {"loss": loss.item(), "grad_norm": grad_norm.item(), "acc": acc.item()}

    def train_with_grad_acc(self, data, target):
        self.master_weight.zero_grad()
        self.model_weight.zero_grad()
        ds = torch.chunk(data, self.grad_acc_steps)
        ts = torch.chunk(target, self.grad_acc_steps)
        losses = []
        acces = []
        self.grad = {}
        for d, t in zip(ds, ts):
            self.master_params_to_model_params()
            output = self.model_weight(d)
            loss = self.loss_fn(output, t)
            loss = loss * self.grad_scaling
            loss.backward()
            acces.append((output.argmax(dim=1) == t).float().mean().item())
            losses.append(loss.item())
        self.model_grads_to_master_grads()
        self.master_grad_apply(lambda x: x / self.grad_scaling / self.grad_acc_steps)
        grad_norm = nn.utils.clip_grad_norm_(self.master_weight.parameters(), self.grad_clip)
        self.optimizer.step()
        self.scheduler.step()
        return {"loss": numpy.mean(losses),  "acc": numpy.mean(acces), "grad_norm": grad_norm.item()}

        


X_train, y_train, X_test, y_test = loadCIFAR10(device)

master_weight = CNN().to(dtype).to(device)
number = qtorch.FloatingPoint(8, MAN_WIDTH)
master_weight = replace_linear_with_quantized(master_weight, number, number, "stochastic")
model_weight = copy.deepcopy(master_weight)
print(master_weight)

opt = torch.optim.SGD(master_weight.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY, nesterov=True)
scheduler = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, epochs=EPOCHS, steps_per_epoch=len(X_train)//BATCH_SIZE, pct_start=PCT_START, anneal_strategy='linear',
                                                div_factor=1e16, final_div_factor=0.07)

training_time = stepi = 0
result_log = {}
bar = tqdm.tqdm(range(EPOCHS * len(X_train) // BATCH_SIZE))
wrapper = MasterWeightOptimizerWrapper(master_weight, model_weight, opt, scheduler, weight_quant=qtorch.quant.quantizer(number), grad_acc_steps=GRAD_ACC_STEPS)

def test(network, dataset):
    network.eval()
    correct = 0
    total_loss = 0
    criterion = torch.nn.CrossEntropyLoss()
    i = 0
    n = 0
    with torch.no_grad():
        for data, target in dataset:
            i += 1
            n += len(data)
            output = network(data)
            total_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    accuracy = correct / n
    avg_loss = total_loss / i
    network.train()
    return {"test_acc": accuracy, "test_loss": avg_loss}
                                       
test_result_m = {}
for epoch in range(EPOCHS):
    start_time = time.perf_counter()
    train_losses, train_accs = [], []
    master_weight.train()
    model_weight.train()

    for X,y in getBatches(X_train,y_train, True):
        stepi += 1
        result_log.update(wrapper.train_with_grad_acc(X, y))
        bar.update(1)
        bar.set_postfix(result_log)
        wandb.log(result_log | test_result_m)

    wrapper.master_params_to_model_params(quantize=False)
    result_log.update(grad_on_dataset(model_weight, X_train, y_train))

    training_time += time.perf_counter()-start_time

    test_result_m = test(model_weight, getBatches(X_test, y_test, False))

    print(f'epoch % 2d  test loss m %.3f   test acc m %.3f  training time %.2f'%(epoch+1, test_result_m["test_loss"], test_result_m["test_acc"], training_time))
  