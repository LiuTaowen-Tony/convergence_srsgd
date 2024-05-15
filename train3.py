import timm
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

import torch
from torch.optim.lr_scheduler import _LRScheduler
from optimizers.Signum import Signum

class CosineAnnealingLinearWarmup(_LRScheduler):
    def __init__(
            self,
            optimizer: torch.optim.Optimizer,
            first_cycle_steps: int,
            min_lrs: list[float] = None,
            cycle_mult: float = 1.,
            warmup_steps: int = 0,
            gamma: float = 1.,
            last_epoch: int = -1,
            min_lrs_pow: int = None,
                 ):

        '''
        :param optimizer: warped optimizer
        :param first_cycle_steps: number of steps for the first scheduling cycle
        :param min_lrs: same as eta_min, min value to reach for each param_groups learning rate
        :param cycle_mult: cycle steps magnification
        :param warmup_steps: number of linear warmup steps
        :param gamma: decreasing factor of the max learning rate for each cycle
        :param last_epoch: index of the last epoch
        :param min_lrs_pow: power of 10 factor of decrease of max_lrs (ex: min_lrs_pow=2, min_lrs = max_lrs * 10 ** -2
        '''
        assert warmup_steps < first_cycle_steps, "Warmup steps should be smaller than first cycle steps"
        assert min_lrs_pow is None and min_lrs is not None or min_lrs_pow is not None and min_lrs is None, \
            "Only one of min_lrs and min_lrs_pow should be specified"
        
        # inferred from optimizer param_groups
        max_lrs = [g["lr"] for g in optimizer.state_dict()['param_groups']]

        if min_lrs_pow is not None:
            min_lrs = [i * (10 ** -min_lrs_pow) for i in max_lrs]

        if min_lrs is not None:
            assert len(min_lrs)==len(max_lrs),\
                "The length of min_lrs should be the same as max_lrs, but found {} and {}".format(
                    len(min_lrs), len(max_lrs)
                )

        self.first_cycle_steps = first_cycle_steps  # first cycle step size
        self.cycle_mult = cycle_mult  # cycle steps magnification
        self.base_max_lrs = max_lrs  # first max learning rate
        self.max_lrs = max_lrs  # max learning rate in the current cycle
        self.min_lrs = min_lrs  # min learning rate
        self.warmup_steps = warmup_steps  # warmup step size
        self.gamma = gamma  # decrease rate of max learning rate by cycle

        self.cur_cycle_steps = first_cycle_steps  # first cycle step size
        self.cycle = 0  # cycle count
        self.step_in_cycle = last_epoch  # step size of the current cycle

        super().__init__(optimizer, last_epoch)

        assert len(optimizer.param_groups) == len(self.max_lrs),\
            "Expected number of max learning rates provided ({}) to be the same as the number of groups parameters ({})".format(
                len(max_lrs), len(optimizer.param_groups))
        
        assert len(optimizer.param_groups) == len(self.min_lrs),\
            "Expected number of min learning rates provided ({}) to be the same as the number of groups parameters ({})".format(
                len(max_lrs), len(optimizer.param_groups))

        # set learning rate min_lr
        self.init_lr()

    def init_lr(self):
        self.base_lrs = []
        for i, param_groups in enumerate(self.optimizer.param_groups):
            param_groups['lr'] = self.min_lrs[i]
            self.base_lrs.append(self.min_lrs[i])

    def get_lr(self):
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            return [(max_lr - base_lr) * self.step_in_cycle / self.warmup_steps + base_lr for (max_lr, base_lr) in
                    zip(self.max_lrs, self.base_lrs)]
        else:
            return [base_lr + (max_lr - base_lr) \
                    * (1 + math.cos(math.pi * (self.step_in_cycle - self.warmup_steps) \
                                    / (self.cur_cycle_steps - self.warmup_steps))) / 2
                    for (max_lr, base_lr) in zip(self.max_lrs, self.base_lrs)]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = int(
                    (self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult) + self.warmup_steps
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(math.log((epoch / self.first_cycle_steps * (self.cycle_mult - 1) + 1), self.cycle_mult))
                    self.cycle = n
                    self.step_in_cycle = epoch - int(
                        self.first_cycle_steps * (self.cycle_mult ** n - 1) / (self.cycle_mult - 1))
                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** (n)
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch

        self.max_lrs = [base_max_lr * (self.gamma ** self.cycle) for base_max_lr in self.base_max_lrs]
        self.last_epoch = math.floor(epoch)

        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]

# scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=10, last_epoch=200)


device = torch.device("cuda")
dtype = torch.float32

default_config = {
    "batch_size": 512,
    "lr_factor": 0.1,
    "man_width": 2,
}
wandb.init(project="man_width_vs_batchsize", config=default_config)

# sweep_config = {
#    "method": "bayes"
# }
# metric = {"name": "test_loss", "goal": "minimize"}
# parameter_dict = {
#    "batch_size": {"distribution": "q_log_uniform_values", "q": 8, "low": 256, "high": 1024},
#    "lr_factor": {"distribution": "uniform", "min": 5e-5, "max": 5e-3},
#    "pct_start": {"distribution": "uniform", "min": 0.05, "max": 0.5},
#    "grad_acc_steps": {"distribution": "q_log_uniform_values", "q": 8, "low": 1, "high": 64},

# }
wandbconfig = wandb.config

KL_WEIGHT = 0.9
DATASET_SIZE = 60000
BATCH_SIZE = wandbconfig.batch_size
STEPS_PER_EPOCH = DATASET_SIZE // BATCH_SIZE
EPOCHS = 5000 // STEPS_PER_EPOCH + 1  #math.ceil(STEPS / (DATASET_SIZE / BATCH_SIZE))
MOMENTUM = 0.0
WEIGHT_DECAY = 5e-4
LR = wandbconfig.lr_factor 
GRAD_ACC_STEPS = 1
W,H = 32,32
CUTSIZE = 8
CROPSIZE = 4
MAN_WIDTH = wandbconfig.man_width
IS_FULL_PRECISION = MAN_WIDTH == 23
# PCT_START = wandbconfig.pct_start
STEPS_PER_EPOCH = DATASET_SIZE / BATCH_SIZE
TOTAL_STEPS= STEPS_PER_EPOCH * EPOCHS


# class CNN(nn.Module):
#     def __init__(self):
#         super().__init__()

#         dims = [3,64,128,128,128,256,512,512,512]
#         seq = []
#         for i in range(len(dims)-1):
#             c_in,c_out = dims[i],dims[i+1]
#             seq.append( nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False) )
#             if c_out == c_in * 2:
#               seq.append( nn.MaxPool2d(2) )
#             seq.append( nn.BatchNorm2d(c_out) )
#             seq.append( nn.CELU(alpha=0.075) )
#         self.seq = nn.Sequential(*seq, nn.MaxPool2d(4), nn.Flatten(), nn.Linear(dims[-1], 10, bias=False))

#     def forward(self, x):
#         x = self.seq(x) / 8
#         return x

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.downsample = nn.AvgPool2d(2)
        self.flatten = nn.Flatten()
        self.fn1 = nn.Linear(196, 50)
        self.relu1 = nn.ReLU()
        self.fn2 = nn.Linear(50, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.downsample(x)
        x = self.flatten(x)
        x = self.fn1(x)
        x = self.relu1(x)
        x = self.fn2(x)
        x = self.softmax(x)
        return x

def loadMNIST(device):
  #   train = torchvision.datasets.CIFAR10(root="./cifar10", train=True, download=True)
  #   test  = torchvision.datasets.CIFAR10(root="./cifar10", train=False, download=True)
    train = torchvision.datasets.MNIST(root="./mnist", train=True, download=True)
    test  = torchvision.datasets.MNIST(root="./mnist", train=False, download=True)
    ret = [torch.tensor(i, device=device) for i in (train.data, train.targets, test.data, test.targets)]
    std, mean = torch.std_mean(ret[0].float(),dim=(0,1,2),unbiased=True,keepdim=True)
    for i in [0,2]: ret[i] = ((ret[i]-mean)/std).to(dtype)
    return ret

def getBatches(X,y, istrain):
  if istrain:
    perm = torch.randperm(len(X), device=device)
    X,y = X[perm],y[perm]

    # Crop = ([(y0,x0) for x0 in range(CROPSIZE+1) for y0 in range(CROPSIZE+1)], 
    #     lambda img, y0, x0 : nn.ReflectionPad2d(CROPSIZE)(img)[..., y0:y0+H, x0:x0+W])
    # FlipLR = ([(True,),(False,)], 
    #     lambda img, choice : torch.flip(img,[-1]) if choice else img)
    # def cutout(img,y0,x0):
    #   img[..., y0:y0+CUTSIZE, x0:x0+CUTSIZE] = 0
    #   return img
    # Cutout = ([(y0,x0) for x0 in range(W+1-CUTSIZE) for y0 in range(H+1-CUTSIZE)], cutout)

    # for options, transform in (Crop, FlipLR, Cutout):
    #   optioni = torch.randint(len(options),(len(X),), device=device)
    #   for i in range(len(options)):
    #     X[optioni==i] = transform(X[optioni==i], *options[i])

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
        if IS_FULL_PRECISION:
            return self.network(x)
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

    # def train_direct(self, data, target):
    #     self.master_weight.zero_grad()
    #     output = self.master_weight(data)
    #     loss = self.loss_fn(output, target)
    #     loss.backward()
    #     acc = (output.argmax(dim=1) == target).float().mean()
    #     self.optimizer.step()
    #     self.scheduler.step()
    #     return {"loss": loss.item(), "acc": acc.item()}

    def train_on_batch(self, data, target):
        self.master_weight.zero_grad()
        self.model_weight.zero_grad()
        output = self.model_weight(data)
        loss = self.loss_fn(output, target)
        # loss = loss * self.grad_scaling
        loss.backward()
        acc = (output.argmax(dim=1) == target).float().mean()
        self.model_grads_to_master_grads()
        # self.master_grad_apply(lambda x: x / self.grad_scaling)
        # grad_norm = nn.utils.clip_grad_norm_(self.master_weight.parameters(), self.grad_clip)
        self.optimizer.step()
        self.scheduler.step()
        self.master_params_to_model_params()
        return {"loss": loss.item(),  "acc": acc.item()}

    def train_with_KLD(self, data, target):
        self.master_weight.zero_grad()
        self.model_weight.zero_grad()
        ce = nn.CrossEntropyLoss(label_smoothing=0.2)
        kld = nn.KLDivLoss(reduction="batchmean")
        self.master_params_to_model_params()
        logits1 = self.model_weight(data)
        self.master_params_to_model_params()
        logits2 = self.model_weight(data)
        kl_weight = KL_WEIGHT
        ce_loss = (ce(logits1, target) + ce(logits2, target)) / 2
        kl_1 = kld(F.log_softmax(logits1, dim=-1), F.softmax(logits2, dim=-1)).sum(-1)
        kl_2 = kld(F.log_softmax(logits2, dim=-1), F.softmax(logits1, dim=-1)).sum(-1)
        kl_loss = kl_1 + kl_2 / 2
        loss = (1 - kl_weight) * ce_loss + kl_weight * kl_loss
        output = (logits1 + logits2) / 2
        acc = (output.argmax(dim=1) == target).float().mean()
        loss.backward()
        self.model_grads_to_master_grads()
        self.master_grad_apply(lambda x: x / self.grad_scaling)
        grad_norm = nn.utils.clip_grad_norm_(self.master_weight.parameters(), self.grad_clip)
        self.optimizer.step()
        self.scheduler.step()
        return {"loss": loss.item(), "grad_norm": grad_norm.item(), "acc": acc.item(),
                "ce_loss": ce_loss.item(), "kl_loss": kl_loss.item()}

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

        


X_train, y_train, X_test, y_test = loadMNIST(device)

# master_weight = CNN().to(dtype).to(device)
# master_weight = torchvision.models.resnet

class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = torchvision.models.resnet18(num_classes=10)
    def forward(self, x):
        return self.resnet(x) / 8

# master_weight = nn.Sequential(torchvision.models.resnet18(num_classes=10), nn.Softmax(dim=1))
# master_weight = ResNet()
master_weight = CNN()
master_weight = master_weight.to(dtype).to(device)
from torchsummary import summary

summary(master_weight, (1, 28, 28))
number = qtorch.FloatingPoint(8, MAN_WIDTH)
master_weight = replace_linear_with_quantized(master_weight, number, number, "stochastic")
model_weight = copy.deepcopy(master_weight)

opt = torch.optim.SGD(master_weight.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
# dummy scheduler
scheduler = torch.optim.lr_scheduler.ConstantLR(opt)
# scheduler = CosineAnnealingLinearWarmup(
#     optimizer = opt,
#     min_lrs = [1e-5],
#     first_cycle_steps = int(TOTAL_STEPS) ,
#     warmup_steps = int(TOTAL_STEPS * PCT_START),
#     gamma = 0.9
# )
# scheduler = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, epochs=EPOCHS, steps_per_epoch=len(X_train)//BATCH_SIZE, pct_start=PCT_START, anneal_strategy='linear',
                                                # div_factor=1e16, final_div_factor=0.07)
# scheduler = CosineLRScheduler(opt, t_initial=EPOCHS, lr_min=2e-8,
#                   cycle_mul=1.0, cycle_decay=1.0, cycle_limit=1,
#                   warmup_t=10, warmup_lr_init=1e-6, warmup_prefix=False, t_in_epochs=True,
#                   noise_range_t=None, noise_pct=0.67, noise_std=1.0,
#                   noise_seed=42, k_decay=1.0, initialize=True)

training_time = stepi = 0
result_log = {}
bar = tqdm.tqdm(range(EPOCHS * len(X_train) // BATCH_SIZE))
wrapper = MasterWeightOptimizerWrapper(master_weight, model_weight, opt, scheduler, weight_quant=qtorch.quant.quantizer(number), grad_acc_steps=GRAD_ACC_STEPS)

@torch.no_grad()
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
                                       
@torch.no_grad()
def weight_distance(initial_weight, final_weight):
    initial_weight.eval()
    final_weight.eval()
    distance = 0
    for initial, final in zip(initial_weight.parameters(), final_weight.parameters()):
        distance += torch.norm(initial - final) ** 2
    initial_weight.train()
    final_weight.train()
    return math.sqrt(distance.item())

@torch.no_grad()
def weight_cos_distance(initial_weight, final_weight):
    initial_weight.eval()
    final_weight.eval()
    dot_prod = 0
    for initial, final in zip(initial_weight.parameters(), final_weight.parameters()):
        dot_prod += torch.sum(initial.flatten() * final.flatten())
    initial_weight.train()
    final_weight.train()
    cos_distance = dot_prod / (weight_norm(initial_weight) * weight_norm(final_weight))
    return cos_distance.item()


@torch.no_grad()
def weight_norm(network):
    network.eval()
    norm = 0
    for param in network.parameters():
        norm += torch.norm(param) ** 2
    network.train()
    return math.sqrt(norm.item())
                                       
test_result_m = {}
initial_weight = copy.deepcopy(master_weight)
print(weight_norm(initial_weight))
for epoch in range(EPOCHS):
    start_time = time.perf_counter()
    train_losses, train_accs = [], []
    master_weight.train()
    model_weight.train()

    for X,y in getBatches(X_train,y_train, True):
        stepi += 1
        result_log.update(wrapper.train_on_batch(X, y))
        bar.update(1)
        bar.set_postfix(result_log)
        # result_log["weight_dist"] = weight_distance(initial_weight, master_weight)
        # result_log["weight_norm"] = weight_norm(master_weight)
        # result_log["weight_cos_dist"] = weight_cos_distance(initial_weight, master_weight)
        result_log["lr"] = opt.param_groups[0]["lr"]
        wandb.log(result_log | test_result_m)
        if stepi >= 5000:
            break

    wrapper.master_params_to_model_params(quantize=False)
    result_log.update(grad_on_dataset(model_weight, X_train, y_train))

    training_time += time.perf_counter()-start_time

    test_result_m = test(model_weight, getBatches(X_test, y_test, False))
    with open("results.txt", "a+") as f:
        f.write(f"{MAN_WIDTH}, {BATCH_SIZE}, {LR},  {test_result_m['test_loss']}, {test_result_m['test_acc']}, {grad_on_dataset(model_weight, X_train, y_train)['grad_norm_entire']}\n")

    print(f'epoch % 2d  test loss m %.3f   test acc m %.3f  training time %.2f'%(epoch+1, test_result_m["test_loss"], test_result_m["test_acc"], training_time))
  