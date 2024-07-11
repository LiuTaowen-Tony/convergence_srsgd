import torch.utils
import torch
from torch import nn
import math

BATCH_SIZE = 64
LR = 0.005
DEVICE="cuda"



class PLinear():
    def __init__(self, in_features, out_features):
        self.weight = torch.randn((in_features, out_features), device=DEVICE) * math.sqrt(1 / (in_features + out_features))
        self.bias = torch.zeros((out_features), device=DEVICE)

        self.x_p = torch.ones((BATCH_SIZE, in_features), requires_grad=True, device=DEVICE)
        self.w_p = torch.ones((in_features, out_features), requires_grad=True, device=DEVICE)
        self.go1_p = torch.ones((BATCH_SIZE,out_features), requires_grad=True, device=DEVICE)
        self.go2_p = torch.ones((BATCH_SIZE,out_features), requires_grad=True, device=DEVICE)
        self.wb_p = torch.ones((out_features, in_features), requires_grad=True, device=DEVICE)
        self.xb_p = torch.ones((BATCH_SIZE, in_features), requires_grad=True, device=DEVICE)


    def my_forward(self, x):
        self.save_x = x
        print("x", x.var())
        x = x * self.x_p
        w = self.weight * self.w_p
        return torch.matmul(x, w) + self.bias

    def my_backward(self, grad_out):
        # self.gox_p = torch.matmul(self.save_x.T, grad_out)
        # grad_out : bs x out
        print("g", grad_out.var())
        x_grad = torch.matmul(grad_out * self.go1_p, self.weight.T * self.wb_p)
        self.x_grad = x_grad
        self.grad_out = grad_out

        self.w_grad = torch.matmul(self.save_x.T * self.xb_p.T, grad_out * self.go2_p)
        self.b_grad = torch.sum(grad_out, dim=0)

        return x_grad

    def my_update(self, lr):
        self.weight = torch.tensor(self.weight - lr * self.w_grad)
        self.bias = torch.tensor(self.bias - lr * self.b_grad)


class PReLU():
    def my_forward(self, x):
        self.save_x = x
        return torch.where(x > 0, x, 0.0)

    def my_backward(self, grad_out):
        return torch.where(self.save_x > 0, grad_out, 0.0)

    def my_update(self, lr):
        pass

class PMSE():
    def my_forward(self, x, y):
        self.save_x = x
        # convert to one hot
        acc = (torch.argmax(x, dim=1) == y).sum().item() / BATCH_SIZE
        y = torch.zeros((BATCH_SIZE, 10), device=DEVICE).scatter(1, y.view(-1, 1), 1)
        self.save_y = y
        return torch.mean((x - y) ** 2), acc

    def my_backward(self):
        return 2 * (self.save_x - self.save_y) / BATCH_SIZE

    def my_update(self, lr):
        pass


my_nn = {
    "l1" : PLinear(784, 100),
    "relu1" : PReLU(),
    "l2" : PLinear(100, 100),
    "relu2" : PReLU(),
    "l3" : PLinear(100, 100),
    "relu3" : PReLU(),
    "l4" : PLinear(100, 100),
    "relu4" : PReLU(),
    "l5" : PLinear(100, 10),
}

mse = PMSE()
def train_batch(x, y):
    with torch.no_grad():
        x = x.view(BATCH_SIZE, -1)
        for name, layer in my_nn.items():
            x = layer.my_forward(x)
        loss, acc = mse.my_forward(x, y)
        grad = mse.my_backward()
        for name, layer in list(my_nn.items())[::-1]:
            grad = layer.my_backward(grad)
            layer.my_update(LR)
        return loss, acc

def grads(x, y):
    x = x.view(BATCH_SIZE, -1)
    for name, layer in my_nn.items():
        x = layer.my_forward(x)
    loss = mse.my_forward(x, y)
    grad = mse.my_backward()
    for name, layer in list(my_nn.items())[::-1]:
        grad = layer.my_backward(grad)
    

def get_grad_else_0(x, y):
    try:
        x: torch.Tensor
        x = x.view(-1)
        xx = torch.tensor(x)
        # xx = x
        cos = torch.cosine_similarity(x, xx, 0)
        # cos = x.norm()
        a = torch.autograd.grad(1e7 * cos, y, retain_graph=True)
        return a[0].abs().sum().item()
    except Exception as e:
        # print(e)
        return 0.0

def get_stats(logger):
    stats_inputs = {}
    for name1, i in my_nn.items():
        if "relu" not in name1:
            weight_grad = i.w_grad
            logger.log({f"{name1}_grad_norm": weight_grad.norm().item()})
            logger.log({f"{name1}_weight_spectral_norm": torch.linalg.norm(i.weight, ord=2).item()})
            logger.log({f"{name1}_x_spectral_norm": torch.linalg.norm(i.save_x, ord=2).item()})
            logger.log({f"{name1}_x_grad_spectral_norm": torch.linalg.norm(i.x_grad, ord=2).item()})
            logger.log({f"{name1}_grad_out_spectral_norm": torch.linalg.norm(i.grad_out, ord=2).item()})
            logger.log({f"{name1}_x_var": i.save_x.var().item()})
            logger.log({f"{name1}_x_grad_var": i.x_grad.var().item()})
            logger.log({f"{name1}_grad_out_var": i.grad_out.var().item()})
            for name2, j in my_nn.items():
                if "relu" not in name2:
                    stats = {
                        f"{name1}_{name2}_xp": get_grad_else_0(weight_grad, j.x_p, ),
                        f"{name1}_{name2}_wp": get_grad_else_0(weight_grad, j.w_p, ),
                        f"{name1}_{name2}_go1p": get_grad_else_0(weight_grad, j.go1_p, ),
                        f"{name1}_{name2}_go2p": get_grad_else_0(weight_grad, j.go2_p, ),
                        f"{name1}_{name2}_wbp": get_grad_else_0(weight_grad, j.wb_p, ),
                        f"{name1}_{name2}_xbp": get_grad_else_0(weight_grad, j.xb_p, ),
                    }
                    logger.log(stats)


import load_data 

X_train, y_train, X_test, y_test = load_data.loadMNISTData("cuda")
X_train = (X_train - X_train.mean())/ X_train.std()
from low_precision_utils import metrics
logger = metrics.Logger()
i = 0
torch.manual_seed(0)

while True:
    for X, y in load_data.getBatches(X_train, y_train, BATCH_SIZE):
        if i % 10 == 0:
            grads(X, y)
            get_stats(logger)
            loss, acc = train_batch(X, y)
            logger.log({"loss": loss.item()})
            logger.log({"acc": acc})
            logger.to_csv("mnist_record_stats_new.csv")
        else:
            train_batch(X, y)
        i += 1
    
