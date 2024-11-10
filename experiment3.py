import torch.nn.grad
import tqdm
import torch
from torch import nn
from argparse import ArgumentParser
from low_precision_utils import quant
from low_precision_utils import metrics
from ml_utils import log_util
from utils import load_data

parser = ArgumentParser()

parser.add_argument("--model", type=str, default="mnist_mlp")
parser.add_argument("--dataset", type=str, default="mnist")
parser.add_argument("--lr", type=float, default=0.003)
parser.add_argument("--weight_decay", type=float, default=0.0)
parser.add_argument("--momentum", type=float, default=0.0)
parser.add_argument("--steps", type=int, default=20000)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--std_weight2", type=float, default=0.0)
parser.add_argument("--std_z2", type=float, default=0.0)
parser.add_argument("--std_dz2", type=float, default=0.0)
parser.add_argument("--std_dz1", type=float, default=0.0)
parser.add_argument("--std_a1", type=float, default=0.0)
parser.add_argument("--std_a0", type=float, default=0.0)
log_util.add_log_args(parser, defualt_experiment_name="mnist3")
args = parser.parse_args()

logger = log_util.Logger.from_args(args)

dtype = torch.float32
device = torch.device("cuda")
data_loader = load_data.get_data_loader(args.dataset)
X_train, y_train, X_test, y_test = data_loader(device)


def softmax_crossentropy_backward(logits, labels):
    batch_size = logits.size(0)
    probs = torch.softmax(logits, dim=-1)
    labels_one_hot = torch.zeros_like(logits)
    labels_one_hot[torch.arange(batch_size), labels] = 1.0
    grad = (probs - labels_one_hot)
    return grad

def softmax_crossentropy(logits, labels):
    batch_size = logits.size(0)
    probs = torch.softmax(logits, dim=-1)
    labels_one_hot = torch.zeros_like(logits)
    labels_one_hot[torch.arange(batch_size), labels] = 1.0
    loss = -torch.sum(labels_one_hot * torch.log(probs + 1e-8)) 
    return loss

def compute_acc(logits, labels):
    preds = torch.argmax(logits, dim=-1)
    acc = torch.sum(preds == labels).item() / len(labels)
    return acc

class Model():
    def __init__(self):
        super().__init__()
        self.weight1 = torch.rand(784, 100, dtype=dtype, device=device) * (2 / (100 + 784))
        self.weight2 = torch.rand(100, 10, dtype=dtype, device=device) * (2 / (100 + 10))
        self.bias1 = torch.zeros(100, dtype=dtype, device=device)
        self.bias2 = torch.zeros(10, dtype=dtype, device=device)
        self.z1 = None
        self.a1 = None
        self.z2 = None

    def forward(self, x):
        self.a0 = x
        z1 = x @ self.weight1 + self.bias1
        self.z1 = z1
        a1 = torch.relu(z1)
        self.a1 = a1
        z2 = a1 @ self.weight2 + self.bias2
        self.z2 = z2
        return z2

    def noisy_back(self, y, std_weight2=0.0, std_z2=0.0, std_dz2=0.0, 
                   std_dz1=0.0, std_a1=0.0, std_a0=0.0):
        dz2 = softmax_crossentropy_backward(self.z2, y)
        eps_z2 = torch.randn_like(self.z2) * std_z2
        eps_weight2 = torch.randn_like(self.weight2) * std_weight2
        da1 = (dz2 + eps_z2) @ (self.weight2 + eps_weight2).t()
        dz1 = da1 * (self.z1 > 0).to(torch.float32)

        self.dz2 = dz2
        self.dz1 = dz1
        self.da1 = da1

        eps_a1 = torch.randn_like(self.a1) * std_a1
        eps_dz2 = torch.randn_like(dz2) * std_dz2
        dw2 = (self.a1 + eps_a1).t() @ (dz2 + eps_dz2)
        db2 = dz2.sum(dim=0)

        eps_a0 = torch.randn_like(self.a0) * std_a0
        eps_dz1 = torch.randn_like(dz1) * std_dz1
        dw1 = (self.a0 + eps_a0).t() @ (dz1 + eps_dz1)
        db1 = dz1.sum(dim=0)

        self.dw1 = dw1 / args.batch_size
        self.db1 = db1 / args.batch_size
        self.dw2 = dw2 / args.batch_size
        self.db2 = db2 / args.batch_size

    def weight_update(self):
        lr = args.lr
        self.weight1 -= lr * self.dw1 
        self.bias1 -= lr * self.db1 
        self.weight2 -= lr * self.dw2 
        self.bias2 -= lr * self.db2 


model = Model()
print(model)

@torch.no_grad()
def test(network: nn.Module):
    network.forward(X_test)
    loss = softmax_crossentropy(network.z2, y_test)
    accuracy = compute_acc(network.z2, y_test)
    return {"test_acc": accuracy, "test_loss": loss}

bar = tqdm.tqdm(range(args.steps))
stepi = 0
grad_norm = 0

while True:
    with torch.no_grad():
        for X,y in load_data.getBatches(X_train,y_train,args.batch_size):
            stepi += 1
            logits = model.forward(X)
            loss = softmax_crossentropy(logits, y)
            acc = compute_acc(logits, y)
            logger.log({
                "loss": loss.item() / args.batch_size,
                "acc": acc,
            })
            model.noisy_back(y, args.std_weight2, args.std_z2, args.std_dz2, args.std_dz1, args.std_a1, args.std_a0)
            model.weight_update()
            bar.update(1)
            bar.set_postfix(loss=loss.item() / args.batch_size, acc=acc, grad_norm=grad_norm)
            if stepi % (args.wandb_interval) == 0 or stepi >= args.steps:
                test_result = test(model)
                model.forward(X_train[:2048])
                model.noisy_back(y_train[:2048])
                grad_norm = torch.sqrt((model.dw2 ** 2).sum()).item()
                test_result["grad_norm"] = grad_norm
                logger.log_same_iter(test_result)
            if stepi >= args.steps:
                exit()
