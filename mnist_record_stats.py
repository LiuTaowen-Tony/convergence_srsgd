from low_precision_utils import utils
from low_precision_utils import metrics

import load_data
import models

import wandb
import tqdm
import torch
import qtorch

from argparse import ArgumentParser

model = models.MNIST_CNN().to(torch.device("cuda"))

model = utils.QuantWrapper(model, utils.FP32_SCHEME)
data_loader = load_data.get_data_loader("mnist")
X_train, y_train, X_test, y_test = data_loader(torch.device("cuda"))
opt = torch.optim.SGD(model.parameters(), lr=0.03, momentum=0.0)
scheduler = torch.optim.lr_scheduler.ConstantLR(opt)
result_log = {}
test_result = {}

wandb.init(project="metrics", config={})

stepi = 0

def build_test_scheme(fman, bman, rounding, same):
    fnumber = qtorch.FloatingPoint(8, fman)
    bnumber = qtorch.FloatingPoint(8, bman)
    return utils.QuantScheme(
        fnumber=fnumber,
        bnumber=bnumber,
        wnumber=fnumber,
        fround_mode=rounding,
        bround_mode=rounding,
        wround_mode=rounding,
        same_input=same,
        same_weight=same,
        bfnumber=bnumber,
        bfround_mode=rounding,
        bwnumber=bnumber,
        bwround_mode=rounding,
    )

ntest_schemes = [
    build_test_scheme(0, 0, "nearest", True),
    build_test_scheme(1, 1, "nearest", True),
    build_test_scheme(2, 2, "nearest", True),
    build_test_scheme(3, 2, "nearest", True),
    build_test_scheme(7, 7, "nearest", True),

]

stest_schemes = [
    build_test_scheme(0, 0, "stochastic", False),
    build_test_scheme(1, 1, "stochastic", False),
    build_test_scheme(2, 2, "stochastic", False),
    build_test_scheme(3, 2, "stochastic", False),
    build_test_scheme(7, 7, "stochastic", False),
]

logger = metrics.Logger()
batch_sizes = [4, 32, 64, 128, 256, 512, 2048]

def describe_scheme(bs, scheme: utils.QuantScheme):
    return f"bs{bs}_f{scheme.fnumber.man}_b{scheme.bnumber.man}_r{scheme.fround_mode}_{scheme.same_input}"

def test():
    result = {}
    model.apply_quant_scheme(utils.FP32_SCHEME)
    loss_acc = model.module.loss_acc(X_test, y_test)
    total_loss = loss_acc["loss"].item()
    acc = loss_acc["acc"] 
    result.update({"test_loss": total_loss, "test_acc": acc})
    for bs in batch_sizes:
        for X, y in load_data.getBatches(X_train, y_train, bs): break
        model.apply_quant_scheme(utils.FP32_SCHEME)
        # loss = model.module.loss_acc(X, y)["loss"]
        # eigenvalue = metrics.power_iteration_find_hessian_eigen(loss, model.parameters())
        # result[f"bs{bs}_abs_hessian"] = abs(eigenvalue)
        # result[f"bs{bs}_hessian"] = eigenvalue
        grad_norm = metrics.grad_on_dataset(model.module, X, y)
        result.update({f"bs{bs}_{k}": v for k, v in grad_norm.items()})
        model.zero_grad()
        for scheme in ntest_schemes:
            bias = metrics.grad_bias_deterministic(model, scheme, X, y)
            result[f"{describe_scheme(bs, scheme)}_bias"] = bias
        for scheme in stest_schemes:
            var, bias = metrics.grad_bias_std_norm(model, scheme, X, y)
            result[f"{describe_scheme(bs, scheme)}_bias"] = bias
            result[f"{describe_scheme(bs, scheme)}_var"] = var
    return result


try:
    while True:
        for X,y in load_data.getBatches(X_train,y_train,128):
            model.train()
            stepi += 1
            model.zero_grad()
            loss, acc = model.module.loss_acc(X, y).values()
            loss.backward()
            opt.step()
            scheduler.step()
            result_log.update({
                "loss": loss.item(),
                "acc": acc,
                "lr": opt.param_groups[0]["lr"]
            })
            if stepi % (10) == 0:
                test_result = test()
                wandb.log(result_log | test_result)
                logger.log(result_log | test_result)
            if stepi >= 10000:
                exit()
finally:
    logger.to_csv("mnist_record_stats.csv")
    wandb.finish()