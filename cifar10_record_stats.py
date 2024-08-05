from low_precision_utils import utils
from low_precision_utils import metrics

import load_data
import models

import torch
import qtorch

from argparse import ArgumentParser
EPOCHS = 7
BATCH_SIZE=512

model = models.CNN().to(torch.device("cuda"))
model = utils.QuantWrapper(model, utils.FP32_SCHEME)
data_loader = load_data.loadCIFAR10

X_train, y_train, X_test, y_test = data_loader(torch.device("cuda"))
opt = torch.optim.SGD(model.parameters(), lr=0.6, momentum=0.9, nesterov=True)
scheduler = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=0.6, total_steps=EPOCHS*len(X_train)//BATCH_SIZE, pct_start=0.2)
result_log = {}
test_result = {}



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

ntest_schemes = {
    "f0b0n": build_test_scheme(0, 0, "nearest", False),
    "f1b1n": build_test_scheme(1, 1, "nearest", False),
    "f2b2n": build_test_scheme(2, 2, "nearest", False),
}

# we know that if we have unbiased estimator of gradient
# we have
# 1. we converge to a noise ball around local mimimum for non-convex problem with fixed learning rate
#       the size of the noise ball is determined by the variance of the gradient, which futher determined by the batch size
#       and the precision of computation 
# 2. we converge to the global minimum for convex problem with fixed learning rate
# 3. we converge to a local minimum for non-convex problem with decaying learning rate (havn't showed)
# 4. we showed that if we have unbiased estimator of gradient, we can make a trade-off between the bias and the variance

# question 1:
# in mixed precision training, do we have 0 mean for a practical machine learning problem?
# question 2:
# how much bias in the gradient can we tolerate?
# question 3:
# what is the quant scheme that has the 
# question 4:
# given a computation budge, how to find the best quant scheme for each matrix?


# how to make a quant scheme that has less bias, but variance can be reduced as increasing the batchsize

stest_schemes = {
    # "stem3leaf0": utils.QuantScheme.build(
    #     fnumber=3, bnumber=3, wnumber=3,
    #     bwnumber=3, bfnumber=0, bnumber2=0,
    # ),
    # "f0b0s": build_test_scheme(0, 0, "stochastic", False),
    # "f1b1s": build_test_scheme(1, 1, "stochastic", False),
    # "f2b2s": build_test_scheme(2, 2, "stochastic", False),
    # "f3b3s": build_test_scheme(3, 3, "stochastic", False),
    # "f7b7s": build_test_scheme(7, 7, "stochastic", False),
    # "b20other23": utils.QuantScheme.build(
    #     fnumber=23, bnumber=23, wnumber=23,
    #     bwnumber=23, bfnumber=23, bnumber2=0,
    # ),
    # "bf0other23": utils.QuantScheme.build(
    #     fnumber=23, bnumber=23, wnumber=23,
    #     bwnumber=23, bfnumber=0, bnumber2=23,
    # ),
    # "bw0other23": utils.QuantScheme.build(
    #     fnumber=23, bnumber=23, wnumber=23,
    #     bwnumber=0, bfnumber=23, bnumber2=23,
    # ),
    # "w0other23": utils.QuantScheme.build(
    #     fnumber=23, bnumber=23, wnumber=0,
    #     bwnumber=23, bfnumber=23, bnumber2=23,
    # ),
    # "b0other23": utils.QuantScheme.build(
    #     fnumber=23, bnumber=0, wnumber=23,
    #     bwnumber=23, bfnumber=23, bnumber2=23,
    # ),
    # "f0other23": utils.QuantScheme.build(
    #     fnumber=0, bnumber=23, wnumber=23,
    #     bwnumber=23, bfnumber=23, bnumber2=23,
    # ),
    # "stem23leaf0": utils.QuantScheme.build(
    #     fnumber=23, bnumber=23, wnumber=23,
    #     bwnumber=23, bfnumber=0, bnumber2=0,
    # ),
    # "stem23leaf3": utils.QuantScheme.build(
    #     fnumber=23, bnumber=23, wnumber=23,
    #     bwnumber=23, bfnumber=3, bnumber2=3,
    # ),
    # "stem3leaf0": utils.QuantScheme.build(
    #     fnumber=3, bnumber=3, wnumber=3,
    #     bwnumber=3, bfnumber=0, bnumber2=0,
    # ),
    # "stem3leaf23": utils.QuantScheme.build(
    #     fnumber=3, bnumber=3, wnumber=3,
    #     bwnumber=3, bfnumber=23, bnumber2=23,
    # ),
    # "stem3leaf3": utils.QuantScheme.build(
    #     fnumber=3, bnumber=3, wnumber=3,
    #     bwnumber=3, bfnumber=3, bnumber2=3,
    # ),
    # "f0b0s": build_test_scheme(0, 0, "nearest", True),
    # "f1b1s": build_test_scheme(1, 1, "nearest", True),
    # "f2b2s": build_test_scheme(2, 2, "nearest", True),
    # "f3b2s": build_test_scheme(3, 2, "nearest", True),
    # "f7b7s": build_test_scheme(7, 7, "nearest", True),
}

logger = metrics.Logger()
batch_sizes = [512, 8192]

def describe_scheme(bs, scheme: utils.QuantScheme):
    return f"bs{bs}_f{scheme.fnumber.man}_b{scheme.bnumber.man}_r{scheme.fround_mode}"

def test():
    result = {}
    model.apply_quant_scheme(utils.FP32_SCHEME)
    with torch.no_grad():
        loss_acc = model.module.loss_acc(X_test, y_test)
        total_loss = loss_acc["loss"].item()
        acc = loss_acc["acc"] 
        result.update({"test_loss": total_loss, "test_acc": acc})
    # del loss_acc, total_loss, acc
    for bs in batch_sizes:
        print(bs)
        for X, y in load_data.getBatches(X_train, y_train, bs): break
        model.apply_quant_scheme(utils.FP32_SCHEME)
        grad_norm = metrics.grad_on_dataset(model.module, X, y)
        result.update({f"bs{bs}_{k}": v for k, v in grad_norm.items()})
        model.zero_grad()
        for k,v in ntest_schemes.items():
            dot, e_error_norm = metrics.grad_error_metrics_deterministic(model, v, X, y)
            result[f"bs{bs}_{k}_dot"] = dot
            result[f"bs{bs}_{k}_e_error_norm"] = e_error_norm
            result[f"bs{bs}_{k}_bias_norm"] = e_error_norm
        for k,v in stest_schemes.items():
            dot, e_error_norm, bias_norm = metrics.grad_error_metrics(model, v, X, y)
            result[f"bs{bs}_{k}_dot"] = dot
            result[f"bs{bs}_{k}_e_error_norm"] = e_error_norm
            result[f"bs{bs}_{k}_bias_norm"] = bias_norm
    return result

stepi = 0
for epoch in range(EPOCHS):
    augX, augY = load_data.augment_data(X_train, y_train)
    model.apply_quant_scheme(utils.FP32_SCHEME)
    for X,y in load_data.getBatches(augX,augY,BATCH_SIZE):
        model.train()
        stepi += 1
        model.zero_grad()
        loss, acc = model.module.loss_acc(X, y).values()
        loss.backward()
        opt.step()
        scheduler.step()
    test_result = test()
    logger.log(test_result)
    print(loss, acc)

logger.to_csv("out/result_tables/cifar10_record_stats.csv")
