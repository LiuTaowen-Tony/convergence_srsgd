from low_precision_utils import quant
from low_precision_utils import metrics

import ml_utils.log_util
from utils import load_data
from utils import models

import torch
import ml_utils

EPOCHS = 1
BATCH_SIZE=512

model = models.CNN().to(torch.device("cuda"))
model = quant.QuantWrapper(model, quant.FP32_SCHEME)
data_loader = load_data.loadCIFAR10

X_train, y_train, X_test, y_test = data_loader(torch.device("cuda"))
opt = torch.optim.SGD(model.parameters(), lr=0.6, momentum=0.9, nesterov=True)
scheduler = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=0.6, total_steps=EPOCHS*len(X_train)//BATCH_SIZE, pct_start=0.2)
result_log = {}
test_result = {}




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

fx8 = quant.ScaledIntQuant(fl=8)
fx6 = quant.ScaledIntQuant(fl=6)
fx4 = quant.ScaledIntQuant(fl=4)
fx8d = quant.ScaledIntQuant(fl=8, round_mode="nearest")
fx4d = quant.ScaledIntQuant(fl=4, round_mode="nearest")
fp0 = quant.FPQuant(man=0)
fp0d = quant.FPQuant(man=0, round_mode="nearest")


stest_schemes = {
    # "leaf_only16" : quant.QuantScheme(goweight=fx16, bact=fx16),
    # "leaf_only22" : quant.QuantScheme(bact=fx4),
    "forward4" : quant.QuantScheme(act=fx4, weight=fx4),
    "forward8" : quant.QuantScheme(act=fx8, weight=fx8),
    "allfp0" : quant.QuantScheme(act=fp0, weight=fp0, goweight=fp0, bact=fp0, goact=fp0, bweight=fp0),
    "all4" : quant.QuantScheme(act=fx4, weight=fx4, goweight=fx4, bact=fx4, goact=fx4, bweight=fx4),
    "all8" : quant.QuantScheme(act=fx8, weight=fx8, goweight=fx8, bact=fx8, goact=fx8, bweight=fx8),
    "back8" : quant.QuantScheme(goweight=fx8, bact=fx8, goact=fx8, bweight=fx8),
    "back4" : quant.QuantScheme(goweight=fx4, bact=fx4, goact=fx4, bweight=fx4),
    "leaf8" : quant.QuantScheme(goweight=fx8, bact=fx8),
    "leaf4" : quant.QuantScheme(goweight=fx4, bact=fx4),
}

ntest_schemes = {
    "forward4d" : quant.QuantScheme(act=fx4d, weight=fx4d),
    "forward8d" : quant.QuantScheme(act=fx8d, weight=fx8d),
    "allfp0d" : quant.QuantScheme(act=fp0d, weight=fp0d, goweight=fp0d, bact=fp0d, goact=fp0d, bweight=fp0d),
    "all4d" : quant.QuantScheme(act=fx4d, weight=fx4d, goweight=fx4d, bact=fx4d, goact=fx4d, bweight=fx4d),
    "all8d" : quant.QuantScheme(act=fx8d, weight=fx8d, goweight=fx8d, bact=fx8d, goact=fx8d, bweight=fx8d),
    "back8d" : quant.QuantScheme(goweight=fx8d, bact=fx8d, goact=fx8d, bweight=fx8d),
    "back4d" : quant.QuantScheme(goweight=fx4d, bact=fx4d, goact=fx4d, bweight=fx4d),
    "leaf8d" : quant.QuantScheme(goweight=fx8d, bact=fx8d),
    "leaf4d" : quant.QuantScheme(goweight=fx4d, bact=fx4d),
}

logger = ml_utils.log_util.Logger(1)
batch_sizes = [512]

def test():
    result = {}
    model.apply_quant_scheme(quant.FP32_SCHEME)
    with torch.no_grad():
        loss_acc = model.module.loss_acc(X_test, y_test)
        total_loss = loss_acc["loss"].item()
        acc = loss_acc["acc"] 
        result.update({"test_loss": total_loss, "test_acc": acc})
    # del loss_acc, total_loss, acc
    for bs in batch_sizes:
        print(bs)
        for X, y in load_data.getBatches(X_train, y_train, bs): break
        model.apply_quant_scheme(quant.FP32_SCHEME)
        # grad_norm = metrics.grad_on_dataset(model.module, X, y)
        # result.update({f"&bs={bs}&metrics={k}": v for k, v in grad_norm.items()})
        model.zero_grad()
        for i,v in stest_schemes.items():
            for rounds in [25, 100, 400]:
                # r = metrics.per_layer_grad_error_metrics(model, v, X, y, rounds)
                # result.update({f"&bs={bs}&scheme={i}&rounds={rounds}{k}" : v for k, v in r.items()})
                cos, var, bias = metrics.grad_error_metrics(model, v, X, y, rounds)
                result.update({
                    f"&bs={bs}&scheme={i}&rounds={rounds}&metric=cos" : cos,
                    f"&bs={bs}&scheme={i}&rounds={rounds}&metric=var" : var,
                    f"&bs={bs}&scheme={i}&rounds={rounds}&metric=bias" : bias
                })
        for i,v in ntest_schemes.items():
            # r = metrics.per_layer_grad_error_metrics_deterministic(model, v, X, y)
            # result.update({f"&bs={bs}&scheme={i}&rounds={rounds}{k}" : v for k, v in r.items()})
            cos, var, bias = metrics.grad_error_metrics_deterministic(model, v, X, y)
            result.update({
                f"&bs={bs}&scheme={i}&rounds={rounds}&metric=cos" : cos,
                f"&bs={bs}&scheme={i}&rounds={rounds}&metric=var" : var,
                f"&bs={bs}&scheme={i}&rounds={rounds}&metric=bias" : bias
            })
    return result

stepi = 0
for epoch in range(EPOCHS):
    augX, augY = load_data.augment_data(X_train, y_train)
    model.apply_quant_scheme(quant.FP32_SCHEME)
    for X,y in load_data.getBatches(augX,augY,BATCH_SIZE):
        model.train()
        stepi += 1
        model.zero_grad()
        loss, acc = model.module.loss_acc(X, y).values()
        loss.backward()
        opt.step()
        scheduler.step()
    test_result = test()
    print(test_result)
    logger.log(test_result)
    print(loss, acc)

logger.to_csv("out/result_tables/cifar10_record_stats.csv")
