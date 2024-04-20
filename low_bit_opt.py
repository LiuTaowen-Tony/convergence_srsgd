from copy import deepcopy
from torch.optim import Optimizer, SGD, Adam
from torch import nn


class MasterWeightOptimizerWrapper():
    def __init__(
            self,
            master_weight,
            model_weight,
            optimizer,
            weight_quant=None,
            grad_scaling=1.0,
    ):
        self.master_weight = master_weight
        self.model_weight = model_weight
        self.optimizer = optimizer
        self.weight_quant = weight_quant
        self.grad_scaling = grad_scaling

    # --- for mix precision training ---
    def model_grads_to_master_grads(self):
        for model, master in zip(self.model_weight.parameters(), self.master_weight.parameters()):
            if master.grad is None:
                master.grad = master.data.new(*master.data.size())
            master.grad.data.copy_(self.grad_quant(model.grad.data))

    def master_grad_apply(self, fn):
        for master in (self.master_weight.parameters()):
            if master.grad is None:
                master.grad = master.data.new(*master.data.size())
            master.grad.data = fn(master.grad.data)

    def master_params_to_model_params(self):
        for model, master in zip(self.model_weight.parameters(), self.master_weight.parameters()):
            model.data.copy_(self.weight_quant(master.data))

    def _apply_model_weights(self, model, quant_func):
        for m in model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                m.weight.data = quant_func(m.weight.data)
                if m.bias is not None:
                    m.bias.data = quant_func(m.bias.data)

    def train_on_loss(self, loss):
        loss = loss * self.loss_scale
        opt = self.optimizer
        self.model_weight.zero_grad()
        self.model_weight.backward(loss)
        self.model_grads_to_master_grads()
        self.master_grad_apply(lambda x: x / self.loss_scale)
        nn.utils.clip_grad_norm_(self.master_weight.parameters(), self.grad_clip)
        opt.step()
        self.master_params_to_model_params()
        self._apply_model_weights(self.model_weight, self.weight_quant)
        return loss.item()
