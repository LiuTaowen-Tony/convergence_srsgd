# %%
from copy import deepcopy
import torch
from torch import nn
import qtorch
from qtorch.quant import Quantizer
import tqdm

from torchvision import datasets, transforms
import torchvision
import argparse

parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--man_width', type=int, default=5)
parser.add_argument('--group', type=str)
parser.add_argument('--id', type=str)
parser.add_argument("--round_mode", type=str, default="stochastic")
parser.add_argument("--steps", type=int, default=10000)
parser.add_argument("--scheduler", type=str, default="fix")

# import namespace
from argparse import Namespace

args = parser.parse_args()
args = Namespace()
args.batch_size = 256
args.grad_acc = 1
args.lr = 0.1 / 1024
args.man_width = 20
args.group = "play_ground"
args.id = "full_precision"
args.steps = 10000
args.scheduler = "warmuplinear"
args.round_mode = "stochastic"
device = "cuda"

# %%

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Adjust size to match model's expected input
    transforms.ToTensor(),          # Convert to tensor (and scales to [0, 1])
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalization
])
TrainData = datasets.CIFAR10("./imagenet", transform=transform)
data = torch.tensor(TrainData.data).to(device).float()
data = data.permute(0, 3, 1, 2)

class MyDataLoader():
    def __init__(self, data, targets, batch_size):
        self.data = data
        self.targets = targets
        self.batch_size = batch_size
        self.idx = list(range(len(data)))
        random.shuffle(self.idx)


    def __iter__(self):
        random.shuffle(self.idx)
        for i in range(0, len(data), self.batch_size):
            idx = self.idx[i:i + self.batch_size]
            if len(idx) == 0: continue
            yield self.data[idx], self.targets[idx]

    def __len__(self):
        return len(self.data) // self.batch_size

fnumber = qtorch.FloatingPoint(8, args.man_width)
bnumber = qtorch.FloatingPoint(8, args.man_width)


# can hack any changes to each convolution group that you want directly in here
class ConvGroup(nn.Module):
    def __init__(self, channels_in, channels_out):
        super().__init__()
        self.channels_in  = channels_in
        self.channels_out = channels_out

        self.pool1 = nn.MaxPool2d(2)
        self.conv1 = nn.Conv2d(channels_in,  channels_out, kernel_size=3, padding="same", bias=False)
        self.conv2 = nn.Conv2d(channels_out, channels_out, kernel_size=3, padding="same", bias=False)

        self.norm1 = nn.BatchNorm2d(channels_out)
        self.norm2 = nn.BatchNorm2d(channels_out)

        self.activ = nn.GELU()


    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.norm1(x)
        x = self.activ(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.activ(x)

        return x

class FastGlobalMaxPooling(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        # Previously was chained torch.max calls.
        # requires less time than AdaptiveMax2dPooling -- about ~.3s for the entire run, in fact (which is pretty significant! :O :D :O :O <3 <3 <3 <3)
        return torch.amax(x, dim=(2,3)) # Global maximum pooling
    


def get_patches(x, patch_shape=(3, 3), dtype=torch.float32):
    # This uses the unfold operation (https://pytorch.org/docs/stable/generated/torch.nn.functional.unfold.html?highlight=unfold#torch.nn.functional.unfold)
    # to extract a _view_ (i.e., there's no data copied here) of blocks in the input tensor. We have to do it twice -- once horizontally, once vertically. Then
    # from that, we get our kernel_size*kernel_size patches to later calculate the statistics for the whitening tensor on :D
    c, (h, w) = x.shape[1], patch_shape
    return x.unfold(2,h,1).unfold(3,w,1).transpose(1,3).reshape(-1,c,h,w).to(dtype) # TODO: Annotate?

def get_whitening_parameters(patches):
    # As a high-level summary, we're basically finding the high-dimensional oval that best fits the data here.
    # We can then later use this information to map the input information to a nicely distributed sphere, where also
    # the most significant features of the inputs each have their own axis. This significantly cleans things up for the
    # rest of the neural network and speeds up training.
    n,c,h,w = patches.shape
    est_covariance = torch.cov(patches.view(n, c*h*w).t())
    eigenvalues, eigenvectors = torch.linalg.eigh(est_covariance, UPLO='U') # this is the same as saying we want our eigenvectors, with the specification that the matrix be an upper triangular matrix (instead of a lower-triangular matrix)
    return eigenvalues.flip(0).view(-1, 1, 1, 1), eigenvectors.t().reshape(c*h*w,c,h,w).flip(0)

# Run this over the training set to calculate the patch statistics, then set the initial convolution as a non-learnable 'whitening' layer
def init_whitening_conv(layer, train_set=None, num_examples=None, previous_block_data=None, pad_amount=None, freeze=True, whiten_splits=None):
    if train_set is not None and previous_block_data is None:
        if pad_amount > 0:
            previous_block_data = train_set[:num_examples,:,pad_amount:-pad_amount,pad_amount:-pad_amount] # if it's none, we're at the beginning of our network.
        else:
            previous_block_data = train_set[:num_examples,:,:,:]

    # chunking code to save memory for smaller-memory-size (generally consumer) GPUs
    if whiten_splits is None:
         previous_block_data_split = [previous_block_data] # If we're whitening in one go, then put it in a list for simplicity to reuse the logic below
    else:
         previous_block_data_split = previous_block_data.split(whiten_splits, dim=0) # Otherwise, we split this into different chunks to keep things manageable

    eigenvalue_list, eigenvector_list = [], []
    for data_split in previous_block_data_split:
        eigenvalues, eigenvectors = get_whitening_parameters(get_patches(data_split, patch_shape=layer.weight.data.shape[2:]))
        eigenvalue_list.append(eigenvalues)
        eigenvector_list.append(eigenvectors)

    eigenvalues  = torch.stack(eigenvalue_list,  dim=0).mean(0)
    eigenvectors = torch.stack(eigenvector_list, dim=0).mean(0)
    # i believe the eigenvalues and eigenvectors come out in float32 for this because we implicitly cast it to float32 in the patches function (for numerical stability)
    set_whitening_conv(layer, eigenvalues.to(dtype=layer.weight.dtype), eigenvectors.to(dtype=layer.weight.dtype), freeze=freeze)
    data = layer(previous_block_data.to(dtype=layer.weight.dtype))
    return data

def set_whitening_conv(conv_layer, eigenvalues, eigenvectors, eps=1e-2, freeze=True):
    shape = conv_layer.weight.data.shape
    eigenvectors_sliced = (eigenvectors/torch.sqrt(eigenvalues+eps))[-shape[0]:, :, :, :] # set the first n filters of the weight data to the top n significant (sorted by importance) filters from the eigenvectors
    conv_layer.weight.data = torch.cat((eigenvectors_sliced, -eigenvectors_sliced), dim=0)
    ## We don't want to train this, since this is implicitly whitening over the whole dataset
    ## For more info, see David Page's original blogposts (link in the README.md as of this commit.)
    if freeze: 
        conv_layer.weight.requires_grad = False


#############################################
#            Network Definition             #
#############################################
hyp = {
    'opt': {
        'non_bias_lr':    1.525 / 512,
        'scaling_factor': 1./9,
        'percent_start': .23,
        'loss_scale_scaler': 1./32, # * Regularizer inside the loss summing (range: ~1/512 - 16+). FP8 should help with this somewhat too, whenever it comes out. :)
    },
    'net': {
        'whitening': {
            'kernel_size': 2,
            'num_examples': 50000,
        },
        'batch_norm_momentum': .4, # * Don't forget momentum is 1 - momentum here (due to a quirk in the original paper... >:( )
        'cutmix_size': 3,
        'cutmix_epochs': 6,
        'pad_amount': 2,
        'base_depth': 64 ## This should be a factor of 8 in some way to stay tensor core friendly
    },
    'misc': {
        'ema': {
            'epochs': 10, # Slight bug in that this counts only full epochs and then additionally runs the EMA for any fractional epochs at the end too
            'decay_base': .95,
            'decay_pow': 3.,
            'every_n_steps': 5,
        },
        'train_epochs': 12.1,
        'device': 'cuda',
        'data_location': 'data.pt',
    }
}


scaler = 2. ## You can play with this on your own if you want, for the first beta I wanted to keep things simple (for now) and leave it out of the hyperparams dict
depths = {
    'init':   round(scaler**-1*hyp['net']['base_depth']), # 32  w/ scaler at base value
    'block1': round(scaler** 0*hyp['net']['base_depth']), # 64  w/ scaler at base value
    'block2': round(scaler** 2*hyp['net']['base_depth']), # 256 w/ scaler at base value
    'block3': round(scaler** 3*hyp['net']['base_depth']), # 512 w/ scaler at base value
    'num_classes': 10
}

class SpeedyConvNet(nn.Module):
    def __init__(self, network_dict):
        super().__init__()
        self.net_dict = network_dict # flexible, defined in the make_net function

    # This allows you to customize/change the execution order of the network as needed.
    def forward(self, x):
        if not self.training:
            x = torch.cat((x, torch.flip(x, (-1,))))
        x = self.net_dict['initial_block']['whiten'](x)
        x = self.net_dict['initial_block']['activation'](x)
        x = self.net_dict['conv_group_1'](x)
        x = self.net_dict['conv_group_2'](x)
        x = self.net_dict['conv_group_3'](x)
        x = self.net_dict['pooling'](x)
        x = self.net_dict['linear'](x)
        if not self.training:
            # Average the predictions from the lr-flipped inputs during eval
            orig, flipped = x.split(x.shape[0]//2, dim=0)
            x = .5 * orig + .5 * flipped
        return x

## This is actually (I believe) a pretty clean implementation of how to do something like this, since shifted-square masks unique to each depth-channel can actually be rather
## tricky in practice. That said, if there's a better way, please do feel free to submit it! This can be one of the harder parts of the code to understand (though I personally get
## stuck on the fold/unfold process for the lower-level convolution calculations.
def make_random_square_masks(inputs, mask_size):
    ##### TODO: Double check that this properly covers the whole range of values. :'( :')
    if mask_size == 0:
        return None # no need to cutout or do anything like that since the patch_size is set to 0
    is_even = int(mask_size % 2 == 0)
    in_shape = inputs.shape

    # seed centers of squares to cutout boxes from, in one dimension each
    mask_center_y = torch.empty(in_shape[0], dtype=torch.long, device=inputs.device).random_(mask_size//2-is_even, in_shape[-2]-mask_size//2-is_even)
    mask_center_x = torch.empty(in_shape[0], dtype=torch.long, device=inputs.device).random_(mask_size//2-is_even, in_shape[-1]-mask_size//2-is_even)

    # measure distance, using the center as a reference point
    to_mask_y_dists = torch.arange(in_shape[-2], device=inputs.device).view(1, 1, in_shape[-2], 1) - mask_center_y.view(-1, 1, 1, 1)
    to_mask_x_dists = torch.arange(in_shape[-1], device=inputs.device).view(1, 1, 1, in_shape[-1]) - mask_center_x.view(-1, 1, 1, 1)

    to_mask_y = (to_mask_y_dists >= (-(mask_size // 2) + is_even)) * (to_mask_y_dists <= mask_size // 2)
    to_mask_x = (to_mask_x_dists >= (-(mask_size // 2) + is_even)) * (to_mask_x_dists <= mask_size // 2)

    final_mask = to_mask_y * to_mask_x ## Turn (y by 1) and (x by 1) boolean masks into (y by x) masks through multiplication. Their intersection is square, hurray! :D

    return final_mask

def batch_crop(inputs, crop_size):
    with torch.no_grad():
        crop_mask_batch = make_random_square_masks(inputs, crop_size)
        cropped_batch = torch.masked_select(inputs, crop_mask_batch).view(inputs.shape[0], inputs.shape[1], crop_size, crop_size)
        return cropped_batch

def batch_flip_lr(batch_images, flip_chance=.5):
    with torch.no_grad():
        # TODO: Is there a more elegant way to do this? :') :'((((
        return torch.where(torch.rand_like(batch_images[:, 0, 0, 0].view(-1, 1, 1, 1)) < flip_chance, torch.flip(batch_images, (-1,)), batch_images)


def make_net():
    # TODO: A way to make this cleaner??
    # Note, you have to specify any arguments overlapping with defaults (i.e. everything but in/out depths) as kwargs so that they are properly overridden (TODO cleanup somehow?)
    whiten_conv_depth = 3*hyp['net']['whitening']['kernel_size']**2
    network_dict = nn.ModuleDict({
        'initial_block': nn.ModuleDict({
            'whiten': nn.Conv2d(3, whiten_conv_depth*2, kernel_size=hyp['net']['whitening']['kernel_size'], padding=0),
            'activation': nn.GELU(),
        }),
        'conv_group_1': ConvGroup(whiten_conv_depth*2, depths['block1']),
        'conv_group_2': ConvGroup(depths['block1'],    depths['block2']),
        'conv_group_3': ConvGroup(depths['block2'],    depths['block3']),
        'pooling': FastGlobalMaxPooling(),
        'linear': nn.Linear(depths['block3'], depths['num_classes'], bias=False),
    })

    net = SpeedyConvNet(network_dict)
    net = net.to(hyp['misc']['device'])
    net = net.to(memory_format=torch.channels_last) # to appropriately use tensor cores/avoid thrash while training
    net.train()


    ## Initialize the whitening convolution
    with torch.no_grad():
        # Initialize the first layer to be fixed weights that whiten the expected input values of the network be on the unit hypersphere. (i.e. their...average vector length is 1.?, IIRC)
        init_whitening_conv(net.net_dict['initial_block']['whiten'],
                            data.index_select(0, torch.randperm(data.shape[0], device=data.device)),
                            num_examples=hyp['net']['whitening']['num_examples'],
                            pad_amount=hyp['net']['pad_amount'],
                            whiten_splits=5000) ## Hardcoded for now while we figure out the optimal whitening number
                                                ## If you're running out of memory (OOM) feel free to decrease this, but
                                                ## the index lookup in the dataloader may give you some trouble depending
                                                ## upon exactly how memory-limited you are


        for layer_name in net.net_dict.keys():
            if 'conv_group' in layer_name:
                # Create an implicit residual via a dirac-initialized tensor
                dirac_weights_in = torch.nn.init.dirac_(torch.empty_like(net.net_dict[layer_name].conv1.weight))

                # Add the implicit residual to the already-initialized convolutional transition layer.
                # One can use more sophisticated initializations, but this one appeared worked best in testing.
                # What this does is brings up the features from the previous residual block virtually, so not only 
                # do we have residual information flow within each block, we have a nearly direct connection from
                # the early layers of the network to the loss function.
                std_pre, mean_pre = torch.std_mean(net.net_dict[layer_name].conv1.weight.data)
                net.net_dict[layer_name].conv1.weight.data = net.net_dict[layer_name].conv1.weight.data + dirac_weights_in 
                std_post, mean_post = torch.std_mean(net.net_dict[layer_name].conv1.weight.data)

                # Renormalize the weights to match the original initialization statistics
                net.net_dict[layer_name].conv1.weight.data.sub_(mean_post).div_(std_post).mul_(std_pre).add_(mean_pre)

                ## We do the same for the second layer in each convolution group block, since this only
                ## adds a simple multiplier to the inputs instead of the noise of a randomly-initialized
                ## convolution. This can be easily scaled down by the network, and the weights can more easily
                ## pivot in whichever direction they need to go now.
                ## The reason that I believe that this works so well is because a combination of MaxPool2d
                ## and the nn.GeLU function's positive bias encouraging values towards the nearly-linear
                ## region of the GeLU activation function at network initialization. I am not currently
                ## sure about this, however, it will require some more investigation. For now -- it works! D:
                torch.nn.init.dirac_(net.net_dict[layer_name].conv2.weight)


    return net

def get_network():
    network = make_net()
    return network

class ActivationQuantizedLinear(nn.Linear):
    def __init__(self, in_features, out_features, fnumber, bnumber, round_mode):
        super(ActivationQuantizedLinear, self).__init__(in_features, out_features)
        self.quant = Quantizer(forward_number=fnumber, forward_rounding=round_mode)
        self.quant_b = Quantizer(backward_number=bnumber, backward_rounding=round_mode)

    def forward(self, x):
        x = self.quant(x)
        x = super().forward(x)
        x = self.quant_b(x)
        return x

class ActivationQuantizedConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, padding, bias, fnumber, bnumber, round_mode):
        super(ActivationQuantizedConv2d, self).__init__(in_channels, out_channels, kernel_size, padding=padding, bias=bias)
        self.quant = Quantizer(forward_number=fnumber, forward_rounding=round_mode)
        self.quant_b = Quantizer(backward_number=bnumber, backward_rounding=round_mode)

    def forward(self, x):
        x = self.quant(x)
        x = super().forward(x)
        x = self.quant_b(x)
        return x

class QuantisedWrapper(nn.Module):
    def __init__(self, network, fnumber, bnumber, round_mode):
        super(QuantisedWrapper, self).__init__()
        self.network = network
        self.quant = Quantizer(forward_number=fnumber, forward_rounding=round_mode)
        self.quant_b = Quantizer(backward_number=bnumber, backward_rounding=round_mode)

    def forward(self, x):
        x = self.quant(x)
        x = self.network(x)
        x = self.quant_b(x)
        return x

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


network = get_network()
network_q = replace_linear_with_quantized(deepcopy(network), fnumber, bnumber, args.round_mode)

# %%

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
        self.grad_clip = 100.0
        ce = nn.CrossEntropyLoss()
        self.loss_fn = lambda x, y: ce(x, y) * 1024
        self.acc_count = 0

    # --- for mix precision training ---
    def model_grads_to_master_grads(self):
        for model, master in zip(self.model_weight.parameters(), self.master_weight.parameters()):
            if master.grad is None:
                master.grad = master.data.new(*master.data.size())
            if model.grad is None:
                continue
            master.grad.data.copy_(model.grad.data)

    def master_grad_apply(self, fn):
        for master in (self.master_weight.parameters()):
            if master.grad is None:
                master.grad = master.data.new(*master.data.size())
            master.grad.data = fn(master.grad.data)

    def master_params_to_model_params(self):
        for model, master in zip(self.model_weight.parameters(), self.master_weight.parameters()):
            model.data.copy_(self.weight_quant(master.data))

    def train_on_batch(self, data, target):
        if args.man_width == 23:
            self.master_weight.zero_grad()
            output = self.master_weight(data)
            loss = self.loss_fn(output, target)
            loss.backward()
            self.optimizer.step()
            return loss

        self.master_weight.zero_grad()
        self.model_weight.zero_grad()
        ds = data.chunk(args.grad_acc)
        ts = target.chunk(args.grad_acc)
        for d, t in zip(ds, ts):
            output = self.model_weight(d)
            loss = self.loss_fn(output, t)
            loss.backward()
        self.model_grads_to_master_grads()
        self.master_grad_apply(lambda x: x / self.grad_scaling / args.grad_acc)
        nn.utils.clip_grad_norm_(self.master_weight.parameters(), self.grad_clip)
        self.optimizer.step()
        self.master_params_to_model_params()
        return loss
        self.model_weight.zero_grad()
        self.master_weight.zero_grad()
        output = self.model_weight(data)
        loss = self.loss_fn(output, target)
        loss = loss * self.grad_scaling
        loss.backward()
        self.model_grads_to_master_grads()
        self.master_grad_apply(lambda x: x / self.grad_scaling)
        nn.utils.clip_grad_norm_(self.master_weight.parameters(), self.grad_clip)
        self.optimizer.step()
        self.master_params_to_model_params()
        return loss


# %%
import torch
import torch.nn as nn
import wandb
device = "cuda"


def grad_on_dataset(network, dataloader):
    network.train()
    total_norm = 0
    network.zero_grad()
    idx = list(range(len(dataloader.data)))
    random.shuffle(idx)
    idx = idx[:2048]
    data = dataloader.data[idx]
    target = dataloader.targets[idx]
    output = network(data)
    loss = torch.nn.CrossEntropyLoss()(output, target)
    loss.backward()
    total_norm += nn.utils.clip_grad_norm_(network.parameters(), float('inf'))
    network.zero_grad()
    network.eval()
    return {"grad_norm": total_norm.item()}

def test(network, dataset):
    network.eval()
    correct = 0
    total_loss = 0
    criterion = torch.nn.CrossEntropyLoss()
    i = 0
    with torch.no_grad():
        for data, target in dataset:
            i += 1
            output = network(data)
            total_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    accuracy = correct / len(dataset.data) 
    avg_loss = total_loss / i
    network.train()
    return {"acc": accuracy, "test_loss": avg_loss}

class MovingAvg():
    def __init__(self, beta=0.9):
        self.beta = beta
        self.average = None

    def update(self, value):
        if self.average is None:
            self.average = value
        else:
            self.average = self.beta * self.average + (1 - self.beta) * value

    def get(self):
        return self.average

class MovingAvgStat():
    def __init__(self, beta):
        self.beta = beta
        self.stats = {}

    def add_value(self, stats):
        for key in stats:
            if key not in self.stats:
                self.stats[key] = MovingAvg(self.beta)
            self.stats[key].update(stats[key])

    def get(self):
        return {f"{key}_mov_avg": self.stats[key].get() for key in self.stats}

def report_stats(network):
    stats = {}
    i = 0
    for _, m in enumerate(network.modules()):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            stats[f"{i}_w_norm"] = m.weight.data.norm().item()
            stats[f"{i}_w_mean"] = m.weight.data.mean().item()
            stats[f"{i}_w_std"] = m.weight.data.std().item()
            stats[f"{i}_w_max"] = m.weight.data.max().item()
            stats[f"{i}_w_min"] = m.weight.data.min().item()
            stats[f"{i}_g_norm"] = torch.sqrt((m.weight.grad.data ** 2).sum()).item() if m.weight.grad is not None else 0
            stats[f"{i}_g_mean"] = m.weight.grad.data.mean().item() if m.weight.grad is not None else 0
            stats[f"{i}_g_std"] = m.weight.grad.data.std().item() if m.weight.grad is not None else 0
            i += 1
    return stats

def train(network, dataset, test_dataset, steps, lr=args.lr):
    wandb.init(project="playground sr", group=args.group, id=args.id)
    wandb.watch(network, log='all', log_freq=30)

    if args.scheduler == "inverse":
        lr = 100.0
    optimizer = torch.optim.SGD(network.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)
    elif args.scheduler == "inverse":
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda x: 1 / (x + 1))
    elif args.scheduler == "fix":
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda x: 1)
    elif args.scheduler == "warmuplinear":
        initial_div_factor = 1e16 # basically to make the initial lr ~0 or so :D
        final_lr_ratio = .07 # Actually pretty important, apparently!
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,  max_lr=lr, pct_start=0.23, div_factor=initial_div_factor, final_div_factor=1./(initial_div_factor*final_lr_ratio), total_steps=steps, anneal_strategy='linear', cycle_momentum=False)
    iteration = 0
    running_weight = replace_linear_with_quantized(deepcopy(network), fnumber, bnumber, args.round_mode)
    running_weight.to(device)
    wrapper = MasterWeightOptimizerWrapper(network, model_weight=running_weight, optimizer=optimizer, weight_quant=qtorch.quant.quantizer(fnumber, forward_rounding=args.round_mode), grad_scaling=1.0)

    bar = tqdm.tqdm(total=steps)
    result_log = {}
    while True:
        for data, target in (dataset):
            if iteration >= steps:
                return network
            bar.update(1)
            iteration += 1
            loss = wrapper.train_on_batch(data, target)
            scheduler.step()
            result_log["loss"] = loss.item()

            if iteration % 100 == 0:
                test_metrics = test(network, test_dataset)
                result_log.update(test_metrics)
            if iteration % 300 == 0:
                result_log.update(grad_on_dataset(network, dataset))
            result_log.update({"lr": scheduler.get_last_lr()[0]})
                
            bar.set_postfix(result_log)
            wandb.log(result_log)


# %%
import random

network = get_network()

device = "cuda"
targets = torch.tensor(TrainData.targets).to(device)
train_set_end = len(data) * 9 // 10
test_set_end = len(data)
train_loader = MyDataLoader(data[:train_set_end], targets[:train_set_end], args.batch_size)
test_loader = MyDataLoader(data[train_set_end:test_set_end], targets[train_set_end:test_set_end], test_set_end - train_set_end)
network = network.to(device)
print(network)
train(network,train_loader, test_loader, steps=args.steps)
# %%
