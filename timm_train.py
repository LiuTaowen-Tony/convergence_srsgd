import torch
import timm
import timm.data

model = timm.create_model('resnet18')

transform = timm.data.create_transform()
dataset = timm.data.create_loader("cifar10", batch_size=128, num_workers=4)

from timm.data import rand_augment_transform
from timm.data.random_erasing import RandomErasing
from torchvision import transforms

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

#--aa rand-m9-mstd0.5 --remode pixel --reprob 0.2 --amp --lr .016
tfm = rand_augment_transform(
    config_str='rand-m9-mstd0.5', 
)
random_erase = RandomErasing(probability=1, mode='pixel', device='cpu')
