from datasets import Pokemon
from autoencoder import ConvAE2d
import torch
from torch import relu
from torch.optim import Adam
from torch.nn import MSELoss, ZeroPad2d
from apex.fp16_utils import FP16_Optimizer
from trainer.utils import rgetattr

################################################
#           TRAINER RELATED VARIABLES          #
################################################

MODEL = ConvAE2d
DATASET = Pokemon
OPTIMIZER = Adam
LOSS = MSELoss
LOGDIR = '/outdir/ae_bottleneck/pokemon/'
APEX = FP16_Optimizer


model = {
    'channels': [3*4**x for x in range(6)],
    'n_residual': (2, 2),
    'kernel_size': (3, 3),
    'activation': relu,
    'affine': True,
    'padding': ZeroPad2d,
    'input_channels': 3
}

dataset = {
    'folder': '/datadir/pokemon/',
    'fraction': 0.8,
    'scale': 0.5
}

dataloader = {
    'batch_size': 32,
    'num_workers': 4
}

loss = {
    # loss keyword arguments go here
}

optimizer = {
    'lr': 0.0005
}

apex = {
    'dynamic_loss_scale': True,
    'dynamic_loss_args': {'init_scale': 2**16},
    'verbose': False
}

trainer = {
    'storage': 'storage.hdf5',
    'split_sample': lambda x: (x, x),
    'transformation': lambda x: x[0],
    'loss_decay': 0.9
}

cuda =True
dtype = torch.float16
seed = 0

################################################
#       EXPERIMENT RELATED VARIABLES           #
################################################

validationset = {
    'folder': dataset['folder'],
    'fraction': dataset['fraction'] - 1,
    'scale': dataset['scale']
}

sizes = [4, 8, 16]
input_size = 256*dataset['scale']
sizes = {size: [int((model['input_channels']*input_size**2)/(4**i*size**2)) for i in range(4)] for size in sizes}
superfluous_strides = 0
n_samples = min(8, dataloader['batch_size'])
n_epochs = 1000
save_interval = n_epochs//10


def remove_stride(trainer):
    global superfluous_strides
    global model

    n_convs = len(model['channels']) - 1
    for i in range(superfluous_strides):
        rgetattr(trainer.model, f'conv{n_convs - i}.convolution').stride = (1, 1)
        rgetattr(trainer.model, f'dconv{i+1}.upsampling').scale_factor = 1.0

    return trainer


mod_trainer = ['remove_stride']
