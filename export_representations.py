from trainer import Config, Trainer
import torch as pt
from torch.utils.data import DataLoader
import os
import sys
from math import log2


def get_codes(trainer):
    codes = []
    labels = []
    with pt.no_grad():
        for batch in trainer.dataloader:
            labels.append(batch[1:])
            s, _ = trainer._transform(batch)
            c = trainer.model.encode(s)
            codes.append(c.cpu())
    codes = pt.cat(codes)
    labels = [pt.cat(l) for l in list(zip(*labels))]
    samples = list(zip(codes, *labels))
    return samples


user = sys.argv[1]
image_sizes = {'stl10': 96, 'pokemon': 128, 'celeba': 96}
dataset = 'celeba'
image_size = image_sizes[dataset]
raw = False
sizes = [3, 6, 12, 96] if image_size/2**int(log2(image_size)) > 1 else [4, 8, 16, 128]
dest_base = ''  # export directory for the latent codes
model_base_dir = ''  # base directory where the trained models are stored

for size in sizes:
    channels = [3] if raw else reversed([int((3*image_size**2)/(4**i*size**2)) for i in range(4)])
    for channel in channels:

        folder = f'{model_base_dir}/{dataset}/{size}/{channel}/ae'

        config = Config.from_file(f'{folder}/config.py') if not raw else Config.from_file(f'configs/{dataset}_config_ae.py')
        config.LOGDIR = folder
        val = True
        if 'stl10' in dataset:
            config.dataset['fraction'] = 1
            config.dataset['split'] = 'test'
            val = False
        config.superfluous_strides = log2(size)//2
        trainer = Trainer.from_config(config, copy_config=False)
        if size > 16:
            trainer.model.encode = lambda x: x
        else:
            config.remove_stride(trainer)
            trainer.load_latest(folder)
            for parameter in trainer.model.parameters():
                parameter.requires_grad = False

        dest = dest_base + f'{size}/{channel}/'
        if not os.path.isdir(dest):
            os.makedirs(dest)
        print(dest)
        samples = get_codes(trainer)
        pt.save(samples, dest+'codes_train.pt')

        if val:
            trainer.dataloader = DataLoader(config.DATASET(**config.validationset))
            samples = get_codes(trainer)
            pt.save(samples, dest+'codes_test.pt')

        del trainer
        del samples
        pt.cuda.empty_cache()


