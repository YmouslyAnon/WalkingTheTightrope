from trainer import Config
import os

from math import log2
from train import run_training


def prep_config(dataset, size, channel, local=''):

    config = Config.from_file(f'configs/{dataset}_config_ae.py')

    if local:
        global DATADIR
        config.LOGDIR = local
        if 'stl10' in dataset:
            config.dataset['folder'] = DATADIR
        else:
            config.dataset['folder'] = os.path.join(DATADIR, dataset)
        config.validationset['folder'] = config.dataset['folder']

    config.model['channels'][-1] = channel
    config.LOGDIR += f'{size}/{channel}/ae'
    config.superfluous_strides = int(log2(size / min(config.sizes)))
    return config


DATADIR = ''  # directory containing the
LOCAL = os.getcwd() + '/'
DATASETS = ['pokemon', 'stl10', 'celeba']
DATASET = DATASETS[2]
SIZES = [(6, [48])]  # used to run experiments on ly for a certain shape instead of all
SIZES = SIZES or Config.from_file(f'configs/{DATASET}_config_ae.py').sizes.items()

if __name__ == '__main__':

    for size, channels in SIZES:
        for channel in channels:

            config = prep_config(DATASET, size, channel, LOCAL)
            print('running', size, channel)
            config.save('temp_config.py', False)
            config = Config.from_file('temp_config.py')
            run_training(config)
