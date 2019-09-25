import os
import torch as pt
from torch.utils.data import DataLoader
from torch.nn import Module, Linear, BCEWithLogitsLoss
import pandas as pd
from trainer import Trainer, events
from datasets import Representations, Folds


class Classifier(Module):

    def __init__(self, sizes, activation=pt.relu, activation_on_final_layer=False):

        super(Classifier, self).__init__()
        self.activation = activation
        self.activation_on_final_layer = activation_on_final_layer
        self.layers = []
        for i, (n_in, n_out) in enumerate(zip(sizes[:-1], sizes[1:])):
            self.layers.append(Linear(n_in, n_out))
            self.add_module(f'layer{i}', self.layers[-1])

    def forward(self, x):

        for layer in self.layers:
            x = layer(x)
            if layer is not self.layers[-1] or self.activation_on_final_layer:
                x = self.activation(x)

        return x


def classify_and_evaluate(dataset, sizes, modes, n_classes, im_size, outdir, reshape, evaluate, folder, criterion,
                          n_workers=8, n_epochs=200):
    min_size = im_size // 2**5
    min_channels = (3 * im_size ** 2) // (4 ** 3 * min_size ** 2)
    min_vol = min_channels*min_size**2
    results = []
    for size in sizes:
        if size <= 16:
            n_channels = reversed([(3 * im_size ** 2) // (4 ** j * size ** 2) for j in range(4)])
        else:
            n_channels = [3]
        for channels in n_channels:
            for mode in modes:

                data = f'{folder}/{dataset}/{size}/{channels}/codes_{mode}.pt'
                folds = Folds(n_folds=5, dataset=Representations(data, fraction=1, transformation=reshape))

                for i, (trainset, valset) in enumerate(folds):
                    in_size = channels * size ** 2
                    model = Classifier([in_size, n_classes]).cuda()
                    optimizer = pt.optim.SGD(model.parameters(), lr=1 * min_vol / in_size, weight_decay=0.01)

                    trainloader = DataLoader(trainset, batch_size=len(trainset), num_workers=n_workers, shuffle=True)
                    valloader = DataLoader(valset, batch_size=len(valset), num_workers=n_workers, shuffle=False)

                    logdir = f'{outdir}/{dataset}/{size}/{channels}/classifier/{mode}/{i}/'
                    os.makedirs(logdir) if not os.path.isdir(logdir) else None
                    trainer = Trainer(model=model, criterion=criterion, optimizer=optimizer, dataloader=trainloader,
                                      logdir=logdir)

                    trainer.register_event_handler(events.EACH_EPOCH, trainer.validate, dataloader=valloader)
                    trainer.train(n_epochs=n_epochs, resume=True)
                    result = evaluate(trainer, valloader)
                    result['fold'] = i
                    result['size'] = size
                    result['channels'] = channels
                    result['dataset'] = dataset
                    result['mode'] = mode
                    #result.to_csv(os.path.join(trainer.logdir, 'result.csv'))
                    results.append(result)

    results = pd.concat(results, ignore_index=True)
    results.to_csv(os.path.join(outdir, 'results-loss.csv'))
    return results
