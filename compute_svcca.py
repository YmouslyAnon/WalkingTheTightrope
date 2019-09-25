from math import log2
import numpy as np
import pandas as pd
import torch as pt
from svcca import compute_svcca


image_sizes = {'pokemon': 128, 'celeba': 96, 'stl10': 96}
subs = {'train': {'stl10': 8, 'pokemon': 1, 'celeba': 10},
        'test': {'stl10': 4, 'pokemon': 1, 'celeba': 5}}
base = ''  # folder that contains the exported representations
mode = 'train'  # 'test'
for dataset in image_sizes.keys():
    try:
        prepped = pt.load(f'{dataset}-{mode}-prepped-for-svcca.pt')
    except:
        prepped = {mode: {dataset: {} for dataset in ('pokemon', 'stl10', 'celeba')}
                   for mode in ('train', 'test')}
    image_size = image_sizes[dataset]
    sub = subs[mode][dataset]
    sizes = [3, 6, 12, 96] if image_size/2**int(log2(image_size)) > 1 else [4, 8, 16, 128]
    processed = np.zeros((13, 13))
    processed[-1, -1] = 1 # exclude svcca of input vs input due to computation time (is 1 anyway)
    results = []
    try:
        for i, size1 in enumerate(sizes):
            n_channels = reversed([(3 * image_size ** 2) // (4 ** j * size1 ** 2) for j in range(4)])
            n_channels = n_channels if size1 <= 16 else [3]
            for j, channels1 in enumerate(n_channels):
                col = i*4+j

                for k, size2 in enumerate(sizes):
                    n_channels = reversed([(3 * image_size ** 2) // (4 ** j * size2 ** 2) for j in range(4)])
                    n_channels = n_channels if size2 <= 16 else [3]
                    for l, channels2 in enumerate(n_channels):

                        row = k*4+l
                        if processed[col, row]:
                            continue

                        if size1 == size2 and channels1 == channels2:
                            result = {'size1': size1, 'channels1': channels1, 'size2': size2, 'channels2': channels2,
                                      'mean': 1, 'std': 0}
                        else:
                            res = compute_svcca(base, dataset, size1, channels1, size2, channels2, mode,
                                                svd=0.99, sub=sub, prepped=prepped, dft=False)
                            print(f'{size1, channels1} vs {size2, channels2}', res['mean'].mean(), res['mean'].std())
                            result = {'size1': size1, 'channels1': channels1, 'size2': size2, 'channels2': channels2,
                                      'mean': res['mean'].mean(), 'std': res['mean'].std()}
                        results.append(result)
                        processed[row, col] = 1
                        processed[col, row] = 1
    finally:
        results = pd.DataFrame(results)
        results.to_csv(f'{dataset}-{mode}-svcca.csv')
        pt.save(prepped, f'{dataset}-{mode}-prepped-for-svcca.pt')
