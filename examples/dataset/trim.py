# Remove unused variables from the dataset (to get a lighter repo)

import numpy as np

file_name = input('Enter complete file name:')

dataset_lite = {}
keys = {'A_average', 'A_std', 'S_average', 'S_std', 't', 'gradU'}

with np.load(file_name) as data:
    temp = dict(data)
    for key in keys:
        dataset_lite[key] = temp[key]

np.savez_compressed(file_name[:-4]+'_lite', **dataset_lite)
