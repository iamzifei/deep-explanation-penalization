import itertools
import os


params_to_vary = {
    'regularizer_rate':   [0, 10, -1],
    'seed':   [x for x in range(5)],


}

ks = sorted(params_to_vary.keys())
vals = [params_to_vary[k] for k in ks]
param_combinations = list(itertools.product(*vals))  # list of tuples
# print(param_combinations)
# print(ks)
# print(vals)

for i in range(len(param_combinations)):
    param_str = 'python train.py '
    for j, key in enumerate(ks):
        param_str += '--'+key + ' ' + str(param_combinations[i][j]) + ' '
        print(param_str)

    os.system(param_str)
