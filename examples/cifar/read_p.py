import argparse
import os
import numpy as np
from sklearn.model_selection import ParameterSampler, ParameterGrid

from space_conf import space
def main(args):
    existing_pth = args.existing
    temp_pth = args.temp
    with open(existing_pth, "r") as file:
        existing_hyperparameters = file.read().splitlines()
    ps = ParameterSampler(space, n_iter=1)
    for p in ps:
        new_hp = ' '.join(['{name}:{value}'.format(name=k, value=v) for k, v in p.items()])
    if new_hp not in existing_hyperparameters:
        with open(existing_pth,"a") as file:
            file.write(new_hp+'\n')
        with open(temp_pth,"w") as file:
            file.write(new_hp)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description = 'Generate a hyperparameter')
    parser.add_argument('existing',type = str,
            help = 'exisiting hyperparameter file')
    parser.add_argument('temp',type = str,
            help = 'temporary hyperparameter file')
    args = parser.parse_args()
    main(args)

