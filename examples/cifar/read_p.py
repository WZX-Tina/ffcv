import argparse
import os
import numpy as np
from sklearn.model_selection import ParameterSampler, ParameterGrid

from space_conf import space,data
def main(args):
    existing_pth = args.existing
    temp_pth = args.temp
    with open(existing_pth, "r") as file:
        existing_hyperparameters = file.read().splitlines()
    ps = ParameterSampler(space, n_iter=1)
    for p in ps:
        print(p)
        new_hp = ''.join(['  {name}: {value}\n'.format(name=k, value=v) for k, v in p.items()])
        print(new_hp)
    if new_hp not in existing_hyperparameters:
        with open(existing_pth,"a") as file: 
            file.write(new_hp)
        with open(temp_pth,"w") as file:
            results = "training:\n"
            results+=new_hp
            dt = "data:\n"
            for key,value in data.items():
                dt+= f"  {key}: {value}\n"
            file.write(dt+'\n'+results+'\n')
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description = 'Generate a hyperparameter')
    parser.add_argument('existing',type = str,
            help = 'exisiting hyperparameter file')
    parser.add_argument('temp',type = str,
            help = 'temporary hyperparameter file')
    args = parser.parse_args()
    main(args)

