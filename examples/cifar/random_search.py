import argparse
import os
import numpy as np
from sklearn.model_selection import ParameterSampler, ParameterGrid

from space_conf import space,data
import random
def main(args):
    # existing_pth = args.existing
    temp_pth = args.temp
    sn = args.seed
    # with open(existing_pth, "r") as file:
    #     existing_hyperparameters = file.read().splitlines()
    ps = ParameterSampler(space, n_iter=64, random_state=sn)
    k = 1
    for p in ps:
        p['lr_peak_epoch'] =int(random.random()*p['epochs'])
        new_hp = ''.join(['  {name}: {value}\n'.format(name=k, value=v) for k, v in p.items()])
        print('new_hp',new_hp)
    # if new_hp not in existing_hyperparameters:
    #     with open(existing_pth,"a") as file: 
    #         file.write(new_hp)
        with open('config/'+temp_pth+'-'+str(sn)+'-'+str(k)+'.yaml',"w") as file:
            results = "training:\n"
            results+=new_hp
            dt = "data:\n"
            for key,value in data.items():
                dt+= f"  {key}: {value}\n"
            seed = {'seednum': sn}
            sd = "seed:\n"
            for key,value in seed.items():
                sd+=f"  {key}: {value}\n"
            file.write(dt+'\n'+results+'\n'+sd+'\n')
        k += 1
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description = 'Generate a hyperparameter')
    # parser.add_argument('existing',type = str,
    #         help = 'exisiting hyperparameter file')
    parser.add_argument('temp',type = str,
            help = 'temporary hyperparameter file')
    parser.add_argument("seed", type=int,
            help="random seed")
    args = parser.parse_args()
    main(args)

