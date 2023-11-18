

from ConfigSpace import Configuration, ConfigurationSpace, Float,Integer, Categorical, Normal,Constant

from smac import HyperparameterOptimizationFacade, Scenario,BlackBoxFacade
from smac.initial_design import RandomInitialDesign
from smac.runhistory.dataclasses import TrialValue
import numpy as np
import torch as ch
from torch.cuda.amp import GradScaler, autocast
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, lr_scheduler
from tqdm import tqdm
import yaml
import subprocess
import json
from ConfigSpace.hyperparameters import UniformFloatHyperparameter
import random
import argparse
from argparse import ArgumentParser
class resnet:
    def __init__(self, seed=0):
        self.sn = seed
    @property
    def configspace(self) -> ConfigurationSpace:
        cs = ConfigurationSpace(seed = self.sn)
        batch_size = Constant('batch_size',4196)
        lr = UniformFloatHyperparameter('lr',lower = 0.0001,upper=10,log = True)
        epochs = Integer('epochs',(30,300))
        momentum = Float('momentum',(0,1))
        weight_decay = UniformFloatHyperparameter('weight_decay',lower=0.000001,upper=0.1,log = True)
        label_smoothing = Float('label_smoothing',(0,0.5))
        lr_tta = Constant('lr_tta',value = 'True')
        num_workers = Constant('num_workers',value=8)
        lr_peak_epoch = Float('lr_peak_epoch',(0,1))
        cs.add_hyperparameters([batch_size,lr,epochs,momentum,weight_decay,label_smoothing,lr_tta,num_workers,lr_peak_epoch])
        return cs


    def train(self,config:Configuration,seed:int=0,sn:int=0):
    # Convert the hyperparameters to their appropriate types
        batch_size = int(config['batch_size'])
        lr = config['lr']
        epochs = int(config['epochs'])
        momentum = config['momentum']
        weight_decay = config['weight_decay']
        label_smoothing = config['label_smoothing']
        lr_tta = config['lr_tta']
        num_workers = int(config['num_workers'])
        lr_peak_epoch = int(random.random()*config['epochs'])
        
        # Generate the YAML configuration file
        yaml_config = {
            'batch_size': batch_size,
            'lr': lr,
            'epochs': epochs,
            'momentum': momentum,
            'weight_decay': weight_decay,
            'label_smoothing': label_smoothing,
            'lr_tta': lr_tta,
            'num_workers': num_workers,
            'lr_peak_epoch': lr_peak_epoch
        }
        data =  {
            'gpu': 0,
            'num_workers': 8,
            'train_dataset': '/scratch/zd2260/cifar_train.beton',
            'val_dataset': '/scratch/zd2260/cifar_test.beton'
            }
        seed = {'seednum': sn}
        new_hp = ''.join(['  {name}: {value}\n'.format(name=k, value=v) for k, v in yaml_config.items()])
        results = "training:\n"
        results+=new_hp
        dt = "data:\n"
        sd = "seed:\n"
        for key,value in seed.items():
            sd+=f"  {key}: {value}\n"
        for key,value in data.items():
            dt+= f"  {key}: {value}\n"
        print(dt+'\n'+results+'\n'+sd+'\n')
        with open(f'config{sn}.yaml', 'w') as f:
            f.write(dt+'\n'+results+'\n'+sd+'\n')
        command = ['python', 'train_cifar_100.py', '--config-file', f'config{sn}.yaml']
        process = subprocess.Popen(command, stdout=subprocess.PIPE)
        process.wait()

        # Read the evaluation score from the subprocess output
        results = None
        with open(f'result{sn}.json') as f:
            results = json.load(f)
            # print(results)
        evaluation_score = float(results['test'][-1])
        print(evaluation_score)

        return -evaluation_score 





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("seed", type=int,
                    help="random seed")

    args = vars(parser.parse_args())['seed']
    print(args)
    random.seed(args)
    model = resnet(args)

    # Scenario object
    scenario = Scenario(model.configspace, deterministic=False, n_trials=100,seed = args)

    intensifier_gp = BlackBoxFacade.get_intensifier(
        scenario,
        max_config_calls=1,  # We basically use one seed per config only
    )
    initial = RandomInitialDesign(scenario,n_configs = 8)


    # Now we use SMAC to find the best hyperparameters
    smac = BlackBoxFacade(
        scenario,
        model.train,
        intensifier=intensifier_gp,  # No intensifier initially
        initial_design=initial,  # No initial design, as we'll handle it manually
        overwrite=True,
    )

    # We can ask SMAC which trials should be evaluated next
    for i in range(80):
    
        info = smac.ask()
        assert info.seed is not None
        print(info.config)
        cost = model.train(config=info.config, seed=info.seed,sn=args)
        value = TrialValue(cost=cost, time=0.5)

        smac.tell(info, value)
