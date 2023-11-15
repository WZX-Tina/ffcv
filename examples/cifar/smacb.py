

from ConfigSpace import Configuration, ConfigurationSpace, Float,Integer, Categorical, Normal,Constant

from smac import HyperparameterOptimizationFacade, Scenario
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
class resnet:
    @property
    def configspace(self) -> ConfigurationSpace:
        cs = ConfigurationSpace(seed = 0)
        batch_size = Constant('batch_size',4196)
        lr = UniformFloatHyperparameter('lr',(0.0001,10),log = True)
        epochs = Integer('epochs',(30,300))
        momentum = Float('momentum',(0,1))
        weight_decay = UniformFloatHyperparameter('weight_decay',(0.000001,0.1),log = True)
        label_smoothing = Float('label_smoothing',(0,0.5))
        lr_tta = Constant('lr_tta',value = True)
        num_workers = Constant('num_workers',value=8)
        lr_peak_epoch = Float('lr_peak_epoch',(0,1))
        cs.add_hyperparameters([batch_size,lr,epochs,momentum,weight_decay,label_smoothing,lr_tta,num_workers,lr_peak_epoch])
        return cs


    def train(self,config:Configuration,seed:int=0):
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
            'train_dataset': '/tmp/cifar_train.beton',
            'val_dataset': '/tmp/cifar_test.beton'
            }
        new_hp = ''.join(['  {name}: {value}\n'.format(name=k, value=v) for k, v in yaml_config.items()])
        results = "training:\n"
        results+=new_hp
        dt = "data:\n"
        for key,value in data.items():
            dt+= f"  {key}: {value}\n"
        print(dt+'\n'+results+'\n')
        with open('config.yaml', 'w') as f:
            f.write(dt+'\n'+results+'\n')
        command = ['python', 'train_cifar_100.py', '--config-file', 'config.yaml']
        process = subprocess.Popen(command, stdout=subprocess.PIPE)
        process.wait()

        # Read the evaluation score from the subprocess output
        results = None
        with open('results.json') as f:
            results = json.load(f)
            # print(results)
        evaluation_score = float(results['test'][-1])
        print(evaluation_score)

        return -evaluation_score 





if __name__ == "__main__":
    model = resnet()

    # Scenario object
    scenario = Scenario(model.configspace, deterministic=False, n_trials=100)

    intensifier = HyperparameterOptimizationFacade.get_intensifier(
        scenario,
        max_config_calls=1,  # We basically use one seed per config only
    )

    # Now we use SMAC to find the best hyperparameters
    smac = HyperparameterOptimizationFacade(
        scenario,
        model.train,
        intensifier=intensifier,
        overwrite=True,
    )

    # We can ask SMAC which trials should be evaluated next
    for _ in range(10):
        info = smac.ask()
        assert info.seed is not None
        cost = model.train(config=info.config, seed=info.seed)
        value = TrialValue(cost=cost, time=0.5)

        smac.tell(info, value)
