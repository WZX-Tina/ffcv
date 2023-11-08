import numpy as np

# number of trees T = 1 - 3
# paths traversed in tree (beam search) P = 10
# max no of labels in leaf M = 100
# misclass. penality for classifiers C = 10 (1) log loss (squared hinge loss)

# P = 5, 10, 15, 20, 50, 100 ?
data =  {
  'gpu': 0,
  'num_workers': 8,
  'train_dataset': '/tmp/cifar_train.beton',
  'val_dataset': '/tmp/cifar_test.beton'
  }
space = {
    'batch_size': [64,128],                            # T in Parabel paper
    'epochs': [100,200,300,400,500],  # M
    'lr': [0.001,0.01,0.1],
    'lr_peak_epoch':[5],
    'momentum': [0.9,0.5,0.1],            # C
    'weight_decay':[0.0005],
    'label_smoothing':[0.01,0.1,0.5],
    'lr_tta':[True],
    'num_workers':[8]
}



# Note: you can also specify probability distributions, e.g.,
# import scipy
#     'C': scipy.stats.expon(scale=100),
#     'gamma': scipy.stats.expon(scale=.1),
