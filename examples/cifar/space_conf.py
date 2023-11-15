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
    'batch_size': [4196],                            # T in Parabel paper
    'epochs': np.arange(35,301).tolist(),  # M
    'lr': np.logspace(-4,1,num=100).tolist(),
    'lr_peak_epoch':[5],
    'momentum': np.linspace(0,1,num=100).tolist(),            # C
    'weight_decay': np.logspace(-6, -1, num=100).tolist(),
    'label_smoothing':np.linspace(0, 0.5, num=100),
    'lr_tta':[True],
    'num_workers':[8]
}



# Note: you can also specify probability distributions, e.g.,
# import scipy
#     'C': scipy.stats.expon(scale=100),
#     'gamma': scipy.stats.expon(scale=.1),
