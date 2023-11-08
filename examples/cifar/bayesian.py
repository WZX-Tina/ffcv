import skopt
from skopt import gp_minimize
from fastargs import get_current_config, Param, Section
from fastargs.decorators import param
from argparse import ArgumentParser
import numpy as np
import torch as ch
from torch.cuda.amp import GradScaler, autocast
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, lr_scheduler
from tqdm import tqdm
SPACE = [
   skopt.space.Integer(16, 512, name='batch_size'),
   skopt.space.Real(0.001, 0.5, name='lr', prior='log-uniform'),
   skopt.space.Integer(100, 500, name='epochs'),
   skopt.space.Real(0.1, 1.0, name='momentum', prior='uniform'),
   skopt.space.Real(0.0005,0.005,name = 'weight_decay'),
   skopt.space.Real(0.01,0.5,name = 'label_smoothing',prior = 'uniform'),
   skopt.space.Categorical(['True','False'],name = 'lr_tta'),
   skopt.space.Integer(8,9,name = 'num_workers'),
   skopt.space.Integer(5,10,name = 'lr_peak_epoch')
]
from train_cifar_100 import evaluate
from train_cifar_100 import construct_model,make_dataloaders

@skopt.utils.use_named_args(SPACE)
def train_evaluate(batch_size, lr, epochs, momentum, weight_decay, label_smoothing, lr_tta, num_workers, lr_peak_epoch):
    ##
    model = construct_model()
    loaders, start_time = make_dataloaders('/tmp/cifar_train.beton','/tmp/cifar_test.beton',batch_size,num_workers)
    ##
    opt = SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    iters_per_epoch = len(loaders['train'])
    # Cyclic LR with single triangle
    lr_schedule = np.interp(np.arange((epochs+1) * iters_per_epoch),
                            [0, lr_peak_epoch * iters_per_epoch, epochs * iters_per_epoch],
                            [0, 1, 0])
    scheduler = lr_scheduler.LambdaLR(opt, lr_schedule.__getitem__)
    scaler = GradScaler()
    loss_fn = CrossEntropyLoss(label_smoothing=label_smoothing)

    for _ in range(epochs):
        for ims, labs in tqdm(loaders['train']):
            opt.zero_grad(set_to_none=True)
            with autocast():
                out = model(ims)
                loss = loss_fn(out, labs)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            scheduler.step()
    model.eval()
    accuracies = {}
    with ch.no_grad():
        for name in ['train', 'test']:
            total_correct, total_num = 0., 0.
            for ims, labs in tqdm(loaders[name]):
                with autocast():
                    out = model(ims)
                    if lr_tta:
                        out += model(ims.flip(-1))
                    total_correct += out.argmax(1).eq(labs).sum().cpu().item()
                    total_num += ims.shape[0]
            accuracy = round(total_correct / total_num * 100, 2)
            print(f'{name} accuracy: {total_correct / total_num * 100:.1f}%')
            accuracies[name] = accuracy
    return -accuracies['test']

    #Set resnet18 params:

    #return evaluation results


# Print the best hyperparameters found





if __name__ == "__main__":
    
    res = gp_minimize(train_evaluate, SPACE, n_calls=20, random_state=42)
    best_batch_size, best_lr, best_epochs, best_momentum, best_weight_decay, best_label_smoothing, best_lr_tta, best_num_workers, best_lr_peak_epoch = res.x
    print("Best Hyperparameters:")
    print(f"Batch Size: {best_batch_size}")
    print(f"Learning Rate: {best_lr}")
    print(f"Epochs: {best_epochs}")
    print(f"Momentum: {best_momentum}")
    print(f"Weight Decay: {best_weight_decay}")
    print(f"Label Smoothing: {best_label_smoothing}")
    print(f"LR TTA: {best_lr_tta}")
    print(f"Num Workers: {best_num_workers}")
    print(f"LR Peak Epoch: {best_lr_peak_epoch}")
