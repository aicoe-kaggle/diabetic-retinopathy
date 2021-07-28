import ray
from ray.util.sgd import TorchTrainer
from ray.util.sgd.torch import TrainingOperator

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, CIFAR10
from torchvision.transforms import Compose, PILToTensor, Normalize, ConvertImageDtype

import numpy as np

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer_list = nn.ModuleList()
        self.layer_list.append(nn.Conv2d(1, 16, 3, 1))
        self.layer_list.append(nn.Conv2d(16, 32, 3, 1))
        self.layer_list.append(nn.Conv2d(32, 64, 3, 1))
        self.activation = nn.ReLU()

        linear_input = ((28 - 2*len(self.layer_list))**2)*self.layer_list[-1].out_channels
        self.output = nn.Linear(linear_input, 10)

    def forward(self, x):
        for layer in self.layer_list:
            x = self.activation(layer(x))
        
        x = self.output(x.reshape(-1, self.output.in_features)) #logits
        
        return x


ray.init()

class MyTrainingOperator(TrainingOperator):
    def setup(self, config):
        if config['dataset'] == 'MNIST':

            ds_train = MNIST(root=config['download_loc'], 
                             train=True, 
                             download=True,
                             transform=Compose([PILToTensor(), ConvertImageDtype(dtype=torch.float32), Normalize((128.,), (255.,))]))

            ds_val = MNIST(root=config['download_loc'], 
                           train=False, 
                           download=True,
                           transform=Compose([PILToTensor(), ConvertImageDtype(dtype=torch.float32), Normalize((128.,), (255.,))]))
        
        elif config['dataset'] == 'CIFAR10':
        
            ds_train = CIFAR10(root=config['download_loc'], 
                               train=True, 
                               download=True,
                               transform=PILToTensor())

            ds_val = CIFAR10(root=config['download_loc'], 
                             train=False, 
                             download=True,
                             transform=PILToTensor())


        dl_train = DataLoader(ds_train, batch_size=config['batch_size'])
        dl_val = DataLoader(ds_val, batch_size=config['batch_size'])

        net = Net()
        optimizer = optim.Adam(net.parameters(), lr=1e-2)
        criterion = nn.CrossEntropyLoss()

        self.model, self.optimizer, self.criterion = \
            self.register(models=net,
                          optimizers=optimizer,
                          criterion=criterion,
                          schedulers=None)
        self.register_data(train_loader=dl_train, validation_loader=dl_val)

#uses DDP
num_workers = 6
batch_size_per_worker = 32

trainer = TorchTrainer(
    training_operator_cls = MyTrainingOperator,
    num_workers = num_workers,
    use_gpu = False,
    config = {'dataset': 'MNIST', 'download_loc': './', 'batch_size': num_workers*batch_size_per_worker})

for n in range(10):
    print(f'----------Epoch {n}---------------')
    stats = trainer.train() #one epoch
    print(stats)
    val_stats = trainer.validate()
    print(val_stats)

trainer.shutdown()