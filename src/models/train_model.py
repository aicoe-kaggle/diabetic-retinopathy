import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
from torchvision.datasets import VisionDataset
from torchvision.datasets import MNIST, FashionMNIST, CIFAR100
from torchvision import transforms
from torch.utils.data import Subset

import numpy as np
import matplotlib.pylab as plt
from PIL import Image
plt.ion()

#this might need changes if multiples GPUs found on a node
device = "cuda" if torch.cuda.is_available() else "cpu"
data_loc = '/home/sanjay/kaggle/diabetic-retinopathy/data/raw'

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
        
    def forward(self, inp):
        return inp.flatten(start_dim=1, end_dim=-1)

def get_mnist_ds(DOWNLOAD_PATH):
  #Get raw data
  mnist_train = MNIST(DOWNLOAD_PATH, 
                      train=True, 
                      download=True,
                      transform = transforms.Compose([transforms.ToTensor()]))

  mnist_test = MNIST(DOWNLOAD_PATH, 
                     train=False, 
                     download=True,
                     transform = transforms.Compose([transforms.ToTensor()]))

  return mnist_train, mnist_test

def create_dataloader(dataset, batch_size, num_workers=8):
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             num_workers=num_workers,
                                             pin_memory=True)
    
    return dataloader

def plot_samples(dataset, N_samples=3):
    for idx in np.random.choice(dataset.data.shape[0], N_samples):
        plt.figure()
        plt.imshow(dataset.data[idx])
        print(dataset.targets[idx], dataset.classes[dataset.targets[idx]])

def compute_accuracy(dataloader, 
                     net, 
                     process_pred=lambda x: x.argmax(dim=1).float()):

    net.eval()
    pred, targets = torch.tensor([]), torch.tensor([])

    with torch.no_grad(): #context manager for inference since we don't need the memory footprint of gradients
        for idx, (data_example, data_target) in enumerate(dataloader):
            data_example = data_example.to(device)

            #make predictions
            label_pred = process_pred(net(data_example))
                
            #concat and store both predictions and targets
            label_pred = label_pred.to('cpu')
            pred = torch.cat((pred, label_pred))
            targets = torch.cat((targets, data_target.float()))

    assert(pred.shape == targets.shape)
    accuracy = torch.sum(pred == targets).item() / pred.shape[0]
    print(f'Accuracy = {accuracy:.4f}')


def train_network(net, 
                  N_epochs,
                  train_dataloader,
                  criterion,
                  optimizer,
                  device,
                  preprocess_target=None,
                  print_freq=10
                  ):

    net.train()
    net.to(device)

    for epoch in range(N_epochs):
        loss_list = []
        for idx, (data_example, data_target) in enumerate(train_dataloader):
            data_example = data_example.to(device)
            data_target = data_target.to(device)

            if preprocess_target is not None:
                data_target = preprocess_target(data_target)
                #data_target = convert_to_binary(data_target)
                
            pred = net(data_example)

            loss = criterion(pred, data_target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_list.append(loss.item())

        if epoch % print_freq == 0:        
            print(f'Epoch = {epoch} Loss = {np.mean(loss_list)}')    
        
    return net, loss_list

def run(N_epochs=10, batch_size=8, lr=1e-2):
  train_ds, test_ds = get_mnist_ds('./')

  plot_samples(train_ds, N_samples=3)

  train_dl = create_dataloader(train_ds, batch_size, num_workers=8)
  test_dl = create_dataloader(test_ds, batch_size, num_workers=8)  

  flattened_dimensions = np.prod(train_ds.data[0].shape)
  n_unique_classes = len(np.unique(train_ds.targets))
  net = nn.Sequential(Flatten(), 
                    nn.Linear(flattened_dimensions, 100),
                    nn.ReLU(),
                    nn.Linear(100, n_unique_classes),
                   )

  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(net.parameters(), lr=lr)

  net, loss_list = train_network(net,
                                 N_epochs,
                                 train_dl,
                                 criterion,
                                 optimizer,
                                 device,
                                 preprocess_target=None,
                                 print_freq=10
                                )

  return net