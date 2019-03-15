import sys
import numpy as np
import pandas as pd

from loss import MultiLabelNBLoss
from naive_bayes import NaiveBayes
from dataset import VOCDataset, collate_wrapper

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from torchvision import transforms
import torchvision.models as models

directory = 'VOC2012'
use_cuda = 1
batch_size = 32
num_epochs = 50
learning_rate = 1e-3
device = torch.device("cuda" if use_cuda else "cpu")

def print_nb_matrix(dataset, mat):
    cols = ["x={}".format(key) for key in dataset.labels_dict.keys()]
    rows = ["P({}|x)".format(key) for key in dataset.labels_dict.keys()]
    mat = pd.DataFrame(mat, columns=cols, index=rows).round(5).T
    print(mat)

tr = transforms.Compose([transforms.CenterCrop(224), transforms.ToTensor()])

# Get the NB matrix from the dataset,
# counting multiple instances of labels.
nb_dataset = VOCDataset(directory, 'train', transforms=tr, multi_instance=True)
nb = NaiveBayes(nb_dataset, 1)
mat = nb.get_nb_matrix()
print_nb_matrix(nb_dataset, mat)
mat = torch.Tensor(mat).to(device)


# Define the training dataset, removing
# multiple instances for the training problem.
train = VOCDataset(directory, 'train', transforms=tr, multi_instance=False)
train_loader = DataLoader(train, batch_size=batch_size, collate_fn=collate_wrapper, shuffle=True, num_workers=4)

model = models.resnet18(pretrained=True)
model.fc = nn.Linear(512, len(train.labels_dict))
model.to(device)
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

loss_function = MultiLabelNBLoss()
train_losses = []
val_losses = []

model.train()

for epoch in range(1, num_epochs + 1):
    for idx, batch in enumerate(train_loader):  
        data = batch.image.to(device)
        target = batch.labels.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_function(mat, output, target)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        print('Epoch: {}, Samples: {}/{}, Loss: {}'.format(epoch, idx*batch_size,
                                                           len(train_loader)*batch_size,
                                                           loss.item()))
        train_loss = torch.mean(torch.tensor(train_losses))
    print('\nEpoch: {}'.format(epoch))
    print('Training set: Average loss: {:.4f}'.format(train_loss))