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

from sklearn.metrics import average_precision_score

directory = 'VOC2012'
use_cuda = 1
batch_size = 32
num_epochs = 1
learning_rate = 1e-3
device = torch.device("cuda" if use_cuda else "cpu")

def print_nb_matrix(dataset, mat):
    cols = ["x={}".format(key) for key in dataset.labels_dict.keys()]
    rows = ["P({}|x)".format(key) for key in dataset.labels_dict.keys()]
    mat = pd.DataFrame(mat, columns=cols, index=rows).round(5).T
    print(mat)

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    losses = []
    for idx, batch in enumerate(train_loader):  
        data = batch.image.to(device)
        target = batch.labels.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_function(output, target)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        print('Epoch: {}, Samples: {}/{}, Loss: {}'.format(epoch, idx*batch_size,
                                                           len(train_loader)*batch_size,
                                                           loss.item()))
        train_loss = torch.mean(torch.tensor(losses))

    print('\nEpoch: {}'.format(epoch))
    print('Training set: Average loss: {:.4f}'.format(train_loss))
    return train_loss

def validate(model, device, val_loader):
    model.eval()
    val_loss = 0
    mAP = 0
    
    with torch.no_grad():
        for idx, batch in enumerate(val_loader):
            data = batch.image.to(device)
            target = batch.labels.to(device)
            output = model(data)
            batch_loss = loss_function(output, target)
            val_loss += batch_loss.item()
            pred = (torch.sigmoid(output).data > 0.5).float()
            t = target.to('cpu')
            p = pred.to('cpu')
            AP = sum([average_precision_score(t[i], p[i]) for i in range(len(t))])
            mAP += AP/len(t)

    # divide by the number of batches of batch size
    # get the average validation over all bins
    val_loss /= len(val_loader)
    mAP /= len(val_loader)
    print('Validation set: Average loss: {:.4f}, mAP: {:.4f}'.format(
        val_loss, mAP))
    return val_loss

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
train_set = VOCDataset(directory, 'train', transforms=tr, multi_instance=False)
train_loader = DataLoader(train_set, batch_size=batch_size, collate_fn=collate_wrapper, shuffle=True, num_workers=4)

val_set = VOCDataset(directory, 'val', transforms=tr)
val_loader = DataLoader(val_set, batch_size=batch_size, collate_fn=collate_wrapper, shuffle=True, num_workers=4)

model = models.resnet18(pretrained=True)
model.fc = nn.Linear(512, 20)
model.to(device)
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

# ====================================== #
# Use either:                            #
# loss_function = nn.BCEWithLogitsLoss() #
# loss_function = MultiLabelNBLoss(mat)  #
# ====================================== #
loss_function = MultiLabelNBLoss(mat)

train_losses = []
val_losses = []

for epoch in range(1, num_epochs + 1):
    train_loss = train(model, device, train_loader, optimizer, epoch)
    val_loss = validate(model, device, val_loader)
    
    if (len(val_losses) > 0) and (val_loss < min(val_losses)):
        torch.save(model.state_dict(), "model_{}_{:.4f}.pt".format(epoch, val_loss))
        print("Saving model (epoch {}) with lowest validation loss: {}"
            .format(epoch, val_loss))

    train_losses.append(train_loss)
    val_losses.append(val_loss)
