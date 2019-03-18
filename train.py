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

torch.manual_seed(0)

def print_nb_matrix(dataset, mat):
    cols = ["x={}".format(key) for key in dataset.labels_dict.keys()]
    rows = ["P({}|x)".format(key) for key in dataset.labels_dict.keys()]
    mat = pd.DataFrame(mat, columns=cols, index=rows).round(5).T
    print(mat)

def train(model, device, train_loader, optimizer, epoch, loss_function):
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

def validate(model, device, val_loader, loss_function):
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
            AP = average_precision_score(t.numpy(), p.numpy())
            #AP = sum([average_precision_score(t[i], p[i]) for i in range(len(t))])
            mAP += AP

    # divide by the number of batches of batch size
    # get the average validation over all bins
    val_loss /= len(val_loader)
    mAP /= len(val_loader)
    print('Validation set: Average loss: {:.4f}, mAP: {:.4f}'.format(
        val_loss, mAP))
    return val_loss, mAP

def main(mode=1, num_epochs=1, num_workers=0, lr=learning_rate, sc=learning_rate, model_name=None):
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
    train_loader = DataLoader(train_set, batch_size=batch_size, collate_fn=collate_wrapper, shuffle=True, num_workers=num_workers)
    
    val_set = VOCDataset(directory, 'val', transforms=tr)
    val_loader = DataLoader(val_set, batch_size=batch_size, collate_fn=collate_wrapper, shuffle=True, num_workers=num_workers)
    
    if model_name == None:
        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(512, 20)
    else:
        try:
            model = models.resnet18(pretrained=True)
            model.fc = nn.Linear(512, 20)
            model.load_state_dict(torch.load(model_name + '.pt'))
        except:
            model = models.resnet18(pretrained=True)
            model.fc = nn.Linear(512, 20)
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    
    # ====================================== #
    # Use either:                            #
    # loss_function = nn.BCEWithLogitsLoss() #
    # loss_function = MultiLabelNBLoss(mat)  #
    # ====================================== #
    if mode == 0:
        loss_function = nn.BCEWithLogitsLoss()
    elif mode == 1:        
        loss_function = MultiLabelNBLoss(mat)
    if mode == 0:
        mode_ = 'BCE'
    elif mode == 1:
        mode_ = 'NB'
        
    if model_name == None:
        train_losses = []
        val_losses = []
        mAPs = []
    else:
        print('Loading history')
        train_losses = np.load('train_history_{}_{}.npy'.format(mode_, model_name)).tolist()
        val_losses = np.load('val_history_{}_{}.npy'.format(mode_, model_name)).tolist()
        mAPs = np.load('mAP_history_{}_{}.npy'.format(mode_, model_name)).tolist()
        
    model_save_name = ''
    if model_name != None:
        curr_epoch = int(model_name.split('_')[-2])
    else:
        curr_epoch = 0
    
    try:
        for epoch in range(1, num_epochs + 1):
            train_loss = train(model, device, train_loader, optimizer, curr_epoch+1, loss_function)
            val_loss, mAP = validate(model, device, val_loader, loss_function)
            
            if (len(val_losses) > 0) and (val_loss < min(val_losses)):
                torch.save(model.state_dict(), "lr{}_sc{}_model_{}_{}_{:.4f}.pt".format(lr, sc, mode_, curr_epoch+1, val_loss))
                print("Saving model (epoch {}) with lowest validation loss: {}"
                    .format(epoch, val_loss))
        
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            mAPs.append(mAP)
            torch.save(model.state_dict(), 'temp_model.pt')
            curr_epoch += 1
        model_save_name = "stop_lr{}_sc{}_model_{}_{}_{:.4f}.pt".format(lr, sc, mode_, curr_epoch, val_losses[-1])
        torch.save(model.state_dict(), model_save_name)
    except KeyboardInterrupt:
        model.load_state_dict(torch.load('temp_model.pt'))
        model_save_name = "pause_lr{}_sc{}_model_{}_{}_{:.4f}.pt".format(lr, sc, mode_, curr_epoch, val_losses[-1])
        torch.save(model.state_dict(), model_save_name)
        print("Saving model (epoch {}) with current validation loss: {}".format(curr_epoch, val_losses[-1]))
    
    train_history = np.array(train_losses)
    val_history = np.array(val_losses)
    mAP_history = np.array(mAPs)
    
    print('Saving history')
    np.save("train_history_{}_{}".format(mode_, model_save_name[5:-3]), train_history)
    np.save("val_history_{}_{}".format(mode_, model_save_name[5:-3]), val_history)
    np.save("mAP_history_{}_{}".format(mode_, model_save_name[5:-3]), mAP_history)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        args = sys.argv[1:]
        if len(args) == 5:
            main(mode=int(args[0]), num_epochs=int(args[1]), num_workers=int(args[2]), 
                 lr=float(args[3]), sc=float(args[4]))
        elif len(args) == 6:
            main(mode=int(args[0]), num_epochs=int(args[1]), num_workers=int(args[2]), 
                 lr=float(args[3]), sc=float(args[4]), model_name=args[5])
        else:
            response = '''Wrong number of arguments, please enter the following arguments:
1. Mode (int)
2. Max epochs (int)
3. Number of worker threads (int)
4. Learning rate (float)
5. Scaling constant (float)
6. Target model file name (optional)'''
            print(response)
    else:
        main(mode=0, num_epochs=20, num_workers=0, lr=1e-3, sc=1e-3)