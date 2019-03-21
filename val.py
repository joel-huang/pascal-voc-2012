import sys
import pickle
import numpy as np
import pandas as pd

from dataset import VOCDataset, collate_wrapper

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from torchvision import transforms
import torchvision.models as models

from eval import get_AP

torch.multiprocessing.set_sharing_strategy('file_system')

directory = 'VOC2012'
use_cuda = 1
batch_size = 48
device = torch.device("cuda" if use_cuda else "cpu")

torch.manual_seed(0)

def validate(model, device, val_loader, loss_function):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for idx, batch in enumerate(val_loader):
            data = batch.image.to(device)
            target = batch.labels.to(device)
            output = model(data)
            batch_loss = loss_function(output, target)
            val_loss += batch_loss.item()
            pred = torch.sigmoid(output)
            if idx == 0:
                predictions = pred
                targets = target
            else:
                predictions = torch.cat((predictions, pred))
                targets = torch.cat((targets, target))

    # divide by the number of batches of batch size
    # get the average validation over all bins
    val_loss /= len(val_loader)
    print('Validation set: Average loss: {:.4f}'.format(val_loss))
    print('                AP: {:.4f}'.format(
        get_AP(predictions.reshape(-1, 20), targets.reshape(-1, 20))))
    return val_loss, predictions, targets

def main(model_name=None):
    tr = transforms.Compose([transforms.RandomResizedCrop(300),
                             transforms.ToTensor(),
                             transforms.Normalize([0.4589, 0.4355, 0.4032],[0.2239, 0.2186, 0.2206])])

    val_set = VOCDataset(directory, 'val', transforms=tr)
    val_loader = DataLoader(val_set, batch_size=batch_size, collate_fn=collate_wrapper, shuffle=False, num_workers=16)

    model = models.resnet34(pretrained=True)
    model.fc = nn.Linear(512, 20)
    model.load_state_dict(torch.load(model_name + '.pt'))
    model.to(device)

    classwise_frequencies = np.array(list(val_set.classes_count.values()))
    minimum_frequency = np.min(classwise_frequencies)
    loss_weights = minimum_frequency / classwise_frequencies
    loss_weights = torch.Tensor(loss_weights).to(device)
    loss_function = nn.BCEWithLogitsLoss(weight=loss_weights)

    val_loss, predictions, targets = validate(model, device, val_loader, loss_function)

    print("Saving raw predictions for validation pass...")
    with open("{}_validation.pkl".format(model_name), 'wb') as f:
        pred_targets = torch.cat((predictions.unsqueeze(0), targets.unsqueeze(0)))
        pickle.dump(pred_targets, f)
    f.close()

if __name__ == '__main__':
    if len(sys.argv) == 2:
        model_name = str(sys.argv[1])
        main(model_name)
    else:
        response = '''Wrong number of arguments, please enter the following arguments: 1. Target model file name'''
        print(response)
