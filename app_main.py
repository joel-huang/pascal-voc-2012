import sys
import numpy as np
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from torchvision import transforms
import torchvision.models as models

from flask import Flask

import matplotlib.pyplot as plt
    
torch.manual_seed(0)

class CustomNet(nn.Module):
    def __init__(self, base_net=None):
        super().__init__()
        self.base_net = base_net
        self.pool = nn.AdaptiveAvgPool3d((3,224,224))
        
    def forward(self, x):
        self.base_net.eval()
        x_mod = self.pool(x)
        with torch.no_grad():
            out = self.base_net(x_mod)
        return out

class App(object):
    def __init__(self, imgs_dir='./VOC2012/JPEGImages/', pred_dir='./logs/attempt1/', models_dir='../logs/attempt1/'):
        self.imgs_dir = imgs_dir
        self.pred_dir = pred_dir
        self.models_dir = models_dir
        self.model = self._load_default_model()
        self.img_names = self.load_img_names('val')
        
    def _load_default_model(self):
        return self.load_model('stop_lr0.001_sc0.001_model_BCE_40_0.1131')
        
    def load_model(self, model_name):
        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(512, 20)
        model.load_state_dict(torch.load(self.models_dir + model_name + '.pt'))
        model = CustomNet(model)
        return model
        
    def load_img_names(self, dataset='val'):
        textfile_dir = './VOC2012/ImageSets/Main/'
        img_file_names = textfile_dir + dataset + '.txt'
        return img_file_names
        
    def main(self):
        pass
        
def load_pickle(file_path='./logs/attempt1/pred_BCE_40.pkl'):
    with open(file_path, 'rb') as f:
        ret = pickle.load(f)
    return ret
    
def get_preds_and_gt(pickled, output_type='numpy'):
    preds, gt = [], []
    for pair in pickled:
        preds.append(pair['pred'])
        gt.append(pair['target'])
    pred_tensor = torch.cat(preds, dim=0)
    gt_tensor = torch.cat(gt, dim=0)
    if output_type == 'numpy':
        return pred_tensor.cpu().numpy(), gt_tensor.cpu().numpy()
    return pred_tensor, gt_tensor
    
def rank_by_class(scores, col=0):
    out = np.copy(scores)
    new_order = out[:,col].argsort()[::-1]
    return out[new_order], new_order
    
def _test_pickle_and_rank():
    temp = load_pickle()
    preds, gt = get_preds_and_gt(temp, 'numpy')
    class_0, class_0_idx = rank_by_class(preds, 0)
    class_1, class_1_idx = rank_by_class(preds, 1)
    print(class_0[:10,0], class_0_idx[:10])
    print(class_1[:10,1], class_1_idx[:10])
    
def get_tail_acc(sorted_scores, sorted_gt, t_min, t_max, t_num):
    t_vals = np.linspace(t_min, t_max, t_num)
    tail_accs = []
    for t_val in t_vals:
        tp = 0
        fp = 0
        for (score, target) in zip(sorted_scores, sorted_gt):
            pred = 0
            if score > t_val:
                pred = 1
            if pred == target:
                tp += 1
            else:
                fp += 1
        tail_accs.append(tp/(tp+fp))
        
    return tail_accs
                    
def get_tail_acc_classwise(scores, gt, t_min=0.5, t_num=10):
    t_max = np.max(scores)
    tail_accs = []
    t_vals = np.linspace(t_min, t_max, t_num)
    for i in range(20):
        sorted_scores, idx_order = rank_by_class(scores, i)
        tail_accs_i = get_tail_acc(sorted_scores[:,i], gt[idx_order][:,i], t_min, t_max, t_num)
        tail_accs.append(tail_accs_i)
    return tail_accs, t_vals
    
def _plot_tail_acc():
    pickled = load_pickle()
    scores, gt = get_preds_and_gt(pickled, 'numpy')
    tail_accs, t_vals = get_tail_acc_classwise(scores, gt, 0.5, 10)
    tail_accs = np.array(tail_accs)
    ave_tail_accs = np.mean(tail_accs, axis=0)
    plt.figure()
    plt.plot(t_vals, ave_tail_accs)
    plt.xticks(t_vals)
    plt.ylabel('Tailaccs')
    plt.xlabel('t value')
    plt.show()
    
app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello, World!'   
    
if __name__ == '__main__':
    #_test_pickle_and_rank()
    _plot_tail_acc()
        