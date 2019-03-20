import sys
import numpy as np

import torch
import torch.nn as nn
import torchvision.models as models

from metrics import *
from flask import Flask, render_template, request

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

class Backend(object):
    def __init__(self, imgs_dir='./VOC2012/JPEGImages/', pred_dir='./logs/attempt1/', models_dir='./logs/attempt1/'):
        self.imgs_dir = imgs_dir
        self.pred_dir = pred_dir
        self.models_dir = models_dir
        self.model = self._load_default_model()
        self.img_paths = self._load_img_paths()
        self.val_order = self._load_val_order()
        
    def _load_default_model(self):
        return self.load_model('stop_lr0.001_sc0.001_model_BCE_40_0.1131')
    
    def _load_val_order(self):
        return np.load('./logs/val_order.npy')
        
    def _load_img_paths(self):    
        image_paths = []
        image_path_file = './VOC2012/ImageSets/Main/val.txt' 
        with open(image_path_file) as f:
            for image_path in f.readlines():
                candidate_path = image_path.split(' ')[0].strip('\n')
                image_paths.append(candidate_path)
        return image_paths
    
    def load_model(self, model_name):
        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(512, 20)
        model.load_state_dict(torch.load(self.models_dir + model_name + '.pt'))
        model = CustomNet(model)
        return model

    def load_img(self, idx):
        idx = int(idx[0])
        return self.imgs_dir + self.img_paths[idx] + '.jpg'
    
    def get_prediction(self, x):
        scores = self.model(x)
        scores = scores.cpu().numpy()
        preds = []
        for i in scores:
            pred = 0
            if i > 0.5:
                pred = 1
            preds.append(pred)
        return preds
        
    def main(self):
        pass
    
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')
    
@app.route('/predict')
def predict():
    return render_template('predict.html')
    
@app.route('/predict', methods=['GET'])
def get_url():
    text = request.form['file_path']
    
    return render_template('predict_output.html')
    
@app.route('/browse')
def browse():
    return render_template('browse.html')
    
if __name__ == '__main__':
    backend = Backend()
    app.run(debug=False)
    
    #_test_pickle_and_rank()
    #_plot_tail_acc('./logs/attempt2/pred_NB_40.pkl')
    #_test_tail_acc()
        