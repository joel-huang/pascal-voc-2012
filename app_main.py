import sys, os
import numpy as np
import scipy as sp
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from torchvision import transforms
import torchvision.models as models
from sklearn.metrics import average_precision_score


from tkinter import *
from tkinter import font
from tkinter import filedialog
from tkinter.ttk import *
from PIL import Image, ImageTk
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

class Backend(object):
    def __init__(self, imgs_dir='VOC2012/JPEGImages/', pred_dir='logs/attempt7/', models_dir='logs/attempt7/'):
        self.imgs_dir = imgs_dir
        self.pred_dir = pred_dir
        self.models_dir = models_dir
        self.model = self._load_default_model()
        self.img_paths = self._load_img_paths()
        self.val_order = self._load_val_order()
        self.labels_dict = self.get_labels_dict()
        self.predict_img_counter = 0
        
    def get_labels_dict(self):
        return {
            0: 'aeroplane' ,
            1: 'bicycle' ,
            2: 'bird' ,
            3: 'boat' ,
            4: 'bottle' ,
            5: 'bus' ,
            6: 'car' ,
            7: 'cat' ,
            8: 'chair' ,
            9: 'cow' ,
            10: 'diningtable' ,
            11: 'dog' ,
            12: 'horse' ,
            13: 'motorbike' ,
            14: 'person' ,
            15: 'pottedplant' ,
            16: 'sheep' ,
            17: 'sofa' ,
            18: 'train' ,
            19: 'tvmonitor'
        }
        
    def _load_default_model(self):
        return self.load_model('stop_lr0.005_sc0.001_model_BCE_50_0.0437')
    
    def _load_val_order(self):
        return np.load('logs/val_order.npy')
        
    def _load_img_paths(self):    
        image_paths = []
        image_path_file = 'VOC2012/ImageSets/Main/val.txt' 
        with open(image_path_file) as f:
            for image_path in f.readlines():
                candidate_path = image_path.split(' ')[0].strip('\n')
                image_paths.append(candidate_path)
        return image_paths
    
    def load_model(self, model_name):
        model = models.resnet34(pretrained=True)
        model.fc = nn.Linear(512, 20)
        model.load_state_dict(torch.load(self.models_dir + model_name + '.pt'))
        model = CustomNet(model)
        return model

    def load_img(self, idx):
        if type(idx) != int:
            idx = int(idx[0])
        print(idx, self.predict_img_counter)
        return self.imgs_dir + self.img_paths[idx] + '.jpg'
    
    def pred_from_path(self, file_path):
        img = Image.open(file_path)
        scores = self.get_prediction(img)
        output = []
        for i, score in enumerate(scores):
            if score > 0.5:
                output.append(self.labels_dict[i])
        return scores.cpu().numpy(), output

                
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
        
class App(Tk):
    def __init__(self):
        super().__init__()

        default_font = font.nametofont("TkDefaultFont")
        default_font.configure(size=10)

        self.title("Deep Learning Mini Project")
        self.geometry("1024x576")
        self.tabs = []        
        
        self.notebook = Notebook(self)

        self.notebook.pack(fill=BOTH, expand=1)
        self.backend = Backend()
        #self.img_names_file = os.path.join(
        #        'data/VOCdevkit/VOC2012/ImageSets/Main/val.txt')

    def add_new_tab(self, tab, name):
        self.tabs.append(tab)
        self.notebook.add(tab, text=name)
             
class PredictTab(Frame):
    
    def __init__(self,app):
        super().__init__(app)
        
        self.classes = app.backend.labels_dict
        self.backend = app.backend
        centre = Frame(self)
        centre.pack()
        centre.classes = self.classes
        
        image_frame = Frame(centre)
        image_frame.config(border = 5, relief = RAISED)
        image_frame.pack(side=LEFT,fill=BOTH,padx=20,pady=20)
        
        self.image_holder = Label(image_frame)
        
        self.choose_img_button = Button(image_frame, 
                                        text="Choose Image",
                                        command=self.predict_image)
        self.choose_img_button.pack(side=BOTTOM,padx=20,pady=20)
                
        self.results_frame = ResultsFrame(centre)
        self.results_frame.config(border = 5, relief = RAISED)
        
        device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device_name)
        self.model = app.backend.model
        self.model.to(self.device)
        
        self.transform = transforms.Compose([transforms.Resize(224),
                                transforms.CenterCrop(224),
                                transforms.ToTensor()])
        
    def get_image(self):
        
        path = filedialog.askopenfilename(filetypes=[("Image File",'*')])
        
        image_name = path[path.rfind('/')+1:]
        
        if path != '':
            img = Image.open(path).convert('RGB')
        else:
            img = None
            
        return img, image_name

    def display_image(self, img):
        img_w,img_h = img.size
        limit = 400
        if img_w >= img_h:
            resized_image = img.resize((limit,int(img_h*limit/img_w)),
                                             Image.ANTIALIAS)
        else:
            resized_image = img.resize((int(img_w*limit/img_h),limit),
                                             Image.ANTIALIAS)
        tkimage = ImageTk.PhotoImage(resized_image)
        self.image_holder.image = tkimage
        self.image_holder.config(image=tkimage)

    def get_pred(self, img):
        
        self.model.eval()
        threshold = 0.5
        
        with torch.no_grad():
            data = self.transform(img).to(self.device)
            scores = torch.sigmoid(self.model(data.unsqueeze(0))).squeeze(0).tolist()
            prediction = []
            for i in range(len(scores)):
                if scores[i] >= threshold:
                    prediction.append(self.classes[i])
                    
        return prediction, scores
    
    def predict_image(self):
        img = self.get_image()
        if img[0] != None:
            self.display_image(img[0])
            self.results_frame.loading(img[1])
            
            prediction, scores = self.get_pred(img[0])
            
            self.image_holder.pack(side=TOP,padx=20,pady=20)
            self.results_frame.pack(side=RIGHT,fill=BOTH,padx=20,pady=20)
            self.results_frame.display_results(prediction, scores)
        
class ResultsFrame(Frame):
    def __init__(self,predict_tab):
        super().__init__(predict_tab)
        
        self.classes = predict_tab.classes
        
        centre_frame = Frame(self)
        centre_frame.pack(fill=BOTH)
        
        pred_title = Label(centre_frame,
                           text = 'Prediction Results')
        pred_title.grid(row = 0, column = 0, padx = 10, pady = 10)
        
        self.image_name = Label(centre_frame,
                            text = '(No image selected)')
        self.image_name.grid(row = 1, column = 0, padx = 10, pady = 10)
        
        self.pred_result = Label(centre_frame,
                            text = '(No image selected)')
        self.pred_result.grid(row = 2, column = 0, padx = 10, pady = 10)
        
        score_grid = Frame(centre_frame)
        score_grid.grid(row = 3, column = 0, padx = 10, pady = 10)
        
        self.score_list = []
        num_cols = 4
        for i in range(len(self.classes)):
            
            class_score_frame = Frame(score_grid)
            class_score_frame.config(border = 5, relief = SUNKEN)
            class_score_frame.grid(row = int(i/num_cols),
                                   column = int(i%num_cols),
                                   sticky=N+S+E+W)
            
            class_name = Label(class_score_frame,
                               text = self.classes[i])
            class_name.pack(side = TOP)
            
            self.score_list.append(Label(class_score_frame))
            self.score_list[i].pack(side = TOP)
            
    def loading(self, image_name):
        self.pred_result.config(text = 'Loading...')
        for score in self.score_list:
            score.config(text = '...')
        
        if len(image_name) <= 50:
            self.image_name.config(text = image_name)
        else:
            name = image_name.split('/')[-1]
            truncated_name = ' ... '.join(name)#.join([image_name[:39], image_name[-7:]])
            self.image_name.config(text = truncated_name)
                   
    def display_results(self, prediction, scores):
        
        if len(prediction) == 0:
            prediction_text = 'Prediction: None'
        else:
            prediction_text = 'Prediction: ' + ', '.join(prediction)
        self.pred_result.config(text = prediction_text)
        
        for i in range(len(self.score_list)):
            self.score_list[i].config(text = str(round(scores[i],5)))    
 
class BrowseTab(Frame):
    
    def __init__(self, app):
        super().__init__(app)
        
        self.backend = app.backend
        self.img_paths = self.backend.img_paths
        self.classes = self.backend.labels_dict
        
        self.browse_frame = ScrollFrame(self)
        self.browse_frame.pack(side=BOTTOM, fill=BOTH, expand=1)
        
        self.cls_select_frame = ClassSelectFrame(self)
        self.cls_select_frame.pack(side=TOP, fill=X)
        
    def class_select(self, cls):
        self.browse_frame.change_class(cls)
        self.cls_select_frame.change_page(self.browse_frame.page_number)
        
    def prev_page(self):
        if self.browse_frame.page_number > 1:
            self.browse_frame.prev_page()
            self.cls_select_frame.change_page(self.browse_frame.page_number)
        else:
            self.cls_select_frame.change_page(self.browse_frame.page_limit)
        
    def next_page(self):
        if self.browse_frame.page_number < self.browse_frame.page_limit:
            self.browse_frame.next_page()
            self.cls_select_frame.change_page(self.browse_frame.page_number)
        else:
            self.cls_select_frame.change_page(1)
    
class ClassSelectFrame(Frame):
    
    def __init__(self, browse_tab):
        super().__init__(browse_tab)
        
        self.classes = browse_tab.classes
        self.img_dir = browse_tab.backend.imgs_dir
        
        self.page_nav = Frame(self)
        self.page_nav.pack(side=TOP,pady=10)
        
        self.prev_page_button = Button(self.page_nav,
                                command = lambda:browse_tab.prev_page(),
                                text = '<')
        self.prev_page_button.grid(row = 0, column = 0, padx = 10)
        
        page = browse_tab.browse_frame.page_number
        self.num_images = len(browse_tab.browse_frame.image_names)
        self.page_size = browse_tab.browse_frame.page_size
        self.image_range = Label(self.page_nav,
                                 text = 'Page {} of {} - Images {} to {}'
                                 .format(page,
                                    int(self.num_images/self.page_size)+1,
                                    self.page_size*(page-1)+1,
                                    min(self.page_size*page,self.num_images)))
        self.image_range.grid(row = 0, column = 1, padx = 10)
        
        self.next_page_button = Button(self.page_nav,
                                command = lambda:browse_tab.next_page(),
                                text = '>')
        self.next_page_button.grid(row = 0, column = 2, padx = 10)
        
        self.sorted_by = Label(self,
                               text = 'Now viewing: '+self.classes[browse_tab.browse_frame.cls])
        self.sorted_by.pack(side=TOP,pady=10)
        
        self.cls_nav = Frame(self)
        self.cls_nav.pack(side=BOTTOM)
        
        cls_rows = 2
        cls_cols = len(self.classes)/cls_rows
        for i in range(len(self.classes)):
                        
            class_button = Button(self.cls_nav,
                             command = lambda idx = i: browse_tab.class_select(idx),
                             text = self.classes[i])

            class_button.grid(
                    row=int(i/cls_cols),
                    column=int(i%cls_cols))

    def change_page(self, page):
        self.image_range.config(text = 'Page {} of {} - Images {} to {}'
                                .format(page,
                                   int(self.num_images/self.page_size)+1,
                                   self.page_size*(page-1)+1,
                                   min(self.page_size*page,self.num_images)))
        self.sorted_by.config(text = 'Sorted by: '+self.classes[browse_tab.browse_frame.cls])
    
class ImageFrame(Frame):
    
    def __init__(self,frame):
        super().__init__(frame)
        
        self.img_dir = 'VOC2012/JPEGImages/'
        self.img_names_file = frame.img_names_file
        
        self.text_holder = Frame(self)
        self.text_holder.grid(row=0,
                              column=1,
                              sticky = N+S+E+W,
                              padx = 20)
        
        self.rank = Label(self.text_holder)
        self.rank.config(anchor=W)
        self.rank.grid(row=0,column=0,sticky=E+W,padx=5,pady=5)
        
        self.title = Label(self.text_holder)
        self.title.config(anchor=W)
        self.title.grid(row=1,column=0,sticky=E+W,padx=5,pady=5)
        
        self.img = Label(self)
        self.img.grid(row=0,column=0,sticky = N+S+E+W)
        
        self.score = Label(self.text_holder)
        self.score.config(anchor=W)
        self.score.grid(row=2,column=0,sticky=E+W,padx=5,pady=5)
        
        self.prediction = Label(self.text_holder)
        self.prediction.config(anchor=W)
        self.prediction.grid(row=3,column=0,sticky=E+W,padx=5,pady=5)
        
    def change(self,idx,title,score,prediction):
        
        self.rank.config(text = 'Rank '+str(idx+1))
        self.title.config(text = 'Image: '+title)
        
        img_path = os.path.join(self.img_dir,
                                    title+'.jpg')
        pil_image = Image.open(img_path)
        img_w,img_h = pil_image.size
        limit = 360
        if img_w >= img_h:
            resized_image = pil_image.resize((limit,int(img_h*limit/img_w)),
                                             Image.ANTIALIAS)
        else:
            resized_image = pil_image.resize((int(img_w*limit/img_h),limit),
                                             Image.ANTIALIAS)
        self.img.image = ImageTk.PhotoImage(resized_image)
        self.img.config(image = self.img.image)
        
        self.score.config(text = 'Score: ' + str(score))
        
        self.prediction.config(text = 'Prediction: ' + ', '.join(prediction))
        
class ScrollFrame(Frame):
    def __init__(self, parent):
        super().__init__(parent) # create a frame (self)

        self.backend = parent.backend
        self.img_names_file = parent.backend.img_paths
        self.image_names = parent.backend.img_paths
            
        self.classes = parent.classes
        self.cls = 0
        
        device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device_name)
        self.val_scores, self.gt = load_scores_and_gt()
        self.sorted_preds, self.sorted_img_idxs = [], []
        for i in range(20):
            preds, img_idxs = rank_by_class(self.val_scores, i)
            self.sorted_preds.append(preds.tolist())
            self.sorted_img_idxs.append(img_idxs.tolist())
        self.sorted_preds = np.array(self.sorted_preds)
        self.sorted_img_idxs = np.array(self.sorted_img_idxs)
        
        self.canvas = Canvas(self, borderwidth=0, background="#ffffff")

        self.viewPort = Frame(self.canvas)
        self.viewPort.img_names_file = self.img_names_file
        self.viewPort.pack(fill=BOTH,expand=True)
        
        self.page_number = 1
        
        self.page_size = 20
        self.page_limit = int(len(self.image_names)/self.page_size)+1
        
        self.cols = 5
        
        self.image_list = []
        
        for i in range(self.page_size):
            
            self.image_list.append(ImageFrame(self.viewPort))
            self.image_list[i].config(border=10,relief=SUNKEN)
            self.image_list[i].pack(side=TOP,fill=BOTH,expand=True)
            
        self.goto_page(self.page_number)
            
        self.vsb = Scrollbar(self, orient="vertical", command=self.canvas.yview) #place a scrollbar on self 
        self.canvas.configure(yscrollcommand=self.vsb.set)                          #attach scrollbar action to scroll of canvas

        self.vsb.pack(side="right", fill="y")                                       #pack scrollbar to right of self
        self.canvas.pack(side=LEFT, fill=BOTH, expand=True)                     #pack canvas to left of self and expand to fil
        self.canvas.create_window((4,4), window=self.viewPort, anchor="nw",            #add view port frame to canvas
                                  tags="self.viewPort")

        self.viewPort.bind("<Configure>", self.on_frame_config)                       

    def on_frame_config(self, event):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))                
        
    def goto_page(self, pg):
        last_cell = min(len(self.image_list),len(self.image_names)-(pg-1)*self.page_size)
        for i in range(len(self.image_list)):
            if i < last_cell:
                img_idx = (pg-1)*self.page_size+i
                true_img_idx = self.sorted_img_idxs[self.cls,img_idx]
                print(true_img_idx, self.backend.val_order[true_img_idx][0])
                title = self.image_names[int(self.backend.val_order[true_img_idx][0])]
                score = self.sorted_preds[self.cls,true_img_idx,self.cls]
                print(score)
                preds = self.sorted_preds[self.cls,true_img_idx,:]
                prediction = []
                for k in range(20):
                    if preds[k] > 0.5:
                        prediction.append(self.classes[k])
                self.image_list[i].change(img_idx,title,score,prediction)
                self.image_list[i].pack()
            else:
                self.image_list[i].grid_remove()
        
    def next_page(self):
        self.page_number += 1
        self.goto_page(self.page_number)
        
    def prev_page(self):
        self.page_number -= 1
        self.goto_page(self.page_number)
        
    def change_class(self,cls):
        self.page_number = 1
        self.cls = cls
        self.goto_page(self.page_number)
    
def load_pickle(file_path='logs/attempt1/pred_BCE_40.pkl'):
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

def load_scores_and_gt(file_path='logs/attempt1/pred_BCE_40.pkl'):
    return get_preds_and_gt(load_pickle(file_path))
    
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
    t_vals = np.linspace(t_min, t_max, t_num, endpoint=False)
    tail_accs = []
    for t_val in t_vals:
        tp = 0
        fp = 0
        start = True
        for (score, target) in zip(sorted_scores, sorted_gt):
            pred = 0
            if score > t_val:
                pred = 1
                if pred == target:
                    tp += 1
                else:
                    fp += 1
        tail_acc = 0
        if tp+fp > 0:
            tail_acc = tp/(tp+fp)
        #print(t_val, tp, fp)
        tail_accs.append(tail_acc)
        
    return tail_accs
    
def get_mAP(scores, gt):
    mAP = 0
    for i in range(scores.shape[0]):
        pred = np.array([1 if scores[i,j] > 0.5 else 0 for j in range(scores.shape[1])])
        AP = average_precision_score(pred, gt[i])
        mAP += AP
    mAP /= scores.shape[0]
    return mAP
    
def _test_tail_acc():
    pickled = load_pickle()
    scores, gt = get_preds_and_gt(pickled, 'numpy')
    sorted_scores, idx_order = rank_by_class(scores, 0)
    t_max = np.max(scores)
    print(t_max)
    tail_accs = get_tail_acc(sorted_scores[:,0], gt[idx_order][:,0], 0.5, t_max, 10)
    print(tail_accs)
                
def get_tail_acc_classwise(scores, gt, t_min=0.5, t_num=10):
    t_max = np.max(scores)
    tail_accs = []
    t_vals = np.linspace(t_min, t_max, t_num, endpoint=False)
    for i in range(20):
        sorted_scores, idx_order = rank_by_class(scores, i)
        tail_accs_i = get_tail_acc(sorted_scores[:50,i], gt[idx_order][:50,i], t_min, t_max, t_num)
        tail_accs.append(tail_accs_i)
    return tail_accs, t_vals
    
def _plot_tail_acc(file_path):
    pickled = load_pickle(file_path)
    scores, gt = get_preds_and_gt(pickled, 'numpy')
    tail_accs, t_vals = get_tail_acc_classwise(scores, gt, 0.5, 10)
    tail_accs = np.array(tail_accs)
    ave_tail_accs = np.mean(tail_accs, axis=0)
    plt.figure()
    plt.plot(t_vals, ave_tail_accs, 'b', t_vals, ave_tail_accs, 'ro')
    plt.xticks(t_vals)
    plt.title('Average of classwise tailaccs over top-50 ranked images in each class')
    plt.ylabel('Tailaccs')
    plt.xlabel('t value')
    plt.show()
    
if __name__ == "__main__":
    
    app = App()
    predict_tab = PredictTab(app)
    app.add_new_tab(predict_tab,"Predict on a Single Image")
    browse_tab = BrowseTab(app)
    app.add_new_tab(browse_tab,"Browse Top-ranked Predictions")
    
    app.mainloop()

    
"""    
class MyFlask(Flask):
    def __init__(self, name, **kwargs):
        super().__init__(name, static_folder=kwargs['static_folder'], static_url_path=kwargs['static_url_path'])
        self.predict_img_counter = 0

app = Flask(__name__, static_folder='.', static_url_path='')
backend = Backend()
predict_img_counter = 0
max_img_counter = backend.val_order.shape[0]

@app.route('/')
def home():
    return render_template('home.html')
    
@app.route('/', methods=['GET'])
def home_buttons():
    btn_val = request.form['btn']
    if btn_val == 'Get prediction for single image':
        file_path = backend.load_img(0)
        return render_template('predict.html', img_src=file_path, img_num=0)
    else:
        return render_template('browse.html')
  
@app.route('/predict', methods=['GET'])
def predict():
    btn_val = request.form['btn']
    if btn_val == 'Next':
        app.predict_img_counter += 1
        app.predict_img_counter %= max_img_counter
        file_path = backend.load_img(app.predict_img_counter)
        return render_template('predict.html', img_src=file_path)
    elif btn_val == 'Previous':
        if app.predict_img_counter == 0:
            app.predict_img_counter -= 1
        else:
            app.predict_img_counter = max_img_counter-1
        file_path = backend.load_img(app.predict_img_counter)
        return render_template('predict.html', img_src=file_path)
    elif btn_val == 'Submit':
        file_path = request.form['file_path']
        output = backend.pred_from_path(file_path)
        return render_template('predict_output.html', scores=output[0], pred=output[1])
        
    
@app.route('/browse')
def browse():
    return render_template('browse.html')
    
if __name__ == '__main__':
    app.run(debug=True, use_reloader=True)
    
    #_plot_tail_acc('logs/attempt1/pred_BCE_40.pkl')
    #_plot_tail_acc('logs/attempt2/pred_NB_40.pkl')
"""        