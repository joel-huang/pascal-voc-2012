import torch
import pickle
import numpy as np

import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score

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

def load_scores_and_gt(file_path='./logs/attempt1/pred_BCE_40.pkl'):
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
    t_vals = np.linspace(t_min, t_max, t_num)
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
        AP = average_precision_score(gt[i], pred)
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
    t_vals = np.linspace(t_min, t_max, t_num)
    for i in range(20):
        sorted_scores, idx_order = rank_by_class(scores, i)
        tail_accs_i = get_tail_acc(sorted_scores[:,i], gt[idx_order][:,i], t_min, t_max, t_num)
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
    plt.ylabel('Tailaccs')
    plt.xlabel('t value')
    plt.show()