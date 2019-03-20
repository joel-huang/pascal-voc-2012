import pickle
import numpy as np
from sklearn.metrics import average_precision_score

def get_AP(pred, target):
    return average_precision_score(target.cpu(), pred.cpu())

def get_AP_from_pkl(name):
    with open(name, 'rb') as f:
        x = pickle.load(f)
    f.close()
    pred, target = x[0], x[1]
    return get_AP(pred, target)

def write_AP_from_pkl(start, stop):
    APs = []
    for i in range(start, stop+1):
        name = 'pred_BCE_' + str(i) + '.pkl'
        AP = get_AP_from_pkl(name)
        print('Saving AP{}={}...'.format(str(i), AP))
        APs.append(AP)
    APs = np.array(APs)
    np.save('AP_history_{}_{}.npy'.format(start, stop), APs)
