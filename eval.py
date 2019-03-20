import sys
import metrics
from metrics import *

if len(sys.argv) > 1:
    args = sys.argv[1:]
    name = args[0]
    pkl = load_pickle(name)
    pred, target = get_preds_and_gt(pkl)
    print(get_mAP(pred, target))