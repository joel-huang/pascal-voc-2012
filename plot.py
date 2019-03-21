import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from matplotlib import rc

rc('mathtext', default='regular')
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (10, 10),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)

many_train = np.load('logs/attempt8/train_history_BCE_lr0.005_sc0.001_model_BCE_50_0.0536.npy')
many_val = np.load('logs/attempt8/val_history_BCE_lr0.005_sc0.001_model_BCE_50_0.0536.npy')
many_ap = np.load('logs/attempt8/AP_history_1_50.npy')

flip_rot_train = np.load('logs/attempt5/train_history_BCE_lr0.005_sc0.001_model_BCE_50_0.0470.npy')
flip_rot_val = np.load('logs/attempt5/val_history_BCE_lr0.005_sc0.001_model_BCE_50_0.0470.npy')
flip_rot_ap = np.load('logs/attempt5/AP_history_1_50.npy')

flip_train = np.load('logs/attempt6/train_history_BCE_lr0.005_sc0.001_model_BCE_50_0.0434.npy')
flip_val = np.load('logs/attempt6/val_history_BCE_lr0.005_sc0.001_model_BCE_50_0.0434.npy')
flip_ap = np.load('logs/attempt6/AP_history_1_50.npy')

none_train = np.load('logs/attempt7/train_history_BCE_lr0.005_sc0.001_model_BCE_50_0.0437.npy')
none_val = np.load('logs/attempt7/val_history_BCE_lr0.005_sc0.001_model_BCE_50_0.0437.npy')
none_ap = np.load('logs/attempt7/AP_history_1_50.npy')

fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(25,15))
ax2=ax.twinx()

epoch_list = np.arange(1, 51)
xticks = np.arange(0, 51, 10)

ax.tick_params(labelsize=30)
ax2.tick_params(labelsize=30)
ax.set_xticks(xticks)
ax.set_title("BCELoss and Average Precision", fontsize=30)
ax.set_xlabel("Epoch", fontsize=30)
ax.set_ylabel("Loss", fontsize=30)
ax.plot(epoch_list, flip_rot_train, label="Training loss", linewidth=4, color='red', linestyle=':')
ax.plot(epoch_list, flip_rot_val, label="Validation loss", linewidth=4, color='red', linestyle='--')
ax2.plot(epoch_list, flip_rot_ap, label="Average Precision", linewidth=4, color='red', linestyle='-')
ax.plot(epoch_list, flip_train, linewidth=4, color='blue', linestyle=':')
ax.plot(epoch_list, flip_val, linewidth=4, color='blue', linestyle='--')
ax2.plot(epoch_list, flip_ap, linewidth=4, color='blue', linestyle='-')
ax.plot(epoch_list, none_train, linewidth=4, color='green', linestyle=':')
ax.plot(epoch_list, none_val, linewidth=4, color='green', linestyle='--')
ax2.plot(epoch_list, none_ap, linewidth=4, color='green', linestyle='-')
ax.plot(epoch_list, many_train, linewidth=4, color='purple', linestyle=':')
ax.plot(epoch_list, many_val, linewidth=4, color='purple', linestyle='--')
ax2.plot(epoch_list, many_ap, linewidth=4, color='purple', linestyle='-')
ax.legend(loc='center', bbox_to_anchor=(0.8, 0.3), fontsize=30)
ax2.legend(loc='center', bbox_to_anchor=(0.8, 0.6), fontsize=30)

plt.savefig('train_val_ap.png', format='png', bbox_inches='tight')
