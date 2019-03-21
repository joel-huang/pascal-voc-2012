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

none_train = np.load('logs/attempt12/train_history_BCE_lr0.01_sc0.001_model_BCE_50_0.0321.npy')
none_val = np.load('logs/attempt12/val_history_BCE_lr0.01_sc0.001_model_BCE_50_0.0321.npy')
none_ap = np.load('logs/attempt12/AP_history_1_50.npy')

none2_train = np.load('logs/attempt11/train_history_BCE_lr0.01_sc0.001_model_BCE_50_0.0324.npy')
none2_val = np.load('logs/attempt11/val_history_BCE_lr0.01_sc0.001_model_BCE_50_0.0324.npy')
none2_ap = np.load('logs/attempt11/AP_history_1_50.npy')

none3_train = np.load('logs/attempt13/train_history_BCE_lr0.01_sc0.001_model_BCE_50_0.0334.npy')
none3_val = np.load('logs/attempt13/val_history_BCE_lr0.01_sc0.001_model_BCE_50_0.0334.npy')
none3_ap = np.load('logs/attempt13/AP_history_1_50.npy')

fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(25,15))
ax2=ax.twinx()

epoch_list = np.arange(1, 51)
xticks = np.arange(0, 51, 10)

ax.tick_params(labelsize=30)
ax2.tick_params(labelsize=30)
ax.set_xticks(xticks)
ax.set_title("Training phase", fontsize=30)
ax.set_xlabel("Epoch", fontsize=30)
ax.set_ylabel("Loss", fontsize=30)
ax2.set_ylabel("AP", fontsize=30)

ax.plot(epoch_list, none_train, linewidth=4, color='blue', linestyle=':', label="Training loss")
ax.plot(epoch_list, none_val, linewidth=4, color='blue', linestyle='--', label="Validation loss")
ax2.plot(epoch_list, none_ap, linewidth=4, color='blue', linestyle='-', label="Average Precision")

ax.plot(epoch_list, none2_train, linewidth=4, color='green', linestyle=':')
ax.plot(epoch_list, none2_val, linewidth=4, color='green', linestyle='--')
ax2.plot(epoch_list, none2_ap, linewidth=4, color='green', linestyle='-')

ax.plot(epoch_list, none3_train, linewidth=4, color='red', linestyle=':')
ax.plot(epoch_list, none3_val, linewidth=4, color='red', linestyle='--')
ax2.plot(epoch_list, none3_ap, linewidth=4, color='red', linestyle='-')

ax.legend(loc='center', bbox_to_anchor=(0.8, 0.3), fontsize=30)
ax2.legend(loc='center', bbox_to_anchor=(0.8, 0.6), fontsize=30)

plt.savefig('train_flipped_rot.png', format='png', bbox_inches='tight')
