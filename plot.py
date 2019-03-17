import numpy as np
import matplotlib.pyplot as plt

bcetrain = np.concatenate((np.load('train_history_BCE.npy'), np.load('train_history_BCE_100.npy')), axis=None)
bceval =  np.concatenate((np.load('val_history_BCE.npy'), np.load('val_history_BCE_100.npy')), axis=None)
bcemap = np.concatenate((np.load('mAP_history_BCE.npy'), np.load('mAP_history_BCE_100.npy')), axis=None)

fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(20,20))
ax2 = ax.twinx()

epoch_list = np.arange(1, 101)
xticks = np.arange(0, 100, 10)

ax.set_xticks(xticks)
ax.set_title("BCELoss")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax2.set_ylabel("mAP")
ax.plot(epoch_list, bcetrain, label="Training loss")
ax.plot(epoch_list, bceval, label="Validation loss")
ax2.plot(epoch_list, bcemap, label="mAP", c='g')
ax.legend(loc='upper right')
ax2	.legend(loc='lower right')

plt.show()