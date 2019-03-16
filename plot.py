import numpy as np
import matplotlib.pyplot as plt

bcetrain = np.load('train_history_BCE.npy')
bceval =  np.load('val_history_BCE.npy')
bcemap = np.load('mAP_history_BCE.npy')

nbtrain = np.load('train_history_NB.npy')
nbval =  np.load('val_history_NB.npy')
nbmap = np.load('mAP_history_NB.npy')

fig, ax = plt.subplots(nrows=1,ncols=2,figsize=(20,20))

ax2 = np.array([a.twinx() for a in ax])

epoch_list = np.arange(1, 11)

ax[0].set_xticks(epoch_list)
ax[0].set_title("BCELoss")
ax[0].set_xlabel("Epoch")
ax[0].set_ylabel("Loss")
ax2[1].set_ylabel("mAP")
ax[0].plot(epoch_list, bcetrain, label="Training loss")
ax[0].plot(epoch_list, bceval, label="Validation loss")
ax2[0].plot(epoch_list, bcemap, label="mAP", c='g')
ax[0].legend(loc='upper right')
ax2[0].legend(loc='lower right')

ax[1].set_xticks(epoch_list)
ax[1].set_title("NBLoss")
ax[1].set_xlabel("Epoch")
ax[1].set_ylabel("Loss")
ax2[1].set_ylabel("mAP")
ax[1].plot(epoch_list, nbtrain, label="Training loss")
ax[1].plot(epoch_list, nbval, label="Validation loss")
ax2[1].plot(epoch_list, nbmap, label="mAP", c='g')
ax[1].legend(loc='upper right')
ax2[1].legend(loc='lower right')

plt.show()