import numpy as np
import matplotlib.pyplot as plt

bcetrain = np.load('train_history_BCE_lr0.001_sc0.001_model_BCE_40_0.1131.npy')
bceval =  np.load('val_history_BCE_lr0.001_sc0.001_model_BCE_40_0.1131.npy')

fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(20,20))

epoch_list = np.arange(1, 41)
xticks = np.arange(0, 40, 10)

ax.set_xticks(xticks)
ax.set_title("BCELoss")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.plot(epoch_list, bcetrain, label="Training loss")
ax.plot(epoch_list, bceval, label="Validation loss")
ax.legend(loc='upper right')

plt.show()