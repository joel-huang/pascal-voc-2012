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
fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(25,18))
ax2 = ax.twinx()

from dataset import VOCDataset
dataset = VOCDataset('VOC2012', 'train')
keys = list(dataset.classes_count.keys())
values = np.array(list(dataset.classes_count.values()))
minv = np.min(values)
maxv = np.max(values)
newvalues = (minv/values)

ax.tick_params(labelsize=20)
ax2.tick_params(labelsize=20)
ax2.set_ylim(top=maxv)
plt.setp(ax.get_xticklabels(), **{"rotation": 45, "ha": "right"})

ax.set_title("Training set distribution", fontsize=30)
ax.set_xlabel("Classes", fontsize=30)
ax.set_ylabel("Count", fontsize=30)
ax2.set_ylabel("Weights", fontsize=30)

ax.bar(keys, values, color='blue')
ax2.bar(keys, np.multiply(values, newvalues), color='red')
plt.savefig('classes.png', format='png', bbox_inches='tight')

