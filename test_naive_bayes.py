import numpy as np
import pandas as pd
from dataset import VOCDataset
from naive_bayes import NaiveBayes

# Instantiate the dataset and retrieve the NB matrix.
dataset = VOCDataset('VOC2012', 'train', multi_instance=True)
mat = NaiveBayes(dataset, 1).get_nb_matrix()

# Check if rows sum up to 1. They should; Each element
# in the # row is a conditional probability,
# conditioned on the same class label.
assert(np.allclose(np.sum(mat, axis=0), np.ones(20), atol=1e-8))

# Print results.
cols = ["x={}".format(key) for key in dataset.labels_dict.keys()]
rows = ["P({}|x)".format(key) for key in dataset.labels_dict.keys()]
mat = pd.DataFrame(mat, columns=cols, index=rows).round(5).T
print(mat)
