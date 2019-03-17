import numpy as np

class NaiveBayes:
    def __init__(self, dataset, smoothing_constant):
        self.dataset = dataset
        self.num_classes = len(self.dataset.labels_dict)
        self.smoothing_constant = smoothing_constant
        
    def get_nb_matrix(self):
        weight_mat = np.zeros((self.num_classes, self.num_classes))
        for count, i in enumerate(self.dataset.data):
            labels = i['labels']
            relations = {}
            for current in labels:
                for other in labels:
                    if other is not current:
                        try:
                            relations[(current,other)] += 1
                        except:
                            relations[(current,other)] = 1
            for key in relations.keys():
                i = self.dataset.labels_dict[key[0]]
                j = self.dataset.labels_dict[key[1]]
                weight_mat[i,j] += 1
        smoothing = np.full((self.num_classes, self.num_classes),
            self.smoothing_constant)
        dividend = weight_mat + smoothing
        class_probabilties = np.sum(weight_mat, axis=0)
        divisor = class_probabilties + smoothing*len(self.dataset.labels_dict)
        nb_mat = dividend / divisor
        return nb_mat
