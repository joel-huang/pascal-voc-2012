import os
import glob
import numpy as np
import xml.etree.ElementTree as ET

import torch
from torch.utils.data import Dataset

image_directory = 'VOC2012/JPEGImages'
label_directory = 'VOC2012/Annotations'

class VOCDataset(Dataset):
    def __init__(self, image_directory, label_directory, verbose=False):
        self.verbose = verbose
        self.image_directory = image_directory
        self.label_directory = label_directory
        self.labels_dict = self.get_labels_dict()
        self.data = self._load_all_image_paths_labels(self.label_directory)
        print(self._count_classes())

    def __len__(self):
        return len(data)

    def __getitem__(self, idx):
        pass
        # how to do multilabel balancing?

    def plot_classes(self):
        import matplotlib.pyplot as plt
        count_dict = self._count_classes()
        x = count_dict.values()
        y = count_dict.keys()
        plt.figure(figsize=(20,20))
        plt.bar(y,x)
        plt.show()

    def _count_classes(self):
        count_dict = {x: 0 for x in self.get_labels_dict()}
        for pairs in self.data:
            for label_list in pairs.values():
                for label in np.unique(label_list):
                    count_dict[label] += 1
        return count_dict

    def _get_label_path(self, image_path):
        label_title = image_path.split('JPEGImages/')[-1].strip('.jpg') + '.xml'
        label_path = os.path.join(self.label_directory, label_title)
        return label_path

    def _load_all_image_paths_labels(self, label_directory):
        all_image_paths_labels = []
        images_list = glob.glob(os.path.join(image_directory, '*'))
        xml_path_list = [self._get_label_path(image_path)
                        for image_path in images_list]
        for image_path, xml_path in zip(images_list, xml_path_list):
            labels = self._get_labels_from_xml(xml_path)
            if self.verbose:
                print("Loading labels of size {} for {}...".format(
                    len(labels),image_path))
            image_path_labels = {image_path: labels}
            all_image_paths_labels.append(image_path_labels)
        return all_image_paths_labels

    def _get_labels_from_xml(self, xml_path):
        labels = []
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for child in root.iter('object'):
            labels.append(child.find('name').text)
        return labels

    def get_labels_dict(self):
        return {
            'aeroplane' :    0,
            'bicycle' :      1,
            'bird' :         2,
            'boat' :         3,
            'bottle' :       4,
            'bus' :          5,
            'car' :          6,
            'cat' :          7,
            'chair' :        8,
            'cow' :          9,
            'diningtable' :  10,
            'dog' :          11,
            'horse' :        12,
            'motorbike' :    13,
            'person' :       14,
            'pottedplant' :  15,
            'sheep' :        16,
            'sofa' :         17,
            'train' :        18,
            'tvmonitor' :    19
        }

v = VOCDataset(image_directory, label_directory, verbose=True)