# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 20:38:41 2019

@author: Daniel
"""
import os
import glob
import numpy as np
from PIL import Image
import xml.etree.ElementTree as ET
np.set_printoptions(edgeitems=30, linewidth=100000)

class NB():
    
    def __init__(self, directory, split, smoothing_constant=1, multi_instance=False, verbose=False):
        self.directory = directory
        self.split = split
        self.smoothing_constant = smoothing_constant
        self.multi_instance = multi_instance
        self.verbose = verbose
        self.labels_dict = self.get_labels_dict()
        self.data = self._load_all_image_paths_labels(split)
        self.weight_mat = self._get_weights()
        
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

    def _get_images_list(self, split):
        image_paths = []
        image_path_file = os.path.join(self.directory, 'ImageSets/Main', split + '.txt')
        with open(image_path_file) as f:
            for image_path in f.readlines():
                candidate_path = image_path.split(' ')[0].strip('\n')
                image_paths.append(candidate_path)
        return image_paths

    def _get_xml_file_path(self, image_name):
        xml_name = image_name + '.xml'
        xml_path = os.path.join(self.directory, 'Annotations', xml_name)
        return xml_path

    def _load_all_image_paths_labels(self, split):
        label_count = 0
        all_image_paths_labels = []
        images_list = self._get_images_list(split)
        xml_path_list = [self._get_xml_file_path(image_path)
                        for image_path in images_list]
        for image_path, xml_path in zip(images_list, xml_path_list):
            image_path = os.path.join(self.directory, 'JPEGImages', image_path + '.jpg')
            assert(image_path not in all_image_paths_labels)
            if self.multi_instance:
                labels = self._get_labels_from_xml(xml_path)
            else:
                labels = list(np.unique(self._get_labels_from_xml(xml_path)))
            label_count += len(labels)
            if self.verbose:
                print("Loading labels of size {} for {}...".format(
                    len(labels), image_path))
            image_path_labels = {'image_path': image_path,
                                 'labels': labels}
            all_image_paths_labels.append(image_path_labels)

        print("SET: {} | TOTAL IMAGES: {}".format(self.split, len(all_image_paths_labels)))
        print("SET: {} | TOTAL LABELS: {}".format(self.split, label_count))
        return all_image_paths_labels

    def _get_labels_from_xml(self, xml_path):
        labels = []
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for child in root.iter('object'):
            labels.append(child.find('name').text)
        return labels
    
    def _get_weights(self):
        weight_mat = np.zeros((20,20))
        #print(weight_mat.shape)
        for count, i in enumerate(self.data):
            labels = i['labels']
            relations = {}
            for l0 in labels:
                for l1 in labels:
                    try:
                        relations[(l0,l1)] += 1.0
                    except:
                        relations[(l0,l1)] = 1.0
                    if l0 != l1:
                        try:
                            relations[(l1,l0)] += 1.0
                        except:
                            relations[(l1,l0)] = 1.0
            for key in relations.keys():
                i0, i1 = self.labels_dict[key[0]], self.labels_dict[key[1]]
                weight_mat[i0, i1] += 1#/relations[key]
                #if count < 10:
                #    print(i0,i1, relations[key])
                #    print(weight_mat[i0, i1])
        smoothing = np.full((20,20), self.smoothing_constant)
        weight_mat += smoothing
        weight_mat = weight_mat/(len(self.data)+self.smoothing_constant)
        return weight_mat
            
if __name__ == '__main__':
    nb = NB('VOC2012', 'train', 1, multi_instance=True)
    print(np.round(nb.weight_mat,5))