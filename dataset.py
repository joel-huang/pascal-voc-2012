import os
import glob
import numpy as np
from PIL import Image
import xml.etree.ElementTree as ET

import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

directory = 'VOC2012'

class VOCDataset(Dataset):
    def __init__(self, directory, split, transforms=None, multi_instance=False, verbose=False):
        self.split = split
        self.verbose = verbose
        self.directory = directory
        self.transforms = transforms
        self.multi_instance = multi_instance
        self.labels_dict = self.get_labels_dict()
        self.data = self._load_all_image_paths_labels(split)
        self._count_classes()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self._load_image(self.data[idx]['image_path'])
        if self.transforms is not None:
            image = self.transforms(image)
        labels = self.data[idx]['labels']
        return (image, labels)

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
            for label_list in pairs['labels']:
                for label in np.unique(label_list):
                    count_dict[label] += 1
        print(count_dict)

    def _load_image(self, image_path):
        img = Image.open(image_path)
        assert(img.mode == 'RGB')
        return img        

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

class VOCBatch:
    def __init__(self, data):
        transposed_data = list(zip(*data))
        self.image = torch.stack(transposed_data[0], 0)
        self.labels = transposed_data[1]
    def pin_memory(self):
        self.image = self.image.pin_memory()
        self.labels = self.labels.pin_memory()
        return self

def collate_wrapper(batch):
    return VOCBatch(batch)

tr = transforms.Compose([transforms.CenterCrop(224), transforms.ToTensor()])
train = VOCDataset(directory, 'train', transforms=tr, multi_instance=True)
train_loader = DataLoader(train, batch_size=16, collate_fn=collate_wrapper, shuffle=True, num_workers=4)

for _, batch in enumerate(train_loader):
    print(batch.image.shape, batch.labels)