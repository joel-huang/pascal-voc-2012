from torchvision import transforms
from torch.utils.data import DataLoader
from dataset import VOCDataset, collate_wrapper

directory = 'VOC2012'

tr = transforms.Compose([transforms.CenterCrop(224), transforms.ToTensor()])
train = VOCDataset(directory, 'train', transforms=tr, multi_instance=True)
train_loader = DataLoader(train, batch_size=16, collate_fn=collate_wrapper, shuffle=True, num_workers=4)

"""
How to enumerate across the DataLoader:

for _, batch in enumerate(train_loader):
    batch_of_image_tensors = batch.image
    batch of label_lists = batch.labels
"""