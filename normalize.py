from dataset import VOCDataset, collate_wrapper
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

tr = transforms.Compose([transforms.RandomResizedCrop(300), transforms.ToTensor()])
dataset = VOCDataset('VOC2012', 'train', transforms=tr)
loader = DataLoader(dataset, batch_size=48, collate_fn=collate_wrapper, shuffle=False, num_workers=16)

mean = 0.
std = 0.
nb_samples = 0.

for _, batch in enumerate(loader):
    data = batch.image
    data = data.view(data.size(0), data.size(1), -1)
    mean += data.mean(2).sum(0)
    std += data.std(2).sum(0)
    nb_samples += data.size(0)

mean /= nb_samples
std /= nb_samples

print(mean,std)
