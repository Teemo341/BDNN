import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image



def getSVHN(batch_size, test_batch_size, img_size, **kwargs):
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    print("Building SVHN data loader with {} workers".format(num_workers))

    def target_transform(target):
        new_target = target - 1
        if new_target == -1:
            new_target = 9
        return new_target

    ds = []

    train_loader = DataLoader(
        datasets.SVHN(
            root='../data/svhn', split='train', download=True,
            transform=transforms.Compose([
                transforms.Resize(img_size),
                transforms.ToTensor(),
            ]),
            target_transform=target_transform,
        ),
        batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)
    ds.append(train_loader)

    test_loader = DataLoader(
        datasets.SVHN(
            root='../data/svhn', split='test', download=True,
            transform=transforms.Compose([
                transforms.Resize(img_size),
                transforms.ToTensor(),
            ]),
            target_transform=target_transform
        ),
        batch_size=batch_size, shuffle=False, drop_last=True, **kwargs)
    ds.append(test_loader)
    return ds

def getCIFAR10(batch_size, test_batch_size, img_size, **kwargs):
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    print("Building CIFAR-10 data loader with {} workers".format(num_workers))
    ds = []
    train_loader = DataLoader(
        datasets.CIFAR10(
            root='../data/cifar10', train=True, download=True,
            transform=transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.Resize(img_size),
                transforms.ToTensor(),
            ])),
        batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)
    ds.append(train_loader)
    test_loader = DataLoader(
        datasets.CIFAR10(
            root='../data/cifar10', train=False, download=True,
            transform=transforms.Compose([
                transforms.Resize(img_size),
                transforms.ToTensor(),
            ])),
        batch_size=batch_size, shuffle=False, drop_last=True, **kwargs)
    ds.append(test_loader)

    return ds




def getCIFAR100(batch_size, test_batch_size, img_size, **kwargs):
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    print("Building CIFAR-100 data loader with {} workers".format(num_workers))
    ds = []
    train_loader = DataLoader(
        datasets.CIFAR100(
            root='../data/cifar100', train=True, download=True,
            transform=transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.Resize(img_size),
                transforms.ToTensor(),
            ])),
        batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)
    ds.append(train_loader)
    test_loader = DataLoader(
        datasets.CIFAR100(
            root='../data/cifar100', train=False, download=True,
            transform=transforms.Compose([
                transforms.Resize(img_size),
                transforms.ToTensor(),
            ])),
        batch_size=batch_size, shuffle=False, drop_last=True, **kwargs)
    ds.append(test_loader)

    return ds

class tiny_imagenet_train(Dataset):
    def __init__(self, root, transform=None):
        self.root = root+'/train'
        self.transform = transform
        self.classes = os.listdir(root+'/train')
        self.classes.sort()
        self.class_to_idx = {self.classes[i]: i for i in range(len(self.classes))}
        self.images = []
        self.targets = []
        for i, cls in enumerate(self.classes):
            img_dir = os.path.join(root, 'train', cls, 'images')
            for img_name in os.listdir(img_dir):
                img_path = os.path.join(img_dir, img_name)
                self.images.append(img_path)
                self.targets.append(i)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        target = self.targets[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, target
    
class tiny_imagenet_val(Dataset):
    def __init__(self, root, transform=None):
        self.root = root+'/val'
        self.transform = transform
        self.classes = os.listdir(root+'/train')
        self.classes.sort()
        self.class_to_idx = {self.classes[i]: i for i in range(len(self.classes))}
        self.images = []
        self.targets = []
        with open(os.path.join(root, 'val', 'val_annotations.txt')) as f:
            for line in f:
                img_name, cls = line.split('\t')[:2]
                img_path = os.path.join(root, 'val', 'images', img_name)
                self.images.append(img_path)
                self.targets.append(self.class_to_idx[cls])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        target = self.targets[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, target
    

def getImagenetTiny(batch_size, test_batch_size, img_size, **kwargs):
    num_workers = kwargs.setdefault('num_workers', 4)
    kwargs.pop('input_size', None)
    print("Building Tiny ImageNet data loader with {} workers".format(num_workers))
    ds = []
    train_loader = DataLoader(
        tiny_imagenet_train(
            root='../data/tiny-imagenet-200/',
            transform=transforms.Compose([
                transforms.RandomCrop(64, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.Resize(img_size),
                transforms.ToTensor(),
            ])),
        batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)
    ds.append(train_loader)
    test_loader = DataLoader(
        tiny_imagenet_val(
            root='../data/tiny-imagenet-200/',
            transform=transforms.Compose([
                transforms.Resize(img_size),
                transforms.ToTensor(),
            ])),
        batch_size=batch_size, shuffle=False, drop_last=True, **kwargs)
    ds.append(test_loader)

    return ds







def getDataSet(data_type, batch_size,test_batch_size, imageSize):
    if data_type == 'cifar10':
        train_loader, test_loader = getCIFAR10(batch_size, test_batch_size, imageSize)
    elif data_type == 'svhn':
        train_loader, test_loader = getSVHN(batch_size, test_batch_size, imageSize)
    elif data_type == 'cifar100':
        train_loader, test_loader = getCIFAR100(batch_size, test_batch_size, imageSize)
    elif data_type == 'imagenet':
        train_loader, test_loader = getImagenetTiny(batch_size, test_batch_size, imageSize)
       
    return train_loader, test_loader


if __name__ == '__main__':
    train_loader, test_loader = getDataSet('imagenet', 256, 1000, 28)
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        print(inputs.shape)
        print(targets.shape)