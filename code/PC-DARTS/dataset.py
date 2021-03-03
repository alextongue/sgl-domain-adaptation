import torchvision.datasets as dset
import utils
import torch.utils.data as data
from PIL import Image
import os

# DataLoader for MNIST-M dataset
class MnistmGetLoader(data.Dataset):
    def __init__(self, data_root, data_list, transform=None):
        self.root = data_root
        self.transform = transform

        f = open(data_list, 'r')
        data_list = f.readlines()
        f.close()

        self.n_data = len(data_list)

        self.img_paths = []
        self.img_labels = []

        for data in data_list:
            self.img_paths.append(data[:-3])
            self.img_labels.append(data[-2])

    def __getitem__(self, item):
        img_paths, labels = self.img_paths[item], self.img_labels[item]
        imgs = Image.open(os.path.join(self.root, img_paths)).convert('RGB')

        if self.transform is not None:
            imgs = self.transform(imgs)
            labels = int(labels)

        return imgs, labels

    def __len__(self):
        return self.n_data


def get_data_transforms( args ):
    train_transform, valid_transform = None, None
    if args.set == 'cifar10' or args.set == 'cifar100':
        train_transform, valid_transform = utils._data_transforms_cifar10(args)
    elif args.set == 'mnist':
        train_transform, valid_transform = utils._data_transforms_mnist(args)
    elif args.set == 'mnistm':
        train_transform, valid_transform = utils._data_transforms_mnistm(args)
    else:
        assert False and f'Unrecognized dataset: {args.set}'
    return train_transform, valid_transform

def get_train_dataset( args ):
    train_data = None
    train_transform, _ = get_data_transforms( args )
    if args.set == 'cifar100':
        train_data = dset.CIFAR100(root=args.data, train=True, 
                download=True, transform=train_transform)
    elif args.set == 'cifar10':
        train_data = dset.CIFAR10(root=args.data, train=True, 
                download=True, transform=train_transform)
    elif args.set == 'mnist':
        train_data = dset.MNIST(
            root=args.data,
            train=True,
            transform=train_transform,
            download=True
        )
    elif args.set == 'mnistm':
        mnistm_root = os.path.join( args.data, 'mnist_m' )
        train_list = os.path.join(mnistm_root, 'mnist_m_train_labels.txt')
        train_data = MnistmGetLoader(
            data_root=os.path.join(mnistm_root, 'mnist_m_train'),
            data_list=train_list,
            transform=train_transform
        )
    else:
        assert False and f'Unrecognized dataset: {args.set}'
    return train_data
