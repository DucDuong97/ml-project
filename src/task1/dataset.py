import torch
import torchvision.transforms as transforms
import os
import errno
import tarfile
from typing import Any, Callable, Dict, IO, List, Optional, Tuple, Union
import warnings
import numpy as np



# Change this to the path where you want to download the dataset to
DEFAULT_ROOT = './data/symbols'

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                                                                         #
#                You may change this file below, but it is not necessary.                 #
#   For a better understanding of data preprocessing though, we recommend reading it.     #
#                                                                                         #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


class StrangeSymbols(torch.utils.data.Dataset):
    urls = [
        'https://seafile.rlp.net/f/b4e03d983fea4e5790b0/?dl=1',
    ]

    def __init__(self, 
                 root: str=DEFAULT_ROOT, 
                 train: bool = True,
                 transform: Optional[Callable] = None):
        self.root = os.path.expanduser(root)
        self.train = train  # training set or test set
        self.transform = transform
        self.download()

        if self.transform == None:
            self.transform = transforms.Compose([
                transforms.Lambda(self._norm),
                transforms.Normalize((0.5, ), (0.5, ))
                ])

        if self.train:
            self.train_data = torch.load(os.path.join(self.root, 'training_data.pt'))
            self.train_labels = torch.load(os.path.join(self.root, 'training_labels.pt'))
        else:
            self.test_data = torch.load(os.path.join(self.root, 'test_data.pt'))

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img = self.test_data[index]
        
        img = img.unsqueeze(0)
        img = img.to(torch.float64)

        if self.transform is not None:
            img = self.transform(img)
        return (img, target) if self.train else (img, float("NaN"))

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, 'training_data.pt')) and \
            os.path.exists(os.path.join(self.root, 'test_data.pt')) and \
            os.path.exists(os.path.join(self.root, 'training_labels.pt'))

    def download(self):
        from six.moves import urllib
        if self._check_exists():
            return
        # download files
        try:
            os.makedirs(self.root)
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        for url in self.urls:
            print('Downloading ' + url)
            data = urllib.request.urlopen(url)
            filename = 'strange_symbols.tar.gz'
            file_path = os.path.join(self.root, filename)
            with open(file_path, 'wb') as f:
                f.write(data.read())
            tarfile.open(file_path, mode='r:gz').extractall(self.root)
            os.unlink(file_path)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        return fmt_str

    @staticmethod
    def _norm(x):
        x = x/255
        return x


def get_strange_symbols_train_loader(batch_size, root=DEFAULT_ROOT, transform=None):
    trainset = StrangeSymbols(root=root, train=True, transform=transform)
    return torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)

def get_strange_symbols_train_data(root=DEFAULT_ROOT, transform=None):
    trainset = StrangeSymbols(root=root, train=True, transform=transform)
    batch_size = len(trainset)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)
    return next(iter(trainloader))

def get_strange_symbols_test_data(root=DEFAULT_ROOT, transform=None):
    testset = StrangeSymbols(root=root, train=False, transform=transform)
    batch_size = len(testset)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)
    return next(iter(testloader))


# if __name__ == '__main__':
#     train_loader = get_strange_symbols_train_loader(batch_size=32)
#     X,y = next(iter(train_loader))
#     print(f'The tensor containing a batch of data points is of shape {X.shape}, and of labels is of shape {y.shape}')

#     train_data, train_labels = get_strange_symbols_train_data()
#     print(train_data.shape, train_labels.shape)

#     test_data, test_labels = get_strange_symbols_test_data()
#     print(test_data.shape, test_labels.shape)

#     print(test_labels)

    