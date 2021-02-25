import numpy as np
from diagan.datasets.color_mnist import ColoredMNIST
from diagan.datasets.gaussian import get_25gaussian_dataset
from diagan.datasets.mnist_fmnist import MNIST_FMNIST
from diagan.datasets.transform import get_transform
from torch.utils.data import Dataset
from torchvision.datasets.celeba import CelebA
from torchvision.datasets.cifar import CIFAR10

DATASET_DICT = {
    'cifar10': CIFAR10,
    'celeba': CelebA,
    'color_mnist': ColoredMNIST,
    'mnist_fmnist': MNIST_FMNIST,
}

class WeightedDataset(Dataset):
    def __init__(self, dataset, weights=None):
        self.dataset = dataset
        self.weights = weights if weights is not None else np.ones(len(dataset))
    
    def __getitem__(self, index):
        data, target = self.dataset.__getitem__(index)
        return data, target, self.weights[index], index
    
    def __len__(self):
        return len(self.dataset)


def get_predefined_dataset(dataset_name, root, weights=None, **kwargs):
    if dataset_name == '25gaussian':
        dataset = get_25gaussian_dataset()
    else:
        transform = get_transform(dataset_name)
        dataset = DATASET_DICT[dataset_name](root=root, transform=transform, download=True, **kwargs)
    return WeightedDataset(dataset=dataset, weights=weights)
