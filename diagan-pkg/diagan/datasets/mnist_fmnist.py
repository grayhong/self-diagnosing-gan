import os
import pickle
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils import data
from torchvision.datasets import MNIST, FashionMNIST


class MixedMNIST(MNIST):

    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=True, major_ratio=0.1, num_data=10000):
        super().__init__(root, train=train, transform=transform,
                         target_transform=target_transform,
                         download=download)
        self.random = True

        self.major_ratio = major_ratio
        self.num_data = num_data

        self.FMNIST_dataset = FashionMNIST('./dataset/fmnist', train=True, download=True)
        
        save_path = Path(root) / f'mnist_fmnist-{major_ratio}-n{num_data}'
        if save_path.is_dir():
            print(f'use existing mnist_fmnist from {save_path}')
            self.data = pickle.load(open(save_path / 'data.pkl', 'rb'))
            self.targets = pickle.load(open(save_path / 'targets.pkl', 'rb'))
            self.mixed_targets = pickle.load(open(save_path / 'mixed_targets.pkl', 'rb'))
        else:
            save_path.mkdir(parents=True, exist_ok=True)
            self.data = self.data[:num_data]
            self.data, self.targets, self.mixed_targets = self.build_mixed_mnist()

            indices = np.arange(len(self.data))
            self._shuffle(indices)

            self.data = self.data[indices].numpy()
            self.targets = self.targets[indices]
            self.mixed_targets = self.mixed_targets[indices]
            pickle.dump(self.data, open(save_path / 'data.pkl', 'wb'))
            pickle.dump(self.targets, open(save_path / 'targets.pkl', 'wb'))
            pickle.dump(self.mixed_targets, open(save_path / 'mixed_targets.pkl', 'wb'))

        self.labels = self.mixed_targets
        num_each_bias = [sum(self.mixed_targets == t) for t in range(2)]
        print(f'num_each_bias: {num_each_bias} total: {len(self.data)}')

    @property
    def raw_folder(self):
        return os.path.join(self.root, 'raw')

    @property
    def processed_folder(self):
        return os.path.join(self.root, 'processed')

    def _shuffle(self, iteratable):
        if self.random:
            np.random.shuffle(iteratable)

    def _make_biased_mnist(self, indices, label):
        raise NotImplementedError

    def build_mixed_mnist(self):
        """Build mixed MNIST.
        """
        n_labels = self.targets.max().item() + 1

        bias_indices = dict()
        random_indices = np.random.permutation(self.num_data)
        num_major = int(self.num_data * self.major_ratio)
        bias_indices[0] = random_indices[:num_major]
        bias_indices[1] = random_indices[num_major:]

        data = torch.ByteTensor()
        targets = torch.LongTensor()
        mixed_targets = []

        for bias_label, indices in bias_indices.items():
            _data, _targets = self._make_mixed_mnist(indices, bias_label)
            data = torch.cat([data, _data])
            targets = torch.cat([targets, _targets])
            mixed_targets.extend([bias_label] * len(indices))

        mixed_targets = torch.LongTensor(mixed_targets)
        return data, targets, mixed_targets

    def __getitem__(self, index):
        img, target, mixed_target = self.data[index], int(self.targets[index]), int(self.mixed_targets[index])
        img = Image.fromarray(img, mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, mixed_target


class MNIST_FMNIST(MixedMNIST):
    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=True, major_ratio=0.1, num_data=10000):
        super().__init__(root, train=train, transform=transform,
                                                target_transform=target_transform,
                                                download=download,
                                                major_ratio=major_ratio, num_data=num_data,)


    def _make_mixed_mnist(self, indices, label):
        if label == 0:
            return self.data[indices], self.targets[indices]
        else:
            return self.FMNIST_dataset.data[indices], self.FMNIST_dataset.targets[indices]