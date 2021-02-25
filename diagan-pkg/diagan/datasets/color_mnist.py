# Code based on https://github.com/clovaai/rebias/blob/master/datasets/colour_mnist.py
import os
import pickle
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils import data
from torchvision.datasets import MNIST


class BiasedMNIST(MNIST):
    COLOUR_MAP = [[255, 0, 0], [0, 255, 0], ]

    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False, major_ratio=0.1, num_data=10000):
        super().__init__(root, train=train, transform=transform,
                         target_transform=target_transform,
                         download=download)
        self.random = True

        self.major_ratio = major_ratio
        self.num_data = num_data
        
        save_path = Path(root) / f'color_mnist-rd{major_ratio}-n{num_data}'
        if save_path.is_dir():
            print(f'use existing color_mnist from {save_path}')
            self.data = pickle.load(open(save_path / 'data.pkl', 'rb'))
            self.targets = pickle.load(open(save_path / 'targets.pkl', 'rb'))
            self.biased_targets = pickle.load(open(save_path / 'biased_targets.pkl', 'rb'))
        else:
            save_path.mkdir(parents=True, exist_ok=True)
            self.data = self.data[:num_data]
            self.data, self.targets, self.biased_targets = self.build_biased_mnist()

            indices = np.arange(len(self.data))
            self._shuffle(indices)

            self.data = self.data[indices].numpy()
            self.targets = self.targets[indices]
            self.biased_targets = self.biased_targets[indices]
            pickle.dump(self.data, open(save_path / 'data.pkl', 'wb'))
            pickle.dump(self.targets, open(save_path / 'targets.pkl', 'wb'))
            pickle.dump(self.biased_targets, open(save_path / 'biased_targets.pkl', 'wb'))

        self.labels = self.biased_targets
        num_each_bias = [sum(self.biased_targets == t) for t in range(len(self.COLOUR_MAP))]
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

    def build_biased_mnist(self):
        """Build biased MNIST.
        """
        n_labels = self.targets.max().item() + 1

        bias_indices = dict()
        random_indices = np.random.permutation(self.num_data)
        num_major = int(self.num_data * self.major_ratio)
        bias_indices[0] = random_indices[:num_major]
        bias_indices[1] = random_indices[num_major:]

        data = torch.ByteTensor()
        targets = torch.LongTensor()
        biased_targets = []

        for bias_label, indices in bias_indices.items():
            _data, _targets = self._make_biased_mnist(indices, bias_label)
            data = torch.cat([data, _data])
            targets = torch.cat([targets, _targets])
            biased_targets.extend([bias_label] * len(indices))

        biased_targets = torch.LongTensor(biased_targets)
        return data, targets, biased_targets

    def __getitem__(self, index):
        img, target, biased_target = self.data[index], int(self.targets[index]), int(self.biased_targets[index])
        img = Image.fromarray(img.astype(np.uint8), mode='RGB')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, biased_target


class ColoredMNIST(BiasedMNIST):
    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False, major_ratio=0.1, num_data=10000):
        super().__init__(root, train=train, transform=transform,
                                                target_transform=target_transform,
                                                download=download,
                                                major_ratio=major_ratio, num_data=num_data,)

    def _binary_to_colour(self, data, colour):
        bg_data = torch.zeros_like(data)
        bg_data[data == 0] = 0
        bg_data[data != 0] = 1
        bg_data = torch.stack([bg_data, bg_data, bg_data], dim=3)
        bg_data = bg_data * torch.ByteTensor(colour)
        bg_data = bg_data.permute(0, 3, 1, 2)

        data = bg_data
        return data.permute(0, 2, 3, 1)

    def _make_biased_mnist(self, indices, label):
        return self._binary_to_colour(self.data[indices], self.COLOUR_MAP[label % len(self.COLOUR_MAP)]), self.targets[indices]