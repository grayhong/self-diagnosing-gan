"""
Loads randomly sampled images from datasets for computing metrics.
"""
import os
import random
from io import BytesIO

import lmdb
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image

from torch_mimicry.datasets import data_utils



class MultiResolutionDataset(Dataset):
    def __init__(self, path, transform, resolution=256):
        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)

        with self.env.begin(write=False) as txn:
            self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))

        self.resolution = resolution
        self.transform = transform
        self.blacklist = np.array([40650])
        self.length -= len(self.blacklist)
        print(f'MultiResolutionDataset len: {self.length}')
        # self.check_consistency()
    
    def get_index(self, idx):
        shift = sum(self.blacklist <= idx)
        return idx + shift
    
    def check_consistency(self):
        for index in range(self.length):
            with self.env.begin(write=False) as txn:
                key = f'{self.resolution}-{str(index).zfill(5)}'.encode('utf-8')
                img_bytes = txn.get(key)

            buffer = BytesIO(img_bytes)
            try:
                img = Image.open(buffer)
            except:
                print(f'Exception at {index}')

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        index = self.get_index(index)
        with self.env.begin(write=False) as txn:
            key = f'{self.resolution}-{str(index).zfill(5)}'.encode('utf-8')
            img_bytes = txn.get(key)

        buffer = BytesIO(img_bytes)
        img = Image.open(buffer)
        img = np.asarray(img)
        # img = self.transform(img)
        # img = img.numpy().transpose(1, 2, 0)

        return img


def get_random_images(dataset, num_samples):
    """
    Randomly sample without replacement num_samples images.

    Args:
        dataset (Dataset): Torch Dataset object for indexing elements.
        num_samples (int): The number of images to randomly sample.

    Returns:
        ndarray: Batch of num_samples images in np array form.
    """
    choices = np.random.choice(range(len(dataset)),
                               size=num_samples,
                               replace=False)

    images = []
    for choice in choices:
        # for choice in range(100):
        img = np.array(dataset[choice])
        img = np.expand_dims(img, axis=0)
        images.append(img)
    images = np.concatenate(images, axis=0)

    return images



def get_ffhq_images(num_samples, root='./dataset', size=256):

    transform = transforms.Compose(
        [
            # transforms.RandomHorizontalFlip(),
            # transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ]
    )

    dataset = MultiResolutionDataset(root, transform, size)
    images = get_random_images(dataset, num_samples)

    return images


def sample_dataset_images(dataset, num_samples):
    """
    Randomly samples the dataset for images.

    Args:
        dataset (Dataset): Torch dataset object to sample images from.
        num_samples (int): The number of images to randomly sample.

    Returns:
        ndarray: Numpy array of images with first dim as batch size.
    """
    # Check if sufficient images
    if len(dataset) < num_samples:
        raise ValueError(
            "Given dataset has less than num_samples images: {} given but requires at least {}."
            .format(len(dataset), num_samples))

    choices = random.sample(range(len(dataset)), num_samples)
    images = []
    for i in choices:
        data = dataset[i]

        # Case of iterable, assumes first arg is data.
        if isinstance(data, tuple) or isinstance(data, list):
            img = data[0]
        else:
            img = data

        img = torch.unsqueeze(img, 0)
        images.append(img)

    images = np.concatenate(images, axis=0)

    return images


def get_dataset_images(dataset, data_path='./dataset', num_samples=50000, **kwargs):
    """
    Randomly sample num_samples images based on input dataset name.

    Args:
        dataset (str/Dataset): Dataset to load images from.
        num_samples (int): The number of images to randomly sample.

    Returns:
        ndarray: Batch of num_samples images from a dataset in np array form.
            The final format is of (N, H, W, 3) shape for TF inference.
    """
    if isinstance(dataset, str):
        if dataset == "ffhq":
            images = get_ffhq_images(num_samples, root=data_path, size=256, **kwargs)

        else:
            raise ValueError("Invalid dataset name {}.".format(dataset))

    elif issubclass(type(dataset), torch.utils.data.Dataset):
        images = sample_dataset_images(dataset, num_samples)

    else:
        raise ValueError("dataset must be of type str or a Dataset object.")

    # Check shape and permute if needed
    if images.shape[1] == 3:
        images = images.transpose((0, 2, 3, 1))

    # Ensure the values lie within the correct range, otherwise there might be some
    # preprocessing error from the library causing ill-valued scores.
    if np.min(images) < 0 or np.max(images) > 255:
        print(
            "INFO: Some pixel values lie outside of [0, 255]. Clipping values.."
        )
        images = np.clip(images, 0, 255)

    return images
